"""Run bicubic vs LatentSR metrics on CelebA val pairs"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import save_image

from latentsr.datasets.onthefly_sr_latent import upsample_bicubic
from latentsr.metrics.image_metrics import (
    LPIPSMetric,
    batch_metrics,
    dataset_filename,
    format_metric_table,
    sobel_magnitude,
    summarize_values,
    write_per_image_csv,
    write_summary_files,
)
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import encode_lr_latents, save_sr_comparison_grid
from latentsr.super_resolution.sample import sample_conditional_latents
from latentsr.vae.latent import decode_scaled, encode_scaled
from latentsr.vae.vae import VAE
from latentsr.vae.whitening import ChannelWhitening


def _latent_pair_stats(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    delta = pred - target
    mse = delta.pow(2).mean(dim=(1, 2, 3))
    return {
        "mse": mse,
        "rmse": mse.clamp_min(0.0).sqrt(),
        "l2": delta.flatten(1).norm(dim=1),
        "cosine": F.cosine_similarity(pred.flatten(1), target.flatten(1)),
    }


@torch.no_grad()
def evaluate_sr(
    model: ConditionalLatentDDPM,
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int = 64,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    compute_lpips: bool = True,
    include_soft_decode: bool = True,
    show_progress: bool = True,
    grid_images: int = 8,
    output_dir: str | Path | None = None,
    noise_seed: int = 42,
    start_index: int = 0,
    whitener: ChannelWhitening | None = None,
) -> dict[str, Any]:
    """Evaluate bicubic + soft decode vs LatentSR on a val loader.

    Reverse-diffusion noise is generated per ``val_index`` from ``noise_seed``,
    so pairing two checkpoints does not depend on batch size.

    Soft-decode always uses **raw** ``z_lr``. The UNet condition uses whitened
    ``z_lr`` when ``whitener`` is set (matched whitened DDPM).
    """
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)

    lpips_fn: LPIPSMetric | None = None
    if compute_lpips:
        lpips_fn = LPIPSMetric(net="alex", device=device)

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_image_rows: list[dict[str, Any]] = []
    remaining = max(int(num_images), 1)
    next_index = int(start_index)
    grid_lr = grid_pred = grid_hr = grid_bicubic = grid_soft = None

    iterator = tqdm(loader, desc="evaluate", leave=False) if show_progress else loader
    dataset = loader.dataset
    for lr, hr in iterator:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        hr = hr[:take].to(device)
        indices = list(range(next_index, next_index + take))

        bicubic = upsample_bicubic(lr, hr_size)
        z_hr = encode_scaled(vae, hr, latent_scale=latent_scale)
        z_lr_raw = encode_lr_latents(
            vae,
            lr,
            hr_size=hr_size,
            latent_scale=latent_scale,
            apply_whiten=False,
        )
        z_lr_cond = (
            whitener.transform(z_lr_raw) if whitener is not None else z_lr_raw
        )
        z_sr = sample_conditional_latents(
            model,
            z_lr_cond,
            val_indices=indices,
            noise_seed=noise_seed,
            show_progress=False,
        )
        pred = decode_scaled(vae, z_sr, latent_scale=latent_scale).clamp(0.0, 1.0)
        soft = decode_scaled(vae, z_lr_raw, latent_scale=latent_scale).clamp(0.0, 1.0)

        methods = {"bicubic": bicubic, "soft_decode": soft, "latentsr": pred}
        batch_metric_tensors: dict[str, dict[str, torch.Tensor]] = {}
        for name, output in methods.items():
            metrics = batch_metrics(output, hr, lpips_fn=lpips_fn)
            batch_metric_tensors[name] = metrics
            for key, values in metrics.items():
                scores[name][key].extend(values.detach().cpu().tolist())

        # Latent distances stay in raw space for soft-decode comparability.
        z_lr_stats = _latent_pair_stats(z_lr_raw, z_hr)
        z_sr_stats = _latent_pair_stats(z_sr, z_hr)
        hr_edge = sobel_magnitude(hr).mean(dim=(1, 2, 3))

        for i, val_index in enumerate(indices):
            row: dict[str, Any] = {
                "val_index": int(val_index),
                "filename": dataset_filename(dataset, int(val_index)),
            }
            for method, metric_map in batch_metric_tensors.items():
                for key, values in metric_map.items():
                    row[f"{method}_{key}"] = float(values[i].detach().cpu().item())
            for prefix, stats in (("z_lr", z_lr_stats), ("z_sr", z_sr_stats)):
                for key, values in stats.items():
                    row[f"{prefix}_{key}"] = float(values[i].detach().cpu().item())
            row["hr_edge_energy"] = float(hr_edge[i].detach().cpu().item())
            per_image_rows.append(row)

        if grid_lr is None:
            n_grid = min(grid_images, take)
            grid_lr = lr[:n_grid].cpu()
            grid_pred = pred[:n_grid].cpu()
            grid_hr = hr[:n_grid].cpu()
            grid_bicubic = bicubic[:n_grid].cpu()
            grid_soft = soft[:n_grid].cpu()

        remaining -= take
        next_index += take

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method, metric_map in scores.items():
        summary[method] = {
            metric: summarize_values(vals) for metric, vals in metric_map.items()
        }

    result: dict[str, Any] = {
        "summary": summary,
        "num_images": int(summary.get("latentsr", {}).get("psnr", {}).get("n", 0)),
        "table": format_metric_table(summary),
        "per_image": per_image_rows,
        "noise_seed": int(noise_seed),
        "start_index": int(start_index),
    }

    if output_dir is not None and grid_lr is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = save_sr_comparison_grid(
            grid_lr,
            grid_pred,
            hr=grid_hr,
            output_path=output_dir / "eval_compare.png",
            hr_size=hr_size,
            include_soft_decode=include_soft_decode and grid_soft is not None,
            soft=grid_soft,
        )
        save_image(grid_bicubic, output_dir / "eval_bicubic.png", nrow=4, padding=2)
        save_image(grid_pred, output_dir / "eval_latentsr.png", nrow=4, padding=2)
        save_image(grid_hr, output_dir / "eval_hr.png", nrow=4, padding=2)
        if include_soft_decode and grid_soft is not None:
            save_image(grid_soft, output_dir / "eval_soft_decode.png", nrow=4, padding=2)
        result["grid_path"] = str(grid_path)
        write_summary_files(
            output_dir,
            summary,
            result["num_images"],
            extra={
                "noise_seed": int(noise_seed),
                "start_index": int(start_index),
                "per_image_noise": True,
            },
        )
        write_per_image_csv(output_dir / "per_image.csv", per_image_rows)

    return result
