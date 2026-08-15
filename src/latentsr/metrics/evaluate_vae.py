"""VAE bottleneck eval: decode(z_hr) / decode(z_lr) / bicubic vs HR.

No reverse diffusion — this measures what the frozen VAE already loses
before LatentSR runs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image

from latentsr.datasets.onthefly_sr_latent import upsample_bicubic
from latentsr.metrics.image_metrics import (
    LPIPSMetric,
    batch_metrics,
    format_metric_table,
    summarize_values,
    write_summary_files,
)
from latentsr.vae.latent import decode_scaled, encode_scaled
from latentsr.vae.vae import VAE

# Phase 10 LatentSR PSNR on 64 CelebA val images (for the printed gate note).
KNOWN_LATENTSR_PSNR = 26.25


class _MomentAccumulator:
    """Population mean / std over a stream of tensor elements."""

    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().reshape(-1).double()
        self.n += int(values.numel())
        self.total += float(values.sum().item())
        self.total_sq += float(values.pow(2).sum().item())

    def std(self, eps: float = 1e-6) -> float:
        if self.n == 0:
            return float("nan")
        mean = self.total / self.n
        var = max(self.total_sq / self.n - mean * mean, 0.0)
        return max(var**0.5, eps)

    def mean(self) -> float:
        if self.n == 0:
            return float("nan")
        return self.total / self.n


def format_bottleneck_note(
    summary: dict[str, dict[str, dict[str, float]]],
    *,
    latentsr_psnr: float | None = KNOWN_LATENTSR_PSNR,
) -> str:
    """Human-readable gate: is HR autoencoding the SR bottleneck?"""
    vae_psnr = summary["vae_hr"]["psnr"]["mean"]
    soft_psnr = summary["soft_decode"]["psnr"]["mean"]
    bicubic_psnr = summary["bicubic"]["psnr"]["mean"]
    lines = [
        f"VAE HR recon PSNR: {vae_psnr:.2f}  |  "
        f"soft decode(z_lr): {soft_psnr:.2f}  |  "
        f"bicubic: {bicubic_psnr:.2f}",
    ]
    if latentsr_psnr is not None:
        lines.append(
            f"Known LatentSR PSNR (Phase 10, 64 images): {latentsr_psnr:.2f}"
        )
        if vae_psnr >= latentsr_psnr + 3.0:
            lines.append(
                "Gate: decode(z_hr) is clearly above LatentSR — HR autoencoding "
                "is likely NOT the bottleneck (lean Q2: SR-aware z_lr)."
            )
        elif vae_psnr <= latentsr_psnr + 1.0:
            lines.append(
                "Gate: decode(z_hr) is close to LatentSR — the VAE may be the "
                "bottleneck (lean Q1: edge/frequency VAE)."
            )
        else:
            lines.append(
                "Gate: mixed — VAE recon is somewhat better than LatentSR. "
                "Run Q1 as a control; Q2 may still matter."
            )
    if abs(soft_psnr - bicubic_psnr) < 1.0:
        lines.append(
            "decode(z_lr) ≈ bicubic: the LR encoder mostly stores the blurry "
            "image, so an SR-aware z_lr objective (Q2) is relevant."
        )
    return "\n".join(lines)


def save_vae_bottleneck_grid(
    lr: torch.Tensor,
    bicubic: torch.Tensor,
    soft: torch.Tensor,
    vae_hr: torch.Tensor,
    hr: torch.Tensor,
    output_path: str | Path,
    *,
    hr_size: int = 128,
) -> Path:
    """Grid rows: nearest(LR) | bicubic | decode(z_lr) | decode(z_hr) | HR."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = lr.shape[0]
    lr_nn = F.interpolate(lr, size=(hr_size, hr_size), mode="nearest")
    rows = [
        lr_nn.cpu().clamp(0.0, 1.0),
        bicubic.cpu().clamp(0.0, 1.0),
        soft.cpu().clamp(0.0, 1.0),
        vae_hr.cpu().clamp(0.0, 1.0),
        hr.cpu().clamp(0.0, 1.0),
    ]
    grid = make_grid(torch.cat(rows, dim=0), nrow=n, padding=2)
    save_image(grid, output_path)
    return output_path


@torch.no_grad()
def evaluate_vae(
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int = 64,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    compute_lpips: bool = True,
    show_progress: bool = True,
    grid_images: int = 8,
    output_dir: str | Path | None = None,
    latentsr_psnr: float | None = KNOWN_LATENTSR_PSNR,
) -> dict[str, Any]:
    """Score bicubic, decode(z_lr), and decode(z_hr) against HR."""
    vae.eval()
    vae.to(device)

    lpips_fn: LPIPSMetric | None = None
    if compute_lpips:
        lpips_fn = LPIPSMetric(net="alex", device=device)

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    z_mse_values: list[float] = []
    z_cosine_values: list[float] = []
    mu_hr_moments = _MomentAccumulator()
    mu_lr_moments = _MomentAccumulator()

    remaining = max(int(num_images), 1)
    grid_lr = grid_bicubic = grid_soft = grid_vae_hr = grid_hr = None

    iterator = tqdm(loader, desc="evaluate-vae", leave=False) if show_progress else loader
    for lr, hr in iterator:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        hr = hr[:take].to(device)

        bicubic = upsample_bicubic(lr, hr_size)
        z_hr = encode_scaled(vae, hr, latent_scale=latent_scale)
        z_lr = encode_scaled(vae, bicubic, latent_scale=latent_scale)
        vae_hr = decode_scaled(vae, z_hr, latent_scale=latent_scale).clamp(0.0, 1.0)
        soft = decode_scaled(vae, z_lr, latent_scale=latent_scale).clamp(0.0, 1.0)

        methods = {"bicubic": bicubic, "soft_decode": soft, "vae_hr": vae_hr}
        for name, output in methods.items():
            metrics = batch_metrics(output, hr, lpips_fn=lpips_fn)
            for key, values in metrics.items():
                scores[name][key].extend(values.detach().cpu().tolist())

        mu_hr = z_hr / latent_scale
        mu_lr = z_lr / latent_scale
        mu_hr_moments.update(mu_hr)
        mu_lr_moments.update(mu_lr)
        z_mse_values.extend((z_lr - z_hr).pow(2).mean(dim=(1, 2, 3)).cpu().tolist())
        z_cosine_values.extend(
            F.cosine_similarity(z_lr.flatten(1), z_hr.flatten(1)).cpu().tolist()
        )

        if grid_lr is None:
            n_grid = min(grid_images, take)
            grid_lr = lr[:n_grid].cpu()
            grid_bicubic = bicubic[:n_grid].cpu()
            grid_soft = soft[:n_grid].cpu()
            grid_vae_hr = vae_hr[:n_grid].cpu()
            grid_hr = hr[:n_grid].cpu()

        remaining -= take

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method, metric_map in scores.items():
        summary[method] = {
            metric: summarize_values(vals) for metric, vals in metric_map.items()
        }

    mu_hr_std = mu_hr_moments.std()
    suggested_scale = (
        1.0 / mu_hr_std if math.isfinite(mu_hr_std) and mu_hr_std > 0 else float("nan")
    )
    latent_stats = {
        "latent_scale_used": float(latent_scale),
        "mu_hr_std": mu_hr_std,
        "mu_hr_mean": mu_hr_moments.mean(),
        "mu_lr_std": mu_lr_moments.std(),
        "mu_lr_mean": mu_lr_moments.mean(),
        "latent_scale_suggested": suggested_scale,
        "z_mse": summarize_values(z_mse_values),
        "z_cosine": summarize_values(z_cosine_values),
        "n_latent_elements_hr": mu_hr_moments.n,
    }
    bottleneck_note = format_bottleneck_note(summary, latentsr_psnr=latentsr_psnr)

    result: dict[str, Any] = {
        "summary": summary,
        "num_images": int(summary.get("vae_hr", {}).get("psnr", {}).get("n", 0)),
        "table": format_metric_table(summary),
        "latent_stats": latent_stats,
        "bottleneck_note": bottleneck_note,
    }

    if output_dir is not None and grid_lr is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = save_vae_bottleneck_grid(
            grid_lr,
            grid_bicubic,
            grid_soft,
            grid_vae_hr,
            grid_hr,
            output_dir / "eval_vae_compare.png",
            hr_size=hr_size,
        )
        save_image(grid_bicubic, output_dir / "eval_bicubic.png", nrow=4, padding=2)
        save_image(grid_soft, output_dir / "eval_soft_decode.png", nrow=4, padding=2)
        save_image(grid_vae_hr, output_dir / "eval_vae_hr.png", nrow=4, padding=2)
        save_image(grid_hr, output_dir / "eval_hr.png", nrow=4, padding=2)
        result["grid_path"] = str(grid_path)
        write_summary_files(
            output_dir,
            summary,
            result["num_images"],
            extra={
                "latent_stats": latent_stats,
                "bottleneck_note": bottleneck_note,
            },
        )

    return result
