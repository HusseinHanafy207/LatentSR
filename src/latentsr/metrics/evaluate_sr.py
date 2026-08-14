"""Run bicubic vs LatentSR metrics on CelebA val pairs (Phase 10)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import save_image

from latentsr.datasets.onthefly_sr_latent import upsample_bicubic
from latentsr.metrics.image_metrics import (
    LPIPSMetric,
    batch_metrics,
    format_metric_table,
    summarize_values,
)
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import (
    save_sr_comparison_grid,
    soft_decode_from_lr,
    super_resolve,
)
from latentsr.vae.vae import VAE


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
    include_soft_decode: bool = False,
    show_progress: bool = True,
    grid_images: int = 8,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate bicubic (+ optional soft decode) vs LatentSR on a val loader."""
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)

    lpips_fn: LPIPSMetric | None = None
    if compute_lpips:
        lpips_fn = LPIPSMetric(net="alex", device=device)

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    remaining = max(int(num_images), 1)
    grid_lr = grid_pred = grid_hr = grid_bicubic = grid_soft = None

    iterator = tqdm(loader, desc="evaluate", leave=False) if show_progress else loader
    for lr, hr in iterator:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        hr = hr[:take].to(device)

        bicubic = upsample_bicubic(lr, hr_size)
        pred = super_resolve(
            model,
            vae,
            lr,
            hr_size=hr_size,
            latent_scale=latent_scale,
            show_progress=False,
        )
        methods = {"bicubic": bicubic, "latentsr": pred}
        if include_soft_decode:
            soft = soft_decode_from_lr(
                vae, lr, hr_size=hr_size, latent_scale=latent_scale
            )
            methods["soft_decode"] = soft
        else:
            soft = None

        for name, output in methods.items():
            metrics = batch_metrics(output, hr, lpips_fn=lpips_fn)
            for key, values in metrics.items():
                scores[name][key].extend(values.detach().cpu().tolist())

        if grid_lr is None:
            n_grid = min(grid_images, take)
            grid_lr = lr[:n_grid].cpu()
            grid_pred = pred[:n_grid].cpu()
            grid_hr = hr[:n_grid].cpu()
            grid_bicubic = bicubic[:n_grid].cpu()
            if soft is not None:
                grid_soft = soft[:n_grid].cpu()

        remaining -= take

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for method, metric_map in scores.items():
        summary[method] = {
            metric: summarize_values(vals) for metric, vals in metric_map.items()
        }

    result: dict[str, Any] = {
        "summary": summary,
        "num_images": int(summary.get("latentsr", {}).get("psnr", {}).get("n", 0)),
        "table": format_metric_table(summary),
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
        result["grid_path"] = str(grid_path)
        _write_summary_files(output_dir, summary, result["num_images"])

    return result


def _write_summary_files(
    output_dir: Path,
    summary: dict[str, dict[str, dict[str, float]]],
    num_images: int,
) -> None:
    payload = {"num_images": num_images, "methods": summary}
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for method, metrics in summary.items():
        row: dict[str, Any] = {"method": method, "n": num_images}
        for metric, stats in metrics.items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
