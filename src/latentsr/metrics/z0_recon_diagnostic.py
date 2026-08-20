"""Decode ẑ0 at selected reverse-process timesteps (image-quality diagnostic).

Same paired noise as ``evaluate_sr`` / the latent timestep diagnostic.
At each requested t, decode ``x̂0(t) = D(ẑ0(t))`` and score vs HR (and vs LR).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from latentsr.datasets.sr_pairs import downsample_bicubic
from latentsr.metrics.image_metrics import (
    LPIPSMetric,
    batch_metrics,
    psnr,
    summarize_values,
)
from latentsr.metrics.timestep_diagnostic import latent_cosine, latent_rmse
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.super_resolution.sample import (
    _STEP_SALT,
    predict_x0_from_eps,
    seeded_noise_like,
)
from latentsr.vae.latent import decode_scaled
from latentsr.vae.vae import VAE

DEFAULT_Z0_RECON_TIMESTEPS = (800, 700, 650, 600, 500, 400, 300, 200, 100, 0)
_VAE1_COLOR = "#4c78a8"
_VAESR_COLOR = "#f58518"


def _score_decode(
    image: torch.Tensor,
    hr: torch.Tensor,
    lr: torch.Tensor,
    *,
    lr_size: int,
    lpips_fn: LPIPSMetric | None,
) -> dict[str, torch.Tensor]:
    metrics = batch_metrics(image, hr, lpips_fn=lpips_fn)
    lr_hat = downsample_bicubic(image, lr_size)
    metrics["lr_psnr"] = psnr(lr_hat, lr, reduction="none")
    keep = ["psnr", "ssim", "edge_mae", "lr_psnr"]
    if "lpips" in metrics:
        keep.append("lpips")
    return {k: metrics[k] for k in keep}


@torch.no_grad()
def run_z0_recon_diagnostic(
    model_a: ConditionalLatentDDPM,
    vae_a: VAE,
    model_b: ConditionalLatentDDPM,
    vae_b: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    timesteps: Sequence[int] = DEFAULT_Z0_RECON_TIMESTEPS,
    num_images: int = 64,
    hr_size: int = 128,
    lr_size: int = 32,
    latent_scale_a: float = 1.0,
    latent_scale_b: float = 1.0,
    noise_seed: int = 42,
    start_index: int = 0,
    output_dir: str | Path | None = None,
    show_progress: bool = True,
    compute_lpips: bool = True,
    grid_images: int = 4,
    baseline_name: str = "vae1",
    candidate_name: str = "vae_sr",
) -> dict[str, Any]:
    """Paired reverse chain; decode ẑ0 at ``timesteps`` and score images."""
    if model_a.num_timesteps != model_b.num_timesteps:
        raise ValueError(
            f"timestep mismatch: {model_a.num_timesteps} vs {model_b.num_timesteps}"
        )
    num_t = int(model_a.num_timesteps)
    selected = tuple(int(t) for t in timesteps)
    if not selected:
        raise ValueError("timesteps must be non-empty")
    bad = [t for t in selected if t < 0 or t >= num_t]
    if bad:
        raise ValueError(f"timesteps {bad} outside [0, {num_t - 1}]")
    selected_set = set(selected)

    for model, vae in ((model_a, vae_a), (model_b, vae_b)):
        model.eval()
        vae.eval()
        model.to(device)
        vae.to(device)

    lpips_fn: LPIPSMetric | None = None
    if compute_lpips:
        lpips_fn = LPIPSMetric(net="alex", device=device)

    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_image: list[dict[str, Any]] = []
    grids: dict[int, dict[str, torch.Tensor]] = {}

    remaining = max(int(num_images), 1)
    next_index = int(start_index)
    pbar = (
        tqdm(total=remaining, desc="z0-recon", leave=False) if show_progress else None
    )

    for lr, hr in loader:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        hr = hr[:take].to(device)
        indices = list(range(next_index, next_index + take))

        z_lr_a = encode_lr_latents(
            vae_a, lr, hr_size=hr_size, latent_scale=latent_scale_a
        )
        z_lr_b = encode_lr_latents(
            vae_b, lr, hr_size=hr_size, latent_scale=latent_scale_b
        )
        x_a = seeded_noise_like(z_lr_a, indices, base_seed=noise_seed, salt=0)
        x_b = x_a.clone()

        for t in range(num_t - 1, -1, -1):
            t_batch = torch.full((take,), t, device=device, dtype=torch.long)
            eps_a = model_a.predict_noise(x_a, t_batch, z_lr_a)
            eps_b = model_b.predict_noise(x_b, t_batch, z_lr_b)
            z0_a = predict_x0_from_eps(model_a.scheduler, x_a, t_batch, eps_a)
            z0_b = predict_x0_from_eps(model_b.scheduler, x_b, t_batch, eps_b)

            if t in selected_set:
                img_a = decode_scaled(vae_a, z0_a, latent_scale_a).clamp(0.0, 1.0)
                img_b = decode_scaled(vae_b, z0_b, latent_scale_b).clamp(0.0, 1.0)
                scores_a = _score_decode(
                    img_a, hr, lr, lr_size=lr_size, lpips_fn=lpips_fn
                )
                scores_b = _score_decode(
                    img_b, hr, lr, lr_size=lr_size, lpips_fn=lpips_fn
                )
                cos_a = latent_cosine(z0_a, z_lr_a)
                cos_b = latent_cosine(z0_b, z_lr_b)
                rmse_a = latent_rmse(z0_a, z_lr_a)
                rmse_b = latent_rmse(z0_b, z_lr_b)

                for i, val_index in enumerate(indices):
                    row: dict[str, Any] = {"val_index": int(val_index), "t": int(t)}
                    for key, values in scores_a.items():
                        row[f"{baseline_name}_{key}"] = float(values[i].cpu())
                        buckets[t][f"{baseline_name}_{key}"].append(
                            float(values[i].cpu())
                        )
                    for key, values in scores_b.items():
                        row[f"{candidate_name}_{key}"] = float(values[i].cpu())
                        buckets[t][f"{candidate_name}_{key}"].append(
                            float(values[i].cpu())
                        )
                    row[f"{baseline_name}_cosine_z0_z_lr"] = float(cos_a[i].cpu())
                    row[f"{candidate_name}_cosine_z0_z_lr"] = float(cos_b[i].cpu())
                    row[f"{baseline_name}_z0_z_lr_rmse"] = float(rmse_a[i].cpu())
                    row[f"{candidate_name}_z0_z_lr_rmse"] = float(rmse_b[i].cpu())
                    buckets[t][f"{baseline_name}_cosine_z0_z_lr"].append(
                        float(cos_a[i].cpu())
                    )
                    buckets[t][f"{candidate_name}_cosine_z0_z_lr"].append(
                        float(cos_b[i].cpu())
                    )
                    per_image.append(row)

                if t not in grids and grid_images > 0:
                    n_grid = min(int(grid_images), take)
                    grids[t] = {
                        "vae1": img_a[:n_grid].cpu(),
                        "vae_sr": img_b[:n_grid].cpu(),
                        "hr": hr[:n_grid].cpu(),
                    }

            step_noise = seeded_noise_like(
                x_a,
                indices,
                base_seed=noise_seed,
                salt=_STEP_SALT * (int(t) + 1),
            )
            x_a = model_a.scheduler.p_sample_step(
                x_a, t_batch, eps_a, noise=step_noise
            )
            x_b = model_b.scheduler.p_sample_step(
                x_b, t_batch, eps_b, noise=step_noise
            )

        remaining -= take
        next_index += take
        if pbar is not None:
            pbar.update(take)

    if pbar is not None:
        pbar.close()

    n_done = len({row["val_index"] for row in per_image}) if per_image else 0
    rows = []
    for t in selected:
        summary = {key: summarize_values(vals) for key, vals in buckets[t].items()}
        row: dict[str, Any] = {"t": t, "n": n_done}
        for key, stats in summary.items():
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
        if f"{baseline_name}_psnr_mean" in row and f"{candidate_name}_psnr_mean" in row:
            row["delta_psnr_mean"] = (
                row[f"{candidate_name}_psnr_mean"] - row[f"{baseline_name}_psnr_mean"]
            )
        if (
            f"{baseline_name}_lpips_mean" in row
            and f"{candidate_name}_lpips_mean" in row
        ):
            row["delta_lpips_mean"] = (
                row[f"{candidate_name}_lpips_mean"] - row[f"{baseline_name}_lpips_mean"]
            )
        rows.append(row)

    result: dict[str, Any] = {
        "num_images": n_done,
        "timesteps": list(selected),
        "noise_seed": int(noise_seed),
        "start_index": int(start_index),
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "rows": rows,
        "per_image": per_image,
    }
    if output_dir is not None:
        result["paths"] = write_z0_recon_outputs(result, output_dir, grids=grids)
    return result


def format_z0_recon_table(result: dict[str, Any]) -> str:
    baseline = result["baseline_name"]
    candidate = result["candidate_name"]
    n = result["num_images"]
    has_lpips = any(f"{baseline}_lpips_mean" in r for r in result["rows"])
    lines = [
        f"n={n}  decode(ẑ0(t)) vs HR  |  Δ = {candidate} − {baseline}",
        "",
        f"{'t':>5}  {'PSNR':^21}  {'ΔPSNR':>7}  {'SSIM':^17}  "
        + (f"{'LPIPS':^17}  " if has_lpips else "")
        + f"{'edgeMAE':^17}  {'LR-PSNR':^17}  {'cos ẑ0,z_lr':^17}",
        f"{'':>5}  {baseline:>10} {candidate:>10}  {'':>7}  "
        f"{baseline:>8} {candidate:>8}  "
        + (f"{baseline:>8} {candidate:>8}  " if has_lpips else "")
        + f"{baseline:>8} {candidate:>8}  {baseline:>8} {candidate:>8}  "
        f"{baseline:>8} {candidate:>8}",
        "-" * (108 if has_lpips else 90),
    ]
    for row in result["rows"]:
        lpips_bit = ""
        if has_lpips:
            lpips_bit = (
                f"{row[f'{baseline}_lpips_mean']:8.4f} "
                f"{row[f'{candidate}_lpips_mean']:8.4f}  "
            )
        lines.append(
            f"{int(row['t']):5d}  "
            f"{row[f'{baseline}_psnr_mean']:10.3f} {row[f'{candidate}_psnr_mean']:10.3f}  "
            f"{row.get('delta_psnr_mean', float('nan')):7.3f}  "
            f"{row[f'{baseline}_ssim_mean']:8.4f} {row[f'{candidate}_ssim_mean']:8.4f}  "
            f"{lpips_bit}"
            f"{row[f'{baseline}_edge_mae_mean']:8.4f} {row[f'{candidate}_edge_mae_mean']:8.4f}  "
            f"{row[f'{baseline}_lr_psnr_mean']:8.3f} {row[f'{candidate}_lr_psnr_mean']:8.3f}  "
            f"{row[f'{baseline}_cosine_z0_z_lr_mean']:8.4f} "
            f"{row[f'{candidate}_cosine_z0_z_lr_mean']:8.4f}"
        )
    return "\n".join(lines)


def write_z0_recon_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    grids: dict[int, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    csv_path = output_dir / "z0_recon_means.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    per_image = result.get("per_image") or []
    per_path = output_dir / "z0_recon_per_image.csv"
    if per_image:
        with per_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_image[0].keys()))
            writer.writeheader()
            writer.writerows(per_image)

    json_path = output_dir / "z0_recon.json"
    serializable = {k: v for k, v in result.items() if k not in {"paths", "per_image"}}
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    txt_path = output_dir / "z0_recon.txt"
    txt_path.write_text(format_z0_recon_table(result) + "\n", encoding="utf-8")

    plots = write_z0_recon_plots(result, output_dir)
    grid_paths: list[str] = []
    if grids:
        for t in result["timesteps"]:
            if t not in grids:
                continue
            pack = grids[t]
            n = pack["hr"].shape[0]
            grid = make_grid(
                torch.cat([pack["vae1"], pack["vae_sr"], pack["hr"]], dim=0),
                nrow=n,
                padding=2,
            )
            path = output_dir / f"z0_recon_t_{int(t):03d}.png"
            save_image(grid, path)
            grid_paths.append(str(path))

    return {
        "csv": str(csv_path),
        "per_image": str(per_path) if per_image else "",
        "json": str(json_path),
        "table": str(txt_path),
        "plots": [str(p) for p in plots],
        "grids": grid_paths,
    }


def write_z0_recon_plots(
    result: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    rows = result["rows"]
    t_vals = [int(r["t"]) for r in rows]
    baseline = result["baseline_name"]
    candidate = result["candidate_name"]
    written: list[Path] = []

    def _save(fig: plt.Figure, name: str) -> None:
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    specs = [
        ("psnr", "PSNR vs HR (dB)", "decode(ẑ0) PSNR", False),
        ("ssim", "SSIM vs HR", "decode(ẑ0) SSIM", False),
        ("edge_mae", "edge MAE vs HR", "decode(ẑ0) edge MAE", True),
        ("lr_psnr", "LR-consistency PSNR (dB)", "downsample(ẑ0) vs input LR", False),
    ]
    if rows and f"{baseline}_lpips_mean" in rows[0]:
        specs.insert(2, ("lpips", "LPIPS vs HR", "decode(ẑ0) LPIPS", True))

    for key, ylabel, title, lower_better in specs:
        b_col = f"{baseline}_{key}_mean"
        c_col = f"{candidate}_{key}_mean"
        if not rows or b_col not in rows[0]:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(
            t_vals,
            [r[b_col] for r in rows],
            color=_VAE1_COLOR,
            marker="o",
            label=baseline,
        )
        ax.plot(
            t_vals,
            [r[c_col] for r in rows],
            color=_VAESR_COLOR,
            marker="o",
            label=candidate,
        )
        ax.set_xlabel("t  (0 = clean, larger = noisier)")
        suffix = "  (lower better)" if lower_better else ""
        ax.set_ylabel(ylabel + suffix)
        ax.set_title(title)
        ax.legend(frameon=False)
        _save(fig, f"z0_recon_{key}.png")

    if rows and "delta_psnr_mean" in rows[0]:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(
            t_vals,
            [r["delta_psnr_mean"] for r in rows],
            color="#c44e52",
            marker="o",
            label="Δ PSNR (VAE-SR − VAE-1)",
        )
        ax.axhline(0.0, color="#888888", lw=0.8)
        ax.set_xlabel("t  (0 = clean, larger = noisier)")
        ax.set_ylabel("Δ PSNR (dB)")
        ax.set_title("Does the VAE-SR ẑ0 image advantage survive to t=0?")
        ax.legend(frameon=False)
        _save(fig, "z0_recon_delta_psnr.png")

    return written
