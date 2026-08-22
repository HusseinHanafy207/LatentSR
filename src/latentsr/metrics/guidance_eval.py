"""Stage 1 guidance eval: cache D(z_lr), λ_g sanity, timed run, 3-condition eval."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from latentsr.datasets.onthefly_sr_latent import upsample_bicubic
from latentsr.metrics.evaluate_sr import _latent_pair_stats
from latentsr.metrics.image_metrics import (
    LPIPSMetric,
    batch_metrics,
    dataset_filename,
    format_metric_table,
    psnr,
    sobel_magnitude,
    summarize_values,
    write_per_image_csv,
    write_summary_files,
)
from latentsr.metrics.paired_stats import (
    compare_per_image,
    load_per_image_csv,
    write_comparison_outputs,
)
from latentsr.metrics.timestep_diagnostic import latent_cosine, latent_rmse
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.guidance import (
    BASELINE_WINDOW,
    EARLY_WINDOW,
    LATE_WINDOW,
    TRAJECTORY_TIMESTEPS,
    GuidanceWindow,
    cache_soft_decodes,
    dps_guidance_step,
    iter_indexed_batches,
    sample_guided_latents,
)
from latentsr.super_resolution.inference import save_sr_comparison_grid
from latentsr.super_resolution.sample import STEP_SALT, seeded_noise_like
from latentsr.vae.latent import decode_scaled, encode_scaled
from latentsr.vae.vae import VAE

SOFT_DECODE_PSNR_REF = 28.48
SOFT_DECODE_LPIPS_REF = 0.119
BICUBIC_PSNR_REF = 26.13
BICUBIC_LPIPS_REF = 0.282
UNGUIDED_PSNR_REF = 26.48
UNGUIDED_LPIPS_REF = 0.0685

SANITY_TIMESTEPS = (800, 650, 500, 300, 100)
LAMBDA_CANDIDATES = (0.001, 0.01, 0.1, 1.0)


def recommend_lambda_g(
    rows: list[dict[str, Any]],
    *,
    lambdas: tuple[float, ...] = LAMBDA_CANDIDATES,
    target: float = 0.3,
    lo: float = 0.1,
    hi: float = 0.5,
) -> dict[str, Any]:
    """Pick λ_g from R(t) only (never from PSNR). Prefers mean R in [0.1, 0.5]."""
    if not rows:
        raise ValueError("no R(t) rows")
    means: dict[float, float] = {}
    for lam in lambdas:
        key = f"R_lambda_{lam:g}"
        vals = [float(r[key]) for r in rows]
        means[lam] = sum(vals) / len(vals)
    in_band = [lam for lam, mean_r in means.items() if lo <= mean_r <= hi]
    pool = in_band if in_band else list(lambdas)
    chosen = min(pool, key=lambda lam: abs(means[lam] - target))
    return {
        "lambda_g": float(chosen),
        "mean_R": {str(lam): means[lam] for lam in lambdas},
        "in_band": [float(x) for x in in_band],
        "note": (
            "R(t) in 0.1–0.5"
            if in_band
            else "no candidate in 0.1–0.5; closest to 0.3 — re-check scale"
        ),
    }


PRE_REGISTERED = """
Pre-registered interpretation (fix BEFORE looking at Stage-1 numbers)
---------------------------------------------------------------------
PSNR↑, LPIPS ≈0.068 or better
    Strong positive: conditioning preservation helps without perceptual collapse.
PSNR↑ but LPIPS moving toward 0.119
    Over-guidance / soft-decode collapse.
PSNR ≈ baseline, LPIPS ≈ baseline, but cos(ẑ0, z_lr) clearly higher
    Mechanistic null: conditioning preserved, does not improve the final sample.
PSNR ≈ baseline, LPIPS ≈ baseline, AND trajectory barely differs
    Implementation/scale failure — re-check λ_g; do not report as a scientific null.
Both early and late show a result
    Stage 2 strength sweep on the better window.
Only early or only late shows a result
    Stop. Localize the conditioning-loss region. Do not run a combined schedule.
Neither works, but trajectory confirms guidance had a measured effect
    Negative result: preservation is not sufficient. Write it up. Do not train next.
""".strip()

REFERENCE_BANNER = (
    f"Reference (n=64, seed=42): bicubic {BICUBIC_PSNR_REF:.2f} dB / "
    f"LPIPS {BICUBIC_LPIPS_REF:.3f}  |  soft-decode {SOFT_DECODE_PSNR_REF:.2f} / "
    f"{SOFT_DECODE_LPIPS_REF:.3f}  |  unguided LatentSR {UNGUIDED_PSNR_REF:.2f} / "
    f"{UNGUIDED_LPIPS_REF:.4f}"
)


def windows_for_name(name: str, *, every: int = 1) -> GuidanceWindow:
    key = name.strip().lower()
    if key == "baseline":
        return BASELINE_WINDOW
    if key == "early":
        if every == 1:
            return EARLY_WINDOW
        return GuidanceWindow(t_low=0, t_high=800, every=every)
    if key == "late":
        if every == 1:
            return LATE_WINDOW
        return GuidanceWindow(t_low=0, t_high=500, every=every)
    raise ValueError(f"unknown condition {name!r}; use baseline, early, or late")


def check_soft_decode_cache(
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int = 64,
    start_index: int = 0,
    hr_size: int = 128,
    latent_scale: float = 1.0,
) -> dict[str, float]:
    """Step 0: PSNR(D(z_lr), HR) must match ~28.48 on the eval 64."""
    vae.eval()
    vae.to(device)
    scores: list[float] = []
    for lr, hr, _idx in iter_indexed_batches(
        loader, start_index=start_index, num_images=num_images
    ):
        lr = lr.to(device)
        hr = hr.to(device)
        _z_lr, d_zlr = cache_soft_decodes(
            vae, lr, hr_size=hr_size, latent_scale=latent_scale
        )
        scores.extend(psnr(d_zlr, hr, reduction="none").cpu().tolist())
    stats = summarize_values(scores)
    stats["expected"] = SOFT_DECODE_PSNR_REF
    stats["abs_error"] = abs(stats["mean"] - SOFT_DECODE_PSNR_REF)
    stats["ok"] = bool(stats["abs_error"] < 0.08)
    return stats


def measure_lambda_ratios(
    model: ConditionalLatentDDPM,
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int = 5,
    start_index: int = 64,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    noise_seed: int = 42,
    timesteps: tuple[int, ...] = SANITY_TIMESTEPS,
    lambdas: tuple[float, ...] = LAMBDA_CANDIDATES,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Step 2: R(t)=||λ g||/||Δz_DDPM|| on held-out images (not the eval 64)."""
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)
    log_set = {int(t) for t in timesteps}
    rows: list[dict[str, Any]] = []
    batches = list(
        iter_indexed_batches(loader, start_index=start_index, num_images=num_images)
    )
    iterator = tqdm(batches, desc="lambda-sanity", leave=False) if show_progress else batches
    for lr, hr, indices in iterator:
        del hr
        lr = lr.to(device)
        z_lr, d_zlr = cache_soft_decodes(
            vae, lr, hr_size=hr_size, latent_scale=latent_scale
        )
        z = seeded_noise_like(z_lr, indices, base_seed=noise_seed, salt=0)
        for t_int in range(model.num_timesteps - 1, -1, -1):
            t_batch = torch.full((z.shape[0],), t_int, device=device, dtype=torch.long)
            step_noise = seeded_noise_like(
                z, indices, base_seed=noise_seed, salt=STEP_SALT * (t_int + 1)
            )
            need = t_int in log_set
            z, extras = dps_guidance_step(
                model,
                vae,
                z,
                t_batch,
                z_lr,
                d_zlr,
                latent_scale=latent_scale,
                lambda_g=1.0,
                active=need,
                step_noise=step_noise,
                need_diagnostics=need,
                apply_correction=False,
            )
            if not need:
                continue
            for i, val_index in enumerate(indices):
                g_norm = float(extras["grad_norm"][i].cpu())
                d_norm = float(extras["delta_norm"][i].cpu())
                loss = float(extras["loss"][i].cpu())
                row: dict[str, Any] = {
                    "val_index": int(val_index),
                    "t": int(t_int),
                    "grad_norm": g_norm,
                    "ddpm_step_norm": d_norm,
                    "L_g": loss,
                }
                for lam in lambdas:
                    row[f"R_lambda_{lam:g}"] = (lam * g_norm) / max(d_norm, 1e-8)
                rows.append(row)
    return rows


def time_guided_image(
    model: ConditionalLatentDDPM,
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    lambda_g: float,
    window: GuidanceWindow = EARLY_WINDOW,
    start_index: int = 64,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    noise_seed: int = 42,
) -> dict[str, float]:
    """Step 3: one full early-guided trajectory; estimate 64×2 cost."""
    model.to(device)
    vae.to(device)
    lr, _hr, indices = next(
        iter_indexed_batches(loader, start_index=start_index, num_images=1)
    )
    lr = lr[:1].to(device)
    indices = indices[:1]
    z_lr, d_zlr = cache_soft_decodes(vae, lr, hr_size=hr_size, latent_scale=latent_scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    sample_guided_latents(
        model,
        vae,
        z_lr,
        d_zlr,
        latent_scale=latent_scale,
        lambda_g=lambda_g,
        window=window,
        val_indices=indices,
        noise_seed=noise_seed,
        show_progress=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "seconds_one_image_early": elapsed,
        "hours_64x2_guided": (elapsed * 64 * 2) / 3600.0,
        "lambda_g": float(lambda_g),
    }


def run_guidance_condition(
    model: ConditionalLatentDDPM,
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    condition: str,
    lambda_g: float,
    num_images: int = 64,
    start_index: int = 0,
    hr_size: int = 128,
    lr_size: int = 32,
    latent_scale: float = 1.0,
    noise_seed: int = 42,
    compute_lpips: bool = True,
    grid_images: int = 8,
    output_dir: str | Path | None = None,
    log_timesteps: tuple[int, ...] = TRAJECTORY_TIMESTEPS,
    guide_every: int = 1,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Step 4: one condition on the eval set, with trajectory snapshots."""
    window = windows_for_name(condition, every=guide_every)
    if condition == "baseline":
        lambda_g = 0.0
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)
    lpips_fn = LPIPSMetric(net="alex", device=device) if compute_lpips else None

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_image: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    grid_pack: dict[str, torch.Tensor] | None = None
    dataset = loader.dataset
    log_set = {int(t) for t in log_timesteps}

    batches = list(
        iter_indexed_batches(loader, start_index=start_index, num_images=num_images)
    )
    pbar = (
        tqdm(total=num_images, desc=f"guidance-{condition}", leave=False)
        if show_progress
        else None
    )
    for lr, hr, indices in batches:
        lr = lr.to(device)
        hr = hr.to(device)
        z_lr, d_zlr = cache_soft_decodes(
            vae, lr, hr_size=hr_size, latent_scale=latent_scale
        )
        z_hr = encode_scaled(vae, hr, latent_scale)
        bicubic = upsample_bicubic(lr, hr_size)

        def _on_log(
            t_int: int,
            extras: dict[str, torch.Tensor],
            *,
            _idx=indices,
            _z_lr=z_lr,
            _hr=hr,
        ) -> None:
            z0 = extras["z0"]
            decoded = extras.get("decoded_z0")
            if decoded is None:
                decoded = decode_scaled(vae, z0, latent_scale).clamp(0.0, 1.0)
            img_metrics = batch_metrics(decoded, _hr, lpips_fn=lpips_fn)
            for i, val_index in enumerate(_idx):
                row: dict[str, Any] = {
                    "val_index": int(val_index),
                    "t": int(t_int),
                    "cosine_z0_z_lr": float(latent_cosine(z0[i : i + 1], _z_lr[i : i + 1]).cpu()),
                    "z0_z_lr_rmse": float(latent_rmse(z0[i : i + 1], _z_lr[i : i + 1]).cpu()),
                    "L_g": float(extras["loss"][i].cpu()),
                    "R": float(extras["r"][i].cpu()),
                    "guidance_active": int(extras["guidance_active"][i].cpu()),
                    "ddpm_step_norm": float(extras["delta_norm"][i].cpu()),
                    "grad_norm": float(extras["grad_norm"][i].cpu()),
                }
                for key, values in img_metrics.items():
                    row[f"z0_{key}"] = float(values[i].cpu())
                traj_rows.append(row)

        z_sr = sample_guided_latents(
            model,
            vae,
            z_lr,
            d_zlr,
            latent_scale=latent_scale,
            lambda_g=lambda_g,
            window=window,
            val_indices=indices,
            noise_seed=noise_seed,
            log_timesteps=tuple(log_set),
            on_log=_on_log,
            show_progress=False,
        )
        pred = decode_scaled(vae, z_sr, latent_scale).clamp(0.0, 1.0)
        methods = {"bicubic": bicubic, "soft_decode": d_zlr, "latentsr": pred}
        batch_metric_tensors: dict[str, dict[str, torch.Tensor]] = {}
        for name, output in methods.items():
            metrics = batch_metrics(output, hr, lpips_fn=lpips_fn)
            batch_metric_tensors[name] = metrics
            for key, values in metrics.items():
                scores[name][key].extend(values.detach().cpu().tolist())
        z_lr_stats = _latent_pair_stats(z_lr, z_hr)
        z_sr_stats = _latent_pair_stats(z_sr, z_hr)
        hr_edge = sobel_magnitude(hr).mean(dim=(1, 2, 3))
        for i, val_index in enumerate(indices):
            row = {
                "val_index": int(val_index),
                "filename": dataset_filename(dataset, int(val_index)),
                "condition": condition,
                "lambda_g": float(lambda_g),
            }
            for method, metric_map in batch_metric_tensors.items():
                for key, values in metric_map.items():
                    row[f"{method}_{key}"] = float(values[i].cpu())
            for prefix, stats in (("z_lr", z_lr_stats), ("z_sr", z_sr_stats)):
                for key, values in stats.items():
                    row[f"{prefix}_{key}"] = float(values[i].cpu())
            row["hr_edge_energy"] = float(hr_edge[i].cpu())
            per_image.append(row)
        if grid_pack is None:
            n_grid = min(grid_images, lr.shape[0])
            grid_pack = {
                "lr": lr[:n_grid].cpu(),
                "pred": pred[:n_grid].cpu(),
                "hr": hr[:n_grid].cpu(),
                "soft": d_zlr[:n_grid].cpu(),
            }
        if pbar is not None:
            pbar.update(len(indices))
    if pbar is not None:
        pbar.close()

    summary = {
        method: {metric: summarize_values(vals) for metric, vals in metric_map.items()}
        for method, metric_map in scores.items()
    }
    result: dict[str, Any] = {
        "condition": condition,
        "lambda_g": float(lambda_g),
        "window": {"t_low": window.t_low, "t_high": window.t_high, "every": window.every},
        "num_images": len(per_image),
        "noise_seed": int(noise_seed),
        "start_index": int(start_index),
        "hr_size": int(hr_size),
        "lr_size": int(lr_size),
        "latent_scale": float(latent_scale),
        "summary": summary,
        "table": format_metric_table(summary),
        "per_image": per_image,
        "trajectory": traj_rows,
        "reference": REFERENCE_BANNER,
        "pre_registered": PRE_REGISTERED,
    }
    if output_dir is not None and grid_pack is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_sr_comparison_grid(
            grid_pack["lr"],
            grid_pack["pred"],
            hr=grid_pack["hr"],
            output_path=output_dir / "eval_compare.png",
            hr_size=hr_size,
            include_soft_decode=True,
            soft=grid_pack["soft"],
        )
        write_summary_files(
            output_dir,
            summary,
            result["num_images"],
            extra={
                "condition": condition,
                "lambda_g": float(lambda_g),
                "noise_seed": int(noise_seed),
                "guidance_convention": "dps_z_t_grad_applied_to_z_tm1",
                "guide_every": int(guide_every),
                "lr_size": int(lr_size),
            },
        )
        write_per_image_csv(output_dir / "per_image.csv", per_image)
        if traj_rows:
            write_per_image_csv(output_dir / "trajectory.csv", traj_rows)
        (output_dir / "protocol.txt").write_text(
            REFERENCE_BANNER + "\n\n" + PRE_REGISTERED + "\n", encoding="utf-8"
        )
        result["output_dir"] = str(output_dir)
    return result


def compare_guidance_conditions(
    baseline_csv: Path,
    candidate_csv: Path,
    *,
    output_dir: Path,
    baseline_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    baseline = load_per_image_csv(baseline_csv)
    candidate = load_per_image_csv(candidate_csv)
    result = compare_per_image(
        baseline,
        candidate,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
    )
    write_comparison_outputs(result, output_dir)
    return result
