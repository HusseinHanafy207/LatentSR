"""Diagnostic 2: Deterministic DDIM (eta=0) vs Ancestral DDPM (eta=1).

Evaluates the exact same checkpoint on the exact same initial noise x_T and seed:
    - DDPM: Stochastic ancestral reverse chain (eta=1.0)
    - DDIM: Deterministic probability-flow ODE trajectory (eta=0.0)

Compares:
    PSNR, LPIPS, SSIM, cos_peak, cos_{t=0}, collapse (cos_peak - cos_{t=0}).

Fork Decision:
    - If DDIM substantially reduces late collapse:
      Sampling stochasticity is a major part of the problem.
    - If the same collapse remains under DDIM:
      The learned denoising objective / score function parameterization
      is the stronger suspect.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from latentsr.metrics.collapse_geometry import collapse_from_cosine_curve
from latentsr.metrics.image_metrics import (
    LPIPSMetric,
    batch_metrics,
    dataset_filename,
    summarize_values,
)
from latentsr.metrics.paired_stats import (
    bootstrap_mean_ci,
    sign_flip_permutation_pvalue,
)
from latentsr.metrics.timestep_diagnostic import latent_cosine
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import (
    encode_lr_latents,
    save_sr_comparison_grid,
    upsample_bicubic,
)
from latentsr.super_resolution.sample import (
    _STEP_SALT,
    ddim_step,
    predict_x0_from_eps,
    seeded_noise_like,
)
from latentsr.vae.latent import decode_scaled, encode_scaled
from latentsr.vae.vae import VAE
from latentsr.vae.whitening import ChannelWhitening


@torch.no_grad()
def run_ddpm_vs_ddim_comparison(
    model: ConditionalLatentDDPM,
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int = 64,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    noise_seed: int = 42,
    start_index: int = 0,
    compute_lpips: bool = True,
    lpips_fn: LPIPSMetric | None = None,
    whitener: ChannelWhitening | None = None,
    show_progress: bool = True,
    grid_images: int = 8,
) -> dict[str, Any]:
    """Run paired reverse chains (DDPM vs DDIM) using identical x_T and seed.

    Returns dict containing per-image metrics, cosine trajectories,
    summary tables, and visual grids.
    """
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)

    num_t = int(model.num_timesteps)
    if compute_lpips and lpips_fn is None:
        lpips_fn = LPIPSMetric(net="alex", device=device)

    # Storage for both samplers
    samplers = ("ddpm", "ddim")
    curves: dict[str, list[torch.Tensor]] = {s: [] for s in samplers}
    metrics_per_sampler: dict[str, dict[str, list[float]]] = {
        s: {k: [] for k in ("psnr", "ssim", "lpips", "cos_peak", "cos_t0", "t_peak", "collapse", "z_err", "z_cos")}
        for s in samplers
    }
    per_image_rows: list[dict[str, Any]] = []

    grid_lr = None
    grid_hr = None
    grid_bicubic = None
    grid_soft = None
    grid_ddpm = None
    grid_ddim = None

    remaining = max(int(num_images), 1)
    next_index = int(start_index)
    dataset = loader.dataset

    pbar = (
        tqdm(
            total=remaining,
            desc="ddpm-vs-ddim",
            unit="img",
            leave=True,
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )

    for lr, hr in loader:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr_b = lr[:take].to(device)
        hr_b = hr[:take].to(device)
        batch_idx = list(range(next_index, next_index + take))

        # Ground truth clean latents and bicubic
        z_hr = encode_scaled(vae, hr_b, latent_scale=latent_scale)
        bicubic_b = upsample_bicubic(lr_b, hr_size)
        z_lr_raw = encode_lr_latents(
            vae,
            lr_b,
            hr_size=hr_size,
            latent_scale=latent_scale,
            apply_whiten=False,
        )
        z_lr_cond = whitener.transform(z_lr_raw) if whitener is not None else z_lr_raw
        soft_b = decode_scaled(vae, z_lr_raw, latent_scale=latent_scale).clamp(0.0, 1.0)

        # Same initial noise x_T for both DDPM and DDIM
        x_T = seeded_noise_like(z_lr_raw, batch_idx, base_seed=noise_seed, salt=0)

        batch_out: dict[str, dict[str, torch.Tensor]] = {}

        for sampler_name in samplers:
            x = x_T.clone()
            cos_hist = torch.empty(take, num_t, dtype=torch.float32)

            steps = range(num_t - 1, -1, -1)
            if show_progress:
                steps = tqdm(
                    steps,
                    desc=f"{sampler_name} reverse t",
                    unit="t",
                    leave=False,
                    dynamic_ncols=True,
                    mininterval=0.5,
                )

            for t in steps:
                t_batch = torch.full((take,), t, device=device, dtype=torch.long)
                eps = model.predict_noise(x, t_batch, z_lr_cond)
                z0 = predict_x0_from_eps(model.scheduler, x, t_batch, eps)
                cos_hist[:, t] = latent_cosine(z0, z_lr_cond).detach().cpu()

                if sampler_name == "ddim":
                    # Deterministic probability-flow ODE: eta=0.0, zero step noise
                    x = ddim_step(
                        model.scheduler,
                        x,
                        t_batch,
                        eps,
                        eta=0.0,
                        noise=None,
                    )
                else:
                    # Stochastic ancestral DDPM: eta=1.0, seeded step noise
                    step_noise = seeded_noise_like(
                        x,
                        batch_idx,
                        base_seed=noise_seed,
                        salt=_STEP_SALT * (int(t) + 1),
                    )
                    x = model.scheduler.p_sample_step(
                        x, t_batch, eps, noise=step_noise
                    )

            pred_img = decode_scaled(vae, x, latent_scale=latent_scale).clamp(0.0, 1.0)
            img_metrics = batch_metrics(
                pred_img,
                hr_b,
                lpips_fn=lpips_fn if compute_lpips else None,
            )
            coll = collapse_from_cosine_curve(cos_hist)
            z_err = (x - z_hr).pow(2).flatten(1).mean(dim=1).cpu()
            z_cos = latent_cosine(x, z_hr).cpu()

            curves[sampler_name].append(cos_hist)
            batch_out[sampler_name] = {
                "pred": pred_img,
                "cos_hist": cos_hist,
                "psnr": img_metrics["psnr"].cpu(),
                "ssim": img_metrics["ssim"].cpu(),
                "lpips": img_metrics.get("lpips", torch.zeros(take)).cpu(),
                "cos_peak": coll["cos_peak"].cpu(),
                "cos_t0": coll["cos_t0"].cpu(),
                "t_peak": coll["t_peak"].cpu().float(),
                "collapse": coll["collapse"].cpu(),
                "z_err": z_err,
                "z_cos": z_cos,
            }

            for k in ("psnr", "ssim", "lpips", "cos_peak", "cos_t0", "t_peak", "collapse", "z_err", "z_cos"):
                metrics_per_sampler[sampler_name][k].extend(
                    batch_out[sampler_name][k].tolist()
                )

        # Assemble per-image comparison rows
        for i, val_idx in enumerate(batch_idx):
            fn = dataset_filename(dataset, val_idx)
            row: dict[str, Any] = {"val_index": val_idx, "filename": fn}
            for s in samplers:
                for k in ("psnr", "ssim", "lpips", "cos_peak", "cos_t0", "t_peak", "collapse", "z_err", "z_cos"):
                    row[f"{s}_{k}"] = float(batch_out[s][k][i].item())

            # Paired deltas (DDIM - DDPM)
            row["delta_collapse"] = row["ddim_collapse"] - row["ddpm_collapse"]
            row["delta_psnr"] = row["ddim_psnr"] - row["ddpm_psnr"]
            row["delta_ssim"] = row["ddim_ssim"] - row["ddpm_ssim"]
            row["delta_lpips"] = row["ddim_lpips"] - row["ddpm_lpips"]
            row["delta_cos_peak"] = row["ddim_cos_peak"] - row["ddpm_cos_peak"]
            row["delta_cos_t0"] = row["ddim_cos_t0"] - row["ddpm_cos_t0"]
            per_image_rows.append(row)

        if grid_lr is None:
            n_grid = min(grid_images, take)
            grid_lr = lr_b[:n_grid].cpu()
            grid_hr = hr_b[:n_grid].cpu()
            grid_bicubic = bicubic_b[:n_grid].cpu()
            grid_soft = soft_b[:n_grid].cpu()
            grid_ddpm = batch_out["ddpm"]["pred"][:n_grid].cpu()
            grid_ddim = batch_out["ddim"]["pred"][:n_grid].cpu()

        remaining -= take
        next_index += take
        if pbar is not None:
            pbar.update(take)

    if pbar is not None:
        pbar.close()

    # Compile summaries
    summary: dict[str, Any] = {}
    for s in samplers:
        summary[s] = {
            k: summarize_values(metrics_per_sampler[s][k])
            for k in ("psnr", "ssim", "lpips", "cos_peak", "cos_t0", "t_peak", "collapse", "z_err", "z_cos")
        }

    # Paired delta analysis
    delta_keys = (
        "delta_collapse",
        "delta_psnr",
        "delta_ssim",
        "delta_lpips",
        "delta_cos_peak",
        "delta_cos_t0",
    )
    deltas: dict[str, Any] = {}
    for dk in delta_keys:
        vals = np.array([r[dk] for r in per_image_rows], dtype=np.float64)
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci_lo, ci_hi = bootstrap_mean_ci(vals, n_boot=2000, seed=noise_seed)
        p_val = sign_flip_permutation_pvalue(vals, n_perm=2000, seed=noise_seed)
        deltas[dk] = {
            "mean": m,
            "std": s,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "p_value": p_val,
        }

    # Fork Decision Evaluation
    mean_coll_ddpm = summary["ddpm"]["collapse"]["mean"]
    mean_coll_ddim = summary["ddim"]["collapse"]["mean"]
    delta_coll_mean = deltas["delta_collapse"]["mean"]
    delta_coll_ci_hi = deltas["delta_collapse"]["ci95_high"]
    coll_pct_reduction = (
        (mean_coll_ddpm - mean_coll_ddim) / max(mean_coll_ddpm, 1e-8) * 100.0
    )

    # Substantial reduction criteria: reduction >= 25% AND upper CI < 0.0
    if coll_pct_reduction >= 25.0 and delta_coll_ci_hi < 0.0:
        fork_outcome = "STOCHASTICITY_DRIVEN"
        fork_verdict = (
            f"DDIM substantially reduces late collapse by {coll_pct_reduction:.1f}% "
            f"({mean_coll_ddpm:.4f} -> {mean_coll_ddim:.4f}, Δ={delta_coll_mean:+.4f}, "
            f"95% CI [{deltas['delta_collapse']['ci95_low']:+.4f}, {delta_coll_ci_hi:+.4f}]). "
            "Conclusion: Sampling stochasticity (Brownian noise in ancestral DDPM) is a major "
            "driver of late trajectory drift. Deterministic probability-flow ODE keeps the trajectory on track."
        )
    else:
        fork_outcome = "OBJECTIVE_PARAMETERIZATION_DRIVEN"
        fork_verdict = (
            f"Late collapse persists under deterministic DDIM "
            f"(DDPM collapse: {mean_coll_ddpm:.4f}, DDIM collapse: {mean_coll_ddim:.4f}, "
            f"Δ={delta_coll_mean:+.4f}, reduction: {coll_pct_reduction:.1f}%). "
            "Conclusion: The learned score function / denoising parameterization is the primary root cause. "
            "Even along the deterministic ODE trajectory with zero noise, the UNet itself steers the latent "
            "state away from the condition at low noise timesteps."
        )

    cat_curves = {s: torch.cat(curves[s], dim=0) for s in samplers}

    return {
        "num_images": len(per_image_rows),
        "noise_seed": int(noise_seed),
        "summary": summary,
        "deltas": deltas,
        "fork_outcome": fork_outcome,
        "fork_verdict": fork_verdict,
        "per_image": per_image_rows,
        "curves": cat_curves,
        "grid_tensors": {
            "lr": grid_lr,
            "hr": grid_hr,
            "bicubic": grid_bicubic,
            "soft": grid_soft,
            "ddpm": grid_ddpm,
            "ddim": grid_ddim,
        },
    }


def format_ddpm_vs_ddim_table(result: dict[str, Any]) -> str:
    """Format side-by-side comparison table."""
    summary = result["summary"]
    deltas = result["deltas"]

    metrics = [
        ("PSNR (dB)", "psnr", "delta_psnr", True),
        ("SSIM", "ssim", "delta_ssim", True),
        ("LPIPS (alex)", "lpips", "delta_lpips", False),
        ("cos_peak", "cos_peak", "delta_cos_peak", True),
        ("cos_t0", "cos_t0", "delta_cos_t0", True),
        ("Collapse Score", "collapse", "delta_collapse", False),
        ("Latent MSE", "z_err", None, False),
        ("Latent Cosine", "z_cos", None, True),
    ]

    headers = ["Metric", "DDPM (eta=1)", "DDIM (eta=0)", "Δ (DDIM - DDPM)", "95% CI", "p-val"]
    widths = [18, 16, 16, 18, 20, 10]
    lines: list[str] = []
    lines.append(" | ".join(h.center(w) for h, w in zip(headers, widths)))
    lines.append("-+-".join("-" * w for w in widths))

    for name, key, dkey, higher_better in metrics:
        ddpm_m = summary["ddpm"][key]["mean"]
        ddpm_s = summary["ddpm"][key]["std"]
        ddim_m = summary["ddim"][key]["mean"]
        ddim_s = summary["ddim"][key]["std"]
        col_ddpm = f"{ddpm_m:.4f} ± {ddpm_s:.4f}"
        col_ddim = f"{ddim_m:.4f} ± {ddim_s:.4f}"

        if dkey and dkey in deltas:
            d_m = deltas[dkey]["mean"]
            d_ci_lo = deltas[dkey]["ci95_low"]
            d_ci_hi = deltas[dkey]["ci95_high"]
            pval = deltas[dkey]["p_value"]
            col_delta = f"{d_m:+.4f}"
            col_ci = f"[{d_ci_lo:+.4f}, {d_ci_hi:+.4f}]"
            col_p = f"{pval:.4f}" if pval >= 0.0001 else "<0.0001"
        else:
            col_delta = "-"
            col_ci = "-"
            col_p = "-"

        row = [name, col_ddpm, col_ddim, col_delta, col_ci, col_p]
        lines.append(" | ".join(f.rjust(w) for f, w in zip(row, widths)))

    return "\n".join(lines)


def generate_ddpm_vs_ddim_report(
    result: dict[str, Any],
    *,
    model_name: str = "Q2 Checkpoint",
) -> str:
    """Generate complete diagnostic text report with fork decision."""
    table_str = format_ddpm_vs_ddim_table(result)
    n = result.get("num_images", 0)
    seed = result.get("noise_seed", 42)
    fork_verdict = result.get("fork_verdict", "")
    fork_outcome = result.get("fork_outcome", "")

    report = f"""================================================================================
DIAGNOSTIC 2: DETERMINISTIC DDIM (eta=0) VS ANCESTRAL DDPM (eta=1)
Model: {model_name}
Evaluated Images: {n} | Base Seed: {seed}
Both samplers evaluated on the EXACT SAME initial noise x_T and validation images.
================================================================================

--- HEAD-TO-HEAD SUMMARY TABLE ---
{table_str}

--- FORK DECISION ANALYSIS ---
Outcome: {fork_outcome}

{fork_verdict}

Detailed Fork Implications:
  - If DDIM substantially reduces late collapse:
    Sampling stochasticity is a major part of the problem. Ancestral DDPM adds
    Brownian noise at every reverse step; near t=0, even small stochastic kicks
    can derail the trajectory away from the conditioning manifold. Switching to
    deterministic ODE integration (DDIM, DPM-Solver, or RiT flow matching)
    bypasses this noise accumulation.

  - If the same collapse remains under DDIM:
    The learned denoising objective / conditioning architecture is the stronger
    suspect. Because the deterministic probability flow ODE follows the vector
    field dx/dt = -0.5 * beta_t * [x + eps_theta(x, t, z_lr)], collapse under
    DDIM proves that the model's learned score eps_theta actively pushes the
    trajectory away from z_lr at low noise timesteps. Zero new sampler tuning
    will fix this without architectural or objective changes (e.g. conditioning
    preservation loss, AdaGN/cross-attention, or RiT).
================================================================================
"""
    return report


def plot_ddpm_vs_ddim_curves(
    result: dict[str, Any],
    output_path: Path | str,
    *,
    model_name: str = "Q2 Checkpoint",
) -> Path:
    """Plot mean +/- std cosine curves for DDPM vs DDIM overlaid on one graph."""
    ddpm_curves = result["curves"]["ddpm"].numpy()  # (N, T)
    ddim_curves = result["curves"]["ddim"].numpy()  # (N, T)
    num_t = ddpm_curves.shape[1]
    timesteps = np.arange(num_t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Cosine Trajectories Overlay
    ax = axes[0]
    m_ddpm = ddpm_curves.mean(axis=0)
    s_ddpm = ddpm_curves.std(axis=0)
    m_ddim = ddim_curves.mean(axis=0)
    s_ddim = ddim_curves.std(axis=0)

    ax.plot(timesteps, m_ddpm, color="#1f77b4", lw=2.2, label=f"DDPM (η=1, mean peak={m_ddpm.max():.3f}, t0={m_ddpm[0]:.3f})")
    ax.fill_between(timesteps, m_ddpm - s_ddpm, m_ddpm + s_ddpm, color="#1f77b4", alpha=0.18)

    ax.plot(timesteps, m_ddim, color="#d62728", lw=2.2, label=f"DDIM (η=0, mean peak={m_ddim.max():.3f}, t0={m_ddim[0]:.3f})")
    ax.fill_between(timesteps, m_ddim - s_ddim, m_ddim + s_ddim, color="#d62728", alpha=0.18)

    ax.set_xlabel("Diffusion Timestep t (0 = clean prediction, 1000 = noise)")
    ax.set_ylabel("Cosine Similarity: cos(ẑ0(x_t, t), z_lr)")
    ax.set_title("Trajectory of ẑ0 Alignment with Condition z_lr")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Panel 2: Per-Image Collapse Score Scatter: DDPM vs DDIM
    ax2 = axes[1]
    coll_ddpm = np.array([r["ddpm_collapse"] for r in result["per_image"]])
    coll_ddim = np.array([r["ddim_collapse"] for r in result["per_image"]])
    lim_max = max(float(coll_ddpm.max()), float(coll_ddim.max()), 0.4) + 0.05
    lim_min = min(float(coll_ddpm.min()), float(coll_ddim.min()), 0.0) - 0.05

    ax2.scatter(coll_ddpm, coll_ddim, alpha=0.75, c="#4c78a8", edgecolors="none", s=28)
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="y = x (no change)")
    ax2.set_xlim(lim_min, lim_max)
    ax2.set_ylim(lim_min, lim_max)
    ax2.set_xlabel("DDPM Collapse Score (cos_peak − cos_t0)")
    ax2.set_ylabel("DDIM Collapse Score (cos_peak − cos_t0)")
    ax2.set_title("Per-Image Collapse: DDPM vs DDIM")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    fig.suptitle(
        f"Diagnostic 2: DDPM (Stochastic) vs DDIM (Deterministic)\n{model_name} (Same x_T)",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_ddpm_vs_ddim_grid(
    grid_tensors: dict[str, torch.Tensor | None],
    output_path: Path | str,
    *,
    hr_size: int = 128,
) -> Path:
    """Save visual grid: LR | Soft-Decode | DDPM (eta=1) | DDIM (eta=0) | HR."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lr = grid_tensors["lr"]
    hr = grid_tensors["hr"]
    soft = grid_tensors["soft"]
    ddpm = grid_tensors["ddpm"]
    ddim = grid_tensors["ddim"]

    n = min(len(lr), 8)
    # 5 columns: LR (bicubic), Soft Decode, DDPM, DDIM, HR
    bicubic = upsample_bicubic(lr[:n], hr_size).cpu()
    cols = [
        bicubic,
        soft[:n].cpu() if soft is not None else bicubic,
        ddpm[:n].cpu(),
        ddim[:n].cpu(),
        hr[:n].cpu(),
    ]
    col_names = ["Bicubic LR", "Soft Decode", "DDPM (η=1)", "DDIM (η=0)", "Ground Truth HR"]

    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for r in range(n):
        for c in range(5):
            ax = axes[r, c]
            img = cols[c][r].permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0.0, 1.0))
            ax.axis("off")
            if r == 0:
                ax.set_title(col_names[c], fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_ddpm_vs_ddim_results(
    result: dict[str, Any],
    output_dir: Path | str,
    *,
    model_name: str = "Q2 Checkpoint",
    hr_size: int = 128,
) -> dict[str, Path]:
    """Save all Diagnostic 2 artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # 1. Summary JSON
    json_path = output_dir / "ddpm_vs_ddim_summary.json"
    summary_data = {
        "model_name": model_name,
        "num_images": result.get("num_images"),
        "noise_seed": result.get("noise_seed"),
        "summary": result.get("summary"),
        "deltas": result.get("deltas"),
        "fork_outcome": result.get("fork_outcome"),
        "fork_verdict": result.get("fork_verdict"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    saved["json"] = json_path

    # 2. Per-Image CSV
    csv_path = output_dir / "ddpm_vs_ddim_per_image.csv"
    per_image = result.get("per_image", [])
    if per_image:
        fieldnames = list(per_image[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image)
        saved["per_image_csv"] = csv_path

    # 3. Text Report
    report_path = output_dir / "ddpm_vs_ddim_report.txt"
    report_text = generate_ddpm_vs_ddim_report(result, model_name=model_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    saved["report"] = report_path

    # 4. Trajectory Plot
    plot_path = output_dir / "ddpm_vs_ddim_trajectories.png"
    plot_ddpm_vs_ddim_curves(result, plot_path, model_name=model_name)
    saved["trajectories_plot"] = plot_path

    # 5. Visual Grid
    if result.get("grid_tensors"):
        grid_path = output_dir / "ddpm_vs_ddim_compare_grid.png"
        save_ddpm_vs_ddim_grid(result["grid_tensors"], grid_path, hr_size=hr_size)
        saved["compare_grid"] = grid_path

    return saved
