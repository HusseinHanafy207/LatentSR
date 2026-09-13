"""Condition-switch rollout diagnostic (zero retraining).

Starts every reverse trajectory with the correct condition ``z_lr^true``.
At a switch timestep ``τ``, permanently replaces the condition with a
batch-shuffled ``z_lr^shuf`` for every remaining step ``t = τ, …, 0``.

Evaluated at
    τ ∈ {800, 650, 500, 300, 200, 100, 50}
plus an always-true baseline (never switch).

Measures final PSNR / LPIPS / SSIM and latent error vs ``z_hr``.

Interpretation:
    When the explicit condition ceases to be *causally necessary* —
    i.e. switching to a wrong condition after τ no longer hurts final quality —
    late reverse steps are no longer using ``z_lr`` to steer the sample.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from latentsr.metrics.condition_utility import shuffle_condition
from latentsr.metrics.image_metrics import (
    LPIPSMetric,
    batch_metrics,
    dataset_filename,
    summarize_values,
)
from latentsr.metrics.timestep_diagnostic import latent_cosine
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.super_resolution.sample import (
    _STEP_SALT,
    ddim_step,
    seeded_noise_like,
)
from latentsr.vae.latent import decode_scaled, encode_scaled
from latentsr.vae.vae import VAE
from latentsr.vae.whitening import ChannelWhitening

DEFAULT_SWITCH_TIMES: tuple[int, ...] = (800, 650, 500, 300, 200, 100, 50)
ALWAYS_TRUE_KEY = "always_true"


def _advance_step(
    model: ConditionalLatentDDPM,
    x: torch.Tensor,
    t: int,
    z_cond: torch.Tensor,
    *,
    batch_idx: Sequence[int],
    noise_seed: int,
    sampler: str,
    ddim_eta: float,
) -> torch.Tensor:
    """One reverse step with the given condition."""
    take = x.shape[0]
    device = x.device
    t_batch = torch.full((take,), t, device=device, dtype=torch.long)
    eps = model.predict_noise(x, t_batch, z_cond)
    if sampler == "ddim":
        step_noise = None
        if ddim_eta > 0.0:
            step_noise = seeded_noise_like(
                x,
                batch_idx,
                base_seed=noise_seed,
                salt=_STEP_SALT * (int(t) + 1),
            )
        return ddim_step(
            model.scheduler,
            x,
            t_batch,
            eps,
            eta=ddim_eta,
            noise=step_noise,
        )
    step_noise = seeded_noise_like(
        x,
        batch_idx,
        base_seed=noise_seed,
        salt=_STEP_SALT * (int(t) + 1),
    )
    return model.scheduler.p_sample_step(x, t_batch, eps, noise=step_noise)


def _latent_pair_metrics(
    z_pred: torch.Tensor,
    z_hr: torch.Tensor,
    z_lr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-image latent MSE / RMSE / cosine vs HR and LR."""
    mse_hr = (z_pred - z_hr).pow(2).flatten(1).mean(dim=1)
    return {
        "latent_mse_hr": mse_hr,
        "latent_rmse_hr": mse_hr.clamp_min(0.0).sqrt(),
        "latent_cos_hr": F.cosine_similarity(
            z_pred.flatten(1), z_hr.flatten(1), dim=1
        ),
        "latent_mse_lr": (z_pred - z_lr).pow(2).flatten(1).mean(dim=1),
        "latent_cos_lr": F.cosine_similarity(
            z_pred.flatten(1), z_lr.flatten(1), dim=1
        ),
    }


@torch.no_grad()
def run_condition_switch_rollout(
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
    switch_times: Sequence[int] = DEFAULT_SWITCH_TIMES,
    sampler: str = "ddpm",
    ddim_eta: float = 0.0,
    compute_lpips: bool = True,
    lpips_fn: LPIPSMetric | None = None,
    whitener: ChannelWhitening | None = None,
    show_progress: bool = True,
    grid_images: int = 4,
) -> dict[str, Any]:
    """Run condition-switch rollouts for each ``τ`` plus always-true baseline.

    For each switch time ``τ``:
      - Steps ``t > τ`` use ``z_lr^true``
      - Steps ``t ≤ τ`` permanently use ``z_lr^shuffled``

    Prefixes before the earliest remaining switch are shared via checkpoints
    of the always-true trajectory, so each batch pays for one full true
    reverse plus ``|τ|`` short shuffled suffixes.
    """
    sampler = sampler.lower().strip()
    if sampler not in ("ddpm", "ddim"):
        raise ValueError(f"Unknown sampler '{sampler}', expected 'ddpm' or 'ddim'")

    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)

    num_t = int(model.num_timesteps)
    switch_list = sorted(
        {int(t) for t in switch_times if 0 <= int(t) < num_t},
        reverse=True,
    )
    if not switch_list:
        raise ValueError(
            f"No valid switch times in {list(switch_times)} for T={num_t}"
        )
    switch_set = set(switch_list)
    condition_keys = [ALWAYS_TRUE_KEY] + [f"tau_{t}" for t in switch_list]

    if compute_lpips and lpips_fn is None:
        lpips_fn = LPIPSMetric(net="alex", device=device)

    metric_names = (
        "psnr",
        "ssim",
        "lpips",
        "latent_mse_hr",
        "latent_rmse_hr",
        "latent_cos_hr",
        "latent_mse_lr",
        "latent_cos_lr",
    )
    scores: dict[str, dict[str, list[float]]] = {
        k: {m: [] for m in metric_names} for k in condition_keys
    }
    per_image_rows: list[dict[str, Any]] = []
    grid_tensors: dict[str, torch.Tensor | None] = {
        "lr": None,
        "hr": None,
        ALWAYS_TRUE_KEY: None,
        **{f"tau_{t}": None for t in switch_list},
    }

    remaining = max(int(num_images), 2)
    next_index = int(start_index)
    dataset = loader.dataset
    total_images = 0

    pbar = (
        tqdm(
            total=remaining,
            desc=f"cond-switch ({sampler})",
            unit="img",
            leave=True,
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )

    for lr, hr in loader:
        if remaining <= 1:
            break
        take = min(lr.shape[0], remaining)
        if take < 2:
            break

        lr_b = lr[:take].to(device)
        hr_b = hr[:take].to(device)
        batch_idx = list(range(next_index, next_index + take))

        z_hr = encode_scaled(vae, hr_b, latent_scale=latent_scale)
        z_lr_raw = encode_lr_latents(
            vae,
            lr_b,
            hr_size=hr_size,
            latent_scale=latent_scale,
            apply_whiten=False,
        )
        z_lr_true = whitener.transform(z_lr_raw) if whitener is not None else z_lr_raw
        z_lr_shuf = shuffle_condition(z_lr_true, shift=1)

        x_T = seeded_noise_like(z_lr_raw, batch_idx, base_seed=noise_seed, salt=0)

        # --- Always-true reverse; checkpoint x just before each switch τ ---
        x = x_T.clone()
        checkpoints: dict[int, torch.Tensor] = {}
        steps = range(num_t - 1, -1, -1)
        if show_progress:
            steps = tqdm(
                steps,
                desc="always-true reverse",
                unit="t",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )
        for t in steps:
            if t in switch_set:
                checkpoints[t] = x.clone()
            x = _advance_step(
                model,
                x,
                t,
                z_lr_true,
                batch_idx=batch_idx,
                noise_seed=noise_seed,
                sampler=sampler,
                ddim_eta=ddim_eta,
            )
        finals: dict[str, torch.Tensor] = {ALWAYS_TRUE_KEY: x}

        # --- Fork at each τ: continue with permanently shuffled condition ---
        tau_iter = switch_list
        if show_progress:
            tau_iter = tqdm(
                switch_list,
                desc="switch forks",
                unit="τ",
                leave=False,
                dynamic_ncols=True,
            )
        for tau in tau_iter:
            x_sw = checkpoints[tau].clone()
            for t in range(tau, -1, -1):
                x_sw = _advance_step(
                    model,
                    x_sw,
                    t,
                    z_lr_shuf,
                    batch_idx=batch_idx,
                    noise_seed=noise_seed,
                    sampler=sampler,
                    ddim_eta=ddim_eta,
                )
            finals[f"tau_{tau}"] = x_sw

        # --- Decode + score ---
        batch_metrics_by_key: dict[str, dict[str, torch.Tensor]] = {}
        for key, z_final in finals.items():
            pred = decode_scaled(vae, z_final, latent_scale=latent_scale).clamp(0.0, 1.0)
            img = batch_metrics(
                pred,
                hr_b,
                lpips_fn=lpips_fn if compute_lpips else None,
            )
            lat = _latent_pair_metrics(z_final, z_hr, z_lr_true)
            packed = {
                "psnr": img["psnr"].detach().cpu(),
                "ssim": img["ssim"].detach().cpu(),
                "lpips": img.get("lpips", torch.zeros(take)).detach().cpu(),
                "latent_mse_hr": lat["latent_mse_hr"].detach().cpu(),
                "latent_rmse_hr": lat["latent_rmse_hr"].detach().cpu(),
                "latent_cos_hr": lat["latent_cos_hr"].detach().cpu(),
                "latent_mse_lr": lat["latent_mse_lr"].detach().cpu(),
                "latent_cos_lr": lat["latent_cos_lr"].detach().cpu(),
                "pred": pred.detach().cpu(),
            }
            batch_metrics_by_key[key] = packed
            for m in metric_names:
                scores[key][m].extend(packed[m].tolist())

        for i, val_idx in enumerate(batch_idx):
            row: dict[str, Any] = {
                "val_index": int(val_idx),
                "filename": dataset_filename(dataset, int(val_idx)),
            }
            for key in condition_keys:
                for m in metric_names:
                    row[f"{key}_{m}"] = float(batch_metrics_by_key[key][m][i].item())
            # Deltas vs always-true (negative PSNR delta = quality loss from switch)
            base_psnr = row[f"{ALWAYS_TRUE_KEY}_psnr"]
            base_lpips = row[f"{ALWAYS_TRUE_KEY}_lpips"]
            base_mse = row[f"{ALWAYS_TRUE_KEY}_latent_mse_hr"]
            for tau in switch_list:
                key = f"tau_{tau}"
                row[f"{key}_delta_psnr"] = row[f"{key}_psnr"] - base_psnr
                row[f"{key}_delta_lpips"] = row[f"{key}_lpips"] - base_lpips
                row[f"{key}_delta_latent_mse_hr"] = (
                    row[f"{key}_latent_mse_hr"] - base_mse
                )
            per_image_rows.append(row)

        if grid_tensors["lr"] is None:
            n_grid = min(grid_images, take)
            grid_tensors["lr"] = lr_b[:n_grid].cpu()
            grid_tensors["hr"] = hr_b[:n_grid].cpu()
            for key in condition_keys:
                grid_tensors[key] = batch_metrics_by_key[key]["pred"][:n_grid]

        total_images += take
        remaining -= take
        next_index += take
        if pbar is not None:
            pbar.update(take)

    if pbar is not None:
        pbar.close()
    if total_images < 2:
        raise ValueError("Need at least 2 images for condition shuffling.")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for key in condition_keys:
        summary[key] = {
            m: summarize_values(scores[key][m]) for m in metric_names
        }

    # Mean deltas vs always-true across switch times
    delta_summary: list[dict[str, Any]] = []
    for tau in switch_list:
        key = f"tau_{tau}"
        d_psnr = np.array(
            [r[f"{key}_delta_psnr"] for r in per_image_rows], dtype=np.float64
        )
        d_lpips = np.array(
            [r[f"{key}_delta_lpips"] for r in per_image_rows], dtype=np.float64
        )
        d_mse = np.array(
            [r[f"{key}_delta_latent_mse_hr"] for r in per_image_rows],
            dtype=np.float64,
        )
        delta_summary.append(
            {
                "tau": tau,
                "psnr_mean": summary[key]["psnr"]["mean"],
                "psnr_std": summary[key]["psnr"]["std"],
                "lpips_mean": summary[key]["lpips"]["mean"],
                "lpips_std": summary[key]["lpips"]["std"],
                "latent_mse_hr_mean": summary[key]["latent_mse_hr"]["mean"],
                "latent_rmse_hr_mean": summary[key]["latent_rmse_hr"]["mean"],
                "latent_cos_hr_mean": summary[key]["latent_cos_hr"]["mean"],
                "latent_cos_lr_mean": summary[key]["latent_cos_lr"]["mean"],
                "delta_psnr_mean": float(d_psnr.mean()),
                "delta_psnr_std": float(d_psnr.std(ddof=1)) if len(d_psnr) > 1 else 0.0,
                "delta_lpips_mean": float(d_lpips.mean()),
                "delta_lpips_std": float(d_lpips.std(ddof=1))
                if len(d_lpips) > 1
                else 0.0,
                "delta_latent_mse_hr_mean": float(d_mse.mean()),
                "delta_latent_mse_hr_std": float(d_mse.std(ddof=1))
                if len(d_mse) > 1
                else 0.0,
            }
        )

    return {
        "num_images": total_images,
        "noise_seed": int(noise_seed),
        "sampler": sampler,
        "ddim_eta": float(ddim_eta),
        "switch_times": switch_list,
        "condition_keys": condition_keys,
        "summary": summary,
        "delta_summary": delta_summary,
        "per_image": per_image_rows,
        "grid_tensors": grid_tensors,
        "always_true_key": ALWAYS_TRUE_KEY,
    }


def format_condition_switch_table(result: dict[str, Any]) -> str:
    """Aligned text table of always-true + each switch τ."""
    summary = result["summary"]
    deltas = {row["tau"]: row for row in result["delta_summary"]}
    base = summary[ALWAYS_TRUE_KEY]

    headers = [
        "τ (switch)",
        "PSNR",
        "ΔPSNR",
        "LPIPS",
        "ΔLPIPS",
        "Latent MSE",
        "ΔLatent MSE",
        "cos(ẑ, z_hr)",
        "cos(ẑ, z_lr)",
    ]
    widths = [12, 10, 10, 10, 10, 12, 12, 14, 14]
    lines = [
        " | ".join(h.center(w) for h, w in zip(headers, widths)),
        "-+-".join("-" * w for w in widths),
    ]

    def _row(label: str, psnr_m: float, d_psnr: str, lpips_m: float, d_lpips: str,
             mse_m: float, d_mse: str, cos_hr: float, cos_lr: float) -> str:
        fields = [
            label,
            f"{psnr_m:.3f}",
            d_psnr,
            f"{lpips_m:.4f}",
            d_lpips,
            f"{mse_m:.5f}",
            d_mse,
            f"{cos_hr:.4f}",
            f"{cos_lr:.4f}",
        ]
        return " | ".join(f.rjust(w) for f, w in zip(fields, widths))

    lines.append(
        _row(
            "always_true",
            base["psnr"]["mean"],
            "  —",
            base["lpips"]["mean"],
            "  —",
            base["latent_mse_hr"]["mean"],
            "  —",
            base["latent_cos_hr"]["mean"],
            base["latent_cos_lr"]["mean"],
        )
    )
    for tau in result["switch_times"]:
        key = f"tau_{tau}"
        s = summary[key]
        d = deltas[tau]
        lines.append(
            _row(
                str(tau),
                s["psnr"]["mean"],
                f"{d['delta_psnr_mean']:+.3f}",
                s["lpips"]["mean"],
                f"{d['delta_lpips_mean']:+.4f}",
                s["latent_mse_hr"]["mean"],
                f"{d['delta_latent_mse_hr_mean']:+.5f}",
                s["latent_cos_hr"]["mean"],
                s["latent_cos_lr"]["mean"],
            )
        )
    return "\n".join(lines)


def generate_condition_switch_report(
    result: dict[str, Any],
    *,
    model_name: str = "Q2 Checkpoint",
) -> str:
    """Text report: when does the condition stop being causally necessary?"""
    table = format_condition_switch_table(result)
    deltas = result["delta_summary"]
    n = result["num_images"]
    sampler = result["sampler"]

    # Find earliest (largest τ) where |ΔPSNR| is small → condition no longer needed
    # and latest (smallest τ) where switching still hurts substantially.
    # Threshold: ΔPSNR > -0.5 dB means switch barely hurts.
    causal_cutoff = None
    for row in sorted(deltas, key=lambda r: r["tau"], reverse=True):
        if row["delta_psnr_mean"] > -0.5:
            causal_cutoff = row["tau"]
            break

    if causal_cutoff is None:
        verdict = (
            "Switching the condition at every tested τ still hurts final quality "
            "substantially (ΔPSNR ≤ −0.5 dB). The explicit condition remains causally "
            "necessary deep into the reverse trajectory."
        )
    elif causal_cutoff == max(r["tau"] for r in deltas):
        verdict = (
            f"Even switching as early as τ={causal_cutoff} barely hurts "
            f"(ΔPSNR > −0.5 dB). The reverse chain largely ignores z_lr after "
            f"the noisiest steps — condition ceases to be causally necessary early."
        )
    else:
        hurt = [r for r in deltas if r["tau"] > causal_cutoff]
        hurt_str = (
            ", ".join(str(r["tau"]) for r in sorted(hurt, key=lambda r: -r["tau"]))
            if hurt
            else "none"
        )
        verdict = (
            f"Switching still hurts for τ ∈ {{{hurt_str}}}, but by τ={causal_cutoff} "
            f"(and later / smaller τ) the quality drop is small (ΔPSNR > −0.5 dB). "
            f"The explicit condition ceases to be causally necessary around τ≈{causal_cutoff}."
        )

    return f"""================================================================================
CONDITION-SWITCH ROLLOUT DIAGNOSTIC
Model: {model_name}
Images: {n} | Sampler: {sampler} | Seed: {result.get('noise_seed', 42)}
================================================================================

Protocol:
  Start every sample with z_lr^true.
  At switch timestep τ, permanently replace with z_lr^shuffled for all t ≤ τ.
  Switch times: {result['switch_times']}
  Baseline: always_true (never switch).

This asks when the explicit condition ceases to be *causally necessary*,
rather than testing condition sensitivity at one isolated timestep.

--- RESULTS ---
{table}

Δ columns are vs always_true (same x_T, same images).
  ΔPSNR < 0  → switching hurt reconstruction quality
  ΔLPIPS > 0 → switching hurt perceptual quality
  ΔLatent MSE > 0 → switching increased error to z_hr

--- VERDICT ---
{verdict}
================================================================================
"""


def plot_condition_switch(
    result: dict[str, Any],
    output_path: Path | str,
    *,
    model_name: str = "Q2 Checkpoint",
) -> Path:
    """Plot PSNR / LPIPS / latent MSE vs switch time τ."""
    deltas = sorted(result["delta_summary"], key=lambda r: r["tau"], reverse=True)
    taus = [r["tau"] for r in deltas]
    base = result["summary"][ALWAYS_TRUE_KEY]

    psnr_vals = [r["psnr_mean"] for r in deltas]
    lpips_vals = [r["lpips_mean"] for r in deltas]
    mse_vals = [r["latent_mse_hr_mean"] for r in deltas]
    d_psnr = [r["delta_psnr_mean"] for r in deltas]
    d_lpips = [r["delta_lpips_mean"] for r in deltas]
    d_mse = [r["delta_latent_mse_hr_mean"] for r in deltas]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(taus, psnr_vals, "o-", color="#1f77b4", lw=2, label="switch at τ")
    ax.axhline(
        base["psnr"]["mean"],
        color="#2ca02c",
        ls="--",
        lw=1.5,
        label=f"always_true ({base['psnr']['mean']:.2f} dB)",
    )
    ax.set_xlabel("Switch timestep τ (larger = switch earlier / noisier)")
    ax.set_ylabel("Final PSNR (dB)")
    ax.set_title("1. Final PSNR vs Condition-Switch Time")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(taus, lpips_vals, "o-", color="#d62728", lw=2, label="switch at τ")
    ax.axhline(
        base["lpips"]["mean"],
        color="#2ca02c",
        ls="--",
        lw=1.5,
        label=f"always_true ({base['lpips']['mean']:.4f})",
    )
    ax.set_xlabel("Switch timestep τ")
    ax.set_ylabel("Final LPIPS (↓ better)")
    ax.set_title("2. Final LPIPS vs Condition-Switch Time")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(taus, mse_vals, "o-", color="#9467bd", lw=2, label="switch at τ")
    ax.axhline(
        base["latent_mse_hr"]["mean"],
        color="#2ca02c",
        ls="--",
        lw=1.5,
        label="always_true",
    )
    ax.set_xlabel("Switch timestep τ")
    ax.set_ylabel("||ẑ_final − z_hr||²")
    ax.set_title("3. Latent Error to z_hr vs Switch Time")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(taus, d_psnr, "o-", color="#1f77b4", lw=2, label="ΔPSNR")
    ax.plot(taus, [10 * d for d in d_lpips], "s-", color="#d62728", lw=2, label="10×ΔLPIPS")
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Switch timestep τ")
    ax.set_ylabel("Δ vs always_true")
    ax.set_title("4. Causal Cost of Switching (ΔPSNR, 10×ΔLPIPS)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle(
        f"Condition-Switch Rollout\n{model_name} ({result['sampler']})",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_condition_switch_grid(
    grid_tensors: dict[str, torch.Tensor | None],
    switch_times: Sequence[int],
    output_path: Path | str,
) -> Path | None:
    """Visual grid: LR | always_true | switch@τ… | HR (subset of τ for readability)."""
    if grid_tensors.get("lr") is None or grid_tensors.get("hr") is None:
        return None

    # Show always_true + up to 4 representative switch times + HR
    show_taus = list(switch_times)
    if len(show_taus) > 4:
        # Pick evenly spaced across the list (already sorted descending)
        idxs = np.linspace(0, len(show_taus) - 1, 4).round().astype(int)
        show_taus = [show_taus[i] for i in idxs]

    col_keys = [ALWAYS_TRUE_KEY] + [f"tau_{t}" for t in show_taus]
    col_labels = ["Always True"] + [f"Switch τ={t}" for t in show_taus]

    lr = grid_tensors["lr"]
    hr = grid_tensors["hr"]
    assert lr is not None and hr is not None
    n = lr.shape[0]
    n_cols = 2 + len(col_keys)  # LR + conditions + HR

    fig, axes = plt.subplots(n, n_cols, figsize=(2.4 * n_cols, 2.4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    def _show(ax: Any, img: torch.Tensor, title: str | None = None) -> None:
        ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
        ax.axis("off")
        if title is not None:
            ax.set_title(title, fontsize=9, fontweight="bold")

    for r in range(n):
        _show(axes[r, 0], lr[r], "LR" if r == 0 else None)
        for c, (key, label) in enumerate(zip(col_keys, col_labels)):
            img = grid_tensors.get(key)
            if img is not None:
                _show(axes[r, 1 + c], img[r], label if r == 0 else None)
            else:
                axes[r, 1 + c].axis("off")
        _show(axes[r, -1], hr[r], "HR" if r == 0 else None)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def save_condition_switch_results(
    result: dict[str, Any],
    output_dir: Path | str,
    *,
    model_name: str = "Q2 Checkpoint",
) -> dict[str, Path]:
    """Save JSON, CSVs, report, plot, and optional visual grid."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    json_path = output_dir / "condition_switch_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "num_images": result["num_images"],
                "noise_seed": result["noise_seed"],
                "sampler": result["sampler"],
                "ddim_eta": result["ddim_eta"],
                "switch_times": result["switch_times"],
                "summary": result["summary"],
                "delta_summary": result["delta_summary"],
            },
            f,
            indent=2,
        )
    saved["json"] = json_path

    delta_csv = output_dir / "condition_switch_by_tau.csv"
    if result["delta_summary"]:
        fieldnames = list(result["delta_summary"][0].keys())
        # Prepend always_true row for convenience
        base = result["summary"][ALWAYS_TRUE_KEY]
        base_row = {
            "tau": "always_true",
            "psnr_mean": base["psnr"]["mean"],
            "psnr_std": base["psnr"]["std"],
            "lpips_mean": base["lpips"]["mean"],
            "lpips_std": base["lpips"]["std"],
            "latent_mse_hr_mean": base["latent_mse_hr"]["mean"],
            "latent_rmse_hr_mean": base["latent_rmse_hr"]["mean"],
            "latent_cos_hr_mean": base["latent_cos_hr"]["mean"],
            "latent_cos_lr_mean": base["latent_cos_lr"]["mean"],
            "delta_psnr_mean": 0.0,
            "delta_psnr_std": 0.0,
            "delta_lpips_mean": 0.0,
            "delta_lpips_std": 0.0,
            "delta_latent_mse_hr_mean": 0.0,
            "delta_latent_mse_hr_std": 0.0,
        }
        rows = [base_row] + list(result["delta_summary"])
        with open(delta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        saved["by_tau_csv"] = delta_csv

    per_csv = output_dir / "condition_switch_per_image.csv"
    if result["per_image"]:
        fieldnames = list(result["per_image"][0].keys())
        with open(per_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result["per_image"])
        saved["per_image_csv"] = per_csv

    report_path = output_dir / "condition_switch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(generate_condition_switch_report(result, model_name=model_name))
    saved["report"] = report_path

    plot_path = output_dir / "condition_switch_plot.png"
    plot_condition_switch(result, plot_path, model_name=model_name)
    saved["plot"] = plot_path

    grid_path = save_condition_switch_grid(
        result.get("grid_tensors") or {},
        result["switch_times"],
        output_dir / "condition_switch_grid.png",
    )
    if grid_path is not None:
        saved["grid"] = grid_path

    return saved
