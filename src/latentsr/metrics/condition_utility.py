"""Diagnostic 1: Condition Utility as a function of diffusion timestep.

Evaluates:
    epsilon_theta(z_t, z_lr^true, t)  vs  epsilon_theta(z_t, z_lr^shuffled, t)

where z_lr^shuffled comes from another image in the batch (derangement via roll).

Measures across timesteps (e.g. t in {999, 800, 650, 500, 300, 100, 0}):
    1. epsilon-MSE and epsilon-RMSE: ||eps_true - eps_shuf||^2
    2. z0_hat-MSE and z0_hat-RMSE: ||z0_hat_true - z0_hat_shuf||^2
    3. Prediction Cosine: cos(z0_hat_true, z0_hat_shuf)
    4. Condition Specificity Gap: cos(z0_hat_true, z_lr^true) - cos(z0_hat_shuf, z_lr^true)
    5. Ground-Truth Advantage (when z_hr is available):
       ||z0_hat_shuf - z_hr||^2 - ||z0_hat_true - z_hr||^2

Directly tests the hypothesis:
    Condition utility is strong at noisy/mid timesteps and drops toward t=0,
    demonstrating whether the model ignores the LR condition near t=0.
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

from latentsr.metrics.image_metrics import summarize_values
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.super_resolution.sample import (
    _STEP_SALT,
    ddim_step,
    predict_x0_from_eps,
    seeded_noise_like,
)
from latentsr.vae.latent import encode_scaled
from latentsr.vae.vae import VAE
from latentsr.vae.whitening import ChannelWhitening

DEFAULT_MILESTONES: tuple[int, ...] = (999, 800, 650, 500, 300, 100, 0)


def shuffle_condition(z_lr: torch.Tensor, shift: int = 1) -> torch.Tensor:
    """Create a shuffled condition bank with guaranteed derangement.

    Every sample in the batch gets a condition from a different image:
        z_lr_shuffled[i] = z_lr[(i - shift) % B]

    Args:
        z_lr: Condition latents of shape ``(B, C, H, W)``.
        shift: Circular shift offset (default 1).

    Returns:
        Tensor of shape ``(B, C, H, W)`` where no element receives its own condition
        (guaranteed when B >= 2 and shift % B != 0).
    """
    batch_size = z_lr.shape[0]
    if batch_size < 2:
        raise ValueError(
            f"Cannot shuffle condition within batch of size {batch_size}. "
            "Batch size must be >= 2 for condition utility diagnostic."
        )
    eff_shift = shift % batch_size
    if eff_shift == 0:
        eff_shift = 1
    return torch.roll(z_lr, shifts=eff_shift, dims=0)


def compute_step_condition_utility(
    scheduler: Any,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_true: torch.Tensor,
    eps_shuf: torch.Tensor,
    z_lr_true: torch.Tensor,
    z_lr_shuf: torch.Tensor,
    *,
    z_hr: torch.Tensor | None = None,
    eps_gt: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-image condition utility metrics at timestep t.

    Returns dictionary of 1D tensors of shape ``(B,)`` with per-image metrics.
    """
    z0_true = predict_x0_from_eps(scheduler, x_t, t, eps_true)
    z0_shuf = predict_x0_from_eps(scheduler, x_t, t, eps_shuf)

    # 1. Epsilon differences
    eps_diff = eps_true - eps_shuf
    eps_mse = eps_diff.pow(2).flatten(1).mean(dim=1)
    eps_rmse = eps_mse.clamp_min(0.0).sqrt()
    eps_true_norm = eps_true.flatten(1).norm(dim=1).clamp_min(1e-8)
    eps_rel_diff = eps_diff.flatten(1).norm(dim=1) / eps_true_norm
    eps_cos = F.cosine_similarity(eps_true.flatten(1), eps_shuf.flatten(1), dim=1)

    # 2. Predicted z0 differences
    z0_diff = z0_true - z0_shuf
    z0_mse = z0_diff.pow(2).flatten(1).mean(dim=1)
    z0_rmse = z0_mse.clamp_min(0.0).sqrt()
    z0_cos = F.cosine_similarity(z0_true.flatten(1), z0_shuf.flatten(1), dim=1)

    # 3. Alignment with condition
    cos_z0_true_lr_true = F.cosine_similarity(
        z0_true.flatten(1), z_lr_true.flatten(1), dim=1
    )
    cos_z0_shuf_lr_true = F.cosine_similarity(
        z0_shuf.flatten(1), z_lr_true.flatten(1), dim=1
    )
    cos_z0_shuf_lr_shuf = F.cosine_similarity(
        z0_shuf.flatten(1), z_lr_shuf.flatten(1), dim=1
    )
    specificity_gap = cos_z0_true_lr_true - cos_z0_shuf_lr_true

    out: dict[str, torch.Tensor] = {
        "eps_mse": eps_mse,
        "eps_rmse": eps_rmse,
        "eps_rel_diff": eps_rel_diff,
        "eps_cos": eps_cos,
        "z0_mse": z0_mse,
        "z0_rmse": z0_rmse,
        "z0_cos": z0_cos,
        "cos_z0_true_lr_true": cos_z0_true_lr_true,
        "cos_z0_shuf_lr_true": cos_z0_shuf_lr_true,
        "cos_z0_shuf_lr_shuf": cos_z0_shuf_lr_shuf,
        "specificity_gap": specificity_gap,
    }

    # 4. Ground-truth latent advantage (if z_hr available)
    if z_hr is not None:
        err_true = (z0_true - z_hr).pow(2).flatten(1).mean(dim=1)
        err_shuf = (z0_shuf - z_hr).pow(2).flatten(1).mean(dim=1)
        # Positive advantage => true condition yields lower error to clean z_hr
        out["z0_err_true"] = err_true
        out["z0_err_shuf"] = err_shuf
        out["z0_condition_advantage"] = err_shuf - err_true
        out["cos_z0_true_hr"] = F.cosine_similarity(
            z0_true.flatten(1), z_hr.flatten(1), dim=1
        )
        out["cos_z0_shuf_hr"] = F.cosine_similarity(
            z0_shuf.flatten(1), z_hr.flatten(1), dim=1
        )

    # 5. Ground-truth noise advantage (if eps_gt available, e.g. in forward q-sample)
    if eps_gt is not None:
        eps_err_true = (eps_true - eps_gt).pow(2).flatten(1).mean(dim=1)
        eps_err_shuf = (eps_shuf - eps_gt).pow(2).flatten(1).mean(dim=1)
        out["eps_err_true"] = eps_err_true
        out["eps_err_shuf"] = eps_err_shuf
        out["eps_condition_advantage"] = eps_err_shuf - eps_err_true

    return out


class _UtilityAccumulator:
    """Accumulates running sums and sums of squares for timestep metrics."""

    def __init__(self, num_timesteps: int, metric_keys: Sequence[str]) -> None:
        self.num_timesteps = int(num_timesteps)
        self.keys = tuple(metric_keys)
        self.counts = [0] * self.num_timesteps
        self._sum = {k: torch.zeros(self.num_timesteps, dtype=torch.float64) for k in self.keys}
        self._sq = {k: torch.zeros(self.num_timesteps, dtype=torch.float64) for k in self.keys}

    def add_batch(self, t: int, metrics: dict[str, torch.Tensor]) -> None:
        batch_n = 0
        for k in self.keys:
            if k in metrics:
                vals = metrics[k].detach().reshape(-1).double().cpu()
                self._sum[k][t] += vals.sum()
                self._sq[k][t] += vals.pow(2).sum()
                batch_n = len(vals)
        self.counts[t] += batch_n

    def compute_summary(self) -> dict[str, dict[str, list[float]]]:
        """Returns mean and std per metric across all timesteps."""
        out: dict[str, dict[str, list[float]]] = {}
        for k in self.keys:
            means: list[float] = []
            stds: list[float] = []
            for t in range(self.num_timesteps):
                n = self.counts[t]
                if n > 0:
                    m = float(self._sum[k][t] / n)
                    var = float(self._sq[k][t] / n) - (m ** 2)
                    s = float(np.sqrt(max(var, 0.0)))
                else:
                    m = 0.0
                    s = 0.0
                means.append(m)
                stds.append(s)
            out[k] = {"mean": means, "std": stds}
        return out


@torch.no_grad()
def evaluate_condition_utility_reverse(
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
    sampler: str = "ddpm",
    ddim_eta: float = 0.0,
    show_progress: bool = True,
    whitener: ChannelWhitening | None = None,
    milestones: Sequence[int] = DEFAULT_MILESTONES,
    eval_all_timesteps: bool = True,
) -> dict[str, Any]:
    """Measure condition utility along the actual reverse sampling trajectory.

    At each timestep t (or milestones), evaluates:
        eps_true = eps_theta(z_t, z_lr_true, t)
        eps_shuf = eps_theta(z_t, z_lr_shuffled, t)
    and measures ||eps_true - eps_shuf||^2 and ||z0_true - z0_shuf||^2.
    Trajectory advances using eps_true.
    """
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)

    num_t = int(model.num_timesteps)
    milestone_set = {int(m) for m in milestones if 0 <= int(m) < num_t}
    target_timesteps = set(range(num_t)) if eval_all_timesteps else milestone_set

    metric_keys = (
        "eps_mse",
        "eps_rmse",
        "eps_rel_diff",
        "eps_cos",
        "z0_mse",
        "z0_rmse",
        "z0_cos",
        "cos_z0_true_lr_true",
        "cos_z0_shuf_lr_true",
        "cos_z0_shuf_lr_shuf",
        "specificity_gap",
        "z0_err_true",
        "z0_err_shuf",
        "z0_condition_advantage",
        "cos_z0_true_hr",
        "cos_z0_shuf_hr",
    )
    accum = _UtilityAccumulator(num_t, metric_keys)

    # For milestone detailed table
    milestone_per_image: dict[int, list[dict[str, float]]] = {
        m: [] for m in sorted(milestone_set, reverse=True)
    }

    remaining = max(int(num_images), 2)
    next_index = int(start_index)
    total_images_processed = 0

    pbar = (
        tqdm(
            total=remaining,
            desc=f"cond-utility ({sampler})",
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
            # Need at least 2 images for derangement shuffle
            break
        lr = lr[:take].to(device)
        hr_b = hr[:take].to(device)
        batch_idx = list(range(next_index, next_index + take))

        # Encode ground truth and condition
        z_hr = encode_scaled(vae, hr_b, latent_scale=latent_scale)
        z_lr_raw = encode_lr_latents(
            vae,
            lr,
            hr_size=hr_size,
            latent_scale=latent_scale,
            apply_whiten=False,
        )
        z_lr_true = whitener.transform(z_lr_raw) if whitener is not None else z_lr_raw
        z_lr_shuf = shuffle_condition(z_lr_true, shift=1)

        # Initial noise x_T
        x = seeded_noise_like(z_lr_raw, batch_idx, base_seed=noise_seed, salt=0)

        steps = range(num_t - 1, -1, -1)
        if show_progress:
            steps = tqdm(
                steps,
                desc=f"reverse t ({sampler})",
                unit="t",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )

        for t in steps:
            t_batch = torch.full((take,), t, device=device, dtype=torch.long)
            eps_true = model.predict_noise(x, t_batch, z_lr_true)

            if t in target_timesteps:
                eps_shuf = model.predict_noise(x, t_batch, z_lr_shuf)
                step_metrics = compute_step_condition_utility(
                    model.scheduler,
                    x,
                    t_batch,
                    eps_true,
                    eps_shuf,
                    z_lr_true,
                    z_lr_shuf,
                    z_hr=z_hr,
                )
                accum.add_batch(t, step_metrics)

                if t in milestone_set:
                    for i, img_idx in enumerate(batch_idx):
                        row = {"val_index": img_idx, "t": t}
                        for k, v in step_metrics.items():
                            row[k] = float(v[i].item())
                        milestone_per_image[t].append(row)

            # Advance reverse trajectory with eps_true
            if sampler == "ddim":
                step_noise = None
                if ddim_eta > 0.0:
                    step_noise = seeded_noise_like(
                        x,
                        batch_idx,
                        base_seed=noise_seed,
                        salt=_STEP_SALT * (int(t) + 1),
                    )
                x = ddim_step(
                    model.scheduler,
                    x,
                    t_batch,
                    eps_true,
                    eta=ddim_eta,
                    noise=step_noise,
                )
            else:
                step_noise = seeded_noise_like(
                    x,
                    batch_idx,
                    base_seed=noise_seed,
                    salt=_STEP_SALT * (int(t) + 1),
                )
                x = model.scheduler.p_sample_step(
                    x, t_batch, eps_true, noise=step_noise
                )

        total_images_processed += take
        remaining -= take
        next_index += take
        if pbar is not None:
            pbar.update(take)

    if pbar is not None:
        pbar.close()

    curves = accum.compute_summary()
    alphas_cumprod = model.scheduler.alphas_cumprod.detach().cpu().tolist()

    # Compile milestone summary
    milestone_summary: list[dict[str, Any]] = []
    for m in sorted(milestone_set, reverse=True):
        m_row: dict[str, Any] = {
            "t": m,
            "alpha_bar": float(alphas_cumprod[m]),
            "sqrt_alpha_bar": float(np.sqrt(alphas_cumprod[m])),
            "sqrt_one_minus_alpha_bar": float(np.sqrt(max(1.0 - alphas_cumprod[m], 0.0))),
        }
        for k in metric_keys:
            if k in curves:
                m_row[f"{k}_mean"] = curves[k]["mean"][m]
                m_row[f"{k}_std"] = curves[k]["std"][m]
        milestone_summary.append(m_row)

    return {
        "mode": "reverse",
        "sampler": sampler,
        "ddim_eta": float(ddim_eta),
        "num_images": total_images_processed,
        "num_timesteps": num_t,
        "milestones": list(sorted(milestone_set, reverse=True)),
        "curves": curves,
        "milestone_summary": milestone_summary,
        "milestone_per_image": milestone_per_image,
        "alphas_cumprod": alphas_cumprod,
    }


@torch.no_grad()
def evaluate_condition_utility_forward(
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
    show_progress: bool = True,
    whitener: ChannelWhitening | None = None,
    milestones: Sequence[int] = DEFAULT_MILESTONES,
    eval_all_timesteps: bool = True,
) -> dict[str, Any]:
    """Measure condition utility on exact training-marginal forward states q(z_t | z_hr).

    For each timestep t:
        z_t = sqrt(ab_t) * z_hr + sqrt(1 - ab_t) * eps_gt
    Evaluates:
        eps_true = eps_theta(z_t, z_lr_true, t)
        eps_shuf = eps_theta(z_t, z_lr_shuffled, t)
    Here, the exact ground-truth noise eps_gt and clean latent z_hr are known.
    """
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)

    num_t = int(model.num_timesteps)
    milestone_set = {int(m) for m in milestones if 0 <= int(m) < num_t}
    target_timesteps = sorted(
        list(range(num_t)) if eval_all_timesteps else sorted(milestone_set),
        reverse=True,
    )

    metric_keys = (
        "eps_mse",
        "eps_rmse",
        "eps_rel_diff",
        "eps_cos",
        "z0_mse",
        "z0_rmse",
        "z0_cos",
        "cos_z0_true_lr_true",
        "cos_z0_shuf_lr_true",
        "cos_z0_shuf_lr_shuf",
        "specificity_gap",
        "z0_err_true",
        "z0_err_shuf",
        "z0_condition_advantage",
        "cos_z0_true_hr",
        "cos_z0_shuf_hr",
        "eps_err_true",
        "eps_err_shuf",
        "eps_condition_advantage",
    )
    accum = _UtilityAccumulator(num_t, metric_keys)
    milestone_per_image: dict[int, list[dict[str, float]]] = {
        m: [] for m in sorted(milestone_set, reverse=True)
    }

    remaining = max(int(num_images), 2)
    next_index = int(start_index)
    total_images_processed = 0

    pbar = (
        tqdm(
            total=remaining,
            desc="cond-utility (forward-q)",
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
        lr = lr[:take].to(device)
        hr_b = hr[:take].to(device)
        batch_idx = list(range(next_index, next_index + take))

        z_hr = encode_scaled(vae, hr_b, latent_scale=latent_scale)
        z_lr_raw = encode_lr_latents(
            vae,
            lr,
            hr_size=hr_size,
            latent_scale=latent_scale,
            apply_whiten=False,
        )
        z_lr_true = whitener.transform(z_lr_raw) if whitener is not None else z_lr_raw
        z_lr_shuf = shuffle_condition(z_lr_true, shift=1)

        for t in target_timesteps:
            t_batch = torch.full((take,), t, device=device, dtype=torch.long)
            # Seeded noise for reproducible q-sample
            eps_gt = seeded_noise_like(
                z_hr,
                batch_idx,
                base_seed=noise_seed,
                salt=_STEP_SALT * (int(t) + 1),
            )
            # q_sample: z_t = sqrt(ab_t) * z_hr + sqrt(1 - ab_t) * eps_gt
            z_t = model.scheduler.q_sample(z_hr, t_batch, noise=eps_gt)

            eps_true = model.predict_noise(z_t, t_batch, z_lr_true)
            eps_shuf = model.predict_noise(z_t, t_batch, z_lr_shuf)

            step_metrics = compute_step_condition_utility(
                model.scheduler,
                z_t,
                t_batch,
                eps_true,
                eps_shuf,
                z_lr_true,
                z_lr_shuf,
                z_hr=z_hr,
                eps_gt=eps_gt,
            )
            accum.add_batch(t, step_metrics)

            if t in milestone_set:
                for i, img_idx in enumerate(batch_idx):
                    row = {"val_index": img_idx, "t": t}
                    for k, v in step_metrics.items():
                        row[k] = float(v[i].item())
                    milestone_per_image[t].append(row)

        total_images_processed += take
        remaining -= take
        next_index += take
        if pbar is not None:
            pbar.update(take)

    if pbar is not None:
        pbar.close()

    curves = accum.compute_summary()
    alphas_cumprod = model.scheduler.alphas_cumprod.detach().cpu().tolist()

    milestone_summary: list[dict[str, Any]] = []
    for m in sorted(milestone_set, reverse=True):
        m_row: dict[str, Any] = {
            "t": m,
            "alpha_bar": float(alphas_cumprod[m]),
            "sqrt_alpha_bar": float(np.sqrt(alphas_cumprod[m])),
            "sqrt_one_minus_alpha_bar": float(np.sqrt(max(1.0 - alphas_cumprod[m], 0.0))),
        }
        for k in metric_keys:
            if k in curves:
                m_row[f"{k}_mean"] = curves[k]["mean"][m]
                m_row[f"{k}_std"] = curves[k]["std"][m]
        milestone_summary.append(m_row)

    return {
        "mode": "forward",
        "sampler": "none (q-sample)",
        "ddim_eta": 0.0,
        "num_images": total_images_processed,
        "num_timesteps": num_t,
        "milestones": list(sorted(milestone_set, reverse=True)),
        "curves": curves,
        "milestone_summary": milestone_summary,
        "milestone_per_image": milestone_per_image,
        "alphas_cumprod": alphas_cumprod,
    }


def format_milestone_table(milestone_summary: list[dict[str, Any]]) -> str:
    """Format milestone summary into a clean aligned text table."""
    headers = [
        "t",
        "ᾱ_t",
        "ε-MSE",
        "ε-RMSE",
        "ẑ0-MSE",
        "cos(ẑ0_t, ẑ0_s)",
        "Δcos(z_lr)",
        "Advantage(z_hr)",
    ]
    col_w = [6, 8, 11, 10, 11, 17, 14, 16]
    lines: list[str] = []
    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_w))
    sep_line = "-+-".join("-" * w for w in col_w)
    lines.append(header_line)
    lines.append(sep_line)

    for row in milestone_summary:
        t_val = str(row["t"])
        ab_val = f"{row.get('alpha_bar', 0.0):.4f}"
        eps_mse = f"{row.get('eps_mse_mean', 0.0):.5f}"
        eps_rmse = f"{row.get('eps_rmse_mean', 0.0):.4f}"
        z0_mse = f"{row.get('z0_mse_mean', 0.0):.5f}"
        z0_cos = f"{row.get('z0_cos_mean', 0.0):.4f}"
        gap = f"{row.get('specificity_gap_mean', 0.0):+.4f}"
        adv = f"{row.get('z0_condition_advantage_mean', 0.0):+.5f}"
        fields = [t_val, ab_val, eps_mse, eps_rmse, z0_mse, z0_cos, gap, adv]
        lines.append(" | ".join(f.rjust(w) for f, w in zip(fields, col_w)))

    return "\n".join(lines)


def generate_condition_utility_report(
    result: dict[str, Any],
    *,
    model_name: str = "Q2 Checkpoint",
) -> str:
    """Generate comprehensive diagnostic interpretation text."""
    ms = result["milestone_summary"]
    table_str = format_milestone_table(ms)
    mode = result.get("mode", "reverse")
    sampler = result.get("sampler", "ddpm")
    n_img = result.get("num_images", 0)

    # Find highest noise milestone (e.g. 999 or 800) and lowest noise milestone (e.g. 100 or 0)
    ms_by_t = {row["t"]: row for row in ms}
    high_t = 800 if 800 in ms_by_t else (999 if 999 in ms_by_t else max(ms_by_t.keys()))
    low_t = 0 if 0 in ms_by_t else (100 if 100 in ms_by_t else min(ms_by_t.keys()))

    eps_mse_high = ms_by_t[high_t].get("eps_mse_mean", 0.0)
    eps_mse_low = ms_by_t[low_t].get("eps_mse_mean", 0.0)
    z0_mse_high = ms_by_t[high_t].get("z0_mse_mean", 0.0)
    z0_mse_low = ms_by_t[low_t].get("z0_mse_mean", 0.0)
    cos_high = ms_by_t[high_t].get("z0_cos_mean", 0.0)
    cos_low = ms_by_t[low_t].get("z0_cos_mean", 0.0)

    eps_ratio = (eps_mse_high / max(eps_mse_low, 1e-9))
    z0_ratio = (z0_mse_high / max(z0_mse_low, 1e-9))

    is_ignored_at_t0 = (eps_mse_low < 0.1 * eps_mse_high) or (cos_low > 0.98)
    verdict = (
        "CONFIRMED: Condition utility collapses toward t=0. The model ignores z_lr "
        "at low noise timesteps because z_t already reveals the image structure."
        if is_ignored_at_t0
        else "Condition utility remains non-negligible at low noise timesteps."
    )

    report = f"""================================================================================
DIAGNOSTIC 1: CONDITION UTILITY AS A FUNCTION OF TIMESTEP
Model: {model_name}
Mode: {mode} | Sampler: {sampler} | Evaluated Images: {n_img}
================================================================================

Hypothesis:
  epsilon_theta(z_t, z_lr^true, t) vs epsilon_theta(z_t, z_lr^shuffled, t)
  Utility should be strong at noisy/mid timesteps and fall toward t=0,
  because at low noise z_t itself reveals z_hr and the training loss contains
  no explicit penalty for discarding z_lr at the end of the trajectory.

--- MILESTONE EVALUATION TABLE ---
{table_str}

--- DIAGNOSTIC QUANTITIES ---
High noise timestep (t={high_t}):
  eps-MSE(true vs shuf) : {eps_mse_high:.6f}
  z0-MSE(true vs shuf)  : {z0_mse_high:.6f}
  cos(z0_true, z0_shuf) : {cos_high:.4f}

Low noise timestep (t={low_t}):
  eps-MSE(true vs shuf) : {eps_mse_low:.6f}
  z0-MSE(true vs shuf)  : {z0_mse_low:.6f}
  cos(z0_true, z0_shuf) : {cos_low:.4f}

Utility Ratio (high / low):
  eps-MSE Ratio : {eps_ratio:.2f}x
  z0-MSE Ratio  : {z0_ratio:.2f}x

--- VERDICT ---
{verdict}
================================================================================
"""
    return report


def plot_condition_utility(
    result: dict[str, Any],
    output_path: Path | str,
    *,
    model_name: str = "Q2 Checkpoint",
) -> Path:
    """Save 4-panel diagnostic plot of condition utility across timesteps."""
    curves = result["curves"]
    alphas = np.array(result.get("alphas_cumprod", []))
    num_t = result.get("num_timesteps", len(curves["eps_mse"]["mean"]))
    timesteps = np.arange(num_t)
    mode = result.get("mode", "reverse")
    sampler = result.get("sampler", "ddpm")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Epsilon-MSE across t
    ax = axes[0, 0]
    eps_m = np.array(curves["eps_mse"]["mean"])
    eps_s = np.array(curves["eps_mse"]["std"])
    ax.plot(timesteps, eps_m, color="#1f77b4", lw=2, label="ε-MSE (true vs shuf)")
    ax.fill_between(timesteps, np.maximum(eps_m - eps_s, 0), eps_m + eps_s, color="#1f77b4", alpha=0.2)
    ax.set_xlabel("Diffusion Timestep t (0 = clean, 1000 = noise)")
    ax.set_ylabel("||ε(z_t, z_lr^true) − ε(z_t, z_lr^shuf)||^2")
    ax.set_title("1. Epsilon Sensitivity to LR Condition")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Panel 2: z0-MSE across t
    ax = axes[0, 1]
    z0_m = np.array(curves["z0_mse"]["mean"])
    z0_s = np.array(curves["z0_mse"]["std"])
    ax.plot(timesteps, z0_m, color="#d62728", lw=2, label="ẑ0-MSE (true vs shuf)")
    ax.fill_between(timesteps, np.maximum(z0_m - z0_s, 0), z0_m + z0_s, color="#d62728", alpha=0.2)
    ax.set_xlabel("Diffusion Timestep t")
    ax.set_ylabel("||ẑ0(true) − ẑ0(shuf)||^2")
    ax.set_title("2. Predicted Clean Latent Difference ||ẑ0^true − ẑ0^shuf||^2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Panel 3: Cosine Similarity between z0_true and z0_shuf
    ax = axes[1, 0]
    cos_m = np.array(curves["z0_cos"]["mean"])
    ax.plot(timesteps, cos_m, color="#2ca02c", lw=2, label="cos(ẑ0^true, ẑ0^shuf)")
    if "cos_z0_true_lr_true" in curves:
        ax.plot(
            timesteps,
            np.array(curves["cos_z0_true_lr_true"]["mean"]),
            color="#ff7f0e",
            lw=1.5,
            ls="--",
            label="cos(ẑ0^true, z_lr^true)",
        )
    ax.set_xlabel("Diffusion Timestep t")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("3. Directional Alignment of Predictions")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Panel 4: Condition Advantage or Specificity Gap
    ax = axes[1, 1]
    if "z0_condition_advantage" in curves and any(curves["z0_condition_advantage"]["mean"]):
        adv_m = np.array(curves["z0_condition_advantage"]["mean"])
        ax.plot(timesteps, adv_m, color="#9467bd", lw=2, label="ẑ0 Advantage (Error Reduction to z_hr)")
        ax.axhline(0.0, color="gray", ls=":", lw=1)
    elif "specificity_gap" in curves:
        gap_m = np.array(curves["specificity_gap"]["mean"])
        ax.plot(timesteps, gap_m, color="#8c564b", lw=2, label="Specificity Gap: cos(z0, lr_true) - cos(z0, lr_shuf)")
        ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Diffusion Timestep t")
    ax.set_ylabel("Metric Value")
    ax.set_title("4. Condition Utility / Specificity Gap")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.suptitle(
        f"Diagnostic 1: Condition Utility as a Function of Timestep\n{model_name} ({mode} mode, {sampler})",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_condition_utility_results(
    result: dict[str, Any],
    output_dir: Path | str,
    *,
    model_name: str = "Q2 Checkpoint",
) -> dict[str, Path]:
    """Save all artifacts for Diagnostic 1: JSON, CSVs, report, and plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # 1. Summary JSON (without big curves to keep lightweight)
    json_path = output_dir / "condition_utility_summary.json"
    summary_data = {
        "model_name": model_name,
        "mode": result.get("mode"),
        "sampler": result.get("sampler"),
        "ddim_eta": result.get("ddim_eta"),
        "num_images": result.get("num_images"),
        "num_timesteps": result.get("num_timesteps"),
        "milestones": result.get("milestones"),
        "milestone_summary": result.get("milestone_summary"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    saved["json"] = json_path

    # 2. Milestones CSV
    ms_csv_path = output_dir / "condition_utility_milestones.csv"
    if result.get("milestone_summary"):
        ms = result["milestone_summary"]
        fieldnames = list(ms[0].keys())
        with open(ms_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ms)
        saved["milestones_csv"] = ms_csv_path

    # 3. All-t curves CSV
    curves_csv_path = output_dir / "condition_utility_curves.csv"
    curves = result.get("curves", {})
    if "eps_mse" in curves:
        num_t = len(curves["eps_mse"]["mean"])
        alphas = result.get("alphas_cumprod", [0.0] * num_t)
        rows: list[dict[str, Any]] = []
        for t in range(num_t):
            r: dict[str, Any] = {"t": t, "alpha_bar": alphas[t] if t < len(alphas) else 0.0}
            for k, stat in curves.items():
                r[f"{k}_mean"] = stat["mean"][t]
                r[f"{k}_std"] = stat["std"][t]
            rows.append(r)
        with open(curves_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        saved["curves_csv"] = curves_csv_path

    # 4. Text Report
    report_path = output_dir / "condition_utility_report.txt"
    report_text = generate_condition_utility_report(result, model_name=model_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    saved["report"] = report_path

    # 5. Diagnostic Plot
    plot_path = output_dir / "condition_utility_plot.png"
    plot_condition_utility(result, plot_path, model_name=model_name)
    saved["plot"] = plot_path

    return saved
