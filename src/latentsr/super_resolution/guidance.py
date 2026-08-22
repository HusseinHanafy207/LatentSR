"""Inference-time conditioning-consistency guidance (Stage 1, no training).

Convention (fixed for all Stage-1 conditions)
--------------------------------------------
DPS-style, Chung et al. (2022)-like correction:

    1. ε̂ = UNet(z_t, z_lr, t)          # no UNet backward (ε̂ detached)
    2. ẑ0 = (z_t − √(1−ᾱ_t) ε̂) / √ᾱ_t
    3. z_{t-1} = p_sample_step(z_t, ε̂, t)     # unguided DDPM posterior
    4. if t is in the guidance window:
           L_g = mean_{c,h,w} || D(ẑ0) − D(z_lr) ||²
           g   = ∇_{z_t} L_g                  # backprop through D, not UNet
           z_{t-1} ← z_{t-1} − λ_g · g

``D`` is the frozen VAE decoder (weights ``requires_grad=False``). Gradient
flows through the ẑ0 affine and the decoder, then is applied as a correction
to the **already-sampled** ``z_{t-1}``.

Guidance window: active iff ``t_low <= t <= t_high`` (scheduler index;
t=0 is clean, t=T−1 is noise). Early = (0, 800), late = (0, 500).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.super_resolution.sample import (
    STEP_SALT,
    predict_x0_from_eps,
    seeded_noise_like,
)
from latentsr.vae.latent import decode_scaled
from latentsr.vae.vae import VAE


@dataclass(frozen=True)
class GuidanceWindow:
    """Inclusive scheduler-t range. ``t_high=None`` disables guidance."""

    t_low: int = 0
    t_high: int | None = None
    every: int = 1

    def active(self, t: int) -> bool:
        if self.t_high is None:
            return False
        if not (int(self.t_low) <= int(t) <= int(self.t_high)):
            return False
        return ((int(self.t_high) - int(t)) % max(int(self.every), 1)) == 0


BASELINE_WINDOW = GuidanceWindow(t_high=None)
EARLY_WINDOW = GuidanceWindow(t_low=0, t_high=800)
LATE_WINDOW = GuidanceWindow(t_low=0, t_high=500)

TRAJECTORY_TIMESTEPS = (
    800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50, 0,
)


def per_image_l2(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(dim=1)


def cache_soft_decodes(
    vae: VAE,
    lr: torch.Tensor,
    *,
    hr_size: int = 128,
    latent_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(z_lr, D(z_lr))``. ``D(z_lr)`` is the fixed guidance target."""
    z_lr = encode_lr_latents(vae, lr, hr_size=hr_size, latent_scale=latent_scale)
    d_zlr = decode_scaled(vae, z_lr, latent_scale, allow_grad=False).clamp(0.0, 1.0)
    return z_lr, d_zlr


def guidance_loss(decoded_z0: torch.Tensor, d_zlr: torch.Tensor) -> torch.Tensor:
    """Per-image mean squared error ``||D(ẑ0) − D(z_lr)||²`` (mean over C,H,W)."""
    if decoded_z0.shape != d_zlr.shape:
        raise ValueError(
            f"shape mismatch: ẑ0 image {tuple(decoded_z0.shape)} vs "
            f"D(z_lr) {tuple(d_zlr.shape)}"
        )
    return (decoded_z0 - d_zlr).pow(2).mean(dim=(1, 2, 3))


def dps_guidance_step(
    model: ConditionalLatentDDPM,
    vae: VAE,
    z_t: torch.Tensor,
    t: torch.Tensor,
    z_lr: torch.Tensor,
    d_zlr: torch.Tensor,
    *,
    latent_scale: float,
    lambda_g: float,
    active: bool,
    step_noise: torch.Tensor | None,
    need_diagnostics: bool,
    apply_correction: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One reverse step. See module docstring for the z_t vs z_{t-1} convention."""
    z_t = z_t.detach()
    with torch.no_grad():
        eps = model.predict_noise(z_t, t, z_lr)
        z_prev = model.scheduler.p_sample_step(z_t, t, eps, noise=step_noise)
        delta = z_prev - z_t
        delta_norm = per_image_l2(delta)

    extras: dict[str, torch.Tensor] = {
        "delta_norm": delta_norm,
        "guidance_active": torch.full(
            (z_t.shape[0],), float(active), device=z_t.device
        ),
    }

    if not (active or need_diagnostics):
        extras["z0"] = predict_x0_from_eps(model.scheduler, z_t, t, eps)
        extras["loss"] = torch.full((z_t.shape[0],), float("nan"), device=z_t.device)
        extras["grad_norm"] = torch.zeros(z_t.shape[0], device=z_t.device)
        extras["r"] = torch.full((z_t.shape[0],), float("nan"), device=z_t.device)
        return z_prev.detach(), extras

    z_req = z_t.detach().requires_grad_(bool(active))
    z0 = predict_x0_from_eps(model.scheduler, z_req, t, eps.detach())
    decoded = decode_scaled(vae, z0, latent_scale, allow_grad=bool(active))
    loss = guidance_loss(decoded, d_zlr)
    extras["z0"] = z0.detach()
    extras["loss"] = loss.detach()
    extras["decoded_z0"] = decoded.detach().clamp(0.0, 1.0)

    if not active:
        extras["grad_norm"] = torch.zeros(z_t.shape[0], device=z_t.device)
        extras["r"] = torch.full((z_t.shape[0],), float("nan"), device=z_t.device)
        return z_prev.detach(), extras

    grad = torch.autograd.grad(loss.sum(), z_req, retain_graph=False)[0]
    grad_norm = per_image_l2(grad)
    extras["grad_norm"] = grad_norm.detach()
    denom = delta_norm.clamp_min(1e-8)
    extras["r"] = (float(lambda_g) * grad_norm / denom).detach()
    z_out = z_prev
    if apply_correction:
        z_out = z_prev - float(lambda_g) * grad
    return z_out.detach(), extras


def sample_guided_latents(
    model: ConditionalLatentDDPM,
    vae: VAE,
    z_lr: torch.Tensor,
    d_zlr: torch.Tensor,
    *,
    latent_scale: float = 1.0,
    lambda_g: float = 0.0,
    window: GuidanceWindow = BASELINE_WINDOW,
    val_indices: Sequence[int] | None = None,
    noise_seed: int | None = None,
    noise: torch.Tensor | None = None,
    log_timesteps: Sequence[int] | None = None,
    on_log: Callable[[int, dict[str, torch.Tensor]], None] | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Full reverse chain with optional DPS guidance. Same seed contract as eval."""
    device = z_lr.device
    model.eval()
    vae.eval()
    if d_zlr.shape[0] != z_lr.shape[0]:
        raise ValueError(
            f"D(z_lr) batch {d_zlr.shape[0]} != z_lr batch {z_lr.shape[0]}"
        )
    if noise_seed is not None:
        if val_indices is None:
            raise ValueError("val_indices is required when noise_seed is set")
        if len(val_indices) != z_lr.shape[0]:
            raise ValueError(
                f"val_indices length {len(val_indices)} != batch {z_lr.shape[0]}"
            )
        if noise is None:
            noise = seeded_noise_like(
                z_lr, val_indices, base_seed=noise_seed, salt=0
            )
    if noise is None:
        z = torch.randn_like(z_lr)
    else:
        if noise.shape != z_lr.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} != z_lr shape {tuple(z_lr.shape)}"
            )
        z = noise.to(device=device, dtype=z_lr.dtype)

    log_set = {int(t) for t in (log_timesteps or ())}
    timesteps = range(model.num_timesteps - 1, -1, -1)
    iterator: Any = (
        tqdm(timesteps, desc="guided-sampling", leave=False)
        if show_progress
        else timesteps
    )
    for t_int in iterator:
        t_batch = torch.full((z.shape[0],), t_int, device=device, dtype=torch.long)
        step_noise = None
        if noise_seed is not None:
            assert val_indices is not None
            step_noise = seeded_noise_like(
                z,
                val_indices,
                base_seed=noise_seed,
                salt=STEP_SALT * (int(t_int) + 1),
            )
        need_log = t_int in log_set and on_log is not None
        active = window.active(t_int) and float(lambda_g) != 0.0
        z, extras = dps_guidance_step(
            model,
            vae,
            z,
            t_batch,
            z_lr,
            d_zlr,
            latent_scale=latent_scale,
            lambda_g=lambda_g,
            active=active,
            step_noise=step_noise,
            need_diagnostics=need_log,
            apply_correction=True,
        )
        if need_log and on_log is not None:
            extras["z"] = z.detach()
            on_log(int(t_int), extras)
    return z


def iter_indexed_batches(
    loader: DataLoader,
    *,
    start_index: int,
    num_images: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[int]]]:
    """Yield ``(lr, hr, val_indices)`` after skipping ``start_index`` samples."""
    skipped = 0
    remaining = max(int(num_images), 1)
    next_index = int(start_index)
    target_skip = max(int(start_index), 0)
    for lr, hr in loader:
        if remaining <= 0:
            break
        if skipped < target_skip:
            drop = min(lr.shape[0], target_skip - skipped)
            skipped += drop
            if drop == lr.shape[0]:
                continue
            lr = lr[drop:]
            hr = hr[drop:]
        take = min(lr.shape[0], remaining)
        lr = lr[:take]
        hr = hr[:take]
        indices = list(range(next_index, next_index + take))
        yield lr, hr, indices
        remaining -= take
        next_index += take

