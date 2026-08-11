"""Conditional reverse sampling in latent space (Phase 8/9)."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.vae.latent import decode_scaled
from latentsr.vae.vae import VAE


@torch.no_grad()
def sample_conditional_latents(
    model: ConditionalLatentDDPM,
    z_lr: torch.Tensor,
    *,
    show_progress: bool = True,
) -> torch.Tensor:
    """Denoise from noise to ``z_hr``, conditioned on ``z_lr`` (no clamping)."""
    device = z_lr.device
    model.eval()
    z = torch.randn_like(z_lr)
    timesteps = range(model.num_timesteps - 1, -1, -1)
    iterator = (
        tqdm(timesteps, desc="sr-sampling", leave=False) if show_progress else timesteps
    )
    for t in iterator:
        t_batch = torch.full((z.shape[0],), t, device=device, dtype=torch.long)
        noise_pred = model.predict_noise(z, t_batch, z_lr)
        z = model.scheduler.p_sample_step(z, t_batch, noise_pred)
    return z


@torch.no_grad()
def sample_sr_images(
    model: ConditionalLatentDDPM,
    vae: VAE,
    z_lr: torch.Tensor,
    *,
    latent_scale: float = 1.0,
    show_progress: bool = True,
) -> torch.Tensor:
    """``z_lr`` → conditional DDPM → decode_scaled → RGB ``[0, 1]``."""
    z_hr = sample_conditional_latents(model, z_lr, show_progress=show_progress)
    images = decode_scaled(vae, z_hr, latent_scale=latent_scale)
    return images.clamp(0.0, 1.0)
