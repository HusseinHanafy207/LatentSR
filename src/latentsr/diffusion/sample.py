"""Latent Diffusion sampling: noise → z_scaled → decode → RGB.

Important: do **not** use ``generative_models.ddpm.sample`` for latents — that
helper clamps outputs to ``[0, 1]``, which destroys the latent distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from generative_models.ddpm import DDPM
from generative_models.ddpm.sampler import p_sample
from torchvision.utils import save_image
from tqdm.auto import tqdm

from latentsr.vae.latent import decode_scaled, load_frozen_vae
from latentsr.vae.vae import VAE


@torch.no_grad()
def sample_latents(
    model: DDPM,
    num_samples: int,
    *,
    latent_channels: int = 4,
    latent_size: int = 32,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Run the reverse diffusion chain in latent space (no value clamping)."""
    device = device or next(model.unet.parameters()).device
    model.eval()

    z = torch.randn(
        num_samples, latent_channels, latent_size, latent_size, device=device
    )
    timesteps = range(model.num_timesteps - 1, -1, -1)
    iterator = (
        tqdm(timesteps, desc="sampling", leave=False) if show_progress else timesteps
    )
    for t in iterator:
        z = p_sample(model, z, t)
    return z


@torch.no_grad()
def sample_ldm(
    ddpm: DDPM,
    vae: VAE,
    num_samples: int,
    *,
    latent_channels: int = 4,
    latent_size: int = 32,
    latent_scale: float = 1.0,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Full LDM pipeline: noise → latent DDPM → decode_scaled → RGB ``[0, 1]``."""
    device = device or next(ddpm.unet.parameters()).device
    vae.to(device)
    z_scaled = sample_latents(
        ddpm,
        num_samples,
        latent_channels=latent_channels,
        latent_size=latent_size,
        device=device,
        show_progress=show_progress,
    )
    images = decode_scaled(vae, z_scaled, latent_scale=latent_scale)
    return images.clamp(0.0, 1.0)


def save_ldm_grid(
    images: torch.Tensor,
    output_path: str | Path,
    *,
    nrow: int = 8,
) -> Path:
    """Save an RGB sample grid."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.cpu().clamp(0.0, 1.0), output_path, nrow=nrow, padding=2)
    return output_path


def load_ldm_components(
    ddpm_checkpoint: str | Path,
    *,
    vae_checkpoint: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[DDPM, VAE, dict[str, Any]]:
    """Load latent DDPM + frozen VAE; VAE path falls back to DDPM config metadata."""
    from latentsr.diffusion.build import load_latent_ddpm_checkpoint

    ddpm, checkpoint = load_latent_ddpm_checkpoint(
        ddpm_checkpoint, map_location=map_location
    )
    config = checkpoint.get("config") or {}
    vae_path = (
        vae_checkpoint
        or checkpoint.get("vae_checkpoint")
        or config.get("vae_checkpoint")
    )
    if not vae_path:
        raise ValueError(
            "VAE checkpoint not found in DDPM metadata; pass --vae-checkpoint."
        )
    latent_scale = float(
        checkpoint.get("latent_scale", config.get("latent_scale", 1.0))
    )
    vae, _vae_ckpt = load_frozen_vae(vae_path, map_location=map_location)
    meta = {
        "config": config,
        "latent_scale": latent_scale,
        "vae_checkpoint": str(vae_path),
        "ddpm_epoch": checkpoint.get("epoch"),
    }
    return ddpm, vae, meta
