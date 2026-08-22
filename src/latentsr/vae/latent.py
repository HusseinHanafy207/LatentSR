"""Frozen-VAE encode/decode helpers with Stable-Diffusion-style latent scaling.

Contract for all later phases (LDM / SR):

    z_scaled = latent_scale * mu          # encode_scaled
    x_hat    = decode(z_scaled / latent_scale)  # decode_scaled

Default ``latent_scale=1.0``. After the VAE is frozen you may retune scale to
approximately ``1 / std(mu)`` over a calibration batch if diffusion is unstable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from latentsr.vae.checkpointing import load_vae_checkpoint
from latentsr.vae.vae import VAE


def freeze_vae(model: nn.Module) -> nn.Module:
    """Put ``model`` in eval mode and disable gradients on all parameters."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_frozen_vae(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[VAE, dict[str, Any]]:
    """Load a VAE checkpoint, freeze it, and return ``(model, checkpoint)``.

    The returned model is in ``eval`` mode with ``requires_grad=False`` on every
    parameter — suitable for on-the-fly encoding in diffusion / SR training.
    """
    model, checkpoint = load_vae_checkpoint(path, map_location=map_location)
    freeze_vae(model)
    return model, checkpoint


def is_frozen(model: nn.Module) -> bool:
    """Return True if every parameter has ``requires_grad=False``."""
    params = list(model.parameters())
    if not params:
        return True
    return all(not p.requires_grad for p in params)


@torch.no_grad()
def encode_scaled(
    vae: VAE,
    x: torch.Tensor,
    latent_scale: float = 1.0,
) -> torch.Tensor:
    """Encode images to scaled latents: ``latent_scale * mu``.

    Uses the posterior mean (deterministic) — preferred for diffusion / SR.
    """
    if latent_scale <= 0:
        raise ValueError(f"latent_scale must be > 0, got {latent_scale}")
    mu, _logvar = vae.encode(x)
    return mu * latent_scale


def decode_scaled(
    vae: VAE,
    z_scaled: torch.Tensor,
    latent_scale: float = 1.0,
    *,
    allow_grad: bool = False,
) -> torch.Tensor:
    """Decode scaled latents: ``decode(z_scaled / latent_scale)``.

    Decoder weights stay frozen. Pass ``allow_grad=True`` to backprop
    *through* the decoder (guidance), not into it.
    """
    if latent_scale <= 0:
        raise ValueError(f"latent_scale must be > 0, got {latent_scale}")
    if allow_grad:
        return vae.decode(z_scaled / latent_scale)
    with torch.no_grad():
        return vae.decode(z_scaled / latent_scale)


@torch.no_grad()
def estimate_latent_scale(
    vae: VAE,
    images: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> float:
    """Suggest ``latent_scale ≈ 1 / std(mu)`` from a calibration batch.

    Returns a Python float. Call after freezing the VAE; store the value in
    diffusion configs when ``1.0`` proves too unstable.
    """
    mu, _ = vae.encode(images)
    std = float(mu.std().clamp_min(eps).item())
    return 1.0 / std
