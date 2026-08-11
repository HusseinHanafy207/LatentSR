"""On-the-fly image → scaled-latent encoding (Phase 3, no disk cache).

Diffusion trainers should keep loading **pixels** from CelebA and call
:class:`OnTheFlyLatentEncoder` each step:

    images = batch_to_images(batch).to(device)
    z = latent_encoder(images)   # (B, 4, 32, 32), no_grad

Changing ``latent_scale`` or the VAE checkpoint does **not** require rebuilding
any cached ``.pt`` files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from latentsr.vae.latent import encode_scaled, is_frozen, load_frozen_vae
from latentsr.vae.vae import VAE


def batch_to_images(batch: Any) -> torch.Tensor:
    """Extract the image tensor from a ``(image, …)`` batch or bare tensor."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


class OnTheFlyLatentEncoder:
    """Frozen VAE wrapper that maps RGB batches to scaled latents.

    Parameters
    ----------
    vae:
        Already-frozen :class:`~latentsr.vae.vae.VAE` (eval, no grad).
    latent_scale:
        Multiplier applied to ``mu`` (default ``1.0``).
    """

    def __init__(self, vae: VAE, latent_scale: float = 1.0) -> None:
        if not is_frozen(vae):
            raise ValueError(
                "OnTheFlyLatentEncoder expects a frozen VAE "
                "(use load_frozen_vae / freeze_vae)."
            )
        if latent_scale <= 0:
            raise ValueError(f"latent_scale must be > 0, got {latent_scale}")
        self.vae = vae
        self.latent_scale = float(latent_scale)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        latent_scale: float = 1.0,
        map_location: str | torch.device = "cpu",
    ) -> OnTheFlyLatentEncoder:
        """Build an encoder from a VAE checkpoint path."""
        vae, _ckpt = load_frozen_vae(checkpoint, map_location=map_location)
        return cls(vae, latent_scale=latent_scale)

    def to(self, device: torch.device | str) -> OnTheFlyLatentEncoder:
        self.vae.to(device)
        return self

    def set_latent_scale(self, latent_scale: float) -> None:
        """Update scale without touching VAE weights or any disk cache."""
        if latent_scale <= 0:
            raise ValueError(f"latent_scale must be > 0, got {latent_scale}")
        self.latent_scale = float(latent_scale)

    @property
    def latent_channels(self) -> int:
        return int(self.vae.latent_channels)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 3, H, W)`` → ``(B, C_z, H/4, W/4)`` scaled latents."""
        return encode_scaled(self.vae, images, self.latent_scale)

    def encode_batch(self, batch: Any, device: torch.device | None = None) -> torch.Tensor:
        """Pull images from a dataloader batch and encode on ``device``."""
        images = batch_to_images(batch)
        if device is not None:
            images = images.to(device, non_blocking=True)
        return self(images)
