"""On-the-fly SR latent pairs (Phase 7, no disk cache).

For each pixel pair ``(lr, hr)``:

    lr_up = bicubic_upsample(lr → hr_size)
    z_lr  = latent_scale * encode(lr_up).mu
    z_hr  = latent_scale * encode(hr).mu

Both latents share spatial size ``(C_z, hr_size/4, hr_size/4)`` so Phase 8 can
concatenate ``[x_t | z_lr]`` channel-wise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from latentsr.datasets.onthefly_latent import OnTheFlyLatentEncoder
from latentsr.vae.latent import is_frozen, load_frozen_vae
from latentsr.vae.vae import VAE
from latentsr.vae.whitening import ChannelWhitening, load_channel_whitening


def upsample_bicubic(lr: torch.Tensor, hr_size: int) -> torch.Tensor:
    """Bicubic upsample ``(B, C, h, w)`` or ``(C, h, w)`` to ``hr_size``."""
    if lr.ndim == 3:
        lr_b = lr.unsqueeze(0)
        out = F.interpolate(
            lr_b,
            size=(hr_size, hr_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    elif lr.ndim == 4:
        out = F.interpolate(
            lr,
            size=(hr_size, hr_size),
            mode="bicubic",
            align_corners=False,
        )
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(lr.shape)}")
    return out.clamp(0.0, 1.0)


class OnTheFlySRLatentEncoder:
    """Frozen VAE wrapper that maps ``(lr, hr)`` batches to ``(z_lr, z_hr)``.

    Optional ``whitener`` is applied to **z_lr only** (condition). HR targets
    stay raw. Soft-decode must use ``encode_lr_raw`` / unwhitened latents.
    """

    def __init__(
        self,
        vae: VAE,
        *,
        latent_scale: float = 1.0,
        hr_size: int = 128,
        whitener: ChannelWhitening | None = None,
    ) -> None:
        if not is_frozen(vae):
            raise ValueError(
                "OnTheFlySRLatentEncoder expects a frozen VAE "
                "(use load_frozen_vae / freeze_vae)."
            )
        if latent_scale <= 0:
            raise ValueError(f"latent_scale must be > 0, got {latent_scale}")
        self.encoder = OnTheFlyLatentEncoder(vae, latent_scale=latent_scale)
        self.hr_size = int(hr_size)
        self.whitener = whitener

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        latent_scale: float = 1.0,
        hr_size: int = 128,
        map_location: str | torch.device = "cpu",
        whitener: ChannelWhitening | None = None,
        whiten_path: str | Path | None = None,
    ) -> OnTheFlySRLatentEncoder:
        vae, _ = load_frozen_vae(checkpoint, map_location=map_location)
        if whitener is None and whiten_path is not None:
            whitener = load_channel_whitening(whiten_path)
        return cls(
            vae, latent_scale=latent_scale, hr_size=hr_size, whitener=whitener
        )

    def to(self, device: torch.device | str) -> OnTheFlySRLatentEncoder:
        self.encoder.to(device)
        return self

    def set_latent_scale(self, latent_scale: float) -> None:
        self.encoder.set_latent_scale(latent_scale)

    @property
    def vae(self) -> VAE:
        return self.encoder.vae

    @property
    def latent_scale(self) -> float:
        return self.encoder.latent_scale

    @property
    def latent_channels(self) -> int:
        return self.encoder.latent_channels

    def encode_lr_raw(self, lr: torch.Tensor) -> torch.Tensor:
        """Bicubic↑ + encode; **no** whitening (for soft-decode / guidance)."""
        lr_up = upsample_bicubic(lr, self.hr_size)
        return self.encoder(lr_up)

    def condition_from_raw(self, z_lr_raw: torch.Tensor) -> torch.Tensor:
        """Apply frozen whitener to a raw LR latent (identity if unset)."""
        if self.whitener is None:
            return z_lr_raw
        return self.whitener.transform(z_lr_raw)

    def __call__(
        self, lr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode LR/HR → ``(z_lr_cond, z_hr_raw)`` of matching spatial size."""
        if hr.shape[-2:] != (self.hr_size, self.hr_size):
            raise ValueError(
                f"Expected HR spatial {(self.hr_size, self.hr_size)}, "
                f"got {tuple(hr.shape[-2:])}"
            )
        z_lr_raw = self.encode_lr_raw(lr)
        z_hr = self.encoder(hr)
        z_lr = self.condition_from_raw(z_lr_raw)
        if z_lr.shape != z_hr.shape:
            raise RuntimeError(
                f"Latent shape mismatch: z_lr {tuple(z_lr.shape)} vs "
                f"z_hr {tuple(z_hr.shape)}"
            )
        return z_lr, z_hr

    def encode_batch(
        self, batch: Any, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a dataloader ``(lr, hr)`` batch → whitened cond + raw HR."""
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise TypeError("Expected batch to be (lr, hr, …)")
        lr, hr = batch[0], batch[1]
        if device is not None:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
        return self(lr, hr)
