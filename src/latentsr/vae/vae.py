"""Convolutional VAE composing spatial encoder and decoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from latentsr.vae.decoder import Decoder
from latentsr.vae.encoder import Encoder


class VAE(nn.Module):
    """β-VAE with spatial latents for LDM-style pipelines.

    Default: RGB ``128×128`` → latent ``(4, 32, 32)`` (downsample factor 4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_mult=self.channel_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_mult=self.channel_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` of ``q(z|x)``."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to RGB in ``[0, 1]``."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = Encoder.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def reconstruct(self, x: torch.Tensor, *, use_mean: bool = True) -> torch.Tensor:
        """Encode then decode; use ``mu`` when ``use_mean`` else a posterior sample."""
        mu, logvar = self.encode(x)
        z = mu if use_mean else Encoder.reparameterize(mu, logvar)
        return self.decode(z)

    def sample(
        self,
        num_samples: int,
        latent_size: int = 32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample ``z ~ N(0, I)`` and decode (prior samples; often blurry for β-VAE)."""
        device = device or next(self.parameters()).device
        z = torch.randn(
            num_samples,
            self.latent_channels,
            latent_size,
            latent_size,
            device=device,
        )
        return self.decode(z)

    def config_dict(self) -> dict[str, Any]:
        """Architecture kwargs for checkpoint metadata / rebuild."""
        return {
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "base_channels": self.base_channels,
            "channel_mult": list(self.channel_mult),
            "num_res_blocks": self.num_res_blocks,
            "dropout": self.dropout,
        }
