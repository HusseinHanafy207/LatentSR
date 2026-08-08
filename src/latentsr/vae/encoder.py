"""Convolutional VAE encoder → spatial Gaussian posterior."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from latentsr.vae.blocks import Downsample, ResidualBlock, group_norm


class Encoder(nn.Module):
    """Maps RGB images to ``(mu, logvar)`` with spatial shape ``(C_z, H/f, W/f)``.

    Default ``f=4``: ``128×128`` → ``32×32`` latent map.
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
        if len(channel_mult) < 2:
            raise ValueError("channel_mult must have at least two stages for f=4")

        self.latent_channels = latent_channels
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        blocks: list[nn.Module] = []
        ch = base_channels
        # Last multiplier is the bottleneck width at latent resolution.
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                if ch != out_ch:
                    blocks.append(nn.Conv2d(ch, out_ch, kernel_size=1))
                    ch = out_ch
                blocks.append(ResidualBlock(ch, dropout=dropout))
            if i < len(channel_mult) - 1:
                next_ch = base_channels * channel_mult[i + 1]
                blocks.append(Downsample(ch, next_ch))
                ch = next_ch

        self.backbone = nn.ModuleList(blocks)
        self.norm_out = group_norm(ch)
        self.conv_out = nn.Conv2d(ch, 2 * latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        for module in self.backbone:
            h = module(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
