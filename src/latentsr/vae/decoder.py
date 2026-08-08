"""Convolutional VAE decoder ← spatial latents."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from latentsr.vae.blocks import ResidualBlock, Upsample, group_norm


class Decoder(nn.Module):
    """Maps spatial latents ``(C_z, H/f, W/f)`` back to RGB in ``[0, 1]``."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channel_mult) < 2:
            raise ValueError("channel_mult must have at least two stages for f=4")

        # Mirror encoder: start from bottleneck channels, upsample toward image.
        mults = list(reversed(channel_mult))
        ch = base_channels * channel_mult[-1]
        self.conv_in = nn.Conv2d(latent_channels, ch, kernel_size=3, padding=1)

        blocks: list[nn.Module] = []
        for i, mult in enumerate(mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                if ch != out_ch:
                    blocks.append(nn.Conv2d(ch, out_ch, kernel_size=1))
                    ch = out_ch
                blocks.append(ResidualBlock(ch, dropout=dropout))
            if i < len(mults) - 1:
                next_ch = base_channels * mults[i + 1]
                blocks.append(Upsample(ch, next_ch))
                ch = next_ch

        self.backbone = nn.ModuleList(blocks)
        self.norm_out = group_norm(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        for module in self.backbone:
            h = module(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.sigmoid(h)
