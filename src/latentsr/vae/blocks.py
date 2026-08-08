"""GroupNorm + SiLU residual block used by the convolutional VAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with as many groups as possible up to ``max_groups``."""
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResidualBlock(nn.Module):
    """Two 3×3 convolutions with GroupNorm and a residual skip."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = group_norm(channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class Downsample(nn.Module):
    """Stride-2 convolution that halves spatial size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor 2× upsample followed by a 3×3 convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
