"""AdaGN / spatial FiLM conditioning for latent SR (replaces concat).

``x_t`` stays ``C_z`` channels. ``z_lr`` is projected to a spatial pyramid and
injected at every UNet scale:

    h' = (1 + γ(z_lr)) ⊙ h + β(z_lr)

γ, β are 1×1 convs, zero-initialized so the block is identity at the start of
training. Residual blocks, attention, and the time UNet skeleton are imported
from ``generative_models.ddpm`` (not vendored).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from generative_models.ddpm.time_embedding import TimestepEmbedding
from generative_models.ddpm.unet import (
    AttentionBlock,
    Downsample,
    ResidualBlock,
    Upsample,
    _group_norm,
)


class SpatialFiLM(nn.Module):
    """Per-pixel affine modulation from a condition map of matching resolution."""

    def __init__(self, cond_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(cond_channels, out_channels * 2, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.shape[-2:] != h.shape[-2:]:
            cond = F.interpolate(
                cond, size=h.shape[-2:], mode="bilinear", align_corners=False
            )
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return (1.0 + gamma) * h + beta


class AdaGNLatentUNet(nn.Module):
    """Time-conditioned UNet with multi-scale FiLM from ``z_lr`` (no channel concat)."""

    def __init__(
        self,
        latent_channels: int = 4,
        *,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        image_size: int = 32,
        time_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        if latent_channels < 1:
            raise ValueError(f"latent_channels must be >= 1, got {latent_channels}")
        if not channel_mult:
            raise ValueError("channel_mult must be non-empty")

        self.latent_channels = latent_channels
        self.in_channels = latent_channels
        self.out_channels = latent_channels
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)
        self.image_size = image_size
        self.condition_type = "adagn"

        time_dim = time_embedding_dim or base_channels * 4
        self.time_dim = time_dim
        self.time_embed = TimestepEmbedding(embedding_dim=time_dim)

        cond_ch = base_channels
        self.cond_in = nn.Conv2d(latent_channels, cond_ch, kernel_size=3, padding=1)
        self.cond_downs = nn.ModuleList(
            nn.Conv2d(cond_ch, cond_ch, kernel_size=3, stride=2, padding=1)
            for _ in range(len(self.channel_mult) - 1)
        )

        self.conv_in = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)

        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enc_films = nn.ModuleList()
        skip_channels: list[int] = [base_channels]
        ch = base_channels
        resolution = image_size

        for level, mult in enumerate(self.channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                block = ResidualBlock(ch, out_ch, time_dim=time_dim, dropout=dropout)
                attn: nn.Module
                if resolution in self.attention_resolutions:
                    attn = AttentionBlock(out_ch)
                else:
                    attn = nn.Identity()
                blocks.append(nn.ModuleList([block, attn]))
                ch = out_ch
                skip_channels.append(ch)
            self.encoder.append(blocks)
            self.enc_films.append(SpatialFiLM(cond_ch, ch))

            if level != len(self.channel_mult) - 1:
                self.downs.append(Downsample(ch))
                skip_channels.append(ch)
                resolution //= 2
            else:
                self.downs.append(nn.Identity())

        self.mid_block1 = ResidualBlock(ch, ch, time_dim=time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_dim=time_dim, dropout=dropout)
        self.mid_film = SpatialFiLM(cond_ch, ch)

        self.decoder = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.dec_films = nn.ModuleList()
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block = ResidualBlock(
                    ch + skip_ch, out_ch, time_dim=time_dim, dropout=dropout
                )
                attn = (
                    AttentionBlock(out_ch)
                    if resolution in self.attention_resolutions
                    else nn.Identity()
                )
                blocks.append(nn.ModuleList([block, attn]))
                ch = out_ch
            self.decoder.append(blocks)
            self.dec_films.append(SpatialFiLM(cond_ch, ch))

            if level != 0:
                self.ups.append(Upsample(ch))
                resolution *= 2
            else:
                self.ups.append(nn.Identity())

        self.conv_out = nn.Sequential(
            _group_norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, latent_channels, kernel_size=3, padding=1),
        )

    def _condition_pyramid(self, z_lr: torch.Tensor) -> list[torch.Tensor]:
        feat = F.silu(self.cond_in(z_lr))
        feats = [feat]
        for down in self.cond_downs:
            feat = F.silu(down(feat))
            feats.append(feat)
        return feats

    def _cond_for(
        self, feats: list[torch.Tensor], spatial: tuple[int, int]
    ) -> torch.Tensor:
        for feat in feats:
            if feat.shape[-2:] == spatial:
                return feat
        return F.interpolate(
            feats[-1], size=spatial, mode="bilinear", align_corners=False
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, z_lr: torch.Tensor
    ) -> torch.Tensor:
        if x_t.shape != z_lr.shape:
            raise ValueError(
                f"x_t shape {tuple(x_t.shape)} must match z_lr {tuple(z_lr.shape)}"
            )
        if x_t.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} latent channels, got {x_t.shape[1]}"
            )
        if t.shape != (x_t.shape[0],):
            raise ValueError(f"t shape {tuple(t.shape)} must be ({x_t.shape[0]},)")

        cond_feats = self._condition_pyramid(z_lr)
        time_emb = self.time_embed(t)
        h = self.conv_in(x_t)
        skips = [h]

        for level, blocks in enumerate(self.encoder):
            for block, attn in blocks:
                h = block(h, time_emb)
                h = attn(h)
                skips.append(h)
            h = self.enc_films[level](h, self._cond_for(cond_feats, h.shape[-2:]))
            h = self.downs[level](h)
            if not isinstance(self.downs[level], nn.Identity):
                skips.append(h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        h = self.mid_film(h, self._cond_for(cond_feats, h.shape[-2:]))

        for level, blocks in enumerate(self.decoder):
            for block, attn in blocks:
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)
                h = attn(h)
            h = self.dec_films[level](h, self._cond_for(cond_feats, h.shape[-2:]))
            h = self.ups[level](h)

        return self.conv_out(h)


def build_adagn_unet_from_config(config: dict[str, Any]) -> AdaGNLatentUNet:
    return AdaGNLatentUNet(
        latent_channels=int(config.get("latent_channels", 4)),
        base_channels=int(config.get("base_channels", 64)),
        channel_mult=tuple(int(m) for m in config.get("channel_mult", [1, 2, 4])),
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        attention_resolutions=tuple(
            int(r) for r in config.get("attention_resolutions", [16, 8])
        ),
        dropout=float(config.get("dropout", 0.1)),
        image_size=int(config.get("latent_size", 32)),
    )
