"""LR-conditioned UNet for latent super-resolution.

Concat layout (Phase 8):

    [noisy_z_hr | z_lr]  →  in_channels = 2 * C_z
    UNet → predicted noise ε̂  (out_channels = C_z)

AdaGN / FiLM (Q2 follow-up): ``x_t`` stays ``C_z`` channels; ``z_lr`` modulates
features at every UNet scale. Set ``condition_type: adagn`` in the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from generative_models.ddpm import NoiseScheduler, UNet
from generative_models.ddpm.forward import forward_diffuse

from latentsr.super_resolution.adagn import build_adagn_unet_from_config


class ConditionedLatentUNet(nn.Module):
    """Thin wrapper: concat ``[x_t | z_lr]``, then call ``generative_models.ddpm.UNet``."""

    def __init__(self, latent_channels: int = 4, **unet_kwargs: Any) -> None:
        super().__init__()
        if latent_channels < 1:
            raise ValueError(f"latent_channels must be >= 1, got {latent_channels}")
        if "in_channels" in unet_kwargs or "out_channels" in unet_kwargs:
            raise TypeError(
                "Pass latent_channels=... instead of in_channels/out_channels; "
                "ConditionedLatentUNet sets those from the concat layout."
            )
        self.latent_channels = latent_channels
        self.in_channels = 2 * latent_channels
        self.out_channels = latent_channels
        self.condition_type = "concat"
        self.unet = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            **unet_kwargs,
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
        return self.unet(torch.cat([x_t, z_lr], dim=1), t)


class ConditionalLatentDDPM(nn.Module):
    """ε-prediction DDPM on ``z_hr``, conditioned on ``z_lr``."""

    def __init__(
        self,
        unet: nn.Module,
        scheduler: NoiseScheduler | None = None,
        *,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler or NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

    @property
    def num_timesteps(self) -> int:
        return self.scheduler.num_timesteps

    def predict_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, z_lr: torch.Tensor
    ) -> torch.Tensor:
        return self.unet(x_t, t, z_lr)

    def forward(
        self,
        z_hr: torch.Tensor,
        z_lr: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t, t, noise = forward_diffuse(self.scheduler, z_hr, t=t, noise=noise)
        noise_pred = self.predict_noise(x_t, t, z_lr)
        return noise_pred, noise, t


def build_conditioned_latent_ddpm_from_config(
    config: dict[str, Any],
) -> ConditionalLatentDDPM:
    """Build concat or AdaGN LatentSR from a YAML config dict."""
    condition_type = str(config.get("condition_type", "concat")).strip().lower()
    if condition_type in {"adagn", "film"}:
        unet: nn.Module = build_adagn_unet_from_config(config)
    elif condition_type == "concat":
        latent_channels = int(config.get("latent_channels", 4))
        latent_size = int(config.get("latent_size", 32))
        unet = ConditionedLatentUNet(
            latent_channels=latent_channels,
            base_channels=int(config.get("base_channels", 64)),
            channel_mult=tuple(int(m) for m in config.get("channel_mult", [1, 2, 4])),
            num_res_blocks=int(config.get("num_res_blocks", 2)),
            attention_resolutions=tuple(
                int(r) for r in config.get("attention_resolutions", [16, 8])
            ),
            dropout=float(config.get("dropout", 0.1)),
            image_size=latent_size,
        )
    else:
        raise ValueError(
            f"Unknown condition_type={condition_type!r}; use 'concat' or 'adagn'."
        )
    return ConditionalLatentDDPM(
        unet=unet,
        num_timesteps=int(config.get("num_timesteps", 1000)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
    )


def load_conditioned_latent_ddpm_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ConditionalLatentDDPM, dict[str, Any]]:
    """Load a Phase-8 checkpoint; returns ``(model, checkpoint_dict)``."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint missing config dict: {path}")
    model = build_conditioned_latent_ddpm_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
