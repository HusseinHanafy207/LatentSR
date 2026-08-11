"""Build / load latent-space DDPM models (generative_models.ddpm backbone)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from generative_models.ddpm import DDPM, NoiseScheduler, UNet


def build_latent_ddpm_from_config(config: dict[str, Any]) -> DDPM:
    """Construct a DDPM that operates on VAE latents (default 4×32×32)."""
    latent_channels = int(config.get("latent_channels", config.get("in_channels", 4)))
    latent_size = int(config.get("latent_size", config.get("image_size", 32)))
    channel_mult = tuple(int(m) for m in config.get("channel_mult", [1, 2, 4]))
    attention_resolutions = tuple(
        int(r) for r in config.get("attention_resolutions", [16, 8])
    )

    scheduler = NoiseScheduler(
        num_timesteps=int(config.get("num_timesteps", 1000)),
        beta_start=float(config.get("beta_start", 1e-4)),
        beta_end=float(config.get("beta_end", 0.02)),
    )
    unet = UNet(
        in_channels=latent_channels,
        out_channels=latent_channels,
        base_channels=int(config.get("base_channels", 64)),
        channel_mult=channel_mult,
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        attention_resolutions=attention_resolutions,
        dropout=float(config.get("dropout", 0.1)),
        image_size=latent_size,
    )
    return DDPM(unet=unet, scheduler=scheduler)


def load_latent_ddpm_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[DDPM, dict[str, Any]]:
    """Load a latent DDPM checkpoint; returns ``(model, checkpoint_dict)``."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint missing config dict: {path}")
    model = build_latent_ddpm_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
