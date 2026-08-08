"""Checkpoint helpers for the convolutional VAE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from latentsr.vae.vae import VAE


def build_vae_from_config(config: dict[str, Any]) -> VAE:
    """Construct a :class:`VAE` from a training / checkpoint config dict."""
    channel_mult = config.get("channel_mult", [1, 2, 4])
    return VAE(
        in_channels=int(config.get("in_channels", 3)),
        latent_channels=int(config.get("latent_channels", 4)),
        base_channels=int(config.get("base_channels", 128)),
        channel_mult=tuple(int(m) for m in channel_mult),
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        dropout=float(config.get("dropout", 0.0)),
    )


def load_vae_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[VAE, dict[str, Any]]:
    """Load a VAE checkpoint; returns ``(model, checkpoint_dict)`` in eval mode."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    arch = checkpoint.get("arch") or {}
    config = checkpoint.get("config") or {}
    build_cfg = {**config, **arch}
    model = build_vae_from_config(build_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
