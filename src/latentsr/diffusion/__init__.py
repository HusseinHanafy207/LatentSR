"""Latent DDPM wrappers built on generative_models.ddpm."""

from latentsr.diffusion.build import (
    build_latent_ddpm_from_config,
    load_latent_ddpm_checkpoint,
)
from latentsr.diffusion.trainer import LatentDDPMTrainer

__all__ = [
    "LatentDDPMTrainer",
    "build_latent_ddpm_from_config",
    "load_latent_ddpm_checkpoint",
]
