"""Latent DDPM wrappers built on generative_models.ddpm."""

from latentsr.diffusion.build import (
    build_latent_ddpm_from_config,
    load_latent_ddpm_checkpoint,
)
from latentsr.diffusion.sample import (
    load_ldm_components,
    sample_latents,
    sample_ldm,
    save_ldm_grid,
)
from latentsr.diffusion.trainer import LatentDDPMTrainer

__all__ = [
    "LatentDDPMTrainer",
    "build_latent_ddpm_from_config",
    "load_latent_ddpm_checkpoint",
    "load_ldm_components",
    "sample_latents",
    "sample_ldm",
    "save_ldm_grid",
]
