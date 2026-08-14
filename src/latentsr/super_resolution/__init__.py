"""Latent super-resolution (Phase 8+)."""

from latentsr.super_resolution.condition import (
    ConditionedLatentUNet,
    ConditionalLatentDDPM,
    build_conditioned_latent_ddpm_from_config,
    load_conditioned_latent_ddpm_checkpoint,
)
from latentsr.super_resolution.inference import (
    encode_lr_latents,
    load_sr_components,
    prepare_lr_batch,
    save_sr_comparison_grid,
    soft_decode_from_lr,
    super_resolve,
)
from latentsr.super_resolution.sample import sample_conditional_latents, sample_sr_images
from latentsr.super_resolution.trainer import LatentSRTrainer

__all__ = [
    "ConditionedLatentUNet",
    "ConditionalLatentDDPM",
    "LatentSRTrainer",
    "build_conditioned_latent_ddpm_from_config",
    "encode_lr_latents",
    "load_conditioned_latent_ddpm_checkpoint",
    "load_sr_components",
    "prepare_lr_batch",
    "sample_conditional_latents",
    "sample_sr_images",
    "save_sr_comparison_grid",
    "soft_decode_from_lr",
    "super_resolve",
]
