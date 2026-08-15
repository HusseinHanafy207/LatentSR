"""Convolutional VAE for spatial latents (Phase 1+)."""

from latentsr.vae.checkpointing import build_vae_from_config, load_vae_checkpoint
from latentsr.vae.decoder import Decoder
from latentsr.vae.encoder import Encoder
from latentsr.vae.latent import (
    decode_scaled,
    encode_scaled,
    estimate_latent_scale,
    freeze_vae,
    is_frozen,
    load_frozen_vae,
)
from latentsr.vae.loss import SRAwareVAELoss, VAELoss
from latentsr.vae.trainer import SRAwareVAETrainer, VAETrainer
from latentsr.vae.vae import VAE

__all__ = [
    "Decoder",
    "Encoder",
    "VAE",
    "VAELoss",
    "SRAwareVAELoss",
    "VAETrainer",
    "SRAwareVAETrainer",
    "build_vae_from_config",
    "decode_scaled",
    "encode_scaled",
    "estimate_latent_scale",
    "freeze_vae",
    "is_frozen",
    "load_frozen_vae",
    "load_vae_checkpoint",
]
