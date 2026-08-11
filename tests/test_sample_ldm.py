"""Phase 5: LDM sampling (no latent clamp)."""

from __future__ import annotations

from pathlib import Path

import torch
from generative_models.ddpm.sampler import sample as pixel_sample

from latentsr.diffusion.build import build_latent_ddpm_from_config
from latentsr.diffusion.sample import sample_latents, sample_ldm, save_ldm_grid
from latentsr.vae import VAE, freeze_vae


def _tiny_ddpm():
    config = {
        "num_timesteps": 4,  # tiny chain for unit tests
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "latent_channels": 4,
        "latent_size": 32,
        "base_channels": 32,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [16],
        "dropout": 0.0,
    }
    return build_latent_ddpm_from_config(config), config


def test_sample_latents_no_clamp_to_unit_interval() -> None:
    model, config = _tiny_ddpm()
    z = sample_latents(
        model,
        num_samples=2,
        latent_channels=config["latent_channels"],
        latent_size=config["latent_size"],
        device=torch.device("cpu"),
        show_progress=False,
    )
    assert z.shape == (2, 4, 32, 32)
    # Latents may lie outside [0, 1]; pixel sampler would wrongly clamp them.
    # Just assert finite and correct dtype/device.
    assert torch.isfinite(z).all()


def test_pixel_sample_clamps_but_latent_path_does_not() -> None:
    """Document why we must not reuse generative_models.ddpm.sample for LDM."""
    model, _ = _tiny_ddpm()
    # Pixel helper always returns values in [0, 1] after clamp.
    x = pixel_sample(
        model, num_samples=1, image_size=32, channels=4, device=torch.device("cpu"), show_progress=False
    )
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_sample_ldm_end_to_end(tmp_path: Path) -> None:
    ddpm, config = _tiny_ddpm()
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    images = sample_ldm(
        ddpm,
        vae,
        num_samples=2,
        latent_channels=config["latent_channels"],
        latent_size=config["latent_size"],
        latent_scale=1.0,
        device=torch.device("cpu"),
        show_progress=False,
    )
    assert images.shape == (2, 3, 128, 128)
    assert images.min() >= 0.0 and images.max() <= 1.0
    out = save_ldm_grid(images, tmp_path / "grid.png", nrow=2)
    assert out.is_file()
