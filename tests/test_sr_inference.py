"""Phase 9: LR → HR inference helpers + comparison grids."""

from __future__ import annotations

from pathlib import Path

import torch

from latentsr.super_resolution.condition import build_conditioned_latent_ddpm_from_config
from latentsr.super_resolution.inference import (
    encode_lr_latents,
    prepare_lr_batch,
    save_sr_comparison_grid,
    soft_decode_from_lr,
    super_resolve,
)
from latentsr.vae import VAE, freeze_vae


def _tiny_sr_model():
    config = {
        "latent_channels": 4,
        "latent_size": 32,
        "base_channels": 32,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [16],
        "dropout": 0.0,
        "num_timesteps": 4,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "hr_size": 128,
        "lr_size": 32,
    }
    return build_conditioned_latent_ddpm_from_config(config), config


def test_prepare_lr_batch_from_hr() -> None:
    hr = torch.rand(2, 3, 128, 128)
    lr, hr_out = prepare_lr_batch(hr, lr_size=32, hr_size=128)
    assert lr.shape == (2, 3, 32, 32)
    assert hr_out is not None and hr_out.shape == (2, 3, 128, 128)


def test_prepare_lr_batch_from_lr() -> None:
    lr_in = torch.rand(1, 3, 32, 32)
    lr, hr_out = prepare_lr_batch(lr_in, lr_size=32, hr_size=128)
    assert lr.shape == (1, 3, 32, 32)
    assert hr_out is None


def test_encode_lr_latents_shape() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    lr = torch.rand(2, 3, 32, 32)
    z_lr = encode_lr_latents(vae, lr, hr_size=128, latent_scale=1.0)
    assert z_lr.shape == (2, 4, 32, 32)


def test_super_resolve_end_to_end(tmp_path: Path) -> None:
    model, _ = _tiny_sr_model()
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    lr = torch.rand(2, 3, 32, 32)
    pred = super_resolve(
        model,
        vae,
        lr,
        hr_size=128,
        latent_scale=1.0,
        show_progress=False,
    )
    assert pred.shape == (2, 3, 128, 128)
    assert pred.min() >= 0.0 and pred.max() <= 1.0

    soft = soft_decode_from_lr(vae, lr, hr_size=128, latent_scale=1.0)
    out = save_sr_comparison_grid(
        lr,
        pred,
        hr=torch.rand(2, 3, 128, 128),
        output_path=tmp_path / "compare.png",
        hr_size=128,
        include_soft_decode=True,
        soft=soft,
    )
    assert out.is_file()
