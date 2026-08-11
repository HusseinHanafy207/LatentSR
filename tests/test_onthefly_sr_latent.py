"""Phase 7: on-the-fly SR latent pairs."""

from __future__ import annotations

from pathlib import Path

import torch

from latentsr.datasets.onthefly_sr_latent import (
    OnTheFlySRLatentEncoder,
    upsample_bicubic,
)
from latentsr.vae import VAE, freeze_vae


def test_upsample_shapes() -> None:
    lr = torch.rand(2, 3, 32, 32)
    hr = upsample_bicubic(lr, 128)
    assert hr.shape == (2, 3, 128, 128)


def test_sr_latent_encoder_shapes() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    enc = OnTheFlySRLatentEncoder(vae, latent_scale=1.0, hr_size=128)
    lr = torch.rand(2, 3, 32, 32)
    hr = torch.rand(2, 3, 128, 128)
    z_lr, z_hr = enc(lr, hr)
    assert z_lr.shape == (2, 4, 32, 32)
    assert z_hr.shape == (2, 4, 32, 32)


def test_sr_latent_rejects_unfrozen() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    try:
        OnTheFlySRLatentEncoder(vae)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "frozen" in str(exc).lower()


def test_set_scale_changes_magnitude() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    enc = OnTheFlySRLatentEncoder(vae, latent_scale=1.0, hr_size=128)
    lr = torch.rand(1, 3, 32, 32)
    hr = torch.rand(1, 3, 128, 128)
    z1_lr, z1_hr = enc(lr, hr)
    enc.set_latent_scale(2.0)
    z2_lr, z2_hr = enc(lr, hr)
    assert torch.allclose(z2_lr, z1_lr * 2.0, atol=1e-5)
    assert torch.allclose(z2_hr, z1_hr * 2.0, atol=1e-5)


def test_from_checkpoint_if_present() -> None:
    ckpt = Path("outputs/vae/checkpoints/checkpoint_epoch_050.pt")
    if not ckpt.is_file():
        return
    enc = OnTheFlySRLatentEncoder.from_checkpoint(ckpt, latent_scale=1.0)
    lr = torch.rand(2, 3, 32, 32)
    hr = torch.rand(2, 3, 128, 128)
    z_lr, z_hr = enc(lr, hr)
    assert z_lr.shape == z_hr.shape == (2, enc.latent_channels, 32, 32)
