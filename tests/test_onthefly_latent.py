"""Phase 3: on-the-fly latent encoding (no disk cache)."""

from __future__ import annotations

from pathlib import Path

import torch

from latentsr.datasets.onthefly_latent import OnTheFlyLatentEncoder, batch_to_images
from latentsr.vae import VAE, freeze_vae


def test_batch_to_images() -> None:
    images = torch.rand(4, 3, 128, 128)
    assert batch_to_images(images) is images
    assert batch_to_images((images, "attrs")) is images


def test_onthefly_encoder_shapes() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    encoder = OnTheFlyLatentEncoder(vae, latent_scale=1.0)
    images = torch.rand(2, 3, 128, 128)
    z = encoder(images)
    assert z.shape == (2, 4, 32, 32)


def test_onthefly_rejects_unfrozen_vae() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    try:
        OnTheFlyLatentEncoder(vae, latent_scale=1.0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "frozen" in str(exc).lower()


def test_set_latent_scale_no_cache() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    encoder = OnTheFlyLatentEncoder(vae, latent_scale=1.0)
    images = torch.rand(1, 3, 128, 128)
    z1 = encoder(images)
    encoder.set_latent_scale(2.0)
    z2 = encoder(images)
    assert torch.allclose(z2, z1 * 2.0, atol=1e-5)


def test_encode_batch_from_tuple() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    encoder = OnTheFlyLatentEncoder(vae, latent_scale=1.0)
    images = torch.rand(3, 3, 128, 128)
    z = encoder.encode_batch((images, torch.zeros(3, 40)))
    assert z.shape == (3, 4, 32, 32)


def test_from_checkpoint_if_present() -> None:
    ckpt = Path("outputs/vae/checkpoints/checkpoint_epoch_050.pt")
    if not ckpt.is_file():
        return
    encoder = OnTheFlyLatentEncoder.from_checkpoint(ckpt, latent_scale=1.0)
    images = torch.rand(2, 3, 128, 128)
    z = encoder.encode_batch(images)
    assert z.shape == (2, encoder.latent_channels, 32, 32)
