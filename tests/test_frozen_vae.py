"""Phase 2: frozen VAE + latent_scale helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from latentsr.vae import VAE
from latentsr.vae.latent import (
    decode_scaled,
    encode_scaled,
    estimate_latent_scale,
    freeze_vae,
    is_frozen,
    load_frozen_vae,
)


def test_freeze_vae_disables_grad() -> None:
    model = VAE(base_channels=32, num_res_blocks=1)
    assert any(p.requires_grad for p in model.parameters())
    freeze_vae(model)
    assert is_frozen(model)
    assert not model.training


def test_encode_decode_scaled_identity_at_scale_one() -> None:
    model = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(model)
    x = torch.rand(2, 3, 128, 128)
    z = encode_scaled(model, x, latent_scale=1.0)
    assert z.shape == (2, 4, 32, 32)
    recon_scaled = decode_scaled(model, z, latent_scale=1.0)
    recon_mu = model.reconstruct(x, use_mean=True)
    assert torch.allclose(recon_scaled, recon_mu, atol=1e-6)


def test_encode_decode_scaled_roundtrip_arbitrary_scale() -> None:
    model = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(model)
    x = torch.rand(1, 3, 128, 128)
    scale = 0.18215  # SD-like placeholder
    z = encode_scaled(model, x, latent_scale=scale)
    recon = decode_scaled(model, z, latent_scale=scale)
    recon_mu = model.reconstruct(x, use_mean=True)
    assert torch.allclose(recon, recon_mu, atol=1e-5)


def test_estimate_latent_scale_positive() -> None:
    model = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(model)
    x = torch.rand(4, 3, 128, 128)
    scale = estimate_latent_scale(model, x)
    assert scale > 0
    assert isinstance(scale, float)


def test_load_frozen_vae_real_checkpoint() -> None:
    ckpt = Path("outputs/vae/checkpoints/checkpoint_epoch_050.pt")
    if not ckpt.is_file():
        return  # skip when artifacts are not present
    model, checkpoint = load_frozen_vae(ckpt, map_location="cpu")
    assert checkpoint["epoch"] == 50
    assert is_frozen(model)
    x = torch.rand(2, 3, 128, 128)
    z = encode_scaled(model, x, 1.0)
    assert z.shape[1:] == (model.latent_channels, 32, 32)
    recon = decode_scaled(model, z, 1.0)
    assert recon.shape == x.shape
    # No parameter should receive a gradient through encode_scaled.
    assert all(p.grad is None for p in model.parameters())
