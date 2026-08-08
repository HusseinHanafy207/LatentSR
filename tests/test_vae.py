"""Convolutional VAE shape and loss smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch

from latentsr.utils.config import load_config
from latentsr.vae import VAE, VAELoss, build_vae_from_config, load_vae_checkpoint


def test_vae_forward_shapes() -> None:
    model = VAE(
        in_channels=3,
        latent_channels=4,
        base_channels=32,
        channel_mult=(1, 2, 4),
        num_res_blocks=1,
    )
    x = torch.rand(2, 3, 128, 128)
    recon, mu, logvar = model(x)
    assert recon.shape == (2, 3, 128, 128)
    assert mu.shape == (2, 4, 32, 32)
    assert logvar.shape == (2, 4, 32, 32)
    assert torch.all((recon >= 0) & (recon <= 1))


def test_vae_reconstruct_uses_mean() -> None:
    model = VAE(base_channels=32, num_res_blocks=1)
    model.eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        a = model.reconstruct(x, use_mean=True)
        b = model.reconstruct(x, use_mean=True)
    assert torch.allclose(a, b)


def test_vae_loss_finite() -> None:
    criterion = VAELoss(kl_weight=1e-4)
    recon = torch.rand(2, 3, 128, 128)
    images = torch.rand(2, 3, 128, 128)
    mu = torch.randn(2, 4, 32, 32)
    logvar = torch.zeros(2, 4, 32, 32)
    total, recon_loss, kl_loss = criterion(recon, images, mu, logvar)
    assert total.ndim == 0
    assert torch.isfinite(total)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)


def test_build_from_config_and_checkpoint(tmp_path: Path) -> None:
    config = load_config(Path("configs/vae_celeba.yaml"))
    config["base_channels"] = 32
    config["num_res_blocks"] = 1
    model = build_vae_from_config(config)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        recon, _, _ = model(x)

    ckpt_path = tmp_path / "vae.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "metrics": {},
            "config": config,
            "arch": model.config_dict(),
        },
        ckpt_path,
    )
    loaded, ckpt = load_vae_checkpoint(ckpt_path)
    assert ckpt["epoch"] == 1
    with torch.no_grad():
        recon2 = loaded.reconstruct(x, use_mean=True)
    assert recon2.shape == recon.shape


def test_one_train_step() -> None:
    model = VAE(base_channels=32, num_res_blocks=1)
    criterion = VAELoss(kl_weight=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 128, 128)
    opt.zero_grad()
    recon, mu, logvar = model(x)
    loss, _, _ = criterion(recon, x, mu, logvar)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
