"""AdaGN / FiLM latent UNet (concat replacement)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from generative_models.losses import DDPMLoss
from torch.utils.data import DataLoader, TensorDataset

from latentsr.datasets.onthefly_sr_latent import OnTheFlySRLatentEncoder
from latentsr.super_resolution.adagn import AdaGNLatentUNet
from latentsr.super_resolution.condition import (
    build_conditioned_latent_ddpm_from_config,
    load_conditioned_latent_ddpm_checkpoint,
)
from latentsr.super_resolution.trainer import LatentSRTrainer
from latentsr.vae import VAE, freeze_vae


def _tiny_adagn_config() -> dict:
    return {
        "condition_type": "adagn",
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
    }


def test_adagn_forward_shape() -> None:
    unet = AdaGNLatentUNet(
        latent_channels=4,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        dropout=0.0,
        image_size=32,
    )
    assert unet.in_channels == 4
    x_t = torch.randn(2, 4, 32, 32)
    z_lr = torch.randn(2, 4, 32, 32)
    t = torch.tensor([0, 3])
    out = unet(x_t, t, z_lr)
    assert out.shape == (2, 4, 32, 32)


def test_adagn_film_is_identity_at_init() -> None:
    unet = AdaGNLatentUNet(
        latent_channels=4,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        dropout=0.0,
        image_size=32,
    )
    unet.eval()
    x_t = torch.randn(2, 4, 32, 32)
    t = torch.tensor([1, 2])
    z_a = torch.randn(2, 4, 32, 32)
    z_b = torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        y_a = unet(x_t, t, z_a)
        y_b = unet(x_t, t, z_b)
    assert torch.allclose(y_a, y_b, atol=1e-5)


def test_adagn_uses_condition_after_film_init() -> None:
    unet = AdaGNLatentUNet(
        latent_channels=4,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        dropout=0.0,
        image_size=32,
    )
    torch.nn.init.normal_(unet.enc_films[0].proj.weight, std=0.05)
    unet.eval()
    x_t = torch.randn(2, 4, 32, 32)
    t = torch.tensor([1, 2])
    with torch.no_grad():
        y_a = unet(x_t, t, torch.randn(2, 4, 32, 32))
        y_b = unet(x_t, t, torch.randn(2, 4, 32, 32))
    assert not torch.allclose(y_a, y_b, atol=1e-4)


def test_build_adagn_from_config() -> None:
    model = build_conditioned_latent_ddpm_from_config(_tiny_adagn_config())
    assert getattr(model.unet, "condition_type", None) == "adagn"
    z_hr = torch.randn(2, 4, 32, 32)
    z_lr = torch.randn(2, 4, 32, 32)
    noise_pred, noise, t = model(z_hr, z_lr)
    assert noise_pred.shape == z_hr.shape
    assert noise.shape == z_hr.shape
    assert t.shape == (2,)


def test_unknown_condition_type_raises() -> None:
    with pytest.raises(ValueError, match="condition_type"):
        build_conditioned_latent_ddpm_from_config(
            {**_tiny_adagn_config(), "condition_type": "crossattn"}
        )


def test_adagn_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = {
        **_tiny_adagn_config(),
        "epochs": 1,
        "amp": False,
        "device": "cpu",
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "sample_dir": str(tmp_path / "samples"),
        "log_dir": str(tmp_path / "logs"),
        "vae_checkpoint": "synthetic",
        "latent_scale": 1.0,
        "hr_size": 128,
        "validate_every": 1,
        "checkpoint_every": 1,
        "sample_every": 0,
        "seed": 0,
    }
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    sr_encoder = OnTheFlySRLatentEncoder(vae, latent_scale=1.0, hr_size=128)
    loader = DataLoader(
        TensorDataset(torch.rand(4, 3, 32, 32), torch.rand(4, 3, 128, 128)),
        batch_size=2,
    )
    model = build_conditioned_latent_ddpm_from_config(config)
    trainer = LatentSRTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        criterion=DDPMLoss(),
        train_loader=loader,
        val_loader=loader,
        sr_encoder=sr_encoder,
        config=config,
    )
    trainer.train()
    latest = Path(config["checkpoint_dir"]) / "latest.pt"
    loaded, ckpt = load_conditioned_latent_ddpm_checkpoint(latest)
    assert ckpt["config"]["condition_type"] == "adagn"
    assert getattr(loaded.unet, "condition_type") == "adagn"
    z_lr = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        pred, _, _ = loaded(torch.randn_like(z_lr), z_lr)
    assert pred.shape == z_lr.shape
