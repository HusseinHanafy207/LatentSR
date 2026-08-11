"""Phase 8: conditioned latent UNet + SR trainer smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch
from generative_models.losses import DDPMLoss
from torch.utils.data import DataLoader, TensorDataset

from latentsr.datasets.onthefly_sr_latent import OnTheFlySRLatentEncoder
from latentsr.super_resolution.condition import (
    ConditionedLatentUNet,
    ConditionalLatentDDPM,
    build_conditioned_latent_ddpm_from_config,
    load_conditioned_latent_ddpm_checkpoint,
)
from latentsr.super_resolution.sample import sample_conditional_latents
from latentsr.super_resolution.trainer import LatentSRTrainer
from latentsr.vae import VAE, freeze_vae


def test_conditioned_unet_concat_channels() -> None:
    unet = ConditionedLatentUNet(
        latent_channels=4,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        image_size=32,
        dropout=0.0,
    )
    assert unet.in_channels == 8
    assert unet.out_channels == 4
    x_t = torch.randn(2, 4, 32, 32)
    z_lr = torch.randn(2, 4, 32, 32)
    t = torch.tensor([0, 1])
    out = unet(x_t, t, z_lr)
    assert out.shape == (2, 4, 32, 32)


def test_conditional_ddpm_forward() -> None:
    model = build_conditioned_latent_ddpm_from_config(
        {
            "latent_channels": 4,
            "latent_size": 32,
            "base_channels": 32,
            "channel_mult": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [16],
            "dropout": 0.0,
            "num_timesteps": 10,
        }
    )
    z_hr = torch.randn(2, 4, 32, 32)
    z_lr = torch.randn(2, 4, 32, 32)
    noise_pred, noise, t = model(z_hr, z_lr)
    assert noise_pred.shape == z_hr.shape
    assert noise.shape == z_hr.shape
    assert t.shape == (2,)


def test_sample_conditional_latents_short_chain() -> None:
    model = build_conditioned_latent_ddpm_from_config(
        {
            "latent_channels": 4,
            "latent_size": 32,
            "base_channels": 32,
            "channel_mult": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [16],
            "dropout": 0.0,
            "num_timesteps": 4,
        }
    )
    z_lr = torch.randn(1, 4, 32, 32)
    z_hr = sample_conditional_latents(model, z_lr, show_progress=False)
    assert z_hr.shape == z_lr.shape


def test_latent_sr_trainer_one_epoch(tmp_path: Path) -> None:
    config = {
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
        "sample_every": 0,  # skip slow full sampling in unit test
        "latent_channels": 4,
        "latent_size": 32,
        "base_channels": 32,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [16],
        "dropout": 0.0,
        "num_timesteps": 10,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "seed": 0,
    }
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    sr_encoder = OnTheFlySRLatentEncoder(vae, latent_scale=1.0, hr_size=128)

    lr = torch.rand(4, 3, 32, 32)
    hr = torch.rand(4, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

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
    assert latest.is_file()
    loaded, ckpt = load_conditioned_latent_ddpm_checkpoint(latest)
    assert ckpt["epoch"] == 1
    assert ckpt["latent_scale"] == 1.0
    z_lr = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        pred, _, _ = loaded(torch.randn_like(z_lr), z_lr)
    assert pred.shape == z_lr.shape
