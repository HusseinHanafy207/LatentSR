"""Phase 4: latent-space DDPM build + one training step."""

from __future__ import annotations

from pathlib import Path

import torch
from generative_models.losses import DDPMLoss
from torch.utils.data import DataLoader, TensorDataset

from latentsr.datasets.onthefly_latent import OnTheFlyLatentEncoder
from latentsr.diffusion import (
    LatentDDPMTrainer,
    build_latent_ddpm_from_config,
    load_latent_ddpm_checkpoint,
)
from latentsr.utils.config import load_config
from latentsr.vae import VAE, freeze_vae


def _tiny_config(tmp_path: Path) -> dict:
    config = load_config(Path("configs/latent_ddpm.yaml"))
    config["base_channels"] = 32
    config["num_res_blocks"] = 1
    config["attention_resolutions"] = [8]
    config["channel_mult"] = [1, 2]
    config["latent_size"] = 32
    config["latent_channels"] = 4
    config["batch_size"] = 2
    config["epochs"] = 1
    config["amp"] = False
    config["device"] = "cpu"
    config["checkpoint_dir"] = str(tmp_path / "ckpt")
    config["sample_dir"] = str(tmp_path / "samples")
    config["log_dir"] = str(tmp_path / "logs")
    config["vae_checkpoint"] = "synthetic"
    config["latent_scale"] = 1.0
    config["validate_every"] = 1
    config["checkpoint_every"] = 1
    return config


def test_build_latent_ddpm_shapes() -> None:
    config = load_config(Path("configs/latent_ddpm.yaml"))
    config["base_channels"] = 32
    config["num_res_blocks"] = 1
    model = build_latent_ddpm_from_config(config)
    z = torch.randn(2, 4, 32, 32)
    noise_pred, noise, t = model(z)
    assert noise_pred.shape == z.shape
    assert noise.shape == z.shape
    assert t.shape == (2,)


def test_latent_ddpm_one_train_epoch(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    # Tiny UNet path: channel_mult length 2 → 32→16
    config["channel_mult"] = [1, 2]
    config["attention_resolutions"] = [16]

    vae = VAE(base_channels=32, num_res_blocks=1, channel_mult=(1, 2, 4))
    freeze_vae(vae)
    encoder = OnTheFlyLatentEncoder(vae, latent_scale=1.0)

    images = torch.rand(4, 3, 128, 128)
    loader = DataLoader(TensorDataset(images), batch_size=2)

    model = build_latent_ddpm_from_config(config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = LatentDDPMTrainer(
        model=model,
        optimizer=opt,
        criterion=DDPMLoss(),
        train_loader=loader,
        val_loader=loader,
        latent_encoder=encoder,
        config=config,
    )
    trainer.train()
    latest = Path(config["checkpoint_dir"]) / "latest.pt"
    assert latest.is_file()
    loaded, ckpt = load_latent_ddpm_checkpoint(latest)
    assert ckpt["epoch"] == 1
    assert ckpt["latent_scale"] == 1.0
    assert "vae_checkpoint" in ckpt
    z = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        pred, _, _ = loaded(z)
    assert pred.shape == z.shape


def test_checkpoint_records_scale_and_vae(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    config["channel_mult"] = [1, 2]
    config["attention_resolutions"] = [16]
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    encoder = OnTheFlyLatentEncoder(vae, latent_scale=0.5)
    config["latent_scale"] = 0.5
    config["vae_checkpoint"] = "outputs/vae/checkpoints/checkpoint_epoch_050.pt"

    images = torch.rand(2, 3, 128, 128)
    loader = DataLoader(TensorDataset(images), batch_size=2)
    model = build_latent_ddpm_from_config(config)
    trainer = LatentDDPMTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        criterion=DDPMLoss(),
        train_loader=loader,
        latent_encoder=encoder,
        config=config,
    )
    metrics = {"loss": 0.1, "lr": 1e-3, "epoch_time": 0.0}
    trainer.save_checkpoint(1, metrics, save_snapshot=True)
    ckpt = torch.load(Path(config["checkpoint_dir"]) / "latest.pt", weights_only=False)
    assert ckpt["latent_scale"] == 0.5
    assert "checkpoint_epoch_050" in ckpt["vae_checkpoint"]
