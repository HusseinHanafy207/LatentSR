"""Timestep diagnostic: ẑ0 vs z_lr along the reverse chain."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from generative_models.ddpm import NoiseScheduler
from latentsr.metrics.timestep_diagnostic import (
    latent_cosine,
    latent_rmse,
    run_timestep_diagnostic,
)
from latentsr.super_resolution.condition import build_conditioned_latent_ddpm_from_config
from latentsr.super_resolution.sample import predict_x0_from_eps
from latentsr.vae import VAE, freeze_vae


def _tiny_concat_config(**overrides: object) -> dict:
    cfg: dict = {
        "condition_type": "concat",
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
    cfg.update(overrides)
    return cfg


def test_predict_x0_recovers_clean_latent() -> None:
    scheduler = NoiseScheduler(num_timesteps=8, beta_start=1e-4, beta_end=0.02)
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn_like(x0)
    t = torch.tensor([3, 7])
    x_t = scheduler.q_sample(x0, t, noise=eps)
    hat = predict_x0_from_eps(scheduler, x_t, t, eps)
    assert torch.allclose(hat, x0, atol=1e-5)


def test_latent_rmse_and_cosine() -> None:
    a = torch.ones(2, 4, 2, 2)
    b = torch.ones(2, 4, 2, 2)
    assert torch.allclose(latent_rmse(a, b), torch.zeros(2))
    assert torch.allclose(latent_cosine(a, a * 3), torch.ones(2))


def test_identical_models_zero_z0_gap(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    clone = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    clone.load_state_dict(model.state_dict())
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)

    lr = torch.rand(3, 3, 32, 32)
    hr = torch.rand(3, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

    result = run_timestep_diagnostic(
        model,
        vae,
        clone,
        vae,
        loader,
        device=torch.device("cpu"),
        num_images=3,
        hr_size=128,
        noise_seed=42,
        output_dir=tmp_path,
        show_progress=False,
    )
    assert result["num_images"] == 3
    assert result["num_timesteps"] == 4
    assert (tmp_path / "timestep_means.csv").is_file()
    assert (tmp_path / "cosine_z0_z_lr.png").is_file()
    for row in result["rows"]:
        assert row["z_lr_rmse_mean"] == pytest.approx(0.0, abs=1e-5)
        assert row["z0_rmse_sr_vs_vae1_mean"] == pytest.approx(0.0, abs=1e-5)


def test_different_z_lr_is_constant_in_t() -> None:
    torch.manual_seed(1)
    model_a = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    model_b = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    vae_a = VAE(base_channels=32, num_res_blocks=1)
    vae_b = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae_a)
    freeze_vae(vae_b)

    lr = torch.rand(2, 3, 32, 32)
    hr = torch.rand(2, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)
    result = run_timestep_diagnostic(
        model_a,
        vae_a,
        model_b,
        vae_b,
        loader,
        device=torch.device("cpu"),
        num_images=2,
        hr_size=128,
        noise_seed=42,
        show_progress=False,
    )
    gaps = [row["z_lr_rmse_mean"] for row in result["rows"]]
    assert max(gaps) > 1e-4
    assert max(gaps) - min(gaps) < 1e-8
