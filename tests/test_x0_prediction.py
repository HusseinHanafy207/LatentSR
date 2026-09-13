"""Unit tests for x0-prediction parameterization (ε-matched loss)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from generative_models.ddpm import NoiseScheduler
from generative_models.losses import DDPMLoss

from latentsr.super_resolution.condition import (
    ConditionalLatentDDPM,
    build_conditioned_latent_ddpm_from_config,
    load_conditioned_latent_ddpm_checkpoint,
)
from latentsr.super_resolution.parameterization import (
    normalize_prediction_type,
    predict_eps_from_x0,
    predict_x0_from_eps,
)
from latentsr.super_resolution.sample import sample_conditional_latents


def _tiny_cfg(**overrides):
    cfg = {
        "latent_channels": 4,
        "latent_size": 16,
        "base_channels": 16,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [8],
        "dropout": 0.0,
        "num_timesteps": 5,
        "prediction_type": "eps",
    }
    cfg.update(overrides)
    return cfg


def test_normalize_prediction_type() -> None:
    assert normalize_prediction_type("eps") == "eps"
    assert normalize_prediction_type("epsilon") == "eps"
    assert normalize_prediction_type("x0") == "x0"
    assert normalize_prediction_type("z0") == "x0"
    with pytest.raises(ValueError):
        normalize_prediction_type("v")


def test_eps_x0_roundtrip() -> None:
    scheduler = NoiseScheduler(num_timesteps=10)
    x_t = torch.randn(3, 4, 8, 8)
    t = torch.tensor([0, 4, 9], dtype=torch.long)
    eps = torch.randn(3, 4, 8, 8)

    x0 = predict_x0_from_eps(scheduler, x_t, t, eps)
    eps_back = predict_eps_from_x0(scheduler, x_t, t, x0)
    assert torch.allclose(eps_back, eps, atol=1e-5)

    x0_2 = torch.randn(3, 4, 8, 8)
    eps_2 = predict_eps_from_x0(scheduler, x_t, t, x0_2)
    x0_back = predict_x0_from_eps(scheduler, x_t, t, eps_2)
    assert torch.allclose(x0_back, x0_2, atol=1e-5)


def test_matched_loss_equals_snr_weighted_x0_mse() -> None:
    """L_eps = SNR(t) * L_x0 when eps_hat comes from x0_hat conversion."""
    scheduler = NoiseScheduler(num_timesteps=20)
    z0 = torch.randn(4, 4, 8, 8)
    eps = torch.randn(4, 4, 8, 8)
    t = torch.tensor([1, 5, 10, 15], dtype=torch.long)
    x_t = scheduler.q_sample(z0, t, noise=eps)

    # Pretend network predicted a wrong x0
    x0_hat = z0 + 0.1 * torch.randn_like(z0)
    eps_hat = predict_eps_from_x0(scheduler, x_t, t, x0_hat)

    # Per-sample MSE
    eps_mse = (eps_hat - eps).pow(2).flatten(1).mean(dim=1)
    x0_mse = (x0_hat - z0).pow(2).flatten(1).mean(dim=1)

    ab = scheduler._extract(scheduler.alphas_cumprod, t, z0.shape).flatten(1)[:, 0]
    snr = ab / (1.0 - ab).clamp_min(1e-12)
    assert torch.allclose(eps_mse, snr * x0_mse, rtol=1e-4, atol=1e-5)


def test_x0_model_predict_noise_matches_oracle_conversion() -> None:
    """If UNet returns true z0, predict_noise must recover true eps."""
    model = build_conditioned_latent_ddpm_from_config(_tiny_cfg(prediction_type="x0"))
    assert model.prediction_type == "x0"

    z_hr = torch.randn(2, 4, 16, 16)
    z_lr = torch.randn(2, 4, 16, 16)
    t = torch.tensor([1, 3], dtype=torch.long)
    eps = torch.randn_like(z_hr)
    x_t = model.scheduler.q_sample(z_hr, t, noise=eps)

    # Monkeypatch UNet to return true clean latent
    def _oracle_x0(x, tt, zl):
        return z_hr

    model.unet.forward = _oracle_x0  # type: ignore[method-assign]
    eps_hat = model.predict_noise(x_t, t, z_lr)
    assert torch.allclose(eps_hat, eps, atol=1e-5)

    x0_hat = model.predict_x0(x_t, t, z_lr)
    assert torch.allclose(x0_hat, z_hr, atol=1e-5)


def test_x0_forward_loss_finite_and_trainable() -> None:
    model = build_conditioned_latent_ddpm_from_config(_tiny_cfg(prediction_type="x0"))
    criterion = DDPMLoss()
    z_hr = torch.randn(2, 4, 16, 16)
    z_lr = torch.randn(2, 4, 16, 16)
    noise_pred, noise, t = model(z_hr, z_lr)
    loss = criterion(noise_pred, noise)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_eps_model_default_unchanged() -> None:
    model = build_conditioned_latent_ddpm_from_config(_tiny_cfg())
    assert model.prediction_type == "eps"
    z_lr = torch.randn(1, 4, 16, 16)
    x = torch.randn(1, 4, 16, 16)
    t = torch.tensor([2])
    raw = model.predict_raw(x, t, z_lr)
    eps = model.predict_noise(x, t, z_lr)
    assert torch.equal(raw, eps)


def test_sample_conditional_latents_x0() -> None:
    model = build_conditioned_latent_ddpm_from_config(
        _tiny_cfg(prediction_type="x0", num_timesteps=4)
    )
    z_lr = torch.randn(2, 4, 16, 16)
    z = sample_conditional_latents(
        model,
        z_lr,
        val_indices=[0, 1],
        noise_seed=7,
        show_progress=False,
        sampler="ddpm",
    )
    assert z.shape == z_lr.shape
    assert torch.isfinite(z).all()

    z_ddim = sample_conditional_latents(
        model,
        z_lr,
        val_indices=[0, 1],
        noise_seed=7,
        show_progress=False,
        sampler="ddim",
        ddim_eta=0.0,
    )
    assert z_ddim.shape == z_lr.shape


def test_checkpoint_roundtrip_preserves_prediction_type(tmp_path: Path) -> None:
    cfg = _tiny_cfg(prediction_type="x0", hr_size=128, vae_checkpoint="dummy.pt")
    model = build_conditioned_latent_ddpm_from_config(cfg)
    ckpt = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "prediction_type": "x0",
        "latent_scale": 1.0,
    }
    path = tmp_path / "x0.pt"
    torch.save(ckpt, path)

    loaded, raw = load_conditioned_latent_ddpm_checkpoint(path)
    assert loaded.prediction_type == "x0"
    assert raw["config"]["prediction_type"] == "x0"


def test_old_checkpoint_defaults_to_eps(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    cfg.pop("prediction_type", None)
    model = build_conditioned_latent_ddpm_from_config(cfg)
    path = tmp_path / "old.pt"
    torch.save(
        {"epoch": 1, "model_state_dict": model.state_dict(), "config": cfg},
        path,
    )
    loaded, _ = load_conditioned_latent_ddpm_checkpoint(path)
    assert loaded.prediction_type == "eps"
