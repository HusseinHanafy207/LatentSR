"""Unit tests for DDIM sampling, condition utility diagnostic, and DDPM vs DDIM."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from generative_models.ddpm import NoiseScheduler
from latentsr.metrics.condition_utility import (
    compute_step_condition_utility,
    evaluate_condition_utility_forward,
    evaluate_condition_utility_reverse,
    format_milestone_table,
    generate_condition_utility_report,
    plot_condition_utility,
    save_condition_utility_results,
    shuffle_condition,
)
from latentsr.metrics.ddim_diagnostic import (
    format_ddpm_vs_ddim_table,
    generate_ddpm_vs_ddim_report,
    plot_ddpm_vs_ddim_curves,
    run_ddpm_vs_ddim_comparison,
    save_ddpm_vs_ddim_grid,
    save_ddpm_vs_ddim_results,
)
from latentsr.super_resolution.condition import build_conditioned_latent_ddpm_from_config
from latentsr.super_resolution.sample import (
    ddim_step,
    predict_x0_from_eps,
    sample_conditional_latents,
)
from latentsr.vae import VAE, freeze_vae


def _tiny_sr_model():
    cfg = {
        "num_timesteps": 5,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "latent_channels": 4,
        "latent_size": 16,
        "base_channels": 16,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [8],
        "dropout": 0.0,
    }
    return build_conditioned_latent_ddpm_from_config(cfg), cfg


def _tiny_vae():
    vae = VAE(
        in_channels=3,
        latent_channels=4,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
    )
    freeze_vae(vae)
    return vae


def test_ddim_step_deterministic_eta_zero() -> None:
    scheduler = NoiseScheduler(num_timesteps=10)
    x_t = torch.randn(4, 4, 16, 16)
    t = torch.tensor([5, 5, 5, 5], dtype=torch.long)
    eps = torch.randn(4, 4, 16, 16)

    # Calling twice with eta=0 gives byte-exact identical output
    step1 = ddim_step(scheduler, x_t, t, eps, eta=0.0)
    step2 = ddim_step(scheduler, x_t, t, eps, eta=0.0)
    assert torch.equal(step1, step2)


def test_ddim_step_at_t_zero_equals_predict_x0() -> None:
    scheduler = NoiseScheduler(num_timesteps=10)
    x_t = torch.randn(3, 4, 16, 16)
    t = torch.zeros(3, dtype=torch.long)
    eps = torch.randn(3, 4, 16, 16)

    # At t=0, alpha_prev=1.0, so x_{t-1} must equal predict_x0_from_eps
    step0 = ddim_step(scheduler, x_t, t, eps, eta=0.0)
    x0_hat = predict_x0_from_eps(scheduler, x_t, t, eps)
    assert torch.allclose(step0, x0_hat, atol=1e-6)


def test_ddim_step_shape_mismatch_raises() -> None:
    scheduler = NoiseScheduler(num_timesteps=10)
    x_t = torch.randn(2, 4, 16, 16)
    t = torch.tensor([3, 3])
    bad_eps = torch.randn(2, 8, 16, 16)
    with pytest.raises(ValueError, match="must match"):
        ddim_step(scheduler, x_t, t, bad_eps)


def test_sample_conditional_latents_ddim() -> None:
    model, cfg = _tiny_sr_model()
    z_lr = torch.randn(2, 4, 16, 16)
    # Test DDIM sampler with eta=0.0
    z_ddim = sample_conditional_latents(
        model,
        z_lr,
        val_indices=[0, 1],
        noise_seed=123,
        show_progress=False,
        sampler="ddim",
        ddim_eta=0.0,
    )
    assert z_ddim.shape == (2, 4, 16, 16)
    assert torch.isfinite(z_ddim).all()

    # Repeat with same seed and check determinism
    z_ddim_2 = sample_conditional_latents(
        model,
        z_lr,
        val_indices=[0, 1],
        noise_seed=123,
        show_progress=False,
        sampler="ddim",
        ddim_eta=0.0,
    )
    assert torch.allclose(z_ddim, z_ddim_2, atol=1e-5)


def test_shuffle_condition_derangement() -> None:
    z_lr = torch.randn(5, 4, 8, 8)
    shuf = shuffle_condition(z_lr, shift=1)
    assert shuf.shape == z_lr.shape

    # For each sample i, it must receive condition from (i - 1) % 5
    for i in range(5):
        expected_src = (i - 1) % 5
        assert torch.equal(shuf[i], z_lr[expected_src])
        assert not torch.equal(shuf[i], z_lr[i])


def test_shuffle_condition_batch_size_one_raises() -> None:
    z_lr = torch.randn(1, 4, 8, 8)
    with pytest.raises(ValueError, match="Batch size must be >= 2"):
        shuffle_condition(z_lr)


def test_compute_step_condition_utility() -> None:
    scheduler = NoiseScheduler(num_timesteps=10)
    x_t = torch.randn(4, 4, 16, 16)
    t = torch.tensor([4, 4, 4, 4], dtype=torch.long)
    eps_true = torch.randn(4, 4, 16, 16)
    eps_shuf = torch.randn(4, 4, 16, 16)
    z_lr_true = torch.randn(4, 4, 16, 16)
    z_lr_shuf = torch.randn(4, 4, 16, 16)
    z_hr = torch.randn(4, 4, 16, 16)

    metrics = compute_step_condition_utility(
        scheduler,
        x_t,
        t,
        eps_true,
        eps_shuf,
        z_lr_true,
        z_lr_shuf,
        z_hr=z_hr,
    )
    assert metrics["eps_mse"].shape == (4,)
    assert metrics["z0_mse"].shape == (4,)
    assert (metrics["eps_mse"] >= 0.0).all()
    assert (metrics["z0_mse"] >= 0.0).all()
    assert (metrics["z0_cos"] >= -1.0001).all() and (metrics["z0_cos"] <= 1.0001).all()
    assert "specificity_gap" in metrics
    assert "z0_condition_advantage" in metrics


def test_condition_utility_pipeline_reverse_and_forward(tmp_path: Path) -> None:
    model, cfg = _tiny_sr_model()
    vae = _tiny_vae()
    device = torch.device("cpu")

    # Synthetic loader: 4 pairs (lr: 32x32, hr: 64x64)
    lr = torch.rand(4, 3, 32, 32)
    hr = torch.rand(4, 3, 64, 64)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

    # 1. Reverse Mode
    rev_res = evaluate_condition_utility_reverse(
        model,
        vae,
        loader,
        device=device,
        num_images=4,
        hr_size=64,
        latent_scale=1.0,
        noise_seed=42,
        sampler="ddpm",
        milestones=(4, 2, 0),
        eval_all_timesteps=True,
        show_progress=False,
    )
    assert rev_res["num_images"] == 4
    assert len(rev_res["milestone_summary"]) == 3
    assert "eps_mse" in rev_res["curves"]

    saved_rev = save_condition_utility_results(
        rev_res, tmp_path / "cond_util_rev", model_name="TinyTest"
    )
    assert saved_rev["json"].is_file()
    assert saved_rev["milestones_csv"].is_file()
    assert saved_rev["report"].is_file()
    assert saved_rev["plot"].is_file()

    # 2. Forward Mode
    fwd_res = evaluate_condition_utility_forward(
        model,
        vae,
        loader,
        device=device,
        num_images=4,
        hr_size=64,
        latent_scale=1.0,
        noise_seed=42,
        milestones=(4, 2, 0),
        eval_all_timesteps=True,
        show_progress=False,
    )
    assert fwd_res["num_images"] == 4
    assert "eps_mse" in fwd_res["curves"]

    saved_fwd = save_condition_utility_results(
        fwd_res, tmp_path / "cond_util_fwd", model_name="TinyTest Fwd"
    )
    assert saved_fwd["json"].is_file()
    assert saved_fwd["plot"].is_file()


def test_ddpm_vs_ddim_pipeline(tmp_path: Path) -> None:
    model, cfg = _tiny_sr_model()
    vae = _tiny_vae()
    device = torch.device("cpu")

    lr = torch.rand(4, 3, 32, 32)
    hr = torch.rand(4, 3, 64, 64)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

    comp_res = run_ddpm_vs_ddim_comparison(
        model,
        vae,
        loader,
        device=device,
        num_images=4,
        hr_size=64,
        latent_scale=1.0,
        noise_seed=42,
        compute_lpips=False,
        show_progress=False,
        grid_images=2,
    )

    assert comp_res["num_images"] == 4
    assert "ddpm" in comp_res["summary"]
    assert "ddim" in comp_res["summary"]
    assert "delta_collapse" in comp_res["deltas"]
    assert "fork_outcome" in comp_res
    assert "fork_verdict" in comp_res

    table_text = format_ddpm_vs_ddim_table(comp_res)
    assert "DDPM (eta=1)" in table_text
    assert "DDIM (eta=0)" in table_text
    assert "Collapse Score" in table_text

    saved = save_ddpm_vs_ddim_results(
        comp_res, tmp_path / "ddpm_vs_ddim", model_name="TinyTest", hr_size=64
    )
    assert saved["json"].is_file()
    assert saved["per_image_csv"].is_file()
    assert saved["report"].is_file()
    assert saved["trajectories_plot"].is_file()
    assert saved["compare_grid"].is_file()
