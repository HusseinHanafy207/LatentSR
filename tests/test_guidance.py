"""Inference-time DPS guidance (Stage 1)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from latentsr.super_resolution.condition import build_conditioned_latent_ddpm_from_config
from latentsr.super_resolution.guidance import (
    BASELINE_WINDOW,
    EARLY_WINDOW,
    LATE_WINDOW,
    cache_soft_decodes,
    dps_guidance_step,
    iter_indexed_batches,
    sample_guided_latents,
)
from latentsr.super_resolution.sample import sample_conditional_latents
from latentsr.vae import VAE, freeze_vae
from latentsr.vae.latent import decode_scaled


def _tiny_concat_config() -> dict:
    return {
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


def test_decode_scaled_grads_flow_through_frozen_decoder() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    z = torch.randn(1, 4, 32, 32, requires_grad=True)
    img = decode_scaled(vae, z, allow_grad=True)
    img.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert all(p.grad is None for p in vae.parameters())


def test_windows() -> None:
    assert not BASELINE_WINDOW.active(100)
    assert EARLY_WINDOW.active(800)
    assert EARLY_WINDOW.active(0)
    assert not EARLY_WINDOW.active(801)
    assert LATE_WINDOW.active(500)
    assert not LATE_WINDOW.active(501)
    assert LATE_WINDOW.active(0)


def test_iter_indexed_batches_skips_eval_set() -> None:
    lr = torch.arange(8).float().view(8, 1, 1, 1).expand(8, 3, 32, 32)
    hr = torch.arange(8).float().view(8, 1, 1, 1).expand(8, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=3)
    batches = list(iter_indexed_batches(loader, start_index=5, num_images=2))
    indices = [i for _lr, _hr, idx in batches for i in idx]
    assert indices == [5, 6]


def test_unguided_matches_standard_sampler() -> None:
    torch.manual_seed(0)
    model = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    z_lr = torch.randn(2, 4, 32, 32)
    d_zlr = torch.rand(2, 3, 128, 128)
    a = sample_conditional_latents(
        model, z_lr, val_indices=[10, 11], noise_seed=42, show_progress=False
    )
    for window in (BASELINE_WINDOW, EARLY_WINDOW):
        b = sample_guided_latents(
            model,
            vae,
            z_lr,
            d_zlr,
            lambda_g=0.0,
            window=window,
            val_indices=[10, 11],
            noise_seed=42,
            show_progress=False,
        )
        assert torch.allclose(a, b, atol=1e-5), window


def test_guidance_gradient_nonzero_and_apply_false_keeps_state() -> None:
    torch.manual_seed(1)
    model = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    z_t = torch.randn(1, 4, 32, 32)
    z_lr = torch.randn(1, 4, 32, 32)
    d_zlr = torch.rand(1, 3, 128, 128)
    t = torch.tensor([3])
    z_next, extras = dps_guidance_step(
        model,
        vae,
        z_t,
        t,
        z_lr,
        d_zlr,
        latent_scale=1.0,
        lambda_g=0.1,
        active=True,
        step_noise=torch.zeros_like(z_t),
        need_diagnostics=True,
        apply_correction=False,
    )
    z_unguided, _ = dps_guidance_step(
        model,
        vae,
        z_t,
        t,
        z_lr,
        d_zlr,
        latent_scale=1.0,
        lambda_g=0.0,
        active=False,
        step_noise=torch.zeros_like(z_t),
        need_diagnostics=False,
        apply_correction=True,
    )
    assert torch.allclose(z_next, z_unguided, atol=1e-5)
    assert float(extras["grad_norm"][0]) > 0.0
    assert torch.isfinite(extras["loss"]).all()


def test_cache_soft_decode_shapes() -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    lr = torch.rand(2, 3, 32, 32)
    z_lr, d_zlr = cache_soft_decodes(vae, lr, hr_size=128)
    assert z_lr.shape == (2, 4, 32, 32)
    assert d_zlr.shape == (2, 3, 128, 128)
    assert float(d_zlr.min()) >= 0.0 and float(d_zlr.max()) <= 1.0


def test_correction_equals_lambda_times_grad() -> None:
    torch.manual_seed(2)
    model = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    z_t = torch.randn(1, 4, 32, 32)
    z_lr = torch.randn(1, 4, 32, 32)
    d_zlr = torch.rand(1, 3, 128, 128)
    t = torch.tensor([3])
    noise = torch.zeros_like(z_t)
    z_off, extras = dps_guidance_step(
        model,
        vae,
        z_t,
        t,
        z_lr,
        d_zlr,
        latent_scale=1.0,
        lambda_g=0.25,
        active=True,
        step_noise=noise,
        need_diagnostics=True,
        apply_correction=False,
    )
    z_on, extras_on = dps_guidance_step(
        model,
        vae,
        z_t,
        t,
        z_lr,
        d_zlr,
        latent_scale=1.0,
        lambda_g=0.25,
        active=True,
        step_noise=noise,
        need_diagnostics=True,
        apply_correction=True,
    )
    delta = z_off - z_on
    expected = 0.25 * float(extras["grad_norm"][0])
    assert expected > 0.0
    assert abs(float(delta.flatten(1).norm()) - expected) < 1e-5
    assert torch.allclose(extras["grad_norm"], extras_on["grad_norm"], atol=1e-6)


def test_guidance_checkpoint_roundtrip(tmp_path) -> None:
    from latentsr.metrics.guidance_eval import (
        accumulate_scores_from_rows,
        load_guidance_checkpoint,
        soft_decode_psnr_mean,
    )
    from latentsr.metrics.image_metrics import write_per_image_csv

    rows = [
        {
            "val_index": 0,
            "filename": "a.jpg",
            "condition": "late",
            "lambda_g": 200.0,
            "soft_decode_psnr": 28.48,
            "latentsr_psnr": 27.6,
            "latentsr_lpips": 0.069,
        },
        {
            "val_index": 3,
            "filename": "b.jpg",
            "condition": "late",
            "lambda_g": 200.0,
            "soft_decode_psnr": 28.50,
            "latentsr_psnr": 27.7,
            "latentsr_lpips": 0.070,
        },
    ]
    write_per_image_csv(tmp_path / "per_image.csv", rows)
    write_per_image_csv(
        tmp_path / "trajectory.csv",
        [
            {"val_index": 0, "t": 500, "R": 0.04},
            {"val_index": 9, "t": 500, "R": 0.05},
        ],
    )
    loaded, traj, done = load_guidance_checkpoint(
        tmp_path, start_index=0, num_images=4
    )
    assert done == {0, 3}
    assert [int(r["val_index"]) for r in loaded] == [0, 3]
    assert len(traj) == 1
    assert int(traj[0]["val_index"]) == 0
    scores: dict = {"soft_decode": {"psnr": []}, "latentsr": {"psnr": [], "lpips": []}}
    accumulate_scores_from_rows(loaded, scores)
    assert scores["soft_decode"]["psnr"] == [28.48, 28.50]
    mean_soft = soft_decode_psnr_mean(loaded)
    assert mean_soft is not None
    assert abs(mean_soft - 28.49) < 1e-9


def test_recommend_lambda_prefers_in_band() -> None:
    from latentsr.metrics.guidance_eval import recommend_lambda_g

    rows = [
        {
            "R_lambda_0.001": 0.002,
            "R_lambda_0.01": 0.02,
            "R_lambda_0.1": 0.25,
            "R_lambda_1": 2.5,
        }
    ]
    rec = recommend_lambda_g(rows)
    assert rec["lambda_g"] == 0.1
    assert 0.1 in rec["in_band"]
