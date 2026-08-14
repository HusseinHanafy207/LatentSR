"""Phase 10: PSNR / SSIM / LPIPS helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from latentsr.metrics.evaluate_sr import evaluate_sr
from latentsr.metrics.image_metrics import (
    batch_metrics,
    format_metric_table,
    psnr,
    ssim,
    summarize_values,
)
from latentsr.super_resolution.condition import build_conditioned_latent_ddpm_from_config
from latentsr.vae import VAE, freeze_vae


def test_psnr_identical_is_high() -> None:
    x = torch.rand(2, 3, 32, 32)
    assert psnr(x, x, reduction="mean").item() > 40.0


def test_ssim_identical_near_one() -> None:
    x = torch.rand(2, 3, 32, 32)
    assert ssim(x, x, reduction="mean").item() > 0.99


def test_batch_metrics_shapes() -> None:
    pred = torch.rand(3, 3, 64, 64)
    gt = torch.rand(3, 3, 64, 64)
    metrics = batch_metrics(pred, gt, lpips_fn=None)
    assert metrics["psnr"].shape == (3,)
    assert metrics["ssim"].shape == (3,)
    assert "lpips" not in metrics


def test_summarize_and_table() -> None:
    stats = summarize_values([1.0, 2.0, 3.0])
    assert stats["n"] == 3
    assert abs(stats["mean"] - 2.0) < 1e-6
    table = format_metric_table(
        {
            "bicubic": {"psnr": {"mean": 20.0, "std": 1.0, "n": 3}},
            "latentsr": {"psnr": {"mean": 22.0, "std": 1.5, "n": 3}},
        }
    )
    assert "bicubic" in table and "latentsr" in table


def test_evaluate_sr_smoke(tmp_path: Path) -> None:
    config = {
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
    model = build_conditioned_latent_ddpm_from_config(config)
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)

    lr = torch.rand(4, 3, 32, 32)
    hr = torch.rand(4, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

    result = evaluate_sr(
        model,
        vae,
        loader,
        device=torch.device("cpu"),
        num_images=3,
        hr_size=128,
        latent_scale=1.0,
        compute_lpips=False,
        include_soft_decode=True,
        show_progress=False,
        grid_images=2,
        output_dir=tmp_path,
    )
    assert result["num_images"] == 3
    assert "bicubic" in result["summary"]
    assert "latentsr" in result["summary"]
    assert "soft_decode" in result["summary"]
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "eval_compare.png").is_file()
