"""PSNR / SSIM / LPIPS helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
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
from latentsr.super_resolution.sample import seeded_noise_like
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
    assert metrics["edge_mae"].shape == (3,)
    assert metrics["freq_low"].shape == (3,)
    assert metrics["freq_mid"].shape == (3,)
    assert metrics["freq_high"].shape == (3,)
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
    per_image_path = tmp_path / "per_image.csv"
    assert per_image_path.is_file()
    rows = result["per_image"]
    assert [row["val_index"] for row in rows] == [0, 1, 2]
    for row in rows:
        assert "latentsr_psnr" in row
        assert "soft_decode_psnr" in row
        assert "bicubic_psnr" in row
        assert "z_lr_rmse" in row
        assert "z_sr_rmse" in row
        assert "hr_edge_energy" in row
    assert "soft_decode" in result["summary"]


def _tiny_sr():
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
    return build_conditioned_latent_ddpm_from_config(config)


def test_seeded_noise_independent_of_batching() -> None:
    z = torch.zeros(4, 4, 8, 8)
    all_noise = seeded_noise_like(z, [0, 1, 2, 3], base_seed=42)
    first = seeded_noise_like(z[:2], [0, 1], base_seed=42)
    second = seeded_noise_like(z[2:], [2, 3], base_seed=42)
    assert torch.equal(all_noise[:2], first)
    assert torch.equal(all_noise[2:], second)
    other = seeded_noise_like(z[:1], [0], base_seed=7)
    assert not torch.equal(all_noise[:1], other)
    salted = seeded_noise_like(z[:1], [0], base_seed=42, salt=10_007)
    assert not torch.equal(all_noise[:1], salted)


def test_evaluate_sr_noise_independent_of_batch_size(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _tiny_sr()
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    lr = torch.rand(3, 3, 32, 32)
    hr = torch.rand(3, 3, 128, 128)

    def _run(batch_size: int, seed: int, folder: str) -> list[float]:
        loader = DataLoader(TensorDataset(lr, hr), batch_size=batch_size)
        result = evaluate_sr(
            model,
            vae,
            loader,
            device=torch.device("cpu"),
            num_images=3,
            hr_size=128,
            compute_lpips=False,
            show_progress=False,
            grid_images=1,
            output_dir=tmp_path / folder,
            noise_seed=seed,
        )
        return [row["latentsr_psnr"] for row in result["per_image"]]

    batched = _run(3, 42, "bs3")
    sequential = _run(1, 42, "bs1")
    other_seed = _run(3, 99, "seed99")
    assert batched == pytest.approx(sequential, abs=1e-5)
    assert batched != pytest.approx(other_seed, abs=1e-5)
