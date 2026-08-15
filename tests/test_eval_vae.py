"""Phase A: VAE bottleneck eval (no diffusion)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from latentsr.metrics.evaluate_vae import evaluate_vae, format_bottleneck_note
from latentsr.metrics.image_metrics import edge_mae, frequency_band_error
from latentsr.vae import VAE, freeze_vae


def test_edge_mae_identical_is_zero() -> None:
    x = torch.rand(2, 3, 32, 32)
    assert float(edge_mae(x, x, reduction="mean").item()) < 1e-6


def test_edge_mae_blur_is_positive() -> None:
    x = torch.rand(2, 3, 32, 32)
    blur = F.avg_pool2d(x, kernel_size=4, stride=1, padding=2)
    blur = F.interpolate(blur, size=32, mode="bilinear", align_corners=False)
    assert float(edge_mae(blur, x, reduction="mean").item()) > 1e-4


def test_frequency_bands_identical_near_zero() -> None:
    x = torch.rand(2, 3, 32, 32)
    bands = frequency_band_error(x, x, reduction="mean")
    assert set(bands) == {"freq_low", "freq_mid", "freq_high"}
    for value in bands.values():
        assert float(value.item()) < 1e-4


def test_frequency_high_band_detects_blur() -> None:
    coords = torch.arange(32)
    checker = ((coords[:, None] + coords[None, :]) % 2).float()
    x = checker.expand(2, 3, 32, 32).contiguous()
    blur = F.interpolate(
        F.avg_pool2d(x, kernel_size=4, stride=4),
        size=32,
        mode="nearest",
    )
    bands = frequency_band_error(blur, x, reduction="mean")
    assert float(bands["freq_high"].item()) > float(bands["freq_low"].item())


def test_bottleneck_note_mentions_gate() -> None:
    summary = {
        "vae_hr": {"psnr": {"mean": 33.0, "std": 1.0, "n": 8}},
        "soft_decode": {"psnr": {"mean": 26.2, "std": 1.0, "n": 8}},
        "bicubic": {"psnr": {"mean": 26.1, "std": 1.0, "n": 8}},
    }
    note = format_bottleneck_note(summary, latentsr_psnr=26.25)
    assert "NOT the bottleneck" in note
    assert "Q2" in note
    assert "decode(z_lr) ≈ bicubic" in note


def test_evaluate_vae_smoke(tmp_path: Path) -> None:
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    lr = torch.rand(4, 3, 32, 32)
    hr = torch.rand(4, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

    result = evaluate_vae(
        vae,
        loader,
        device=torch.device("cpu"),
        num_images=3,
        hr_size=128,
        latent_scale=1.0,
        compute_lpips=False,
        show_progress=False,
        grid_images=2,
        output_dir=tmp_path,
        latentsr_psnr=26.25,
    )
    assert result["num_images"] == 3
    assert set(result["summary"]) == {"bicubic", "soft_decode", "vae_hr"}
    for method in result["summary"].values():
        assert "psnr" in method
        assert "ssim" in method
        assert "edge_mae" in method
        assert "freq_high" in method
        assert "lpips" not in method
    stats = result["latent_stats"]
    assert stats["latent_scale_suggested"] > 0
    assert stats["z_mse"]["n"] == 3
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "eval_vae_compare.png").is_file()
    assert "Gate:" in result["bottleneck_note"]
