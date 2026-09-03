"""Unit tests for RiT-style representation geometry metrics."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from latentsr.metrics.representation_geometry import (
    compare_vae_geometries,
    covariance_condition_number,
    cumulative_variance,
    effective_rank,
    excess_kurtosis_stats,
    flatten_latents,
    format_geometry_table,
    pca_eigenvalues,
    run_representation_geometry,
    transport_condition_numbers,
    twonn_bootstrap,
    twonn_intrinsic_dim,
    write_geometry_report,
)
from latentsr.vae import VAE, freeze_vae


def test_flatten_latents_shape() -> None:
    z = torch.randn(5, 4, 8, 8)
    flat = flatten_latents(z)
    assert flat.shape == (5, 256)
    assert flat.dtype == torch.float64


def test_isotropic_gaussian_high_effective_rank() -> None:
    torch.manual_seed(0)
    # Near-isotropic cloud in 32D.
    x = torch.randn(400, 32, dtype=torch.float64)
    eigs = pca_eigenvalues(x)
    erank = effective_rank(eigs)
    # Full rank would be 32; isotropic noise should be high.
    assert erank > 20.0
    kappa = covariance_condition_number(eigs, num_samples=x.shape[0])["kappa"]
    assert kappa < 5.0


def test_rank1_cloud_low_effective_rank() -> None:
    torch.manual_seed(1)
    direction = torch.randn(16, dtype=torch.float64)
    coeffs = torch.randn(200, 1, dtype=torch.float64)
    x = coeffs * direction
    eigs = pca_eigenvalues(x)
    assert effective_rank(eigs) < 2.5
    kappa = covariance_condition_number(
        eigs, num_samples=x.shape[0], relative_floor=1e-12
    )["kappa"]
    assert kappa > 1e3


def test_cumulative_variance_ends_at_one() -> None:
    torch.manual_seed(2)
    x = torch.randn(100, 20, dtype=torch.float64)
    cum = cumulative_variance(pca_eigenvalues(x))
    assert abs(float(cum[-1].item()) - 1.0) < 1e-6


def test_transport_kappa_at_t0_is_one() -> None:
    eigs = torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64)
    rows = transport_condition_numbers(eigs, t_values=(0.0, 1.0))
    assert abs(rows[0]["t"] - 0.0) < 1e-12
    assert abs(rows[0]["kappa"] - 1.0) < 1e-6
    assert rows[1]["kappa"] == covariance_condition_number(
        eigs, num_samples=eigs.numel() + 1
    )["kappa"]


def test_gaussian_low_excess_kurtosis() -> None:
    torch.manual_seed(3)
    x = torch.randn(2000, 64, dtype=torch.float64)
    stats = excess_kurtosis_stats(x)
    assert stats["median_abs_kurtosis"] < 0.3
    assert stats["frac_abs_lt_0_5"] > 0.8


def test_twonn_bootstrap_full_sample_runs_once() -> None:
    torch.manual_seed(7)
    x = torch.randn(30, 8, dtype=torch.float64)
    stats = twonn_bootstrap(x, num_bootstraps=10, subsample_size=30, seed=0)
    assert stats["twonn_n_boot"] == 1.0
    assert stats["twonn_std"] == 0.0
    assert stats["twonn_mean"] == stats["twonn_full"]


def test_twonn_recovers_low_dim_manifold() -> None:
    torch.manual_seed(4)
    # 2D linear subspace embedded in 32D.
    basis = torch.linalg.qr(torch.randn(32, 2, dtype=torch.float64)).Q
    coeffs = torch.randn(800, 2, dtype=torch.float64)
    x = coeffs @ basis.T
    d_hat = twonn_intrinsic_dim(x)
    assert 1.3 < d_hat < 3.2


def test_compare_and_write_report(tmp_path: Path) -> None:
    torch.manual_seed(5)
    baseline = {
        "z_hr": torch.randn(40, 4, 4, 4),
        "z_lr": torch.randn(40, 4, 4, 4) * 0.5,
    }
    candidate = {
        "z_hr": torch.randn(40, 4, 4, 4),
        "z_lr": torch.randn(40, 4, 4, 4),
    }
    report = compare_vae_geometries(
        baseline,
        candidate,
        twonn_bootstraps=3,
        twonn_subsample=30,
        seed=0,
    )
    assert set(report["spaces"]) == {
        "vae1_hr",
        "vae1_lr",
        "vae_sr_hr",
        "vae_sr_lr",
    }
    table = format_geometry_table(report)
    assert "TwoNN" in table or "twonn" in table.lower() or "vae1_hr" in table
    path = write_geometry_report(report, tmp_path)
    assert path.is_file()
    assert (tmp_path / "pca_cumulative_variance.png").is_file()
    assert (tmp_path / "summary.txt").is_file()


def test_run_representation_geometry_tiny_vae(tmp_path: Path) -> None:
    torch.manual_seed(6)
    vae_a = VAE(base_channels=32, channel_mult=(1, 2, 4), num_res_blocks=1)
    vae_b = VAE(base_channels=32, channel_mult=(1, 2, 4), num_res_blocks=1)
    # Make candidate different so stacks are not identical by accident.
    with torch.no_grad():
        for p in vae_b.parameters():
            p.add_(0.01)
    freeze_vae(vae_a)
    freeze_vae(vae_b)

    lr = torch.rand(6, 3, 32, 32)
    hr = torch.rand(6, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=3)

    report = run_representation_geometry(
        vae_a,
        vae_b,
        loader,
        device=torch.device("cpu"),
        num_images=6,
        output_dir=tmp_path,
        hr_size=128,
        twonn_bootstraps=2,
        twonn_subsample=5,
        seed=0,
        show_progress=False,
    )
    assert report["spaces"]["vae1_hr"]["num_samples"] == 6
    assert (tmp_path / "metrics.json").is_file()
