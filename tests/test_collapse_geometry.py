"""Collapse score + local geometry correlation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from latentsr.metrics.collapse_geometry import (
    bootstrap_correlation_ci,
    collapse_from_cosine_curve,
    correlate_collapse_with_geometry,
    enrich_correlations_from_rows,
    pearson_r,
    run_collapse_geometry,
)
from latentsr.metrics.local_geometry import knn_indices, local_geometry_at_points
from latentsr.super_resolution.condition import build_conditioned_latent_ddpm_from_config
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


def test_collapse_from_cosine_curve_per_image_peak() -> None:
    cos = torch.tensor(
        [
            [0.5, 0.9, 0.7, 0.6],
            [0.8, 0.85, 0.95, 0.7],
        ],
        dtype=torch.float64,
    )
    stats = collapse_from_cosine_curve(cos)
    assert list(stats["t_peak"].tolist()) == [1, 2]
    assert torch.allclose(stats["collapse"], torch.tensor([0.4, 0.15], dtype=torch.float64))


def test_collapse_fixed_peak() -> None:
    cos = torch.tensor([[0.5, 0.9, 0.7], [0.4, 0.6, 0.8]], dtype=torch.float64)
    stats = collapse_from_cosine_curve(cos, fixed_t_peak=1)
    assert list(stats["t_peak"].tolist()) == [1, 1]
    assert torch.allclose(stats["collapse"], torch.tensor([0.4, 0.2], dtype=torch.float64))


def test_pearson_perfect() -> None:
    x = torch.arange(10.0).numpy()
    assert abs(pearson_r(x, 2 * x + 1) - 1.0) < 1e-9


def test_bootstrap_correlation_ci_stable_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    y = 0.8 * x + 0.2 * rng.normal(size=64)
    stats = bootstrap_correlation_ci(x, y, n_boot=200, seed=0)
    assert stats["pearson_r"] > 0.5
    assert stats["pearson_lo"] > 0.0  # CI should not cross 0 for this signal
    assert stats["pearson_hi"] >= stats["pearson_lo"]
    assert stats["pearson_frac_sign_flip"] < 0.05


def test_knn_excludes_self() -> None:
    torch.manual_seed(0)
    x = torch.randn(20, 8, dtype=torch.float64)
    idx, dists = knn_indices(x, x, k=5, exclude_self=True)
    for i in range(20):
        assert i not in idx[i].tolist()
        assert float(dists[i, 0].item()) > 0.0


def test_local_geometry_shapes() -> None:
    torch.manual_seed(1)
    q = torch.randn(12, 4, 2, 2)
    r = torch.randn(40, 4, 2, 2)
    r[:12] = q
    stats = local_geometry_at_points(q, r, k=8, exclude_self=True)
    assert stats["local_erank"].shape == (12,)
    assert stats["local_kappa"].shape == (12,)
    assert stats["density"].shape == (12,)
    assert torch.all(stats["nn_dist"] > 0)
    assert torch.all(stats["local_erank"] > 0)


def test_correlate_collapse_with_geometry() -> None:
    collapse = torch.linspace(0.0, 1.0, 30)
    geom = {
        "local_erank": 1.0 - collapse,
        "local_kappa": collapse.clone(),
        "nn_dist": collapse.clone(),
        "mean_knn_dist": collapse.clone(),
        "density": 1.0 - collapse,
    }
    corr = correlate_collapse_with_geometry(collapse, geom, n_boot=50, seed=0)
    assert corr["local_erank"]["pearson_r"] < -0.99
    assert corr["density"]["pearson_r"] < -0.99
    assert "pearson_lo" in corr["density"]
    assert corr["local_kappa"]["pearson_r"] > 0.99


def test_enrich_from_rows_bootstrap_only() -> None:
    rows = []
    for i in range(40):
        dens = float(i)
        rows.append(
            {
                "val_index": i,
                "vae1_collapse": dens * 0.05,
                "vae1_density": dens * 0.5,
                "vae1_mean_knn_dist": 2.0 / (dens + 1.0),
                "vae1_local_erank": 10.0,
                "vae1_local_kappa": 1.0,
                "vae1_nn_dist": 0.5,
                "vae1_psnr": 26.0,
                "vae1_lpips": 0.2,
                "vae_sr_collapse": dens * 0.1,
                "vae_sr_density": dens,
                "vae_sr_mean_knn_dist": 1.0 / (dens + 1.0),
                "vae_sr_local_erank": 10.0,
                "vae_sr_local_kappa": 1.0,
                "vae_sr_nn_dist": 0.5,
                "vae_sr_psnr": 30.0 - 0.05 * dens,
                "vae_sr_lpips": 0.1 + 0.001 * dens,
            }
        )
    result = enrich_correlations_from_rows(rows, n_boot=100, seed=1)
    dens_c = result["models"]["vae_sr"]["correlations"]["density"]
    assert dens_c["pearson_r"] > 0.9
    assert "pearson_lo" in dens_c
    q = result["models"]["vae_sr"]["quality_correlations"]
    assert "psnr" in q and "lpips" in q
    assert q["psnr"]["density"]["pearson_r"] < 0.0
    assert "paired_delta" in result
    assert "psnr" in result["paired_delta"]["correlations"]
    assert "delta_psnr" in result["per_image"][0]


def test_paired_delta_analysis_unit() -> None:
    from latentsr.metrics.collapse_geometry import paired_delta_analysis

    rows = [
        {
            "val_index": i,
            "vae1_psnr": 20.0,
            "vae_sr_psnr": 20.0 + 0.1 * i,
            "vae1_lpips": 0.3,
            "vae_sr_lpips": 0.3 - 0.001 * i,
            "vae1_density": 1.0,
            "vae_sr_density": 1.0 + 0.05 * i,
            "vae1_mean_knn_dist": 1.0,
            "vae_sr_mean_knn_dist": 1.0 - 0.01 * i,
        }
        for i in range(30)
    ]
    out = paired_delta_analysis(
        rows, baseline_name="vae1", candidate_name="vae_sr", n_boot=50, seed=0
    )
    assert out is not None
    assert out["correlations"]["psnr"]["density"]["pearson_r"] > 0.9
    assert "delta_psnr" in rows[0]


def test_run_collapse_geometry_end_to_end(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    clone = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    clone.load_state_dict(model.state_dict())
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)

    lr = torch.rand(8, 3, 32, 32)
    hr = torch.rand(8, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

    result = run_collapse_geometry(
        {"vae1": model, "vae_sr": clone},
        {"vae1": vae, "vae_sr": vae},
        loader,
        device=torch.device("cpu"),
        num_images=6,
        reference_images=6,
        knn=3,
        hr_size=128,
        noise_seed=42,
        n_boot=20,
        compute_image_metrics=True,
        compute_lpips=False,
        baseline_name="vae1",
        candidate_name="vae_sr",
        knn_grid=[2, 3],
        reference_grid=[6],
        output_dir=tmp_path,
        show_progress=False,
    )
    assert result["num_images"] == 6
    assert "vae_sr" in result["models"]
    assert "vae1" in result["models"]
    row = result["per_image"][0]
    assert "vae_sr_collapse" in row
    assert "vae_sr_psnr" in row
    assert "vae1_psnr" in row
    assert "quality_correlations" in result["models"]["vae_sr"]
    assert "paired_delta" in result
    assert "delta_psnr" in row
    assert result["models"]["vae_sr"].get("robustness")
    dens = result["models"]["vae_sr"]["correlations"]["density"]
    assert "pearson_lo" in dens and "pearson_hi" in dens
    assert (tmp_path / "collapse_geometry_per_image.csv").is_file()
    assert (tmp_path / "delta_geometry_vs_delta_quality.png").is_file()
