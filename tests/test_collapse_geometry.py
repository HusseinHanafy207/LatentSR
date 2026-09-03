"""Collapse score + local geometry correlation tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from latentsr.metrics.collapse_geometry import (
    collapse_from_cosine_curve,
    correlate_collapse_with_geometry,
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
    # t=0 weak, mid peak, t=end mid — C = peak − cos0
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
    # Put query points into the reference cloud so leave-one-out applies.
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
        "local_erank": 1.0 - collapse,  # perfect negative
        "local_kappa": collapse.clone(),
        "nn_dist": collapse.clone(),
        "mean_knn_dist": collapse.clone(),
        "density": 1.0 - collapse,
    }
    corr = correlate_collapse_with_geometry(collapse, geom)
    assert corr["local_erank"]["pearson_r"] < -0.99
    assert corr["density"]["pearson_r"] < -0.99
    assert corr["local_kappa"]["pearson_r"] > 0.99


def test_run_collapse_geometry_end_to_end(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)

    lr = torch.rand(6, 3, 32, 32)
    hr = torch.rand(6, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)

    result = run_collapse_geometry(
        {"tiny": model},
        {"tiny": vae},
        loader,
        device=torch.device("cpu"),
        num_images=5,
        reference_images=5,
        knn=3,
        hr_size=128,
        noise_seed=42,
        output_dir=tmp_path,
        show_progress=False,
    )
    assert result["num_images"] == 5
    assert "tiny" in result["models"]
    assert len(result["per_image"]) == 5
    row = result["per_image"][0]
    assert "tiny_collapse" in row
    assert "tiny_local_erank" in row
    assert (tmp_path / "collapse_geometry_per_image.csv").is_file()
    assert (tmp_path / "collapse_geometry.json").is_file()
    assert "correlations" in result["models"]["tiny"]
