"""Paired permutation / bootstrap / Spearman helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentsr.metrics.image_metrics import write_per_image_csv
from latentsr.metrics.paired_stats import (
    bootstrap_mean_ci,
    compare_per_image,
    format_comparison_table,
    load_per_image_csv,
    sign_flip_permutation_pvalue,
    spearman_rho,
    write_comparison_outputs,
)


def test_permutation_pvalue_zero_diffs() -> None:
    diffs = np.zeros(16)
    p_value = sign_flip_permutation_pvalue(diffs, n_perm=199, seed=0)
    assert p_value == pytest.approx(1.0)


def test_permutation_pvalue_constant_shift() -> None:
    diffs = np.ones(32)
    p_value = sign_flip_permutation_pvalue(diffs, n_perm=199, seed=0)
    assert p_value == pytest.approx(1.0 / 200.0)


def test_bootstrap_ci_constant() -> None:
    diffs = np.full(20, 0.25)
    lo, hi = bootstrap_mean_ci(diffs, n_boot=200, seed=0)
    assert lo == pytest.approx(0.25)
    assert hi == pytest.approx(0.25)


def test_spearman_perfect() -> None:
    x = np.arange(10, dtype=np.float64)
    assert spearman_rho(x, x) == pytest.approx(1.0)
    assert spearman_rho(x, -x) == pytest.approx(-1.0)


def _synthetic_rows(offset: float) -> list[dict]:
    rows = []
    for i in range(8):
        rows.append(
            {
                "val_index": i,
                "filename": f"{i:06d}.jpg",
                "bicubic_psnr": 26.0 - 0.1 * i,
                "soft_decode_psnr": 26.0 + offset + 0.2 * i,
                "soft_decode_lpips": 0.20 - 0.01 * offset,
                "latentsr_psnr": 26.2 + 0.5 * offset,
                "latentsr_lpips": 0.07 - 0.001 * offset,
                "latentsr_ssim": 0.83 + 0.01 * offset,
                "latentsr_edge_mae": 0.12 - 0.01 * offset,
                "z_lr_rmse": 0.80 - 0.2 * offset,
                "z_sr_rmse": 0.40,
                "hr_edge_energy": 0.05 + 0.01 * i,
            }
        )
    return rows


def test_compare_per_image_and_outputs(tmp_path: Path) -> None:
    baseline_path = tmp_path / "base.csv"
    candidate_path = tmp_path / "cand.csv"
    write_per_image_csv(baseline_path, _synthetic_rows(0.0))
    write_per_image_csv(candidate_path, _synthetic_rows(1.0))

    result = compare_per_image(
        load_per_image_csv(baseline_path),
        load_per_image_csv(candidate_path),
        n_perm=199,
        n_boot=199,
        seed=0,
    )
    assert result["n"] == 8
    psnr = result["metrics"]["psnr"]
    assert psnr["mean_delta"] == pytest.approx(0.5)
    assert psnr["ci_excludes_zero"]
    assert psnr["p_permutation"] < 0.05
    assert "delta_psnr_soft_vs_sr" in result["correlations"]

    out = tmp_path / "compare"
    paths = write_comparison_outputs(result, out)
    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()
    table = format_comparison_table(result)
    assert "psnr" in table
    assert (out / "delta_psnr_soft_vs_sr.png").is_file()
    assert (out / "z_lr_rmse_vs_z_sr_rmse.png").is_file()


def test_compare_rejects_index_mismatch(tmp_path: Path) -> None:
    left = _synthetic_rows(0.0)
    right = _synthetic_rows(1.0)[1:]
    write_per_image_csv(tmp_path / "a.csv", left)
    write_per_image_csv(tmp_path / "b.csv", right)
    with pytest.raises(ValueError, match="val_index mismatch"):
        compare_per_image(
            load_per_image_csv(tmp_path / "a.csv"),
            load_per_image_csv(tmp_path / "b.csv"),
            n_perm=19,
            n_boot=19,
        )
