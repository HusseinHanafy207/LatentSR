"""Paired permutation tests and bootstrap CIs"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_METRICS = ("psnr", "lpips")
SECONDARY_METRICS = ("ssim", "edge_mae")
LOWER_IS_BETTER = {"lpips", "edge_mae", "freq_low", "freq_mid", "freq_high"}


def load_per_image_csv(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load ``per_image.csv`` keyed by ``val_index``."""
    path = Path(path)
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        for raw in reader:
            if "val_index" not in raw or raw["val_index"] == "":
                raise ValueError(f"Missing val_index in {path}")
            index = int(raw["val_index"])
            parsed: dict[str, Any] = {"val_index": index}
            for key, value in raw.items():
                if key == "val_index":
                    continue
                if value is None or value == "":
                    parsed[key] = None
                    continue
                if key in {"filename", "condition"}:
                    parsed[key] = value
                    continue
                parsed[key] = float(value)
            rows[index] = parsed
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def aligned_values(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    column: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return paired arrays for a column, intersecting on ``val_index``."""
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError("No shared val_index between the two CSVs.")
    missing_b = [i for i in shared if baseline[i].get(column) is None]
    missing_c = [i for i in shared if candidate[i].get(column) is None]
    if missing_b or missing_c:
        raise ValueError(
            f"Column {column!r} missing for val_index "
            f"{(missing_b or missing_c)[:8]}"
        )
    left = np.array([float(baseline[i][column]) for i in shared], dtype=np.float64)
    right = np.array([float(candidate[i][column]) for i in shared], dtype=np.float64)
    return left, right, shared


def bootstrap_mean_ci(
    diffs: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    diffs = np.asarray(diffs, dtype=np.float64).reshape(-1)
    n = int(diffs.size)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(n_boot, n))
    means = diffs[index].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def sign_flip_permutation_pvalue(
    diffs: np.ndarray,
    *,
    n_perm: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation p-value (sign flips of Δ, Phipson–Smyth)."""
    diffs = np.asarray(diffs, dtype=np.float64).reshape(-1)
    n = int(diffs.size)
    if n == 0:
        return float("nan")
    observed = float(np.abs(diffs.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_abs = np.abs((signs * diffs).mean(axis=1))
    count = int(np.sum(perm_abs >= observed - 1e-15))
    return (count + 1) / (n_perm + 1)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """1-based ranks with average ties."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    sorter = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[sorter[j + 1]] == x[sorter[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[sorter[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ties)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _metric_block(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
    lower_is_better: bool,
) -> dict[str, Any]:
    diffs = candidate - baseline
    lo, hi = bootstrap_mean_ci(diffs, n_boot=n_boot, alpha=0.05, seed=seed)
    p_value = sign_flip_permutation_pvalue(diffs, n_perm=n_perm, seed=seed)
    mean_delta = float(diffs.mean())
    return {
        "n": int(diffs.size),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_delta": mean_delta,
        "ci95_low": lo,
        "ci95_high": hi,
        "p_permutation": p_value,
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "lower_is_better": lower_is_better,
        "candidate_better_mean": bool(
            mean_delta < 0.0 if lower_is_better else mean_delta > 0.0
        ),
    }


def compare_per_image(
    baseline_rows: dict[int, dict[str, Any]],
    candidate_rows: dict[int, dict[str, Any]],
    *,
    baseline_name: str = "vae1",
    candidate_name: str = "vae_sr",
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired stats + pre-registered correlations (Δ = candidate − baseline)."""
    shared = sorted(set(baseline_rows) & set(candidate_rows))
    only_b = sorted(set(baseline_rows) - set(candidate_rows))
    only_c = sorted(set(candidate_rows) - set(baseline_rows))
    if only_b or only_c:
        raise ValueError(
            "val_index mismatch: "
            f"only in baseline={only_b[:8]} only in candidate={only_c[:8]}"
        )

    metric_results: dict[str, dict[str, Any]] = {}
    all_metrics = list(PRIMARY_METRICS) + list(SECONDARY_METRICS)
    for metric in all_metrics:
        column = f"latentsr_{metric}"
        left, right, _ = aligned_values(baseline_rows, candidate_rows, column)
        metric_results[metric] = _metric_block(
            left,
            right,
            n_perm=n_perm,
            n_boot=n_boot,
            seed=seed,
            lower_is_better=metric in LOWER_IS_BETTER,
        )
        metric_results[metric]["role"] = (
            "primary" if metric in PRIMARY_METRICS else "secondary"
        )

    left_soft, right_soft, indices = aligned_values(
        baseline_rows, candidate_rows, "soft_decode_psnr"
    )
    left_sr, right_sr, _ = aligned_values(
        baseline_rows, candidate_rows, "latentsr_psnr"
    )
    delta_soft_psnr = right_soft - left_soft
    delta_sr_psnr = right_sr - left_sr

    left_soft_lpips, right_soft_lpips, _ = aligned_values(
        baseline_rows, candidate_rows, "soft_decode_lpips"
    )
    left_sr_lpips, right_sr_lpips, _ = aligned_values(
        baseline_rows, candidate_rows, "latentsr_lpips"
    )
    delta_soft_lpips = right_soft_lpips - left_soft_lpips
    delta_sr_lpips = right_sr_lpips - left_sr_lpips

    bicubic, _, _ = aligned_values(
        baseline_rows, candidate_rows, "bicubic_psnr"
    )
    hr_edge, _, _ = aligned_values(
        baseline_rows, candidate_rows, "hr_edge_energy"
    )

    correlations = {
        "delta_psnr_soft_vs_sr": spearman_rho(delta_soft_psnr, delta_sr_psnr),
        "delta_lpips_soft_vs_sr": spearman_rho(delta_soft_lpips, delta_sr_lpips),
        "delta_psnr_sr_vs_bicubic": spearman_rho(bicubic, delta_sr_psnr),
        "delta_psnr_sr_vs_hr_edge": spearman_rho(hr_edge, delta_sr_psnr),
    }

    per_image = []
    for i, val_index in enumerate(indices):
        per_image.append(
            {
                "val_index": int(val_index),
                "filename": baseline_rows[val_index].get("filename")
                or candidate_rows[val_index].get("filename")
                or "",
                "delta_psnr_soft": float(delta_soft_psnr[i]),
                "delta_psnr_sr": float(delta_sr_psnr[i]),
                "delta_lpips_soft": float(delta_soft_lpips[i]),
                "delta_lpips_sr": float(delta_sr_lpips[i]),
                "bicubic_psnr": float(bicubic[i]),
                "hr_edge_energy": float(hr_edge[i]),
                f"{baseline_name}_z_lr_rmse": float(
                    baseline_rows[val_index]["z_lr_rmse"]
                ),
                f"{candidate_name}_z_lr_rmse": float(
                    candidate_rows[val_index]["z_lr_rmse"]
                ),
                f"{baseline_name}_z_sr_rmse": float(
                    baseline_rows[val_index]["z_sr_rmse"]
                ),
                f"{candidate_name}_z_sr_rmse": float(
                    candidate_rows[val_index]["z_sr_rmse"]
                ),
            }
        )

    return {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "n": len(indices),
        "delta_definition": "candidate - baseline",
        "n_perm": n_perm,
        "n_boot": n_boot,
        "seed": seed,
        "metrics": metric_results,
        "correlations": correlations,
        "per_image": per_image,
        "series": {
            "delta_psnr_soft": delta_soft_psnr.tolist(),
            "delta_psnr_sr": delta_sr_psnr.tolist(),
            "delta_lpips_soft": delta_soft_lpips.tolist(),
            "delta_lpips_sr": delta_sr_lpips.tolist(),
            "bicubic_psnr": bicubic.tolist(),
            "hr_edge_energy": hr_edge.tolist(),
            f"{baseline_name}_z_lr_rmse": [
                float(baseline_rows[i]["z_lr_rmse"]) for i in indices
            ],
            f"{candidate_name}_z_lr_rmse": [
                float(candidate_rows[i]["z_lr_rmse"]) for i in indices
            ],
            f"{baseline_name}_z_sr_rmse": [
                float(baseline_rows[i]["z_sr_rmse"]) for i in indices
            ],
            f"{candidate_name}_z_sr_rmse": [
                float(candidate_rows[i]["z_sr_rmse"]) for i in indices
            ],
        },
    }


def format_comparison_table(result: dict[str, Any]) -> str:
    """Human-readable paired-test table."""
    lines = [
        f"n={result['n']}  Δ = {result['candidate_name']} − {result['baseline_name']}",
        f"{'metric':<12} {'role':<10} {'Δ mean':>10} {'95% CI':>22} "
        f"{'p_perm':>10} {'CI≠0':>6}",
        "-" * 78,
    ]
    for metric, stats in result["metrics"].items():
        ci = f"[{stats['ci95_low']:+.4f}, {stats['ci95_high']:+.4f}]"
        lines.append(
            f"{metric:<12} {stats['role']:<10} {stats['mean_delta']:+10.4f} "
            f"{ci:>22} {stats['p_permutation']:10.4g} "
            f"{str(stats['ci_excludes_zero']):>6}"
        )
    lines.append("")
    lines.append("Spearman ρ (pre-registered):")
    for name, rho in result["correlations"].items():
        lines.append(f"  {name}: {rho:.4f}")
    return "\n".join(lines)


def write_scatter_plots(result: dict[str, Any], output_dir: Path) -> list[Path]:
    """Pre-registered diagnostic scatters (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    series = result["series"]
    baseline = result["baseline_name"]
    candidate = result["candidate_name"]
    written: list[Path] = []

    def _scatter(
        x: list[float],
        y: list[float],
        *,
        xlabel: str,
        ylabel: str,
        title: str,
        filename: str,
        identity: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.scatter(x, y, s=22, alpha=0.8, c="#1f4e79", edgecolors="none")
        ax.axhline(0.0, color="#888888", lw=0.8)
        ax.axvline(0.0, color="#888888", lw=0.8)
        if identity and x and y:
            lo = min(min(x), min(y))
            hi = max(max(x), max(y))
            ax.plot([lo, hi], [lo, hi], color="#c44e52", lw=1.0, ls="--", label="y = x")
            ax.legend(frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    _scatter(
        series["delta_psnr_soft"],
        series["delta_psnr_sr"],
        xlabel="Δ PSNR soft-decode (dB)",
        ylabel="Δ PSNR LatentSR (dB)",
        title="Does a larger VAE gain transfer to SR?",
        filename="delta_psnr_soft_vs_sr.png",
    )
    _scatter(
        series["delta_lpips_soft"],
        series["delta_lpips_sr"],
        xlabel="Δ LPIPS soft-decode",
        ylabel="Δ LPIPS LatentSR",
        title="Does a larger VAE LPIPS gain transfer to SR?",
        filename="delta_lpips_soft_vs_sr.png",
    )
    _scatter(
        series["bicubic_psnr"],
        series["delta_psnr_sr"],
        xlabel="Bicubic PSNR (dB)",
        ylabel="Δ PSNR LatentSR (dB)",
        title="SR gain vs baseline difficulty",
        filename="delta_psnr_sr_vs_bicubic.png",
    )
    _scatter(
        series["hr_edge_energy"],
        series["delta_psnr_sr"],
        xlabel="HR Sobel energy",
        ylabel="Δ PSNR LatentSR (dB)",
        title="SR gain vs HR edge energy",
        filename="delta_psnr_sr_vs_hr_edge.png",
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(
        series[f"{baseline}_z_lr_rmse"],
        series[f"{baseline}_z_sr_rmse"],
        s=22,
        alpha=0.8,
        c="#4c78a8",
        label=baseline,
        edgecolors="none",
    )
    ax.scatter(
        series[f"{candidate}_z_lr_rmse"],
        series[f"{candidate}_z_sr_rmse"],
        s=22,
        alpha=0.8,
        c="#f58518",
        label=candidate,
        edgecolors="none",
    )
    all_z = (
        series[f"{baseline}_z_lr_rmse"]
        + series[f"{baseline}_z_sr_rmse"]
        + series[f"{candidate}_z_lr_rmse"]
        + series[f"{candidate}_z_sr_rmse"]
    )
    lo, hi = min(all_z), max(all_z)
    ax.plot([lo, hi], [lo, hi], color="#c44e52", lw=1.0, ls="--", label="y = x")
    ax.set_xlabel(r"RMSE($z_{lr}$, $z_{hr}$)")
    ax.set_ylabel(r"RMSE($z_{sr}$, $z_{hr}$)")
    ax.set_title("LR latent error vs DDPM latent error")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "z_lr_rmse_vs_z_sr_rmse.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axes[0].hist(series["delta_psnr_sr"], bins=12, color="#1f4e79", alpha=0.85)
    axes[0].axvline(0.0, color="#888888", lw=0.8)
    axes[0].set_title("Δ PSNR LatentSR")
    axes[0].set_xlabel("dB")
    axes[1].hist(series["delta_lpips_sr"], bins=12, color="#c44e52", alpha=0.85)
    axes[1].axvline(0.0, color="#888888", lw=0.8)
    axes[1].set_title("Δ LPIPS LatentSR")
    axes[1].set_xlabel("LPIPS")
    fig.tight_layout()
    hist_path = output_dir / "delta_hist.png"
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)
    written.append(hist_path)

    return written


def write_comparison_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, CSV, text table, and scatter PNGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = format_comparison_table(result)
    (output_dir / "paired_stats.txt").write_text(table + "\n", encoding="utf-8")

    serializable = {k: v for k, v in result.items() if k != "series"}
    (output_dir / "paired_stats.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )

    per_image = result["per_image"]
    if per_image:
        fieldnames = list(per_image[0].keys())
        with (output_dir / "paired_deltas.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image)

    plots = write_scatter_plots(result, output_dir)
    return {
        "table": str(output_dir / "paired_stats.txt"),
        "json": str(output_dir / "paired_stats.json"),
        "csv": str(output_dir / "paired_deltas.csv"),
        "plots": [str(p) for p in plots],
    }
