"""Phase 1.5: correlate late reverse-chain collapse with local z_lr geometry.

For each image:

    C_i = cos(ẑ0(t_peak), z_lr) − cos(ẑ0(0), z_lr)

Higher C_i ⇒ stronger late-stage collapse of the mid-chain copy.

Local geometry around each z_lr (leave-one-out k-NN in a reference cloud):

    local effective rank, local κ, neighborhood density, NN distance.

Also scores final LatentSR decode vs HR (PSNR / LPIPS) on the **same**
reverse samples, and bootstraps correlation CIs (image-level resample).

Exploratory / correlational — not causal.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from latentsr.metrics.image_metrics import LPIPSMetric, batch_metrics
from latentsr.metrics.local_geometry import local_geometry_at_points
from latentsr.metrics.paired_stats import spearman_rho
from latentsr.metrics.timestep_diagnostic import latent_cosine
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.super_resolution.sample import (
    _STEP_SALT,
    predict_x0_from_eps,
    seeded_noise_like,
)
from latentsr.vae.latent import decode_scaled
from latentsr.vae.vae import VAE


_GEOM_KEYS = (
    "local_erank",
    "local_kappa",
    "nn_dist",
    "mean_knn_dist",
    "density",
)
_QUALITY_KEYS = ("psnr", "lpips")
# density ≈ 1/mean_knn_dist — treat as one neighborhood signal, not two discoveries.
_HEADLINE_GEOM = ("density", "mean_knn_dist")
_DELTA_GEOM = ("density", "mean_knn_dist")
_DELTA_QUALITY = ("psnr", "lpips")


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size < 3 or ya.size != xa.size:
        return float("nan")
    if float(xa.std()) < 1e-12 or float(ya.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def bootstrap_correlation_ci(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Image-level percentile bootstrap CI for Pearson r and Spearman ρ.

    Resamples paired indices with replacement ``n_boot`` times (default 1000).
    Also reports the fraction of bootstrap draws whose Pearson r has the
    opposite sign from the point estimate (cheap stability check).
    """
    xa = np.asarray(
        x.detach().cpu().numpy() if torch.is_tensor(x) else x,
        dtype=np.float64,
    ).reshape(-1)
    ya = np.asarray(
        y.detach().cpu().numpy() if torch.is_tensor(y) else y,
        dtype=np.float64,
    ).reshape(-1)
    if xa.size != ya.size or xa.size < 3:
        return {
            "pearson_r": float("nan"),
            "pearson_lo": float("nan"),
            "pearson_hi": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_lo": float("nan"),
            "spearman_hi": float("nan"),
            "n": float(xa.size),
            "n_boot": float(max(int(n_boot), 0)),
            "pearson_frac_sign_flip": float("nan"),
        }

    point_p = pearson_r(xa, ya)
    point_s = float(spearman_rho(xa, ya))
    n = int(xa.size)
    n_boot = max(int(n_boot), 0)
    if n_boot == 0:
        return {
            "pearson_r": point_p,
            "pearson_lo": float("nan"),
            "pearson_hi": float("nan"),
            "spearman_rho": point_s,
            "spearman_lo": float("nan"),
            "spearman_hi": float("nan"),
            "n": float(n),
            "n_boot": 0.0,
            "pearson_frac_sign_flip": float("nan"),
        }

    rng = np.random.default_rng(int(seed))
    boots_p = np.empty(n_boot, dtype=np.float64)
    boots_s = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots_p[b] = pearson_r(xa[idx], ya[idx])
        boots_s[b] = float(spearman_rho(xa[idx], ya[idx]))

    def _ci(vals: np.ndarray) -> tuple[float, float]:
        ok = vals[np.isfinite(vals)]
        if ok.size == 0:
            return float("nan"), float("nan")
        lo, hi = np.quantile(ok, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lo), float(hi)

    plo, phi = _ci(boots_p)
    slo, shi = _ci(boots_s)
    finite_p = boots_p[np.isfinite(boots_p)]
    if finite_p.size == 0 or not np.isfinite(point_p) or point_p == 0.0:
        frac_flip = float("nan")
    else:
        frac_flip = float(np.mean(np.sign(finite_p) != np.sign(point_p)))

    return {
        "pearson_r": point_p,
        "pearson_lo": plo,
        "pearson_hi": phi,
        "spearman_rho": point_s,
        "spearman_lo": slo,
        "spearman_hi": shi,
        "n": float(n),
        "n_boot": float(n_boot),
        "pearson_frac_sign_flip": frac_flip,
    }


def collapse_from_cosine_curve(
    cos_curve: torch.Tensor,
    *,
    fixed_t_peak: int | None = None,
) -> dict[str, torch.Tensor]:
    """``cos_curve`` is ``(N, T)`` with columns indexed by diffusion t.

    ``t_peak`` is per-image ``argmax_t cos`` unless ``fixed_t_peak`` is set.
    """
    if cos_curve.ndim != 2:
        raise ValueError(f"Expected (N, T), got {tuple(cos_curve.shape)}")
    n, num_t = cos_curve.shape
    cos0 = cos_curve[:, 0]
    if fixed_t_peak is None:
        t_peak = cos_curve.argmax(dim=1)
    else:
        tp = int(fixed_t_peak)
        if not (0 <= tp < num_t):
            raise ValueError(f"fixed_t_peak={tp} out of range for T={num_t}")
        t_peak = torch.full((n,), tp, dtype=torch.long)
    arange = torch.arange(n)
    cos_peak = cos_curve[arange, t_peak]
    collapse = cos_peak - cos0
    return {
        "t_peak": t_peak,
        "cos_peak": cos_peak,
        "cos_t0": cos0,
        "collapse": collapse,
    }


@torch.no_grad()
def collect_z_lr_bank(
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    start_index: int = 0,
    show_progress: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    """Encode ``z_lr`` for ``num_images`` val pairs; return ``(N,C,H,W), indices``."""
    vae.eval()
    chunks: list[torch.Tensor] = []
    indices: list[int] = []
    remaining = max(int(num_images), 1)
    next_index = int(start_index)
    pbar = (
        tqdm(total=remaining, desc="encode z_lr bank", unit="img", leave=False)
        if show_progress
        else None
    )
    for lr, _hr in loader:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        z = encode_lr_latents(vae, lr, hr_size=hr_size, latent_scale=latent_scale)
        chunks.append(z.cpu())
        indices.extend(range(next_index, next_index + take))
        remaining -= take
        next_index += take
        if pbar is not None:
            pbar.update(take)
    if pbar is not None:
        pbar.close()
    if not chunks:
        raise ValueError("No z_lr collected; check num_images / loader.")
    return torch.cat(chunks, dim=0), indices


@torch.no_grad()
def reverse_cosine_curves(
    model: ConditionalLatentDDPM,
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    noise_seed: int = 42,
    start_index: int = 0,
    show_progress: bool = True,
    compute_image_metrics: bool = True,
    compute_lpips: bool = True,
    lpips_fn: LPIPSMetric | None = None,
) -> dict[str, Any]:
    """Run the reverse chain; return cos curves, z_lr, and optional PSNR/LPIPS.

    Final decode uses the same seeded reverse trajectory as the cosine log
    (``x`` after ``t=0``), matching ``sample_conditional_latents``.
    """
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)
    num_t = int(model.num_timesteps)
    if compute_image_metrics and compute_lpips and lpips_fn is None:
        lpips_fn = LPIPSMetric(net="alex", device=device)

    cos_chunks: list[torch.Tensor] = []
    z_chunks: list[torch.Tensor] = []
    psnr_chunks: list[torch.Tensor] = []
    lpips_chunks: list[torch.Tensor] = []
    indices: list[int] = []
    remaining = max(int(num_images), 1)
    next_index = int(start_index)
    pbar = (
        tqdm(total=remaining, desc="reverse cos curves", unit="img", leave=True)
        if show_progress
        else None
    )

    for lr, hr in loader:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        hr_b = hr[:take].to(device)
        batch_idx = list(range(next_index, next_index + take))
        z_lr = encode_lr_latents(
            vae, lr, hr_size=hr_size, latent_scale=latent_scale
        )
        x = seeded_noise_like(z_lr, batch_idx, base_seed=noise_seed, salt=0)
        cos_hist = torch.empty(take, num_t, dtype=torch.float32)

        for t in range(num_t - 1, -1, -1):
            t_batch = torch.full((take,), t, device=device, dtype=torch.long)
            eps = model.predict_noise(x, t_batch, z_lr)
            z0 = predict_x0_from_eps(model.scheduler, x, t_batch, eps)
            cos_hist[:, t] = latent_cosine(z0, z_lr).detach().cpu()
            step_noise = seeded_noise_like(
                x,
                batch_idx,
                base_seed=noise_seed,
                salt=_STEP_SALT * (int(t) + 1),
            )
            x = model.scheduler.p_sample_step(x, t_batch, eps, noise=step_noise)

        cos_chunks.append(cos_hist)
        z_chunks.append(z_lr.cpu())
        if compute_image_metrics:
            pred = decode_scaled(vae, x, latent_scale=latent_scale).clamp(0.0, 1.0)
            metrics = batch_metrics(
                pred,
                hr_b,
                lpips_fn=lpips_fn if compute_lpips else None,
            )
            psnr_chunks.append(metrics["psnr"].detach().cpu().float())
            if compute_lpips and "lpips" in metrics:
                lpips_chunks.append(metrics["lpips"].detach().cpu().float())
        indices.extend(batch_idx)
        remaining -= take
        next_index += take
        if pbar is not None:
            pbar.update(take)

    if pbar is not None:
        pbar.close()
    if not cos_chunks:
        raise ValueError("No reverse curves collected.")

    out: dict[str, Any] = {
        "cos": torch.cat(cos_chunks, dim=0),
        "indices": indices,
        "z_lr": torch.cat(z_chunks, dim=0),
    }
    if psnr_chunks:
        out["psnr"] = torch.cat(psnr_chunks, dim=0)
    if lpips_chunks:
        out["lpips"] = torch.cat(lpips_chunks, dim=0)
    return out


def correlate_vectors(
    target: np.ndarray | torch.Tensor,
    features: dict[str, np.ndarray | torch.Tensor],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap Pearson/Spearman of ``target`` vs each named feature."""
    out: dict[str, dict[str, float]] = {}
    for key, feat in features.items():
        out[key] = bootstrap_correlation_ci(
            target, feat, n_boot=n_boot, seed=seed
        )
    return out


def correlate_collapse_with_geometry(
    collapse: torch.Tensor | np.ndarray,
    geometry: dict[str, torch.Tensor],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Pearson + Spearman of C_i vs each local geometry scalar (+ bootstrap CI)."""
    feats = {k: geometry[k] for k in _GEOM_KEYS if k in geometry}
    return correlate_vectors(collapse, feats, n_boot=n_boot, seed=seed)


def paired_delta_analysis(
    per_image: list[dict[str, Any]],
    *,
    baseline_name: str,
    candidate_name: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Correlate Δgeometry with ΔPSNR / ΔLPIPS (candidate − baseline).

    Asks: when VAE-SR changes an image's representation geometry, does that
    change predict whether the image actually benefits in SR quality?
    Stronger than absolute density↔PSNR (which confounds easy faces).
    """
    if not per_image:
        return None
    b, c = baseline_name, candidate_name
    need = [f"{b}_psnr", f"{c}_psnr", f"{b}_density", f"{c}_density"]
    if any(k not in per_image[0] for k in need):
        return None

    deltas_q: dict[str, np.ndarray] = {}
    for qkey in _DELTA_QUALITY:
        bcol, ccol = f"{b}_{qkey}", f"{c}_{qkey}"
        if bcol not in per_image[0] or ccol not in per_image[0]:
            continue
        bv = np.array([r[bcol] for r in per_image], dtype=np.float64)
        cv = np.array([r[ccol] for r in per_image], dtype=np.float64)
        deltas_q[qkey] = cv - bv

    deltas_g: dict[str, np.ndarray] = {}
    for gkey in _DELTA_GEOM:
        bcol, ccol = f"{b}_{gkey}", f"{c}_{gkey}"
        if bcol not in per_image[0] or ccol not in per_image[0]:
            continue
        bv = np.array([r[bcol] for r in per_image], dtype=np.float64)
        cv = np.array([r[ccol] for r in per_image], dtype=np.float64)
        deltas_g[gkey] = cv - bv

    if not deltas_q or not deltas_g:
        return None

    # Write delta columns onto rows for CSV / plots.
    for i, row in enumerate(per_image):
        for qkey, arr in deltas_q.items():
            row[f"delta_{qkey}"] = float(arr[i])
        for gkey, arr in deltas_g.items():
            row[f"delta_{gkey}"] = float(arr[i])

    correlations: dict[str, dict[str, dict[str, float]]] = {}
    for qkey, qdelta in deltas_q.items():
        correlations[qkey] = correlate_vectors(
            qdelta, deltas_g, n_boot=n_boot, seed=seed
        )

    summary = {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "n": len(per_image),
        "delta_means": {
            **{f"delta_{k}": float(v.mean()) for k, v in deltas_q.items()},
            **{f"delta_{k}": float(v.mean()) for k, v in deltas_g.items()},
        },
        "correlations": correlations,
        "note": (
            "Δ = candidate − baseline on the same val_index. "
            "density and mean_knn_dist are essentially inverse neighborhood "
            "signals — not independent discoveries."
        ),
    }
    return summary


def neighborhood_robustness_sweep(
    *,
    z_eval: torch.Tensor,
    z_ref: torch.Tensor,
    collapse: torch.Tensor | np.ndarray,
    quality: dict[str, torch.Tensor | np.ndarray],
    knn_grid: list[int],
    reference_grid: list[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Recompute headline corrs for a small (k, reference_n) grid (no reverse)."""
    rows_out: list[dict[str, Any]] = []
    ref_max = int(z_ref.shape[0])
    for ref_n in reference_grid:
        use_n = min(int(ref_n), ref_max)
        if use_n < 3:
            continue
        z_sub = z_ref[:use_n]
        for k in knn_grid:
            k_use = int(k)
            if k_use < 2 or k_use >= use_n:
                continue
            geom = local_geometry_at_points(
                z_eval, z_sub, k=k_use, exclude_self=True
            )
            headline = {g: geom[g] for g in _HEADLINE_GEOM}
            collapse_corr = correlate_vectors(
                collapse, headline, n_boot=n_boot, seed=seed
            )
            entry: dict[str, Any] = {
                "knn": k_use,
                "reference_n": use_n,
                "requested_reference_n": int(ref_n),
                "collapse_correlations": collapse_corr,
            }
            qcorr: dict[str, dict[str, dict[str, float]]] = {}
            for qkey, qvals in quality.items():
                qcorr[qkey] = correlate_vectors(
                    qvals, headline, n_boot=n_boot, seed=seed + 1
                )
            if qcorr:
                entry["quality_correlations"] = qcorr
            rows_out.append(entry)
    return rows_out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return "nan"
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(float(obj))
    return obj


def _fmt_corr(block: dict[str, float]) -> str:
    r = block.get("pearson_r", float("nan"))
    lo = block.get("pearson_lo", float("nan"))
    hi = block.get("pearson_hi", float("nan"))
    rho = block.get("spearman_rho", float("nan"))
    flip = block.get("pearson_frac_sign_flip", float("nan"))
    return (
        f"r={r:+.3f} [{lo:+.3f},{hi:+.3f}]  "
        f"ρ={rho:+.3f}  signflip={flip:.2f}"
    )


def write_collapse_geometry_report(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = result["per_image"]
    csv_path = output_dir / "collapse_geometry_per_image.csv"
    if rows:
        fields = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    json_path = output_dir / "collapse_geometry.json"
    payload = {k: v for k, v in result.items() if k != "paths"}
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8"
    )

    table = format_collapse_geometry_table(result)
    txt_path = output_dir / "collapse_geometry.txt"
    txt_path.write_text(table + "\n", encoding="utf-8")

    plots = plot_collapse_geometry(result, output_dir)
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "table": str(txt_path),
        "plots": [str(p) for p in plots],
    }


def format_collapse_geometry_table(result: dict[str, Any]) -> str:
    lines = [
        "Collapse ↔ local z_lr geometry (exploratory)",
        f"n={result['num_images']}  knn={result.get('knn', '—')}  "
        f"reference_n={result.get('reference_n', '—')}  "
        f"n_boot={result.get('n_boot', '—')}",
        "",
        "Note: density ≈ 1/mean_knn_dist — one neighborhood signal, not two.",
        "",
    ]
    for model_name, block in result["models"].items():
        lines.append(f"[{model_name}]")
        if "collapse_mean" in block:
            lines.append(
                f"  mean C_i (per-image peak) = {block['collapse_mean']:.4f} "
                f"± {block['collapse_std']:.4f}"
            )
        if "t_peak_mean" in block:
            lines.append(
                f"  mean t_peak = {block['t_peak_mean']:.1f}  "
                f"global mean-curve peak t = {block['global_t_peak']}"
            )
        if "psnr_mean" in block:
            lines.append(
                f"  LatentSR PSNR = {block['psnr_mean']:.3f}±{block['psnr_std']:.3f}"
                + (
                    f"  LPIPS = {block['lpips_mean']:.4f}±{block['lpips_std']:.4f}"
                    if "lpips_mean" in block
                    else ""
                )
            )
        lines.append("  C_i ↔ neighborhood (bootstrap CI):")
        for key in _HEADLINE_GEOM:
            if key in block.get("correlations", {}):
                lines.append(f"    {key:<14}  {_fmt_corr(block['correlations'][key])}")
        qcorr = block.get("quality_correlations") or {}
        if qcorr:
            lines.append("  absolute geometry ↔ absolute quality:")
            for qkey in _QUALITY_KEYS:
                if qkey not in qcorr:
                    continue
                for gkey in _HEADLINE_GEOM:
                    if gkey in qcorr[qkey]:
                        lines.append(
                            f"    {gkey:<14} ↔ {qkey:<5}  {_fmt_corr(qcorr[qkey][gkey])}"
                        )
        rob = block.get("robustness") or []
        if rob:
            lines.append("  robustness (C_i ↔ density) over (k, ref_n):")
            for entry in rob:
                dens = entry["collapse_correlations"].get("density", {})
                lines.append(
                    f"    k={entry['knn']:<3} ref={entry['reference_n']:<5}  "
                    f"{_fmt_corr(dens)}"
                )
        lines.append("")

    delta = result.get("paired_delta")
    if delta:
        lines.append(
            f"[paired Δ = {delta['candidate_name']} − {delta['baseline_name']}]"
        )
        means = delta.get("delta_means") or {}
        if means:
            bits = "  ".join(f"{k}={v:+.4f}" for k, v in means.items())
            lines.append(f"  mean deltas: {bits}")
        lines.append("  Δgeometry ↔ Δquality (key test):")
        for qkey, geom_map in (delta.get("correlations") or {}).items():
            for gkey in _DELTA_GEOM:
                if gkey in geom_map:
                    lines.append(
                        f"    Δ{gkey:<12} ↔ Δ{qkey:<5}  {_fmt_corr(geom_map[gkey])}"
                    )
        lines.append("")

    lines.append(
        "Decision rule: if neighborhood predicts collapse but not absolute "
        "quality / Δquality, demote this branch. If it predicts both, whitening "
        "becomes more interesting."
    )
    return "\n".join(lines)


def plot_collapse_geometry(
    result: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    rows = result["per_image"]
    written: list[Path] = []
    if not rows:
        return written

    model_names = list(result.get("models") or [])
    if not model_names:
        # Infer from column prefixes.
        for key in rows[0]:
            if key.endswith("_collapse"):
                model_names.append(key[: -len("_collapse")])

    for model_name in model_names:
        prefix = f"{model_name}_"
        c_key = f"{prefix}collapse"
        if c_key not in rows[0]:
            continue
        c = np.array([r[c_key] for r in rows], dtype=np.float64)
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))
        axes_flat = axes.ravel()
        geom_plot = list(_GEOM_KEYS)
        for ax, gkey in zip(axes_flat, geom_plot, strict=False):
            col = f"{prefix}{gkey}"
            if col not in rows[0]:
                ax.set_visible(False)
                continue
            g = np.array([r[col] for r in rows], dtype=np.float64)
            ax.scatter(g, c, s=18, alpha=0.75, edgecolors="none")
            ax.set_xlabel(gkey)
            ax.set_ylabel("C_i (collapse)")
            ax.set_title(f"{model_name}: C vs {gkey}", fontsize=9)
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(geom_plot) :]:
            ax.set_visible(False)
        fig.tight_layout()
        path = output_dir / f"collapse_vs_geometry_{model_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        has_psnr = f"{prefix}psnr" in rows[0]
        has_lpips = f"{prefix}lpips" in rows[0]
        if not (has_psnr or has_lpips):
            continue
        if f"{prefix}density" not in rows[0]:
            continue
        n_plot_rows = int(has_psnr) + int(has_lpips)
        fig, axes = plt.subplots(n_plot_rows, 2, figsize=(8.5, 3.6 * n_plot_rows))
        if n_plot_rows == 1:
            axes = np.asarray([axes])
        dens = np.array([r[f"{prefix}density"] for r in rows], dtype=np.float64)
        mk = np.array(
            [r[f"{prefix}mean_knn_dist"] for r in rows], dtype=np.float64
        )
        row_i = 0
        if has_psnr:
            psnr = np.array([r[f"{prefix}psnr"] for r in rows], dtype=np.float64)
            axes[row_i, 0].scatter(dens, psnr, s=18, alpha=0.75, edgecolors="none")
            axes[row_i, 0].set_xlabel("density")
            axes[row_i, 0].set_ylabel("PSNR")
            axes[row_i, 0].set_title(f"{model_name}: density vs PSNR")
            axes[row_i, 0].grid(True, alpha=0.3)
            axes[row_i, 1].scatter(mk, psnr, s=18, alpha=0.75, edgecolors="none")
            axes[row_i, 1].set_xlabel("mean_knn_dist")
            axes[row_i, 1].set_ylabel("PSNR")
            axes[row_i, 1].set_title(f"{model_name}: mean_knn vs PSNR")
            axes[row_i, 1].grid(True, alpha=0.3)
            row_i += 1
        if has_lpips:
            lpips = np.array([r[f"{prefix}lpips"] for r in rows], dtype=np.float64)
            axes[row_i, 0].scatter(dens, lpips, s=18, alpha=0.75, edgecolors="none")
            axes[row_i, 0].set_xlabel("density")
            axes[row_i, 0].set_ylabel("LPIPS")
            axes[row_i, 0].set_title(f"{model_name}: density vs LPIPS")
            axes[row_i, 0].grid(True, alpha=0.3)
            axes[row_i, 1].scatter(mk, lpips, s=18, alpha=0.75, edgecolors="none")
            axes[row_i, 1].set_xlabel("mean_knn_dist")
            axes[row_i, 1].set_ylabel("LPIPS")
            axes[row_i, 1].set_title(f"{model_name}: mean_knn vs LPIPS")
            axes[row_i, 1].grid(True, alpha=0.3)
        fig.tight_layout()
        qpath = output_dir / f"geometry_vs_quality_{model_name}.png"
        fig.savefig(qpath, dpi=150)
        plt.close(fig)
        written.append(qpath)

    # Paired Δgeometry vs Δquality (candidate − baseline).
    if (
        result.get("paired_delta")
        and "delta_psnr" in rows[0]
        and "delta_density" in rows[0]
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
        dd = np.array([r["delta_density"] for r in rows], dtype=np.float64)
        dm = np.array([r["delta_mean_knn_dist"] for r in rows], dtype=np.float64)
        if "delta_psnr" in rows[0]:
            dp = np.array([r["delta_psnr"] for r in rows], dtype=np.float64)
            axes[0, 0].scatter(dd, dp, s=18, alpha=0.75, edgecolors="none")
            axes[0, 0].set_xlabel("Δ density")
            axes[0, 0].set_ylabel("Δ PSNR")
            axes[0, 0].set_title("Δdensity vs ΔPSNR")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 1].scatter(dm, dp, s=18, alpha=0.75, edgecolors="none")
            axes[0, 1].set_xlabel("Δ mean_knn_dist")
            axes[0, 1].set_ylabel("Δ PSNR")
            axes[0, 1].set_title("Δmean_knn vs ΔPSNR")
            axes[0, 1].grid(True, alpha=0.3)
        if "delta_lpips" in rows[0]:
            dl = np.array([r["delta_lpips"] for r in rows], dtype=np.float64)
            axes[1, 0].scatter(dd, dl, s=18, alpha=0.75, edgecolors="none")
            axes[1, 0].set_xlabel("Δ density")
            axes[1, 0].set_ylabel("Δ LPIPS")
            axes[1, 0].set_title("Δdensity vs ΔLPIPS")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 1].scatter(dm, dl, s=18, alpha=0.75, edgecolors="none")
            axes[1, 1].set_xlabel("Δ mean_knn_dist")
            axes[1, 1].set_ylabel("Δ LPIPS")
            axes[1, 1].set_title("Δmean_knn vs ΔLPIPS")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
        fig.tight_layout()
        dpath = output_dir / "delta_geometry_vs_delta_quality.png"
        fig.savefig(dpath, dpi=150)
        plt.close(fig)
        written.append(dpath)
    return written


def enrich_correlations_from_rows(
    per_image: list[dict[str, Any]],
    model_names: list[str] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    knn: int | None = None,
    reference_n: int | None = None,
    baseline_name: str = "vae1",
    candidate_name: str = "vae_sr",
) -> dict[str, Any]:
    """Recompute bootstrap CIs (+ quality / Δ corrs) from an existing per-image table.

    Cheap path for already-finished runs: no reverse chain required.
    Cannot recompute (k, ref) robustness without cached latents.
    """
    if not per_image:
        raise ValueError("per_image is empty")
    if model_names is None:
        model_names = []
        for key in per_image[0]:
            if key.endswith("_collapse"):
                model_names.append(key[: -len("_collapse")])
    models: dict[str, Any] = {}
    for name in model_names:
        prefix = f"{name}_"
        if f"{prefix}collapse" not in per_image[0]:
            continue
        collapse = np.array(
            [r[f"{prefix}collapse"] for r in per_image], dtype=np.float64
        )
        geom = {
            k: np.array([r[f"{prefix}{k}"] for r in per_image], dtype=np.float64)
            for k in _GEOM_KEYS
            if f"{prefix}{k}" in per_image[0]
        }
        corr = correlate_vectors(collapse, geom, n_boot=n_boot, seed=seed)
        block: dict[str, Any] = {
            "collapse_mean": float(collapse.mean()),
            "collapse_std": float(collapse.std()),
            "correlations": corr,
        }
        qcorr: dict[str, dict[str, dict[str, float]]] = {}
        for qkey in _QUALITY_KEYS:
            col = f"{prefix}{qkey}"
            if col not in per_image[0]:
                continue
            q = np.array([r[col] for r in per_image], dtype=np.float64)
            block[f"{qkey}_mean"] = float(q.mean())
            block[f"{qkey}_std"] = float(q.std())
            headline = {k: geom[k] for k in _HEADLINE_GEOM if k in geom}
            qcorr[qkey] = correlate_vectors(q, headline, n_boot=n_boot, seed=seed)
        if qcorr:
            block["quality_correlations"] = qcorr
        models[name] = block

    paired = paired_delta_analysis(
        per_image,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        n_boot=n_boot,
        seed=seed,
    )
    result: dict[str, Any] = {
        "num_images": len(per_image),
        "knn": knn,
        "reference_n": reference_n,
        "n_boot": int(n_boot),
        "models": models,
        "per_image": per_image,
        "note": (
            "Bootstrap / quality / Δ correlations recomputed from per-image rows. "
            "If PSNR/LPIPS columns are missing, re-run the full diagnostic. "
            "Robustness (k, ref) sweeps require a full re-run with --robustness."
        ),
    }
    if paired is not None:
        result["paired_delta"] = paired
    return result


def load_per_image_collapse_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            parsed: dict[str, Any] = {}
            for k, v in raw.items():
                if v is None or v == "":
                    continue
                if k == "val_index":
                    parsed[k] = int(v)
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            if "val_index" in parsed:
                rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


@torch.no_grad()
def run_collapse_geometry(
    models: dict[str, ConditionalLatentDDPM],
    vaes: dict[str, VAE],
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int,
    reference_images: int | None = None,
    knn: int = 32,
    hr_size: int = 128,
    latent_scales: dict[str, float] | None = None,
    noise_seed: int = 42,
    start_index: int = 0,
    fixed_t_peak: int | None = None,
    n_boot: int = 1000,
    compute_image_metrics: bool = True,
    compute_lpips: bool = True,
    baseline_name: str | None = None,
    candidate_name: str | None = None,
    knn_grid: list[int] | None = None,
    reference_grid: list[int] | None = None,
    output_dir: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Reverse-chain collapse + local z_lr geometry + quality / Δ correlations.

    ``models`` / ``vaes`` share keys (e.g. ``vae1``, ``vae_sr``). Each model is
    paired with its VAE. Reference cloud size defaults to ``num_images``.

    Optional ``knn_grid`` / ``reference_grid`` recompute neighborhood geometry
    only (no extra reverse) for a small robustness sweep.
    """
    if not models:
        raise ValueError("models must be non-empty")
    if set(models) != set(vaes):
        raise ValueError("models and vaes must share the same keys")
    latent_scales = latent_scales or {k: 1.0 for k in models}
    names = list(models.keys())
    if baseline_name is None:
        baseline_name = names[0]
    if candidate_name is None:
        candidate_name = names[-1]

    knn_grid = list(knn_grid or [])
    reference_grid = list(reference_grid or [])
    ref_n = int(reference_images) if reference_images is not None else int(num_images)
    if reference_grid:
        ref_n = max(ref_n, max(int(x) for x in reference_grid))
    if ref_n < num_images:
        raise ValueError(
            f"reference_images ({ref_n}) must be >= num_images ({num_images})"
        )
    if knn < 2:
        raise ValueError(f"knn must be >= 2 for local PCA, got {knn}")

    per_image: list[dict[str, Any]] = [
        {"val_index": start_index + i} for i in range(num_images)
    ]
    model_blocks: dict[str, Any] = {}
    lpips_fn: LPIPSMetric | None = None
    if compute_image_metrics and compute_lpips:
        lpips_fn = LPIPSMetric(net="alex", device=device)

    for name in models:
        model = models[name]
        vae = vaes[name]
        scale = float(latent_scales.get(name, 1.0))
        if show_progress:
            print(f"[{name}] reverse cosine curves (n={num_images})…", flush=True)
        packed = reverse_cosine_curves(
            model,
            vae,
            loader,
            device=device,
            num_images=num_images,
            hr_size=hr_size,
            latent_scale=scale,
            noise_seed=noise_seed,
            start_index=start_index,
            show_progress=show_progress,
            compute_image_metrics=compute_image_metrics,
            compute_lpips=compute_lpips,
            lpips_fn=lpips_fn,
        )
        cos = packed["cos"]
        idx = packed["indices"]
        z_eval = packed["z_lr"]

        if ref_n == num_images:
            z_ref = z_eval
            ref_idx = idx
        else:
            if show_progress:
                print(f"[{name}] encode z_lr reference bank (n={ref_n})…", flush=True)
            z_ref, ref_idx = collect_z_lr_bank(
                vae,
                loader,
                device=device,
                num_images=ref_n,
                hr_size=hr_size,
                latent_scale=scale,
                start_index=start_index,
                show_progress=show_progress,
            )

        mean_curve = cos.mean(dim=0)
        global_t_peak = int(mean_curve.argmax().item())
        peak_mode = fixed_t_peak if fixed_t_peak is not None else None
        stats = collapse_from_cosine_curve(cos, fixed_t_peak=peak_mode)
        stats_fixed = collapse_from_cosine_curve(
            cos, fixed_t_peak=global_t_peak if fixed_t_peak is None else fixed_t_peak
        )

        # Primary geometry uses min(requested primary ref, bank) and knn.
        primary_ref_n = (
            int(reference_images)
            if reference_images is not None
            else int(num_images)
        )
        primary_ref_n = min(primary_ref_n, int(z_ref.shape[0]))
        z_ref_primary = z_ref[:primary_ref_n]

        if show_progress:
            print(
                f"[{name}] local geometry (k={knn}, ref={z_ref_primary.shape[0]})…",
                flush=True,
            )
        geom = local_geometry_at_points(
            z_eval, z_ref_primary, k=knn, exclude_self=True
        )
        corr = correlate_collapse_with_geometry(
            stats["collapse"], geom, n_boot=n_boot, seed=noise_seed
        )
        corr_fixed = correlate_collapse_with_geometry(
            stats_fixed["collapse"], geom, n_boot=n_boot, seed=noise_seed + 1
        )

        quality: dict[str, torch.Tensor] = {}
        if "psnr" in packed:
            quality["psnr"] = packed["psnr"]
        if "lpips" in packed:
            quality["lpips"] = packed["lpips"]

        qcorr: dict[str, dict[str, dict[str, float]]] = {}
        headline_geom = {k: geom[k] for k in _HEADLINE_GEOM if k in geom}
        for qkey, qvals in quality.items():
            qcorr[qkey] = correlate_vectors(
                qvals, headline_geom, n_boot=n_boot, seed=noise_seed + 2
            )

        robustness: list[dict[str, Any]] = []
        if knn_grid and reference_grid:
            if show_progress:
                print(
                    f"[{name}] robustness sweep knn={knn_grid} "
                    f"ref={reference_grid}…",
                    flush=True,
                )
            robustness = neighborhood_robustness_sweep(
                z_eval=z_eval,
                z_ref=z_ref,
                collapse=stats["collapse"],
                quality=quality,
                knn_grid=knn_grid,
                reference_grid=reference_grid,
                n_boot=n_boot,
                seed=noise_seed,
            )

        prefix = f"{name}_"
        for i, val_i in enumerate(idx):
            row = per_image[i] if i < len(per_image) else {"val_index": val_i}
            row["val_index"] = int(val_i)
            row[f"{prefix}t_peak"] = int(stats["t_peak"][i].item())
            row[f"{prefix}cos_peak"] = float(stats["cos_peak"][i].item())
            row[f"{prefix}cos_t0"] = float(stats["cos_t0"][i].item())
            row[f"{prefix}collapse"] = float(stats["collapse"][i].item())
            row[f"{prefix}collapse_fixedpeak"] = float(
                stats_fixed["collapse"][i].item()
            )
            for gkey in _GEOM_KEYS:
                row[f"{prefix}{gkey}"] = float(geom[gkey][i].item())
            for qkey, qvals in quality.items():
                row[f"{prefix}{qkey}"] = float(qvals[i].item())
            if i < len(per_image):
                per_image[i] = row
            else:
                per_image.append(row)

        block: dict[str, Any] = {
            "collapse_mean": float(stats["collapse"].mean().item()),
            "collapse_std": float(stats["collapse"].std(unbiased=False).item()),
            "t_peak_mean": float(stats["t_peak"].float().mean().item()),
            "global_t_peak": global_t_peak,
            "cos_peak_mean": float(stats["cos_peak"].mean().item()),
            "cos_t0_mean": float(stats["cos_t0"].mean().item()),
            "correlations": corr,
            "correlations_fixed_peak": corr_fixed,
            "reference_indices_head": ref_idx[: min(8, len(ref_idx))],
        }
        if "psnr" in quality:
            block["psnr_mean"] = float(quality["psnr"].mean().item())
            block["psnr_std"] = float(quality["psnr"].std(unbiased=False).item())
        if "lpips" in quality:
            block["lpips_mean"] = float(quality["lpips"].mean().item())
            block["lpips_std"] = float(quality["lpips"].std(unbiased=False).item())
        if qcorr:
            block["quality_correlations"] = qcorr
        if robustness:
            block["robustness"] = robustness
        model_blocks[name] = block

    primary_ref_report = (
        int(reference_images) if reference_images is not None else int(num_images)
    )
    result: dict[str, Any] = {
        "num_images": int(num_images),
        "reference_n": primary_ref_report,
        "knn": int(knn),
        "noise_seed": int(noise_seed),
        "start_index": int(start_index),
        "fixed_t_peak": fixed_t_peak,
        "n_boot": int(n_boot),
        "compute_image_metrics": bool(compute_image_metrics),
        "knn_grid": knn_grid,
        "reference_grid": reference_grid,
        "models": model_blocks,
        "per_image": per_image[:num_images],
        "note": (
            "Exploratory. C_i = cos(ẑ0(t_peak), z_lr) − cos(ẑ0(0), z_lr). "
            "Bootstrap CIs resample images with replacement. "
            "density and mean_knn_dist are inverse neighborhood signals. "
            "paired_delta correlates Δgeometry with ΔPSNR/ΔLPIPS "
            f"({candidate_name} − {baseline_name})."
        ),
    }
    paired = paired_delta_analysis(
        result["per_image"],
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        n_boot=n_boot,
        seed=noise_seed,
    )
    if paired is not None:
        result["paired_delta"] = paired
    elif show_progress and len(models) >= 2:
        print(
            "NOTE: paired Δ analysis skipped (need PSNR+density for both "
            f"{baseline_name} and {candidate_name}).",
            flush=True,
        )

    if output_dir is not None:
        result["paths"] = write_collapse_geometry_report(result, output_dir)
    return result
