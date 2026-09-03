"""Phase 1.5: correlate late reverse-chain collapse with local z_lr geometry.

For each image:

    C_i = cos(ẑ0(t_peak), z_lr) − cos(ẑ0(0), z_lr)

Higher C_i ⇒ stronger late-stage collapse of the mid-chain copy.

Local geometry around each z_lr (leave-one-out k-NN in a reference cloud):

    local effective rank, local κ, neighborhood density, NN distance.

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
from latentsr.vae.vae import VAE


_GEOM_KEYS = (
    "local_erank",
    "local_kappa",
    "nn_dist",
    "mean_knn_dist",
    "density",
)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if xa.size < 3 or ya.size != xa.size:
        return float("nan")
    if float(xa.std()) < 1e-12 or float(ya.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


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
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Run the reverse chain; return ``(cos[N,T], val_indices, z_lr[N,…])``."""
    model.eval()
    vae.eval()
    model.to(device)
    vae.to(device)
    num_t = int(model.num_timesteps)
    cos_chunks: list[torch.Tensor] = []
    z_chunks: list[torch.Tensor] = []
    indices: list[int] = []
    remaining = max(int(num_images), 1)
    next_index = int(start_index)
    pbar = (
        tqdm(total=remaining, desc="reverse cos curves", unit="img", leave=True)
        if show_progress
        else None
    )

    for lr, _hr in loader:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        batch_idx = list(range(next_index, next_index + take))
        z_lr = encode_lr_latents(
            vae, lr, hr_size=hr_size, latent_scale=latent_scale
        )
        x = seeded_noise_like(z_lr, batch_idx, base_seed=noise_seed, salt=0)
        # Store cosine at every t (column = t).
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
        indices.extend(batch_idx)
        remaining -= take
        next_index += take
        if pbar is not None:
            pbar.update(take)

    if pbar is not None:
        pbar.close()
    if not cos_chunks:
        raise ValueError("No reverse curves collected.")
    return torch.cat(cos_chunks, dim=0), indices, torch.cat(z_chunks, dim=0)


def correlate_collapse_with_geometry(
    collapse: torch.Tensor | np.ndarray,
    geometry: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Pearson + Spearman of C_i vs each local geometry scalar."""
    c = np.asarray(
        collapse.detach().cpu().numpy() if torch.is_tensor(collapse) else collapse,
        dtype=np.float64,
    ).reshape(-1)
    out: dict[str, dict[str, float]] = {}
    for key in _GEOM_KEYS:
        if key not in geometry:
            continue
        g = geometry[key].detach().cpu().numpy().astype(np.float64).reshape(-1)
        out[key] = {
            "pearson_r": pearson_r(c, g),
            "spearman_rho": float(spearman_rho(c, g)),
            "n": float(c.size),
        }
    return out


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
        f"n={result['num_images']}  knn={result['knn']}  "
        f"reference_n={result['reference_n']}",
        "",
    ]
    for model_name, block in result["models"].items():
        lines.append(f"[{model_name}]")
        lines.append(
            f"  mean C_i (per-image peak) = {block['collapse_mean']:.4f} "
            f"± {block['collapse_std']:.4f}"
        )
        lines.append(
            f"  mean t_peak = {block['t_peak_mean']:.1f}  "
            f"global mean-curve peak t = {block['global_t_peak']}"
        )
        lines.append(
            f"  {'metric':<16} {'pearson':>9} {'spearman':>9}"
        )
        for key, corr in block["correlations"].items():
            lines.append(
                f"  {key:<16} {corr['pearson_r']:9.3f} {corr['spearman_rho']:9.3f}"
            )
        lines.append("")
    lines.append(
        "Positive corr(C, density) / negative corr(C, local_erank) would mean "
        "worse-behaved neighborhoods collapse more — interpret cautiously."
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

    for model_name in result["models"]:
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
    return written


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
    output_dir: str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Reverse-chain collapse + local z_lr geometry correlation.

    ``models`` / ``vaes`` share keys (e.g. ``vae1``, ``vae_sr``). Each model is
    paired with its VAE. Reference cloud size defaults to ``num_images``.
    """
    if not models:
        raise ValueError("models must be non-empty")
    if set(models) != set(vaes):
        raise ValueError("models and vaes must share the same keys")
    latent_scales = latent_scales or {k: 1.0 for k in models}
    ref_n = int(reference_images) if reference_images is not None else int(num_images)
    if ref_n < num_images:
        raise ValueError(
            f"reference_images ({ref_n}) must be >= num_images ({num_images})"
        )
    if knn < 2:
        raise ValueError(f"knn must be >= 2 for local PCA, got {knn}")

    # One pass over the loader for reverse curves (smaller N).
    # Rebuild iteration by reusing the same loader from the start — callers
    # must pass shuffle=False loaders.
    per_image: list[dict[str, Any]] = [
        {"val_index": start_index + i} for i in range(num_images)
    ]
    model_blocks: dict[str, Any] = {}

    for name in models:
        model = models[name]
        vae = vaes[name]
        scale = float(latent_scales.get(name, 1.0))
        if show_progress:
            print(f"[{name}] reverse cosine curves (n={num_images})…", flush=True)
        cos, idx, z_eval = reverse_cosine_curves(
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
        )
        if idx != [start_index + i for i in range(len(idx))]:
            # Keep actual indices from the pass.
            pass

        # Reference bank: reuse eval z_lr when ref_n == num_images; else re-encode.
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
        # Also compute fixed-peak collapse using this run's mean-curve peak.
        stats_fixed = collapse_from_cosine_curve(
            cos, fixed_t_peak=global_t_peak if fixed_t_peak is None else fixed_t_peak
        )

        if show_progress:
            print(f"[{name}] local geometry (k={knn}, ref={z_ref.shape[0]})…", flush=True)
        geom = local_geometry_at_points(z_eval, z_ref, k=knn, exclude_self=True)
        corr = correlate_collapse_with_geometry(stats["collapse"], geom)
        corr_fixed = correlate_collapse_with_geometry(stats_fixed["collapse"], geom)

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
            if i < len(per_image):
                per_image[i] = row
            else:
                per_image.append(row)

        model_blocks[name] = {
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

    result: dict[str, Any] = {
        "num_images": int(num_images),
        "reference_n": int(ref_n),
        "knn": int(knn),
        "noise_seed": int(noise_seed),
        "start_index": int(start_index),
        "fixed_t_peak": fixed_t_peak,
        "models": model_blocks,
        "per_image": per_image[:num_images],
        "note": (
            "Exploratory correlation only. C_i = cos(ẑ0(t_peak), z_lr) − "
            "cos(ẑ0(0), z_lr) with per-image t_peak=argmax_t cos unless "
            "fixed_t_peak is set; collapse_fixedpeak uses the run's mean-curve peak."
        ),
    }
    if output_dir is not None:
        result["paths"] = write_collapse_geometry_report(result, output_dir)
    return result
