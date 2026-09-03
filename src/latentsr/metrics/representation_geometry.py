"""RiT-style representation geometry diagnostics on VAE latents.

Metrics (Zhang et al., RiT / Facco et al. TwoNN):

    - TwoNN intrinsic dimensionality
    - PCA spectrum + effective rank
    - Covariance condition number κ(H) and transport κ(Σ_t)
    - Per-dimension excess kurtosis

Latents are flattened per image: ``(N, C, H, W) → (N, C·H·W)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from latentsr.datasets.onthefly_sr_latent import upsample_bicubic
from latentsr.vae.latent import encode_scaled
from latentsr.vae.vae import VAE


def flatten_latents(z: torch.Tensor) -> torch.Tensor:
    """``(N, C, H, W)`` or ``(N, D)`` → float64 ``(N, D)`` on CPU."""
    if z.ndim == 2:
        flat = z
    elif z.ndim == 4:
        flat = z.flatten(1)
    else:
        raise ValueError(f"Expected (N, D) or (N, C, H, W), got {tuple(z.shape)}")
    return flat.detach().to(dtype=torch.float64, device="cpu")


def center_features(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def pca_eigenvalues(x: torch.Tensor) -> torch.Tensor:
    """Sample-covariance eigenvalues of centered ``(N, D)``, descending.

    Uses SVD of the centered matrix so the result is well-defined for N ≤ D.
    Rank is at most ``min(N - 1, D)``.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (N, D), got {tuple(x.shape)}")
    n, d = x.shape
    if n < 2:
        raise ValueError(f"Need at least 2 samples for PCA, got {n}")
    xc = center_features(x)
    # Economy SVD: X = U S V^T; cov eigenvalues = S² / (N-1)
    singular = torch.linalg.svdvals(xc)
    eig = (singular.pow(2) / float(n - 1)).clamp_min(0.0)
    # Pad zeros up to ambient dim so spectra are comparable across spaces.
    if eig.numel() < d:
        eig = F.pad(eig, (0, d - eig.numel()))
    return eig


def explained_variance_ratio(eigenvalues: torch.Tensor) -> torch.Tensor:
    total = eigenvalues.sum().clamp_min(1e-12)
    return eigenvalues / total


def cumulative_variance(eigenvalues: torch.Tensor) -> torch.Tensor:
    return explained_variance_ratio(eigenvalues).cumsum(dim=0)


def effective_rank(eigenvalues: torch.Tensor) -> float:
    """``erank = exp(−Σ λ̂ log λ̂)`` with ``λ̂ = λ / Σλ`` (Roy & Vetterli)."""
    ratios = explained_variance_ratio(eigenvalues)
    positive = ratios[ratios > 0]
    if positive.numel() == 0:
        return 0.0
    entropy = -(positive * positive.log()).sum()
    return float(entropy.exp().item())


def covariance_condition_number(
    eigenvalues: torch.Tensor,
    *,
    num_samples: int | None = None,
    relative_floor: float = 1e-12,
) -> dict[str, float]:
    """κ of the sample covariance from its eigenvalue spectrum.

    Uses the leading ``r = min(N-1, D)`` eigenvalues (including tiny ones).
    If ``r < D``, the ambient covariance is singular and ``kappa_ambient`` is
    infinite; ``kappa`` still reports the condition within the observed span.
    """
    if eigenvalues.numel() == 0:
        raise ValueError("empty eigenvalue spectrum")
    d = int(eigenvalues.numel())
    if num_samples is None:
        rank = d
    else:
        rank = max(min(int(num_samples) - 1, d), 1)
    eigs_r = eigenvalues[:rank]
    lam_max = float(eigs_r.max().item())
    if lam_max <= 0:
        return {
            "kappa": float("inf"),
            "kappa_ambient": float("inf"),
            "lambda_max": 0.0,
            "lambda_min_pos": 0.0,
            "numerical_rank": 0.0,
            "observed_rank": float(rank),
            "ambient_dim": float(d),
        }
    floor = max(relative_floor * lam_max, 1e-30)
    lam_min = float(eigs_r.min().clamp_min(floor).item())
    kappa = lam_max / lam_min
    ambient_singular = rank < d
    return {
        "kappa": kappa,
        "kappa_ambient": float("inf") if ambient_singular else kappa,
        "lambda_max": lam_max,
        "lambda_min_pos": lam_min,
        "numerical_rank": float((eigs_r > floor).sum().item()),
        "observed_rank": float(rank),
        "ambient_dim": float(d),
    }


def transport_condition_numbers(
    eigenvalues: torch.Tensor,
    t_values: list[float] | tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0),
    *,
    relative_floor: float = 1e-12,
) -> list[dict[str, float]]:
    """κ(Σ_t) for ``Σ_t = (1-t)² I + t² H`` (RiT / Ahamed et al.).

    Eigenvalues of Σ_t are ``(1-t)² + t² λ_i`` in the PCA basis; unobserved
    ambient directions (λ=0 when N < D) still receive the noise floor
    ``(1-t)²``, so κ is finite for all t < 1.
    """
    d = int(eigenvalues.numel())
    rows: list[dict[str, float]] = []
    for t in t_values:
        tt = float(t)
        noise = (1.0 - tt) ** 2
        signal = tt**2
        sigma_eigs = noise + signal * eigenvalues
        # Full ambient spectrum for transport (noise fills null space).
        stats = covariance_condition_number(
            sigma_eigs,
            num_samples=d + 1,
            relative_floor=relative_floor,
        )
        rows.append(
            {
                "t": tt,
                "kappa": stats["kappa"],
                "lambda_max": stats["lambda_max"],
                "lambda_min_pos": stats["lambda_min_pos"],
                "numerical_rank": stats["numerical_rank"],
                "ambient_dim": float(d),
            }
        )
    return rows

def excess_kurtosis_stats(x: torch.Tensor) -> dict[str, float]:
    """Per-coordinate excess kurtosis summary over feature dim."""
    if x.ndim != 2:
        raise ValueError(f"Expected (N, D), got {tuple(x.shape)}")
    n, d = x.shape
    if n < 4:
        raise ValueError(f"Need at least 4 samples for kurtosis, got {n}")
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    var = centered.pow(2).mean(dim=0).clamp_min(1e-12)
    m4 = centered.pow(4).mean(dim=0)
    kappa = m4 / var.pow(2) - 3.0
    abs_k = kappa.abs()
    return {
        "mean_abs_kurtosis": float(abs_k.mean().item()),
        "median_abs_kurtosis": float(abs_k.median().item()),
        "mean_kurtosis": float(kappa.mean().item()),
        "frac_abs_lt_0_5": float((abs_k < 0.5).float().mean().item()),
        "frac_abs_lt_1_0": float((abs_k < 1.0).float().mean().item()),
        "num_dims": float(d),
        "num_samples": float(n),
    }


def _pairwise_nn_ratios(x: torch.Tensor) -> torch.Tensor:
    """TwoNN ratios μ = r2 / r1 for every point (self-distance excluded)."""
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"TwoNN needs at least 3 samples, got {n}")
    # Chunked cdist to keep peak memory reasonable for N~5k, D~4k.
    chunk = max(1, min(512, n))
    ratios = torch.empty(n, dtype=torch.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dists = torch.cdist(x[start:stop], x, p=2)
        # Mask self distances.
        for i, global_i in enumerate(range(start, stop)):
            dists[i, global_i] = float("inf")
        knn = torch.topk(dists, k=2, largest=False, dim=1).values
        r1 = knn[:, 0].clamp_min(1e-12)
        r2 = knn[:, 1].clamp_min(r1)
        ratios[start:stop] = r2 / r1
    return ratios


def twonn_intrinsic_dim(x: torch.Tensor) -> float:
    """Facco et al. TwoNN MLE: ``d̂ = N / Σ log(r2/r1)``."""
    ratios = _pairwise_nn_ratios(x)
    logs = ratios.clamp_min(1.0 + 1e-12).log()
    return float(x.shape[0] / logs.sum().clamp_min(1e-12).item())


def twonn_bootstrap(
    x: torch.Tensor,
    *,
    num_bootstraps: int = 10,
    subsample_size: int | None = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """RiT-style TwoNN: mean ± std over independent subsamples."""
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"TwoNN needs at least 3 samples, got {n}")
    target = n if subsample_size is None else min(int(subsample_size), n)
    if target < 3:
        raise ValueError(f"subsample_size must be >= 3, got {target}")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    estimates: list[float] = []
    for _ in range(max(int(num_bootstraps), 1)):
        if target == n:
            sample = x
        else:
            idx = torch.randperm(n, generator=g)[:target]
            sample = x[idx]
        estimates.append(twonn_intrinsic_dim(sample))
    vals = torch.tensor(estimates, dtype=torch.float64)
    return {
        "twonn_mean": float(vals.mean().item()),
        "twonn_std": float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        "twonn_n_boot": float(vals.numel()),
        "twonn_subsample": float(target),
        "twonn_full": float(twonn_intrinsic_dim(x)),
    }


def summarize_geometry(
    x: torch.Tensor,
    *,
    name: str,
    twonn_bootstraps: int = 10,
    twonn_subsample: int | None = 5000,
    seed: int = 42,
    top_k_spectrum: int = 256,
) -> dict[str, Any]:
    """Full geometry pack for one latent cloud ``(N, D)``."""
    flat = flatten_latents(x)
    n, d = flat.shape
    eigs = pca_eigenvalues(flat)
    ratios = explained_variance_ratio(eigs)
    cum = cumulative_variance(eigs)
    kappa = covariance_condition_number(eigs, num_samples=n)
    transport = transport_condition_numbers(eigs)
    kurt = excess_kurtosis_stats(flat)
    twonn = twonn_bootstrap(
        flat,
        num_bootstraps=twonn_bootstraps,
        subsample_size=twonn_subsample,
        seed=seed,
    )
    k = min(int(top_k_spectrum), d)
    return {
        "name": name,
        "num_samples": n,
        "ambient_dim": d,
        "twonn": twonn,
        "effective_rank": effective_rank(eigs),
        "covariance": kappa,
        "transport_kappa": transport,
        "kurtosis": kurt,
        "pca": {
            "top_k": k,
            "explained_variance_ratio": ratios[:k].tolist(),
            "cumulative_variance": cum[:k].tolist(),
            "eigenvalues": eigs[:k].tolist(),
            "var_in_top_50": float(cum[min(49, d - 1)].item()),
            "var_in_top_100": float(cum[min(99, d - 1)].item()),
        },
    }


@torch.no_grad()
def collect_vae_latents(
    vae: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    """Encode matched ``(z_hr, z_lr)`` stacks from an SR pair loader."""
    vae.eval()
    z_hr_chunks: list[torch.Tensor] = []
    z_lr_chunks: list[torch.Tensor] = []
    remaining = max(int(num_images), 1)
    iterator = tqdm(loader, desc="encode", leave=False) if show_progress else loader
    for lr, hr in iterator:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        hr = hr[:take].to(device)
        bicubic = upsample_bicubic(lr, hr_size)
        z_hr = encode_scaled(vae, hr, latent_scale=latent_scale)
        z_lr = encode_scaled(vae, bicubic, latent_scale=latent_scale)
        z_hr_chunks.append(z_hr.cpu())
        z_lr_chunks.append(z_lr.cpu())
        remaining -= take
    if not z_hr_chunks:
        raise ValueError("No latents collected; check num_images / dataloader.")
    return {
        "z_hr": torch.cat(z_hr_chunks, dim=0),
        "z_lr": torch.cat(z_lr_chunks, dim=0),
    }


def compare_vae_geometries(
    baseline_latents: dict[str, torch.Tensor],
    candidate_latents: dict[str, torch.Tensor],
    *,
    baseline_name: str = "vae1",
    candidate_name: str = "vae_sr",
    twonn_bootstraps: int = 10,
    twonn_subsample: int | None = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Geometry for HR and LR latents under two frozen VAEs."""
    spaces = {
        f"{baseline_name}_hr": baseline_latents["z_hr"],
        f"{baseline_name}_lr": baseline_latents["z_lr"],
        f"{candidate_name}_hr": candidate_latents["z_hr"],
        f"{candidate_name}_lr": candidate_latents["z_lr"],
    }
    results = {
        name: summarize_geometry(
            z,
            name=name,
            twonn_bootstraps=twonn_bootstraps,
            twonn_subsample=twonn_subsample,
            seed=seed,
        )
        for name, z in spaces.items()
    }
    return {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "spaces": results,
    }


def format_geometry_table(report: dict[str, Any]) -> str:
    spaces: dict[str, Any] = report["spaces"]
    lines = [
        "Representation geometry (RiT-style)",
        f"{'space':<14} {'N':>5} {'D':>5} {'TwoNN':>10} {'erank':>8} "
        f"{'κ(H)':>10} {'|κ| med':>9} {'%|κ|<0.5':>9} {'top50%':>7}",
        "-" * 86,
    ]
    for name, stats in spaces.items():
        tw = stats["twonn"]["twonn_mean"]
        tw_std = stats["twonn"]["twonn_std"]
        lines.append(
            f"{name:<14} {stats['num_samples']:5d} {stats['ambient_dim']:5d} "
            f"{tw:6.2f}±{tw_std:<3.2f} {stats['effective_rank']:8.1f} "
            f"{stats['covariance']['kappa']:10.1f} "
            f"{stats['kurtosis']['median_abs_kurtosis']:9.3f} "
            f"{100.0 * stats['kurtosis']['frac_abs_lt_0_5']:8.1f}% "
            f"{100.0 * stats['pca']['var_in_top_50']:6.1f}%"
        )
    lines.append("")
    lines.append("Transport κ(Σ_t) at t=0.9:")
    for name, stats in spaces.items():
        row = next(r for r in stats["transport_kappa"] if abs(r["t"] - 0.9) < 1e-9)
        lines.append(f"  {name:<14} κ={row['kappa']:.2f}")
    return "\n".join(lines)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_geometry_report(report: dict[str, Any], output_dir: str | Path) -> None:
    """Write PCA cumulative, kurtosis-free spectrum, and transport-κ plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spaces: dict[str, Any] = report["spaces"]
    colors = {
        k: c
        for k, c in zip(
            spaces.keys(),
            ["#4c78a8", "#72b7b2", "#f58518", "#e45756"],
            strict=False,
        )
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, stats in spaces.items():
        cum = stats["pca"]["cumulative_variance"]
        ax.plot(
            range(1, len(cum) + 1),
            cum,
            label=name,
            color=colors.get(name),
            linewidth=1.8,
        )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative variance")
    ax.set_title("PCA cumulative variance spectrum")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save_fig(fig, output_dir / "pca_cumulative_variance.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, stats in spaces.items():
        eigs = torch.tensor(stats["pca"]["eigenvalues"], dtype=torch.float64)
        # Log per-component variance (first K).
        ax.semilogy(
            range(1, eigs.numel() + 1),
            eigs.clamp_min(1e-12),
            label=name,
            color=colors.get(name),
            linewidth=1.5,
        )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue (log)")
    ax.set_title("PCA eigenvalue spectrum")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save_fig(fig, output_dir / "pca_eigenvalue_spectrum.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, stats in spaces.items():
        ts = [r["t"] for r in stats["transport_kappa"]]
        ks = [r["kappa"] for r in stats["transport_kappa"]]
        ax.semilogy(ts, ks, marker="o", label=name, color=colors.get(name))
    ax.set_xlabel("t (noise → data)")
    ax.set_ylabel("κ(Σ_t)")
    ax.set_title("Transport covariance condition number")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save_fig(fig, output_dir / "transport_condition_number.png")

    # Kurtosis summary bars (median |κ| and frac < 0.5).
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    names = list(spaces.keys())
    med = [spaces[n]["kurtosis"]["median_abs_kurtosis"] for n in names]
    frac = [100.0 * spaces[n]["kurtosis"]["frac_abs_lt_0_5"] for n in names]
    bar_colors = [colors.get(n, "#888888") for n in names]
    axes[0].bar(names, med, color=bar_colors)
    axes[0].set_ylabel("median |excess kurtosis|")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Marginal non-Gaussianity")
    axes[1].bar(names, frac, color=bar_colors)
    axes[1].set_ylabel("% dims with |κ| < 0.5")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Near-Gaussian fraction")
    _save_fig(fig, output_dir / "excess_kurtosis_summary.png")


def write_geometry_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    table = format_geometry_table(report)
    (output_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")
    plot_geometry_report(report, output_dir)
    return metrics_path


def run_representation_geometry(
    vae_baseline: VAE,
    vae_candidate: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int,
    output_dir: str | Path,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    baseline_name: str = "vae1",
    candidate_name: str = "vae_sr",
    twonn_bootstraps: int = 10,
    twonn_subsample: int | None = 5000,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Encode both VAEs on the same images and write the geometry report."""
    baseline_latents = collect_vae_latents(
        vae_baseline,
        loader,
        device=device,
        num_images=num_images,
        hr_size=hr_size,
        latent_scale=latent_scale,
        show_progress=show_progress,
    )
    # Re-create iterator by requiring the caller to pass a fresh loader, or
    # rewind by collecting again from the same loader object only if it can be
    # re-iterated. DataLoader is re-iterable; collect again for the candidate.
    candidate_latents = collect_vae_latents(
        vae_candidate,
        loader,
        device=device,
        num_images=num_images,
        hr_size=hr_size,
        latent_scale=latent_scale,
        show_progress=show_progress,
    )
    if baseline_latents["z_hr"].shape != candidate_latents["z_hr"].shape:
        raise ValueError(
            "Latent stack mismatch between VAEs: "
            f"{tuple(baseline_latents['z_hr'].shape)} vs "
            f"{tuple(candidate_latents['z_hr'].shape)}"
        )
    report = compare_vae_geometries(
        baseline_latents,
        candidate_latents,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        twonn_bootstraps=twonn_bootstraps,
        twonn_subsample=twonn_subsample,
        seed=seed,
    )
    report["meta"] = {
        "num_images": int(num_images),
        "hr_size": int(hr_size),
        "latent_scale": float(latent_scale),
        "seed": int(seed),
        "twonn_bootstraps": int(twonn_bootstraps),
        "twonn_subsample": twonn_subsample,
    }
    write_geometry_report(report, output_dir)
    return report
