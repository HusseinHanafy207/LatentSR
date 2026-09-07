"""Whitening geometry checks: κ↓, erank↑, PCA less concentrated.

Used as a gate before the decisive raw-vs-whitened LatentSR evaluation.
"""

from __future__ import annotations

from typing import Any

import torch

from latentsr.metrics.representation_geometry import (
    covariance_condition_number,
    cumulative_variance,
    effective_rank,
    flatten_latents,
    pca_eigenvalues,
)
from latentsr.vae.whitening import ChannelWhitening, channel_covariance_stats


def ambient_geometry_stats(z: torch.Tensor) -> dict[str, float]:
    """Ambient (flattened) PCA geometry for an ``(N,C,H,W)`` latent cloud."""
    flat = flatten_latents(z)
    eigs = pca_eigenvalues(flat)
    kappa = covariance_condition_number(eigs, num_samples=flat.shape[0])
    cum = cumulative_variance(eigs)
    d = int(flat.shape[1])
    return {
        "ambient_erank": effective_rank(eigs),
        "ambient_kappa": float(kappa["kappa"]),
        "ambient_var_top1": float((eigs[0] / eigs.sum().clamp_min(1e-30)).item()),
        "ambient_var_top50": float(cum[min(49, d - 1)].item()),
        "num_images": float(flat.shape[0]),
        "ambient_dim": float(d),
    }


def compare_raw_vs_whitened_z_lr(
    z_raw: torch.Tensor,
    whitener: ChannelWhitening,
    *,
    name: str = "vae_sr",
) -> dict[str, Any]:
    """Compare raw vs whitened ``z_lr`` geometry (channel + ambient).

    Hypothesis checks (RiT-inspired):
      - channel κ decreases
      - channel effective rank increases
      - PCA mass less concentrated (channel top-1 ↓ and ambient top-50 ↓)
    """
    raw_ch = channel_covariance_stats(z_raw)
    raw_amb = ambient_geometry_stats(z_raw)
    z_w = whitener.transform(z_raw)
    white_ch = channel_covariance_stats(z_w)
    white_amb = ambient_geometry_stats(z_w)

    ok_kappa = bool(white_ch["kappa"] < raw_ch["kappa"])
    ok_erank = bool(white_ch["effective_rank"] > raw_ch["effective_rank"] - 1e-9)
    ok_pca_channel = bool(white_ch["var_top1"] < raw_ch["var_top1"] - 1e-12)
    ok_pca_ambient = bool(
        white_amb["ambient_var_top50"] < raw_amb["ambient_var_top50"] - 1e-12
        or white_amb["ambient_var_top1"] < raw_amb["ambient_var_top1"] - 1e-12
    )
    ok_pca = ok_pca_channel or ok_pca_ambient

    return {
        "name": name,
        "raw_channel": raw_ch,
        "raw_ambient": raw_amb,
        "whitened_channel": white_ch,
        "whitened_ambient": white_amb,
        "delta": {
            "channel_kappa": white_ch["kappa"] - raw_ch["kappa"],
            "channel_erank": white_ch["effective_rank"] - raw_ch["effective_rank"],
            "channel_var_top1": white_ch["var_top1"] - raw_ch["var_top1"],
            "ambient_kappa": white_amb["ambient_kappa"] - raw_amb["ambient_kappa"],
            "ambient_erank": white_amb["ambient_erank"] - raw_amb["ambient_erank"],
            "ambient_var_top1": (
                white_amb["ambient_var_top1"] - raw_amb["ambient_var_top1"]
            ),
            "ambient_var_top50": (
                white_amb["ambient_var_top50"] - raw_amb["ambient_var_top50"]
            ),
        },
        "ok_channel_kappa_dropped": ok_kappa,
        "ok_channel_erank_rose": ok_erank,
        "ok_pca_less_concentrated": ok_pca,
        "ok_pca_channel_top1_dropped": ok_pca_channel,
        "ok_pca_ambient_less_concentrated": ok_pca_ambient,
        "geometry_ok": bool(ok_kappa and ok_erank and ok_pca),
    }


def format_whitening_geometry_block(block: dict[str, Any]) -> str:
    lines = [
        f"[{block['name']}] whitening geometry (raw → white)",
        (
            f"  channel κ     {block['raw_channel']['kappa']:.4g} → "
            f"{block['whitened_channel']['kappa']:.4g}  "
            f"(Δ={block['delta']['channel_kappa']:+.4g})  "
            f"κ↓={block['ok_channel_kappa_dropped']}"
        ),
        (
            f"  channel erank {block['raw_channel']['effective_rank']:.4f} → "
            f"{block['whitened_channel']['effective_rank']:.4f}  "
            f"(Δ={block['delta']['channel_erank']:+.4f})  "
            f"erank↑={block['ok_channel_erank_rose']}"
        ),
        (
            f"  channel top1% {100 * block['raw_channel']['var_top1']:.2f} → "
            f"{100 * block['whitened_channel']['var_top1']:.2f}  "
            f"less_conc={block['ok_pca_channel_top1_dropped']}"
        ),
        (
            f"  ambient κ     {block['raw_ambient']['ambient_kappa']:.4g} → "
            f"{block['whitened_ambient']['ambient_kappa']:.4g}  "
            f"(Δ={block['delta']['ambient_kappa']:+.4g})"
        ),
        (
            f"  ambient erank {block['raw_ambient']['ambient_erank']:.1f} → "
            f"{block['whitened_ambient']['ambient_erank']:.1f}  "
            f"(Δ={block['delta']['ambient_erank']:+.1f})"
        ),
        (
            f"  ambient top50%"
            f" {100 * block['raw_ambient']['ambient_var_top50']:.2f} → "
            f"{100 * block['whitened_ambient']['ambient_var_top50']:.2f}  "
            f"less_conc={block['ok_pca_ambient_less_concentrated']}"
        ),
        (
            f"  GATE geometry_ok={block['geometry_ok']}  "
            f"(need κ↓ and erank↑ and PCA less concentrated)"
        ),
    ]
    if not block["geometry_ok"]:
        lines.append(
            "  FAIL: whitening did not change intended geometry — "
            "do not interpret raw-vs-white quality deltas yet."
        )
    return "\n".join(lines)
