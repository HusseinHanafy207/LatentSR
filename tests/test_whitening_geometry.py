"""Whitening geometry gate: κ↓, erank↑, PCA less concentrated."""

from __future__ import annotations

import torch

from latentsr.metrics.whitening_geometry import (
    compare_raw_vs_whitened_z_lr,
    format_whitening_geometry_block,
)
from latentsr.vae.whitening import fit_channel_whitening


def test_whitening_geometry_gate_passes_on_zca() -> None:
    torch.manual_seed(0)
    # Correlated 4-channel cloud so ZCA should clearly change geometry.
    b, c, h, w = 64, 4, 8, 8
    base = torch.randn(b, 1, h, w)
    z = torch.cat(
        [
            base,
            base + 0.1 * torch.randn(b, 1, h, w),
            base + 0.1 * torch.randn(b, 1, h, w),
            0.05 * torch.randn(b, 1, h, w),
        ],
        dim=1,
    )
    acc = fit_channel_whitening(num_channels=4, eps=1e-4, mode="zca")
    acc.update(z)
    whitener = acc.finalize()
    block = compare_raw_vs_whitened_z_lr(z, whitener, name="toy")
    assert block["ok_channel_kappa_dropped"]
    assert block["ok_channel_erank_rose"]
    assert block["ok_pca_less_concentrated"]
    assert block["geometry_ok"]
    text = format_whitening_geometry_block(block)
    assert "κ↓=True" in text
    assert "erank↑=True" in text
    assert "geometry_ok=True" in text
