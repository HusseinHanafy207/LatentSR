"""Local (k-NN) geometry around individual latent condition codes.

Exploratory helpers for correlating neighborhood structure with reverse-chain
collapse. Not causal — leave-one-out k-NN within a reference cloud.
"""

from __future__ import annotations

from typing import Any

import torch

from latentsr.metrics.representation_geometry import (
    covariance_condition_number,
    effective_rank,
    flatten_latents,
    pca_eigenvalues,
)


def pairwise_euclidean(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    """``(N, D)`` vs ``(M, D)`` → ``(N, M)`` float32 distances."""
    a = x.detach().to(dtype=torch.float32)
    b = a if y is None else y.detach().to(dtype=torch.float32)
    return torch.cdist(a, b, p=2)


def knn_indices(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    k: int,
    exclude_self: bool = True,
    self_atol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(idx, dists)`` of shape ``(N, k)`` for each query row.

    When ``exclude_self`` and a query point coincides with a reference row
    (same cloud, leave-one-out), that neighbor is dropped.
    """
    flat_q = flatten_latents(query).float()
    flat_r = flatten_latents(reference).float()
    n, m = flat_q.shape[0], flat_r.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    need = k + (1 if exclude_self else 0)
    if m < need:
        raise ValueError(
            f"Need at least {need} reference points for k={k} "
            f"(exclude_self={exclude_self}), got {m}"
        )

    dists = pairwise_euclidean(flat_q, flat_r)
    if exclude_self:
        # Mask exact/near duplicates (leave-one-out when query ⊂ reference).
        for i in range(n):
            # Prefer exact index match when clouds share ordering.
            if i < m and torch.allclose(flat_q[i], flat_r[i], atol=self_atol, rtol=0.0):
                dists[i, i] = float("inf")
            else:
                j = int(dists[i].argmin().item())
                if float(dists[i, j].item()) <= self_atol:
                    dists[i, j] = float("inf")

    knn = torch.topk(dists, k=k, largest=False, dim=1)
    return knn.indices, knn.values


def local_geometry_at_points(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    k: int = 32,
    exclude_self: bool = True,
) -> dict[str, torch.Tensor]:
    """Per-query local geometry from a k-NN neighborhood in ``reference``.

    Neighborhood for PCA = the query point itself + its ``k`` neighbors
    (``k+1`` rows). Density uses neighbor distances only (excludes self).
    """
    flat_q = flatten_latents(query)
    flat_r = flatten_latents(reference)
    idx, dists = knn_indices(
        flat_q, flat_r, k=k, exclude_self=exclude_self
    )
    n = flat_q.shape[0]
    erank = torch.empty(n, dtype=torch.float64)
    kappa = torch.empty(n, dtype=torch.float64)
    nn_dist = dists[:, 0].double()
    mean_knn = dists.mean(dim=1).double()
    density = (1.0 / mean_knn.clamp_min(1e-12)).double()

    for i in range(n):
        neigh = flat_r[idx[i]]  # (k, D)
        cloud = torch.cat([flat_q[i : i + 1], neigh], dim=0)
        eigs = pca_eigenvalues(cloud)
        erank[i] = effective_rank(eigs)
        # Local ambient dim is D but observed rank ≤ k; report span κ.
        kappa[i] = covariance_condition_number(
            eigs, num_samples=cloud.shape[0]
        )["kappa"]

    return {
        "local_erank": erank,
        "local_kappa": kappa,
        "nn_dist": nn_dist,
        "mean_knn_dist": mean_knn,
        "density": density,
        "knn_indices": idx,
    }


def summarize_local_geometry(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    """Mean of scalar local metrics (for logging)."""
    out: dict[str, float] = {}
    for key in ("local_erank", "local_kappa", "nn_dist", "mean_knn_dist", "density"):
        if key in stats:
            out[f"{key}_mean"] = float(stats[key].mean().item())
            out[f"{key}_std"] = float(stats[key].std(unbiased=False).item())
    return out


def local_geometry_rows(
    val_indices: list[int],
    stats: dict[str, torch.Tensor],
    *,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Flatten local geometry tensors into per-image dict rows."""
    n = len(val_indices)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {"val_index": int(val_indices[i])}
        for key in ("local_erank", "local_kappa", "nn_dist", "mean_knn_dist", "density"):
            row[f"{prefix}{key}"] = float(stats[key][i].item())
        rows.append(row)
    return rows
