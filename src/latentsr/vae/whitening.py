"""Channel-wise ZCA whitening for LR condition latents.

Treat each spatial site as a 4-D sample ``z_{h,w} ∈ R^C`` (C=4). Fit
``μ, Σ`` on **training** latents only, freeze, then apply

    z'_{h,w} = W (z_{h,w} − μ),   W = (Σ + ε I)^{-1/2}

so ``Cov(z') ≈ I`` while the ``H×W`` layout is preserved.

Never fit on val/test. Never apply to HR diffusion targets — conditions only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _sym_matrix_inv_sqrt(
    cov: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(W, eigenvalues, Σ_ε)`` for SPD ``cov`` with ridge ``eps``."""
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Expected square cov, got {tuple(cov.shape)}")
    c = int(cov.shape[0])
    cov_eps = cov + float(eps) * torch.eye(c, dtype=cov.dtype, device=cov.device)
    # eigh is stable for symmetric matrices; eigenvalues ascending.
    evals, evecs = torch.linalg.eigh(cov_eps)
    evals = evals.clamp_min(float(eps))
    inv_sqrt = evals.rsqrt()
    # W = U diag(λ^{-1/2}) U^T
    w = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.transpose(-1, -2)
    return w, evals, cov_eps


class ChannelWhitening:
    """Frozen channel ZCA (or diagonal standardization) for ``(B,C,H,W)`` latents."""

    def __init__(
        self,
        mean: torch.Tensor,
        matrix: torch.Tensor,
        *,
        eps: float,
        mode: str = "zca",
        meta: dict[str, Any] | None = None,
    ) -> None:
        if mean.ndim != 1:
            raise ValueError(f"mean must be (C,), got {tuple(mean.shape)}")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"matrix must be (C,C), got {tuple(matrix.shape)}")
        if int(mean.shape[0]) != int(matrix.shape[0]):
            raise ValueError("mean / matrix channel mismatch")
        if mode not in {"zca", "standardize"}:
            raise ValueError(f"mode must be 'zca' or 'standardize', got {mode!r}")
        self.mean = mean.detach().to(dtype=torch.float64).cpu()
        self.matrix = matrix.detach().to(dtype=torch.float64).cpu()
        self.eps = float(eps)
        self.mode = str(mode)
        self.meta: dict[str, Any] = dict(meta or {})

    @property
    def num_channels(self) -> int:
        return int(self.mean.numel())

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> ChannelWhitening:
        # Stats stay float64 on CPU; transform casts per-call. Device is a no-op
        # kept for API symmetry with modules.
        _ = device, dtype
        return self

    @torch.no_grad()
    def transform(self, z: torch.Tensor) -> torch.Tensor:
        """Apply ``W(z − μ)`` channel-wise; preserves ``(B,C,H,W)`` layout."""
        if z.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(z.shape)}")
        if z.shape[1] != self.num_channels:
            raise ValueError(
                f"Channel mismatch: z has {z.shape[1]}, whitener has {self.num_channels}"
            )
        # Work in float32 for speed; W/μ stored float64.
        mean = self.mean.to(device=z.device, dtype=z.dtype).view(1, -1, 1, 1)
        w = self.matrix.to(device=z.device, dtype=z.dtype)  # (C,C)
        centered = z - mean
        # (B,C,H,W) → (B,H,W,C) @ W^T → …  (rows of W applied to channel vec)
        # z'_c = Σ_d W_{c,d} (z_d − μ_d)  ⇒  out = W @ centered over channel dim
        b, c, h, w_ = centered.shape
        flat = centered.permute(0, 2, 3, 1).reshape(-1, c)  # (N, C)
        out = flat @ w.transpose(0, 1)
        return out.view(b, h, w_, c).permute(0, 3, 1, 2).contiguous()

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "matrix": self.matrix,
            "eps": self.eps,
            "mode": self.mode,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> ChannelWhitening:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        return cls(
            payload["mean"],
            payload["matrix"],
            eps=float(payload.get("eps", 1e-4)),
            mode=str(payload.get("mode", "zca")),
            meta=dict(payload.get("meta") or {}),
        )


def fit_channel_whitening(
    *,
    num_channels: int,
    eps: float = 1e-4,
    mode: str = "zca",
    meta: dict[str, Any] | None = None,
) -> _ChannelWhiteningAccumulator:
    """Return a streaming accumulator; call ``.update(z)`` then ``.finalize()``."""
    return _ChannelWhiteningAccumulator(
        num_channels=num_channels, eps=eps, mode=mode, meta=meta
    )


class _ChannelWhiteningAccumulator:
    """Welford-style online mean + covariance over channel vectors."""

    def __init__(
        self,
        *,
        num_channels: int,
        eps: float,
        mode: str,
        meta: dict[str, Any] | None,
    ) -> None:
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if mode not in {"zca", "standardize"}:
            raise ValueError(f"mode must be 'zca' or 'standardize', got {mode!r}")
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.mode = str(mode)
        self.meta = dict(meta or {})
        self.n = 0
        self._mean = torch.zeros(self.num_channels, dtype=torch.float64)
        self._m2 = torch.zeros(
            self.num_channels, self.num_channels, dtype=torch.float64
        )

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """Accumulate from a ``(B,C,H,W)`` latent batch (any dtype/device)."""
        if z.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(z.shape)}")
        if z.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected C={self.num_channels}, got {z.shape[1]}"
            )
        # (N, C) channel vectors at every spatial site.
        flat = (
            z.detach()
            .to(dtype=torch.float64, device="cpu")
            .permute(0, 2, 3, 1)
            .reshape(-1, self.num_channels)
        )
        n_batch = int(flat.shape[0])
        if n_batch == 0:
            return
        batch_mean = flat.mean(dim=0)
        centered = flat - batch_mean
        batch_cov = centered.transpose(0, 1) @ centered  # sum of outer products

        if self.n == 0:
            self._mean = batch_mean
            self._m2 = batch_cov
            self.n = n_batch
            return

        n_total = self.n + n_batch
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (n_batch / n_total)
        # Chan et al. parallel covariance update.
        self._m2 = (
            self._m2
            + batch_cov
            + torch.outer(delta, delta) * (self.n * n_batch / n_total)
        )
        self.n = n_total

    def covariance(self) -> torch.Tensor:
        if self.n < 2:
            raise ValueError(f"Need at least 2 channel vectors, got n={self.n}")
        return self._m2 / float(self.n - 1)

    def finalize(self) -> ChannelWhitening:
        if self.n < 2:
            raise ValueError(f"Need at least 2 channel vectors to fit, got n={self.n}")
        cov = self.covariance()
        if self.mode == "standardize":
            var = torch.diag(cov).clamp_min(self.eps)
            matrix = torch.diag(var.rsqrt())
            evals = var
        else:
            matrix, evals, _ = _sym_matrix_inv_sqrt(cov, eps=self.eps)
        meta = {
            **self.meta,
            "n_channel_vectors": int(self.n),
            "cov_eigenvalues": evals.detach().cpu().tolist(),
            "cov_condition_raw": float(
                (cov.diag().max() / cov.diag().clamp_min(1e-30).min()).item()
            )
            if self.mode == "standardize"
            else float(
                (
                    torch.linalg.eigvalsh(cov).abs().max()
                    / torch.linalg.eigvalsh(cov).abs().clamp_min(1e-30).min()
                ).item()
            ),
        }
        return ChannelWhitening(
            self._mean.clone(),
            matrix,
            eps=self.eps,
            mode=self.mode,
            meta=meta,
        )


def channel_covariance_stats(z: torch.Tensor) -> dict[str, float]:
    """4×4 channel-cov geometry after pooling all spatial sites.

    Useful to verify whitening: κ should drop and erank rise toward C.
    """
    if z.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(z.shape)}")
    flat = z.detach().to(dtype=torch.float64, device="cpu").permute(0, 2, 3, 1)
    flat = flat.reshape(-1, z.shape[1])
    n, c = flat.shape
    if n < 2:
        raise ValueError("Need >= 2 channel vectors")
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (centered.transpose(0, 1) @ centered) / float(n - 1)
    evals = torch.linalg.eigvalsh(cov).clamp_min(0.0).flip(0)  # descending
    total = evals.sum().clamp_min(1e-30)
    ratios = evals / total
    positive = ratios[ratios > 0]
    erank = float((-(positive * positive.log()).sum()).exp().item()) if positive.numel() else 0.0
    lam_max = float(evals.max().item())
    lam_min = float(evals.min().clamp_min(1e-30).item())
    return {
        "num_vectors": float(n),
        "num_channels": float(c),
        "effective_rank": erank,
        "kappa": lam_max / lam_min,
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "var_top1": float(ratios[0].item()),
    }


def load_channel_whitening(
    path: str | Path | None,
) -> ChannelWhitening | None:
    if path is None or str(path).strip() == "":
        return None
    return ChannelWhitening.load(path)
