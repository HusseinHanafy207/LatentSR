"""DDPM output parameterization helpers (ε ↔ x0).

These are algebraically invertible given the standard forward process

    z_t = √ᾱ_t z_0 + √(1−ᾱ_t) ε.

Keeping the training loss as ``||ε̂ − ε||²`` while the network outputs ``ẑ0``
is equivalent to an SNR-weighted x0 MSE:

    L_ε = (ᾱ_t / (1−ᾱ_t)) ||ẑ0 − z_0||² = SNR(t) L_{x0}.

So changing ``prediction_type`` changes the *function* the UNet must
represent, not the underlying loss functional / schedule / optimizer.
"""

from __future__ import annotations

import torch
from generative_models.ddpm import NoiseScheduler


def predict_x0_from_eps(
    scheduler: NoiseScheduler,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_hat: torch.Tensor,
) -> torch.Tensor:
    """Closed-form ε-prediction → clean latent (no clamp; latents unbounded).

        ẑ0 = (x_t − √(1−ᾱ_t) ε̂) / √ᾱ_t
    """
    if eps_hat.shape != x_t.shape:
        raise ValueError(
            f"eps_hat shape {tuple(eps_hat.shape)} must match x_t {tuple(x_t.shape)}"
        )
    sqrt_ab = scheduler._extract(scheduler.sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_omb = scheduler._extract(
        scheduler.sqrt_one_minus_alphas_cumprod, t, x_t.shape
    )
    return (x_t - sqrt_omb * eps_hat) / sqrt_ab


def predict_eps_from_x0(
    scheduler: NoiseScheduler,
    x_t: torch.Tensor,
    t: torch.Tensor,
    x0_hat: torch.Tensor,
) -> torch.Tensor:
    """Closed-form x0-prediction → ε (matched training / sampling interface).

        ε̂ = (x_t − √ᾱ_t ẑ0) / √(1−ᾱ_t)
    """
    if x0_hat.shape != x_t.shape:
        raise ValueError(
            f"x0_hat shape {tuple(x0_hat.shape)} must match x_t {tuple(x_t.shape)}"
        )
    sqrt_ab = scheduler._extract(scheduler.sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_omb = scheduler._extract(
        scheduler.sqrt_one_minus_alphas_cumprod, t, x_t.shape
    )
    return (x_t - sqrt_ab * x0_hat) / sqrt_omb.clamp_min(1e-12)


def normalize_prediction_type(prediction_type: str) -> str:
    """Map config aliases to canonical ``'eps'`` or ``'x0'``."""
    key = str(prediction_type).strip().lower()
    if key in {"eps", "epsilon", "noise"}:
        return "eps"
    if key in {"x0", "x_0", "z0", "z_0", "sample"}:
        return "x0"
    raise ValueError(
        f"Unknown prediction_type={prediction_type!r}; use 'eps' or 'x0'."
    )
