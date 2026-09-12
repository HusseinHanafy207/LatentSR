"""Conditional reverse sampling in latent space."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tqdm.auto import tqdm

from generative_models.ddpm import NoiseScheduler

from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.vae.latent import decode_scaled
from latentsr.vae.vae import VAE

# torch.Generator.manual_seed is only well-defined for 32-bit non-negative ints.
_SEED_MOD = 2**31 - 1
# Distinct from the x_T salt (0) so t=0 cannot collide with the initial draw.
_STEP_SALT = 10_007
STEP_SALT = _STEP_SALT


def image_noise_seed(base_seed: int, val_index: int, *, salt: int = 0) -> int:
    """Deterministic per-image seed; independent of batch size and eval order."""
    return (int(base_seed) + 1_000_003 * (int(val_index) + 1) + int(salt)) % _SEED_MOD


def seeded_noise_like(
    reference: torch.Tensor,
    val_indices: Sequence[int],
    *,
    base_seed: int,
    salt: int = 0,
) -> torch.Tensor:
    """CPU Gaussian noise per ``val_index``, then moved onto ``reference``.

    Generating on CPU keeps VAE-1 vs VAE-SR pairing identical across devices.
    ``salt`` distinguishes ``x_T`` (0) from reverse-step noise (``_STEP_SALT * (t+1)``).
    """
    if reference.ndim < 1 or reference.shape[0] != len(val_indices):
        raise ValueError(
            f"Expected {len(val_indices)} leading samples, got shape "
            f"{tuple(reference.shape)}"
        )
    spatial = tuple(reference.shape[1:])
    chunks: list[torch.Tensor] = []
    for index in val_indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            image_noise_seed(base_seed, int(index), salt=salt)
        )
        chunks.append(torch.randn(spatial, generator=generator, dtype=torch.float32))
    return torch.stack(chunks, dim=0).to(device=reference.device, dtype=reference.dtype)


def predict_x0_from_eps(
    scheduler: NoiseScheduler,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_hat: torch.Tensor,
) -> torch.Tensor:
    """Closed-form ε-prediction x0 (no clamp; latents are unbounded).

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


def ddim_step(
    scheduler: NoiseScheduler,
    x_t: torch.Tensor,
    t: torch.Tensor,
    noise_pred: torch.Tensor,
    *,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """One reverse step using DDIM (Song et al., 2020).

    When ``eta == 0.0`` (default), the sampling trajectory is completely
    deterministic (Euler-like integration along the probability flow ODE).
    When ``eta == 1.0``, the variance matches the DDPM posterior variance.

    At timestep t=0, ``x_{t-1} = x_{-1} = x_0_pred``, exactly reaching
    the clean prediction.

    Args:
        scheduler: NoiseScheduler instance with precomputed cumprod schedules.
        x_t: Current noisy latents at timestep t, shape ``(B, C, H, W)``.
        t: Integer timesteps per batch element, shape ``(B,)``.
        noise_pred: Predicted noise epsilon_theta(x_t, t, cond), shape ``(B, C, H, W)``.
        eta: DDIM stochasticity multiplier in [0, 1]. 0.0 is deterministic DDIM.
        noise: Optional random Gaussian noise tensor for eta > 0.
    """
    if noise_pred.shape != x_t.shape:
        raise ValueError(
            f"noise_pred shape {tuple(noise_pred.shape)} must match "
            f"x_t shape {tuple(x_t.shape)}"
        )
    if t.shape[0] != x_t.shape[0]:
        raise ValueError(
            f"Batch size mismatch: x_t has batch {x_t.shape[0]}, t has {t.shape[0]}"
        )

    # 1. Closed-form estimate of clean x_0 from x_t and noise_pred
    x0_hat = predict_x0_from_eps(scheduler, x_t, t, noise_pred)

    # 2. Extract schedule parameters
    ab_t = scheduler._extract(scheduler.alphas_cumprod, t, x_t.shape)
    ab_prev = scheduler._extract(scheduler.alphas_cumprod_prev, t, x_t.shape)

    # 3. Compute sigma_t (variance of the stochastic component)
    if eta > 0.0:
        beta_t = scheduler._extract(scheduler.betas, t, x_t.shape)
        var = ((1.0 - ab_prev) / (1.0 - ab_t).clamp_min(1e-12)) * beta_t
        sigma_t = float(eta) * torch.sqrt(var.clamp_min(0.0))
        # Mask out noise at t=0 (where ab_prev == 1.0)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        sigma_t = sigma_t * nonzero_mask
    else:
        sigma_t = torch.zeros_like(ab_t)

    # 4. Direction pointing to x_t: sqrt(1 - ab_prev - sigma_t^2) * eps
    dir_xt_coeff = torch.sqrt((1.0 - ab_prev - sigma_t.pow(2)).clamp_min(0.0))
    dir_xt = dir_xt_coeff * noise_pred

    # 5. Combine deterministic parts: sqrt(ab_prev) * x0_hat + dir_xt
    x_prev = torch.sqrt(ab_prev) * x0_hat + dir_xt

    # 6. Add noise if eta > 0
    if eta > 0.0:
        if noise is None:
            noise = torch.randn_like(x_t)
        elif noise.shape != x_t.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must match x_t shape {tuple(x_t.shape)}"
            )
        x_prev = x_prev + sigma_t * noise

    return x_prev


@torch.no_grad()
def sample_conditional_latents(
    model: ConditionalLatentDDPM,
    z_lr: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    val_indices: Sequence[int] | None = None,
    noise_seed: int | None = None,
    show_progress: bool = True,
    sampler: str = "ddpm",
    ddim_eta: float = 0.0,
) -> torch.Tensor:
    """Denoise from noise to ``z_hr``, conditioned on ``z_lr`` (no clamping).

    Args:
        model: ConditionalLatentDDPM.
        z_lr: Conditioning latents, shape ``(B, C, H, W)``.
        noise: Optional initial noise x_T. Drawn via ``seeded_noise_like`` or randn.
        val_indices: Per-image index for deterministic noise.
        noise_seed: Base seed for reproducible noise generation.
        show_progress: Display tqdm progress bar over timesteps.
        sampler: "ddpm" (stochastic ancestral) or "ddim" (deterministic ODE if ddim_eta=0).
        ddim_eta: DDIM stochasticity parameter (0.0 = deterministic ODE).
    """
    sampler = sampler.lower().strip()
    if sampler not in ("ddpm", "ddim"):
        raise ValueError(f"Unknown sampler '{sampler}', expected 'ddpm' or 'ddim'")

    device = z_lr.device
    model.eval()
    if noise_seed is not None:
        if val_indices is None:
            raise ValueError("val_indices is required when noise_seed is set")
        if len(val_indices) != z_lr.shape[0]:
            raise ValueError(
                f"val_indices length {len(val_indices)} != batch {z_lr.shape[0]}"
            )
        if noise is None:
            noise = seeded_noise_like(
                z_lr, val_indices, base_seed=noise_seed, salt=0
            )
    if noise is None:
        z = torch.randn_like(z_lr)
    else:
        if noise.shape != z_lr.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} != z_lr shape {tuple(z_lr.shape)}"
            )
        z = noise.to(device=device, dtype=z_lr.dtype)
    timesteps = range(model.num_timesteps - 1, -1, -1)
    desc = "sr-sampling (ddim)" if sampler == "ddim" else "sr-sampling (ddpm)"
    iterator = (
        tqdm(
            timesteps,
            desc=desc,
            unit="t",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        if show_progress
        else timesteps
    )
    for t in iterator:
        t_batch = torch.full((z.shape[0],), t, device=device, dtype=torch.long)
        noise_pred = model.predict_noise(z, t_batch, z_lr)
        step_noise = None
        if sampler == "ddim":
            if ddim_eta > 0.0 and noise_seed is not None:
                assert val_indices is not None
                step_noise = seeded_noise_like(
                    z,
                    val_indices,
                    base_seed=noise_seed,
                    salt=_STEP_SALT * (int(t) + 1),
                )
            z = ddim_step(
                model.scheduler,
                z,
                t_batch,
                noise_pred,
                eta=ddim_eta,
                noise=step_noise,
            )
        else:
            if noise_seed is not None:
                assert val_indices is not None
                step_noise = seeded_noise_like(
                    z,
                    val_indices,
                    base_seed=noise_seed,
                    salt=_STEP_SALT * (int(t) + 1),
                )
            z = model.scheduler.p_sample_step(z, t_batch, noise_pred, noise=step_noise)
    return z


@torch.no_grad()
def sample_sr_images(
    model: ConditionalLatentDDPM,
    vae: VAE,
    z_lr: torch.Tensor,
    *,
    latent_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    val_indices: Sequence[int] | None = None,
    noise_seed: int | None = None,
    show_progress: bool = True,
    sampler: str = "ddpm",
    ddim_eta: float = 0.0,
) -> torch.Tensor:
    """``z_lr`` → conditional DDPM → decode_scaled → RGB ``[0, 1]``."""
    z_hr = sample_conditional_latents(
        model,
        z_lr,
        noise=noise,
        val_indices=val_indices,
        noise_seed=noise_seed,
        show_progress=show_progress,
        sampler=sampler,
        ddim_eta=ddim_eta,
    )
    images = decode_scaled(vae, z_hr, latent_scale=latent_scale)
    return images.clamp(0.0, 1.0)
