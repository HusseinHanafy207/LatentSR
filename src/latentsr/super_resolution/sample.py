"""Conditional reverse sampling in latent space (Phase 8/9)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tqdm.auto import tqdm

from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.vae.latent import decode_scaled
from latentsr.vae.vae import VAE

# torch.Generator.manual_seed is only well-defined for 32-bit non-negative ints.
_SEED_MOD = 2**31 - 1
# Distinct from the x_T salt (0) so t=0 cannot collide with the initial draw.
_STEP_SALT = 10_007


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


@torch.no_grad()
def sample_conditional_latents(
    model: ConditionalLatentDDPM,
    z_lr: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    val_indices: Sequence[int] | None = None,
    noise_seed: int | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Denoise from noise to ``z_hr``, conditioned on ``z_lr`` (no clamping).

    Pass ``noise_seed`` + ``val_indices`` to seed both ``x_T`` and every
    reverse-step ``z`` so pairing does not depend on batch size. The scheduler
    already accepts per-step ``noise``; left unset it uses ``torch.randn_like``.
    """
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
    iterator = (
        tqdm(timesteps, desc="sr-sampling", leave=False) if show_progress else timesteps
    )
    for t in iterator:
        t_batch = torch.full((z.shape[0],), t, device=device, dtype=torch.long)
        noise_pred = model.predict_noise(z, t_batch, z_lr)
        step_noise = None
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
) -> torch.Tensor:
    """``z_lr`` → conditional DDPM → decode_scaled → RGB ``[0, 1]``."""
    z_hr = sample_conditional_latents(
        model,
        z_lr,
        noise=noise,
        val_indices=val_indices,
        noise_seed=noise_seed,
        show_progress=show_progress,
    )
    images = decode_scaled(vae, z_hr, latent_scale=latent_scale)
    return images.clamp(0.0, 1.0)
