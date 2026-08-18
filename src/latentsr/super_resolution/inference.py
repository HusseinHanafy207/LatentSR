"""LR → HR inference helpers.

Pipeline:
    LR_32 → bicubic upsample → encode_scaled → z_lr
    noise → ConditionalLatentDDPM(· | z_lr) → z_hr_scaled
    decode_scaled(z_hr_scaled) → HR_128
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from latentsr.datasets.onthefly_sr_latent import upsample_bicubic
from latentsr.super_resolution.condition import (
    ConditionalLatentDDPM,
    load_conditioned_latent_ddpm_checkpoint,
)
from latentsr.super_resolution.sample import sample_sr_images
from latentsr.vae.latent import decode_scaled, encode_scaled, load_frozen_vae
from latentsr.vae.vae import VAE


def load_sr_components(
    sr_checkpoint: str | Path,
    *,
    vae_checkpoint: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[ConditionalLatentDDPM, VAE, dict[str, Any]]:
    """Load conditioned SR DDPM + frozen VAE; VAE path falls back to ckpt metadata."""
    model, checkpoint = load_conditioned_latent_ddpm_checkpoint(
        sr_checkpoint, map_location=map_location
    )
    config = checkpoint.get("config") or {}
    vae_path = (
        vae_checkpoint
        or checkpoint.get("vae_checkpoint")
        or config.get("vae_checkpoint")
    )
    if not vae_path:
        raise ValueError(
            "VAE checkpoint not found in SR metadata; pass --vae-checkpoint."
        )
    vae_path = Path(vae_path)
    if not vae_path.is_file():
        raise FileNotFoundError(
            f"VAE checkpoint not found:\n  {vae_path}\n\n"
            "The SR checkpoint stores the VAE path from the machine that trained it "
            "(often a Kaggle path like /kaggle/working/...). That file is not on Colab. "
            "Pass the Drive checkpoint explicitly, for example:\n"
            "  --vae-checkpoint /content/drive/MyDrive/LatentSR/outputs/vae/checkpoints/checkpoint_epoch_050.pt\n"
            "VAE-SR:\n"
            "  --vae-checkpoint /content/drive/MyDrive/LatentSR/outputs/vae_sr/checkpoints/latest.pt\n"
            "Also use --config configs/eval_sr_colab.yaml and --data-dir /content/data/raw."
        )
    latent_scale = float(
        checkpoint.get("latent_scale", config.get("latent_scale", 1.0))
    )
    vae, _vae_ckpt = load_frozen_vae(vae_path, map_location=map_location)
    meta = {
        "config": config,
        "latent_scale": latent_scale,
        "vae_checkpoint": str(vae_path),
        "sr_epoch": checkpoint.get("epoch"),
        "hr_size": int(config.get("hr_size", 128)),
        "lr_size": int(config.get("lr_size", 32)),
    }
    return model, vae, meta


@torch.no_grad()
def encode_lr_latents(
    vae: VAE,
    lr: torch.Tensor,
    *,
    hr_size: int = 128,
    latent_scale: float = 1.0,
) -> torch.Tensor:
    """``LR → upsample_hr → encode_scaled`` (same conditioning as training)."""
    lr_up = upsample_bicubic(lr, hr_size)
    return encode_scaled(vae, lr_up, latent_scale=latent_scale)


@torch.no_grad()
def super_resolve(
    model: ConditionalLatentDDPM,
    vae: VAE,
    lr: torch.Tensor,
    *,
    hr_size: int = 128,
    latent_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    val_indices: list[int] | None = None,
    noise_seed: int | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Super-resolve an LR batch ``(B, 3, lr, lr)`` → HR ``(B, 3, hr, hr)``."""
    z_lr = encode_lr_latents(
        vae, lr, hr_size=hr_size, latent_scale=latent_scale
    )
    return sample_sr_images(
        model,
        vae,
        z_lr,
        latent_scale=latent_scale,
        noise=noise,
        val_indices=val_indices,
        noise_seed=noise_seed,
        show_progress=show_progress,
    )


def prepare_lr_batch(
    images: torch.Tensor,
    *,
    lr_size: int = 32,
    hr_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize inputs to LR tensors; return ``(lr, optional_hr_for_grid)``.

    - Already ``lr_size``: use as LR (no GT HR).
    - Already ``hr_size``: bicubic-downsample to LR; keep original as HR.
    - Other sizes: resize to ``hr_size`` then downsample (HR = resized).
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected (B, 3, H, W) RGB, got {tuple(images.shape)}")

    h, w = images.shape[-2:]
    if (h, w) == (lr_size, lr_size):
        return images.clamp(0.0, 1.0), None
    if (h, w) == (hr_size, hr_size):
        hr = images.clamp(0.0, 1.0)
        lr = F.interpolate(
            hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        return lr, hr
    hr = F.interpolate(
        images, size=(hr_size, hr_size), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    lr = F.interpolate(
        hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    return lr, hr


def save_sr_comparison_grid(
    lr: torch.Tensor,
    pred: torch.Tensor,
    *,
    hr: torch.Tensor | None = None,
    output_path: str | Path,
    hr_size: int = 128,
    include_soft_decode: bool = False,
    soft: torch.Tensor | None = None,
) -> Path:
    """Save side-by-side grid: nearest(LR) | bicubic | LatentSR [| soft] [| HR]."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = lr.shape[0]
    lr_nn = F.interpolate(lr, size=(hr_size, hr_size), mode="nearest")
    bicubic = upsample_bicubic(lr, hr_size)
    rows = [lr_nn.cpu(), bicubic.cpu(), pred.cpu().clamp(0.0, 1.0)]
    if include_soft_decode and soft is not None:
        rows.append(soft.cpu().clamp(0.0, 1.0))
    if hr is not None:
        rows.append(hr.cpu().clamp(0.0, 1.0))

    grid = make_grid(torch.cat(rows, dim=0), nrow=n, padding=2)
    save_image(grid, output_path)
    return output_path


@torch.no_grad()
def soft_decode_from_lr(
    vae: VAE,
    lr: torch.Tensor,
    *,
    hr_size: int = 128,
    latent_scale: float = 1.0,
) -> torch.Tensor:
    """Baseline: decode(z_lr) without diffusion (blurry upper bound of VAE alone)."""
    z_lr = encode_lr_latents(
        vae, lr, hr_size=hr_size, latent_scale=latent_scale
    )
    return decode_scaled(vae, z_lr, latent_scale=latent_scale).clamp(0.0, 1.0)
