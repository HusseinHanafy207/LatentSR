"""Verify a frozen VAE checkpoint (Phase 2 contract).

Examples:
  python scripts/verify_frozen_vae.py \\
    --checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from latentsr.utils.config import get_device
from latentsr.vae.latent import (
    decode_scaled,
    encode_scaled,
    estimate_latent_scale,
    is_frozen,
    load_frozen_vae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify frozen VAE + latent_scale.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/vae/checkpoints/checkpoint_epoch_050.pt"),
    )
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vae/samples/frozen_roundtrip.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    vae, ckpt = load_frozen_vae(args.checkpoint, map_location=device)
    vae.to(device)

    print(f"Loaded epoch {ckpt.get('epoch')} from {args.checkpoint}")
    print(f"Frozen: {is_frozen(vae)}")
    assert is_frozen(vae), "VAE parameters must have requires_grad=False"

    # Synthetic batch keeps this script runnable without CelebA.
    x = torch.rand(args.batch_size, 3, 128, 128, device=device)
    z = encode_scaled(vae, x, args.latent_scale)
    x_hat = decode_scaled(vae, z, args.latent_scale)
    x_mu = vae.reconstruct(x, use_mean=True)

    assert z.shape == (args.batch_size, vae.latent_channels, 32, 32), z.shape
    assert x_hat.shape == x.shape
    assert torch.allclose(x_hat, x_mu, atol=1e-5), (
        "encode_scaled/decode_scaled at scale=1 must match reconstruct(mu)"
    )

    suggested = estimate_latent_scale(vae, x)
    print(f"latent_scale (config): {args.latent_scale}")
    print(f"suggested 1/std(mu):   {suggested:.6f}")
    print(f"z shape: {tuple(z.shape)}  |  recon MSE vs input: {(x_hat - x).pow(2).mean().item():.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(torch.cat([x.cpu(), x_hat.cpu()], dim=0), nrow=args.batch_size)
    save_image(grid, args.output)
    print(f"Wrote {args.output}")
    print("Phase 2 contract OK.")


if __name__ == "__main__":
    main()
