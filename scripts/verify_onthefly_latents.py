"""Verify on-the-fly latent encoding (Phase 3).

Loads a frozen VAE and encodes one batch of images → scaled latents.
Does **not** write a latent cache to disk.

Examples:
  # Synthetic batch (no CelebA needed)
  python scripts/verify_onthefly_latents.py \\
    --checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt

  # Real CelebA batch
  python scripts/verify_onthefly_latents.py \\
    --config configs/onthefly_latent.yaml \\
    --use-celeba --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.onthefly_latent import OnTheFlyLatentEncoder, batch_to_images
from latentsr.utils.config import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify on-the-fly latent encoding.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/onthefly_latent.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override VAE checkpoint path.",
    )
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--use-celeba",
        action="store_true",
        help="Encode one real CelebA batch instead of synthetic images.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config.exists() else {}

    checkpoint = Path(
        args.checkpoint
        or config.get("vae_checkpoint")
        or "outputs/vae/checkpoints/checkpoint_epoch_050.pt"
    )
    latent_scale = float(
        args.latent_scale
        if args.latent_scale is not None
        else config.get("latent_scale", 1.0)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else config.get("batch_size", 8)
    )
    device = get_device(args.device or str(config.get("device", "auto")))
    latent_channels = int(config.get("latent_channels", 4))
    latent_size = int(config.get("latent_size", 32))
    image_size = int(config.get("image_size", 128))

    encoder = OnTheFlyLatentEncoder.from_checkpoint(
        checkpoint, latent_scale=latent_scale, map_location=device
    ).to(device)

    if args.use_celeba:
        from latentsr.datasets.celeba import get_celeba_dataloaders

        train_loader, _ = get_celeba_dataloaders(
            batch_size=batch_size,
            data_dir=config.get("data_dir", "data/raw"),
            image_size=image_size,
            num_workers=0,
            pin_memory=False,
            download=args.download,
        )
        batch = next(iter(train_loader))
        images = batch_to_images(batch)
        print(f"CelebA batch images: {tuple(images.shape)}")
    else:
        images = torch.rand(batch_size, 3, image_size, image_size)
        print(f"Synthetic batch images: {tuple(images.shape)}")

    z = encoder.encode_batch(images, device=device)
    expected = (images.size(0), latent_channels, latent_size, latent_size)
    assert z.shape == expected, f"expected {expected}, got {tuple(z.shape)}"
    assert z.dtype == torch.float32 or z.dtype == torch.float16
    assert z.device.type == device.type

    # Changing scale must not require any cache rebuild — just a float update.
    encoder.set_latent_scale(latent_scale * 2.0)
    z2 = encoder(images.to(device))
    assert torch.allclose(z2, z * 2.0, rtol=1e-4, atol=1e-4)

    encoder.set_latent_scale(latent_scale)
    print(f"VAE: {checkpoint}")
    print(f"latent_scale: {latent_scale}")
    print(f"z shape: {tuple(z.shape)}  dtype={z.dtype}  device={z.device}")
    print(f"z mean={z.mean().item():.4f}  std={z.std().item():.4f}")
    print("Phase 3 on-the-fly latent path OK (no disk cache).")


if __name__ == "__main__":
    main()
