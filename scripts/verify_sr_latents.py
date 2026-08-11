"""Verify on-the-fly SR latent pairs (Phase 7).

Shows a grid:
  HR | decode(z_hr) | decode(z_lr) | bicubic(LR→HR)

Examples:
  python scripts/verify_sr_latents.py --config configs/onthefly_sr_latent.yaml --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from latentsr.datasets.onthefly_sr_latent import OnTheFlySRLatentEncoder, upsample_bicubic
from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.utils.config import get_device, load_config
from latentsr.vae.latent import decode_scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SR latent pairs.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/onthefly_sr_latent.yaml")
    )
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device or str(config.get("device", "auto")))

    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))
    latent_scale = float(config.get("latent_scale", 1.0))
    latent_channels = int(config.get("latent_channels", 4))
    latent_size = int(config.get("latent_size", 32))

    _, val_loader = get_sr_pair_dataloaders(
        batch_size=max(args.num_pairs, 1),
        data_dir=config.get("data_dir", "data/raw"),
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=0,
        pin_memory=False,
        download=args.download,
    )
    lr, hr = next(iter(val_loader))
    lr = lr[: args.num_pairs]
    hr = hr[: args.num_pairs]

    encoder = OnTheFlySRLatentEncoder.from_checkpoint(
        config["vae_checkpoint"],
        latent_scale=latent_scale,
        hr_size=hr_size,
        map_location=device,
    ).to(device)

    z_lr, z_hr = encoder.encode_batch((lr, hr), device=device)
    expected = (args.num_pairs, latent_channels, latent_size, latent_size)
    assert z_lr.shape == expected, z_lr.shape
    assert z_hr.shape == expected, z_hr.shape

    recon_hr = decode_scaled(encoder.vae, z_hr, latent_scale)
    recon_lr = decode_scaled(encoder.vae, z_lr, latent_scale)
    bicubic = upsample_bicubic(lr.to(device), hr_size)

    grid = make_grid(
        torch.cat(
            [hr.to(device).cpu(), recon_hr.cpu(), recon_lr.cpu(), bicubic.cpu()],
            dim=0,
        ),
        nrow=args.num_pairs,
        padding=2,
    )

    sample_dir = Path(config.get("sample_dir", "outputs/sr_latents"))
    output = args.output or (sample_dir / "sr_latent_pairs.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output)

    mse_hr = F.mse_loss(recon_hr, hr.to(device)).item()
    print(f"z_lr shape: {tuple(z_lr.shape)}")
    print(f"z_hr shape: {tuple(z_hr.shape)}")
    print(f"decode(z_hr) vs HR MSE: {mse_hr:.6f}")
    print(f"Wrote {output}")
    print("Rows: HR | decode(z_hr) | decode(z_lr) | bicubic(LR→HR)")


if __name__ == "__main__":
    main()
