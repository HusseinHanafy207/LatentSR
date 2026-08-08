"""
Reconstruct / interpolate CelebA images with a trained VAE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from latentsr.datasets.celeba import get_celeba_dataloaders
from latentsr.utils.config import get_device, load_config
from latentsr.vae.checkpointing import load_vae_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE reconstruction / interpolation.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a VAE checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae_celeba.yaml"),
        help="Config used for data paths / image size (fallback if ckpt lacks config).",
    )
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vae/samples"),
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Also save latent interpolations between consecutive pairs.",
    )
    parser.add_argument("--steps", type=int, default=8, help="Interpolation steps.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download CelebA if missing (default: false).",
    )
    return parser.parse_args()


@torch.no_grad()
def save_reconstructions(
    model: torch.nn.Module,
    images: torch.Tensor,
    output_path: Path,
) -> None:
    recon = model.reconstruct(images, use_mean=True)
    grid = make_grid(
        torch.cat([images.cpu(), recon.cpu()], dim=0),
        nrow=images.size(0),
        padding=2,
    )
    save_image(grid, output_path)


@torch.no_grad()
def save_interpolations(
    model: torch.nn.Module,
    images: torch.Tensor,
    output_path: Path,
    steps: int,
) -> None:
    mu, _ = model.encode(images)
    rows: list[torch.Tensor] = []
    n = images.size(0)
    for i in range(0, n - 1, 2):
        z0, z1 = mu[i], mu[i + 1]
        alphas = torch.linspace(0.0, 1.0, steps, device=mu.device)
        zs = torch.stack([(1 - a) * z0 + a * z1 for a in alphas], dim=0)
        rows.append(model.decode(zs).cpu())
    if not rows:
        return
    grid = make_grid(torch.cat(rows, dim=0), nrow=steps, padding=2)
    save_image(grid, output_path)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, checkpoint = load_vae_checkpoint(args.checkpoint, map_location=device)
    model.to(device)

    ckpt_config = checkpoint.get("config") or {}
    file_config = load_config(args.config) if args.config.exists() else {}
    config = {**file_config, **ckpt_config}

    _, val_loader = get_celeba_dataloaders(
        batch_size=max(args.num_images, 2),
        data_dir=config.get("data_dir", "data/raw"),
        image_size=int(config.get("image_size", 128)),
        num_workers=0,
        pin_memory=False,
        download=args.download,
    )
    images, _ = next(iter(val_loader))
    images = images[: args.num_images].to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recon_path = args.output_dir / "recon_eval.png"
    save_reconstructions(model, images, recon_path)
    print(f"Wrote {recon_path}")

    if args.interpolate:
        interp_path = args.output_dir / "latent_interp.png"
        save_interpolations(model, images, interp_path, steps=args.steps)
        print(f"Wrote {interp_path}")


if __name__ == "__main__":
    main()
