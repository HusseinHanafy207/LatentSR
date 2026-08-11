"""Sample images from a trained Latent Diffusion Model (Phase 5).

Pipeline:
  noise → latent DDPM → z_scaled → decode_scaled → RGB grid

Examples:
  python scripts/sample_ldm.py \\
    --checkpoint outputs/latent_ddpm/checkpoints/latest.pt \\
    --num-samples 16

  # Colab
  python scripts/sample_ldm.py \\
    --checkpoint /content/drive/MyDrive/LatentSR/outputs/latent_ddpm/checkpoints/latest.pt \\
    --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.diffusion.sample import load_ldm_components, sample_ldm, save_ldm_grid
from latentsr.utils.config import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from LatentSR LDM.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/LatentSR/outputs/latent_ddpm/checkpoints/latest.pt"
        ),
        help="Latent DDPM checkpoint (.pt).",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
        help="Override VAE path (default: read from DDPM checkpoint metadata).",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: next to checkpoint under samples/).",
    )
    parser.add_argument(
        "--latent-scale",
        type=float,
        default=None,
        help="Override latent_scale from checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)

    ddpm, vae, meta = load_ldm_components(
        args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        map_location=device,
    )
    ddpm.to(device)
    vae.to(device)

    config = meta["config"]
    latent_scale = (
        float(args.latent_scale)
        if args.latent_scale is not None
        else float(meta["latent_scale"])
    )
    latent_channels = int(config.get("latent_channels", 4))
    latent_size = int(config.get("latent_size", 32))

    print(f"DDPM epoch: {meta.get('ddpm_epoch')}")
    print(f"VAE: {meta['vae_checkpoint']}")
    print(f"latent_scale: {latent_scale}")
    print(f"Sampling {args.num_samples} images on {device} …")

    images = sample_ldm(
        ddpm,
        vae,
        args.num_samples,
        latent_channels=latent_channels,
        latent_size=latent_size,
        latent_scale=latent_scale,
        device=device,
        show_progress=True,
    )

    if args.output is not None:
        output = args.output
    else:
        # Prefer Drive sample_dir from training config when present.
        sample_dir = config.get("sample_dir")
        if sample_dir:
            output = Path(sample_dir) / "ldm_samples.png"
        else:
            output = Path(args.checkpoint).parent.parent / "samples" / "ldm_samples.png"

    path = save_ldm_grid(images, output, nrow=args.nrow)
    print(f"Wrote {path}")
    print("Phase 5: inspect the grid — faces should be recognizable (not noise).")


if __name__ == "__main__":
    main()
