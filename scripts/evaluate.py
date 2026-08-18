"""Evaluate LatentSR vs bicubic on CelebA val.

Examples:
  python scripts/evaluate.py \\
    --checkpoint outputs/latent_sr/checkpoints/latest.pt \\
    --vae-checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt \\
    --config configs/eval_sr.yaml \\
    --num-images 64 --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.metrics.evaluate_sr import evaluate_sr
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LatentSR vs bicubic (PSNR/SSIM/LPIPS)."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="CelebA root (folder that contains celeba/). Overrides config/checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write metrics + grids (default: config output_dir).",
    )
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute LPIPS (requires: pip install lpips).",
    )
    parser.add_argument(
        "--include-soft-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add decode(z_lr) to the comparison grid (always scored in per_image.csv).",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    torch.manual_seed(int(args.seed if args.seed is not None else config.get("seed", 42)))

    model, vae, meta = load_sr_components(
        args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        map_location=device,
    )
    ckpt_cfg = meta.get("config") or {}
    merged = {**ckpt_cfg, **config}

    hr_size = int(merged.get("hr_size", meta["hr_size"]))
    lr_size = int(merged.get("lr_size", meta["lr_size"]))
    latent_scale = float(merged.get("latent_scale", meta["latent_scale"]))
    num_images = int(
        args.num_images
        if args.num_images is not None
        else merged.get("num_images", 64)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else merged.get("batch_size", 4)
    )
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else merged.get("output_dir", "outputs/eval")
    )
    data_dir = (
        str(args.data_dir)
        if args.data_dir is not None
        else merged.get("data_dir", "data/raw")
    )

    print(f"SR epoch: {meta.get('sr_epoch')}")
    print(f"VAE: {meta['vae_checkpoint']}")
    print(f"latent_scale: {latent_scale}")
    print(f"data_dir: {data_dir}")
    print(f"Evaluating {num_images} val images on {device} …")
    print(
        f"Per-image noise seed={int(args.seed)} "
        "(x_T and every reverse step; independent of batch size; "
        "use the same seed for paired runs)."
    )
    print("Note: each image runs a full reverse diffusion chain (slow on CPU).")

    _, val_loader = get_sr_pair_dataloaders(
        batch_size=batch_size,
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(merged.get("num_workers", 0)),
        pin_memory=bool(merged.get("pin_memory", False)),
        download=args.download,
    )

    try:
        result = evaluate_sr(
            model,
            vae,
            val_loader,
            device=device,
            num_images=num_images,
            hr_size=hr_size,
            latent_scale=latent_scale,
            compute_lpips=bool(args.lpips),
            include_soft_decode=bool(args.include_soft_decode),
            show_progress=True,
            grid_images=int(args.grid_images),
            output_dir=output_dir,
            noise_seed=int(args.seed),
        )
    except ImportError as exc:
        if args.lpips:
            raise SystemExit(
                f"{exc}\nRe-run with --no-lpips or: pip install lpips"
            ) from exc
        raise

    print()
    print(result["table"])
    print()
    print(f"Images scored: {result['num_images']}")
    print(f"Wrote metrics + grids under: {output_dir}")
    print(f"Per-image scores: {output_dir / 'per_image.csv'}")
    if "grid_path" in result:
        print(f"Comparison grid: {result['grid_path']}")
    print(
        "Higher PSNR/SSIM is better; lower LPIPS is better. "
        "LatentSR should beat bicubic on LPIPS (and often look sharper even if PSNR is close)."
    )


if __name__ == "__main__":
    main()
