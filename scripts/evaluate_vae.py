"""Evaluate the frozen VAE bottleneck (no diffusion).

Scores bicubic, decode(z_lr), and decode(z_hr) vs HR, plus latent statistics
and a suggested latent_scale = 1 / std(mu_HR).

Examples:
  python scripts/evaluate_vae.py \\
    --vae-checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt \\
    --config configs/eval_vae.yaml \\
    --num-images 64 --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.metrics.evaluate_vae import evaluate_vae
from latentsr.utils.config import get_device, load_config
from latentsr.vae.latent import is_frozen, load_frozen_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VAE bottleneck: decode(z_hr) / decode(z_lr) vs HR."
    )
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_vae.yaml"))
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grid-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--latent-scale", type=float, default=None)
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

    vae_path = args.vae_checkpoint or config.get("vae_checkpoint")
    if not vae_path:
        raise SystemExit("Pass --vae-checkpoint or set vae_checkpoint in the config.")

    vae, ckpt = load_frozen_vae(vae_path, map_location=device)
    if not is_frozen(vae):
        raise SystemExit("VAE must be frozen (load_frozen_vae failed).")

    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))
    latent_scale = float(
        args.latent_scale
        if args.latent_scale is not None
        else config.get("latent_scale", 1.0)
    )
    num_images = int(
        args.num_images if args.num_images is not None else config.get("num_images", 64)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else config.get("batch_size", 8)
    )
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else config.get("output_dir", "outputs/eval_vae")
    )

    print(f"VAE epoch: {ckpt.get('epoch')}")
    print(f"VAE: {vae_path}")
    print(f"latent_scale (used for encode/decode): {latent_scale}")
    print(f"Evaluating {num_images} val images on {device} (no diffusion) …")

    _, val_loader = get_sr_pair_dataloaders(
        batch_size=batch_size,
        data_dir=config.get("data_dir", "data/raw"),
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    try:
        result = evaluate_vae(
            vae,
            val_loader,
            device=device,
            num_images=num_images,
            hr_size=hr_size,
            latent_scale=latent_scale,
            compute_lpips=bool(args.lpips),
            show_progress=True,
            grid_images=int(args.grid_images),
            output_dir=output_dir,
        )
    except ImportError as exc:
        if args.lpips:
            raise SystemExit(
                f"{exc}\nRe-run with --no-lpips or: pip install lpips"
            ) from exc
        raise

    stats = result["latent_stats"]
    print()
    print(result["table"])
    print()
    print(f"Images scored: {result['num_images']}")
    print(
        f"std(mu_hr)={stats['mu_hr_std']:.6f}  |  "
        f"std(mu_lr)={stats['mu_lr_std']:.6f}"
    )
    print(
        f"suggested latent_scale = 1/std(mu_hr) = {stats['latent_scale_suggested']:.6f}"
    )
    print(
        f"||z_lr - z_hr|| MSE: {stats['z_mse']['mean']:.6f}  |  "
        f"cosine(z_lr, z_hr): {stats['z_cosine']['mean']:.4f}"
    )
    print()
    print(result["bottleneck_note"])
    print()
    print(f"Wrote metrics + grids under: {output_dir}")
    print(f"Per-image scores: {output_dir / 'per_image.csv'}")
    if "grid_path" in result:
        print(f"Comparison grid: {result['grid_path']}")
    print(
        "Higher PSNR/SSIM is better; lower LPIPS / edge_mae / freq_* is better. "
        "decode(z_hr) is the pixel upper bound of this latent. "
        "Store suggested latent_scale in later diffusion configs."
    )


if __name__ == "__main__":
    main()
