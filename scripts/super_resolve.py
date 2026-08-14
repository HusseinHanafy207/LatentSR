"""Super-resolve LR faces with a trained LatentSR checkpoint.

Examples:
  # CelebA val comparison grid: nearest(LR) | bicubic | LatentSR | HR
  python scripts/super_resolve.py \\
    --checkpoint outputs/latent_sr/checkpoints/latest.pt \\
    --from-celeba --num-images 8 --no-download

  # Single LR / HR image (32 or 128); writes comparison PNG + SR PNG
  python scripts/super_resolve.py \\
    --checkpoint outputs/latent_sr/checkpoints/latest.pt \\
    --input path/to/face.png --output outputs/latent_sr/samples/sr_out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.super_resolution.inference import (
    load_sr_components,
    prepare_lr_batch,
    save_sr_comparison_grid,
    soft_decode_from_lr,
    super_resolve,
)
from latentsr.utils.config import get_device, load_config


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LatentSR inference: LR → HR (Phase 9)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Conditioned LatentSR checkpoint (.pt).",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
        help="Override VAE path (default: read from SR checkpoint metadata).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML for data_dir / sample_dir defaults.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="LR/HR image file or directory of images.",
    )
    parser.add_argument(
        "--from-celeba",
        action="store_true",
        help="Sample val CelebA pairs and write a comparison grid.",
    )
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (PNG). Default under sample_dir.",
    )
    parser.add_argument(
        "--include-soft-decode",
        action="store_true",
        help="Also show decode(z_lr) row in the comparison grid.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--latent-scale",
        type=float,
        default=None,
        help="Override latent_scale from checkpoint.",
    )
    return parser.parse_args()


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        return to_tensor(img.convert("RGB"))


def _collect_input_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {input_path}")
        return paths
    raise FileNotFoundError(f"Input not found: {input_path}")


def _default_output(
    args: argparse.Namespace,
    config: dict,
    *,
    stem: str,
) -> Path:
    if args.output is not None:
        return args.output
    sample_dir = config.get("sample_dir")
    if sample_dir:
        return Path(sample_dir) / stem
    return Path(args.checkpoint).parent.parent / "samples" / stem


def main() -> None:
    args = parse_args()
    if bool(args.input) == bool(args.from_celeba):
        raise SystemExit("Specify exactly one of --input or --from-celeba.")

    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device)
    torch.manual_seed(args.seed)

    model, vae, meta = load_sr_components(
        args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        map_location=device,
    )
    model.to(device)
    vae.to(device)

    latent_scale = (
        float(args.latent_scale)
        if args.latent_scale is not None
        else float(meta["latent_scale"])
    )
    hr_size = int(meta["hr_size"])
    lr_size = int(meta["lr_size"])
    ckpt_config = meta.get("config") or {}
    # Prefer optional CLI config overrides for data/sample dirs.
    merged = {**ckpt_config, **config}

    print(f"SR epoch: {meta.get('sr_epoch')}")
    print(f"VAE: {meta['vae_checkpoint']}")
    print(f"latent_scale: {latent_scale}")
    print(f"Device: {device}")

    if args.from_celeba:
        _, val_loader = get_sr_pair_dataloaders(
            batch_size=max(args.num_images, 1),
            data_dir=merged.get("data_dir", "data/raw"),
            hr_size=hr_size,
            lr_size=lr_size,
            num_workers=0,
            pin_memory=False,
            download=args.download,
        )
        lr, hr = next(iter(val_loader))
        lr = lr[: args.num_images].to(device)
        hr = hr[: args.num_images].to(device)
        print(f"Super-resolving {lr.shape[0]} CelebA val pairs …")
        pred = super_resolve(
            model,
            vae,
            lr,
            hr_size=hr_size,
            latent_scale=latent_scale,
            show_progress=True,
        )
        soft = None
        if args.include_soft_decode:
            soft = soft_decode_from_lr(
                vae, lr, hr_size=hr_size, latent_scale=latent_scale
            )
        out = _default_output(args, merged, stem="sr_celeba_compare.png")
        path = save_sr_comparison_grid(
            lr,
            pred,
            hr=hr,
            output_path=out,
            hr_size=hr_size,
            include_soft_decode=args.include_soft_decode,
            soft=soft,
        )
        print(f"Wrote {path}")
        print(
            "Rows: nearest(LR) | bicubic | LatentSR"
            + (" | soft(decode z_lr)" if args.include_soft_decode else "")
            + " | HR"
        )
        return

    paths = _collect_input_paths(args.input)  # type: ignore[arg-type]
    paths = paths[: args.num_images]
    images = torch.stack([_load_rgb_tensor(p) for p in paths])
    lr, hr = prepare_lr_batch(images, lr_size=lr_size, hr_size=hr_size)
    lr = lr.to(device)
    if hr is not None:
        hr = hr.to(device)

    print(f"Super-resolving {lr.shape[0]} image(s) …")
    pred = super_resolve(
        model,
        vae,
        lr,
        hr_size=hr_size,
        latent_scale=latent_scale,
        show_progress=True,
    )
    soft = None
    if args.include_soft_decode:
        soft = soft_decode_from_lr(
            vae, lr, hr_size=hr_size, latent_scale=latent_scale
        )

    compare_out = _default_output(args, merged, stem="sr_compare.png")
    if args.output is not None and args.output.suffix.lower() in IMAGE_EXTS:
        # User asked for a specific file: write SR image there and compare beside it.
        sr_out = args.output
        compare_out = args.output.with_name(args.output.stem + "_compare.png")
    else:
        sr_out = compare_out.with_name("sr_pred.png")

    sr_out.parent.mkdir(parents=True, exist_ok=True)
    save_image(pred.cpu(), sr_out, nrow=min(4, pred.shape[0]), padding=2)
    path = save_sr_comparison_grid(
        lr,
        pred,
        hr=hr,
        output_path=compare_out,
        hr_size=hr_size,
        include_soft_decode=args.include_soft_decode,
        soft=soft,
    )
    print(f"Wrote SR images: {sr_out}")
    print(f"Wrote comparison: {path}")
    labels = "nearest(LR) | bicubic | LatentSR"
    if args.include_soft_decode:
        labels += " | soft(decode z_lr)"
    if hr is not None:
        labels += " | HR"
    print(f"Rows: {labels}")
    print("Phase 9: inspect the grid — LatentSR should look sharper than bicubic.")


if __name__ == "__main__":
    main()
