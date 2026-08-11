"""Visualize CelebA LR / HR pairs (Phase 6).

Shows: LR (upsampled for display) | bicubic-to-HR | HR

Examples:
  python scripts/visualize_sr_pairs.py --config configs/sr_pairs.yaml --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.utils.config import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SR LR/HR pairs.")
    parser.add_argument("--config", type=Path, default=Path("configs/sr_pairs.yaml"))
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: sample_dir/sr_pairs_grid.png).",
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
    device = get_device(args.device or str(config.get("device", "auto")))

    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))
    batch_size = max(args.num_pairs, 1)

    _, val_loader = get_sr_pair_dataloaders(
        batch_size=batch_size,
        data_dir=config.get("data_dir", "data/raw"),
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=0,
        pin_memory=False,
        download=args.download,
    )
    lr, hr = next(iter(val_loader))
    lr = lr[: args.num_pairs].to(device)
    hr = hr[: args.num_pairs].to(device)

    assert lr.shape == (args.num_pairs, 3, lr_size, lr_size), lr.shape
    assert hr.shape == (args.num_pairs, 3, hr_size, hr_size), hr.shape

    # Display LR at HR resolution (nearest) and bicubic upsample for comparison.
    lr_nn = F.interpolate(lr, size=(hr_size, hr_size), mode="nearest")
    lr_bicubic = F.interpolate(
        lr, size=(hr_size, hr_size), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)

    grid = make_grid(
        torch.cat([lr_nn.cpu(), lr_bicubic.cpu(), hr.cpu()], dim=0),
        nrow=args.num_pairs,
        padding=2,
    )

    sample_dir = Path(config.get("sample_dir", "outputs/sr_pairs"))
    output = args.output or (sample_dir / "sr_pairs_grid.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output)

    print(f"LR shape: {tuple(lr.shape)}")
    print(f"HR shape: {tuple(hr.shape)}")
    print(f"Scale: {hr_size // lr_size}×")
    print(f"Wrote {output}")
    print("Rows: nearest(LR) | bicubic(LR→HR) | HR  — pairs should be aligned.")


if __name__ == "__main__":
    main()
