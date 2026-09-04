"""Train conditional latent DDPM for super-resolution.

Examples:
  python scripts/train_sr.py --config configs/latent_sr.yaml --epochs 1 --no-download
  python scripts/train_sr.py --resume .../latest.pt --epochs 50 --no-download
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from generative_models.losses import DDPMLoss

from latentsr.datasets.onthefly_sr_latent import OnTheFlySRLatentEncoder
from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.super_resolution.condition import (
    build_conditioned_latent_ddpm_from_config,
)
from latentsr.super_resolution.trainer import LatentSRTrainer
from latentsr.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LatentSR conditional DDPM.")
    parser.add_argument("--config", type=Path, default=Path("configs/latent_sr.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
        help="Override frozen VAE path from the config.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def print_sanity_checks(train_metrics: dict[str, float], config: dict) -> None:
    print("\nPhase 8 sanity checks")
    print("-" * 24)
    finite = math.isfinite(train_metrics["loss"])
    print(f"[{'OK' if finite else 'FAIL'}] Loss is finite")
    improved = train_metrics["last_batch_loss"] <= train_metrics["first_batch_loss"]
    print(
        f"[{'OK' if improved else 'WARN'}] Batch loss trend: "
        f"{train_metrics['first_batch_loss']:.4f} -> {train_metrics['last_batch_loss']:.4f}"
    )
    latest = Path(config["checkpoint_dir"]) / "latest.pt"
    print(f"[{'OK' if latest.exists() else 'FAIL'}] Checkpoint saved")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.vae_checkpoint is not None:
        config["vae_checkpoint"] = str(args.vae_checkpoint)

    if args.resume is None and config.get("seed") is not None:
        torch.manual_seed(int(config["seed"]))

    train_loader, val_loader = get_sr_pair_dataloaders(
        batch_size=int(config["batch_size"]),
        data_dir=config["data_dir"],
        hr_size=int(config.get("hr_size", 128)),
        lr_size=int(config.get("lr_size", 32)),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    sr_encoder = OnTheFlySRLatentEncoder.from_checkpoint(
        config["vae_checkpoint"],
        latent_scale=float(config.get("latent_scale", 1.0)),
        hr_size=int(config.get("hr_size", 128)),
        map_location="cpu",
        whiten_path=config.get("zlr_whiten_path"),
    )
    if sr_encoder.whitener is not None:
        print(
            f"condition whitening: {config.get('zlr_whiten_path')}  "
            f"mode={sr_encoder.whitener.mode}  eps={sr_encoder.whitener.eps}"
        )
    else:
        print("condition whitening: off (raw z_lr)")
    model = build_conditioned_latent_ddpm_from_config(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"condition_type={config.get('condition_type', 'concat')}  "
        f"params={n_params / 1e6:.2f}M"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = DDPMLoss()

    trainer = LatentSRTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        sr_encoder=sr_encoder,
        config=config,
    )

    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        config["start_epoch"] = int(checkpoint["epoch"])
        if int(config["epochs"]) <= int(checkpoint["epoch"]):
            raise ValueError(
                f"--epochs {config['epochs']} must be > resumed epoch "
                f"{checkpoint['epoch']}"
            )
        print(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"training to epoch {config['epochs']}"
        )

    trainer.train()

    latest = Path(config["checkpoint_dir"]) / "latest.pt"
    checkpoint = torch.load(latest, weights_only=False)
    print(f"\nFinished through epoch {config['epochs']}.")
    print(f"Latest checkpoint: {latest}")
    if int(config["epochs"]) == 1 and args.resume is None:
        print_sanity_checks(checkpoint["metrics"], config)


if __name__ == "__main__":
    main()
