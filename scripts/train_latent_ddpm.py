"""Train unconditional latent DDPM (Phase 4).

Examples:
  python scripts/train_latent_ddpm.py --epochs 1
  python scripts/train_latent_ddpm.py --config configs/latent_ddpm.yaml
  python scripts/train_latent_ddpm.py --resume outputs/latent_ddpm/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from generative_models.losses import DDPMLoss

from latentsr.datasets.celeba import get_celeba_dataloaders
from latentsr.datasets.onthefly_latent import OnTheFlyLatentEncoder
from latentsr.diffusion.build import build_latent_ddpm_from_config
from latentsr.diffusion.trainer import LatentDDPMTrainer
from latentsr.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent-space DDPM.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/latent_ddpm.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def print_sanity_checks(train_metrics: dict[str, float], config: dict) -> None:
    print("\nPhase 4 sanity checks")
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
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    print(f"[{'OK' if ckpt.get('vae_checkpoint') else 'FAIL'}] vae_checkpoint in metadata")
    print(f"[{'OK' if 'latent_scale' in ckpt else 'FAIL'}] latent_scale in metadata")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device

    if args.resume is None and config.get("seed") is not None:
        torch.manual_seed(int(config["seed"]))

    train_loader, val_loader = get_celeba_dataloaders(
        batch_size=int(config["batch_size"]),
        data_dir=config["data_dir"],
        image_size=int(config["image_size"]),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    latent_encoder = OnTheFlyLatentEncoder.from_checkpoint(
        config["vae_checkpoint"],
        latent_scale=float(config.get("latent_scale", 1.0)),
        map_location="cpu",
    )
    model = build_latent_ddpm_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = DDPMLoss()

    trainer = LatentDDPMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        latent_encoder=latent_encoder,
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
