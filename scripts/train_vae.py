"""
Train the convolutional VAE on CelebA (Phase 1).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from latentsr.datasets.celeba import get_celeba_dataloaders
from latentsr.utils.config import load_config
from latentsr.vae.checkpointing import build_vae_from_config
from latentsr.vae.loss import VAELoss
from latentsr.vae.trainer import VAETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LatentSR convolutional VAE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae_celeba.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download CelebA if missing (default: true).",
    )
    return parser.parse_args()


def print_sanity_checks(train_metrics: dict[str, float], config: dict) -> None:
    print("\nPhase 1 sanity checks")
    print("-" * 24)
    finite = all(
        math.isfinite(train_metrics[key])
        for key in ("total_loss", "recon_loss", "kl_loss")
    )
    print(f"[{'OK' if finite else 'FAIL'}] Loss is finite (no NaN/Inf)")

    batch_improved = train_metrics["last_batch_loss"] <= train_metrics["first_batch_loss"]
    print(
        f"[{'OK' if batch_improved else 'WARN'}] Batch loss trend: "
        f"{train_metrics['first_batch_loss']:.6f} -> {train_metrics['last_batch_loss']:.6f}"
    )

    checkpoint_path = Path(config["checkpoint_dir"]) / "latest.pt"
    print(f"[{'OK' if checkpoint_path.exists() else 'FAIL'}] Checkpoint saved")

    train_csv = Path(config["log_dir"]) / config.get(
        "train_metrics_file", "train_metrics.csv"
    )
    print(f"[{'OK' if train_csv.exists() else 'FAIL'}] Train metrics CSV written")

    sample_glob = list(Path(config["sample_dir"]).glob("reconstruction_epoch_*.png"))
    print(f"[{'OK' if sample_glob else 'FAIL'}] Reconstruction image saved")


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

    model = build_vae_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = VAELoss(kl_weight=float(config.get("kl_weight", 1e-4)))

    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        config["start_epoch"] = int(checkpoint["epoch"])
        if int(config["epochs"]) <= int(checkpoint["epoch"]):
            raise ValueError(
                f"--epochs {config['epochs']} must be greater than resumed epoch "
                f"{checkpoint['epoch']}."
            )
        print(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"training to epoch {config['epochs']}"
        )

    trainer.train()

    latest = Path(config["checkpoint_dir"]) / "latest.pt"
    checkpoint = torch.load(latest, weights_only=False)
    train_metrics = checkpoint["metrics"]
    print(f"\nFinished training through epoch {config['epochs']}.")
    print(f"Latest checkpoint: {latest}")

    if int(config["epochs"]) == 1 and args.resume is None:
        print_sanity_checks(train_metrics, config)


if __name__ == "__main__":
    main()
