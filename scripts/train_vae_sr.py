"""Train an SR-aware VAE: HR recon + stop-grad μ_lr → μ_hr alignment.

Fine-tune from VAE-1 (recommended) so HR reconstruction stays a good code:

  python scripts/train_vae_sr.py --config configs/vae_sr_align.yaml \\
    --init-from outputs/vae/checkpoints/checkpoint_epoch_050.pt --no-download
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.utils.config import load_config
from latentsr.vae.checkpointing import build_vae_from_config, load_vae_checkpoint
from latentsr.vae.loss import SRAwareVAELoss
from latentsr.vae.trainer import SRAwareVAETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SR-aware VAE (Q2).")
    parser.add_argument("--config", type=Path, default=Path("configs/vae_sr_align.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Load VAE-1 weights only (fresh optimizer). Preferred over training from scratch.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Full trainer resume.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def print_sanity_checks(train_metrics: dict[str, float], config: dict) -> None:
    print("\nSR-aware VAE sanity checks")
    print("-" * 28)
    finite = all(
        math.isfinite(train_metrics[key])
        for key in ("total_loss", "recon_loss", "kl_loss", "align_loss")
    )
    print(f"[{'OK' if finite else 'FAIL'}] Loss is finite")
    latest = Path(config["checkpoint_dir"]) / "latest.pt"
    print(f"[{'OK' if latest.exists() else 'FAIL'}] Checkpoint saved")
    print(
        f"align_loss={train_metrics['align_loss']:.4f}  "
        f"z_cosine={train_metrics['z_cosine']:.4f}  "
        f"(VAE-1 baseline cosine was ~0.63)"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device
    if args.init_from is not None:
        config["init_from"] = str(args.init_from)
    if args.resume is not None and args.init_from is not None:
        raise SystemExit("Pass only one of --resume or --init-from.")

    if args.resume is None and config.get("seed") is not None:
        torch.manual_seed(int(config["seed"]))

    train_loader, val_loader = get_sr_pair_dataloaders(
        batch_size=int(config["batch_size"]),
        data_dir=config["data_dir"],
        hr_size=int(config.get("hr_size", config.get("image_size", 128))),
        lr_size=int(config.get("lr_size", 32)),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    init_from = args.init_from or config.get("init_from")
    if init_from:
        model, init_ckpt = load_vae_checkpoint(init_from, map_location="cpu")
        print(
            f"Initialized weights from {init_from} "
            f"(epoch {init_ckpt.get('epoch')}); optimizer starts fresh."
        )
    else:
        model = build_vae_from_config(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = SRAwareVAELoss(
        kl_weight=float(config.get("kl_weight", 1e-4)),
        align_weight=float(config.get("align_weight", 1e-3)),
    )

    trainer = SRAwareVAETrainer(
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
                f"--epochs {config['epochs']} must be > resumed epoch "
                f"{checkpoint['epoch']}"
            )
        print(
            f"Resumed from epoch {checkpoint['epoch']}, "
            f"training to epoch {config['epochs']}"
        )

    print(
        f"align_weight={criterion.align_weight}  |  "
        f"kl_weight={criterion.kl_weight}"
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
