"""CLI: Condition-switch rollout diagnostic (zero retraining).

Starts every sample with the correct condition. At each switch time
τ ∈ {800, 650, 500, 300, 200, 100, 50}, permanently replaces it with a
shuffled condition for every remaining step. Measures final PSNR / LPIPS
and latent error vs the always-true baseline.

Answers: when does the explicit condition cease to be causally necessary?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_val_dataloader
from latentsr.metrics.condition_switch import (
    DEFAULT_SWITCH_TIMES,
    format_condition_switch_table,
    run_condition_switch_rollout,
    save_condition_switch_results,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config
from latentsr.vae.whitening import ChannelWhitening


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Condition-switch rollout: start with true z_lr, permanently "
            "replace with shuffled at tau, measure final PSNR/LPIPS/latent error."
        )
    )
    parser.add_argument(
        "--checkpoint",
        "--sr-checkpoint",
        dest="checkpoint",
        type=Path,
        required=True,
        help="Path to trained ConditionalLatentDDPM checkpoint (e.g. Q2).",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help="Path to matching VAE checkpoint.",
    )
    parser.add_argument(
        "--whiten-path",
        "--whitener",
        dest="whiten_path",
        type=Path,
        default=None,
        help="Optional channel whitener (.pt).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Q2 SR",
        help="Label for reports/plots.",
    )
    parser.add_argument(
        "--switch-times",
        type=str,
        default="800,650,500,300,200,100,50",
        help="Comma-separated switch timesteps tau.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["ddpm", "ddim"],
        default="ddpm",
        help="Reverse sampler (default ancestral DDPM).",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM eta (0.0 = deterministic). Ignored for ddpm.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval_sr.yaml"),
    )
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (must be >= 2 for shuffling; default 4).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/condition_switch"),
    )
    parser.add_argument("--grid-images", type=int, default=4)
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    torch.manual_seed(seed)

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.vae_checkpoint.is_file():
        raise SystemExit(f"VAE checkpoint not found: {args.vae_checkpoint}")

    model, vae, meta = load_sr_components(
        args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        map_location=device,
        whiten_path=args.whiten_path,
    )

    whitener = None
    if args.whiten_path is not None:
        if not args.whiten_path.is_file():
            raise SystemExit(f"Whitener not found: {args.whiten_path}")
        whitener = ChannelWhitening.load(args.whiten_path)
    elif meta.get("whitener") is not None:
        whitener = meta["whitener"]

    hr_size = int(meta.get("hr_size", config.get("hr_size", 128)))
    lr_size = int(meta.get("lr_size", config.get("lr_size", 32)))
    latent_scale = float(meta.get("latent_scale", config.get("latent_scale", 1.0)))

    num_images = int(
        args.num_images
        if args.num_images is not None
        else config.get("num_images", 64)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else config.get("batch_size", 4)
    )
    if batch_size < 2:
        print(f"Notice: batch_size={batch_size} < 2; raising to 2 for shuffle.")
        batch_size = 2

    switch_times = [
        int(s.strip()) for s in args.switch_times.split(",") if s.strip()
    ]
    if not switch_times:
        switch_times = list(DEFAULT_SWITCH_TIMES)

    data_dir = args.data_dir or (
        Path(config["data_dir"]) if "data_dir" in config else None
    )
    if data_dir is None:
        raise SystemExit("Must supply --data-dir or set data_dir in config.")

    print("=" * 80)
    print("CONDITION-SWITCH ROLLOUT DIAGNOSTIC")
    print(f"Model: {args.model_name}")
    print(f"SR Checkpoint: {args.checkpoint}")
    print(f"VAE Checkpoint: {args.vae_checkpoint}")
    print(f"Whitener: {args.whiten_path or (whitener is not None)}")
    print(f"Sampler: {args.sampler} (eta={args.ddim_eta})")
    print(f"Switch times tau: {switch_times}")
    print(f"Images: {num_images} (batch {batch_size}) | Seed: {seed} | Device: {device}")
    print("=" * 80)

    val_loader = get_sr_pair_val_dataloader(
        batch_size=batch_size,
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    result = run_condition_switch_rollout(
        model,
        vae,
        val_loader,
        device=device,
        num_images=num_images,
        hr_size=hr_size,
        latent_scale=latent_scale,
        noise_seed=seed,
        switch_times=switch_times,
        sampler=args.sampler,
        ddim_eta=args.ddim_eta,
        compute_lpips=args.lpips,
        whitener=whitener,
        show_progress=True,
        grid_images=args.grid_images,
    )

    saved = save_condition_switch_results(
        result,
        args.output_dir,
        model_name=args.model_name,
    )

    print("\n--- Condition-Switch Table ---")
    print(format_condition_switch_table(result))
    print(f"\nArtifacts saved to {args.output_dir}:")
    for k, p in saved.items():
        print(f"  [{k}] {p}")
    print("\nCondition-switch diagnostic complete.")


if __name__ == "__main__":
    main()
