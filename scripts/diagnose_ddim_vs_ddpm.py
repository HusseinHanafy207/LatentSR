"""Diagnostic 2 CLI: Deterministic DDIM (eta=0) vs Ancestral DDPM (eta=1).

Evaluates the existing Q2 checkpoint (no retraining):
    - DDPM: Stochastic ancestral sampler (eta=1.0)
    - DDIM: Deterministic probability-flow ODE sampler (eta=0.0)

Using the EXACT SAME initial noise x_T and validation images.
Compares: PSNR, LPIPS, SSIM, cos_peak, cos_{t=0}, collapse score.

Evaluates the Fork Decision:
    - If DDIM substantially reduces late collapse:
      Sampling stochasticity is a major part of the problem.
    - If the same collapse remains under DDIM:
      The learned denoising objective / score function parameterization
      is the stronger suspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.metrics.ddim_diagnostic import (
    format_ddpm_vs_ddim_table,
    run_ddpm_vs_ddim_comparison,
    save_ddpm_vs_ddim_results,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config
from latentsr.vae.whitening import ChannelWhitening


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic 2: Deterministic DDIM (eta=0) vs Ancestral DDPM (eta=1)."
    )
    parser.add_argument(
        "--checkpoint",
        "--sr-checkpoint",
        dest="checkpoint",
        type=Path,
        required=True,
        help="Path to trained ConditionalLatentDDPM checkpoint (e.g. Q2 checkpoint).",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help="Path to matching VAE checkpoint (e.g. VAE-SR or VAE-1).",
    )
    parser.add_argument(
        "--whiten-path",
        "--whitener",
        dest="whiten_path",
        type=Path,
        default=None,
        help="Optional path to channel whitener (.pt).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Q2 SR",
        help="Human-readable label for reporting/plots.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval_sr.yaml"),
        help="Path to base config file.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of validation images (default from config or 64).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default from config or 4).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Noise seed for paired initial x_T.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument("--data-dir", type=Path, default=None, help="CelebA data directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/ddpm_vs_ddim"),
        help="Output directory for CSVs, JSON, report, plots, and visual grids.",
    )
    parser.add_argument(
        "--grid-images",
        type=int,
        default=8,
        help="Number of images in side-by-side comparison grid.",
    )
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute LPIPS perceptual metric.",
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

    data_dir = args.data_dir or (
        Path(config["data_dir"]) if "data_dir" in config else None
    )
    if data_dir is None:
        raise SystemExit("Must supply --data-dir or set data_dir in config.")

    print("=" * 80)
    print("DIAGNOSTIC 2: DETERMINISTIC DDIM (eta=0) VS ANCESTRAL DDPM (eta=1)")
    print(f"Model: {args.model_name}")
    print(f"SR Checkpoint: {args.checkpoint}")
    print(f"VAE Checkpoint: {args.vae_checkpoint}")
    print(f"Whitener: {args.whiten_path or (whitener is not None)}")
    print(f"Images: {num_images} (batch size {batch_size}) | Seed: {seed} | Device: {device}")
    print("Zero retraining. Identical x_T noise and validation images for both samplers.")
    print("=" * 80)

    _, val_loader = get_sr_pair_dataloaders(
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        batch_size=batch_size,
        num_workers=int(config.get("num_workers", 2)),
        download=args.download,
        seed=seed,
    )

    result = run_ddpm_vs_ddim_comparison(
        model,
        vae,
        val_loader,
        device=device,
        num_images=num_images,
        hr_size=hr_size,
        latent_scale=latent_scale,
        noise_seed=seed,
        compute_lpips=args.lpips,
        whitener=whitener,
        show_progress=True,
        grid_images=args.grid_images,
    )

    saved = save_ddpm_vs_ddim_results(
        result,
        args.output_dir,
        model_name=args.model_name,
        hr_size=hr_size,
    )

    print("\n--- Head-to-Head Comparison Table ---")
    print(format_ddpm_vs_ddim_table(result))
    print("\n--- Fork Decision ---")
    print(f"Outcome: {result['fork_outcome']}")
    print(result["fork_verdict"])
    print(f"\nArtifacts saved to {args.output_dir}:")
    for k, p in saved.items():
        print(f"  [{k}] {p}")

    print("\nDiagnostic 2 complete successfully.")


if __name__ == "__main__":
    main()
