from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_val_dataloader
from latentsr.metrics.condition_utility import (
    DEFAULT_MILESTONES,
    evaluate_condition_utility_forward,
    evaluate_condition_utility_reverse,
    format_milestone_table,
    save_condition_utility_results,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config
from latentsr.vae.whitening import ChannelWhitening


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic 1: Measure Condition Utility across diffusion timesteps."
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
        "--mode",
        type=str,
        choices=["reverse", "forward", "both"],
        default="reverse",
        help=(
            "'reverse': along actual sampling trajectory; "
            "'forward': on exact training-marginal q(z_t | z_hr); "
            "'both': runs both diagnostics."
        ),
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["ddpm", "ddim"],
        default="ddpm",
        help="Sampler for reverse mode: 'ddpm' (ancestral) or 'ddim'.",
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity eta (0.0 = deterministic ODE).",
    )
    parser.add_argument(
        "--milestones",
        type=str,
        default="999,800,650,500,300,100,0",
        help="Comma-separated milestone timesteps to report.",
    )
    parser.add_argument(
        "--eval-all-timesteps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate condition utility across all 1000 timesteps for complete curve plotting.",
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
        help="Batch size (must be >= 2 for condition shuffling; default 8).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Noise seed.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument("--data-dir", type=Path, default=None, help="CelebA data directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/condition_utility"),
        help="Output directory for CSVs, JSON, report, and plot.",
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

    # Load components
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
        else config.get("batch_size", 8)
    )
    if batch_size < 2:
        print(f"Notice: batch_size={batch_size} < 2. Adjusting to batch_size=2 for condition shuffling.")
        batch_size = 2

    # Parse milestones
    milestones = [int(s.strip()) for s in args.milestones.split(",") if s.strip()]
    if not milestones:
        milestones = list(DEFAULT_MILESTONES)

    data_dir = args.data_dir or (
        Path(config["data_dir"]) if "data_dir" in config else None
    )
    if data_dir is None:
        raise SystemExit("Must supply --data-dir or set data_dir in config.")

    print("=" * 80)
    print("DIAGNOSTIC 1: CONDITION UTILITY AS A FUNCTION OF TIMESTEP")
    print(f"Model: {args.model_name}")
    print(f"SR Checkpoint: {args.checkpoint}")
    print(f"VAE Checkpoint: {args.vae_checkpoint}")
    print(f"Whitener: {args.whiten_path or (whitener is not None)}")
    print(f"Mode: {args.mode} | Sampler: {args.sampler} (eta={args.ddim_eta})")
    print(f"Milestones: {milestones}")
    print(f"Images: {num_images} (batch size {batch_size}) | Device: {device}")
    print("=" * 80)

    # Dataloader (val-only skips building CelebA train partition)
    val_loader = get_sr_pair_val_dataloader(
        batch_size=batch_size,
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("reverse", "both"):
        print("\n>>> Running condition utility along reverse sampling trajectory...")
        rev_result = evaluate_condition_utility_reverse(
            model,
            vae,
            val_loader,
            device=device,
            num_images=num_images,
            hr_size=hr_size,
            latent_scale=latent_scale,
            noise_seed=seed,
            sampler=args.sampler,
            ddim_eta=args.ddim_eta,
            whitener=whitener,
            milestones=milestones,
            eval_all_timesteps=args.eval_all_timesteps,
            show_progress=True,
        )
        rev_dir = args.output_dir if args.mode == "reverse" else (args.output_dir / "reverse")
        saved_rev = save_condition_utility_results(
            rev_result,
            rev_dir,
            model_name=f"{args.model_name} (reverse {args.sampler})",
        )
        print("\n--- Milestone Table (Reverse Sampling) ---")
        print(format_milestone_table(rev_result["milestone_summary"]))
        print(f"\nArtifacts saved to {rev_dir}:")
        for k, p in saved_rev.items():
            print(f"  [{k}] {p}")

    if args.mode in ("forward", "both"):
        print("\n>>> Running condition utility on forward q-sample marginals q(z_t | z_hr)...")
        fwd_result = evaluate_condition_utility_forward(
            model,
            vae,
            val_loader,
            device=device,
            num_images=num_images,
            hr_size=hr_size,
            latent_scale=latent_scale,
            noise_seed=seed,
            whitener=whitener,
            milestones=milestones,
            eval_all_timesteps=args.eval_all_timesteps,
            show_progress=True,
        )
        fwd_dir = args.output_dir if args.mode == "forward" else (args.output_dir / "forward")
        saved_fwd = save_condition_utility_results(
            fwd_result,
            fwd_dir,
            model_name=f"{args.model_name} (forward q-sample)",
        )
        print("\n--- Milestone Table (Forward q-sample) ---")
        print(format_milestone_table(fwd_result["milestone_summary"]))
        print(f"\nArtifacts saved to {fwd_dir}:")
        for k, p in saved_fwd.items():
            print(f"  [{k}] {p}")

    print("\nDiagnostic 1 complete successfully.")


if __name__ == "__main__":
    main()
