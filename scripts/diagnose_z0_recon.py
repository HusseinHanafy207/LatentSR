"""Decode ẑ0 at selected timesteps (no training, no sampler change).

Same concat VAE-1 vs VAE-SR pair and noise seed as the latent diagnostic.
At t in {800,700,650,600,500,400,300,200,100,0}:

    x̂0(t) = decode(ẑ0(t))

then PSNR / SSIM / LPIPS / edge MAE vs HR, and LR-consistency PSNR.

Kaggle (reuse the four checkpoints from the latent diagnostic):

  python scripts/diagnose_z0_recon.py \\
    --config configs/eval_sr.yaml \\
    --baseline-sr PATH_PHASE8 \\
    --baseline-vae PATH_VAE1 \\
    --candidate-sr PATH_Q2 \\
    --candidate-vae PATH_VAE_SR \\
    --output-dir /kaggle/working/outputs/eval_z0_recon \\
    --num-images 64 --batch-size 4 --seed 42 --device cuda --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.metrics.z0_recon_diagnostic import (
    DEFAULT_Z0_RECON_TIMESTEPS,
    format_z0_recon_table,
    run_z0_recon_diagnostic,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode ẑ0 at selected timesteps (VAE-1 vs VAE-SR)."
    )
    parser.add_argument("--baseline-sr", type=Path, required=True)
    parser.add_argument("--baseline-vae", type=Path, required=True)
    parser.add_argument("--candidate-sr", type=Path, required=True)
    parser.add_argument("--candidate-vae", type=Path, required=True)
    parser.add_argument("--baseline-name", type=str, default="vae1")
    parser.add_argument("--candidate-name", type=str, default="vae_sr")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--timesteps",
        type=str,
        default=",".join(str(t) for t in DEFAULT_Z0_RECON_TIMESTEPS),
        help="Comma-separated diffusion timesteps to decode.",
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


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found:\n  {path}")


def _parse_timesteps(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--timesteps is empty")
    return [int(p) for p in parts]


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    torch.manual_seed(seed)
    timesteps = _parse_timesteps(args.timesteps)

    _require_file(args.baseline_sr, "VAE-1 SR checkpoint")
    _require_file(args.baseline_vae, "VAE-1 checkpoint")
    _require_file(args.candidate_sr, "VAE-SR DDPM checkpoint")
    _require_file(args.candidate_vae, "VAE-SR checkpoint")

    model_a, vae_a, meta_a = load_sr_components(
        args.baseline_sr,
        vae_checkpoint=args.baseline_vae,
        map_location=device,
    )
    model_b, vae_b, meta_b = load_sr_components(
        args.candidate_sr,
        vae_checkpoint=args.candidate_vae,
        map_location=device,
    )
    cond_a = getattr(model_a.unet, "condition_type", "concat")
    cond_b = getattr(model_b.unet, "condition_type", "concat")
    print(f"baseline ({args.baseline_name}): epoch={meta_a.get('sr_epoch')} condition={cond_a}")
    print(f"candidate ({args.candidate_name}): epoch={meta_b.get('sr_epoch')} condition={cond_b}")
    print(f"decode ẑ0 at t={list(timesteps)}")

    hr_size = int(meta_a.get("hr_size", config.get("hr_size", 128)))
    lr_size = int(meta_a.get("lr_size", config.get("lr_size", 32)))
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
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else "outputs/eval_z0_recon"
    )
    data_dir = (
        str(args.data_dir)
        if args.data_dir is not None
        else config.get("data_dir", "data/raw")
    )
    if not Path(data_dir).exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    _, val_loader = get_sr_pair_dataloaders(
        batch_size=batch_size,
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    print(
        f"Diagnosing {num_images} val images, {model_a.num_timesteps} reverse steps, "
        f"seed={seed} on {device}"
    )
    result = run_z0_recon_diagnostic(
        model_a,
        vae_a,
        model_b,
        vae_b,
        val_loader,
        device=device,
        timesteps=timesteps,
        num_images=num_images,
        hr_size=hr_size,
        lr_size=lr_size,
        latent_scale_a=float(meta_a["latent_scale"]),
        latent_scale_b=float(meta_b["latent_scale"]),
        noise_seed=seed,
        output_dir=output_dir,
        show_progress=True,
        compute_lpips=bool(args.lpips),
        grid_images=int(args.grid_images),
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
    print()
    print(format_z0_recon_table(result))
    print()
    print(f"Wrote {output_dir}")
    for key, path in (result.get("paths") or {}).items():
        if key in {"plots", "grids"}:
            for item in path:
                print(f"  {item}")
        elif path:
            print(f"  {path}")


if __name__ == "__main__":
    main()
