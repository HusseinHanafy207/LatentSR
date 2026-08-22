"""Stage 1/2 guidance eval. N=256 confirmation: start_index=0, resume on.

  python scripts/evaluate_guidance.py \\
    --config configs/eval_sr.yaml \\
    --sr-checkpoint /kaggle/working/artifacts/latent_sr_q2/latest.pt \\
    --vae-checkpoint /kaggle/working/outputs/vae_sr/checkpoints/latest.pt \\
    --condition late --lambda-g 200 \\
    --output-dir /kaggle/working/outputs/eval_guidance_n256_late_l200 \\
    --seed 42 --num-images 256 --start-index 0 --batch-size 1 \\
    --device cuda --no-download --resume --checkpoint-every 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.metrics.guidance_eval import (
    CONFIRMATION_N256,
    PRE_REGISTERED,
    REFERENCE_BANNER,
    check_soft_decode_cache,
    compare_guidance_conditions,
    run_guidance_condition,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 concat guidance eval (inference only).")
    parser.add_argument("--sr-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=("baseline", "early", "late"),
    )
    parser.add_argument("--lambda-g", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--guide-every", type=int, default=1)
    parser.add_argument("--grid-images", type=int, default=8)
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compare-baseline", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip val_index already in output-dir/per_image.csv (default on).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=8,
        help="Flush per_image.csv every N new images. 0 disables mid-run writes.",
    )
    parser.add_argument(
        "--skip-soft-check",
        action="store_true",
        help="Skip the n=64 soft-decode PSNR gate (do not use on the confirmation run).",
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
    torch.manual_seed(int(args.seed))

    print(REFERENCE_BANNER)
    print()
    print(PRE_REGISTERED)
    print()
    if int(args.num_images) >= 256:
        print(CONFIRMATION_N256)
        print()

    if args.condition != "baseline" and args.lambda_g is None:
        raise SystemExit("--lambda-g is required for early/late (choose from Step 2, not from PSNR).")
    lambda_g = 0.0 if args.condition == "baseline" else float(args.lambda_g)

    model, vae, meta = load_sr_components(
        args.sr_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        map_location=device,
    )
    ctype = getattr(model.unet, "condition_type", "concat")
    print(f"SR epoch={meta.get('sr_epoch')} condition={ctype}")
    print(f"eval condition={args.condition}  λ_g={lambda_g}  guide_every={args.guide_every}")
    if ctype != "concat":
        print("WARNING: Stage 1 protocol is concat + VAE-SR.")

    hr_size = int(meta.get("hr_size", config.get("hr_size", 128)))
    lr_size = int(meta.get("lr_size", config.get("lr_size", 32)))
    latent_scale = float(meta["latent_scale"])
    data_dir = str(args.data_dir or config.get("data_dir", "data/raw"))
    batch_size = int(
        args.batch_size if args.batch_size is not None else config.get("batch_size", 4)
    )
    output_dir = args.output_dir or Path(f"outputs/eval_guidance_{args.condition}")
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

    if int(args.start_index) == 0 and not args.skip_soft_check:
        n_check = min(64, int(args.num_images))
        soft = check_soft_decode_cache(
            vae,
            val_loader,
            device=device,
            num_images=n_check,
            start_index=0,
            hr_size=hr_size,
            latent_scale=latent_scale,
        )
        print(
            f"soft-decode PSNR (n={n_check}, val_index 0..{n_check - 1}): "
            f"{soft['mean']:.4f}  expected {soft['expected']:.2f}  "
            f"abs_error={soft['abs_error']:.4f}  ok={soft['ok']}",
            flush=True,
        )
        if not soft["ok"]:
            raise SystemExit(
                "soft-decode PSNR on the locked first 64 does not match 28.48. "
                "Stop — VAE checkpoint or latent scale is wrong."
            )

    result = run_guidance_condition(
        model,
        vae,
        val_loader,
        device=device,
        condition=args.condition,
        lambda_g=lambda_g,
        num_images=int(args.num_images),
        start_index=int(args.start_index),
        hr_size=hr_size,
        lr_size=lr_size,
        latent_scale=latent_scale,
        noise_seed=int(args.seed),
        compute_lpips=bool(args.lpips),
        grid_images=int(args.grid_images),
        output_dir=output_dir,
        guide_every=int(args.guide_every),
        show_progress=True,
        resume=bool(args.resume),
        checkpoint_every=int(args.checkpoint_every),
    )
    print()
    print(result["table"])
    print()
    print(f"Wrote {output_dir}")
    print(f"  {output_dir / 'per_image.csv'}")
    print(f"  {output_dir / 'trajectory.csv'}")

    if args.compare_baseline is not None:
        cmp_dir = Path(str(output_dir) + "_vs_baseline")
        compare_guidance_conditions(
            args.compare_baseline,
            output_dir / "per_image.csv",
            output_dir=cmp_dir,
            baseline_name="baseline",
            candidate_name=args.condition,
        )
        print(f"Paired comparison: {cmp_dir}")


if __name__ == "__main__":
    main()
