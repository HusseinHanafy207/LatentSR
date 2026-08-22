"""Stage 1 Steps 0–3: cache sanity, λ_g ratio table, one-image timing.

Held-out images start at val_index 64 so they are NOT the eval 64.

  python scripts/guidance_sanity.py \\
    --config configs/eval_sr.yaml \\
    --sr-checkpoint /kaggle/working/artifacts/latent_sr_q2/latest.pt \\
    --vae-checkpoint /kaggle/working/outputs/vae_sr/checkpoints/latest.pt \\
    --output-dir /kaggle/working/outputs/eval_guidance_sanity \\
    --device cuda --no-download
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.metrics.guidance_eval import (
    EARLY_WINDOW,
    LAMBDA_CANDIDATES,
    PRE_REGISTERED,
    REFERENCE_BANNER,
    check_soft_decode_cache,
    measure_lambda_ratios,
    recommend_lambda_g,
    time_guided_image,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 guidance: Step 0 cache check, Step 2 λ_g, Step 3 timing."
    )
    parser.add_argument("--sr-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_guidance_sanity"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--eval-images", type=int, default=64)
    parser.add_argument("--sanity-images", type=int, default=5)
    parser.add_argument("--sanity-start", type=int, default=64)
    parser.add_argument("--lambda-g", type=float, default=0.1, help="Only used for the timing run.")
    parser.add_argument("--skip-time", action="store_true")
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

    if not args.sr_checkpoint.is_file():
        raise SystemExit(f"SR checkpoint not found: {args.sr_checkpoint}")
    if not args.vae_checkpoint.is_file():
        raise SystemExit(f"VAE checkpoint not found: {args.vae_checkpoint}")

    model, vae, meta = load_sr_components(
        args.sr_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        map_location=device,
    )
    ctype = getattr(model.unet, "condition_type", "concat")
    print(f"SR epoch={meta.get('sr_epoch')} condition={ctype} VAE={meta['vae_checkpoint']}")
    if ctype != "concat":
        print("WARNING: Stage 1 protocol is concat + VAE-SR, not AdaGN/FiLM.")

    hr_size = int(meta.get("hr_size", config.get("hr_size", 128)))
    lr_size = int(meta.get("lr_size", config.get("lr_size", 32)))
    latent_scale = float(meta["latent_scale"])
    data_dir = str(args.data_dir or config.get("data_dir", "data/raw"))
    if not Path(data_dir).exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    _, val_loader = get_sr_pair_dataloaders(
        batch_size=int(config.get("batch_size", 4)),
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    print("Step 0: cache D(z_lr) and check soft-decode PSNR on the eval 64 …")
    cache_stats = check_soft_decode_cache(
        vae,
        val_loader,
        device=device,
        num_images=int(args.eval_images),
        start_index=0,
        hr_size=hr_size,
        latent_scale=latent_scale,
    )
    print(
        f"  PSNR(D(z_lr), HR) = {cache_stats['mean']:.4f} ± {cache_stats['std']:.4f}  "
        f"(expected {cache_stats['expected']:.2f}, |err|={cache_stats['abs_error']:.4f})"
    )
    if not cache_stats["ok"]:
        raise SystemExit(
            "Step 0 failed: soft-decode PSNR does not match 28.48 dB. "
            "Fix VAE-SR checkpoint / latent_scale before guidance."
        )

    print(
        f"Step 2: R(t) on {args.sanity_images} held-out val images "
        f"(start_index={args.sanity_start}, not the eval 64) …"
    )
    ratio_rows = measure_lambda_ratios(
        model,
        vae,
        val_loader,
        device=device,
        num_images=int(args.sanity_images),
        start_index=int(args.sanity_start),
        hr_size=hr_size,
        latent_scale=latent_scale,
        noise_seed=int(args.seed),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "lambda_ratios.csv"
    if ratio_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(ratio_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ratio_rows)

    print(f"\n{'t':>5}  " + "  ".join(f"{lam:^10g}" for lam in LAMBDA_CANDIDATES) + "   (mean R)")
    by_t: dict[int, list[dict]] = {}
    for row in ratio_rows:
        by_t.setdefault(int(row["t"]), []).append(row)
    for t in sorted(by_t, reverse=True):
        recs = by_t[t]
        cells = []
        for lam in LAMBDA_CANDIDATES:
            key = f"R_lambda_{lam:g}"
            mean_r = sum(r[key] for r in recs) / len(recs)
            cells.append(f"{mean_r:10.4f}")
        print(f"{t:5d}  " + "  ".join(cells))
    print("Target band: R(t) ≈ 0.1–0.5. Pick one λ_g; do not use PSNR to choose it.")
    suggestion: dict = {}
    if ratio_rows:
        suggestion = recommend_lambda_g(ratio_rows)
        print(
            f"R-table suggestion: λ_g={suggestion['lambda_g']:g}  "
            f"({suggestion['note']}). Confirm on the table before using it."
        )

    timing: dict[str, float] = {}
    if not args.skip_time:
        print(f"\nStep 3: time 1 held-out image with early guidance λ_g={args.lambda_g} …")
        timing = time_guided_image(
            model,
            vae,
            val_loader,
            device=device,
            lambda_g=float(args.lambda_g),
            window=EARLY_WINDOW,
            start_index=int(args.sanity_start),
            hr_size=hr_size,
            latent_scale=latent_scale,
            noise_seed=int(args.seed),
        )
        print(
            f"  {timing['seconds_one_image_early']:.1f}s / image  →  "
            f"~{timing['hours_64x2_guided']:.2f} h for 64 images × 2 guided windows"
        )

    payload = {
        "soft_decode": cache_stats,
        "timing": timing,
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "lambda_suggestion": suggestion,
        "convention": "dps_z_t_grad_applied_to_z_tm1",
    }
    (out / "sanity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
