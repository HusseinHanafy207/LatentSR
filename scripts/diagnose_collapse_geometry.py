"""Phase 1.5: late reverse-chain collapse vs local z_lr geometry.

Exploratory / correlational. Reuses the seeded reverse chain from the
timestep diagnostic; adds per-image

    C_i = cos(ẑ0(t_peak), z_lr) − cos(ẑ0(0), z_lr)

and leave-one-out k-NN local geometry around each condition code.

Kaggle (after git pull):

  python scripts/diagnose_collapse_geometry.py \\
    --config configs/eval_sr.yaml \\
    --baseline-sr /kaggle/working/artifacts/latent_sr/latest.pt \\
    --baseline-vae /kaggle/working/artifacts/vae/checkpoint_epoch_050.pt \\
    --candidate-sr /kaggle/working/artifacts/latent_sr_q2/latest.pt \\
    --candidate-vae /kaggle/working/outputs/vae_sr/checkpoints/latest.pt \\
    --output-dir /kaggle/working/outputs/eval_collapse_geometry \\
    --num-images 64 --reference-images 512 --knn 32 \\
    --batch-size 4 --seed 42 --device cuda --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.metrics.collapse_geometry import (
    format_collapse_geometry_table,
    run_collapse_geometry,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse score C_i vs local z_lr geometry (exploratory)."
    )
    parser.add_argument("--baseline-sr", type=Path, required=True)
    parser.add_argument("--baseline-vae", type=Path, required=True)
    parser.add_argument("--candidate-sr", type=Path, required=True)
    parser.add_argument("--candidate-vae", type=Path, required=True)
    parser.add_argument("--baseline-name", type=str, default="vae1")
    parser.add_argument("--candidate-name", type=str, default="vae_sr")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Images for reverse-chain collapse scores (expensive).",
    )
    parser.add_argument(
        "--reference-images",
        type=int,
        default=None,
        help=(
            "Size of z_lr cloud for k-NN geometry (>= num-images). "
            "Default: same as num-images."
        ),
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=32,
        help="Neighbors for local erank / κ / density (exclude self).",
    )
    parser.add_argument(
        "--fixed-t-peak",
        type=int,
        default=None,
        help="If set, use this t for every image instead of per-image argmax.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="both",
        choices=("both", "baseline", "candidate"),
        help="Which reverse chains to run (candidate-only is faster).",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found:\n  {path}")


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    torch.manual_seed(seed)

    _require_file(args.baseline_sr, "VAE-1 SR checkpoint")
    _require_file(args.baseline_vae, "VAE-1 checkpoint")
    _require_file(args.candidate_sr, "VAE-SR DDPM checkpoint")
    _require_file(args.candidate_vae, "VAE-SR checkpoint")

    models = {}
    vaes = {}
    scales = {}
    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))

    want_base = args.models in ("both", "baseline")
    want_cand = args.models in ("both", "candidate")

    if want_base:
        model_a, vae_a, meta_a = load_sr_components(
            args.baseline_sr,
            vae_checkpoint=args.baseline_vae,
            map_location=device,
        )
        models[args.baseline_name] = model_a
        vaes[args.baseline_name] = vae_a
        scales[args.baseline_name] = float(meta_a["latent_scale"])
        hr_size = int(meta_a.get("hr_size", hr_size))
        lr_size = int(meta_a.get("lr_size", lr_size))
        print(
            f"baseline ({args.baseline_name}): epoch={meta_a.get('sr_epoch')} "
            f"condition={getattr(model_a.unet, 'condition_type', 'concat')}",
            flush=True,
        )

    if want_cand:
        model_b, vae_b, meta_b = load_sr_components(
            args.candidate_sr,
            vae_checkpoint=args.candidate_vae,
            map_location=device,
        )
        models[args.candidate_name] = model_b
        vaes[args.candidate_name] = vae_b
        scales[args.candidate_name] = float(meta_b["latent_scale"])
        hr_size = int(meta_b.get("hr_size", hr_size))
        lr_size = int(meta_b.get("lr_size", lr_size))
        print(
            f"candidate ({args.candidate_name}): epoch={meta_b.get('sr_epoch')} "
            f"condition={getattr(model_b.unet, 'condition_type', 'concat')}",
            flush=True,
        )

    num_images = int(
        args.num_images
        if args.num_images is not None
        else config.get("num_images", 64)
    )
    reference_images = (
        int(args.reference_images)
        if args.reference_images is not None
        else num_images
    )
    if reference_images < num_images:
        raise SystemExit("--reference-images must be >= --num-images")

    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else config.get("batch_size", 4)
    )
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else "outputs/eval_collapse_geometry"
    )
    data_dir = (
        str(args.data_dir)
        if args.data_dir is not None
        else config.get("data_dir", "data/raw")
    )
    if not Path(data_dir).exists():
        raise SystemExit(
            f"data_dir not found: {data_dir}\n"
            "On Kaggle use configs/eval_sr.yaml (data_dir=/kaggle/working/data/raw)."
        )

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
        f"n_collapse={num_images}  n_reference={reference_images}  knn={args.knn}  "
        f"seed={seed}  device={device}",
        flush=True,
    )
    print(
        "C_i = cos(ẑ0(t_peak), z_lr) − cos(ẑ0(0), z_lr); "
        "local geom = leave-one-out k-NN in z_lr cloud.",
        flush=True,
    )

    result = run_collapse_geometry(
        models,
        vaes,
        val_loader,
        device=device,
        num_images=num_images,
        reference_images=reference_images,
        knn=int(args.knn),
        hr_size=hr_size,
        latent_scales=scales,
        noise_seed=seed,
        fixed_t_peak=args.fixed_t_peak,
        output_dir=output_dir,
        show_progress=True,
    )

    print(flush=True)
    print(format_collapse_geometry_table(result), flush=True)
    print(f"\nWrote {output_dir}", flush=True)
    for key, path in (result.get("paths") or {}).items():
        if key == "plots":
            for plot in path:
                print(f"  {plot}", flush=True)
        else:
            print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
