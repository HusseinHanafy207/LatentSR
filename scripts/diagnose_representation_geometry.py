"""RiT-style geometry: VAE-1 vs VAE-SR latents.

Computes on matched CelebA val images:

  - TwoNN intrinsic dimensionality
  - Effective rank
  - Covariance / transport condition number
  - Excess kurtosis
  - PCA cumulative variance spectrum

For both ``z_hr = encode(HR)`` and ``z_lr = encode(bicubic↑ LR)``.

Full-val / trustworthy κ (need N > ambient D=4096) + TwoNN uncertainty:

  python scripts/diagnose_representation_geometry.py \\
    --baseline-vae outputs/vae/checkpoints/checkpoint_epoch_050.pt \\
    --candidate-vae outputs/vae_sr/checkpoints/latest.pt \\
    --config configs/eval_vae.yaml \\
    --num-images 0 --batch-size 32 --seed 42 --no-download \\
    --twonn-bootstraps 10 --twonn-subsample 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_val_dataloader
from latentsr.metrics.representation_geometry import (
    format_geometry_table,
    run_representation_geometry,
)
from latentsr.utils.config import get_device, load_config
from latentsr.vae.latent import is_frozen, load_frozen_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RiT-style latent geometry: VAE-1 vs VAE-SR."
    )
    parser.add_argument("--baseline-vae", type=Path, required=True)
    parser.add_argument("--candidate-vae", type=Path, required=True)
    parser.add_argument("--baseline-name", type=str, default="vae1")
    parser.add_argument("--candidate-name", type=str, default="vae_sr")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_vae.yaml"))
    parser.add_argument(
        "--num-images",
        type=int,
        default=0,
        help=(
            "Val images to encode. Use 0 for the full val split "
            "(needed for full-rank κ at D=4096; CelebA val ≈ 19867)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest on Kaggle).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_representation_geometry"),
    )
    parser.add_argument(
        "--twonn-bootstraps",
        type=int,
        default=10,
        help="Independent TwoNN subsample draws (need >1 and subsample < N for ±std).",
    )
    parser.add_argument(
        "--twonn-subsample",
        type=int,
        default=5000,
        help=(
            "Points per TwoNN bootstrap. Keep strictly < N for nonzero uncertainty "
            "(default 5000). If >= N, only one full-sample estimate is returned."
        ),
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


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    torch.manual_seed(seed)

    _require_file(args.baseline_vae, "VAE-1 checkpoint")
    _require_file(args.candidate_vae, "VAE-SR checkpoint")

    print(f"device: {device}", flush=True)
    print(f"loading VAEs…", flush=True)
    vae_a, ckpt_a = load_frozen_vae(args.baseline_vae, map_location=device)
    vae_b, ckpt_b = load_frozen_vae(args.candidate_vae, map_location=device)
    if not is_frozen(vae_a) or not is_frozen(vae_b):
        raise SystemExit("Both VAEs must load frozen.")

    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))
    latent_scale = float(
        args.latent_scale
        if args.latent_scale is not None
        else config.get("latent_scale", 1.0)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
    )
    data_dir = (
        str(args.data_dir)
        if args.data_dir is not None
        else config.get("data_dir", "data/raw")
    )
    if not Path(data_dir).exists():
        raise SystemExit(
            f"data_dir not found: {data_dir}\n"
            "Pass --data-dir or set data_dir in the config."
        )

    print(
        f"baseline ({args.baseline_name}): {args.baseline_vae}  "
        f"epoch={ckpt_a.get('epoch')}",
        flush=True,
    )
    print(
        f"candidate ({args.candidate_name}): {args.candidate_vae}  "
        f"epoch={ckpt_b.get('epoch')}",
        flush=True,
    )
    print("Building val-only CelebA loader (no train split)…", flush=True)

    val_loader = get_sr_pair_val_dataloader(
        batch_size=batch_size,
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(args.num_workers),
        pin_memory=bool(config.get("pin_memory", True)) and device.type == "cuda",
        download=bool(args.download),
    )
    val_n = len(val_loader.dataset)
    if int(args.num_images) <= 0:
        num_images = int(val_n)
        print(f"--num-images 0 → using full val set ({num_images})", flush=True)
    else:
        num_images = int(args.num_images)
        if num_images > val_n:
            print(
                f"WARNING: requested {num_images} images but val has {val_n}; "
                f"using {val_n}.",
                flush=True,
            )
            num_images = val_n

    # Ambient latent dim for this VAE (C·H·W); used only for warnings.
    ambient_d = 4 * 16 * 16  # 4096 for the standard LatentSR VAE
    if num_images <= ambient_d:
        print(
            f"WARNING: N={num_images} ≤ D={ambient_d} → sample covariance is "
            f"rank-deficient (rank ≤ N-1). Prefer --num-images 0 (full val).",
            flush=True,
        )

    twonn_subsample = int(args.twonn_subsample)
    twonn_bootstraps = int(args.twonn_bootstraps)
    if twonn_subsample >= num_images:
        print(
            f"WARNING: --twonn-subsample={twonn_subsample} >= N={num_images} → "
            "TwoNN returns a single full-sample estimate with std=0. "
            "Use e.g. --twonn-subsample 5000 for mean±std.",
            flush=True,
        )
    elif twonn_bootstraps < 2:
        print(
            "WARNING: --twonn-bootstraps < 2 → no TwoNN uncertainty.",
            flush=True,
        )

    print(
        f"num_images={num_images}  batch_size={batch_size}  "
        f"latent_scale={latent_scale}  num_workers={args.num_workers}",
        flush=True,
    )
    print(
        f"TwoNN bootstraps={twonn_bootstraps}  subsample={twonn_subsample}",
        flush=True,
    )
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"val size ≈ {val_n}  batches/epoch ≈ {len(val_loader)}", flush=True)

    report = run_representation_geometry(
        vae_a,
        vae_b,
        val_loader,
        device=device,
        num_images=num_images,
        output_dir=args.output_dir,
        hr_size=hr_size,
        latent_scale=latent_scale,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        twonn_bootstraps=twonn_bootstraps,
        twonn_subsample=twonn_subsample,
        seed=seed,
        show_progress=True,
    )
    print(flush=True)
    print(format_geometry_table(report), flush=True)
    print(f"\nWrote {args.output_dir / 'metrics.json'}", flush=True)
    print(
        "Plots: pca_*.png, transport_condition_number.png, excess_kurtosis_summary.png",
        flush=True,
    )


if __name__ == "__main__":
    main()
