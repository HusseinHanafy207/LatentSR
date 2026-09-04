"""Phase 1.5: late reverse-chain collapse vs local z_lr geometry.

Exploratory / correlational. Reuses the seeded reverse chain; adds per-image

    C_i = cos(ẑ0(t_peak), z_lr) − cos(ẑ0(0), z_lr)

leave-one-out k-NN local geometry, **PSNR/LPIPS on the same reverse samples**,
and bootstrap CIs on correlations (default 1000 image-level resamples).

Full Kaggle run:

  python scripts/diagnose_collapse_geometry.py \\
    --config configs/eval_sr.yaml \\
    --baseline-sr /kaggle/working/hf_ckpt/latest.pt \\
    --baseline-vae /kaggle/working/artifacts/vae/checkpoint_epoch_050.pt \\
    --candidate-sr /kaggle/working/artifacts/latent_sr_q2/latest.pt \\
    --candidate-vae /kaggle/working/outputs/vae_sr/checkpoints/latest.pt \\
    --output-dir /kaggle/working/outputs/eval_collapse_geometry \\
    --num-images 64 --reference-images 512 --knn 32 --n-boot 1000 --robustness \\
    --batch-size 4 --seed 42 --device cuda --no-download

Cheap bootstrap-only on an existing per-image CSV (no reverse):

  python scripts/diagnose_collapse_geometry.py \\
    --from-csv /kaggle/working/outputs/eval_collapse_geometry/collapse_geometry_per_image.csv \\
    --output-dir /kaggle/working/outputs/eval_collapse_geometry_boot \\
    --n-boot 1000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.metrics.collapse_geometry import (
    enrich_correlations_from_rows,
    format_collapse_geometry_table,
    load_per_image_collapse_csv,
    run_collapse_geometry,
    write_collapse_geometry_report,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse score C_i vs local z_lr geometry (exploratory)."
    )
    parser.add_argument("--baseline-sr", type=Path, default=None)
    parser.add_argument("--baseline-vae", type=Path, default=None)
    parser.add_argument("--candidate-sr", type=Path, default=None)
    parser.add_argument("--candidate-vae", type=Path, default=None)
    parser.add_argument("--baseline-name", type=str, default="vae1")
    parser.add_argument("--candidate-name", type=str, default="vae_sr")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help=(
            "Skip reverse chain; recompute bootstrap CIs (+ quality corrs if "
            "PSNR/LPIPS columns exist) from collapse_geometry_per_image.csv."
        ),
    )
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
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for correlation CIs (image-level; try 1000–5000).",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help=(
            "Also sweep knn∈{16,32,64} and reference_n∈{512,1024,2048} "
            "(encodes max ref once; no extra reverse)."
        ),
    )
    parser.add_argument(
        "--knn-grid",
        type=str,
        default=None,
        help="Comma list of knn values for robustness (overrides --robustness default).",
    )
    parser.add_argument(
        "--reference-grid",
        type=str,
        default=None,
        help="Comma list of reference sizes for robustness.",
    )
    parser.add_argument(
        "--image-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode final latents and correlate density with PSNR/LPIPS.",
    )
    parser.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include LPIPS in image-metric correlations (needs lpips pkg).",
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


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _run_from_csv(args: argparse.Namespace) -> None:
    csv_path = Path(args.from_csv)
    _require_file(csv_path, "per-image CSV")
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else csv_path.parent / "bootstrap_refresh"
    )
    rows = load_per_image_collapse_csv(csv_path)
    result = enrich_correlations_from_rows(
        rows,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
        knn=int(args.knn) if args.knn is not None else None,
        reference_n=None,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
    result["paths"] = write_collapse_geometry_report(result, output_dir)
    print(format_collapse_geometry_table(result), flush=True)
    print(f"\nWrote {output_dir}", flush=True)
    if result.get("paired_delta") is None:
        print(
            "NOTE: paired Δ analysis needs both models' PSNR + density columns. "
            "Re-run full diagnostic with --models both.",
            flush=True,
        )
    missing_q = []
    for name, block in result["models"].items():
        if "quality_correlations" not in block:
            missing_q.append(name)
    if missing_q:
        print(
            "NOTE: no PSNR/LPIPS columns for "
            + ", ".join(missing_q)
            + " — re-run the full diagnostic (without --from-csv) to check "
            "density vs actual output quality.",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.from_csv is not None:
        _run_from_csv(args)
        return

    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    torch.manual_seed(seed)

    for path, label in (
        (args.baseline_sr, "VAE-1 SR checkpoint"),
        (args.baseline_vae, "VAE-1 checkpoint"),
        (args.candidate_sr, "VAE-SR DDPM checkpoint"),
        (args.candidate_vae, "VAE-SR checkpoint"),
    ):
        if path is None:
            raise SystemExit(
                f"Missing {label}. Pass checkpoints or use --from-csv."
            )
        _require_file(path, label)

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
    knn_grid = _parse_int_list(args.knn_grid)
    reference_grid = _parse_int_list(args.reference_grid)
    if args.robustness:
        knn_grid = knn_grid or [16, 32, 64]
        reference_grid = reference_grid or [512, 1024, 2048]
    if reference_grid:
        reference_images = max(reference_images, max(reference_grid))
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
        f"n_boot={args.n_boot}  image_metrics={args.image_metrics}  "
        f"robustness knn={knn_grid} ref={reference_grid}  "
        f"seed={seed}  device={device}",
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
        n_boot=int(args.n_boot),
        compute_image_metrics=bool(args.image_metrics),
        compute_lpips=bool(args.lpips),
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        knn_grid=knn_grid,
        reference_grid=reference_grid,
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
