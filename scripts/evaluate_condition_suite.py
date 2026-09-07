"""Condition ablation suite: same protocol as VAE-1 / VAE-SR evals.

Runs (per arm, shared val indices + seed):

  1) PSNR / LPIPS (evaluate_sr)
  2) Reverse-chain condition alignment → peak cos, t=0 cos, collapse
  3) Optional RiT geometry on z_lr (raw + whitened when configured)

Arms typically: Phase-8 (VAE-1), Q2 raw (VAE-SR), Q2 whitened (matched retrain).

Kaggle (full suite after whitened retrain):

  python scripts/evaluate_condition_suite.py \\
    --config configs/eval_sr.yaml \\
    --vae1-sr /kaggle/working/hf_ckpt/latest.pt \\
    --vae1-vae /kaggle/working/hf_ckpt/vae/checkpoint_epoch_050.pt \\
    --q2-sr /kaggle/working/hf_ckpt/latent_sr_q2/latest.pt \\
    --q2-vae /kaggle/working/hf_ckpt/vae_sr/latest.pt \\
    --q2-white-sr /kaggle/working/hf_ckpt/latent_sr_q2_whiten/latest.pt \\
    --q2-white-whiten /kaggle/working/outputs/whitening/vae_sr_channel_zca_eps1e-4.pt \\
    --output-dir /kaggle/working/outputs/eval_condition_suite \\
    --num-images 64 --batch-size 4 --seed 42 --device cuda --no-download \\
    --rit --rit-num-images 2048

Skip image metrics with ``--no-image-metrics``; skip RiT with default (no ``--rit``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders, get_sr_pair_val_dataloader
from latentsr.metrics.collapse_geometry import collapse_from_cosine_curve, reverse_cosine_curves
from latentsr.metrics.evaluate_sr import evaluate_sr
from latentsr.metrics.representation_geometry import (
    format_geometry_table,
    run_representation_geometry,
)
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config
from latentsr.vae.latent import is_frozen, load_frozen_vae
from latentsr.vae.whitening import ChannelWhitening


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PSNR/LPIPS + reverse alignment + optional RiT for condition arms."
    )
    p.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    p.add_argument("--vae1-sr", type=Path, required=True)
    p.add_argument("--vae1-vae", type=Path, required=True)
    p.add_argument("--q2-sr", type=Path, required=True)
    p.add_argument("--q2-vae", type=Path, required=True)
    p.add_argument(
        "--q2-white-sr",
        type=Path,
        default=None,
        help="Matched whitened LatentSR checkpoint (omit to skip white arm).",
    )
    p.add_argument(
        "--q2-white-whiten",
        type=Path,
        default=None,
        help="Channel whitener .pt used to train --q2-white-sr.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/eval_condition_suite"))
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--image-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode + PSNR/LPIPS on the same reverse samples as alignment.",
    )
    p.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--sr-eval-grids",
        action="store_true",
        help=(
            "Also run evaluate_sr (extra reverse) for bicubic/soft-decode "
            "comparison grids under each arm."
        ),
    )
    p.add_argument(
        "--rit",
        action="store_true",
        help="Also run RiT geometry (VAE-1 vs VAE-SR LR, + whitened LR if set).",
    )
    p.add_argument(
        "--rit-num-images",
        type=int,
        default=2048,
        help="Images for RiT (0 = full val). Separate from reverse-chain n.",
    )
    p.add_argument(
        "--rit-twonn-subsample",
        type=int,
        default=1000,
        help="TwoNN subsample (< rit-num-images).",
    )
    p.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p.parse_args()


def _require(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found:\n  {path}")


def _load_whitener(path: Path | None) -> ChannelWhitening | None:
    if path is None:
        return None
    _require(path, "whitener")
    return ChannelWhitening.load(path)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return "nan"
        if obj in (float("inf"), float("-inf")):
            return str(obj)
        return obj
    return obj


def _arm_alignment_block(
    cos: torch.Tensor,
    *,
    psnr: torch.Tensor | None,
    lpips: torch.Tensor | None,
) -> dict[str, Any]:
    """Match previous protocol: mean-curve peak + per-image mean collapse."""
    mean_curve = cos.mean(dim=0)
    t_peak = int(mean_curve.argmax().item())
    cos_peak = float(mean_curve[t_peak].item())
    cos_t0 = float(mean_curve[0].item())
    per_img = collapse_from_cosine_curve(cos)
    block: dict[str, Any] = {
        "t_peak": t_peak,
        "cos_peak": cos_peak,
        "cos_t0": cos_t0,
        "collapse": cos_peak - cos_t0,
        "collapse_per_image_mean": float(per_img["collapse"].mean().item()),
        "collapse_per_image_std": float(
            per_img["collapse"].std(unbiased=False).item()
        ),
        "cos_peak_per_image_mean": float(per_img["cos_peak"].mean().item()),
        "cos_t0_per_image_mean": float(per_img["cos_t0"].mean().item()),
    }
    if psnr is not None:
        block["psnr_mean"] = float(psnr.mean().item())
        block["psnr_std"] = float(psnr.std(unbiased=False).item())
    if lpips is not None:
        block["lpips_mean"] = float(lpips.mean().item())
        block["lpips_std"] = float(lpips.std(unbiased=False).item())
    return block


def _run_arm(
    name: str,
    *,
    sr_path: Path,
    vae_path: Path,
    whiten_path: Path | None,
    loader,
    device: torch.device,
    num_images: int,
    hr_size: int,
    seed: int,
    compute_image_metrics: bool,
    compute_lpips: bool,
    sr_eval_grids: bool,
    output_dir: Path,
) -> dict[str, Any]:
    print(f"\n=== arm={name} ===", flush=True)
    model, vae, meta = load_sr_components(
        sr_path,
        vae_checkpoint=vae_path,
        map_location=device,
        whiten_path=whiten_path,
    )
    whitener = _load_whitener(whiten_path)
    if whitener is None:
        whitener = meta.get("whitener")
    scale = float(meta["latent_scale"])
    print(
        f"epoch={meta.get('sr_epoch')}  latent_scale={scale}  "
        f"whiten={whitener is not None}",
        flush=True,
    )

    arm_dir = output_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)

    # One reverse: cosine alignment + optional PSNR/LPIPS (matches prior protocol).
    packed = reverse_cosine_curves(
        model,
        vae,
        loader,
        device=device,
        num_images=num_images,
        hr_size=hr_size,
        latent_scale=scale,
        noise_seed=seed,
        show_progress=True,
        compute_image_metrics=compute_image_metrics,
        compute_lpips=compute_lpips,
        whitener=whitener,
    )
    align = _arm_alignment_block(
        packed["cos"],
        psnr=packed.get("psnr"),
        lpips=packed.get("lpips"),
    )

    eval_out: dict[str, Any] | None = None
    if sr_eval_grids:
        eval_out = evaluate_sr(
            model,
            vae,
            loader,
            device=device,
            num_images=num_images,
            hr_size=hr_size,
            latent_scale=scale,
            compute_lpips=compute_lpips,
            include_soft_decode=True,
            show_progress=True,
            grid_images=min(8, num_images),
            output_dir=arm_dir / "sr_eval",
            noise_seed=seed,
            whitener=whitener,
        )
        print(eval_out["table"], flush=True)

    block = {
        "name": name,
        "sr_checkpoint": str(sr_path),
        "vae_checkpoint": str(vae_path),
        "whiten_path": str(whiten_path) if whiten_path else None,
        "whitener": whitener is not None,
        "sr_epoch": meta.get("sr_epoch"),
        "latent_scale": scale,
        "alignment": align,
        "evaluate_summary": None
        if eval_out is None
        else {
            "num_images": eval_out["num_images"],
            "summary": eval_out.get("summary"),
            "table": eval_out.get("table"),
        },
    }
    (arm_dir / "alignment.json").write_text(
        json.dumps(_json_safe(block), indent=2) + "\n",
        encoding="utf-8",
    )
    return block


def _format_suite_table(arms: list[dict[str, Any]]) -> str:
    lines = [
        "Condition suite (same seed / val prefix)",
        f"{'arm':<16} {'PSNR':>8} {'LPIPS':>8} {'t_peak':>7} "
        f"{'cos_peak':>9} {'cos_t0':>9} {'collapse':>9} {'whiten':>6}",
        "-" * 82,
    ]
    for arm in arms:
        a = arm["alignment"]
        psnr = a.get("psnr_mean")
        lpips = a.get("lpips_mean")
        lines.append(
            f"{arm['name']:<16} "
            f"{(psnr if psnr is not None else float('nan')):8.3f} "
            f"{(lpips if lpips is not None else float('nan')):8.4f} "
            f"{a['t_peak']:7d} {a['cos_peak']:9.4f} {a['cos_t0']:9.4f} "
            f"{a['collapse']:9.4f} {str(arm['whitener']):>6}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    seed = int(args.seed)
    torch.manual_seed(seed)

    for path, label in (
        (args.vae1_sr, "VAE-1 SR"),
        (args.vae1_vae, "VAE-1"),
        (args.q2_sr, "Q2 SR"),
        (args.q2_vae, "VAE-SR"),
    ):
        _require(path, label)
    if args.q2_white_sr is not None:
        _require(args.q2_white_sr, "Q2 whitened SR")
        if args.q2_white_whiten is None:
            raise SystemExit(
                "--q2-white-sr requires --q2-white-whiten "
                "(do not eval a whitened DDPM without its whitener)."
            )
        _require(args.q2_white_whiten, "Q2 whitener")

    data_dir = (
        str(args.data_dir)
        if args.data_dir is not None
        else config.get("data_dir", "data/raw")
    )
    if not Path(data_dir).exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader = get_sr_pair_dataloaders(
        batch_size=int(args.batch_size),
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    arms_spec = [
        ("vae1", args.vae1_sr, args.vae1_vae, None),
        ("q2_raw", args.q2_sr, args.q2_vae, None),
    ]
    if args.q2_white_sr is not None:
        arms_spec.append(
            ("q2_white", args.q2_white_sr, args.q2_vae, args.q2_white_whiten)
        )

    arms: list[dict[str, Any]] = []
    for name, sr, vae, whiten in arms_spec:
        arms.append(
            _run_arm(
                name,
                sr_path=sr,
                vae_path=vae,
                whiten_path=whiten,
                loader=val_loader,
                device=device,
                num_images=int(args.num_images),
                hr_size=hr_size,
                seed=seed,
                compute_image_metrics=bool(args.image_metrics),
                compute_lpips=bool(args.lpips),
                sr_eval_grids=bool(args.sr_eval_grids),
                output_dir=output_dir,
            )
        )

    suite: dict[str, Any] = {
        "num_images": int(args.num_images),
        "seed": seed,
        "device": str(device),
        "arms": arms,
        "note": (
            "Cosine / collapse use the condition each model sees "
            "(whitened iff whitener set). Soft-decode stays raw. "
            "Shared noise_seed for paired reverse chains."
        ),
    }

    if args.rit:
        print("\n=== RiT geometry (z_lr) ===", flush=True)
        rit_loader = get_sr_pair_val_dataloader(
            batch_size=max(8, int(args.batch_size)),
            data_dir=data_dir,
            hr_size=hr_size,
            lr_size=lr_size,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=bool(config.get("pin_memory", False)),
            download=args.download,
        )
        val_n = len(rit_loader.dataset)
        rit_n = int(args.rit_num_images)
        if rit_n <= 0:
            rit_n = val_n
        rit_n = min(rit_n, val_n)
        vae_a, _ = load_frozen_vae(args.vae1_vae, map_location=device)
        vae_b, _ = load_frozen_vae(args.q2_vae, map_location=device)
        if not is_frozen(vae_a) or not is_frozen(vae_b):
            raise SystemExit("VAEs must load frozen for RiT.")
        whitener_b = _load_whitener(args.q2_white_whiten)
        rit_dir = output_dir / "rit_geometry"
        report = run_representation_geometry(
            vae_a,
            vae_b,
            rit_loader,
            device=device,
            num_images=rit_n,
            output_dir=rit_dir,
            hr_size=hr_size,
            latent_scale=1.0,
            baseline_name="vae1",
            candidate_name="vae_sr",
            twonn_bootstraps=10,
            twonn_subsample=min(int(args.rit_twonn_subsample), max(rit_n - 1, 2)),
            seed=seed,
            show_progress=True,
            whitener_candidate_lr=whitener_b,
            include_raw_and_whitened_lr=whitener_b is not None,
        )
        print(format_geometry_table(report), flush=True)
        suite["rit"] = {
            "num_images": rit_n,
            "output_dir": str(rit_dir),
            "spaces": list(report["spaces"].keys()),
            "meta": report.get("meta"),
        }

    table = _format_suite_table(arms)
    print("\n" + table, flush=True)
    (output_dir / "suite_summary.txt").write_text(table + "\n", encoding="utf-8")
    (output_dir / "suite_summary.json").write_text(
        json.dumps(_json_safe(suite), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
