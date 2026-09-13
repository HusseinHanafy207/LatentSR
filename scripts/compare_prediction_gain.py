"""Compare representation gain G under ε-pred vs x0-pred.

Decisive quantity:

    G_eps = PSNR(VAE-SR, ε) − PSNR(VAE-1, ε)     ≈ +0.22 dB (already known)
    G_x0  = PSNR(VAE-SR, x0) − PSNR(VAE-1, x0)

If G_x0 >> G_eps while the VAEs are unchanged, better condition information
was usable — transfer depended on the denoising target parameterization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from latentsr.datasets.sr_pairs import get_sr_pair_val_dataloader
from latentsr.metrics.evaluate_sr import evaluate_sr
from latentsr.super_resolution.inference import load_sr_components
from latentsr.utils.config import get_device, load_config
from latentsr.vae.whitening import ChannelWhitening


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute G_eps vs G_x0 representation-gain comparison."
    )
    p.add_argument("--eps-vae1-sr", type=Path, required=True, help="eps-pred + VAE-1 SR ckpt")
    p.add_argument("--eps-vae1-vae", type=Path, required=True, help="VAE-1 for eps-pred")
    p.add_argument("--eps-vaesr-sr", type=Path, required=True, help="eps-pred + VAE-SR SR ckpt")
    p.add_argument("--eps-vaesr-vae", type=Path, required=True, help="VAE-SR for eps-pred")
    p.add_argument("--x0-vae1-sr", type=Path, required=True, help="x0-pred + VAE-1 SR ckpt")
    p.add_argument("--x0-vae1-vae", type=Path, required=True, help="VAE-1 for x0-pred")
    p.add_argument("--x0-vaesr-sr", type=Path, required=True, help="x0-pred + VAE-SR SR ckpt")
    p.add_argument("--x0-vaesr-vae", type=Path, required=True, help="VAE-SR for x0-pred")
    p.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/prediction_gain"),
    )
    p.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _require(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found:\n  {path}")


def _mean(summary: dict[str, Any], method: str, metric: str) -> float:
    return float(summary[method][metric]["mean"])


def _eval_one(
    *,
    label: str,
    sr_ckpt: Path,
    vae_ckpt: Path,
    val_loader: Any,
    device: torch.device,
    num_images: int,
    hr_size: int,
    noise_seed: int,
    compute_lpips: bool,
    output_dir: Path,
) -> dict[str, Any]:
    model, vae, meta = load_sr_components(
        sr_ckpt, vae_checkpoint=vae_ckpt, map_location=device
    )
    pred_type = meta.get("prediction_type", getattr(model, "prediction_type", "eps"))
    print(
        f"\n>>> Evaluating {label}  "
        f"prediction_type={pred_type}  epoch={meta.get('sr_epoch')}"
    )
    run_dir = output_dir / label
    result = evaluate_sr(
        model,
        vae,
        val_loader,
        device=device,
        num_images=num_images,
        hr_size=hr_size,
        latent_scale=float(meta["latent_scale"]),
        compute_lpips=compute_lpips,
        show_progress=True,
        output_dir=run_dir,
        noise_seed=noise_seed,
        whitener=meta.get("whitener"),
    )
    return {
        "label": label,
        "prediction_type": pred_type,
        "sr_checkpoint": str(sr_ckpt),
        "vae_checkpoint": str(vae_ckpt),
        "sr_epoch": meta.get("sr_epoch"),
        "psnr": _mean(result["summary"], "latentsr", "psnr"),
        "ssim": _mean(result["summary"], "latentsr", "ssim"),
        "lpips": _mean(result["summary"], "latentsr", "lpips")
        if compute_lpips
        else float("nan"),
        "summary": result["summary"],
        "num_images": result["num_images"],
    }


def format_gain_table(rows: dict[str, dict[str, Any]]) -> str:
    """rows keys: eps_vae1, eps_vaesr, x0_vae1, x0_vaesr."""
    g_eps = rows["eps_vaesr"]["psnr"] - rows["eps_vae1"]["psnr"]
    g_x0 = rows["x0_vaesr"]["psnr"] - rows["x0_vae1"]["psnr"]
    headers = ["Cell", "PSNR", "SSIM", "LPIPS"]
    widths = [22, 10, 10, 10]
    lines = [
        " | ".join(h.center(w) for h, w in zip(headers, widths)),
        "-+-".join("-" * w for w in widths),
    ]
    labels = [
        ("eps_vae1", "ε-pred + VAE-1"),
        ("eps_vaesr", "ε-pred + VAE-SR"),
        ("x0_vae1", "x0-pred + VAE-1"),
        ("x0_vaesr", "x0-pred + VAE-SR"),
    ]
    for key, name in labels:
        r = rows[key]
        fields = [
            name,
            f"{r['psnr']:.3f}",
            f"{r['ssim']:.4f}",
            f"{r['lpips']:.4f}" if r["lpips"] == r["lpips"] else "n/a",
        ]
        lines.append(" | ".join(f.rjust(w) for f, w in zip(fields, widths)))

    lines.append("")
    lines.append(f"G_eps = PSNR(VAE-SR, ε)  - PSNR(VAE-1, ε)  = {g_eps:+.3f} dB")
    lines.append(f"G_x0  = PSNR(VAE-SR, x0) - PSNR(VAE-1, x0) = {g_x0:+.3f} dB")
    lines.append(f"ΔG    = G_x0 - G_eps                          = {g_x0 - g_eps:+.3f} dB")
    return "\n".join(lines)


def generate_gain_report(rows: dict[str, dict[str, Any]]) -> str:
    table = format_gain_table(rows)
    g_eps = rows["eps_vaesr"]["psnr"] - rows["eps_vae1"]["psnr"]
    g_x0 = rows["x0_vaesr"]["psnr"] - rows["x0_vae1"]["psnr"]
    delta = g_x0 - g_eps

    if delta >= 0.5:
        verdict = (
            f"STRONG: G_x0 ({g_x0:+.3f} dB) substantially exceeds G_eps ({g_eps:+.3f} dB). "
            "Better VAE-SR condition information was usable; transfer depended on the "
            "denoising target parameterization (x0 vs ε). This is a deeper RiT-style "
            "connection than whitening alone."
        )
    elif delta >= 0.15:
        verdict = (
            f"MODERATE: G_x0 ({g_x0:+.3f} dB) is larger than G_eps ({g_eps:+.3f} dB) "
            f"by {delta:+.3f} dB. x0-prediction improves condition transfer somewhat."
        )
    else:
        verdict = (
            f"NULL / SMALL: G_x0 ({g_x0:+.3f} dB) is similar to G_eps ({g_eps:+.3f} dB). "
            "RiT-style target parameterization alone does not unlock much more of the "
            "VAE-SR side-information gain under this matched loss."
        )

    return f"""================================================================================
PREDICTION-TYPE REPRESENTATION GAIN (G_eps vs G_x0)
================================================================================
Protocol: identical eval images, seed, sampler; only prediction_type and VAE differ.
Loss for x0 models was ε-matched: L = ||eps_hat(z0_hat) - eps||^2.

{table}

--- VERDICT ---
{verdict}
================================================================================
"""


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    seed = int(args.seed)
    torch.manual_seed(seed)

    pairs = [
        ("eps-VAE1 SR", args.eps_vae1_sr),
        ("eps-VAE1 VAE", args.eps_vae1_vae),
        ("eps-VAESR SR", args.eps_vaesr_sr),
        ("eps-VAESR VAE", args.eps_vaesr_vae),
        ("x0-VAE1 SR", args.x0_vae1_sr),
        ("x0-VAE1 VAE", args.x0_vae1_vae),
        ("x0-VAESR SR", args.x0_vaesr_sr),
        ("x0-VAESR VAE", args.x0_vaesr_vae),
    ]
    for label, path in pairs:
        _require(path, label)

    data_dir = args.data_dir or (
        Path(config["data_dir"]) if "data_dir" in config else None
    )
    if data_dir is None:
        raise SystemExit("Must supply --data-dir or set data_dir in config.")

    num_images = int(
        args.num_images if args.num_images is not None else config.get("num_images", 64)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else config.get("batch_size", 4)
    )
    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))

    print("=" * 80)
    print("G_eps vs G_x0 REPRESENTATION-GAIN COMPARISON")
    print(f"Images: {num_images} | Seed: {seed} | Device: {device}")
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "eps_vae1": (args.eps_vae1_sr, args.eps_vae1_vae),
        "eps_vaesr": (args.eps_vaesr_sr, args.eps_vaesr_vae),
        "x0_vae1": (args.x0_vae1_sr, args.x0_vae1_vae),
        "x0_vaesr": (args.x0_vaesr_sr, args.x0_vaesr_vae),
    }
    rows: dict[str, dict[str, Any]] = {}
    for key, (sr_ckpt, vae_ckpt) in specs.items():
        rows[key] = _eval_one(
            label=key,
            sr_ckpt=sr_ckpt,
            vae_ckpt=vae_ckpt,
            val_loader=val_loader,
            device=device,
            num_images=num_images,
            hr_size=hr_size,
            noise_seed=seed,
            compute_lpips=args.lpips,
            output_dir=args.output_dir,
        )

    g_eps = rows["eps_vaesr"]["psnr"] - rows["eps_vae1"]["psnr"]
    g_x0 = rows["x0_vaesr"]["psnr"] - rows["x0_vae1"]["psnr"]

    summary = {
        "num_images": num_images,
        "seed": seed,
        "cells": {
            k: {
                "psnr": v["psnr"],
                "ssim": v["ssim"],
                "lpips": v["lpips"],
                "prediction_type": v["prediction_type"],
                "sr_checkpoint": v["sr_checkpoint"],
                "vae_checkpoint": v["vae_checkpoint"],
                "sr_epoch": v["sr_epoch"],
            }
            for k, v in rows.items()
        },
        "G_eps_dB": g_eps,
        "G_x0_dB": g_x0,
        "delta_G_dB": g_x0 - g_eps,
    }

    json_path = args.output_dir / "prediction_gain_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = args.output_dir / "prediction_gain_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cell",
                "prediction_type",
                "psnr",
                "ssim",
                "lpips",
                "sr_checkpoint",
                "vae_checkpoint",
            ],
        )
        writer.writeheader()
        for key, v in rows.items():
            writer.writerow(
                {
                    "cell": key,
                    "prediction_type": v["prediction_type"],
                    "psnr": v["psnr"],
                    "ssim": v["ssim"],
                    "lpips": v["lpips"],
                    "sr_checkpoint": v["sr_checkpoint"],
                    "vae_checkpoint": v["vae_checkpoint"],
                }
            )

    report = generate_gain_report(rows)
    report_path = args.output_dir / "prediction_gain_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
