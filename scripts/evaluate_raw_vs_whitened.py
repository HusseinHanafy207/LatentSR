"""Decisive eval: raw VAE-SR condition vs whitened VAE-SR condition.

Order (research gate → quality):

  1) Whitening geometry on matched val ``z_lr``:
       κ↓, effective rank↑, PCA less concentrated
  2) Same images + same reverse-process noise:
       PSNR, LPIPS, reverse-chain alignment,
       peak cosine, t=0 cosine, collapse score

Kaggle:

  python scripts/evaluate_raw_vs_whitened.py \\
    --config configs/eval_sr.yaml \\
    --vae-sr /kaggle/working/hf_ckpt/vae_sr/latest.pt \\
    --raw-sr /kaggle/working/hf_ckpt/latent_sr_q2/latest.pt \\
    --white-sr /kaggle/working/hf_ckpt/latent_sr_q2_whiten/latest.pt \\
    --whiten /kaggle/working/outputs/whitening/vae_sr_channel_zca_eps1e-4.pt \\
    --output-dir /kaggle/working/outputs/eval_raw_vs_whitened \\
    --num-images 64 --geom-images 2048 --batch-size 4 --seed 42 \\
    --device cuda --no-download
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders, get_sr_pair_val_dataloader
from latentsr.metrics.collapse_geometry import collapse_from_cosine_curve, reverse_cosine_curves
from latentsr.metrics.timestep_diagnostic import (
    format_timestep_table,
    run_timestep_diagnostic,
)
from latentsr.metrics.whitening_geometry import (
    compare_raw_vs_whitened_z_lr,
    format_whitening_geometry_block,
)
from latentsr.super_resolution.inference import encode_lr_latents, load_sr_components
from latentsr.utils.config import get_device, load_config
from latentsr.vae.latent import load_frozen_vae
from latentsr.vae.whitening import ChannelWhitening


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raw vs whitened VAE-SR condition: geometry gate + paired metrics."
    )
    p.add_argument("--config", type=Path, default=Path("configs/eval_sr.yaml"))
    p.add_argument("--vae-sr", type=Path, required=True, help="Frozen VAE-SR checkpoint.")
    p.add_argument("--raw-sr", type=Path, required=True, help="Q2 concat trained on raw z_lr.")
    p.add_argument(
        "--white-sr",
        type=Path,
        required=True,
        help="Matched Q2 concat trained on whitened z_lr.",
    )
    p.add_argument(
        "--whiten",
        type=Path,
        required=True,
        help="Channel whitener .pt used to train --white-sr.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_raw_vs_whitened"),
    )
    p.add_argument("--num-images", type=int, default=64, help="Reverse-chain images.")
    p.add_argument(
        "--geom-images",
        type=int,
        default=2048,
        help="Val images for whitening geometry gate (encode only).",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--require-geometry-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort before reverse eval if κ↓/erank↑/PCA checks fail.",
    )
    p.add_argument(
        "--skip-paired-timestep",
        action="store_true",
        help="Skip lockstep timestep curves (still runs PSNR/LPIPS + collapse).",
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


@torch.no_grad()
def _collect_z_lr(
    vae,
    loader,
    *,
    device: torch.device,
    num_images: int,
    hr_size: int,
    latent_scale: float,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    remaining = max(int(num_images), 1)
    pbar = tqdm(total=remaining, desc="encode val z_lr", unit="img", leave=False)
    for lr, _hr in loader:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        z = encode_lr_latents(
            vae,
            lr[:take].to(device),
            hr_size=hr_size,
            latent_scale=latent_scale,
            apply_whiten=False,
        )
        chunks.append(z.cpu())
        remaining -= take
        pbar.update(take)
    pbar.close()
    if not chunks:
        raise SystemExit("No z_lr collected for geometry gate.")
    return torch.cat(chunks, dim=0)


def _alignment_from_cos(
    cos: torch.Tensor,
    *,
    psnr: torch.Tensor | None,
    lpips: torch.Tensor | None,
) -> dict[str, Any]:
    mean_curve = cos.mean(dim=0)
    t_peak = int(mean_curve.argmax().item())
    cos_peak = float(mean_curve[t_peak].item())
    cos_t0 = float(mean_curve[0].item())
    per_img = collapse_from_cosine_curve(cos)
    out: dict[str, Any] = {
        "t_peak": t_peak,
        "cos_peak": cos_peak,
        "cos_t0": cos_t0,
        "collapse": cos_peak - cos_t0,
        "collapse_per_image_mean": float(per_img["collapse"].mean().item()),
        "collapse_per_image_std": float(per_img["collapse"].std(unbiased=False).item()),
        "cos_peak_per_image_mean": float(per_img["cos_peak"].mean().item()),
        "cos_t0_per_image_mean": float(per_img["cos_t0"].mean().item()),
    }
    if psnr is not None:
        out["psnr_mean"] = float(psnr.mean().item())
        out["psnr_std"] = float(psnr.std(unbiased=False).item())
    if lpips is not None:
        out["lpips_mean"] = float(lpips.mean().item())
        out["lpips_std"] = float(lpips.std(unbiased=False).item())
    return out


def _delta_block(raw: dict[str, Any], white: dict[str, Any]) -> dict[str, float | None]:
    keys = (
        "psnr_mean",
        "lpips_mean",
        "cos_peak",
        "cos_t0",
        "collapse",
        "t_peak",
    )
    out: dict[str, float | None] = {}
    for k in keys:
        if k in raw and k in white and raw[k] is not None and white[k] is not None:
            out[f"delta_{k}"] = float(white[k]) - float(raw[k])
        else:
            out[f"delta_{k}"] = None
    return out


def _format_decisive_table(
    geometry: dict[str, Any],
    raw_align: dict[str, Any],
    white_align: dict[str, Any],
    delta: dict[str, float | None],
) -> str:
    lines = [
        "Decisive comparison: raw VAE-SR condition vs whitened VAE-SR condition",
        f"geometry_ok={geometry['geometry_ok']}  "
        f"(κ↓={geometry['ok_channel_kappa_dropped']}  "
        f"erank↑={geometry['ok_channel_erank_rose']}  "
        f"PCA↓conc={geometry['ok_pca_less_concentrated']})",
        "",
        f"{'arm':<12} {'PSNR':>8} {'LPIPS':>8} {'t_peak':>7} "
        f"{'cos_peak':>9} {'cos_t0':>9} {'collapse':>9}",
        "-" * 72,
    ]
    for name, a in (("q2_raw", raw_align), ("q2_white", white_align)):
        lines.append(
            f"{name:<12} "
            f"{a.get('psnr_mean', float('nan')):8.3f} "
            f"{a.get('lpips_mean', float('nan')):8.4f} "
            f"{a['t_peak']:7d} {a['cos_peak']:9.4f} {a['cos_t0']:9.4f} "
            f"{a['collapse']:9.4f}"
        )
    lines.append("-" * 72)
    lines.append(
        f"{'Δ(white−raw)':<12} "
        f"{(delta.get('delta_psnr_mean') or float('nan')):8.3f} "
        f"{(delta.get('delta_lpips_mean') or float('nan')):8.4f} "
        f"{(delta.get('delta_t_peak') or float('nan')):7.0f} "
        f"{(delta.get('delta_cos_peak') or float('nan')):9.4f} "
        f"{(delta.get('delta_cos_t0') or float('nan')):9.4f} "
        f"{(delta.get('delta_collapse') or float('nan')):9.4f}"
    )
    lines.append("")
    lines.append(
        "Hypothesis support: lower collapse and/or higher t=0 cosine under "
        "whitening (with PSNR/LPIPS not collapsing) after geometry_ok=True."
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
        (args.vae_sr, "VAE-SR"),
        (args.raw_sr, "raw Q2 SR"),
        (args.white_sr, "whitened Q2 SR"),
        (args.whiten, "channel whitener"),
    ):
        _require(path, label)

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

    whitener = ChannelWhitening.load(args.whiten)

    # ------------------------------------------------------------------
    # 1) Geometry gate (encode only; no diffusion)
    # ------------------------------------------------------------------
    print("=== 1/3 whitening geometry gate ===", flush=True)
    geom_loader = get_sr_pair_val_dataloader(
        batch_size=max(8, int(args.batch_size)),
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        download=bool(args.download),
    )
    vae_geom, _ = load_frozen_vae(args.vae_sr, map_location=device)
    z_lr = _collect_z_lr(
        vae_geom,
        geom_loader,
        device=device,
        num_images=int(args.geom_images),
        hr_size=hr_size,
        latent_scale=float(config.get("latent_scale", 1.0)),
    )
    geometry = compare_raw_vs_whitened_z_lr(z_lr, whitener, name="vae_sr")
    geom_txt = format_whitening_geometry_block(geometry)
    print(geom_txt, flush=True)
    (output_dir / "whitening_geometry.txt").write_text(geom_txt + "\n", encoding="utf-8")
    (output_dir / "whitening_geometry.json").write_text(
        json.dumps(_json_safe(geometry), indent=2) + "\n",
        encoding="utf-8",
    )
    if args.require_geometry_ok and not geometry["geometry_ok"]:
        raise SystemExit(
            "Geometry gate failed (need κ↓, erank↑, PCA less concentrated). "
            "Fix whitening before interpreting quality. "
            "Re-run with --no-require-geometry-ok only to dump metrics anyway."
        )

    # ------------------------------------------------------------------
    # 2) Load matched DDPMs (same VAE-SR; white arm uses whitener)
    # ------------------------------------------------------------------
    print("\n=== 2/3 load raw + whitened LatentSR ===", flush=True)
    model_raw, vae_raw, meta_raw = load_sr_components(
        args.raw_sr,
        vae_checkpoint=args.vae_sr,
        map_location=device,
        whiten_path=None,
    )
    model_white, vae_white, meta_white = load_sr_components(
        args.white_sr,
        vae_checkpoint=args.vae_sr,
        map_location=device,
        whiten_path=args.whiten,
    )
    if meta_raw.get("whitener") is not None:
        print(
            "WARNING: raw checkpoint embeds a whitener path; forcing raw condition off.",
            flush=True,
        )
    scale_raw = float(meta_raw["latent_scale"])
    scale_white = float(meta_white["latent_scale"])
    print(
        f"raw:   epoch={meta_raw.get('sr_epoch')} scale={scale_raw} whiten=False",
        flush=True,
    )
    print(
        f"white: epoch={meta_white.get('sr_epoch')} scale={scale_white} whiten=True",
        flush=True,
    )

    _, val_loader = get_sr_pair_dataloaders(
        batch_size=int(args.batch_size),
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        download=args.download,
    )

    paired = None
    if not args.skip_paired_timestep:
        print("\n=== 2b/3 paired reverse-chain alignment (shared x_T) ===", flush=True)
        paired = run_timestep_diagnostic(
            model_raw,
            vae_raw,
            model_white,
            vae_white,
            val_loader,
            device=device,
            num_images=int(args.num_images),
            hr_size=hr_size,
            latent_scale_a=scale_raw,
            latent_scale_b=scale_white,
            whitener_a=None,
            whitener_b=whitener,
            noise_seed=seed,
            output_dir=output_dir / "timestep_paired",
            show_progress=True,
            baseline_name="q2_raw",
            candidate_name="q2_white",
        )
        print(format_timestep_table(paired), flush=True)

    # ------------------------------------------------------------------
    # 3) PSNR / LPIPS + collapse on same seed (per-arm reverse + decode)
    # ------------------------------------------------------------------
    print("\n=== 3/3 PSNR/LPIPS + collapse (shared noise seed) ===", flush=True)
    packed_raw = reverse_cosine_curves(
        model_raw,
        vae_raw,
        val_loader,
        device=device,
        num_images=int(args.num_images),
        hr_size=hr_size,
        latent_scale=scale_raw,
        noise_seed=seed,
        show_progress=True,
        compute_image_metrics=True,
        compute_lpips=bool(args.lpips),
        whitener=None,
    )
    packed_white = reverse_cosine_curves(
        model_white,
        vae_white,
        val_loader,
        device=device,
        num_images=int(args.num_images),
        hr_size=hr_size,
        latent_scale=scale_white,
        noise_seed=seed,
        show_progress=True,
        compute_image_metrics=True,
        compute_lpips=bool(args.lpips),
        whitener=whitener,
    )
    raw_align = _alignment_from_cos(
        packed_raw["cos"],
        psnr=packed_raw.get("psnr"),
        lpips=packed_raw.get("lpips"),
    )
    white_align = _alignment_from_cos(
        packed_white["cos"],
        psnr=packed_white.get("psnr"),
        lpips=packed_white.get("lpips"),
    )
    # Prefer lockstep mean-curve alignment when available (identical protocol).
    if paired is not None and paired.get("alignment"):
        for name, align in (
            ("q2_raw", raw_align),
            ("q2_white", white_align),
        ):
            block = paired["alignment"].get(name)
            if block:
                align["t_peak"] = int(block["t_peak"])
                align["cos_peak"] = float(block["cos_peak"])
                align["cos_t0"] = float(block["cos_t0"])
                align["collapse"] = float(block["collapse"])

    delta = _delta_block(raw_align, white_align)
    table = _format_decisive_table(geometry, raw_align, white_align, delta)
    print("\n" + table, flush=True)

    report = {
        "comparison": "raw_vae_sr_condition_vs_whitened_vae_sr_condition",
        "num_images": int(args.num_images),
        "geom_images": int(args.geom_images),
        "seed": seed,
        "device": str(device),
        "geometry": geometry,
        "q2_raw": {
            "sr_checkpoint": str(args.raw_sr),
            "sr_epoch": meta_raw.get("sr_epoch"),
            "whitener": False,
            "metrics": raw_align,
        },
        "q2_white": {
            "sr_checkpoint": str(args.white_sr),
            "sr_epoch": meta_white.get("sr_epoch"),
            "whitener": True,
            "whiten_path": str(args.whiten),
            "metrics": white_align,
        },
        "delta_white_minus_raw": delta,
        "note": (
            "Same val prefix and noise_seed. Cosine vs the condition each model "
            "sees. Soft-decode / encode gap stay in raw space. Geometry gate "
            "must pass before interpreting quality deltas."
        ),
    }
    (output_dir / "decisive_summary.txt").write_text(table + "\n", encoding="utf-8")
    (output_dir / "decisive_summary.json").write_text(
        json.dumps(_json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
