"""Verify channel whitening on val LR latents (geometry before vs after).

Compares raw vs whitened z_lr under VAE-1 and/or VAE-SR. Checks that the
4×4 channel covariance κ drops and erank rises; also reports ambient
flattened PCA erank / κ for the RiT-style view.

  python scripts/verify_channel_whitening.py \\
    --baseline-vae /kaggle/working/artifacts/vae/checkpoint_epoch_050.pt \\
    --candidate-vae /kaggle/working/outputs/vae_sr/checkpoints/latest.pt \\
    --whiten-baseline /kaggle/working/outputs/whitening/vae1_channel_zca_eps1e-4.pt \\
    --whiten-candidate /kaggle/working/outputs/whitening/vae_sr_channel_zca_eps1e-4.pt \\
    --config configs/eval_vae.yaml --num-images 2048 --no-download
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from latentsr.datasets.sr_pairs import get_sr_pair_val_dataloader
from latentsr.metrics.representation_geometry import (
    covariance_condition_number,
    effective_rank,
    flatten_latents,
    pca_eigenvalues,
)
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.utils.config import get_device, load_config
from latentsr.vae.latent import load_frozen_vae
from latentsr.vae.whitening import ChannelWhitening, channel_covariance_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify channel whitening geometry on val z_lr."
    )
    parser.add_argument("--baseline-vae", type=Path, default=None)
    parser.add_argument("--candidate-vae", type=Path, default=None)
    parser.add_argument("--baseline-name", type=str, default="vae1")
    parser.add_argument("--candidate-name", type=str, default="vae_sr")
    parser.add_argument("--whiten-baseline", type=Path, default=None)
    parser.add_argument("--whiten-candidate", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_vae.yaml"))
    parser.add_argument("--num-images", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_whitening_verify"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _ambient_stats(z: torch.Tensor) -> dict[str, float]:
    flat = flatten_latents(z)
    eigs = pca_eigenvalues(flat)
    kappa = covariance_condition_number(eigs, num_samples=flat.shape[0])
    return {
        "ambient_erank": effective_rank(eigs),
        "ambient_kappa": kappa["kappa"],
        "ambient_var_top1": float((eigs[0] / eigs.sum().clamp_min(1e-30)).item()),
        "num_images": float(flat.shape[0]),
        "ambient_dim": float(flat.shape[1]),
    }


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
        raise SystemExit("No latents collected.")
    return torch.cat(chunks, dim=0)


def _pack(name: str, z_raw: torch.Tensor, whitener: ChannelWhitening | None) -> dict[str, Any]:
    raw_ch = channel_covariance_stats(z_raw)
    raw_amb = _ambient_stats(z_raw)
    out: dict[str, Any] = {
        "name": name,
        "raw_channel": raw_ch,
        "raw_ambient": raw_amb,
    }
    if whitener is None:
        out["warning"] = "no whitener provided"
        return out
    z_w = whitener.transform(z_raw)
    white_ch = channel_covariance_stats(z_w)
    white_amb = _ambient_stats(z_w)
    out["whitened_channel"] = white_ch
    out["whitened_ambient"] = white_amb
    out["delta"] = {
        "channel_kappa": white_ch["kappa"] - raw_ch["kappa"],
        "channel_erank": white_ch["effective_rank"] - raw_ch["effective_rank"],
        "ambient_kappa": white_amb["ambient_kappa"] - raw_amb["ambient_kappa"],
        "ambient_erank": white_amb["ambient_erank"] - raw_amb["ambient_erank"],
    }
    out["ok_channel_kappa_dropped"] = bool(white_ch["kappa"] < raw_ch["kappa"])
    out["ok_channel_erank_rose"] = bool(
        white_ch["effective_rank"] > raw_ch["effective_rank"] - 1e-6
    )
    return out


def _fmt_block(block: dict[str, Any]) -> str:
    lines = [f"[{block['name']}]"]
    raw_ch = block["raw_channel"]
    lines.append(
        f"  raw  channel κ={raw_ch['kappa']:.3g}  erank={raw_ch['effective_rank']:.3f}  "
        f"top1%={100*raw_ch['var_top1']:.1f}"
    )
    raw_a = block["raw_ambient"]
    lines.append(
        f"  raw  ambient κ={raw_a['ambient_kappa']:.3g}  erank={raw_a['ambient_erank']:.1f}"
    )
    if "whitened_channel" not in block:
        lines.append(f"  WARNING: {block.get('warning', 'missing whitener')}")
        return "\n".join(lines)
    w_ch = block["whitened_channel"]
    w_a = block["whitened_ambient"]
    lines.append(
        f"  white channel κ={w_ch['kappa']:.3g}  erank={w_ch['effective_rank']:.3f}  "
        f"top1%={100*w_ch['var_top1']:.1f}"
    )
    lines.append(
        f"  white ambient κ={w_a['ambient_kappa']:.3g}  erank={w_a['ambient_erank']:.1f}"
    )
    lines.append(
        f"  checks: κ↓={block['ok_channel_kappa_dropped']}  "
        f"erank↑={block['ok_channel_erank_rose']}"
    )
    if not block["ok_channel_kappa_dropped"]:
        lines.append("  FAIL: channel κ did not decrease — debug whitening before training.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    torch.manual_seed(int(args.seed))
    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))
    data_dir = (
        str(args.data_dir)
        if args.data_dir is not None
        else config.get("data_dir", "data/raw")
    )
    if not Path(data_dir).exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    pairs: list[tuple[str, Path, Path | None]] = []
    if args.baseline_vae is not None:
        pairs.append((args.baseline_name, args.baseline_vae, args.whiten_baseline))
    if args.candidate_vae is not None:
        pairs.append((args.candidate_name, args.candidate_vae, args.whiten_candidate))
    if not pairs:
        raise SystemExit("Pass at least one of --baseline-vae / --candidate-vae")

    loader = get_sr_pair_val_dataloader(
        batch_size=int(args.batch_size),
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
        download=bool(args.download),
    )

    report: dict[str, Any] = {
        "num_images": int(args.num_images),
        "latent_scale": float(args.latent_scale),
        "spaces": {},
    }
    print_blocks: list[str] = []
    for name, vae_path, whiten_path in pairs:
        if not vae_path.is_file():
            raise SystemExit(f"{name} VAE not found: {vae_path}")
        vae, _ = load_frozen_vae(vae_path, map_location=device)
        whitener = ChannelWhitening.load(whiten_path) if whiten_path else None
        if whiten_path and whitener is None:
            raise SystemExit(f"Failed to load whitener: {whiten_path}")
        print(f"\n=== {name} ===", flush=True)
        z = _collect_z_lr(
            vae,
            loader,
            device=device,
            num_images=int(args.num_images),
            hr_size=hr_size,
            latent_scale=float(args.latent_scale),
        )
        block = _pack(name, z, whitener)
        report["spaces"][name] = block
        print_blocks.append(_fmt_block(block))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "whitening_verify.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    table = "\n\n".join(print_blocks) + "\n"
    (out_dir / "whitening_verify.txt").write_text(table, encoding="utf-8")
    print("\n" + table, flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
