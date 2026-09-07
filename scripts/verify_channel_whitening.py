"""Verify channel whitening on val LR latents (geometry before vs after).

Checks the RiT-inspired gate before quality eval:
  κ↓, effective rank↑, PCA less concentrated.

  python scripts/verify_channel_whitening.py \\
    --candidate-vae /kaggle/working/hf_ckpt/vae_sr/latest.pt \\
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
from latentsr.metrics.whitening_geometry import (
    compare_raw_vs_whitened_z_lr,
    format_whitening_geometry_block,
)
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.utils.config import get_device, load_config
from latentsr.vae.latent import load_frozen_vae
from latentsr.vae.whitening import ChannelWhitening


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
    any_fail = False
    for name, vae_path, whiten_path in pairs:
        if not vae_path.is_file():
            raise SystemExit(f"{name} VAE not found: {vae_path}")
        vae, _ = load_frozen_vae(vae_path, map_location=device)
        if whiten_path is None:
            raise SystemExit(
                f"{name}: pass --whiten-* so raw vs whitened geometry can be checked."
            )
        if not whiten_path.is_file():
            raise SystemExit(f"whitener not found: {whiten_path}")
        whitener = ChannelWhitening.load(whiten_path)
        print(f"\n=== {name} ===", flush=True)
        z = _collect_z_lr(
            vae,
            loader,
            device=device,
            num_images=int(args.num_images),
            hr_size=hr_size,
            latent_scale=float(args.latent_scale),
        )
        block = compare_raw_vs_whitened_z_lr(z, whitener, name=name)
        report["spaces"][name] = block
        print_blocks.append(format_whitening_geometry_block(block))
        if not block["geometry_ok"]:
            any_fail = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "whitening_verify.json").write_text(
        json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    table = "\n\n".join(print_blocks) + "\n"
    (out_dir / "whitening_verify.txt").write_text(table, encoding="utf-8")
    print("\n" + table, flush=True)
    print(f"Wrote {out_dir}", flush=True)
    if any_fail:
        raise SystemExit(
            "Geometry gate failed for at least one VAE — "
            "do not proceed to raw-vs-whitened quality eval yet."
        )


if __name__ == "__main__":
    main()
