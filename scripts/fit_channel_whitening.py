"""Fit channel-wise ZCA whitening on **training** LR latents only.

Treat each spatial site as a 4-D sample. Estimate μ, Σ on CelebA train,
freeze W=(Σ+εI)^{-1/2}, never refit on val/test.

  python scripts/fit_channel_whitening.py \\
    --vae-checkpoint /kaggle/working/outputs/vae_sr/checkpoints/latest.pt \\
    --config configs/latent_sr_q2.yaml \\
    --output /kaggle/working/outputs/whitening/vae_sr_channel_zca_eps1e-4.pt \\
    --eps 1e-4 --mode zca --max-images 50000 --batch-size 64 --no-download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from latentsr.datasets.sr_pairs import get_sr_pair_dataloaders
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.utils.config import get_device, load_config
from latentsr.vae.latent import load_frozen_vae
from latentsr.vae.whitening import channel_covariance_stats, fit_channel_whitening


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit channel ZCA / standardize whitening on train z_lr."
    )
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/latent_sr_q2.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument(
        "--mode",
        type=str,
        default="zca",
        choices=("zca", "standardize"),
        help="zca = full 4×4 whitening; standardize = diagonal only.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50_000,
        help="Cap on train images (full train ≈ 162k; 50k is usually enough).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None and args.config.exists():
        config = load_config(args.config)

    device = get_device(args.device or str(config.get("device", "auto")))
    torch.manual_seed(int(args.seed))

    if not args.vae_checkpoint.is_file():
        raise SystemExit(f"VAE checkpoint not found: {args.vae_checkpoint}")

    hr_size = int(config.get("hr_size", 128))
    lr_size = int(config.get("lr_size", 32))
    latent_scale = float(
        args.latent_scale
        if args.latent_scale is not None
        else config.get("latent_scale", 1.0)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else config.get("batch_size", 64)
    )
    data_dir = (
        str(args.data_dir)
        if args.data_dir is not None
        else config.get("data_dir", "data/raw")
    )
    if not Path(data_dir).exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    vae, _ = load_frozen_vae(args.vae_checkpoint, map_location=device)
    train_loader, _val = get_sr_pair_dataloaders(
        batch_size=batch_size,
        data_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)) and device.type == "cuda",
        download=args.download,
    )

    print(
        f"Fitting {args.mode} whitening on TRAIN z_lr only  "
        f"eps={args.eps}  max_images={args.max_images}",
        flush=True,
    )
    print(f"VAE: {args.vae_checkpoint}  latent_scale={latent_scale}", flush=True)
    print(f"train size ≈ {len(train_loader.dataset)}", flush=True)

    acc = fit_channel_whitening(
        num_channels=4,
        eps=float(args.eps),
        mode=str(args.mode),
        meta={
            "fit_split": "train",
            "vae_checkpoint": str(args.vae_checkpoint),
            "latent_scale": latent_scale,
            "hr_size": hr_size,
            "lr_size": lr_size,
            "seed": int(args.seed),
            "max_images": int(args.max_images),
        },
    )

    remaining = max(int(args.max_images), 1)
    seen = 0
    pbar = tqdm(total=remaining, desc="fit train z_lr", unit="img")
    for lr, _hr in train_loader:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        z = encode_lr_latents(
            vae, lr, hr_size=hr_size, latent_scale=latent_scale, apply_whiten=False
        )
        acc.update(z.cpu())
        remaining -= take
        seen += take
        pbar.update(take)
    pbar.close()

    whitener = acc.finalize()
    out = whitener.save(args.output)
    raw_stats = {
        "n_channel_vectors": whitener.meta.get("n_channel_vectors"),
        "cov_condition_raw": whitener.meta.get("cov_condition_raw"),
    }
    # Quick check: transform a small held-out train batch already in memory.
    z_check = encode_lr_latents(
        vae,
        lr,
        hr_size=hr_size,
        latent_scale=latent_scale,
        apply_whiten=False,
    )
    before = channel_covariance_stats(z_check.cpu())
    after = channel_covariance_stats(whitener.transform(z_check).cpu())

    print(f"\nWrote {out}", flush=True)
    print(f"fit vectors: {raw_stats['n_channel_vectors']}", flush=True)
    print(
        f"channel κ (last batch raw→white): "
        f"{before['kappa']:.3g} → {after['kappa']:.3g}",
        flush=True,
    )
    print(
        f"channel erank (last batch raw→white): "
        f"{before['effective_rank']:.3f} → {after['effective_rank']:.3f}",
        flush=True,
    )
    print(
        "Next: verify on val with scripts/verify_channel_whitening.py, "
        "then train matched DDPM with configs/latent_sr_q2_whiten.yaml",
        flush=True,
    )


if __name__ == "__main__":
    main()
