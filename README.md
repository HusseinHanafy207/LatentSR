# LatentSR

PyTorch **Latent Diffusion** for **Image Super-Resolution** on CelebA faces.

```
latentsr  →  imports  →  generative_models.ddpm
```

| Setting | Value |
|---------|--------|
| Dataset | CelebA (center-crop 178 → resize) |
| HR / LR | **128×128** / **32×32** (4×) |
| Latents | Conv VAE → `(4, 32, 32)`, on-the-fly encode + `latent_scale` |
| Diffusion | Reuses `generative_models.ddpm` (UNet, schedule, loss) |

---

## Phases

| Phase | Goal |
|-------|------|
| 0 | Package scaffold + `generative-models` dependency |
| 1 | Train convolutional VAE on CelebA 128 |
| 2 | Freeze VAE; `encode_scaled` / `decode_scaled` |
| 3 | On-the-fly image → scaled latent (no disk cache yet) |
| 4 | Unconditional latent DDPM |
| 5 | Verify LDM: noise → latent → decode → image |
| 6 | LR/HR pixel pairs |
| 7 | On-the-fly `(z_lr, z_hr)` pairs |
| 8 | Concat-conditioned latent SR diffusion |
| 9 | LR → HR inference CLI |
| 10 | PSNR / SSIM / LPIPS vs bicubic |
| 11 | Deferred (cache, DDIM, better conditioning, …) |

---

## Setup

This project depends on [`generative-models`](https://github.com/HusseinHanafy207/generative-models) (specifically `generative_models.ddpm`). Clone it and install in editable mode, then LatentSR:

```bash
git clone https://github.com/HusseinHanafy207/generative-models.git
pip install -e ./generative-models
pip install -e ".[dev]"
```

If `generative-models` already lives next to this repo, point `pip` at that checkout instead:

```bash
pip install -e /path/to/generative-models
pip install -e ".[dev]"
```

Smoke check:

```bash
python -c "from generative_models.ddpm import UNet, NoiseScheduler; print('DDPM OK')"
python -c "import latentsr; print(latentsr.__version__)"
pytest tests/ -q
```

### Train the VAE (Phase 1)

```bash
# Sanity epoch (downloads CelebA on first run ~1.4GB)
python scripts/train_vae.py --epochs 1

# Full run
python scripts/train_vae.py --config configs/vae_celeba.yaml

# Reconstructions / latent interpolations
python scripts/recon_vae.py --checkpoint outputs/vae/checkpoints/latest.pt --interpolate
```

### Frozen VAE (Phase 2)

```bash
python scripts/verify_frozen_vae.py \
  --checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt
```

Later phases must load with ``load_frozen_vae`` and use ``encode_scaled`` /
``decode_scaled`` (default ``latent_scale=1.0``).

### On-the-fly latents (Phase 3)

No disk cache — encode each batch live:

```bash
python scripts/verify_onthefly_latents.py \
  --checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt
```

### Latent DDPM (Phase 4)

```bash
# Sanity epoch
python scripts/train_latent_ddpm.py --config configs/latent_ddpm.yaml --epochs 1 --no-download

# Full run / resume
python scripts/train_latent_ddpm.py --config configs/latent_ddpm.yaml --no-download
python scripts/train_latent_ddpm.py --resume outputs/latent_ddpm/checkpoints/latest.pt --epochs 50 --no-download
```

### Sample LDM (Phase 5)

```bash
python scripts/sample_ldm.py \
  --checkpoint /content/drive/MyDrive/LatentSR/outputs/latent_ddpm/checkpoints/latest.pt \
  --num-samples 16 \
  --device cuda
```

Inspect `…/outputs/latent_ddpm/samples/ldm_samples.png` — faces should be recognizable.

### SR pixel pairs (Phase 6)

```bash
python scripts/visualize_sr_pairs.py --config configs/sr_pairs.yaml --no-download
```

Grid rows: nearest(LR) | bicubic(LR→HR) | HR.

### SR latent pairs (Phase 7)

```bash
python scripts/verify_sr_latents.py --config configs/onthefly_sr_latent.yaml --no-download
```

Rows: HR | decode(z_hr) | decode(z_lr) | bicubic(LR→HR).

### Conditional Latent SR (Phase 8)

```bash
!mkdir -p /content/drive/MyDrive/LatentSR/outputs/latent_sr/{checkpoints,samples,logs}
!python scripts/train_sr.py --config configs/latent_sr.yaml --epochs 1 --device cuda --no-download
!python scripts/train_sr.py --config configs/latent_sr.yaml --device cuda --no-download
# Resume:
!python scripts/train_sr.py --config configs/latent_sr.yaml \
  --resume /content/drive/MyDrive/LatentSR/outputs/latent_sr/checkpoints/latest.pt \
  --epochs 50 --device cuda --no-download
```

### SR inference (Phase 9)

```bash
# CelebA val grid: nearest(LR) | bicubic | LatentSR | HR
python scripts/super_resolve.py \
  --checkpoint outputs/latent_sr/checkpoints/latest.pt \
  --vae-checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt \
  --config configs/latent_sr.yaml \
  --from-celeba --num-images 8 --no-download

# Single image / folder (32×32 LR or 128×128 will be downsampled)
python scripts/super_resolve.py \
  --checkpoint outputs/latent_sr/checkpoints/latest.pt \
  --input path/to/face.png \
  --output outputs/latent_sr/samples/sr_out.png
```

Add `--include-soft-decode` to also show `decode(z_lr)` in the comparison grid.

### Evaluate vs bicubic (Phase 10)

```bash
pip install lpips   # once, for LPIPS

python scripts/evaluate.py \
  --checkpoint outputs/latent_sr/checkpoints/latest.pt \
  --vae-checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt \
  --config configs/eval_sr.yaml \
  --num-images 64 --no-download
```

Writes under `output_dir` (default `outputs/eval/`):
- `metrics.csv` / `metrics.json` — mean±std PSNR, SSIM, LPIPS for **bicubic** vs **LatentSR**
- `eval_compare.png` — nearest(LR) | bicubic | LatentSR | HR

Use `--no-lpips` if you skip the `lpips` install. Full reverse diffusion per image is slow; start with `--num-images 16`.

### VAE bottleneck (research Phase A)

No diffusion — measures what the frozen VAE already loses:

```bash
python scripts/evaluate_vae.py \
  --vae-checkpoint outputs/vae/checkpoints/checkpoint_epoch_050.pt \
  --config configs/eval_vae.yaml \
  --num-images 64 --no-download
```

Writes under `output_dir` (default `outputs/eval_vae/`):
- `metrics.csv` / `metrics.json` — PSNR, SSIM, LPIPS, Sobel edge MAE, radial FFT-band error for **bicubic**, **decode(z_lr)**, **decode(z_hr)**
- `std(μ)` and suggested `latent_scale = 1 / std(μ_HR)`
- `eval_vae_compare.png` — nearest(LR) | bicubic | decode(z_lr) | decode(z_hr) | HR

Use `--no-lpips` if you skip the `lpips` install. Compare `decode(z_hr)` to the Phase 10 LatentSR PSNR (~26.25) to decide whether the VAE is the bottleneck.

### SR-aware VAE (research Q2)

Phase A showed `decode(z_hr)` PSNR ~37.3 while LatentSR is ~26.3 and `decode(z_lr)` ≈ bicubic. Fine-tune VAE-1 so `μ_lr` moves toward `sg(μ_hr)` without retraining a new autoencoder from scratch:

```bash
python scripts/train_vae_sr.py \
  --config configs/vae_sr_align.yaml \
  --init-from outputs/vae/checkpoints/checkpoint_epoch_050.pt \
  --no-download
```

Then re-run `scripts/evaluate_vae.py` on the new checkpoint. Watch **cosine(z_lr, z_hr)** (baseline ~0.63) and **soft_decode PSNR**; `vae_hr` PSNR should stay high. Hugging Face uploads go under `vae_sr/` so they do not overwrite the LatentSR DDPM `latest.pt`.

### Matched LatentSR on VAE-SR (Q2)

Same UNet, schedule, and 50 epochs as Phase 8; only the frozen VAE changes. HF uploads go under `latent_sr_q2/`.

```bash
python scripts/train_sr.py \
  --config configs/latent_sr_q2.yaml \
  --vae-checkpoint outputs/vae_sr/checkpoints/latest.pt \
  --epochs 1 --device cuda --no-download
```

Full run / resume:

```bash
python scripts/train_sr.py --config configs/latent_sr_q2.yaml --device cuda --no-download
python scripts/train_sr.py --config configs/latent_sr_q2.yaml \
  --resume path/to/latent_sr_q2/latest.pt --epochs 50 --device cuda --no-download
```

Then evaluate with `scripts/evaluate.py` pointing at the Q2 DDPM and VAE-SR checkpoints.

---

## Layout

```
LatentSR/
├── configs/
├── scripts/
├── src/latentsr/
│   ├── datasets/
│   ├── vae/
│   ├── diffusion/
│   ├── super_resolution/
│   ├── metrics/
│   └── utils/
├── tests/
└── outputs/
```

---

## License

MIT — see [LICENSE](LICENSE).
