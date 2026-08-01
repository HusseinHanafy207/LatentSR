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
pytest tests/test_imports.py -q
```

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
