# LatentSR

Code and technical report for **When Does Better Conditioning Help?**
Representation, injection, and sampling-time guidance in latent diffusion
super-resolution (CelebA faces, 32→128).

A convolutional VAE compresses images; a DDPM denoises the HR latent,
conditioned on the encoded upsampled LR image `z_lr`. This repo is a
controlled diagnosis of that pipeline, not a new SOTA SR architecture.

| | |
|---|---|
| Report | [Technical_report.pdf](Technical_report.pdf) · [LaTeX](paper/) |
| Checkpoints | [HusseinHamouda/LatentSR-checkpoints](https://huggingface.co/HusseinHamouda/LatentSR-checkpoints) |
| DDPM dependency | [`generative-models`](https://github.com/HusseinHanafy207/generative-models) (`generative_models.ddpm`) |

## Findings

Matched CelebA 4× protocol (frozen autoencoders, same UNet skeleton and noise schedule):

- An SR-aware VAE (**VAE-SR**) lifts soft-decode `Dec(z_lr)` by **+2.33 dB** (PSNR 26.15→28.48, LPIPS 0.273→0.119, n=64). Concat LatentSR on the same code gains only **+0.22 dB**. Spatial FiLM vs concat is a null.
- Mid-reverse, `ẑ₀` aligns with `z_lr` (cosine **0.985** at t=656 under VAE-SR); that alignment drops before t=0.
- Decoder-side LR-consistency guidance on the late window, with no extra training, recovers **56%** of the 2.08 dB transfer gap at λ_g=200 (n=256: 26.33→27.50 dB) and slightly improves LPIPS.

Soft-decode scores the *condition*. LatentSR scores the *sampler*. They are different jobs.

## Setup

Python ≥ 3.10.

```bash
git clone https://github.com/HusseinHanafy207/generative-models.git
pip install -e ./generative-models
pip install -e ".[dev,eval]"
```

```bash
python -c "from generative_models.ddpm import UNet, NoiseScheduler; print('DDPM OK')"
python -c "import latentsr; print(latentsr.__version__)"
pytest tests/ -q
```

YAML configs default to Kaggle or Colab paths. For a local machine, set `data_dir` and the output directories in the YAML, or pass `--data-dir` / `--output-dir` on eval scripts. CelebA downloads on first train unless you pass `--no-download`.

## Checkpoints

From [HusseinHamouda/LatentSR-checkpoints](https://huggingface.co/HusseinHamouda/LatentSR-checkpoints):

| Path | What |
|---|---|
| `vae/checkpoint_epoch_050.pt` | Reconstruction VAE (**VAE-1**) |
| `vae_sr/latest.pt` | SR-aware fine-tune (**VAE-SR**) |
| `latest.pt` (repo root) | Concat LatentSR trained on VAE-1 |
| `latent_sr_q2/latest.pt` | Concat LatentSR trained on VAE-SR |
| `latent_sr_adagn_q2/latest.pt` | Spatial FiLM on frozen VAE-SR (`condition_type: adagn`) |

Guidance in the report always pairs Q2 concat epoch-50 with VAE-SR epoch-20. Do not mix the VAE-1 concat checkpoint or the FiLM checkpoint into those runs.

## Reproduce the report tables

Eval reverse-diffusion noise (`x_T` and every reverse step) is seeded **per val index** from `--seed` (default 42), so two checkpoints see the same images and the same noise. Never pool n=64 and n=256.

**VAE / concat / FiLM / timestep / ẑ₀** use n=64. **Guidance confirmation** uses n=256 (`val_index` 0..255).

```bash
python scripts/evaluate.py \
  --checkpoint path/to/vae1_concat.pt \
  --vae-checkpoint path/to/vae/checkpoint_epoch_050.pt \
  --config configs/eval_sr.yaml \
  --data-dir data/raw --output-dir outputs/eval_sr_vae1_paired \
  --num-images 64 --batch-size 4 --seed 42 --no-download

python scripts/evaluate.py \
  --checkpoint path/to/latent_sr_q2/latest.pt \
  --vae-checkpoint path/to/vae_sr/latest.pt \
  --config configs/eval_sr.yaml \
  --data-dir data/raw --output-dir outputs/eval_sr_q2_paired \
  --num-images 64 --batch-size 4 --seed 42 --no-download

python scripts/compare_sr_evals.py \
  --baseline outputs/eval_sr_vae1_paired/per_image.csv \
  --candidate outputs/eval_sr_q2_paired/per_image.csv \
  --baseline-name vae1 --candidate-name vae_sr \
  --output-dir outputs/eval_sr_compare
```

Repeat the Q2 `evaluate.py` call with the FiLM checkpoint for the injection table.

```bash
python scripts/diagnose_timesteps.py \
  --config configs/eval_sr.yaml \
  --baseline-sr path/to/vae1_concat.pt \
  --baseline-vae path/to/vae/checkpoint_epoch_050.pt \
  --candidate-sr path/to/latent_sr_q2/latest.pt \
  --candidate-vae path/to/vae_sr/latest.pt \
  --output-dir outputs/eval_timestep_diagnostic \
  --num-images 64 --batch-size 4 --seed 42 --no-download

python scripts/diagnose_z0_recon.py \
  --config configs/eval_sr.yaml \
  --baseline-sr path/to/vae1_concat.pt \
  --baseline-vae path/to/vae/checkpoint_epoch_050.pt \
  --candidate-sr path/to/latent_sr_q2/latest.pt \
  --candidate-vae path/to/vae_sr/latest.pt \
  --output-dir outputs/eval_z0_recon \
  --num-images 64 --seed 42 --no-download
```

**Representation geometry** (RiT-style; VAE-1 vs VAE-SR, no diffusion).
Use the **full val set** (`--num-images 0`) so sample covariance is full-rank at D=4096, and keep `--twonn-subsample 5000` (strictly `< N`) so TwoNN reports real mean±std:

```bash
python scripts/diagnose_representation_geometry.py \
  --baseline-vae path/to/vae/checkpoint_epoch_050.pt \
  --candidate-vae path/to/vae_sr/latest.pt \
  --config configs/eval_vae.yaml \
  --data-dir data/raw --output-dir outputs/eval_representation_geometry \
  --num-images 0 --batch-size 32 --seed 42 --no-download \
  --twonn-bootstraps 10 --twonn-subsample 5000
```

**Collapse ↔ local geometry** (Phase 1.5; exploratory). Per-image
`C_i = cos(ẑ0(t_peak), z_lr) − cos(ẑ0(0), z_lr)` from the reverse chain, then
Pearson/Spearman vs leave-one-out k-NN local erank / κ / density around each `z_lr`.
`--reference-images` can exceed `--num-images` for a denser neighbor cloud:

```bash
python scripts/diagnose_collapse_geometry.py \
  --config configs/eval_sr.yaml \
  --baseline-sr path/to/vae1_sr/latest.pt \
  --baseline-vae path/to/vae/checkpoint_epoch_050.pt \
  --candidate-sr path/to/latent_sr_q2/latest.pt \
  --candidate-vae path/to/vae_sr/latest.pt \
  --output-dir outputs/eval_collapse_geometry \
  --num-images 64 --reference-images 512 --knn 32 \
  --batch-size 4 --seed 42 --no-download
```

**Guidance** (frozen VAE-SR + Q2 concat, late window). Confirmatory dose is four jobs, n=256, about 21 s/image with a decoder backward:

```bash
python scripts/run_guidance_n256.py --job baseline \
  --sr-checkpoint path/to/latent_sr_q2/latest.pt \
  --vae-checkpoint path/to/vae_sr/latest.pt \
  --config configs/eval_sr.yaml --data-dir data/raw --output-root outputs

python scripts/run_guidance_n256.py --job late50  --sr-checkpoint ... --vae-checkpoint ...
python scripts/run_guidance_n256.py --job late200 --sr-checkpoint ... --vae-checkpoint ...
python scripts/run_guidance_n256.py --job late800 --sr-checkpoint ... --vae-checkpoint ...
python scripts/run_guidance_n256.py --job compare --output-root outputs
```

Re-run the same command after a disconnect; already-finished `val_index` rows are skipped.

Paper figures (from cached CSVs, no resampling):

```bash
python scripts/plot_centerpiece.py
```

## Train from scratch

Point `data_dir` and output dirs in each YAML at local paths first. Configs ship with Kaggle/Colab paths.

```bash
python scripts/train_vae.py --config configs/vae_celeba.yaml
python scripts/train_vae_sr.py --config configs/vae_sr_align.yaml \
  --init-from outputs/vae/checkpoints/checkpoint_epoch_050.pt --no-download
python scripts/train_sr.py --config configs/latent_sr.yaml --no-download
python scripts/train_sr.py --config configs/latent_sr_q2.yaml \
  --vae-checkpoint outputs/vae_sr/checkpoints/latest.pt --no-download
python scripts/train_sr.py --config configs/latent_sr_adagn_q2.yaml \
  --vae-checkpoint outputs/vae_sr/checkpoints/latest.pt --no-download
```

Do not resume a concat checkpoint into FiLM (channel layout differs).
`Kaggle_Notebook.ipynb` is the FiLM (AdaGN) training notebook.

```bash
python scripts/super_resolve.py \
  --checkpoint outputs/latent_sr_q2/checkpoints/latest.pt \
  --vae-checkpoint outputs/vae_sr/checkpoints/latest.pt \
  --config configs/latent_sr_q2.yaml \
  --from-celeba --num-images 8 --no-download
```

## Layout

```
LatentSR/
├── paper/                 # technical report (LaTeX)
├── Technical_report.pdf
├── configs/
├── scripts/
├── src/latentsr/
├── tests/
├── Kaggle_Notebook.ipynb  # FiLM training on Kaggle
└── outputs/               # local runs (gitignored)
```

## License

MIT — see [LICENSE](LICENSE).
