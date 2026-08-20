"""Timestep diagnostic: does concat LatentSR copy ``z_lr`` into ẑ0?

No training. Paired reverse sampling of VAE-1 vs VAE-SR concat DDPMs with the
same ``x_T`` and reverse-step noise as ``evaluate_sr`` (seed + ``val_index``).

Logged at every t (t=0 clean … t=T-1 noise):

    ||z_lr^SR − z_lr^VAE1||     (constant in t; condition gap)
    cos(ẑ0, z_lr)               (per model)
    ||ẑ0 − z_lr||               RMSE, per model
    ||ẑ0^SR − ẑ0^VAE1||         RMSE
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.inference import encode_lr_latents
from latentsr.super_resolution.sample import (
    _STEP_SALT,
    predict_x0_from_eps,
    seeded_noise_like,
)
from latentsr.vae.vae import VAE

_VAE1_COLOR = "#4c78a8"
_VAESR_COLOR = "#f58518"


def latent_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample RMSE over (C, H, W)."""
    return (a - b).pow(2).flatten(1).mean(dim=1).clamp_min(0.0).sqrt()


def latent_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


class _RunningMoments:
    def __init__(self, num_timesteps: int, keys: tuple[str, ...]) -> None:
        self.n = 0
        self._sum = {
            k: torch.zeros(num_timesteps, dtype=torch.float64) for k in keys
        }
        self._sq = {
            k: torch.zeros(num_timesteps, dtype=torch.float64) for k in keys
        }

    def add(self, t: int, key: str, values: torch.Tensor) -> None:
        v = values.detach().reshape(-1).double().cpu()
        self._sum[key][t] += v.sum()
        self._sq[key][t] += v.pow(2).sum()

    def mark_images(self, count: int) -> None:
        self.n += int(count)

    def mean_std(self, key: str) -> tuple[list[float], list[float]]:
        n = max(self.n, 1)
        mean = self._sum[key] / n
        var = (self._sq[key] / n - mean.pow(2)).clamp_min(0.0)
        return mean.tolist(), var.sqrt().tolist()


def _snapshot_t(
    t: int,
    *,
    acc: _RunningMoments,
    z0_a: torch.Tensor,
    z0_b: torch.Tensor,
    z_lr_a: torch.Tensor,
    z_lr_b: torch.Tensor,
    z_lr_rmse: torch.Tensor,
) -> None:
    acc.add(t, "z_lr_rmse", z_lr_rmse)
    acc.add(t, "cosine_z0_z_lr_vae1", latent_cosine(z0_a, z_lr_a))
    acc.add(t, "cosine_z0_z_lr_vaesr", latent_cosine(z0_b, z_lr_b))
    acc.add(t, "z0_z_lr_rmse_vae1", latent_rmse(z0_a, z_lr_a))
    acc.add(t, "z0_z_lr_rmse_vaesr", latent_rmse(z0_b, z_lr_b))
    acc.add(t, "z0_rmse_sr_vs_vae1", latent_rmse(z0_b, z0_a))


@torch.no_grad()
def run_timestep_diagnostic(
    model_a: ConditionalLatentDDPM,
    vae_a: VAE,
    model_b: ConditionalLatentDDPM,
    vae_b: VAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_images: int = 64,
    hr_size: int = 128,
    latent_scale_a: float = 1.0,
    latent_scale_b: float = 1.0,
    noise_seed: int = 42,
    start_index: int = 0,
    output_dir: str | Path | None = None,
    show_progress: bool = True,
    baseline_name: str = "vae1",
    candidate_name: str = "vae_sr",
) -> dict[str, Any]:
    """Paired reverse chain; accumulate ẑ0 metrics vs t.

    ``model_a`` / ``vae_a`` are VAE-1 concat LatentSR.
    ``model_b`` / ``vae_b`` are VAE-SR concat LatentSR.
    """
    if model_a.num_timesteps != model_b.num_timesteps:
        raise ValueError(
            f"timestep mismatch: {model_a.num_timesteps} vs {model_b.num_timesteps}"
        )
    num_t = int(model_a.num_timesteps)
    keys = (
        "z_lr_rmse",
        "cosine_z0_z_lr_vae1",
        "cosine_z0_z_lr_vaesr",
        "z0_z_lr_rmse_vae1",
        "z0_z_lr_rmse_vaesr",
        "z0_rmse_sr_vs_vae1",
    )
    acc = _RunningMoments(num_t, keys)

    for model, vae in ((model_a, vae_a), (model_b, vae_b)):
        model.eval()
        vae.eval()
        model.to(device)
        vae.to(device)

    remaining = max(int(num_images), 1)
    next_index = int(start_index)
    iterator = tqdm(loader, desc="timestep-diag", leave=False) if show_progress else loader

    for lr, _hr in iterator:
        if remaining <= 0:
            break
        take = min(lr.shape[0], remaining)
        lr = lr[:take].to(device)
        indices = list(range(next_index, next_index + take))

        z_lr_a = encode_lr_latents(
            vae_a, lr, hr_size=hr_size, latent_scale=latent_scale_a
        )
        z_lr_b = encode_lr_latents(
            vae_b, lr, hr_size=hr_size, latent_scale=latent_scale_b
        )
        z_lr_gap = latent_rmse(z_lr_b, z_lr_a)

        x_a = seeded_noise_like(z_lr_a, indices, base_seed=noise_seed, salt=0)
        x_b = x_a.clone()

        inner = range(num_t - 1, -1, -1)
        for t in inner:
            t_batch = torch.full((take,), t, device=device, dtype=torch.long)
            eps_a = model_a.predict_noise(x_a, t_batch, z_lr_a)
            eps_b = model_b.predict_noise(x_b, t_batch, z_lr_b)
            z0_a = predict_x0_from_eps(model_a.scheduler, x_a, t_batch, eps_a)
            z0_b = predict_x0_from_eps(model_b.scheduler, x_b, t_batch, eps_b)
            _snapshot_t(
                t,
                acc=acc,
                z0_a=z0_a,
                z0_b=z0_b,
                z_lr_a=z_lr_a,
                z_lr_b=z_lr_b,
                z_lr_rmse=z_lr_gap,
            )
            step_noise = seeded_noise_like(
                x_a,
                indices,
                base_seed=noise_seed,
                salt=_STEP_SALT * (int(t) + 1),
            )
            x_a = model_a.scheduler.p_sample_step(
                x_a, t_batch, eps_a, noise=step_noise
            )
            x_b = model_b.scheduler.p_sample_step(
                x_b, t_batch, eps_b, noise=step_noise
            )

        acc.mark_images(take)
        remaining -= take
        next_index += take

    curves = {key: acc.mean_std(key) for key in keys}
    rows = _rows_from_curves(num_t, acc.n, curves)
    result: dict[str, Any] = {
        "num_images": acc.n,
        "num_timesteps": num_t,
        "noise_seed": int(noise_seed),
        "start_index": int(start_index),
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "rows": rows,
        "highlights": _highlights(rows),
    }
    if output_dir is not None:
        result["paths"] = write_timestep_outputs(result, output_dir)
    return result


def _rows_from_curves(
    num_t: int,
    n: int,
    curves: dict[str, tuple[list[float], list[float]]],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for t in range(num_t):
        row: dict[str, float | int] = {"t": t, "n": n}
        for key, (mean, std) in curves.items():
            row[f"{key}_mean"] = mean[t]
            row[f"{key}_std"] = std[t]
        rows.append(row)
    return rows


def _highlights(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    if not rows:
        return {}
    picks = {0, rows[-1]["t"], rows[len(rows) // 2]["t"]}
    return {
        int(row["t"]): {
            k: row[k]
            for k in row
            if k not in {"t", "n"} and str(k).endswith("_mean")
        }
        for row in rows
        if int(row["t"]) in picks
    }


def write_timestep_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    csv_path = output_dir / "timestep_means.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path = output_dir / "timestep_diagnostic.json"
    serializable = {k: v for k, v in result.items() if k != "paths"}
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    table = format_timestep_table(result)
    txt_path = output_dir / "timestep_diagnostic.txt"
    txt_path.write_text(table + "\n", encoding="utf-8")

    plots = write_timestep_plots(result, output_dir)
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "table": str(txt_path),
        "plots": [str(p) for p in plots],
    }


def format_timestep_table(result: dict[str, Any]) -> str:
    n = result["num_images"]
    t_max = int(result["num_timesteps"]) - 1
    highlights = result.get("highlights") or {}
    lines = [
        f"n={n}  t=0 (clean) … t={t_max} (noise)",
        f"Δz_lr = ||z_lr[{result['candidate_name']}] − z_lr[{result['baseline_name']}]||  (RMSE)",
        "",
        f"{'t':>5}  {'Δz_lr':>8}  {'cos ẑ0,z_lr':^21}  {'||ẑ0−z_lr||':^21}  {'||ẑ0SR−ẑ0VAE1||':>16}",
        f"{'':>5}  {'':>8}  {'VAE-1':>10} {'VAE-SR':>10}  {'VAE-1':>10} {'VAE-SR':>10}",
        "-" * 88,
    ]
    for t in sorted(highlights):
        h = highlights[t]
        lines.append(
            f"{t:5d}  {h['z_lr_rmse_mean']:8.4f}  "
            f"{h['cosine_z0_z_lr_vae1_mean']:10.4f} {h['cosine_z0_z_lr_vaesr_mean']:10.4f}  "
            f"{h['z0_z_lr_rmse_vae1_mean']:10.4f} {h['z0_z_lr_rmse_vaesr_mean']:10.4f}  "
            f"{h['z0_rmse_sr_vs_vae1_mean']:16.4f}"
        )
    return "\n".join(lines)


def write_timestep_plots(
    result: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    rows = result["rows"]
    t = [int(r["t"]) for r in rows]
    written: list[Path] = []
    baseline = result["baseline_name"]
    candidate = result["candidate_name"]

    def _save(fig: plt.Figure, name: str) -> None:
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(
        t,
        [r["cosine_z0_z_lr_vae1_mean"] for r in rows],
        color=_VAE1_COLOR,
        label=rf"cos($\hat{{z}}_0$, $z_{{lr}}$) {baseline}",
    )
    ax.plot(
        t,
        [r["cosine_z0_z_lr_vaesr_mean"] for r in rows],
        color=_VAESR_COLOR,
        label=rf"cos($\hat{{z}}_0$, $z_{{lr}}$) {candidate}",
    )
    ax.set_xlabel("t  (0 = clean, T−1 = noise)")
    ax.set_ylabel("cosine")
    ax.set_title(r"Does $\hat{z}_0$ lock onto $z_{lr}$?")
    ax.legend(frameon=False)
    _save(fig, "cosine_z0_z_lr.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(
        t,
        [r["z0_z_lr_rmse_vae1_mean"] for r in rows],
        color=_VAE1_COLOR,
        label=rf"RMSE($\hat{{z}}_0$, $z_{{lr}}$) {baseline}",
    )
    ax.plot(
        t,
        [r["z0_z_lr_rmse_vaesr_mean"] for r in rows],
        color=_VAESR_COLOR,
        label=rf"RMSE($\hat{{z}}_0$, $z_{{lr}}$) {candidate}",
    )
    ax.set_xlabel("t  (0 = clean, T−1 = noise)")
    ax.set_ylabel("RMSE")
    ax.set_title(r"Distance from $\hat{z}_0$ to the LR code")
    ax.legend(frameon=False)
    _save(fig, "z0_z_lr_rmse.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(
        t,
        [r["z_lr_rmse_mean"] for r in rows],
        color="#888888",
        ls="--",
        label=r"RMSE($z_{lr}^{SR}$, $z_{lr}^{VAE1}$)",
    )
    ax.plot(
        t,
        [r["z0_rmse_sr_vs_vae1_mean"] for r in rows],
        color="#c44e52",
        label=r"RMSE($\hat{z}_0^{SR}$, $\hat{z}_0^{VAE1}$)",
    )
    ax.set_xlabel("t  (0 = clean, T−1 = noise)")
    ax.set_ylabel("RMSE")
    ax.set_title("Condition gap vs predicted-clean gap")
    ax.legend(frameon=False)
    _save(fig, "z0_delta_vs_z_lr_delta.png")

    return written
