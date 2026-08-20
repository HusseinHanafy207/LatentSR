"""Decode-ẑ0 image diagnostic at selected timesteps."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from latentsr.metrics.z0_recon_diagnostic import run_z0_recon_diagnostic
from latentsr.super_resolution.condition import build_conditioned_latent_ddpm_from_config
from latentsr.vae import VAE, freeze_vae


def _tiny_concat_config() -> dict:
    return {
        "condition_type": "concat",
        "latent_channels": 4,
        "latent_size": 32,
        "base_channels": 32,
        "channel_mult": [1, 2],
        "num_res_blocks": 1,
        "attention_resolutions": [16],
        "dropout": 0.0,
        "num_timesteps": 4,
        "beta_start": 1e-4,
        "beta_end": 0.02,
    }


def test_z0_recon_writes_selected_timesteps(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    clone = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    clone.load_state_dict(model.state_dict())
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)

    lr = torch.rand(3, 3, 32, 32)
    hr = torch.rand(3, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)
    result = run_z0_recon_diagnostic(
        model,
        vae,
        clone,
        vae,
        loader,
        device=torch.device("cpu"),
        timesteps=(3, 0),
        num_images=3,
        hr_size=128,
        lr_size=32,
        noise_seed=42,
        output_dir=tmp_path,
        show_progress=False,
        compute_lpips=False,
        grid_images=2,
    )
    assert result["num_images"] == 3
    assert result["timesteps"] == [3, 0]
    assert [row["t"] for row in result["rows"]] == [3, 0]
    for row in result["rows"]:
        assert row["delta_psnr_mean"] == pytest.approx(0.0, abs=1e-4)
        assert row["vae1_psnr_mean"] == pytest.approx(row["vae_sr_psnr_mean"], abs=1e-4)
        assert "vae1_lr_psnr_mean" in row
        assert "vae1_lpips_mean" not in row
    assert (tmp_path / "z0_recon_means.csv").is_file()
    assert (tmp_path / "z0_recon_t_000.png").is_file()
    assert (tmp_path / "z0_recon_t_003.png").is_file()
    assert (tmp_path / "z0_recon_delta_psnr.png").is_file()
    indices = {(row["val_index"], row["t"]) for row in result["per_image"]}
    assert indices == {(0, 3), (0, 0), (1, 3), (1, 0), (2, 3), (2, 0)}


def test_z0_recon_rejects_out_of_range_t() -> None:
    model = build_conditioned_latent_ddpm_from_config(_tiny_concat_config())
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    lr = torch.rand(1, 3, 32, 32)
    hr = torch.rand(1, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=1)
    with pytest.raises(ValueError, match="outside"):
        run_z0_recon_diagnostic(
            model,
            vae,
            model,
            vae,
            loader,
            device=torch.device("cpu"),
            timesteps=(800,),
            num_images=1,
            show_progress=False,
            compute_lpips=False,
            grid_images=0,
        )
