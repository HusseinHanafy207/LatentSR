"""Channel whitening unit tests."""

from __future__ import annotations

from pathlib import Path

import torch

from latentsr.datasets.onthefly_sr_latent import OnTheFlySRLatentEncoder
from latentsr.vae import VAE, freeze_vae
from latentsr.vae.whitening import (
    channel_covariance_stats,
    fit_channel_whitening,
)


def test_zca_makes_channel_cov_near_identity() -> None:
    torch.manual_seed(0)
    # Correlated 4-D channel field.
    a = torch.randn(32, 4, 8, 8)
    # Induce channel correlation: z_c = L @ eps
    l = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [0.5, 0.3, 0.7, 0.0],
            [0.2, 0.4, 0.3, 0.8],
        ]
    )
    z = torch.einsum("cd,bdhw->bchw", l, a)
    before = channel_covariance_stats(z)
    assert before["kappa"] > 5.0

    acc = fit_channel_whitening(num_channels=4, eps=1e-4, mode="zca")
    acc.update(z)
    w = acc.finalize()
    after = channel_covariance_stats(w.transform(z))
    assert after["kappa"] < 1.5
    assert after["effective_rank"] > before["effective_rank"]
    # Off-diagonal of whitened cov should be small.
    flat = w.transform(z).permute(0, 2, 3, 1).reshape(-1, 4).double()
    cov = torch.cov(flat.T)
    off = cov - torch.diag(torch.diag(cov))
    assert float(off.abs().max().item()) < 0.05


def test_standardize_mode_diagonal_only() -> None:
    torch.manual_seed(1)
    z = torch.randn(16, 4, 4, 4) * torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1)
    acc = fit_channel_whitening(num_channels=4, eps=1e-5, mode="standardize")
    acc.update(z)
    w = acc.finalize()
    assert w.mode == "standardize"
    # Matrix should be nearly diagonal.
    off = w.matrix - torch.diag(torch.diag(w.matrix))
    assert float(off.abs().max().item()) < 1e-8


def test_whitening_save_load(tmp_path: Path) -> None:
    torch.manual_seed(2)
    z = torch.randn(8, 4, 4, 4)
    acc = fit_channel_whitening(num_channels=4, eps=1e-4, mode="zca")
    acc.update(z)
    w = acc.finalize()
    path = w.save(tmp_path / "w.pt")
    from latentsr.vae.whitening import ChannelWhitening

    w2 = ChannelWhitening.load(path)
    out1 = w.transform(z)
    out2 = w2.transform(z)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_sr_encoder_whitens_condition_only() -> None:
    torch.manual_seed(3)
    vae = VAE(base_channels=32, num_res_blocks=1)
    freeze_vae(vae)
    z_dummy = torch.randn(4, 4, 8, 8)
    acc = fit_channel_whitening(num_channels=4, eps=1e-4, mode="zca")
    # Match whatever channel count the tiny VAE produces after encode.
    # Encode a probe to learn C.
    from latentsr.datasets.onthefly_sr_latent import upsample_bicubic
    from latentsr.vae.latent import encode_scaled

    lr = torch.rand(2, 3, 32, 32)
    hr = torch.rand(2, 3, 128, 128)
    probe = encode_scaled(vae, upsample_bicubic(lr, 128), latent_scale=1.0)
    c = int(probe.shape[1])
    acc = fit_channel_whitening(num_channels=c, eps=1e-4, mode="zca")
    acc.update(probe)
    # Inflate with more noise for a stable cov.
    for _ in range(5):
        acc.update(torch.randn_like(probe) + probe)
    whitener = acc.finalize()

    enc = OnTheFlySRLatentEncoder(vae, latent_scale=1.0, hr_size=128, whitener=whitener)
    z_lr_cond, z_hr = enc(lr, hr)
    z_lr_raw = enc.encode_lr_raw(lr)
    assert z_lr_cond.shape == z_hr.shape
    assert not torch.allclose(z_lr_cond, z_lr_raw)
    # Whitening must not touch HR path relative to a no-whiten encoder.
    enc_raw = OnTheFlySRLatentEncoder(vae, latent_scale=1.0, hr_size=128, whitener=None)
    _, z_hr2 = enc_raw(lr, hr)
    assert torch.allclose(z_hr, z_hr2)
