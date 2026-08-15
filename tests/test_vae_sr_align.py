"""Q2: SR-aware VAE alignment loss and trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from latentsr.vae import VAE, SRAwareVAELoss, SRAwareVAETrainer, VAELoss


def test_sraware_loss_requires_mu_lr() -> None:
    criterion = SRAwareVAELoss()
    recon = torch.rand(2, 3, 16, 16)
    images = torch.rand(2, 3, 16, 16)
    mu = torch.randn(2, 4, 4, 4)
    logvar = torch.zeros(2, 4, 4, 4)
    try:
        criterion(recon, images, mu, logvar)
    except ValueError as exc:
        assert "mu_lr" in str(exc)
    else:
        raise AssertionError("expected ValueError when mu_lr is omitted")


def test_sraware_align_zero_matches_vae_loss() -> None:
    recon = torch.rand(2, 3, 16, 16)
    images = torch.rand(2, 3, 16, 16)
    mu = torch.randn(2, 4, 4, 4)
    logvar = torch.zeros(2, 4, 4, 4)
    mu_lr = torch.randn(2, 4, 4, 4)
    base = VAELoss(kl_weight=1e-4)
    aligned = SRAwareVAELoss(kl_weight=1e-4, align_weight=0.0)
    total_a, recon_a, kl_a = base(recon, images, mu, logvar)
    total_b, recon_b, kl_b, align_b = aligned(recon, images, mu, logvar, mu_lr)
    assert torch.allclose(total_a, total_b)
    assert torch.allclose(recon_a, recon_b)
    assert torch.allclose(kl_a, kl_b)
    assert align_b.ndim == 0


def test_sraware_stopgrad_on_mu_hr() -> None:
    criterion = SRAwareVAELoss(kl_weight=0.0, align_weight=1.0)
    recon = torch.rand(2, 3, 8, 8)
    images = recon.detach()
    mu_hr = torch.randn(2, 4, 2, 2, requires_grad=True)
    mu_lr = torch.randn(2, 4, 2, 2, requires_grad=True)
    logvar = torch.zeros(2, 4, 2, 2)
    total, _, _, align = criterion(recon, images, mu_hr, logvar, mu_lr)
    total.backward()
    # align uses sg(μ_hr); 0 * KL may still allocate a zero grad on μ_hr.
    if mu_hr.grad is not None:
        assert torch.allclose(mu_hr.grad, torch.zeros_like(mu_hr.grad))
    assert mu_lr.grad is not None
    assert float(mu_lr.grad.abs().sum().item()) > 0
    assert torch.isfinite(align)


def test_sraware_trainer_one_epoch(tmp_path: Path) -> None:
    config = {
        "epochs": 1,
        "amp": False,
        "device": "cpu",
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "sample_dir": str(tmp_path / "samples"),
        "log_dir": str(tmp_path / "logs"),
        "hr_size": 128,
        "validate_every": 1,
        "checkpoint_every": 1,
        "reconstruct_every": 1,
        "seed": 0,
        "align_weight": 0.001,
        "kl_weight": 1e-4,
    }
    vae = VAE(base_channels=32, num_res_blocks=1)
    lr = torch.rand(4, 3, 32, 32)
    hr = torch.rand(4, 3, 128, 128)
    loader = DataLoader(TensorDataset(lr, hr), batch_size=2)
    trainer = SRAwareVAETrainer(
        model=vae,
        optimizer=torch.optim.Adam(vae.parameters(), lr=1e-3),
        criterion=SRAwareVAELoss(kl_weight=1e-4, align_weight=1e-3),
        train_loader=loader,
        val_loader=loader,
        config=config,
    )
    trainer.train()
    latest = Path(config["checkpoint_dir"]) / "latest.pt"
    assert latest.is_file()
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1
    assert "align_loss" in ckpt["metrics"]
    assert "z_cosine" in ckpt["metrics"]
    assert list(tmp_path.joinpath("samples").glob("sr_align_epoch_*.png"))


def test_sraware_hf_upload_every_epoch(tmp_path: Path, monkeypatch) -> None:
    uploaded: list[str] = []

    class _FakeApi:
        def upload_file(self, **kwargs):
            uploaded.append(kwargs["path_in_repo"])

    from types import ModuleType
    import sys

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    config = {
        "epochs": 2,
        "amp": False,
        "device": "cpu",
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "sample_dir": str(tmp_path / "samples"),
        "log_dir": str(tmp_path / "logs"),
        "hr_size": 128,
        "validate_every": 1,
        "checkpoint_every": 1,
        "reconstruct_every": 1,
        "seed": 0,
        "hf_checkpoint_repo": "user/test-repo",
        "hf_checkpoint_subdir": "vae_sr",
    }
    vae = VAE(base_channels=32, num_res_blocks=1)
    loader = DataLoader(
        TensorDataset(torch.rand(4, 3, 32, 32), torch.rand(4, 3, 128, 128)),
        batch_size=2,
    )
    trainer = SRAwareVAETrainer(
        model=vae,
        optimizer=torch.optim.Adam(vae.parameters(), lr=1e-3),
        criterion=SRAwareVAELoss(kl_weight=1e-4, align_weight=1e-3),
        train_loader=loader,
        val_loader=loader,
        config=config,
    )
    trainer.train()

    assert "vae_sr/latest.pt" in uploaded
    assert "vae_sr/checkpoint_epoch_001.pt" in uploaded
    assert "vae_sr/checkpoint_epoch_002.pt" in uploaded
    assert "vae_sr/logs/train_metrics.csv" in uploaded
    assert "vae_sr/logs/val_metrics.csv" in uploaded
    assert "vae_sr/samples/sr_align_epoch_001.png" in uploaded
    assert "vae_sr/samples/sr_align_epoch_002.png" in uploaded
