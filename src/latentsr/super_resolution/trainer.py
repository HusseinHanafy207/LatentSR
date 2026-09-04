"""Train conditional latent DDPM for super-resolution (Phase 8).

Each step:

    (lr, hr) → OnTheFlySRLatentEncoder → (z_lr, z_hr)
    noise_pred, noise, t = ConditionalLatentDDPM(z_hr, z_lr)
    loss = MSE(noise_pred, noise)
"""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from latentsr.datasets.onthefly_sr_latent import OnTheFlySRLatentEncoder, upsample_bicubic
from latentsr.super_resolution.condition import ConditionalLatentDDPM
from latentsr.super_resolution.sample import sample_sr_images
from latentsr.utils.amp import autocast_context, make_grad_scaler
from latentsr.utils.config import get_device
from latentsr.vae.latent import decode_scaled


class LatentSRTrainer:
    """Conditional latent DDPM trainer for 4× super-resolution."""

    TRAIN_FIELDS = ["epoch", "loss", "lr", "epoch_time"]
    VAL_FIELDS = ["epoch", "loss", "epoch_time"]

    def __init__(
        self,
        model: ConditionalLatentDDPM,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        sr_encoder: OnTheFlySRLatentEncoder,
        config: dict[str, Any],
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sr_encoder = sr_encoder
        self.config = config

        self.device = get_device(str(config.get("device", "auto")))
        self.model.to(self.device)
        self.sr_encoder.to(self.device)

        self.use_amp = bool(config.get("amp", False)) and self.device.type == "cuda"
        self.scaler = make_grad_scaler(enabled=self.use_amp)

        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.sample_dir = Path(config["sample_dir"])
        self.log_dir = Path(config["log_dir"])
        for d in (self.checkpoint_dir, self.sample_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.train_metrics_path = self.log_dir / config.get(
            "train_metrics_file", "train_metrics.csv"
        )
        self.val_metrics_path = self.log_dir / config.get(
            "val_metrics_file", "val_metrics.csv"
        )
        self.current_epoch = 0
        self._init_csv_logs()

    def _init_csv_logs(self) -> None:
        if not self.train_metrics_path.exists():
            with self.train_metrics_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.TRAIN_FIELDS).writeheader()
        if self.val_loader is not None and not self.val_metrics_path.exists():
            with self.val_metrics_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.VAL_FIELDS).writeheader()

    def _append_csv_row(
        self, path: Path, fieldnames: list[str], row: dict[str, Any]
    ) -> None:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        num_batches = 0
        first_batch_loss: float | None = None
        last_batch_loss: float | None = None

        context = torch.enable_grad() if training else torch.no_grad()
        phase = "train" if training else "val"
        progress = tqdm(loader, desc=phase, leave=False, dynamic_ncols=True)

        with context:
            for batch in progress:
                z_lr, z_hr = self.sr_encoder.encode_batch(batch, device=self.device)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast_context(
                    enabled=self.use_amp, device_type=self.device.type
                ):
                    noise_pred, noise, _t = self.model(z_hr, z_lr)
                    loss = self.criterion(noise_pred, noise)

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                batch_loss = float(loss.item())
                if first_batch_loss is None:
                    first_batch_loss = batch_loss
                last_batch_loss = batch_loss
                total_loss += batch_loss
                num_batches += 1
                if num_batches % 10 == 0 or num_batches == 1:
                    progress.set_postfix(loss=f"{batch_loss:.4f}")

        metrics = {"loss": total_loss / max(num_batches, 1)}
        if training:
            metrics["first_batch_loss"] = first_batch_loss or 0.0
            metrics["last_batch_loss"] = last_batch_loss or 0.0
        return metrics

    def train_epoch(self) -> dict[str, float]:
        start = time.time()
        metrics = self._run_epoch(self.train_loader, training=True)
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["epoch_time"] = time.time() - start
        return metrics

    def validate(self) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        start = time.time()
        metrics = self._run_epoch(self.val_loader, training=False)
        metrics["epoch_time"] = time.time() - start
        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        *,
        save_snapshot: bool = True,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "vae_checkpoint": self.config.get("vae_checkpoint"),
            "latent_scale": self.config.get("latent_scale", 1.0),
            "zlr_whiten_path": self.config.get("zlr_whiten_path"),
        }
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        size_mb = latest_path.stat().st_size / (1024**2)
        print(f"Saved {latest_path} ({size_mb:.1f} MB, epoch {epoch})")

        if save_snapshot:
            snap = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, snap)
            print(f"Saved snapshot {snap.name}")

        # Optional second local copy (e.g. another mounted path). Still wiped on
        # Kaggle restart unless it points outside /kaggle/working.
        backup_dir = self.config.get("checkpoint_backup_dir")
        if backup_dir:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest_path, backup_dir / "latest.pt")
            if save_snapshot:
                shutil.copy2(
                    latest_path, backup_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                )
            print(f"Backed up checkpoint to {backup_dir}")

    def _hf_remote_path(self, name: str) -> str:
        subdir = str(self.config.get("hf_checkpoint_subdir", "")).strip("/")
        return f"{subdir}/{name}" if subdir else name

    def _persist_remote(self, *, epoch: int, sample_path: Path | None = None) -> None:
        hf_repo = self.config.get("hf_checkpoint_repo")
        if not hf_repo:
            return
        latest_path = self.checkpoint_dir / "latest.pt"
        self._upload_hf_checkpoint(
            latest_path,
            epoch=epoch,
            repo_id=str(hf_repo),
            sample_path=sample_path,
        )

    def _upload_hf_checkpoint(
        self,
        latest_path: Path,
        *,
        epoch: int,
        repo_id: str,
        sample_path: Path | None = None,
    ) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print(
                "hf_checkpoint_repo is set but huggingface_hub is not installed; "
                "skipping upload (pip install huggingface_hub)."
            )
            return
        try:
            api = HfApi()
            uploads: list[tuple[Path, str]] = [
                (latest_path, self._hf_remote_path("latest.pt")),
                (latest_path, self._hf_remote_path(f"checkpoint_epoch_{epoch:03d}.pt")),
            ]
            if self.train_metrics_path.exists():
                uploads.append(
                    (self.train_metrics_path, self._hf_remote_path("logs/train_metrics.csv"))
                )
            if self.val_metrics_path.exists():
                uploads.append(
                    (self.val_metrics_path, self._hf_remote_path("logs/val_metrics.csv"))
                )
            if sample_path is not None and Path(sample_path).exists():
                sample_path = Path(sample_path)
                uploads.append(
                    (sample_path, self._hf_remote_path(f"samples/{sample_path.name}"))
                )
            for local_path, remote_name in uploads:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_name,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"  HF uploaded {remote_name}")
            prefix = str(self.config.get("hf_checkpoint_subdir", "")).strip("/")
            dest = f"hf://{repo_id}/{prefix}/" if prefix else f"hf://{repo_id}/"
            print(f"HF backup complete for epoch {epoch} → {dest} ({len(uploads)} files)")
        except Exception as exc:  # noqa: BLE001 — never crash training on backup failure
            print(f"WARNING: Hugging Face checkpoint upload failed: {exc}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = int(checkpoint["epoch"])
        return checkpoint

    @torch.no_grad()
    def save_qualitative_grid(self, num_images: int = 4) -> Path | None:
        """Compare bicubic | decode(z_lr) | LatentSR | HR on a val batch."""
        if self.val_loader is None:
            return None
        self.model.eval()
        lr, hr = next(iter(self.val_loader))
        lr = lr[:num_images].to(self.device)
        hr = hr[:num_images].to(self.device)
        z_lr_cond, _z_hr = self.sr_encoder(lr, hr)
        z_lr_raw = self.sr_encoder.encode_lr_raw(lr)
        latent_scale = float(self.config.get("latent_scale", 1.0))
        hr_size = int(self.config.get("hr_size", 128))

        bicubic = upsample_bicubic(lr, hr_size)
        # Soft-decode always uses raw (unwhitened) z_lr.
        soft = decode_scaled(self.sr_encoder.vae, z_lr_raw, latent_scale)
        # Short progress bar off for periodic grids during training.
        pred = sample_sr_images(
            self.model,
            self.sr_encoder.vae,
            z_lr_cond,
            latent_scale=latent_scale,
            show_progress=False,
        )

        grid = make_grid(
            torch.cat([bicubic.cpu(), soft.cpu(), pred.cpu(), hr.cpu()], dim=0),
            nrow=num_images,
            padding=2,
        )
        out = self.sample_dir / f"sr_compare_epoch_{self.current_epoch:03d}.png"
        save_image(grid, out)
        return out

    def _print_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        print(f"Epoch [{epoch}/{self.config['epochs']}]")
        print(f"Train Loss:  {train_metrics['loss']:.6f}")
        print(f"Time:        {train_metrics['epoch_time']:.1f} sec")
        if val_metrics is not None:
            print(f"Val Loss:    {val_metrics['loss']:.6f}")

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        self._append_csv_row(
            self.train_metrics_path,
            self.TRAIN_FIELDS,
            {
                "epoch": epoch,
                "loss": f"{train_metrics['loss']:.8f}",
                "lr": f"{train_metrics['lr']:.8f}",
                "epoch_time": f"{train_metrics['epoch_time']:.2f}",
            },
        )
        if val_metrics is not None:
            self._append_csv_row(
                self.val_metrics_path,
                self.VAL_FIELDS,
                {
                    "epoch": epoch,
                    "loss": f"{val_metrics['loss']:.8f}",
                    "epoch_time": f"{val_metrics['epoch_time']:.2f}",
                },
            )

    def train(self) -> None:
        if seed := self.config.get("seed"):
            torch.manual_seed(int(seed))

        start_epoch = int(self.config.get("start_epoch", 0))
        epochs = int(self.config["epochs"])
        checkpoint_every = int(self.config.get("checkpoint_every", 5))
        validate_every = int(self.config.get("validate_every", 5))
        sample_every = int(self.config.get("sample_every", 10))

        n_train = len(self.train_loader.dataset)  # type: ignore[arg-type]
        print(f"Device: {self.device}  |  AMP: {self.use_amp}")
        print(
            f"Train pairs: {n_train}  |  batches/epoch: {len(self.train_loader)}  |  "
            f"batch_size: {self.train_loader.batch_size}"
        )
        print(
            f"VAE: {self.config.get('vae_checkpoint')}  |  "
            f"latent_scale: {self.config.get('latent_scale', 1.0)}  |  "
            f"condition: {self.config.get('condition_type', 'concat')}"
        )
        hf_repo = self.config.get("hf_checkpoint_repo")
        hf_subdir = str(self.config.get("hf_checkpoint_subdir", "")).strip("/")
        if hf_repo:
            dest = f"hf://{hf_repo}/{hf_subdir}/" if hf_subdir else f"hf://{hf_repo}/"
            print(f"HF backup every epoch → {dest}")
        print()

        for epoch in range(start_epoch + 1, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            run_val = self.val_loader is not None and (
                epoch == 1 or epoch == epochs or epoch % validate_every == 0
            )
            val_metrics = self.validate() if run_val else None
            self._print_metrics(epoch, train_metrics, val_metrics)
            self._log_metrics(epoch, train_metrics, val_metrics)

            save_snapshot = epoch == epochs or epoch % checkpoint_every == 0
            self.save_checkpoint(epoch, train_metrics, save_snapshot=save_snapshot)

            sample_path = None
            if sample_every > 0 and (
                epoch == 1 or epoch == epochs or epoch % sample_every == 0
            ):
                # Full 1000-step sample is slow; only a few images.
                print("Generating qualitative SR grid (may take a few minutes)…")
                sample_path = self.save_qualitative_grid(
                    num_images=int(self.config.get("sample_images", 4))
                )
                if sample_path is not None:
                    print(f"Saved comparison grid to {sample_path}")
            self._persist_remote(epoch=epoch, sample_path=sample_path)
            print()
