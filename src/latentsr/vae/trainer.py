"""VAE training loop (CelebA RGB reconstructions)."""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from latentsr.utils.amp import autocast_context, make_grad_scaler
from latentsr.utils.config import get_device


def _upsample_bicubic(lr: torch.Tensor, hr_size: int) -> torch.Tensor:
    """Bicubic upsample a batched LR tensor to ``hr_size`` (no datasets import)."""
    return F.interpolate(
        lr, size=(hr_size, hr_size), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)


def _batch_images(batch: Any) -> torch.Tensor:
    """Pull the image tensor from a ``(image, …)`` batch or bare tensor."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


class VAETrainer:
    """Orchestrates VAE training without owning the model, loss, or data."""

    TRAIN_FIELDS = ["epoch", "total_loss", "recon_loss", "kl_loss", "lr", "epoch_time"]
    VAL_FIELDS = ["epoch", "total_loss", "recon_loss", "kl_loss", "epoch_time"]

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        config: dict[str, Any],
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = get_device(str(config.get("device", "auto")))
        self.model.to(self.device)

        self.use_amp = bool(config.get("amp", False)) and self.device.type == "cuda"
        self.scaler = make_grad_scaler(enabled=self.use_amp)

        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.sample_dir = Path(config["sample_dir"])
        self.log_dir = Path(config["log_dir"])

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

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
            with self.train_metrics_path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.TRAIN_FIELDS).writeheader()

        if self.val_loader is not None and not self.val_metrics_path.exists():
            with self.val_metrics_path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.VAL_FIELDS).writeheader()

    def _append_csv_row(
        self, path: Path, fieldnames: list[str], row: dict[str, Any]
    ) -> None:
        with path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fieldnames).writerow(row)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(mode=training)

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        first_batch_loss: float | None = None
        last_batch_loss: float | None = None

        context = torch.enable_grad() if training else torch.no_grad()
        phase = "train" if training else "val"
        progress = tqdm(loader, desc=phase, leave=False, dynamic_ncols=True)
        with context:
            for batch in progress:
                images = _batch_images(batch).to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast_context(enabled=self.use_amp, device_type=self.device.type):
                    reconstruction, mu, logvar = self.model(images)
                    loss, recon_loss, kl_loss = self.criterion(
                        reconstruction, images, mu, logvar
                    )

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                batch_loss = float(loss.item())
                if first_batch_loss is None:
                    first_batch_loss = batch_loss
                last_batch_loss = batch_loss

                total_loss += batch_loss
                total_recon_loss += float(recon_loss.item())
                total_kl_loss += float(kl_loss.item())
                num_batches += 1
                if num_batches % 10 == 0 or num_batches == 1:
                    progress.set_postfix(
                        loss=f"{batch_loss:.4f}",
                        recon=f"{float(recon_loss.item()):.4f}",
                        kl=f"{float(kl_loss.item()):.4f}",
                    )

        metrics = {
            "total_loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
        }
        if training:
            metrics["first_batch_loss"] = first_batch_loss or 0.0
            metrics["last_batch_loss"] = last_batch_loss or 0.0
        return metrics

    def train_epoch(self) -> dict[str, float]:
        start_time = time.time()
        metrics = self._run_epoch(self.train_loader, training=True)
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["epoch_time"] = time.time() - start_time
        return metrics

    def validate(self) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        start_time = time.time()
        metrics = self._run_epoch(self.val_loader, training=False)
        metrics["epoch_time"] = time.time() - start_time
        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        *,
        save_snapshot: bool = True,
    ) -> None:
        """Always refresh ``latest.pt``; optionally write ``checkpoint_epoch_XXX.pt``."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "arch": self.model.config_dict()
            if hasattr(self.model, "config_dict")
            else {},
        }
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        if save_snapshot:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, epoch_path)
        if alias := self.config.get("checkpoint_alias"):
            torch.save(checkpoint, self.checkpoint_dir / alias)

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = int(checkpoint["epoch"])
        return checkpoint

    def reconstruct_images(self, num_images: int = 8) -> Path:
        self.model.eval()
        batch = next(iter(self.val_loader or self.train_loader))
        images = _batch_images(batch)[:num_images].to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "reconstruct"):
                reconstruction = self.model.reconstruct(images, use_mean=True)
            else:
                reconstruction, _, _ = self.model(images)

        comparison = torch.cat([images.cpu(), reconstruction.cpu()], dim=0)
        grid = make_grid(comparison, nrow=num_images, padding=2)

        output_path = self.sample_dir / f"reconstruction_epoch_{self.current_epoch:03d}.png"
        save_image(grid, output_path)
        return output_path

    def _print_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        epochs = self.config["epochs"]
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"Train Loss:  {train_metrics['total_loss']:.6f}")
        print(f"Recon Loss:  {train_metrics['recon_loss']:.6f}")
        print(f"KL Loss:     {train_metrics['kl_loss']:.6f}")
        print(f"Time:        {train_metrics['epoch_time']:.1f} sec")
        if val_metrics is not None:
            print(f"Val Loss:    {val_metrics['total_loss']:.6f}")
            print(f"Val Recon:   {val_metrics['recon_loss']:.6f}")
            print(f"Val KL:      {val_metrics['kl_loss']:.6f}")

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
                "total_loss": f"{train_metrics['total_loss']:.8f}",
                "recon_loss": f"{train_metrics['recon_loss']:.8f}",
                "kl_loss": f"{train_metrics['kl_loss']:.8f}",
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
                    "total_loss": f"{val_metrics['total_loss']:.8f}",
                    "recon_loss": f"{val_metrics['recon_loss']:.8f}",
                    "kl_loss": f"{val_metrics['kl_loss']:.8f}",
                    "epoch_time": f"{val_metrics['epoch_time']:.2f}",
                },
            )

    def train(self) -> None:
        if seed := self.config.get("seed"):
            torch.manual_seed(seed)

        start_epoch = int(self.config.get("start_epoch", 0))
        epochs = int(self.config["epochs"])
        reconstruct_every = int(self.config.get("reconstruct_every", 5))
        checkpoint_every = int(self.config.get("checkpoint_every", 1))

        n_train = len(self.train_loader.dataset)  # type: ignore[arg-type]
        n_batches = len(self.train_loader)
        validate_every = int(self.config.get("validate_every", 1))
        print(f"Device: {self.device}  |  AMP: {self.use_amp}")
        print(
            f"Train images: {n_train}  |  batches/epoch: {n_batches}  |  "
            f"batch_size: {self.train_loader.batch_size}"
        )
        if self.val_loader is not None:
            print(f"Val images: {len(self.val_loader.dataset)}")  # type: ignore[arg-type]
        hf_repo = self.config.get("hf_checkpoint_repo")
        hf_subdir = str(self.config.get("hf_checkpoint_subdir", "")).strip("/")
        if hf_repo:
            dest = f"hf://{hf_repo}/{hf_subdir}/" if hf_subdir else f"hf://{hf_repo}/"
            print(f"HF backup every epoch → {dest}")
        elif hf_subdir:
            print(
                "WARNING: hf_checkpoint_subdir is set but hf_checkpoint_repo is not; "
                "checkpoints stay on local disk only (Kaggle /kaggle/working is wiped "
                "on restart)."
            )
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

            # Always update latest.pt so Colab interrupts don't lose finished epochs.
            # Numbered snapshots only every checkpoint_every (and final epoch).
            save_snapshot = epoch == epochs or epoch % checkpoint_every == 0
            self.save_checkpoint(epoch, train_metrics, save_snapshot=save_snapshot)
            print(f"Saved latest.pt (epoch {epoch})")

            sample_path = None
            if epoch == 1 or epoch % reconstruct_every == 0 or epoch == epochs:
                sample_path = self.reconstruct_images()
                print(f"Saved reconstruction grid to {sample_path}")
            self._persist_remote(
                epoch=epoch, save_snapshot=save_snapshot, sample_path=sample_path
            )
            print()

    def _persist_remote(
        self,
        *,
        epoch: int,
        save_snapshot: bool,
        sample_path: Path | None,
    ) -> None:
        """Hook for durable off-machine backup (Kaggle). Default: no-op."""
        return


class SRAwareVAETrainer(VAETrainer):
    """VAE trainer with an extra ``μ_lr → sg(μ_hr)`` alignment term.

    Batches must be ``(lr, hr)`` from :func:`get_sr_pair_dataloaders`.
    Reconstruction and KL are computed on HR only.
    """

    TRAIN_FIELDS = [
        "epoch",
        "total_loss",
        "recon_loss",
        "kl_loss",
        "align_loss",
        "z_cosine",
        "lr",
        "epoch_time",
    ]
    VAL_FIELDS = [
        "epoch",
        "total_loss",
        "recon_loss",
        "kl_loss",
        "align_loss",
        "z_cosine",
        "epoch_time",
    ]

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(mode=training)
        hr_size = int(self.config.get("hr_size", self.config.get("image_size", 128)))

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_align_loss = 0.0
        total_cosine = 0.0
        num_batches = 0
        first_batch_loss: float | None = None
        last_batch_loss: float | None = None

        context = torch.enable_grad() if training else torch.no_grad()
        phase = "train" if training else "val"
        progress = tqdm(loader, desc=phase, leave=False, dynamic_ncols=True)
        with context:
            for batch in progress:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise TypeError("SR-aware VAE expects batches of (lr, hr).")
                lr = batch[0].to(self.device, non_blocking=True)
                hr = batch[1].to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast_context(enabled=self.use_amp, device_type=self.device.type):
                    lr_up = _upsample_bicubic(lr, hr_size)
                    reconstruction, mu_hr, logvar_hr = self.model(hr)
                    mu_lr, _logvar_lr = self.model.encode(lr_up)
                    loss, recon_loss, kl_loss, align_loss = self.criterion(
                        reconstruction, hr, mu_hr, logvar_hr, mu_lr
                    )
                    cosine = F.cosine_similarity(
                        mu_lr.flatten(1), mu_hr.detach().flatten(1)
                    ).mean()

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                batch_loss = float(loss.item())
                if first_batch_loss is None:
                    first_batch_loss = batch_loss
                last_batch_loss = batch_loss

                total_loss += batch_loss
                total_recon_loss += float(recon_loss.item())
                total_kl_loss += float(kl_loss.item())
                total_align_loss += float(align_loss.item())
                total_cosine += float(cosine.item())
                num_batches += 1
                if num_batches % 10 == 0 or num_batches == 1:
                    progress.set_postfix(
                        loss=f"{batch_loss:.4f}",
                        recon=f"{float(recon_loss.item()):.4f}",
                        align=f"{float(align_loss.item()):.4f}",
                        cos=f"{float(cosine.item()):.3f}",
                    )

        denom = max(num_batches, 1)
        metrics = {
            "total_loss": total_loss / denom,
            "recon_loss": total_recon_loss / denom,
            "kl_loss": total_kl_loss / denom,
            "align_loss": total_align_loss / denom,
            "z_cosine": total_cosine / denom,
        }
        if training:
            metrics["first_batch_loss"] = first_batch_loss or 0.0
            metrics["last_batch_loss"] = last_batch_loss or 0.0
        return metrics

    def reconstruct_images(self, num_images: int = 4) -> Path:
        """Grid: nearest(LR) | bicubic | decode(z_lr) | decode(z_hr) | HR."""
        self.model.eval()
        hr_size = int(self.config.get("hr_size", self.config.get("image_size", 128)))
        batch = next(iter(self.val_loader or self.train_loader))
        lr = batch[0][:num_images].to(self.device)
        hr = batch[1][:num_images].to(self.device)
        with torch.no_grad():
            lr_up = _upsample_bicubic(lr, hr_size)
            mu_hr, _ = self.model.encode(hr)
            mu_lr, _ = self.model.encode(lr_up)
            vae_hr = self.model.decode(mu_hr)
            soft = self.model.decode(mu_lr)
        lr_nn = F.interpolate(lr, size=(hr_size, hr_size), mode="nearest")
        grid = make_grid(
            torch.cat(
                [
                    lr_nn.cpu(),
                    lr_up.cpu(),
                    soft.cpu(),
                    vae_hr.cpu(),
                    hr.cpu(),
                ],
                dim=0,
            ),
            nrow=lr.shape[0],
            padding=2,
        )
        output_path = self.sample_dir / f"sr_align_epoch_{self.current_epoch:03d}.png"
        save_image(grid, output_path)
        return output_path

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        *,
        save_snapshot: bool = True,
    ) -> None:
        super().save_checkpoint(epoch, metrics, save_snapshot=save_snapshot)
        latest_path = self.checkpoint_dir / "latest.pt"
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

    def _persist_remote(
        self,
        *,
        epoch: int,
        save_snapshot: bool,
        sample_path: Path | None,
    ) -> None:
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
        subdir = str(self.config.get("hf_checkpoint_subdir", "vae_sr")).strip("/")
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
                (latest_path, f"{subdir}/latest.pt"),
                (latest_path, f"{subdir}/checkpoint_epoch_{epoch:03d}.pt"),
            ]
            if self.train_metrics_path.exists():
                uploads.append(
                    (self.train_metrics_path, f"{subdir}/logs/train_metrics.csv")
                )
            if self.val_metrics_path.exists():
                uploads.append(
                    (self.val_metrics_path, f"{subdir}/logs/val_metrics.csv")
                )
            if sample_path is not None and Path(sample_path).exists():
                sample_path = Path(sample_path)
                uploads.append(
                    (sample_path, f"{subdir}/samples/{sample_path.name}")
                )
            for local_path, remote_name in uploads:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_name,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"  HF uploaded {remote_name}")
            print(
                f"HF backup complete for epoch {epoch} → hf://{repo_id}/{subdir}/ "
                f"({len(uploads)} files)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Hugging Face checkpoint upload failed: {exc}")

    def _print_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        super()._print_metrics(epoch, train_metrics, val_metrics)
        print(f"Align Loss:  {train_metrics['align_loss']:.6f}")
        print(f"z_cosine:    {train_metrics['z_cosine']:.4f}")
        if val_metrics is not None:
            print(f"Val Align:   {val_metrics['align_loss']:.6f}")
            print(f"Val cosine:  {val_metrics['z_cosine']:.4f}")

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
                "total_loss": f"{train_metrics['total_loss']:.8f}",
                "recon_loss": f"{train_metrics['recon_loss']:.8f}",
                "kl_loss": f"{train_metrics['kl_loss']:.8f}",
                "align_loss": f"{train_metrics['align_loss']:.8f}",
                "z_cosine": f"{train_metrics['z_cosine']:.8f}",
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
                    "total_loss": f"{val_metrics['total_loss']:.8f}",
                    "recon_loss": f"{val_metrics['recon_loss']:.8f}",
                    "kl_loss": f"{val_metrics['kl_loss']:.8f}",
                    "align_loss": f"{val_metrics['align_loss']:.8f}",
                    "z_cosine": f"{val_metrics['z_cosine']:.8f}",
                    "epoch_time": f"{val_metrics['epoch_time']:.2f}",
                },
            )
