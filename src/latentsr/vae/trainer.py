"""VAE training loop (CelebA RGB reconstructions)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from latentsr.utils.config import get_device


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
        with context:
            for batch in loader:
                images = _batch_images(batch).to(self.device)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                reconstruction, mu, logvar = self.model(images)
                loss, recon_loss, kl_loss = self.criterion(
                    reconstruction, images, mu, logvar
                )

                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_loss = float(loss.item())
                if first_batch_loss is None:
                    first_batch_loss = batch_loss
                last_batch_loss = batch_loss

                total_loss += batch_loss
                total_recon_loss += float(recon_loss.item())
                total_kl_loss += float(kl_loss.item())
                num_batches += 1

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

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
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
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, epoch_path)
        torch.save(checkpoint, latest_path)
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

        for epoch in range(start_epoch + 1, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self._print_metrics(epoch, train_metrics, val_metrics)
            self._log_metrics(epoch, train_metrics, val_metrics)

            if epoch == epochs or epoch % checkpoint_every == 0:
                self.save_checkpoint(epoch, train_metrics)

            if epoch == 1 or epoch % reconstruct_every == 0 or epoch == epochs:
                sample_path = self.reconstruct_images()
                print(f"Saved reconstruction grid to {sample_path}")
            print()
