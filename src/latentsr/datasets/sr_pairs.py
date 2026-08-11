"""CelebA LR / HR pixel pairs for super-resolution (Phase 6).

HR is the CelebA image at ``hr_size`` (default 128).
LR is a bicubic downsample to ``lr_size`` (default 32) → 4× SR.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from latentsr.datasets.celeba import get_celeba_dataset
from latentsr.datasets.loader_utils import build_dataloader_kwargs


def downsample_bicubic(hr: torch.Tensor, lr_size: int) -> torch.Tensor:
    """Bicubic downsample ``(C, H, W)`` or ``(B, C, H, W)`` to ``lr_size``."""
    if hr.ndim == 3:
        hr_b = hr.unsqueeze(0)
        lr = F.interpolate(
            hr_b,
            size=(lr_size, lr_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    elif hr.ndim == 4:
        lr = F.interpolate(
            hr,
            size=(lr_size, lr_size),
            mode="bicubic",
            align_corners=False,
        )
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(hr.shape)}")
    return lr.clamp(0.0, 1.0)


class SuperResolutionPairDataset(Dataset):
    """Wrap an image dataset and yield ``(lr, hr)`` float tensors in ``[0, 1]``.

    The base dataset must return ``(image, …)`` or a bare image tensor of shape
    ``(3, hr_size, hr_size)``.
    """

    def __init__(
        self,
        base: Dataset,
        *,
        hr_size: int = 128,
        lr_size: int = 32,
    ) -> None:
        if hr_size % lr_size != 0:
            raise ValueError(
                f"hr_size ({hr_size}) must be divisible by lr_size ({lr_size})"
            )
        self.base = base
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.scale = hr_size // lr_size

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.base[index]
        hr = item[0] if isinstance(item, (list, tuple)) else item
        if not isinstance(hr, torch.Tensor):
            raise TypeError(f"Expected image tensor, got {type(hr)}")
        if hr.shape[-2:] != (self.hr_size, self.hr_size):
            raise ValueError(
                f"Expected HR spatial size {(self.hr_size, self.hr_size)}, "
                f"got {tuple(hr.shape[-2:])}"
            )
        lr = downsample_bicubic(hr, self.lr_size)
        return lr, hr


def get_sr_pair_datasets(
    data_dir: str | Path,
    *,
    hr_size: int = 128,
    lr_size: int = 32,
    download: bool = True,
) -> tuple[SuperResolutionPairDataset, SuperResolutionPairDataset]:
    """Train / val SR pair datasets over CelebA."""
    train_base = get_celeba_dataset(
        data_dir, train=True, image_size=hr_size, download=download
    )
    val_base = get_celeba_dataset(
        data_dir, train=False, image_size=hr_size, download=download
    )
    return (
        SuperResolutionPairDataset(train_base, hr_size=hr_size, lr_size=lr_size),
        SuperResolutionPairDataset(val_base, hr_size=hr_size, lr_size=lr_size),
    )


def get_sr_pair_dataloaders(
    batch_size: int,
    data_dir: str | Path,
    *,
    hr_size: int = 128,
    lr_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders yielding ``(lr, hr)`` batches."""
    train_ds, val_ds = get_sr_pair_datasets(
        data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        download=download,
    )
    loader_kwargs = build_dataloader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader
