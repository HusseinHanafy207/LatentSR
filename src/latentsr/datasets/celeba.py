"""CelebA loaders for LatentSR (RGB faces in ``[0, 1]``)."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from latentsr.datasets.loader_utils import build_dataloader_kwargs


def _celeba_transform(image_size: int) -> transforms.Compose:
    """Center-crop to the face box, then resize to ``image_size``."""
    return transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )


def get_celeba_dataset(
    data_dir: str | Path,
    *,
    train: bool = True,
    image_size: int = 128,
    download: bool = True,
) -> Dataset:
    """CelebA RGB images as float ``(3, H, W)`` in ``[0, 1]``.

    Uses torchvision ``split='train'`` / ``'valid'``. First download is large
    (~1.4GB) and may require a working Google Drive fetch from torchvision.
    """
    split = "train" if train else "valid"
    return datasets.CelebA(
        root=str(data_dir),
        split=split,
        target_type="attr",
        transform=_celeba_transform(image_size),
        download=download,
    )


def get_celeba_dataloaders(
    batch_size: int,
    data_dir: str | Path,
    *,
    image_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders yielding ``(image, attr)`` batches."""
    train_ds = get_celeba_dataset(
        data_dir, train=True, image_size=image_size, download=download
    )
    val_ds = get_celeba_dataset(
        data_dir, train=False, image_size=image_size, download=download
    )
    loader_kwargs = build_dataloader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
