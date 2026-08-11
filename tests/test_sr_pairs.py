"""Phase 6: LR/HR super-resolution pixel pairs."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from latentsr.datasets.sr_pairs import (
    SuperResolutionPairDataset,
    downsample_bicubic,
)


def test_downsample_shapes() -> None:
    hr = torch.rand(3, 128, 128)
    lr = downsample_bicubic(hr, 32)
    assert lr.shape == (3, 32, 32)
    assert lr.min() >= 0.0 and lr.max() <= 1.0

    hr_b = torch.rand(4, 3, 128, 128)
    lr_b = downsample_bicubic(hr_b, 32)
    assert lr_b.shape == (4, 3, 32, 32)


def test_sr_pair_dataset_shapes() -> None:
    images = torch.rand(5, 3, 128, 128)
    # Fake base dataset yielding (image, label)
    labels = torch.zeros(5, 1)
    base = TensorDataset(images, labels)
    ds = SuperResolutionPairDataset(base, hr_size=128, lr_size=32)
    assert len(ds) == 5
    lr, hr = ds[0]
    assert lr.shape == (3, 32, 32)
    assert hr.shape == (3, 128, 128)
    assert torch.allclose(hr, images[0])


def test_sr_pair_alignment_identity_content() -> None:
    """Downsampling a solid color should preserve the color in LR."""
    hr = torch.ones(3, 128, 128) * torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1)
    lr = downsample_bicubic(hr, 32)
    assert torch.allclose(lr.mean(dim=(1, 2)), torch.tensor([0.2, 0.5, 0.8]), atol=1e-3)


def test_invalid_sizes_raise() -> None:
    images = torch.rand(1, 3, 128, 128)
    base = TensorDataset(images)
    try:
        SuperResolutionPairDataset(base, hr_size=128, lr_size=30)
        assert False, "expected ValueError"
    except ValueError:
        pass
