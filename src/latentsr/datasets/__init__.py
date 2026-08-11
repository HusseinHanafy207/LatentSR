"""Datasets for LatentSR."""

from latentsr.datasets.celeba import get_celeba_dataloaders, get_celeba_dataset
from latentsr.datasets.onthefly_latent import OnTheFlyLatentEncoder, batch_to_images
from latentsr.datasets.sr_pairs import (
    SuperResolutionPairDataset,
    downsample_bicubic,
    get_sr_pair_dataloaders,
    get_sr_pair_datasets,
)

__all__ = [
    "OnTheFlyLatentEncoder",
    "SuperResolutionPairDataset",
    "batch_to_images",
    "downsample_bicubic",
    "get_celeba_dataset",
    "get_celeba_dataloaders",
    "get_sr_pair_datasets",
    "get_sr_pair_dataloaders",
]
