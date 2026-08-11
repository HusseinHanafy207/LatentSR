"""Datasets for LatentSR."""

from latentsr.datasets.celeba import get_celeba_dataloaders, get_celeba_dataset
from latentsr.datasets.onthefly_latent import OnTheFlyLatentEncoder, batch_to_images

__all__ = [
    "OnTheFlyLatentEncoder",
    "batch_to_images",
    "get_celeba_dataset",
    "get_celeba_dataloaders",
]
