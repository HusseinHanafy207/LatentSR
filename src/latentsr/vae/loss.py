"""β-VAE loss: reconstruction (MSE) + weighted KL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """
    Mean MSE reconstruction plus ``kl_weight * KL(q(z|x) || N(0, I))``.
    """

    def __init__(self, kl_weight: float = 1e-4) -> None:
        super().__init__()
        if kl_weight < 0:
            raise ValueError(f"kl_weight must be >= 0, got {kl_weight}")
        self.kl_weight = kl_weight

    def forward(
        self,
        reconstruction: torch.Tensor,
        images: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstruction, images)
        # Mean over batch of per-sample summed KL, then / numel-equivalent:
        # average KL element-wise (stable scale vs spatial latent size).
        kl_per_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_element.mean()
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss
