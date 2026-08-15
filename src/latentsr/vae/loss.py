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


class SRAwareVAELoss(VAELoss):
    """HR reconstruction VAE plus stop-grad alignment of ``μ_lr`` to ``μ_hr``.

    ``L = L_recon(HR) + β L_KL(HR) + λ ||μ_lr - sg(μ_hr)||²``

    ``μ_lr`` is the encoder mean of bicubic-upsampled LR. Stop-grad on ``μ_hr``
    keeps the HR code from collapsing toward the blurry LR code.
    """

    def __init__(self, kl_weight: float = 1e-4, align_weight: float = 1e-3) -> None:
        super().__init__(kl_weight=kl_weight)
        if align_weight < 0:
            raise ValueError(f"align_weight must be >= 0, got {align_weight}")
        self.align_weight = align_weight

    def forward(
        self,
        reconstruction: torch.Tensor,
        images: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mu_lr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_kl_total, recon_loss, kl_loss = super().forward(
            reconstruction, images, mu, logvar
        )
        if mu_lr is None:
            raise ValueError("SRAwareVAELoss requires mu_lr from encode(upsample(LR)).")
        if mu_lr.shape != mu.shape:
            raise ValueError(
                f"mu_lr shape {tuple(mu_lr.shape)} must match mu {tuple(mu.shape)}"
            )
        align_loss = F.mse_loss(mu_lr, mu.detach())
        total_loss = recon_kl_total + self.align_weight * align_loss
        return total_loss, recon_loss, kl_loss, align_loss
