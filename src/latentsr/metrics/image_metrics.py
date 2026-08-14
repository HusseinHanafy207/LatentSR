"""PSNR / SSIM / LPIPS for LatentSR evaluation (Phase 10).

PSNR and SSIM reuse ``generative_models.evaluation``. LPIPS uses the optional
``lpips`` package (AlexNet by default).
"""

from __future__ import annotations

from typing import Any

import torch
from generative_models.evaluation.metrics import psnr as gm_psnr
from generative_models.evaluation.metrics import ssim as gm_ssim


def psnr(
    output: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    data_range: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    return gm_psnr(output, ground_truth, data_range=data_range, reduction=reduction)


def ssim(
    output: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    data_range: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    return gm_ssim(output, ground_truth, data_range=data_range, reduction=reduction)


class LPIPSMetric:
    """Thin wrapper around ``lpips.LPIPS`` (lower is better)."""

    def __init__(
        self,
        *,
        net: str = "alex",
        device: torch.device | str = "cpu",
    ) -> None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS requires the 'lpips' package. Install with: pip install lpips"
            ) from exc
        self._model = lpips.LPIPS(net=net).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(
        self,
        output: torch.Tensor,
        ground_truth: torch.Tensor,
        *,
        reduction: str = "none",
    ) -> torch.Tensor:
        """Compute LPIPS; inputs are RGB in ``[0, 1]`` (converted to ``[-1, 1]``)."""
        if output.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: {tuple(output.shape)} vs {tuple(ground_truth.shape)}"
            )
        if output.ndim == 3:
            output = output.unsqueeze(0)
            ground_truth = ground_truth.unsqueeze(0)
        out = output.to(self.device).float().clamp(0.0, 1.0) * 2.0 - 1.0
        gt = ground_truth.to(self.device).float().clamp(0.0, 1.0) * 2.0 - 1.0
        values = self._model(out, gt).view(-1)
        if reduction == "mean":
            return values.mean()
        if reduction == "none":
            return values
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")


def summarize_values(values: list[float] | torch.Tensor) -> dict[str, float]:
    """Mean / std for a 1D collection of per-image scores."""
    if isinstance(values, list):
        t = torch.tensor(values, dtype=torch.float64)
    else:
        t = values.detach().float().cpu().double().reshape(-1)
    if t.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
        "n": int(t.numel()),
    }


def batch_metrics(
    output: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    lpips_fn: LPIPSMetric | None = None,
    data_range: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Per-image metrics for one batch (``reduction='none'``)."""
    out: dict[str, torch.Tensor] = {
        "psnr": psnr(output, ground_truth, data_range=data_range, reduction="none"),
        "ssim": ssim(output, ground_truth, data_range=data_range, reduction="none"),
    }
    if lpips_fn is not None:
        out["lpips"] = lpips_fn(output, ground_truth, reduction="none").cpu()
    return out


def format_metric_table(results: dict[str, dict[str, Any]]) -> str:
    """Pretty-print ``{method: {metric: {mean,std,n}}}``."""
    methods = list(results.keys())
    metric_names: list[str] = []
    for method in methods:
        for key in results[method]:
            if key not in metric_names:
                metric_names.append(key)

    header = f"{'method':<12}" + "".join(f"{m:>18}" for m in metric_names)
    lines = [header, "-" * len(header)]
    for method in methods:
        cells = [f"{method:<12}"]
        for metric in metric_names:
            stats = results[method].get(metric)
            if not stats:
                cells.append(f"{'—':>18}")
                continue
            cells.append(f"{stats['mean']:.4f}±{stats['std']:.4f}".rjust(18))
        lines.append("".join(cells))
    return "\n".join(lines)
