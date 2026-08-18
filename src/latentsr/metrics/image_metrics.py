"""PSNR / SSIM / LPIPS / structure metrics for LatentSR evaluation.

PSNR and SSIM reuse ``generative_models.evaluation``. LPIPS uses the optional
``lpips`` package (AlexNet by default). Edge MAE and radial FFT-band errors
are local.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from generative_models.evaluation.metrics import psnr as gm_psnr
from generative_models.evaluation.metrics import ssim as gm_ssim

# ITU-R BT.601 luma used for edge / frequency (single-channel) metrics.
_LUMA_WEIGHTS = (0.2989, 0.5870, 0.1140)
_FREQ_BAND_EDGES = (1.0 / 3.0, 2.0 / 3.0)


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


def _as_nchw(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 3:
        return images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W) or (C, H, W), got {tuple(images.shape)}")
    return images


def _reduce_per_image(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")


def rgb_to_luma(images: torch.Tensor) -> torch.Tensor:
    """Convert RGB in ``[0, 1]`` to BT.601 luma, shape ``(B, 1, H, W)``."""
    x = _as_nchw(images).float()
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 RGB channels, got {x.shape[1]}")
    wr, wg, wb = _LUMA_WEIGHTS
    return wr * x[:, 0:1] + wg * x[:, 1:2] + wb * x[:, 2:3]


def sobel_magnitude(images: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of luma, shape ``(B, 1, H, W)``."""
    gray = rgb_to_luma(images)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)


def edge_mae(
    output: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    reduction: str = "none",
) -> torch.Tensor:
    """Per-image MAE of Sobel magnitudes (lower is better)."""
    if output.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(output.shape)} vs {tuple(ground_truth.shape)}"
        )
    mag_out = sobel_magnitude(output)
    mag_gt = sobel_magnitude(ground_truth)
    per_image = (mag_out - mag_gt).abs().mean(dim=(1, 2, 3))
    return _reduce_per_image(per_image, reduction)


def _radial_radius_map(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Normalized radius: 0 at DC, 1 at axis Nyquist."""
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy = (torch.arange(height, device=device, dtype=torch.float32) - cy) / (height / 2.0)
    xx = (torch.arange(width, device=device, dtype=torch.float32) - cx) / (width / 2.0)
    return torch.sqrt(yy[:, None].pow(2) + xx[None, :].pow(2))


def frequency_band_error(
    output: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    reduction: str = "none",
) -> dict[str, torch.Tensor]:
    """MSE of luma FFT magnitudes in low / mid / high radial bands.

    Bands use normalized radius ``r`` (0 at DC, 1 at axis Nyquist):

    - ``freq_low``: ``r < 1/3``
    - ``freq_mid``: ``1/3 <= r < 2/3``
    - ``freq_high``: ``r >= 2/3``
    """
    if output.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(output.shape)} vs {tuple(ground_truth.shape)}"
        )
    gray_out = rgb_to_luma(output)[:, 0]
    gray_gt = rgb_to_luma(ground_truth)[:, 0]
    spec_out = torch.fft.fftshift(torch.fft.fft2(gray_out.float()), dim=(-2, -1)).abs()
    spec_gt = torch.fft.fftshift(torch.fft.fft2(gray_gt.float()), dim=(-2, -1)).abs()
    squared_error = (spec_out - spec_gt).pow(2)

    height, width = gray_out.shape[-2:]
    radius = _radial_radius_map(height, width, squared_error.device)
    low_cut, mid_cut = _FREQ_BAND_EDGES
    masks = {
        "freq_low": radius < low_cut,
        "freq_mid": (radius >= low_cut) & (radius < mid_cut),
        "freq_high": radius >= mid_cut,
    }

    bands: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        count = float(mask.sum().clamp_min(1).item())
        per_image = (squared_error * mask).sum(dim=(-2, -1)) / count
        bands[name] = _reduce_per_image(per_image, reduction)
    return bands


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
        "edge_mae": edge_mae(output, ground_truth, reduction="none"),
    }
    out.update(frequency_band_error(output, ground_truth, reduction="none"))
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

    header = f"{'method':<14}" + "".join(f"{m:>18}" for m in metric_names)
    lines = [header, "-" * len(header)]
    for method in methods:
        cells = [f"{method:<14}"]
        for metric in metric_names:
            stats = results[method].get(metric)
            if not stats:
                cells.append(f"{'—':>18}")
                continue
            cells.append(f"{stats['mean']:.4f}±{stats['std']:.4f}".rjust(18))
        lines.append("".join(cells))
    return "\n".join(lines)


def dataset_filename(dataset: Any, index: int) -> str:
    """Best-effort CelebA filename; empty string if the dataset has none."""
    base = dataset
    seen: set[int] = set()
    while hasattr(base, "base") and id(base) not in seen:
        seen.add(id(base))
        base = base.base
    names = getattr(base, "filename", None)
    if names is None:
        return ""
    try:
        name = names[index]
    except (IndexError, TypeError, KeyError):
        return ""
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    return str(name)


def write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one row per image (wide: ``{method}_{metric}`` plus latent columns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_files(
    output_dir: Path,
    summary: dict[str, dict[str, dict[str, float]]],
    num_images: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write ``metrics.json`` and ``metrics.csv`` under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"num_images": num_images, "methods": summary}
    if extra:
        payload.update(extra)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for method, metrics in summary.items():
        row: dict[str, Any] = {"method": method, "n": num_images}
        for metric, stats in metrics.items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
