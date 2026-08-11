"""AMP helpers compatible with older and newer PyTorch."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def make_grad_scaler(*, enabled: bool) -> Any:
    """Return a GradScaler; no-op capable when ``enabled=False``."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(*, enabled: bool, device_type: str = "cuda"):
    """Return an autocast context (or nullcontext when disabled / CPU)."""
    if not enabled or device_type != "cuda":
        return nullcontext()
    if hasattr(torch.amp, "autocast"):
        # Prefer device_type form when available (PyTorch ≥ 2.1 style).
        try:
            return torch.amp.autocast(device_type, enabled=True)
        except TypeError:
            return torch.cuda.amp.autocast(enabled=True)
    return torch.cuda.amp.autocast(enabled=True)
