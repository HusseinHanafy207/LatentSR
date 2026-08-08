"""Shared DataLoader options (workers, pin_memory)."""

from __future__ import annotations

from typing import Any


def build_dataloader_kwargs(
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool | None = None,
) -> dict[str, Any]:
    """Keyword args for ``torch.utils.data.DataLoader``.

    ``persistent_workers`` / ``prefetch_factor`` are only set when
    ``num_workers > 0`` (PyTorch rejects them otherwise).
    """
    workers = max(0, int(num_workers))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(pin_memory),
    }
    if workers > 0:
        persist = True if persistent_workers is None else bool(persistent_workers)
        kwargs["persistent_workers"] = persist
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return kwargs
