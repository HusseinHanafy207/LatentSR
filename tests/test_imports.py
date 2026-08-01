"""Scaffold / dependency smoke tests."""

from __future__ import annotations

import importlib

import pytest


def test_package_imports() -> None:
    import latentsr

    assert latentsr.__version__ == "0.1.0"
    assert importlib.import_module("latentsr.datasets")
    assert importlib.import_module("latentsr.vae")
    assert importlib.import_module("latentsr.diffusion")
    assert importlib.import_module("latentsr.super_resolution")
    assert importlib.import_module("latentsr.metrics")
    assert importlib.import_module("latentsr.utils")


def test_generative_models_ddpm_available() -> None:
    """This project must import DDPM — not vendor a copy."""
    pytest.importorskip("generative_models")
    ddpm = importlib.import_module("generative_models.ddpm")
    assert hasattr(ddpm, "UNet")
    assert hasattr(ddpm, "NoiseScheduler")
    assert hasattr(ddpm, "forward_diffuse")
    assert hasattr(ddpm, "DDPM")
