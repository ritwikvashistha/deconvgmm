"""Fixtures for the preserved prototype and future production tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE_PATH = PROJECT_ROOT / "prototype" / "xdgmm_jax_prototype.py"
PROTOTYPE_SHA256 = (
    "a6e0b5901eb1ee70a917a28d21df1680657a02e081141d16cd63835fece37b6b"
)


@pytest.fixture(scope="session")
def prototype_module():
    """Load the immutable prototype without treating it as the package API."""

    spec = importlib.util.spec_from_file_location(
        "_xdgmm_jax_preserved_prototype", PROTOTYPE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load preserved prototype at {PROTOTYPE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

