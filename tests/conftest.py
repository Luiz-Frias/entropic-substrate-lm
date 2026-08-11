"""Shared pytest fixtures. All tests here are CPU-only and deterministic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # allow `pytest` without `pip install -e .`
    sys.path.insert(0, str(SRC))
if str(Path(__file__).parent) not in sys.path:  # sibling `helpers` under importlib mode
    sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config: pytest.Config) -> None:
    """Register markers so the suite also runs standalone (strict-markers safe)."""
    config.addinivalue_line("markers", "unit: fast CPU-only tests")
    config.addinivalue_line("markers", "integration: slow / network / GPU tests")


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Pin the global RNG and thread count for every test."""
    torch.manual_seed(1234)
    torch.set_num_threads(2)


@pytest.fixture
def sample_bytes() -> bytes:
    """A small, structured byte corpus for transition counting."""
    from entropic_substrate.data.synthetic import synthetic_text

    return synthetic_text(60_000, seed=7).encode("utf-8")
