"""Shared pytest fixtures. All tests here are CPU-only and deterministic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
if str(SRC) not in sys.path:  # allow `pytest` without `pip install -e .`
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:  # scripts/ is not a package; import exit_gate directly
    # append, not insert(0): scripts/ currently holds only download_enwik8.py
    # and exit_gate.py/run_exit_gate.py, so there's no live collision, but
    # putting it ahead of src/ and the stdlib (as insert(0) would) means any
    # FUTURE top-level script name matching a stdlib or src module would
    # silently shadow it for the whole test session. Lowest priority instead.
    sys.path.append(str(SCRIPTS))
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


@pytest.fixture(autouse=True)
def _configure_logging(capsys: pytest.CaptureFixture[str]) -> None:
    """Rebind loguru's sinks to the CURRENT per-test stdio streams.

    loguru binds a stream sink to the literal object passed to
    ``logger.add()`` at call time. pytest's ``capsys`` swaps
    ``sys.stdout``/``sys.stderr`` with a fresh object per test; a sink
    configured once (at import time, or by an earlier test that went
    through a CLI entry point) would keep writing to a stream a LATER
    test's ``capsys.readouterr()`` never reads from — every DECISION-NEEDED
    / gate-text assertion in this suite would then silently start asserting
    on empty output while still reporting green (see
    ``entropic_substrate.logging``'s module docstring).

    Declaring ``capsys`` as a parameter (not just importing it) makes this
    fixture DEPEND on it, so pytest instantiates capsys's stream swap
    before this fixture's body runs — reconfiguring the sinks any earlier
    would just rebind to the stream capsys is about to replace.

    ``json_path=None``: the structured JSON sink is exercised directly by
    ``tests/test_logging.py``; every other test in this ~300+-test suite
    would otherwise append to ``logs/entropic-substrate.jsonl`` on every
    run for no reason.
    """
    from entropic_substrate.logging import configure_logging

    configure_logging(json_path=None)


@pytest.fixture
def sample_bytes() -> bytes:
    """A small, structured byte corpus for transition counting."""
    from entropic_substrate.data.synthetic import synthetic_text

    return synthetic_text(60_000, seed=7).encode("utf-8")
