"""Shared pytest fixtures. All tests here are CPU-only and deterministic."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from loguru import logger

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


@pytest.fixture(scope="session", autouse=True)
def _isolate_callers_git_context() -> Iterator[None]:
    """Keep hook-owned indexes out of fixture repositories, including module fixtures.

    Git path-only commits export a temporary index. Regressions that exercise
    hostile context inject it inside their own tests, after this isolation.
    """
    with pytest.MonkeyPatch.context() as patch:
        for name in tuple(os.environ):
            if name.startswith("GIT_"):
                patch.delenv(name)
        yield


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Pin the global RNG and thread count for every test.

    Thread count comes from ``tests/xdist_policy.py``: 1 inside an xdist
    worker (``PYTEST_XDIST_WORKER`` set), 2 in a serial session. The old
    hardcoded 2 meant N workers x 2 intra-op threads on an N-CPU runner
    before any subprocess a test spawns (FOR-1618 P4b; the wall-time
    measurement is in that PR's ledger, not here). Imported inside the
    fixture because ``tests/`` joins ``sys.path`` above, after this
    module's own imports.
    """
    from xdist_policy import torch_threads_for

    torch.manual_seed(1234)
    torch.set_num_threads(torch_threads_for(os.environ))


@pytest.fixture(autouse=True)
def _configure_logging(capsys: pytest.CaptureFixture[str]) -> Iterator[None]:
    """Install loguru sinks that follow pytest's capsys stream swaps.

    loguru binds a stream sink to the literal object passed to
    ``logger.add()`` at call time. pytest's ``capsys`` gives each capture
    PHASE (setup, call, teardown) a fresh ``sys.stdout``/``sys.stderr`` and
    closes the previous phase's stream when the phase ends
    (``_pytest/capture.py`` 9.1.1: ``item_capture`` per phase,
    ``CaptureFixture._start`` builds a new ``MultiCapture``,
    ``SysCaptureBase.done`` closes the tmpfile). Sinks bound here, during
    fixture SETUP, would hold the setup phase's stream: closed by the time
    the test body runs, so a body-time ``logger.*`` call produced loguru's
    "Logging error" dump instead of a line, and a ``capsys.readouterr()``
    substring check still passed against the dump's ``Record was: {...}``
    text (see ``entropic_substrate.kernel.runtime.logging``'s module
    docstring).

    RETRACTION (FOR-1649, house rule 4): this docstring used to say capsys
    swaps "a fresh object per test" and that a setup-time rebind therefore
    followed it. Measured false: the swap is per phase, and the object this
    fixture saw at setup was not the one the test body wrote to
    (``tests/test_conftest_sink_binding.py`` pins both facts).

    ``resolve_streams=True`` makes the sinks look ``sys.stdout``/``sys.stderr``
    up on every write, so they follow every phase's swap. Declaring
    ``capsys`` as a parameter still matters: it makes pytest start capsys
    before this body runs, so the setup-phase emissions are captured too.

    ``json_path=None``: the structured JSON sink is exercised directly by
    ``tests/test_logging.py``; every other test in this ~300+-test suite
    would otherwise append to ``logs/entropic-substrate.jsonl`` on every
    run for no reason.

    TEARDOWN removes the sinks before yielding control back to pytest. With
    write-time resolution a late emission (e.g. a slow-to-join background
    thread started by a live integration test) no longer hits a closed
    stream, but a sink that a test body bound through a production CLI entry
    point (``configure_logging()`` without ``resolve_streams``) still holds a
    call-phase stream that is closed at the end of that phase; removing the
    sinks here closes that window for the teardown phase: a late emission
    after this point hits zero handlers and is dropped silently.
    """
    from entropic_substrate.kernel.runtime.logging import configure_logging

    configure_logging(json_path=None, resolve_streams=True)
    yield
    logger.remove()


@pytest.fixture
def sample_bytes() -> bytes:
    """A small, structured byte corpus for transition counting."""
    from entropic_substrate.ladder.capture.custody.synthetic import synthetic_text

    return synthetic_text(60_000, seed=7).encode("utf-8")
