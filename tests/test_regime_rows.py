"""Control tests for tools/regime_rows.py (conductor tooling, p0 ruling 2026-09-19).

The H composed digests are keyed on a measured regime (python, torch) for the
checkpoint reader and (python, pysbd) for the source reader. A phase that must
re-pin a regime row it cannot run locally (CI's Linux regime) computes it with
this tool from measured input strings. The tool is trustworthy only if, fed the
LOCAL regime strings, it reproduces the repo's own identity functions exactly;
that control is what these tests pin.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from platform import python_version

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import regime_rows

from entropic_substrate.ladder.capture import capture
from entropic_substrate.ladder.run import checkpoint

pytestmark = pytest.mark.unit


def test_checkpoint_row_reproduces_the_repo_identity_under_the_local_regime() -> None:
    assert (
        regime_rows.checkpoint_row(python_version(), str(torch.__version__))
        == checkpoint._reader_identity()
    )


def test_source_row_reproduces_the_repo_identity_under_the_local_regime() -> None:
    assert (
        regime_rows.source_row(python_version(), metadata.version("pysbd"))
        == capture._reader_identity()
    )


def test_a_different_regime_string_moves_the_row() -> None:
    """Both arms: the regime strings are hashed inputs, not decoration."""
    local = regime_rows.checkpoint_row(python_version(), str(torch.__version__))
    other = regime_rows.checkpoint_row(python_version(), str(torch.__version__) + "+probe")
    assert other != local
    assert len(other) == 64 and len(local) == 64


def test_cli_prints_both_rows_for_the_given_regime(capsys: pytest.CaptureFixture[str]) -> None:
    regime_rows.main(
        [
            "--python",
            python_version(),
            "--torch",
            str(torch.__version__),
            "--pysbd",
            metadata.version("pysbd"),
        ]
    )
    out = capsys.readouterr().out
    assert checkpoint._reader_identity() in out
    assert capture._reader_identity() in out
