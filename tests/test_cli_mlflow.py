"""CLI-level proof of the MLflow STARTUP ABORT / --tiny opt-in contract.

``http://127.0.0.1:1`` is used as the "unreachable tracking server": nothing
listens on port 1, so the connection is refused immediately (no DNS lookup,
no real network hop) — deterministic and fast enough to stay a unit test,
without needing a live server. Full live-stack behaviour (a REAL MLflow
server actually accepting runs) is covered separately by the
``integration``-marked tests behind ``just container-test``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.ladder.capture.reader.sinks import MLFLOW_TINY_ENV, MLFLOW_TRACKING_ENV

pytestmark = pytest.mark.unit

_UNREACHABLE = "http://127.0.0.1:1"


def _tiny_argv(tmp_path: Path, results: Path) -> list[str]:
    return [
        "train",
        "--exp",
        "e0-byte",
        "--tiny",
        "--device",
        "cpu",
        "--steps",
        "2",
        "--results",
        str(results),
        "--quiet",
    ]


def test_tiny_run_creates_no_mlflow_run_by_default_even_with_uri_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #10: --tiny is untracked by default, even if MLFLOW_TRACKING_URI
    points somewhere — the training must succeed WITHOUT ever touching it.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, _UNREACHABLE)
    monkeypatch.delenv(MLFLOW_TINY_ENV, raising=False)
    results = tmp_path / "out.jsonl"

    code = main(_tiny_argv(tmp_path, results))

    assert code == 0, "an unreachable MLFLOW_TRACKING_URI must not affect an untracked --tiny run"
    records = [json.loads(line) for line in results.read_text().splitlines() if line]
    assert any(r["event"] == "run_end" for r in records), "training must have completed"


def test_tiny_run_with_explicit_opt_in_aborts_on_unreachable_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #5 + #10's other half: opt IN to tracking for --tiny, and an
    unreachable server must abort BEFORE training — no partial results file.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, _UNREACHABLE)
    monkeypatch.setenv(MLFLOW_TINY_ENV, "1")
    results = tmp_path / "out.jsonl"

    code = main(_tiny_argv(tmp_path, results))

    assert code != 0, "an unreachable tracking server must abort the run, not train through it"
    assert not results.exists(), "the run must abort BEFORE any training/logging happens"


def test_non_tiny_run_aborts_on_unreachable_tracking_without_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-tiny runs are tracked whenever MLFLOW_TRACKING_URI is set — no opt-in needed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, _UNREACHABLE)
    monkeypatch.delenv(MLFLOW_TINY_ENV, raising=False)
    results = tmp_path / "out.jsonl"

    code = main(
        [
            "train",
            "--exp",
            "e0-byte",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--results",
            str(results),
            "--quiet",
        ]
    )

    assert code != 0
    assert not results.exists()


def test_untracked_run_is_unaffected_when_tracking_uri_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(MLFLOW_TRACKING_ENV, raising=False)
    monkeypatch.delenv(MLFLOW_TINY_ENV, raising=False)
    results = tmp_path / "out.jsonl"

    assert main(_tiny_argv(tmp_path, results)) == 0
    assert results.exists()
