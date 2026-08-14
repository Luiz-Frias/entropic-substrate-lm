"""Contracts for the live gate's container-diagnostic helpers (`tests/helpers.py`).

These pin two properties that only ever matter on a RED run, which is exactly
why they need unit tests: the code paths under test are the ones no green run
exercises, and the 2026-08-13 live-gate failure proved that an unexercised
diagnostic can be the thing that breaks.

1. ``capture_container_logs`` must NEVER raise. A diagnostic that can fail
   replaces the real exception with its own and destroys the evidence.
2. ``assert_no_foreign_mlflow_stack`` must actually refuse a second concurrent
   mlflow stack — the condition the live gate's timing model forbids in prose
   and, until now, never checked.

Hermetic: ``subprocess.run`` is monkeypatched, so no podman, no containers, and
no dependence on what happens to be running on the machine.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from helpers import assert_no_foreign_mlflow_stack, capture_container_logs

pytestmark = pytest.mark.unit

CONTAINER = "some-project_mlflow_1"


class _Completed:
    """Minimal stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# --------------------------------------------------------- capture_container_logs
def test_capture_container_logs_returns_marker_instead_of_raising_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression. The previous inline call raised ``TimeoutExpired`` from
    inside an ``except`` block, which suppressed the print of the log it had
    just been asked to collect and demoted the genuine setup error to
    ``__context__``. Timing out must degrade to a marker, never to an exception.
    """

    def fake_run(*_args: Any, **_kwargs: Any) -> _Completed:
        raise subprocess.TimeoutExpired(cmd=["podman", "logs", CONTAINER], timeout=30.0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    out = capture_container_logs(CONTAINER, timeout_s=30.0)

    assert "TRUNCATED" in out, out
    assert CONTAINER in out, out


def test_capture_container_logs_keeps_the_partial_output_a_timeout_did_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout is TRUNCATION, not total loss. CPython attaches the bytes read
    so far to ``TimeoutExpired``; those bytes are the most valuable part of a
    hung-container diagnosis and must survive.

    ``output=``/``stderr=`` are passed as BYTES on purpose: ``subprocess.run``
    returns ``str`` under ``text=True`` on the success path but attaches bytes
    to ``TimeoutExpired`` on POSIX. Asserting the str path only would let a
    ``TypeError`` through on the one path that matters.
    """

    def fake_run(*_args: Any, **_kwargs: Any) -> _Completed:
        raise subprocess.TimeoutExpired(
            cmd=["podman", "logs", CONTAINER],
            timeout=30.0,
            output=b"Security middleware enabled. Allowed hosts: allowlisted.host.test\n",
            stderr=b"Waiting for application startup.\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    out = capture_container_logs(CONTAINER)

    assert "Security middleware enabled" in out, out
    assert "Waiting for application startup" in out, out


def test_capture_container_logs_returns_marker_instead_of_raising_when_podman_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine without podman on PATH must produce a diagnosis line, not an
    ``OSError`` that buries whatever the caller was already reporting."""

    def fake_run(*_args: Any, **_kwargs: Any) -> _Completed:
        raise FileNotFoundError(2, "No such file or directory: 'podman'")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out = capture_container_logs(CONTAINER)

    assert "could not run" in out, out


def test_capture_container_logs_merges_stderr_because_podman_logs_splits_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEASURED (2026-08-14): ``podman logs <mlflow container>`` put 214 bytes on
    stdout and 882 on stderr for the same log ``podman compose logs`` delivered
    as 1360 bytes of stdout. Reading only ``.stdout`` — what the previous call
    site did — therefore drops most of an mlflow boot log on the direct form.
    """
    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> _Completed:
        captured["argv"] = argv
        return _Completed(stdout="on-stdout\n", stderr="on-stderr\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out = capture_container_logs(CONTAINER)

    assert "on-stdout" in out, out
    assert "on-stderr" in out, out
    assert captured["argv"] == ["podman", "logs", CONTAINER]


def test_capture_container_logs_says_so_when_there_is_no_output_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty string reads as "I looked and the log was fine"; it must instead
    read as "there was nothing to look at", which is itself a finding."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(returncode=125))

    out = capture_container_logs(CONTAINER)

    assert "no output" in out, out


# ------------------------------------------------ assert_no_foreign_mlflow_stack
def test_guard_refuses_when_another_projects_mlflow_stack_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-08-13 condition, reproduced as a unit: a phase-1 mlflow container
    still up while the phase-2 stack is about to start."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Completed(stdout="entropic-substrate-lm-mlflow-test-sprint_mlflow_1\n"),
    )

    with pytest.raises(RuntimeError, match="another mlflow stack is already running"):
        assert_no_foreign_mlflow_stack("entropic-substrate-lm-mlflow-test-sprint-allowlist")


def test_guard_names_the_offending_container_and_the_recovery_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abort that does not say WHICH stack to stop just relocates the
    five-minute mystery it exists to remove."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(stdout="other_mlflow_1\n"))

    with pytest.raises(RuntimeError) as excinfo:
        assert_no_foreign_mlflow_stack("mine-allowlist")

    message = str(excinfo.value)
    assert "other_mlflow_1" in message, message
    assert "down -v --remove-orphans" in message, message


def test_guard_permits_this_projects_own_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other arm, without which the test above would pass just as happily
    against a guard that refuses everything: a leftover container belonging to
    THIS project is this fixture's own business (its `up`/`down` handle it) and
    must not be mistaken for a foreign stack."""
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Completed(stdout="mine-allowlist_mlflow_1\n")
    )

    assert_no_foreign_mlflow_stack("mine-allowlist")


def test_guard_permits_an_empty_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(stdout="\n\n"))

    assert_no_foreign_mlflow_stack("mine-allowlist")


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (subprocess.TimeoutExpired(cmd=["podman", "ps"], timeout=15.0), "could not verify"),
        (FileNotFoundError(2, "No such file or directory: 'podman'"), "could not verify"),
    ],
    ids=["probe-times-out", "podman-absent"],
)
def test_guard_reports_but_tolerates_a_probe_it_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    expected: str,
) -> None:
    """An UNVERIFIABLE precondition must not become a new flake source. The guard
    exists to remove a failure mode, so it may not add one: it says loudly that
    it could not check, and lets the run proceed.
    """

    def fake_run(*_args: Any, **_kwargs: Any) -> _Completed:
        raise failure

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert_no_foreign_mlflow_stack("mine-allowlist")

    assert expected in capsys.readouterr().out


def test_guard_reports_but_tolerates_a_nonzero_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Completed(stderr="Cannot connect to Podman", returncode=125),
    )

    assert_no_foreign_mlflow_stack("mine-allowlist")

    assert "could not verify" in capsys.readouterr().out
