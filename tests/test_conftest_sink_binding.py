"""Regression pin: the conftest's loguru sinks reach the stream capsys reads.

FOR-1649. ``tests/conftest.py`` configures loguru in an autouse fixture. A
``logger.*`` call made in a test's CALL phase must land in that test's
``capsys.readouterr().err`` as a formatted ``<ts> | LEVEL | message`` line
(``.out`` for ``report=True``), never as loguru's "Logging error" dump.

ROOT CAUSE, read from the installed pytest 9.1.1 (``_pytest/capture.py``):

* ``CaptureManager.item_capture`` wraps EACH of setup, call and teardown
  (``capture.py:858-869``), and ``activate_fixture`` / ``deactivate_fixture``
  run inside it.
* ``CaptureFixture._start`` (``:936-943``) builds a NEW ``MultiCapture``,
  so a NEW ``sys.stderr`` object, when capsys is first started; ``close``
  (``:945-958``) calls ``stop_capturing`` which reaches
  ``SysCaptureBase.done`` and ``self.tmpfile.close()`` (``:407-414``).
* The swap is therefore per PHASE, not per test. A sink that
  ``logger.add(sys.stderr)`` bound during fixture SETUP holds the setup
  phase's stream; it is closed by the time the call phase runs, and every
  record loguru writes there is replaced by a "Logging error in Loguru
  Handler" block whose ``Record was: {...}`` dump still carries the message
  text, so substring assertions kept passing against the dump.

This module never calls ``configure_logging``, ``logger.add`` or
``logger.remove``: it observes only what the autouse conftest fixture
installed. A pin that reconfigured itself would stay GREEN with the
conftest broken.
"""

from __future__ import annotations

import sys
from typing import TextIO

import pytest
from loguru import logger

pytestmark = pytest.mark.unit

_HUMAN_WARNING = "| WARNING  | "


@pytest.fixture
def setup_phase_stderr(capsys: pytest.CaptureFixture[str]) -> TextIO:
    """The ``sys.stderr`` that is current while fixtures are set up.

    Depends on ``capsys`` so it runs after capsys's swap, in the same phase
    in which the autouse conftest fixture binds its sinks.
    """
    return sys.stderr


def test_precondition_setup_stream_is_closed_by_the_call_phase(
    setup_phase_stderr: TextIO,
) -> None:
    """CONTROL, labelled apart from the pin: the hazard exists in this regime.

    If pytest ever keeps one stream for the whole test, the pin below passes
    without proving anything; this test then fails loudly instead.
    """
    call_stderr = sys.stderr
    assert call_stderr is not setup_phase_stderr, (
        f"setup id={id(setup_phase_stderr)} call id={id(call_stderr)}"
    )
    assert setup_phase_stderr.closed, f"setup id={id(setup_phase_stderr)} still open at call"


def test_call_phase_warning_lands_as_one_formatted_stderr_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger.warning("pin-call-phase-marker")
    captured = capsys.readouterr()
    assert "Logging error" not in captured.err, captured.err
    assert "Record was:" not in captured.err, captured.err
    lines = [line for line in captured.err.splitlines() if "pin-call-phase-marker" in line]
    assert len(lines) == 1, captured.err
    assert _HUMAN_WARNING in lines[0], lines[0]
    assert lines[0].endswith(_HUMAN_WARNING + "pin-call-phase-marker"), lines[0]
    assert "pin-call-phase-marker" not in captured.out


def test_call_phase_report_record_lands_bare_on_stdout_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger.bind(report=True).info("pin-report-marker")
    captured = capsys.readouterr()
    assert captured.out == "pin-report-marker\n", repr(captured.out)
    assert "pin-report-marker" not in captured.err
    assert "Logging error" not in captured.err, captured.err


@pytest.fixture
def setup_phase_emission(capsys: pytest.CaptureFixture[str]) -> None:
    """Log during fixture SETUP, after capsys; the call phase reads it."""
    logger.warning("pin-setup-phase-marker")


@pytest.mark.usefixtures("setup_phase_emission")
def test_setup_phase_warning_is_a_formatted_line_too(
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = capsys.readouterr()
    assert "Logging error" not in captured.err, captured.err
    lines = [line for line in captured.err.splitlines() if "pin-setup-phase-marker" in line]
    assert len(lines) == 1, captured.err
    assert _HUMAN_WARNING in lines[0], lines[0]
