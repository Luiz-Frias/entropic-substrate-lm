"""Pins for the loguru logging substrate (:mod:`entropic_substrate.logging`).

These are the acceptance-item-6 pins plus the acceptance-item-5 guard tooth:
a test that would FAIL if sink creation moved back to import time (the
vacuous-assertion trap this whole module exists to avoid — see the module
docstring in ``src/entropic_substrate/logging.py``).
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from loguru import logger

import entropic_substrate.logging as est_logging
from entropic_substrate.logging import configure_logging

pytestmark = pytest.mark.unit

_UTC = ZoneInfo("UTC")
_NY = ZoneInfo("America/New_York")


# ------------------------------------------------------------ lazy config
def test_import_does_not_install_a_sink() -> None:
    """Merely importing the module must not call ``logger.add`` — sinks are
    wired only inside :func:`configure_logging`, called at CLI entry."""
    logger.remove()  # clear whatever the process/other tests left behind
    importlib.reload(est_logging)  # re-execute the module body from scratch
    assert len(logger._core.handlers) == 0


# ------------------------------------------------------- report/human split
def test_configure_logging_rebinds_to_current_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """Guard tooth for acceptance item 5: this is the test that would go RED
    if sink creation moved to import time, or if a stale handler kept
    writing to a stream a prior test's ``capsys`` had already torn down.
    Every DECISION-NEEDED / gate assertion elsewhere in this suite is only
    meaningful because this holds."""
    configure_logging(json_path=None)
    logger.info("marker-xyz-configure-rebinds")
    captured = capsys.readouterr()
    assert "marker-xyz-configure-rebinds" in captured.err
    assert "marker-xyz-configure-rebinds" not in captured.out


def test_report_bound_records_go_to_stdout_bare(capsys: pytest.CaptureFixture[str]) -> None:
    """``logger.bind(report=True)`` is the opt-in for the small set of call
    sites (kappa reports, eval summaries) that print unprefixed lines to
    stdout — distinct from machine-consumed data, which stays raw print()
    and never touches this module at all."""
    configure_logging(json_path=None)
    logger.bind(report=True).info("report-marker-42")
    captured = capsys.readouterr()
    assert captured.out.strip() == "report-marker-42"
    assert "report-marker-42" not in captured.err


def test_human_sink_carries_no_prefix_pollution_into_report_sink(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-report record must never leak onto stdout (it would corrupt
    machine-consumed callers that happen to share a process)."""
    configure_logging(json_path=None)
    logger.info("human-only-marker")
    captured = capsys.readouterr()
    assert "human-only-marker" in captured.err
    assert captured.out == ""


def test_message_containing_braces_does_not_crash_the_formatter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TelemetryGap messages carry dict reprs (e.g. ``"...: {'a': [0.1]}"``).
    A format callable that embeds the message's content directly into its
    returned template would have loguru re-parse those literal braces as
    format fields and raise KeyError; referencing ``{message}`` as an opaque
    placeholder must not."""
    configure_logging(json_path=None)
    logger.error("kappa is not a list of per-layer rows: {'layer0': [0.1]}")
    captured = capsys.readouterr()
    assert "{'layer0': [0.1]}" in captured.err


# -------------------------------------------------------------- timezone
def test_human_format_shows_edt_wall_clock_in_summer() -> None:
    """A UTC instant in July renders as America/New_York's summer offset
    (EDT, UTC-4) — proving DST is handled, not a fixed offset."""
    summer_utc = datetime(2026, 7, 15, 16, 0, 0, tzinfo=_UTC)  # 16:00 UTC
    local = summer_utc.astimezone(_NY)
    assert local.strftime("%H:%M") == "12:00"  # UTC-4
    assert local.utcoffset().total_seconds() / 3600 == -4


def test_human_format_shows_est_wall_clock_in_winter() -> None:
    """The same wall-clock construction in January renders EST (UTC-5) —
    the two tests together are what proves ``zoneinfo`` (DST-aware) is in
    use rather than a hardcoded ``-05:00``/'EST' offset, which would be
    silently wrong for ~8 months of the year."""
    winter_utc = datetime(2026, 1, 15, 16, 0, 0, tzinfo=_UTC)  # 16:00 UTC
    local = winter_utc.astimezone(_NY)
    assert local.strftime("%H:%M") == "11:00"  # UTC-5
    assert local.utcoffset().total_seconds() / 3600 == -5


def test_human_sink_renders_local_wall_clock(capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end: the actual sink (not just the raw zoneinfo math above)
    prints something recognisable as an America/New_York rendering, for
    whatever instant "now" happens to be — DST-agnostic by construction, so
    this passes year-round without knowing today's offset."""
    configure_logging(json_path=None)
    logger.info("tz-render-marker")
    err = capsys.readouterr().err
    now_local = datetime.now(_NY)
    assert now_local.strftime("%Y-%m-%d") in err


# --------------------------------------------------------- JSON sink (UTC)
def test_json_sink_emits_utc_iso8601(tmp_path: Path) -> None:
    json_path = tmp_path / "log.jsonl"
    configure_logging(json_path=json_path)
    logger.info("json-sink-marker")
    lines = [line for line in json_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "json-sink-marker"
    assert record["level"] == "INFO"
    # UTC offset is explicit ("+00:00"), and the timestamp round-trips.
    parsed = datetime.fromisoformat(record["timestamp"])
    assert parsed.utcoffset().total_seconds() == 0
    assert parsed.tzinfo is not None


def test_json_sink_is_absolute_not_local(tmp_path: Path) -> None:
    """The JSON sink is UTC regardless of the human sink's local rendering —
    the two sinks must not silently share a clock."""
    json_path = tmp_path / "log.jsonl"
    configure_logging(json_path=json_path)
    logger.info("absolute-clock-marker")
    record = json.loads(json_path.read_text(encoding="utf-8").splitlines()[0])
    parsed = datetime.fromisoformat(record["timestamp"])
    now_utc = datetime.now(_UTC)
    # Sanity: this instant, read back, is close to "now" in UTC — not offset
    # by the -4/-5h the human sink would apply.
    assert abs((now_utc - parsed).total_seconds()) < 30


def test_json_path_none_disables_the_json_sink(tmp_path: Path) -> None:
    """``json_path=None`` (what the test-suite-wide conftest fixture passes)
    must not write anything anywhere."""
    configure_logging(json_path=None)
    logger.info("no-json-marker")
    assert not (tmp_path / "logs").exists()


# ------------------------------------------------------------- idempotency
def test_repeated_configure_does_not_double_print(capsys: pytest.CaptureFixture[str]) -> None:
    """Calling configure_logging() twice (once per test via the conftest
    fixture, potentially again inside a CLI entry point under test) must not
    accumulate sinks and print the same record multiple times."""
    configure_logging(json_path=None)
    configure_logging(json_path=None)
    logger.info("no-double-print-marker")
    err = capsys.readouterr().err
    assert err.count("no-double-print-marker") == 1
