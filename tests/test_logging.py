"""Pins for the loguru logging substrate (:mod:`entropic_substrate.logging`).

These are the acceptance-item-6 pins plus the acceptance-item-5 guard tooth:
a test that would FAIL if sink creation moved back to import time (the
vacuous-assertion trap this whole module exists to avoid — see the module
docstring in ``src/entropic_substrate/logging.py``).
"""

from __future__ import annotations

import importlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from loguru import logger

import entropic_substrate.kernel.runtime.logging as est_logging
from entropic_substrate.kernel.runtime.logging import configure_logging

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
    # FOR-1649: a bare substring check also matched the `Record was: {...}`
    # dump a stale sink writes, so it stayed green with the guard broken.
    assert "Logging error" not in captured.err, captured.err
    assert "| INFO     | marker-xyz-configure-rebinds" in captured.err
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


class _CountingStream(io.StringIO):
    """A StringIO that counts ``flush()`` calls (loguru's stream sink flushes per write)."""

    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def _swap_stdio(monkeypatch: pytest.MonkeyPatch) -> tuple[_CountingStream, _CountingStream]:
    """Install fresh stand-ins for stdout and stderr; return them in that order."""
    out, err = _CountingStream(), _CountingStream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def test_default_binds_the_streams_current_at_configure_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FOR-1649, default arm: ``resolve_streams`` off keeps the bind-at-call
    behaviour every production caller relies on. A later swap of ``sys.stderr``
    must NOT move the sink (the stream configured first keeps the line)."""
    bound_out, bound_err = _swap_stdio(monkeypatch)
    configure_logging(json_path=None)
    later_out, later_err = _swap_stdio(monkeypatch)
    logger.info("default-arm-human")
    logger.bind(report=True).info("default-arm-report")
    assert "| INFO     | default-arm-human" in bound_err.getvalue()
    assert bound_out.getvalue() == "default-arm-report\n"
    assert later_err.getvalue() == ""
    assert later_out.getvalue() == ""


def test_resolve_streams_follows_a_later_swap_of_both_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FOR-1649, resolving arm: the sinks write to whatever ``sys.stderr`` /
    ``sys.stdout`` is live at WRITE time, in the production human and report
    shapes, and leave the streams current at configure time untouched."""
    bound_out, bound_err = _swap_stdio(monkeypatch)
    configure_logging(json_path=None, resolve_streams=True)
    live_out, live_err = _swap_stdio(monkeypatch)
    logger.warning("resolve-arm-human")
    logger.bind(report=True).info("resolve-arm-report")
    assert "| WARNING  | resolve-arm-human" in live_err.getvalue()
    assert live_out.getvalue() == "resolve-arm-report\n"
    assert bound_err.getvalue() == ""
    assert bound_out.getvalue() == ""


def test_resolve_streams_flushes_the_stream_it_resolved_after_each_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report stream is block-buffered off a TTY; a sink that wrote
    without flushing would hold a report line back. The resolving sink must
    flush the stream it resolved, once per record."""
    live_out, live_err = _swap_stdio(monkeypatch)
    configure_logging(json_path=None, resolve_streams=True)
    logger.info("flush-human")
    logger.bind(report=True).info("flush-report")
    assert live_err.flushes == 1
    assert live_out.flushes == 1


def test_resolve_streams_survives_a_closed_earlier_stream(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure this option exists for: a stream closed after configure
    time (a finished capture phase) must not turn the next record into a
    loguru "Logging error" dump."""
    closed_out, closed_err = _swap_stdio(monkeypatch)
    configure_logging(json_path=None, resolve_streams=True)
    closed_out.close()
    closed_err.close()
    _, live_err = _swap_stdio(monkeypatch)
    logger.error("after-close")
    written = live_err.getvalue()
    assert written.count("| ERROR    | after-close") == 1, written
    # loguru prints its dump to the stderr live at failure time (here the
    # stand-in), so the absence check must read that stream, not capsys.
    assert "Logging error" not in written, written
    assert "Logging error" not in capsys.readouterr().err


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
    placeholder must not.

    Guard tooth: a bare ``"... in captured.err"`` substring check is
    toothless here — when the formatter DOES crash, loguru catches the
    exception and dumps a "--- Logging error in Loguru Handler ---" block
    to stderr that echoes the offending record's ``message`` field verbatim
    (via ``repr()``), so the exact substring this test looks for is present
    on BOTH the success and the crash path. Asserting the absence of
    loguru's own crash-dump markers is what actually distinguishes them —
    reintroducing the f-string bug turns this red.
    """
    configure_logging(json_path=None)
    logger.error("kappa is not a list of per-layer rows: {'layer0': [0.1]}")
    captured = capsys.readouterr()
    assert "Logging error" not in captured.err
    assert "Traceback" not in captured.err
    assert captured.err.count("{'layer0': [0.1]}") == 1
    assert "{'layer0': [0.1]}" in captured.err


# -------------------------------------------------------------- timezone
def test_human_format_shows_edt_wall_clock_in_summer(capsys: pytest.CaptureFixture[str]) -> None:
    """A UTC instant in July, run through the PRODUCTION ``_human_format``
    sink (not a standalone zoneinfo computation), renders America/New_York's
    summer offset (EDT, UTC-4) — proving DST is handled, not a fixed offset.

    Uses ``logger.patch()`` to inject a controlled ``record["time"]`` ahead
    of the real sink, so this exercises ``est_logging._human_format``
    itself: a regression to a hardcoded ``-05:00``/'EST' offset in that
    function would make this assertion fail.
    """
    configure_logging(json_path=None)
    summer_utc = datetime(2026, 7, 15, 16, 0, 0, tzinfo=_UTC)  # 16:00 UTC
    logger.patch(lambda record: record.update(time=summer_utc)).info("dst-summer-marker")
    err = capsys.readouterr().err
    assert "2026-07-15 12:00:00" in err  # EDT, UTC-4
    assert "dst-summer-marker" in err


def test_human_format_shows_est_wall_clock_in_winter(capsys: pytest.CaptureFixture[str]) -> None:
    """Same mechanism as the summer test, for a January instant — EST,
    UTC-5. The two tests together, both driven through the real sink, are
    what proves ``zoneinfo`` (DST-aware) is in use rather than a hardcoded
    offset, which would be silently wrong for ~8 months of the year."""
    configure_logging(json_path=None)
    winter_utc = datetime(2026, 1, 15, 16, 0, 0, tzinfo=_UTC)  # 16:00 UTC
    logger.patch(lambda record: record.update(time=winter_utc)).info("dst-winter-marker")
    err = capsys.readouterr().err
    assert "2026-01-15 11:00:00" in err  # EST, UTC-5
    assert "dst-winter-marker" in err


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


def test_json_path_none_disables_the_json_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``json_path=None`` (what the test-suite-wide conftest fixture passes)
    must not write anything anywhere.

    chdir FIRST, and the assertion stops being vacuous. _DEFAULT_JSON_PATH is
    RELATIVE ("logs/entropic-substrate.jsonl"), so without an isolated cwd a
    regression that ignored ``None`` and fell back to the default would write
    into the repo — under ``logs/``, which is gitignored, so it would not even
    show up in ``git status``. The old assertion could never fail: nothing
    writes to tmp_path when nothing ever cd'd there.
    """
    monkeypatch.chdir(tmp_path)
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
