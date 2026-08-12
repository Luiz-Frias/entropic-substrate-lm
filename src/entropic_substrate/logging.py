"""loguru logging substrate — the human/diagnostic output layer for ``src/``.

:func:`configure_logging` wires three sinks on every call:

* **human stderr** — timestamped in the operator's LOCAL wall clock
  (``America/New_York`` via :mod:`zoneinfo`, so DST is handled rather than a
  fixed offset — pods run UTC, so this is also what renders pod logs in
  operator-local time when tailed from the operator's machine).
* **report stdout** — bare ``{message}``, no timestamp/level prefix. This is
  for the small set of call sites (kappa-divergence reports, human-readable
  eval summaries) that printed unprefixed lines to stdout *before* this
  module existed and still need to: not machine-parsed data (that stays raw
  ``print()``, see below), but a human-facing report block whose lines
  should not each carry a log prefix. Call sites opt in via
  ``logger.bind(report=True).info(...)``.
* **structured JSON** — UTC ISO-8601, appended to ``json_path`` (default
  ``logs/entropic-substrate.jsonl``, already ``.gitignore``d). Absolute time
  for the persisted/machine-parsed side, mirroring the human sink's
  operator-local time for the terminal side.

LAZY BY DESIGN — never called at import. loguru binds a stream sink to the
literal object passed to :func:`loguru.logger.add`, captured once, at
handler-CREATION time. pytest's ``capsys`` fixture replaces
``sys.stdout``/``sys.stderr`` with a fresh object *per test*; a handler
created once (e.g. at import time, or by an earlier test) keeps writing to
whatever stream object it was given, which a later test's
``capsys.readouterr()`` never reads from. Every existing "gate text landed
in stderr" assertion would keep *passing* while asserting on nothing.

The fix is calling :func:`configure_logging` at CLI entry — after argv
parsing starts, never at module import — and, for tests that invoke library
functions directly rather than through a CLI entry point (bypassing that
call entirely), a per-test ``conftest.py`` fixture that rebinds the sinks to
whatever ``sys.stdout``/``sys.stderr`` are live for that test.

MACHINE-CONSUMED STDOUT IS NOT LOGS: ``entropic_substrate.telemetry``'s
``final-eval`` subcommand prints a bare float that ``scripts/run_ladder.sh``
parses via ``$( ) | float()``, and ``entropic_substrate.cli``'s ``eval
--json`` prints one JSON object parsed by ``tests/test_train_smoke.py``.
Both of those two call sites stay raw ``print()`` and never touch this
module — a loguru prefix on either is a hard break, not a formatting nit.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from loguru import Message, Record

__all__ = ["configure_logging", "logger"]

# The operator's timezone — DST-aware. A fixed "-05:00 EST" offset is wrong
# for roughly 8 months of the year (EDT, UTC-4); zoneinfo carries the actual
# IANA transition rules instead of a hand-maintained constant.
_LOCAL_TZ = ZoneInfo("America/New_York")

_DEFAULT_JSON_PATH = Path("logs/entropic-substrate.jsonl")


def _is_report(record: Record) -> bool:
    """True for records emitted via ``logger.bind(report=True)``."""
    return bool(record["extra"].get("report", False))


def _human_format(record: Record) -> str:
    """Format callable for the human stderr sink.

    Returns a TEMPLATE, not the fully-substituted string — loguru runs its
    own ``{}``-substitution over whatever this returns. Embedding
    ``record["message"]``'s *content* directly (e.g. via an f-string) would
    make that content part of the template loguru re-parses, and several
    messages in this codebase carry literal ``{``/``}`` (dict reprs inside
    ``TelemetryGap`` messages, e.g. ``"...: {'layer0': [0.1]}"``) — those
    would raise ``KeyError`` on the second pass. Referencing ``{message}``
    as a placeholder keeps the substituted value opaque (a single
    ``str.format`` pass never re-scans what it inserts), same for
    ``{extra[local_ts]}`` below.
    """
    local = record["time"].astimezone(_LOCAL_TZ)
    record["extra"]["local_ts"] = local.strftime("%Y-%m-%d %H:%M:%S %Z")
    return "{extra[local_ts]} | {level: <8} | {message}\n"


def _report_format(_record: Record) -> str:
    """Format callable for the report stdout sink — bare message, no prefix."""
    return "{message}\n"


def _json_sink(path: Path) -> Callable[[Message], None]:
    """Build a sink function appending one UTC-ISO-8601 JSON object per record.

    A sink *function* rather than ``format=`` on a stream sink: this needs
    full control of the timestamp representation (absolute UTC, explicit
    ``+00:00`` offset), not loguru's own ``serialize=True`` shape.
    """

    def sink(message: Message) -> None:
        record = message.record
        payload: dict[str, Any] = {
            "timestamp": record["time"].astimezone(UTC).isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "message": record["message"],
        }
        if record["exception"] is not None:
            payload["exception"] = str(record["exception"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    return sink


def configure_logging(
    *,
    level: str = "INFO",
    json_path: str | Path | None = _DEFAULT_JSON_PATH,
) -> None:
    """(Re)configure loguru's sinks, bound to the CURRENT stdio streams.

    Idempotent: every call removes all existing sinks first (including
    loguru's own default stderr handler installed on import), so repeated
    calls — once per CLI entry, once per test via the ``conftest.py``
    fixture — never double-print and always follow a since-swapped
    ``sys.stdout``/``sys.stderr``. See the module docstring for why calling
    this lazily (never at import) is load-bearing, not a style choice.

    Args:
        level: minimum level for all three sinks.
        json_path: where the structured JSON sink appends records, or
            ``None`` to skip it entirely (used by the test suite, which
            exercises the JSON sink directly rather than writing
            ``logs/*.jsonl`` on every one of ~300+ test runs).
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        format=_human_format,
        filter=lambda record: not _is_report(record),
        colorize=False,
    )
    logger.add(
        sys.stdout,
        level=level,
        format=_report_format,
        filter=_is_report,
        colorize=False,
    )
    if json_path is not None:
        logger.add(_json_sink(Path(json_path)), level=level)
