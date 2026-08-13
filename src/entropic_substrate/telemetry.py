"""Read side of ``results/*.jsonl`` — the single validated reader for run telemetry.

Two consumers gate on these records: the CLI preflight for the anchored arms
(``train --exp e3-anchor``) and the ladder runner (``scripts/run_ladder.sh``).
They used to each parse the JSONL themselves, which is how one copy came to
validate its kappa rows and the other did not. The admissibility rules live
here once, so a rule pinned by a test holds for both.

What "admissible" means, and why each rule exists:

* the **final** eval record must carry the readout — gating on an older record
  that happens to have kappa hides a run whose late evals lost telemetry;
* rows must be non-empty and numeric — an interrupted writer leaves short rows,
  and ``bool`` is an ``int`` subclass so a JSON ``true`` would otherwise pass;
* values must be finite — ``json`` parses ``NaN``/``Infinity`` back without
  complaint, so a numerically failed run yields a NaN spread that compares
  false against every threshold and reads as a healthy zero-divergence gate.

An inadmissible readout is a :class:`TelemetryGap`, never a value: callers turn
it into a refusal, so nothing downstream can mistake absence for a measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from entropic_substrate.logging import configure_logging, logger

__all__ = [
    "EXIT_TELEMETRY_GAP",
    "KappaReadout",
    "TelemetryGap",
    "final_eval_metric",
    "final_eval_record",
    "main",
    "read_final_kappa",
]

# Callers exit with this when the readout a gate needs is absent or malformed.
EXIT_TELEMETRY_GAP: int = 2


class TelemetryGap(Exception):
    """The readout a gate needs is absent or malformed.

    Deliberately not a value: there is no "empty readout" object to accidentally
    compute a spread from.
    """


@dataclass(frozen=True, slots=True)
class KappaReadout:
    """Validated per-layer, per-head curvatures from one eval record."""

    path: Path
    rows: list[list[float]]

    @property
    def spreads(self) -> list[float]:
        """Within-layer head spread — the E3 divergence signal, per layer."""
        return [max(row) - min(row) for row in self.rows]

    @property
    def max_spread(self) -> float:
        """Largest within-layer head spread across layers."""
        return max(self.spreads)

    def report_lines(self) -> list[str]:
        """Human-readable divergence report (the operator's go/no-go input)."""
        spreads = self.spreads
        lines = [
            f"  layer {i}: {row}  (head spread {spreads[i]:.4f})" for i, row in enumerate(self.rows)
        ]
        lines.append(f"  max within-layer head spread: {self.max_spread:.4f}")
        return lines


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        msg = f"{path}: no such file — the run never wrote telemetry here"
        raise TelemetryGap(msg)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path}: line {number} is not valid JSON ({exc}) — truncated write?"
                raise TelemetryGap(msg) from exc
            if isinstance(record, dict):
                yield record


def final_eval_record(path: str | Path) -> dict[str, Any]:
    """The last ``event == "eval"`` record in ``path``."""
    path = Path(path)
    last: dict[str, Any] | None = None
    for record in _iter_records(path):
        if record.get("event") == "eval":
            last = record
    if last is None:
        msg = f"{path}: no eval record — the run never reached an eval"
        raise TelemetryGap(msg)
    return last


def final_eval_metric(path: str | Path, metric: str) -> float:
    """A finite scalar ``metric`` from the final eval record."""
    path = Path(path)
    record = final_eval_record(path)
    if metric not in record:
        msg = f"{path}: final eval record carries no {metric!r}"
        raise TelemetryGap(msg)
    value = record[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{path}: {metric} is not numeric: {value!r}"
        raise TelemetryGap(msg)
    if not math.isfinite(value):
        msg = f"{path}: non-finite {metric} {value!r}"
        raise TelemetryGap(msg)
    return float(value)


def _validate_rows(path: Path, kappa: Any) -> list[list[float]]:
    if not isinstance(kappa, list) or any(not isinstance(row, list) for row in kappa):
        msg = f"{path}: kappa is not a list of per-layer rows: {kappa!r}"
        raise TelemetryGap(msg)
    if not kappa:
        msg = f"{path}: kappa is empty — no layers reported"
        raise TelemetryGap(msg)
    rows: list[list[float]] = []
    for index, row in enumerate(kappa):
        if not row:
            msg = f"{path}: layer {index} kappa row is empty — telemetry is incomplete"
            raise TelemetryGap(msg)
        # bool is an int subclass, so a JSON `true` would otherwise read as 1.
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in row):
            msg = f"{path}: layer {index} kappa row is not numeric: {row!r}"
            raise TelemetryGap(msg)
        values = [float(v) for v in row]
        if not all(math.isfinite(v) for v in values):
            msg = f"{path}: layer {index} kappa row has non-finite values: {row!r}"
            raise TelemetryGap(msg)
        rows.append(values)
    return rows


def read_final_kappa(path: str | Path) -> KappaReadout:
    """Validated per-head curvatures from the final eval record in ``path``."""
    path = Path(path)
    record = final_eval_record(path)
    if "kappa" not in record or record["kappa"] is None:
        msg = (
            f"{path}: the final eval record carries no kappa telemetry "
            "(a run that evaluated without the curvature readout)"
        )
        raise TelemetryGap(msg)
    return KappaReadout(path=path, rows=_validate_rows(path, record["kappa"]))


# ------------------------------------------------------------------ module CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m entropic_substrate.telemetry",
        description="Validated reads of results/*.jsonl (used by scripts/run_ladder.sh)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("kappa-report", help="print the per-head kappa divergence")
    report.add_argument("path", type=Path)

    metric = sub.add_parser("final-eval", help="print one metric from the final eval")
    metric.add_argument("path", type=Path)
    metric.add_argument("--metric", default="nats_per_byte")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch. Returns a process exit code."""
    configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "kappa-report":
            for line in read_final_kappa(args.path).report_lines():
                logger.bind(report=True).info(line)
        else:
            # MACHINE-CONSUMED STDOUT, NOT LOGS: scripts/run_ladder.sh
            # captures this exact line via `$( )` and feeds it to `float()`.
            # A loguru prefix here breaks the ladder with a ValueError — see
            # entropic_substrate.logging's module docstring. Stays raw
            # print(), never routed through this module's sinks.
            print(final_eval_metric(args.path, args.metric))
    except TelemetryGap as gap:
        logger.error(f"telemetry gap: {gap}")
        return EXIT_TELEMETRY_GAP
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
