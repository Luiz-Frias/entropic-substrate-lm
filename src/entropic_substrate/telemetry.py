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
  false against every threshold and reads as a healthy zero-divergence gate;
* every kappa-bearing record in a series must share ONE shape (layer count,
  and head count per layer) with every other kappa-bearing record in that
  series — chart code derives ``n_layers``/``n_heads`` from the first
  kappa-bearing record and then indexes ``kappa[layer][head]`` across the
  rest, so a later record with a different shape is refused here rather than
  surfacing as a bare ``IndexError`` inside a plotting comprehension;
* a value with a documented domain must land IN it — E1.5c's calibration
  keys (P8, FOR-925) are probabilities/rates in ``[0, 1]`` (ECE, top-1) or
  non-negative (nats/byte, Brier, entropy); a finite-but-impossible reading
  (e.g. a negative entropy) is corruption the same way NaN is, just not
  caught by the finiteness check alone.

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

from entropic_substrate.evidence.d1_telemetry import (
    D1EvalEvidence,
    D1QualityEvidence,
    D1RunEvidence,
    D1RunStatus,
    D1TelemetryGap,
    final_d1_eval_metric,
    read_d1_eval_series,
    read_d1_run_evidence,
)
from entropic_substrate.logging import configure_logging, logger

__all__ = [
    "EXIT_TELEMETRY_GAP",
    "D1EvalEvidence",
    "D1QualityEvidence",
    "D1RunEvidence",
    "D1RunStatus",
    "D1TelemetryGap",
    "EvalRecord",
    "KappaReadout",
    "TelemetryGap",
    "TrainRecord",
    "final_d1_eval_metric",
    "final_eval_metric",
    "final_eval_record",
    "kappa_spreads",
    "main",
    "max_kappa_spread",
    "read_d1_eval_series",
    "read_d1_run_evidence",
    "read_eval_series",
    "read_final_kappa",
    "read_seed",
    "read_train_series",
]

# Callers exit with this when the readout a gate needs is absent or malformed.
EXIT_TELEMETRY_GAP: int = 2


class TelemetryGap(Exception):
    """The readout a gate needs is absent or malformed.

    Deliberately not a value: there is no "empty readout" object to accidentally
    compute a spread from.
    """


def kappa_spreads(rows: Sequence[Sequence[float]]) -> list[float]:
    """Within-layer head spread per layer — the E3 divergence signal.

    The ONE place this arithmetic lives: :attr:`KappaReadout.spreads` (Lane A)
    and the Lane B notebook's per-step aggregation both call it, so the two
    lanes cannot report different spreads for the same run. Empty input is a
    :class:`TelemetryGap` rather than the bare ``ValueError`` ``max(())``
    raises — a caller holding rows that never passed :func:`_validate_rows`
    must fail the same loud way every other reader here does.
    """
    if not rows:
        msg = "kappa spread of an empty matrix — no layers reported"
        raise TelemetryGap(msg)
    spreads: list[float] = []
    for index, row in enumerate(rows):
        if not row:
            msg = f"kappa spread of an empty layer {index} row — telemetry is incomplete"
            raise TelemetryGap(msg)
        spreads.append(max(row) - min(row))
    return spreads


def max_kappa_spread(rows: Sequence[Sequence[float]]) -> float:
    """Largest within-layer head spread across layers — one record's aggregate."""
    return max(kappa_spreads(rows))


@dataclass(frozen=True, slots=True)
class KappaReadout:
    """Validated per-layer, per-head curvatures from one eval record."""

    path: Path
    rows: list[list[float]]

    @property
    def spreads(self) -> list[float]:
        """Within-layer head spread — the E3 divergence signal, per layer."""
        return kappa_spreads(self.rows)

    @property
    def max_spread(self) -> float:
        """Largest within-layer head spread across layers."""
        return max_kappa_spread(self.rows)

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


def _kappa_shape(rows: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Per-layer head counts — the shape a chart's ``kappa[layer][head]``
    indexing assumes holds for every record in the series."""
    return tuple(len(row) for row in rows)


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


# ------------------------------------------------------------------- series
@dataclass(frozen=True, slots=True)
class EvalRecord:
    """One admissible ``event == "eval"`` row from a run's series (charts)."""

    step: int
    loss: float
    nats_per_byte: float
    kappa: list[list[float]] | None
    # P8 (FOR-925): field-calibration diagnostics for p_phi — same "absent
    # channel is absent" per-record optionality as `kappa` above. Present
    # only on runs with a field encoder (state_source="field"); every other
    # arm's eval records simply never carry these keys.
    field_nats_per_byte: float | None = None
    field_brier: float | None = None
    field_ece: float | None = None
    field_entropy: float | None = None
    field_top1: float | None = None


@dataclass(frozen=True, slots=True)
class TrainRecord:
    """One admissible ``event == "train"`` row from a run's series (charts)."""

    step: int
    loss: float
    nats_per_byte: float


def _finite_field(path: Path, record: dict[str, Any], key: str) -> float:
    if key not in record:
        msg = f"{path}: record carries no {key!r}"
        raise TelemetryGap(msg)
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{path}: {key} is not numeric: {value!r}"
        raise TelemetryGap(msg)
    if not math.isfinite(value):
        msg = f"{path}: non-finite {key} {value!r}"
        raise TelemetryGap(msg)
    return float(value)


def _finite_field_or_none(
    path: Path,
    record: dict[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Same admissibility rule as :func:`_finite_field`, but ``key`` is
    OPTIONAL per record (absent -> ``None``) — the convention every "absent
    channel is absent" telemetry key (``kappa``, and now the P8 calibration
    diagnostics) uses. ``minimum``/``maximum`` are inclusive range checks
    for metrics with a documented domain (ECE/top-1 accuracy are
    probabilities in ``[0, 1]``; nats/Brier/entropy are non-negative) — a
    value outside that domain is corruption the same way a non-finite value
    is, so it is refused here rather than passed through as a technically-
    finite-but-impossible reading.
    """
    if key not in record:
        return None
    value = _finite_field(path, record, key)
    if minimum is not None and value < minimum:
        msg = f"{path}: {key} is below its documented minimum {minimum}: {value!r}"
        raise TelemetryGap(msg)
    if maximum is not None and value > maximum:
        msg = f"{path}: {key} is above its documented maximum {maximum}: {value!r}"
        raise TelemetryGap(msg)
    return value


def _int_field(path: Path, record: dict[str, Any], key: str) -> int:
    if key not in record:
        msg = f"{path}: record carries no {key!r}"
        raise TelemetryGap(msg)
    value = record[key]
    # bool is an int subclass — see the module docstring's rationale for kappa.
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{path}: {key} is not an int: {value!r}"
        raise TelemetryGap(msg)
    return value


def read_eval_series(path: str | Path) -> list[EvalRecord]:
    """Every admissible ``event == "eval"`` record in ``path``, in file order.

    Same admissibility rules as :func:`final_eval_metric` / :func:`read_final_kappa`,
    applied to EVERY eval record rather than just the last one: a chart built
    from a partially-validated series could draw a healthy trajectory right up
    to a NaN-poisoned tail record it silently skipped. ``kappa`` is optional
    PER RECORD — E1/E2 runs never report it, and even an E3 run may drop it on
    one eval and not another — but when present it goes through the same
    :func:`_validate_rows` gate :func:`read_final_kappa` uses, AND its shape
    (layer count, head count per layer) must match every other kappa-bearing
    record in the series — see the module docstring's admissibility list.
    """
    path = Path(path)
    records: list[EvalRecord] = []
    shape: tuple[int, ...] | None = None
    shape_step: int | None = None
    for record in _iter_records(path):
        if record.get("event") != "eval":
            continue
        step = _int_field(path, record, "step")
        loss = _finite_field(path, record, "loss")
        nats_per_byte = _finite_field(path, record, "nats_per_byte")
        kappa = record.get("kappa")
        rows = _validate_rows(path, kappa) if kappa is not None else None
        if rows is not None:
            record_shape = _kappa_shape(rows)
            if shape is None:
                shape, shape_step = record_shape, step
            elif record_shape != shape:
                msg = (
                    f"{path}: step {step} kappa is {len(record_shape)} layers with head "
                    f"counts {record_shape}, but step {shape_step} reported {len(shape)} "
                    f"layers with {shape} — the charts index kappa[layer][head] off the "
                    "first kappa-bearing record, so a shape change downstream is a bare "
                    "IndexError inside a plotting comprehension rather than a refusal"
                )
                raise TelemetryGap(msg)
        field_nats_per_byte = _finite_field_or_none(
            path, record, "field_nats_per_byte", minimum=0.0
        )
        field_brier = _finite_field_or_none(path, record, "field_brier", minimum=0.0)
        field_ece = _finite_field_or_none(path, record, "field_ece", minimum=0.0, maximum=1.0)
        field_entropy = _finite_field_or_none(path, record, "field_entropy", minimum=0.0)
        field_top1 = _finite_field_or_none(path, record, "field_top1", minimum=0.0, maximum=1.0)
        calibration_group = {
            "field_nats_per_byte": field_nats_per_byte,
            "field_brier": field_brier,
            "field_ece": field_ece,
            "field_entropy": field_entropy,
            "field_top1": field_top1,
        }
        present = sorted(k for k, v in calibration_group.items() if v is not None)
        if present and len(present) < len(calibration_group):
            # All five travel as ONE group (train.py's eval record emits all
            # or none) — validating each independently would let a record
            # with a lost sibling read as admissible (CR PR #79). Runs after
            # the per-field checks above so corruption keeps its specific
            # error; this fires only on otherwise-valid partial records.
            missing = sorted(set(calibration_group) - set(present))
            msg = (
                f"{path}: step {step} carries a PARTIAL field-calibration group "
                f"({present} without {missing}) — the five calibration metrics "
                "are emitted together or not at all, so a partial group is "
                "telemetry corruption, not an optional channel"
            )
            raise TelemetryGap(msg)
        records.append(
            EvalRecord(
                step=step,
                loss=loss,
                nats_per_byte=nats_per_byte,
                kappa=rows,
                field_nats_per_byte=field_nats_per_byte,
                field_brier=field_brier,
                field_ece=field_ece,
                field_entropy=field_entropy,
                field_top1=field_top1,
            )
        )
    if not records:
        msg = f"{path}: no eval record — the run never reached an eval"
        raise TelemetryGap(msg)
    return records


def read_train_series(path: str | Path) -> list[TrainRecord]:
    """Every admissible ``event == "train"`` record in ``path``, in file order.

    Same fail-loud contract as :func:`read_eval_series`: one non-finite or
    malformed row anywhere in the series invalidates the whole read.
    """
    path = Path(path)
    records: list[TrainRecord] = []
    for record in _iter_records(path):
        if record.get("event") != "train":
            continue
        step = _int_field(path, record, "step")
        loss = _finite_field(path, record, "loss")
        nats_per_byte = _finite_field(path, record, "nats_per_byte")
        records.append(TrainRecord(step=step, loss=loss, nats_per_byte=nats_per_byte))
    if not records:
        msg = f"{path}: no train record — the run never logged a training step"
        raise TelemetryGap(msg)
    return records


def read_seed(path: str | Path) -> int:
    """The ``train.seed`` a run was launched with, from its ``run_start`` record.

    Pairing (E1) and seed-band grouping (charts) both key on this — read
    through the same validated-record iterator as every other reader here
    rather than a second ad-hoc JSONL parse.
    """
    path = Path(path)
    for record in _iter_records(path):
        if record.get("event") != "run_start":
            continue
        config = record.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("train"), dict):
            msg = f"{path}: run_start record carries no config.train"
            raise TelemetryGap(msg)
        train_cfg = config["train"]
        if "seed" not in train_cfg:
            msg = f"{path}: run_start record carries no train.seed"
            raise TelemetryGap(msg)
        seed = train_cfg["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            msg = f"{path}: train.seed is not an int: {seed!r}"
            raise TelemetryGap(msg)
        return seed
    msg = f"{path}: no run_start record — cannot recover the seed"
    raise TelemetryGap(msg)


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
