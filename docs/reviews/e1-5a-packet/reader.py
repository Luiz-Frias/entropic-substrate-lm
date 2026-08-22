"""Validated reader for the packet's JSONL run series.

Prose (docstrings/comments) was stripped from the original for blinding;
logic is unmodified. Parse the data ONLY through these functions.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("reader")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)


EXIT_TELEMETRY_GAP: int = 2


class TelemetryGap(Exception):
    pass


def kappa_spreads(rows: Sequence[Sequence[float]]) -> list[float]:
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
    return max(kappa_spreads(rows))


@dataclass(frozen=True, slots=True)
class KappaReadout:
    path: Path
    rows: list[list[float]]

    @property
    def spreads(self) -> list[float]:
        return kappa_spreads(self.rows)

    @property
    def max_spread(self) -> float:
        return max_kappa_spread(self.rows)

    def report_lines(self) -> list[str]:
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
    return tuple(len(row) for row in rows)


def read_final_kappa(path: str | Path) -> KappaReadout:
    path = Path(path)
    record = final_eval_record(path)
    if "kappa" not in record or record["kappa"] is None:
        msg = f"{path}: the final eval record carries no kappa telemetry (a run that evaluated without the curvature readout)"
        raise TelemetryGap(msg)
    return KappaReadout(path=path, rows=_validate_rows(path, record["kappa"]))


@dataclass(frozen=True, slots=True)
class EvalRecord:
    step: int
    loss: float
    nats_per_byte: float
    kappa: list[list[float]] | None
    field_nats_per_byte: float | None = None
    field_brier: float | None = None
    field_ece: float | None = None
    field_entropy: float | None = None
    field_top1: float | None = None


@dataclass(frozen=True, slots=True)
class TrainRecord:
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
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{path}: {key} is not an int: {value!r}"
        raise TelemetryGap(msg)
    return value


def read_eval_series(path: str | Path) -> list[EvalRecord]:
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
                shape, shape_step = (record_shape, step)
            elif record_shape != shape:
                msg = f"{path}: step {step} kappa is {len(record_shape)} layers with head counts {record_shape}, but step {shape_step} reported {len(shape)} layers with {shape} — the charts index kappa[layer][head] off the first kappa-bearing record, so a shape change downstream is a bare IndexError inside a plotting comprehension rather than a refusal"
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
        present = sorted((k for k, v in calibration_group.items() if v is not None))
        if present and len(present) < len(calibration_group):
            missing = sorted(set(calibration_group) - set(present))
            msg = f"{path}: step {step} carries a PARTIAL field-calibration group ({present} without {missing}) — the five calibration metrics are emitted together or not at all, so a partial group is telemetry corruption, not an optional channel"
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
