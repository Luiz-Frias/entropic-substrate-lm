"""Strict checkpoint-admission codec for immutable :class:`ResourceLedgerState`."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import NoReturn, cast

from entropic_substrate.evidence.resource_ledger import ResourceLedgerState
from entropic_substrate.evidence.resource_provenance import Money, execution_segment_id
from entropic_substrate.evidence.resource_work import CostReading, ReadingMethod, ScalarReading
from entropic_substrate.experiments.d1_primitives import EvidenceClass

__all__ = [
    "ResourceLedgerCodecGap",
    "parse_resource_ledger_state_line",
    "resource_ledger_state_from_mapping",
    "resource_ledger_state_to_mapping",
    "serialize_resource_ledger_state_line",
]

_EVENT = "resource.ledger_state"
_READING_FIELDS = {
    "gpu_train_device_seconds",
    "gpu_eval_device_seconds",
    "cumulative_train_flops",
    "cumulative_eval_flops",
    "energy_joules",
    "rate_estimated_cost",
}
_KEYS = {
    "event",
    "schema_version",
    "logical_run_id",
    "execution_segment_index",
    "execution_segment_id",
    "next_eval_index",
    "last_training_step",
    "trainer_wall_seconds",
    "committed_quality_source_bytes",
    "committed_quality_model_tokens",
    "online_train_token_positions",
    "target_train_token_positions",
    "baseline_calibration_positions",
    "eval_positions",
    "total_model_token_positions",
    "last_optional_readings",
}


class ResourceLedgerCodecGap(Exception):
    """A ledger checkpoint mapping cannot safely restore a cumulative accumulator."""


def _fail(message: str, exc: Exception | None = None) -> NoReturn:
    if exc is None:
        raise ResourceLedgerCodecGap(message)
    raise ResourceLedgerCodecGap(message) from exc


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return value


def _keys(mapping: dict[str, object], expected: set[str], *, label: str) -> None:
    if set(mapping) != expected:
        _fail(f"{label} keys must be exact")


def _enum(enum: type[EvidenceClass] | type[ReadingMethod], value: object, *, label: str) -> object:
    if type(value) is not str:
        _fail(f"{label} must be a canonical enum string")
    try:
        return enum(value)
    except ValueError as exc:
        _fail(f"{label} enum is unsupported", exc)


def _scalar(field: str, mapping: dict[str, object]) -> ScalarReading:
    _keys(mapping, {"kind", "value", "evidence_class", "method"}, label=field)
    if mapping["kind"] != "scalar":
        _fail(f"{field} must be a scalar reading")
    try:
        reading = ScalarReading(
            value=cast(float, mapping["value"]),
            evidence_class=cast(
                EvidenceClass, _enum(EvidenceClass, mapping["evidence_class"], label=field)
            ),
            method=cast(ReadingMethod, _enum(ReadingMethod, mapping["method"], label=field)),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{field} scalar is invalid", exc)
    allowed = {
        "gpu_train_device_seconds": {(EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER)},
        "gpu_eval_device_seconds": {(EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER)},
        "cumulative_train_flops": {
            (EvidenceClass.MEASURED, ReadingMethod.FLOP_PROFILER),
            (EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL),
        },
        "cumulative_eval_flops": {
            (EvidenceClass.MEASURED, ReadingMethod.FLOP_PROFILER),
            (EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL),
        },
        "energy_joules": {
            (EvidenceClass.MEASURED, ReadingMethod.ENERGY_COUNTER),
            (EvidenceClass.DERIVED_FROM_MEASURED, ReadingMethod.ENERGY_INTEGRATED),
        },
    }
    if field not in allowed or (reading.evidence_class, reading.method) not in allowed[field]:
        _fail(f"{field} evidence_class/method is noncanonical")
    return reading


def _cost(field: str, mapping: dict[str, object]) -> CostReading:
    _keys(mapping, {"kind", "amount", "evidence_class", "method"}, label=field)
    if field != "rate_estimated_cost" or mapping["kind"] != "cost":
        _fail(f"{field} must be a rate-estimated cost reading")
    money = _mapping(mapping["amount"], label=f"{field}.amount")
    _keys(money, {"amount", "currency"}, label=f"{field}.amount")
    if type(money["amount"]) is not str:
        _fail(f"{field}.amount.amount must be a canonical decimal string")
    try:
        return CostReading(
            amount=Money(amount=Decimal(money["amount"]), currency=cast(str, money["currency"])),
            evidence_class=cast(
                EvidenceClass, _enum(EvidenceClass, mapping["evidence_class"], label=field)
            ),
            method=cast(ReadingMethod, _enum(ReadingMethod, mapping["method"], label=field)),
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        _fail(f"{field} cost is invalid or noncanonical", exc)


def _last_readings(value: object) -> tuple[tuple[str, ScalarReading | CostReading], ...]:
    readings = _mapping(value, label="last_optional_readings")
    if not set(readings).issubset(_READING_FIELDS):
        _fail("last_optional_readings contains an unregistered meter field")
    parsed: list[tuple[str, ScalarReading | CostReading]] = []
    for field, item in sorted(readings.items()):
        mapping = _mapping(item, label=f"last_optional_readings.{field}")
        parsed.append(
            (
                field,
                _cost(field, mapping)
                if field == "rate_estimated_cost"
                else _scalar(field, mapping),
            )
        )
    return tuple(parsed)


def resource_ledger_state_to_mapping(state: ResourceLedgerState) -> dict[str, object]:
    """Map an already-validated checkpoint state to its exact canonical representation."""
    if type(state) is not ResourceLedgerState:
        _fail("ledger codec requires ResourceLedgerState")
    return {"event": _EVENT, **state.manifest}


def resource_ledger_state_from_mapping(value: object) -> ResourceLedgerState:
    """Refuse noncanonical checkpoint data before it enters accumulator restore."""
    mapping = _mapping(value, label="resource ledger state")
    _keys(mapping, _KEYS, label="resource ledger state")
    if mapping["event"] != _EVENT:
        _fail("resource ledger state event is unsupported")
    try:
        expected_segment_id = execution_segment_id(
            cast(str, mapping["logical_run_id"]), cast(int, mapping["execution_segment_index"])
        )
        total = (
            cast(int, mapping["online_train_token_positions"])
            + cast(int, mapping["target_train_token_positions"])
            + cast(int, mapping["baseline_calibration_positions"])
            + cast(int, mapping["eval_positions"])
        )
    except (TypeError, ValueError) as exc:
        _fail("ledger state identity or total counters are invalid", exc)
    if mapping["execution_segment_id"] != expected_segment_id:
        _fail("execution_segment_id is not derived from logical_run_id/index")
    if mapping["total_model_token_positions"] != total:
        _fail("total_model_token_positions does not match the exact counter sum")
    try:
        return ResourceLedgerState(
            logical_run_id=cast(str, mapping["logical_run_id"]),
            execution_segment_index=cast(int, mapping["execution_segment_index"]),
            next_eval_index=cast(int, mapping["next_eval_index"]),
            last_training_step=cast(int | None, mapping["last_training_step"]),
            trainer_wall_seconds=cast(float, mapping["trainer_wall_seconds"]),
            committed_quality_source_bytes=cast(int, mapping["committed_quality_source_bytes"]),
            committed_quality_model_tokens=cast(int, mapping["committed_quality_model_tokens"]),
            online_train_token_positions=cast(int, mapping["online_train_token_positions"]),
            target_train_token_positions=cast(int, mapping["target_train_token_positions"]),
            baseline_calibration_positions=cast(int, mapping["baseline_calibration_positions"]),
            eval_positions=cast(int, mapping["eval_positions"]),
            last_optional_readings=_last_readings(mapping["last_optional_readings"]),
            schema_version=cast(str, mapping["schema_version"]),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"resource ledger state is invalid: {exc}", exc)


def serialize_resource_ledger_state_line(state: ResourceLedgerState) -> str:
    return json.dumps(
        resource_ledger_state_to_mapping(state),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant {value!r} is forbidden")


def _duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key, value in pairs:
        if key in mapping:
            _fail(f"duplicate JSON key {key!r}")
        mapping[key] = value
    return mapping


def parse_resource_ledger_state_line(line: str) -> ResourceLedgerState:
    if type(line) is not str or not line.strip():
        _fail("resource ledger state line must be nonblank")
    try:
        value = json.loads(line, object_pairs_hook=_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        _fail("resource ledger state line is invalid JSON", exc)
    return resource_ledger_state_from_mapping(value)
