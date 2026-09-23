"""Strict canonical JSON mapping codec for one :class:`ResourceWorkSnapshot`."""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import NoReturn, cast

from entropic_substrate.kernel.evidence_class.resource_provenance import EvidenceGap, Money
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.ladder.capture.resource.resource_work import (
    CostReading,
    ReadingMethod,
    ResourceWorkSnapshot,
    ScalarReading,
)

__all__ = [
    "ResourceWorkCodecGap",
    "parse_resource_snapshot_line",
    "resource_snapshot_from_mapping",
    "resource_snapshot_to_mapping",
    "serialize_resource_snapshot_line",
]

_EVENT = "resource.work_snapshot"
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_KEYS = {
    "event",
    "schema_version",
    "logical_run_id",
    "execution_segment_index",
    "execution_segment_id",
    "training_step",
    "eval_index",
    "observed_at",
    "trainer_wall_seconds",
    "committed_quality_source_bytes",
    "committed_quality_model_tokens",
    "online_train_token_positions",
    "target_train_token_positions",
    "baseline_calibration_positions",
    "eval_positions",
    "total_model_token_positions",
    "gpu_train_device_seconds",
    "gpu_eval_device_seconds",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
    "cumulative_train_flops",
    "cumulative_eval_flops",
    "flop_model_identity",
    "energy_joules",
    "energy_sample_count",
    "rate_estimated_cost",
    "required_counter_evidence",
}


class ResourceWorkCodecGap(Exception):
    """A one-snapshot resource-work wire record is malformed or noncanonical."""


def _fail(message: str, exc: Exception | None = None) -> NoReturn:
    if exc is None:
        raise ResourceWorkCodecGap(message)
    raise ResourceWorkCodecGap(message) from exc


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return value


def _exact_keys(mapping: dict[str, object], expected: set[str], *, label: str) -> None:
    if set(mapping) != expected:
        _fail(f"{label} keys must be exact")


def _timestamp(value: object) -> datetime:
    if type(value) is not str or _TIMESTAMP.fullmatch(value) is None:
        _fail("observed_at must be a canonical UTC Z timestamp")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        _fail("observed_at is invalid", exc)


def _enum(enum: type[EvidenceClass] | type[ReadingMethod], value: object, *, label: str) -> object:
    if type(value) is not str:
        _fail(f"{label} must be a canonical enum string")
    try:
        return enum(value)
    except ValueError as exc:
        _fail(f"{label} is not a supported enum value", exc)


def _gap(mapping: dict[str, object], *, field: str) -> EvidenceGap:
    _exact_keys(mapping, {"kind", "field", "reason"}, label=field)
    if mapping["kind"] != "gap" or mapping["field"] != field:
        _fail(f"{field} gap kind/field is noncanonical")
    try:
        return EvidenceGap(field=cast(str, mapping["field"]), reason=cast(str, mapping["reason"]))
    except (TypeError, ValueError) as exc:
        _fail(f"{field} gap is invalid", exc)


def _scalar(mapping: dict[str, object], *, field: str) -> ScalarReading:
    _exact_keys(mapping, {"kind", "value", "evidence_class", "method"}, label=field)
    if mapping["kind"] != "scalar":
        _fail(f"{field} scalar kind is noncanonical")
    try:
        return ScalarReading(
            value=cast(float, mapping["value"]),
            evidence_class=cast(
                EvidenceClass, _enum(EvidenceClass, mapping["evidence_class"], label=field)
            ),
            method=cast(ReadingMethod, _enum(ReadingMethod, mapping["method"], label=field)),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"{field} scalar is invalid", exc)


def _cost(mapping: dict[str, object], *, field: str) -> CostReading:
    _exact_keys(mapping, {"kind", "amount", "evidence_class", "method"}, label=field)
    if mapping["kind"] != "cost":
        _fail(f"{field} cost kind is noncanonical")
    money = _mapping(mapping["amount"], label=f"{field}.amount")
    _exact_keys(money, {"amount", "currency"}, label=f"{field}.amount")
    if type(money["amount"]) is not str:
        _fail(f"{field}.amount.amount must be a canonical decimal string")
    try:
        amount = Money(amount=Decimal(money["amount"]), currency=cast(str, money["currency"]))
        return CostReading(
            amount=amount,
            evidence_class=cast(
                EvidenceClass, _enum(EvidenceClass, mapping["evidence_class"], label=field)
            ),
            method=cast(ReadingMethod, _enum(ReadingMethod, mapping["method"], label=field)),
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        _fail(f"{field} cost is invalid or noncanonical", exc)


def _optional(
    value: object, *, field: str, cost: bool
) -> ScalarReading | CostReading | EvidenceGap:
    mapping = _mapping(value, label=field)
    kind = mapping.get("kind")
    if kind == "gap":
        return _gap(mapping, field=field)
    if cost:
        return _cost(mapping, field=field)
    return _scalar(mapping, field=field)


def _identity_or_gap(value: object, *, field: str) -> str | EvidenceGap:
    mapping = _mapping(value, label=field)
    if mapping.get("kind") == "gap":
        return _gap(mapping, field=field)
    _exact_keys(mapping, {"kind", "sha256"}, label=field)
    if mapping["kind"] != "identity" or type(mapping["sha256"]) is not str:
        _fail(f"{field} identity is noncanonical")
    return mapping["sha256"]


def _count_or_gap(value: object, *, field: str) -> int | EvidenceGap:
    mapping = _mapping(value, label=field)
    if mapping.get("kind") == "gap":
        return _gap(mapping, field=field)
    _exact_keys(mapping, {"kind", "value"}, label=field)
    if mapping["kind"] != "count" or type(mapping["value"]) is not int:
        _fail(f"{field} count is noncanonical")
    return mapping["value"]


def resource_snapshot_to_mapping(snapshot: ResourceWorkSnapshot) -> dict[str, object]:
    """Map one validated snapshot to the exact canonical wire object."""
    if type(snapshot) is not ResourceWorkSnapshot:
        _fail("resource snapshot codec requires ResourceWorkSnapshot")
    return {"event": _EVENT, **snapshot.manifest}


def resource_snapshot_from_mapping(value: object) -> ResourceWorkSnapshot:
    """Reconstruct one snapshot from a strict, exact canonical mapping."""
    mapping = _mapping(value, label="resource work snapshot")
    _exact_keys(mapping, _KEYS, label="resource work snapshot")
    if mapping["event"] != _EVENT:
        _fail("resource work snapshot event is unsupported")
    if mapping["required_counter_evidence"] != EvidenceClass.MEASURED.value:
        _fail("required_counter_evidence must be canonical measured")
    try:
        return ResourceWorkSnapshot(
            logical_run_id=cast(str, mapping["logical_run_id"]),
            execution_segment_index=cast(int, mapping["execution_segment_index"]),
            execution_segment_id=cast(str, mapping["execution_segment_id"]),
            training_step=cast(int, mapping["training_step"]),
            eval_index=cast(int, mapping["eval_index"]),
            observed_at=_timestamp(mapping["observed_at"]),
            trainer_wall_seconds=cast(float, mapping["trainer_wall_seconds"]),
            committed_quality_source_bytes=cast(int, mapping["committed_quality_source_bytes"]),
            committed_quality_model_tokens=cast(int, mapping["committed_quality_model_tokens"]),
            online_train_token_positions=cast(int, mapping["online_train_token_positions"]),
            target_train_token_positions=cast(int, mapping["target_train_token_positions"]),
            baseline_calibration_positions=cast(int, mapping["baseline_calibration_positions"]),
            eval_positions=cast(int, mapping["eval_positions"]),
            total_model_token_positions=cast(int, mapping["total_model_token_positions"]),
            gpu_train_device_seconds=cast(
                ScalarReading | EvidenceGap,
                _optional(
                    mapping["gpu_train_device_seconds"],
                    field="gpu_train_device_seconds",
                    cost=False,
                ),
            ),
            gpu_eval_device_seconds=cast(
                ScalarReading | EvidenceGap,
                _optional(
                    mapping["gpu_eval_device_seconds"], field="gpu_eval_device_seconds", cost=False
                ),
            ),
            peak_gpu_allocated_bytes=cast(
                ScalarReading | EvidenceGap,
                _optional(
                    mapping["peak_gpu_allocated_bytes"],
                    field="peak_gpu_allocated_bytes",
                    cost=False,
                ),
            ),
            peak_gpu_reserved_bytes=cast(
                ScalarReading | EvidenceGap,
                _optional(
                    mapping["peak_gpu_reserved_bytes"],
                    field="peak_gpu_reserved_bytes",
                    cost=False,
                ),
            ),
            cumulative_train_flops=cast(
                ScalarReading | EvidenceGap,
                _optional(
                    mapping["cumulative_train_flops"], field="cumulative_train_flops", cost=False
                ),
            ),
            cumulative_eval_flops=cast(
                ScalarReading | EvidenceGap,
                _optional(
                    mapping["cumulative_eval_flops"], field="cumulative_eval_flops", cost=False
                ),
            ),
            flop_model_identity=_identity_or_gap(
                mapping["flop_model_identity"], field="flop_model_identity"
            ),
            energy_joules=cast(
                ScalarReading | EvidenceGap,
                _optional(mapping["energy_joules"], field="energy_joules", cost=False),
            ),
            energy_sample_count=_count_or_gap(
                mapping["energy_sample_count"], field="energy_sample_count"
            ),
            rate_estimated_cost=cast(
                CostReading | EvidenceGap,
                _optional(mapping["rate_estimated_cost"], field="rate_estimated_cost", cost=True),
            ),
            schema_version=cast(str, mapping["schema_version"]),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"resource work snapshot is invalid: {exc}", exc)


def serialize_resource_snapshot_line(snapshot: ResourceWorkSnapshot) -> str:
    """Serialize one snapshot deterministically, without non-finite JSON constants."""
    return json.dumps(
        resource_snapshot_to_mapping(snapshot),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant {value!r} is forbidden")


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_resource_snapshot_line(line: str) -> ResourceWorkSnapshot:
    """Parse one nonblank canonical JSON line; this deliberately admits no journal semantics."""
    if type(line) is not str or not line.strip():
        _fail("resource snapshot JSON line must be nonblank")
    try:
        value = json.loads(line, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        _fail("resource snapshot JSON line is invalid JSON", exc)
    return resource_snapshot_from_mapping(value)
