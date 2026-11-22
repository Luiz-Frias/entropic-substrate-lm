"""Strict one-snapshot JSON codec for D1-D5 resource-work evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    EvidenceGap,
    Money,
    execution_segment_id,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.ladder.capture.resource.resource_work import (
    CostReading,
    ReadingMethod,
    ResourceWorkSnapshot,
    ScalarReading,
)
from entropic_substrate.ladder.capture.resource.resource_work_codec import (
    ResourceWorkCodecGap,
    parse_resource_snapshot_line,
    resource_snapshot_from_mapping,
    resource_snapshot_to_mapping,
    serialize_resource_snapshot_line,
)

pytestmark = pytest.mark.unit


def _snapshot() -> ResourceWorkSnapshot:
    logical_run_id = "d3-horizons-seed-5"
    return ResourceWorkSnapshot(
        logical_run_id=logical_run_id,
        execution_segment_index=0,
        execution_segment_id=execution_segment_id(logical_run_id, 0),
        training_step=42,
        eval_index=3,
        observed_at=datetime(2026, 8, 23, 19, 1, 2, 345678, tzinfo=UTC),
        trainer_wall_seconds=123.45678901234567,
        committed_quality_source_bytes=344_064,
        committed_quality_model_tokens=344_064,
        online_train_token_positions=344_064,
        target_train_token_positions=344_064,
        baseline_calibration_positions=512,
        eval_positions=1_024,
        total_model_token_positions=689_664,
        gpu_train_device_seconds=ScalarReading(
            value=120.0,
            evidence_class=EvidenceClass.MEASURED,
            method=ReadingMethod.GPU_DEVICE_COUNTER,
        ),
        gpu_eval_device_seconds=EvidenceGap(
            field="gpu_eval_device_seconds", reason="device collector unavailable"
        ),
        peak_gpu_allocated_bytes=ScalarReading(
            value=8_589_934_592.0,
            evidence_class=EvidenceClass.MEASURED,
            method=ReadingMethod.GPU_DEVICE_COUNTER,
        ),
        peak_gpu_reserved_bytes=EvidenceGap(
            field="peak_gpu_reserved_bytes", reason="device collector unavailable"
        ),
        cumulative_train_flops=ScalarReading(
            value=1.2345678901234568e15,
            evidence_class=EvidenceClass.MODEL_ESTIMATE,
            method=ReadingMethod.FLOP_MODEL,
        ),
        cumulative_eval_flops=EvidenceGap(
            field="cumulative_eval_flops", reason="profiler unavailable"
        ),
        flop_model_identity="f" * 64,
        energy_joules=ScalarReading(
            value=1.2e6,
            evidence_class=EvidenceClass.DERIVED_FROM_MEASURED,
            method=ReadingMethod.ENERGY_INTEGRATED,
        ),
        energy_sample_count=3,
        rate_estimated_cost=CostReading(
            amount=Money(amount=Decimal("0.42"), currency="USD"),
            evidence_class=EvidenceClass.RATE_ESTIMATE,
            method=ReadingMethod.RATE_TIMES_SCOPE,
        ),
    )


def test_snapshot_mapping_and_json_line_round_trip_all_reading_and_gap_variants() -> None:
    snapshot = _snapshot()
    mapping = resource_snapshot_to_mapping(snapshot)
    assert mapping["event"] == "resource.work_snapshot"
    assert mapping["observed_at"] == "2026-08-23T19:01:02.345678Z"
    assert mapping["rate_estimated_cost"]["amount"]["amount"] == "0.42"
    assert resource_snapshot_from_mapping(mapping) == snapshot
    line = serialize_resource_snapshot_line(snapshot)
    assert parse_resource_snapshot_line(line) == snapshot
    assert serialize_resource_snapshot_line(parse_resource_snapshot_line(line)) == line


@pytest.mark.parametrize("mutation", ["unknown", "missing", "bool", "bad-schema", "bad-timestamp"])
def test_mapping_codec_refuses_exact_schema_mutations(mutation: str) -> None:
    mapping = resource_snapshot_to_mapping(_snapshot())
    if mutation == "unknown":
        mapping["unregistered"] = 0
    elif mutation == "missing":
        del mapping["training_step"]
    elif mutation == "bool":
        mapping["trainer_wall_seconds"] = True
    elif mutation == "bad-schema":
        mapping["schema_version"] = "resource-work/v0"
    else:
        mapping["observed_at"] = "2026-08-23T19:01:02+00:00"
    with pytest.raises(ResourceWorkCodecGap):
        resource_snapshot_from_mapping(mapping)


def test_codec_refuses_noncanonical_decimal_and_partial_or_wrong_optional_variant() -> None:
    mapping = resource_snapshot_to_mapping(_snapshot())
    cost = mapping["rate_estimated_cost"]
    assert isinstance(cost, dict)
    amount = cost["amount"]
    assert isinstance(amount, dict)
    amount["amount"] = "0.420"
    with pytest.raises(ResourceWorkCodecGap, match="canonical"):
        resource_snapshot_from_mapping(mapping)
    mapping = resource_snapshot_to_mapping(_snapshot())
    mapping["gpu_eval_device_seconds"] = {
        "kind": "scalar",
        "value": 1.0,
        "evidence_class": EvidenceClass.MEASURED.value,
    }
    with pytest.raises(ResourceWorkCodecGap, match="keys"):
        resource_snapshot_from_mapping(mapping)


@pytest.mark.parametrize(
    "line",
    [
        '{"event":"resource.work_snapshot","trainer_wall_seconds":NaN}',
        '{"event":"resource.work_snapshot","trainer_wall_seconds":Infinity}',
        '{"event":"a","event":"b"}',
        "[]",
        "",
    ],
)
def test_json_line_codec_refuses_duplicates_nonfinite_constants_and_nonobjects(line: str) -> None:
    with pytest.raises(ResourceWorkCodecGap):
        parse_resource_snapshot_line(line)
