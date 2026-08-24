"""Strict checkpoint-admission mapping codec for ResourceLedgerState."""

from __future__ import annotations

from decimal import Decimal

import pytest

from entropic_substrate.evidence.resource_ledger import ResourceLedgerState
from entropic_substrate.evidence.resource_ledger_codec import (
    ResourceLedgerCodecGap,
    parse_resource_ledger_state_line,
    resource_ledger_state_from_mapping,
    resource_ledger_state_to_mapping,
    serialize_resource_ledger_state_line,
)
from entropic_substrate.evidence.resource_provenance import Money
from entropic_substrate.evidence.resource_work import CostReading, ReadingMethod, ScalarReading
from entropic_substrate.experiments.d1_primitives import EvidenceClass

pytestmark = pytest.mark.unit


def _state() -> ResourceLedgerState:
    return ResourceLedgerState(
        logical_run_id="d4-coarse-seed-2",
        execution_segment_index=1,
        next_eval_index=7,
        last_training_step=100,
        trainer_wall_seconds=123.45678901234567,
        committed_quality_source_bytes=88_000,
        committed_quality_model_tokens=80_000,
        online_train_token_positions=100_000,
        target_train_token_positions=100_000,
        baseline_calibration_positions=512,
        eval_positions=4_096,
        last_optional_readings=(
            (
                "gpu_train_device_seconds",
                ScalarReading(
                    value=117.0,
                    evidence_class=EvidenceClass.MEASURED,
                    method=ReadingMethod.GPU_DEVICE_COUNTER,
                ),
            ),
            (
                "rate_estimated_cost",
                CostReading(
                    amount=Money(amount=Decimal("0.42"), currency="USD"),
                    evidence_class=EvidenceClass.RATE_ESTIMATE,
                    method=ReadingMethod.RATE_TIMES_SCOPE,
                ),
            ),
        ),
        last_flop_model_identity="f" * 64,
        last_energy_sample_count=3,
    )


def test_ledger_state_mapping_and_canonical_line_round_trip_exactly() -> None:
    state = _state()
    mapping = resource_ledger_state_to_mapping(state)
    assert mapping["event"] == "resource.ledger_state"
    assert mapping["execution_segment_id"] == "d4-coarse-seed-2:segment:1"
    readings = mapping["last_optional_readings"]
    assert isinstance(readings, dict)
    cost = readings["rate_estimated_cost"]
    assert isinstance(cost, dict)
    amount = cost["amount"]
    assert isinstance(amount, dict)
    assert amount["amount"] == "0.42"
    assert resource_ledger_state_from_mapping(mapping) == state
    line = serialize_resource_ledger_state_line(state)
    assert parse_resource_ledger_state_line(line) == state
    assert serialize_resource_ledger_state_line(parse_resource_ledger_state_line(line)) == line


@pytest.mark.parametrize("mutation", ["unknown", "missing", "bool", "total", "segment"])
def test_checkpoint_mapping_rejects_exact_schema_and_counter_mutations(mutation: str) -> None:
    mapping = resource_ledger_state_to_mapping(_state())
    if mutation == "unknown":
        mapping["invented"] = 0
    elif mutation == "missing":
        del mapping["next_eval_index"]
    elif mutation == "bool":
        mapping["committed_quality_source_bytes"] = True
    elif mutation == "total":
        mapping["total_model_token_positions"] = 0
    else:
        mapping["execution_segment_id"] = "wrong"
    with pytest.raises(ResourceLedgerCodecGap):
        resource_ledger_state_from_mapping(mapping)


def test_checkpoint_codec_rejects_unregistered_or_wrong_variant_last_readings_and_decimal_aliases() -> (
    None
):
    mapping = resource_ledger_state_to_mapping(_state())
    readings = mapping["last_optional_readings"]
    assert isinstance(readings, dict)
    readings["unregistered_meter"] = readings["gpu_train_device_seconds"]
    with pytest.raises(ResourceLedgerCodecGap, match="registered"):
        resource_ledger_state_from_mapping(mapping)
    mapping = resource_ledger_state_to_mapping(_state())
    readings = mapping["last_optional_readings"]
    assert isinstance(readings, dict)
    cost = readings["rate_estimated_cost"]
    assert isinstance(cost, dict)
    amount = cost["amount"]
    assert isinstance(amount, dict)
    amount["amount"] = "0.420"
    with pytest.raises(ResourceLedgerCodecGap, match="canonical"):
        resource_ledger_state_from_mapping(mapping)
    mapping = resource_ledger_state_to_mapping(_state())
    readings = mapping["last_optional_readings"]
    assert isinstance(readings, dict)
    readings["rate_estimated_cost"] = readings["gpu_train_device_seconds"]
    with pytest.raises(ResourceLedgerCodecGap, match="rate_estimated_cost"):
        resource_ledger_state_from_mapping(mapping)


@pytest.mark.parametrize(
    "line",
    [
        '{"event":"resource.ledger_state","trainer_wall_seconds":NaN}',
        '{"event":"resource.ledger_state","trainer_wall_seconds":Infinity}',
        '{"event":"a","event":"b"}',
        "[]",
        "",
    ],
)
def test_checkpoint_line_parser_rejects_nonfinite_duplicate_and_nonobject_values(line: str) -> None:
    with pytest.raises(ResourceLedgerCodecGap):
        parse_resource_ledger_state_line(line)
