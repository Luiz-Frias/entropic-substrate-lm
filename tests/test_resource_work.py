"""Per-eval resource work snapshots preserve counter semantics and evidence gaps."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
    validate_resource_work_series,
)

pytestmark = pytest.mark.unit


def _scalar(value: float, evidence: EvidenceClass, method: ReadingMethod) -> ScalarReading:
    return ScalarReading(value=value, evidence_class=evidence, method=method)


def _snapshot(**overrides: object) -> ResourceWorkSnapshot:
    logical_run_id = "d2-pullback-seed-3"
    payload: dict[str, object] = {
        "logical_run_id": logical_run_id,
        "execution_segment_index": 0,
        "execution_segment_id": execution_segment_id(logical_run_id, 0),
        "training_step": 100,
        "eval_index": 0,
        "observed_at": datetime(2026, 8, 23, 18, 0, tzinfo=UTC),
        "trainer_wall_seconds": 300.0,
        "committed_quality_source_bytes": 819_200,
        "committed_quality_model_tokens": 819_200,
        "online_train_token_positions": 819_200,
        "target_train_token_positions": 819_200,
        "baseline_calibration_positions": 1_024,
        "eval_positions": 2_048,
        "total_model_token_positions": 1_641_472,
        "gpu_train_device_seconds": _scalar(
            290.0, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
        ),
        "gpu_eval_device_seconds": _scalar(
            8.0, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
        ),
        "peak_gpu_allocated_bytes": _scalar(
            8_589_934_592.0,
            EvidenceClass.MEASURED,
            ReadingMethod.GPU_DEVICE_COUNTER,
        ),
        "peak_gpu_reserved_bytes": EvidenceGap(
            field="peak_gpu_reserved_bytes", reason="device counter unavailable"
        ),
        "cumulative_train_flops": _scalar(
            1.2e15, EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL
        ),
        "cumulative_eval_flops": EvidenceGap(
            field="cumulative_eval_flops", reason="profiler unavailable"
        ),
        "flop_model_identity": "f" * 64,
        "energy_joules": _scalar(
            2.3e6, EvidenceClass.DERIVED_FROM_MEASURED, ReadingMethod.ENERGY_INTEGRATED
        ),
        "energy_sample_count": 3,
        "rate_estimated_cost": CostReading(
            amount=Money(amount=Decimal("0.42"), currency="USD"),
            evidence_class=EvidenceClass.RATE_ESTIMATE,
            method=ReadingMethod.RATE_TIMES_SCOPE,
        ),
    }
    payload.update(overrides)
    return ResourceWorkSnapshot(**payload)  # type: ignore[arg-type]


def test_snapshot_distinguishes_committed_quality_exposure_from_full_work_breakdown() -> None:
    snapshot = _snapshot()
    assert snapshot.committed_quality_model_tokens == 819_200
    assert snapshot.total_model_token_positions == (
        snapshot.online_train_token_positions
        + snapshot.target_train_token_positions
        + snapshot.baseline_calibration_positions
        + snapshot.eval_positions
    )
    assert snapshot.manifest["schema_version"] == "resource-work/v2"
    assert snapshot.manifest["observed_at"] == "2026-08-23T18:00:00Z"
    assert snapshot.manifest["baseline_calibration_positions"] == 1_024
    assert snapshot.manifest["required_counter_evidence"] == EvidenceClass.MEASURED.value
    assert snapshot.manifest["cumulative_eval_flops"]["kind"] == "gap"
    changed = replace(
        snapshot,
        committed_quality_source_bytes=snapshot.committed_quality_source_bytes + 1,
    )
    assert changed.cursor_identity != snapshot.cursor_identity


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_model_token_positions", 1, "sum"),
        ("trainer_wall_seconds", -1.0, "trainer_wall_seconds"),
        ("committed_quality_source_bytes", -1, "committed_quality_source_bytes"),
        (
            "gpu_train_device_seconds",
            _scalar(1.0, EvidenceClass.MODEL_ESTIMATE, ReadingMethod.GPU_DEVICE_COUNTER),
            "gpu_train_device_seconds",
        ),
        (
            "energy_joules",
            _scalar(1.0, EvidenceClass.MEASURED, ReadingMethod.ENERGY_INTEGRATED),
            "energy_joules",
        ),
    ],
)
def test_snapshot_rejects_invalid_work_or_evidence_contracts(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _snapshot(**{field: value})


def test_rate_cost_must_be_a_rate_estimate_with_its_declared_method() -> None:
    with pytest.raises(ValueError, match="RATE_ESTIMATE"):
        CostReading(
            amount=Money(amount=Decimal("1"), currency="USD"),
            evidence_class=EvidenceClass.MEASURED,
            method=ReadingMethod.RATE_TIMES_SCOPE,
        )


def test_optional_gap_names_are_field_bound_and_cannot_hide_present_measurements() -> None:
    with pytest.raises(ValueError, match="gpu_eval_device_seconds"):
        _snapshot(
            gpu_eval_device_seconds=EvidenceGap(field="energy_joules", reason="not collected")
        )


def test_series_keeps_required_counters_monotonic_but_allows_total_work_without_new_step() -> None:
    first = _snapshot()
    later = replace(
        first,
        eval_index=1,
        observed_at=first.observed_at + timedelta(minutes=5),
        trainer_wall_seconds=330.0,
        baseline_calibration_positions=2_048,
        eval_positions=4_096,
        total_model_token_positions=1_644_544,
        gpu_train_device_seconds=EvidenceGap(
            field="gpu_train_device_seconds", reason="collector gap"
        ),
        cumulative_train_flops=EvidenceGap(field="cumulative_train_flops", reason="profiler gap"),
    )
    validate_resource_work_series((first, later))
    with pytest.raises(ValueError, match="eval_index"):
        validate_resource_work_series((first, replace(later, eval_index=0)))
    with pytest.raises(ValueError, match="committed_quality_model_tokens"):
        validate_resource_work_series(
            (
                first,
                replace(
                    later, committed_quality_model_tokens=first.committed_quality_model_tokens - 1
                ),
            )
        )


def test_optional_last_nonmissing_readings_remain_monotonic_and_evidence_continuous_across_gaps() -> (
    None
):
    first = _snapshot()
    gap = replace(
        first,
        eval_index=1,
        observed_at=first.observed_at + timedelta(minutes=5),
        gpu_train_device_seconds=EvidenceGap(field="gpu_train_device_seconds", reason="gap"),
    )
    regressed = replace(
        first,
        eval_index=2,
        observed_at=first.observed_at + timedelta(minutes=10),
        gpu_train_device_seconds=_scalar(
            280.0, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
        ),
    )
    with pytest.raises(ValueError, match="gpu_train_device_seconds"):
        validate_resource_work_series((first, gap, regressed))
    changed_method = replace(
        first,
        eval_index=1,
        observed_at=first.observed_at + timedelta(minutes=5),
        cumulative_train_flops=_scalar(1.3e15, EvidenceClass.MEASURED, ReadingMethod.FLOP_PROFILER),
    )
    with pytest.raises(ValueError, match="cumulative_train_flops evidence"):
        validate_resource_work_series((first, changed_method))


def test_segment_transitions_are_consecutive_and_cost_currency_is_continuous() -> None:
    first = _snapshot()
    skipped = replace(
        first,
        execution_segment_index=2,
        execution_segment_id=execution_segment_id(first.logical_run_id, 2),
        eval_index=1,
        observed_at=first.observed_at + timedelta(minutes=5),
    )
    with pytest.raises(ValueError, match="segment"):
        validate_resource_work_series((first, skipped))
    currency_changed = replace(
        first,
        eval_index=1,
        observed_at=first.observed_at + timedelta(minutes=5),
        rate_estimated_cost=CostReading(
            amount=Money(amount=Decimal("0.5"), currency="EUR"),
            evidence_class=EvidenceClass.RATE_ESTIMATE,
            method=ReadingMethod.RATE_TIMES_SCOPE,
        ),
    )
    with pytest.raises(ValueError, match="currency"):
        validate_resource_work_series((first, currency_changed))


@pytest.mark.parametrize(
    "first",
    [
        _snapshot(eval_index=1),
        _snapshot(
            execution_segment_index=1,
            execution_segment_id=execution_segment_id("d2-pullback-seed-3", 1),
        ),
    ],
)
def test_complete_resource_series_starts_at_eval_and_segment_zero(
    first: ResourceWorkSnapshot,
) -> None:
    with pytest.raises(ValueError, match="start"):
        validate_resource_work_series((first,))
