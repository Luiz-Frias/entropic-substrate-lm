"""Resource ledger: attempts consume work; only commits advance quality exposure."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from entropic_substrate.evidence.resource_ledger import (
    ResourceLedger,
    ResourceMeterSnapshot,
    UnavailableMeterStrategy,
)
from entropic_substrate.evidence.resource_provenance import (
    BillingRoundingPolicy,
    BillingUnit,
    Money,
    PurchaseOption,
    QuotedRate,
    RateScope,
    RoundingMode,
    RunStartProvenance,
    SegmentStartMode,
    execution_segment_id,
)
from entropic_substrate.evidence.resource_work import CostReading, ReadingMethod, ScalarReading
from entropic_substrate.experiments.d1_primitives import EvidenceClass

pytestmark = pytest.mark.unit


def _start(segment: int = 0) -> RunStartProvenance:
    logical = "d1-both-seed-9"
    created = datetime(2026, 8, 23, 20, segment * 10, tzinfo=UTC)
    return RunStartProvenance(
        schema_version="resource-telemetry/v2",
        logical_run_id=logical,
        execution_segment_index=segment,
        execution_segment_id=execution_segment_id(logical, segment),
        segment_start_mode=SegmentStartMode.FRESH if segment == 0 else SegmentStartMode.RESUME,
        resume_checkpoint_sha256=None if segment == 0 else "b" * 64,
        accelerator_model="NVIDIA A100 PCIe 80GB",
        accelerator_count=1,
        provider="runpod",
        provider_sku="a100-pcie-80gb",
        region="US-TX-3",
        market="secure-cloud",
        purchase_option=PurchaseOption.ON_DEMAND,
        image_digest="sha256:" + "a" * 64,
        cuda_version="12.6",
        driver_version="560.35.03",
        pytorch_version="2.13.0",
        compiler_enabled=True,
        compiler_version="triton-3.4.0",
        quoted_rate=QuotedRate(
            amount=Money(amount=Decimal("3.19"), currency="USD"),
            billing_unit=BillingUnit.HOUR,
            rounding_policy=BillingRoundingPolicy(
                increment=1, mode=RoundingMode.CEILING, minimum_units=1
            ),
            rate_scope=RateScope.PER_ACCELERATOR,
            source="RunPod pricing API",
            quote_id=f"quote-{segment}",
            observed_at=created,
            evidence_class=EvidenceClass.MEASURED,
        ),
        pod_created_at=created,
        trainer_started_at=created + timedelta(minutes=1),
    )


def _meters(gpu_seconds: float) -> ResourceMeterSnapshot:
    gap = UnavailableMeterStrategy().snapshot()
    return ResourceMeterSnapshot(
        gpu_train_device_seconds=ScalarReading(
            value=gpu_seconds,
            evidence_class=EvidenceClass.MEASURED,
            method=ReadingMethod.GPU_DEVICE_COUNTER,
        ),
        gpu_eval_device_seconds=gap.gpu_eval_device_seconds,
        cumulative_train_flops=gap.cumulative_train_flops,
        cumulative_eval_flops=gap.cumulative_eval_flops,
        energy_joules=gap.energy_joules,
        rate_estimated_cost=CostReading(
            amount=Money(amount=Decimal("0.42"), currency="USD"),
            evidence_class=EvidenceClass.RATE_ESTIMATE,
            method=ReadingMethod.RATE_TIMES_SCOPE,
        ),
    )


class _SequenceMeter:
    def __init__(self, *snapshots: ResourceMeterSnapshot) -> None:
        self._snapshots = list(snapshots)

    def snapshot(self) -> ResourceMeterSnapshot:
        return self._snapshots.pop(0)


def test_failed_attempt_adds_total_work_but_not_committed_quality_then_eval_joins_start_identity() -> (
    None
):
    ledger = ResourceLedger(_start(), monotonic_origin_ns=100)
    ledger.record_baseline(target_positions=3)
    ledger.record_train_attempt(
        online_positions=10,
        target_positions=10,
        quality_source_byte_exposures=17,
        committed=False,
    )
    ledger.record_train_attempt(
        online_positions=10,
        target_positions=10,
        quality_source_byte_exposures=17,
        committed=True,
    )
    snapshot = ledger.record_eval(
        online_positions=4,
        target_positions=4,
        training_step=1,
        observed_at=_start().trainer_started_at,
        monotonic_ns=1_000_000_100,
    )
    assert snapshot.logical_run_id == _start().logical_run_id
    assert snapshot.committed_quality_source_bytes == 17
    assert snapshot.committed_quality_model_tokens == 10
    assert snapshot.online_train_token_positions == 20
    assert snapshot.target_train_token_positions == 20
    assert snapshot.baseline_calibration_positions == 3
    assert snapshot.eval_positions == 8
    assert snapshot.total_model_token_positions == 51
    assert snapshot.gpu_train_device_seconds.field == "gpu_train_device_seconds"


def test_ledger_requires_explicit_byte_exposure_and_monotonic_clock() -> None:
    ledger = ResourceLedger(_start(), monotonic_origin_ns=100)
    with pytest.raises(ValueError, match="quality_source_byte_exposures"):
        ledger.record_train_attempt(
            online_positions=2,
            target_positions=2,
            quality_source_byte_exposures=-1,
            committed=True,
        )
    with pytest.raises(ValueError, match="monotonic_ns"):
        ledger.record_eval(
            online_positions=1,
            target_positions=1,
            training_step=0,
            observed_at=_start().trainer_started_at,
            monotonic_ns=99,
        )


def test_meter_regression_is_rejected_before_snapshot_commit() -> None:
    meter = _SequenceMeter(_meters(10.0), _meters(9.0))
    ledger = ResourceLedger(_start(), monotonic_origin_ns=0, meter_strategy=meter)
    observed = _start().trainer_started_at
    ledger.record_eval(
        online_positions=1,
        target_positions=1,
        training_step=0,
        observed_at=observed,
        monotonic_ns=1,
    )
    with pytest.raises(ValueError, match="gpu_train_device_seconds"):
        ledger.record_eval(
            online_positions=1,
            target_positions=1,
            training_step=0,
            observed_at=observed + timedelta(seconds=1),
            monotonic_ns=2,
        )


def test_checkpoint_restore_carries_counters_optional_continuity_and_next_segment_identity() -> (
    None
):
    meter = _SequenceMeter(_meters(10.0), _meters(11.0))
    first = ResourceLedger(_start(), monotonic_origin_ns=0, meter_strategy=meter)
    observed = _start().trainer_started_at
    before = first.record_eval(
        online_positions=2,
        target_positions=2,
        training_step=0,
        observed_at=observed,
        monotonic_ns=10,
    )
    state = first.current_state()
    assert state.manifest["schema_version"] == "resource-ledger/v1"
    assert state.manifest["total_model_token_positions"] == 4
    assert state.manifest["last_optional_readings"]["gpu_train_device_seconds"]["value"] == 10.0
    restored = ResourceLedger.restore(
        _start(1),
        state,
        monotonic_origin_ns=100,
        meter_strategy=meter,
    )
    after = restored.record_eval(
        online_positions=2,
        target_positions=2,
        training_step=0,
        observed_at=_start(1).trainer_started_at,
        monotonic_ns=110,
    )
    assert after.eval_index == before.eval_index + 1
    assert after.execution_segment_index == 1
    assert after.eval_positions == before.eval_positions + 4
    assert after.trainer_wall_seconds == pytest.approx(before.trainer_wall_seconds + 1e-8)


def test_recovered_tail_work_advances_total_counters_without_quality_or_eval_index() -> None:
    first = ResourceLedger(_start(), monotonic_origin_ns=0)
    first.record_eval(
        online_positions=2,
        target_positions=2,
        training_step=0,
        observed_at=_start().trainer_started_at,
        monotonic_ns=10,
    )
    restored = ResourceLedger.restore(_start(1), first.current_state(), monotonic_origin_ns=100)

    restored.record_discarded_work(
        online_train_positions=4,
        target_train_positions=5,
        eval_positions=9,
    )

    state = restored.current_state()
    assert state.online_train_token_positions == 4
    assert state.target_train_token_positions == 5
    assert state.eval_positions == 13
    assert state.committed_quality_source_bytes == 0
    assert state.committed_quality_model_tokens == 0
    assert state.next_eval_index == 1
