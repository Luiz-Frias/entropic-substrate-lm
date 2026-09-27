"""D1 resource-ledger v2 preserves optional meter identity across resume."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    RESOURCE_TELEMETRY_SCHEMA_VERSION,
    BillingRoundingPolicy,
    BillingUnit,
    EvidenceGap,
    Money,
    PurchaseOption,
    QuotedRate,
    RateScope,
    RoundingMode,
    RunStartProvenance,
    SegmentStartMode,
    execution_segment_id,
    unavailable_resource_activation_registration,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.ladder.capture.resource.resource_ledger import (
    ResourceLedger,
    ResourceMeterSnapshot,
)
from entropic_substrate.ladder.capture.resource.resource_ledger_codec import (
    resource_ledger_state_from_mapping,
    resource_ledger_state_to_mapping,
)
from entropic_substrate.ladder.capture.resource.resource_work import (
    CostReading,
    ReadingMethod,
    ScalarReading,
)

pytestmark = pytest.mark.unit


def _start(segment: int = 0) -> RunStartProvenance:
    logical = "d1-sg-seed-3"
    observed = datetime(2026, 8, 24, 12, segment, tzinfo=UTC)
    return RunStartProvenance(
        schema_version=RESOURCE_TELEMETRY_SCHEMA_VERSION,
        logical_run_id=logical,
        execution_segment_index=segment,
        execution_segment_id=execution_segment_id(logical, segment),
        segment_start_mode=SegmentStartMode.FRESH if segment == 0 else SegmentStartMode.RESUME,
        resume_checkpoint_sha256=None if segment == 0 else "f" * 64,
        accelerator_model="NVIDIA A100 80GB PCIe",
        accelerator_count=1,
        provider="runpod",
        provider_sku="a100-pcie-80gb",
        region="US-TX-3",
        market="secure-cloud",
        purchase_option=PurchaseOption.ON_DEMAND,
        image_digest="sha256:" + "e" * 64,
        cuda_version="12.6",
        driver_version="560.35.03",
        pytorch_version="2.13.0",
        compiler_enabled=False,
        compiler_version="disabled",
        resource_activation=unavailable_resource_activation_registration(),
        quoted_rate=QuotedRate(
            Money(Decimal("1.19"), "USD"),
            BillingUnit.HOUR,
            BillingRoundingPolicy(1, RoundingMode.CEILING, 1),
            RateScope.PER_ACCELERATOR,
            "RunPod pricing API",
            f"quote-{segment}",
            observed,
            EvidenceClass.MEASURED,
        ),
        pod_created_at=observed,
        trainer_started_at=observed,
    )


def _reading(value: float) -> ScalarReading:
    return ScalarReading(
        value=value,
        evidence_class=EvidenceClass.MEASURED,
        method=ReadingMethod.GPU_DEVICE_COUNTER,
    )


def _meters(
    *, identity: str = "a" * 64, energy_count: int = 3, gpu_seconds: float = 1.0
) -> ResourceMeterSnapshot:
    return ResourceMeterSnapshot(
        gpu_train_device_seconds=_reading(gpu_seconds),
        gpu_eval_device_seconds=_reading(gpu_seconds / 2.0),
        peak_gpu_allocated_bytes=_reading(1024.0),
        peak_gpu_reserved_bytes=_reading(2048.0),
        cumulative_train_flops=ScalarReading(
            100.0, EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL
        ),
        cumulative_eval_flops=ScalarReading(
            20.0, EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL
        ),
        flop_model_identity=identity,
        energy_joules=ScalarReading(
            10.0,
            EvidenceClass.DERIVED_FROM_MEASURED,
            ReadingMethod.ENERGY_INTEGRATED,
        ),
        energy_sample_count=energy_count,
        rate_estimated_cost=CostReading(
            Money(Decimal("0.01"), "USD"),
            EvidenceClass.RATE_ESTIMATE,
            ReadingMethod.RATE_TIMES_SCOPE,
        ),
    )


class _Meter:
    def __init__(self, *values: ResourceMeterSnapshot) -> None:
        self._values = list(values)
        self.wall_reads: list[float] = []

    def snapshot(self, *, trainer_wall_seconds: float) -> ResourceMeterSnapshot:
        self.wall_reads.append(trainer_wall_seconds)
        return self._values.pop(0)


def _record(ledger: ResourceLedger, *, clock_ns: int, step: int = 0) -> None:
    ledger.record_eval(
        online_positions=1,
        target_positions=1,
        training_step=step,
        observed_at=_start().trainer_started_at + timedelta(seconds=clock_ns / 1e9),
        monotonic_ns=clock_ns,
    )


def test_v2_state_codec_preserves_identity_count_and_vram_continuity() -> None:
    meter = _Meter(_meters())
    ledger = ResourceLedger(_start(), monotonic_origin_ns=0, meter_strategy=meter)
    _record(ledger, clock_ns=2_000_000_000)
    state = ledger.current_state()
    mapping = resource_ledger_state_to_mapping(state)

    assert meter.wall_reads == [2.0]
    assert mapping["schema_version"] == "resource-ledger/v2"
    assert mapping["last_flop_model_identity"] == "a" * 64
    assert mapping["last_energy_sample_count"] == 3
    assert resource_ledger_state_from_mapping(mapping) == state


@pytest.mark.parametrize(
    ("meters", "message"),
    [
        (_meters(identity="b" * 64, gpu_seconds=2.0), "flop_model_identity"),
        (_meters(energy_count=2, gpu_seconds=2.0), "energy_sample_count"),
    ],
)
def test_resume_refuses_methodology_or_sample_count_regression(
    meters: ResourceMeterSnapshot, message: str
) -> None:
    first = ResourceLedger(_start(), monotonic_origin_ns=0, meter_strategy=_Meter(_meters()))
    _record(first, clock_ns=1)
    restored = ResourceLedger.restore(
        _start(1),
        first.current_state(),
        monotonic_origin_ns=10,
        meter_strategy=_Meter(meters),
    )

    with pytest.raises(ValueError, match=message):
        _record(restored, clock_ns=11)


def test_unavailable_v2_channels_are_gaps_not_zeroes() -> None:
    snapshot = ResourceLedger(_start(), monotonic_origin_ns=0).record_eval(
        online_positions=1,
        target_positions=1,
        training_step=0,
        observed_at=_start().trainer_started_at,
        monotonic_ns=0,
    )

    for field in (
        "gpu_train_device_seconds",
        "peak_gpu_allocated_bytes",
        "flop_model_identity",
        "energy_sample_count",
    ):
        value = getattr(snapshot, field)
        assert type(value) is EvidenceGap
        assert value.field == field
