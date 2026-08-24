"""D1 resource-work v2 binds optional meter values to their evidence metadata."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from entropic_substrate.evidence.resource_provenance import EvidenceGap, Money
from entropic_substrate.evidence.resource_work import (
    CostReading,
    ReadingMethod,
    ResourceWorkSnapshot,
    ScalarReading,
)
from entropic_substrate.evidence.resource_work_codec import (
    resource_snapshot_from_mapping,
    resource_snapshot_to_mapping,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass

pytestmark = pytest.mark.unit


def _scalar(value: float, evidence: EvidenceClass, method: ReadingMethod) -> ScalarReading:
    return ScalarReading(value=value, evidence_class=evidence, method=method)


def _snapshot(**overrides: object) -> ResourceWorkSnapshot:
    values: dict[str, object] = {
        "logical_run_id": "d1-both-seed-7",
        "execution_segment_index": 0,
        "execution_segment_id": "d1-both-seed-7:segment:0",
        "training_step": 50,
        "eval_index": 1,
        "observed_at": datetime(2026, 8, 24, 12, tzinfo=UTC),
        "trainer_wall_seconds": 15.25,
        "committed_quality_source_bytes": 100,
        "committed_quality_model_tokens": 100,
        "online_train_token_positions": 100,
        "target_train_token_positions": 101,
        "baseline_calibration_positions": 10,
        "eval_positions": 20,
        "total_model_token_positions": 231,
        "gpu_train_device_seconds": _scalar(
            12.0, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
        ),
        "gpu_eval_device_seconds": _scalar(
            1.0, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
        ),
        "peak_gpu_allocated_bytes": _scalar(
            12_345.0, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
        ),
        "peak_gpu_reserved_bytes": _scalar(
            16_384.0, EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER
        ),
        "cumulative_train_flops": _scalar(
            2_000.0, EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL
        ),
        "cumulative_eval_flops": _scalar(
            500.0, EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL
        ),
        "flop_model_identity": "a" * 64,
        "energy_joules": _scalar(
            42.0, EvidenceClass.DERIVED_FROM_MEASURED, ReadingMethod.ENERGY_INTEGRATED
        ),
        "energy_sample_count": 3,
        "rate_estimated_cost": CostReading(
            amount=Money(Decimal("0.125"), "USD"),
            evidence_class=EvidenceClass.RATE_ESTIMATE,
            method=ReadingMethod.RATE_TIMES_SCOPE,
        ),
        "schema_version": "resource-work/v2",
    }
    values.update(overrides)
    return ResourceWorkSnapshot(**values)  # type: ignore[arg-type]


def test_v2_round_trip_keeps_vram_flop_identity_and_energy_sample_count() -> None:
    snapshot = _snapshot()
    mapping = resource_snapshot_to_mapping(snapshot)

    assert mapping["schema_version"] == "resource-work/v2"
    assert mapping["peak_gpu_allocated_bytes"]["value"] == 12_345.0  # type: ignore[index]
    assert mapping["flop_model_identity"] == {"kind": "identity", "sha256": "a" * 64}
    assert mapping["energy_sample_count"] == {"kind": "count", "value": 3}
    assert resource_snapshot_from_mapping(mapping) == snapshot


def test_model_estimated_flops_require_a_canonical_methodology_identity() -> None:
    with pytest.raises(ValueError, match="flop_model_identity"):
        _snapshot(
            flop_model_identity=EvidenceGap("flop_model_identity", "model methodology unavailable")
        )
    with pytest.raises(ValueError, match="flop_model_identity"):
        _snapshot(flop_model_identity="not-a-sha256")


def test_integrated_energy_requires_at_least_two_timestamped_samples() -> None:
    with pytest.raises(ValueError, match="energy_sample_count"):
        _snapshot(energy_sample_count=1)
    with pytest.raises(ValueError, match="energy_sample_count"):
        _snapshot(
            energy_sample_count=EvidenceGap("energy_sample_count", "sample counter unavailable")
        )

    no_claim = replace(
        _snapshot(),
        energy_joules=EvidenceGap("energy_joules", "only one power sample"),
        energy_sample_count=1,
    )
    assert no_claim.manifest["energy_joules"]["kind"] == "gap"  # type: ignore[index]


@pytest.mark.parametrize("field", ["peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"])
def test_vram_peaks_cannot_be_model_estimates(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _snapshot(
            **{
                field: _scalar(
                    1.0,
                    EvidenceClass.MODEL_ESTIMATE,
                    ReadingMethod.GPU_DEVICE_COUNTER,
                )
            }
        )


def test_unsupported_channels_remain_field_bound_gaps() -> None:
    snapshot = _snapshot(
        peak_gpu_allocated_bytes=EvidenceGap(
            "peak_gpu_allocated_bytes", "CUDA memory API unavailable"
        ),
        peak_gpu_reserved_bytes=EvidenceGap(
            "peak_gpu_reserved_bytes", "CUDA memory API unavailable"
        ),
        cumulative_train_flops=EvidenceGap("cumulative_train_flops", "FLOP model not configured"),
        cumulative_eval_flops=EvidenceGap("cumulative_eval_flops", "FLOP model not configured"),
        flop_model_identity=EvidenceGap("flop_model_identity", "FLOP model not configured"),
        energy_joules=EvidenceGap("energy_joules", "power samples unavailable"),
        energy_sample_count=EvidenceGap("energy_sample_count", "power samples unavailable"),
    )

    assert snapshot.manifest["peak_gpu_allocated_bytes"]["kind"] == "gap"  # type: ignore[index]
    assert snapshot.manifest["energy_sample_count"]["kind"] == "gap"  # type: ignore[index]
