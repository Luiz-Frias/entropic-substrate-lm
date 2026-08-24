"""D1-D5 execution-segment provenance: exact, complete, and resume-safe."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from entropic_substrate.evidence.resource_provenance import (
    RESOURCE_TELEMETRY_SCHEMA_VERSION,
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
    unavailable_resource_activation_registration,
    validate_execution_segments,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass

pytestmark = pytest.mark.unit


def _rate(**overrides: object) -> QuotedRate:
    payload: dict[str, object] = {
        "amount": Money(amount=Decimal("3.19"), currency="USD"),
        "billing_unit": BillingUnit.HOUR,
        "rounding_policy": BillingRoundingPolicy(
            increment=1,
            mode=RoundingMode.CEILING,
            minimum_units=1,
        ),
        "rate_scope": RateScope.PER_ACCELERATOR,
        "source": "RunPod pricing API",
        "quote_id": "runpod-price-a100-pcie-us-tx-3-20260823",
        "observed_at": datetime(2026, 8, 23, 13, 0, tzinfo=UTC),
        "evidence_class": EvidenceClass.MEASURED,
    }
    payload.update(overrides)
    return QuotedRate(**payload)  # type: ignore[arg-type]


def _start(**overrides: object) -> RunStartProvenance:
    logical_run_id = "d1-confirmatory-both-seed-7"
    payload: dict[str, object] = {
        "schema_version": RESOURCE_TELEMETRY_SCHEMA_VERSION,
        "logical_run_id": logical_run_id,
        "execution_segment_index": 0,
        "execution_segment_id": execution_segment_id(logical_run_id, 0),
        "segment_start_mode": SegmentStartMode.FRESH,
        "resume_checkpoint_sha256": None,
        "accelerator_model": "NVIDIA A100 PCIe 80GB",
        "accelerator_count": 2,
        "provider": "runpod",
        "provider_sku": "a100-pcie-80gb",
        "region": "US-TX-3",
        "market": "secure-cloud",
        "purchase_option": PurchaseOption.ON_DEMAND,
        "image_digest": "sha256:" + "a" * 64,
        "cuda_version": "12.6",
        "driver_version": "560.35.03",
        "pytorch_version": "2.13.0",
        "compiler_enabled": True,
        "compiler_version": "triton-3.4.0",
        "resource_activation": unavailable_resource_activation_registration(),
        "quoted_rate": _rate(),
        "pod_created_at": datetime(2026, 8, 23, 13, 0, tzinfo=UTC),
        "trainer_started_at": datetime(2026, 8, 23, 13, 4, tzinfo=UTC),
    }
    payload.update(overrides)
    return RunStartProvenance(**payload)  # type: ignore[arg-type]


def test_fresh_segment_carries_complete_compute_and_exact_observed_rate_provenance() -> None:
    start = _start()
    assert start.execution_segment_id == "d1-confirmatory-both-seed-7:segment:0"
    assert start.resume_checkpoint_sha256 is None
    assert start.quoted_rate.amount.amount == Decimal("3.19")
    assert start.quoted_rate.amount.canonical_amount == "3.19"
    assert start.quoted_rate.evidence_class is EvidenceClass.MEASURED
    assert start.resource_activation == unavailable_resource_activation_registration()
    assert "evidence_gaps" not in start.manifest
    assert start.identity_digest == start.identity_digest


def test_resume_segments_require_the_exact_next_index_id_and_checkpoint_digest() -> None:
    first = _start()
    resumed = _start(
        execution_segment_index=1,
        execution_segment_id=execution_segment_id(first.logical_run_id, 1),
        segment_start_mode=SegmentStartMode.RESUME,
        resume_checkpoint_sha256="b" * 64,
        pod_created_at=first.trainer_started_at + timedelta(minutes=20),
        trainer_started_at=first.trainer_started_at + timedelta(minutes=24),
    )
    validate_execution_segments((first, resumed))
    with pytest.raises(ValueError, match="resource activation"):
        validate_execution_segments(
            (
                first,
                replace(
                    resumed,
                    resource_activation=replace(
                        resumed.resource_activation,
                        factory_registration_digest="c" * 64,
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="consecutive"):
        validate_execution_segments(
            (
                first,
                _start(
                    execution_segment_index=2,
                    execution_segment_id=execution_segment_id(first.logical_run_id, 2),
                    segment_start_mode=SegmentStartMode.RESUME,
                    resume_checkpoint_sha256="b" * 64,
                ),
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution_segment_id", "not-the-derived-id", "execution_segment_id"),
        ("segment_start_mode", SegmentStartMode.RESUME, "segment 0"),
        ("image_digest", "sha256:" + "A" * 64, "lowercase"),
        ("accelerator_count", 0, "accelerator_count"),
        ("market", "", "market"),
        ("compiler_enabled", False, "compiler_version"),
    ],
)
def test_start_provenance_refuses_incomplete_or_misidentified_core_facts(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _start(**{field: value})


def test_resume_mode_and_checkpoint_digest_are_mutually_load_bearing() -> None:
    with pytest.raises(ValueError, match="resume_checkpoint_sha256"):
        _start(
            execution_segment_index=1,
            execution_segment_id=execution_segment_id("d1-confirmatory-both-seed-7", 1),
            segment_start_mode=SegmentStartMode.RESUME,
            resume_checkpoint_sha256=None,
        )
    with pytest.raises(ValueError, match="fresh segment"):
        _start(resume_checkpoint_sha256="b" * 64)


def test_quote_is_measured_observation_with_exact_decimal_and_structured_rounding() -> None:
    with pytest.raises((TypeError, ValueError), match="Decimal"):
        Money(amount=3.19, currency="USD")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical"):
        Money(amount=Decimal("3.190"), currency="USD")
    tiny = Money(amount=Decimal("4.4E-13"), currency="USD")
    assert tiny.canonical_amount == "0.00000000000044"
    with pytest.raises(ValueError, match="evidence_class"):
        _rate(evidence_class=EvidenceClass.RATE_ESTIMATE)
    with pytest.raises(ValueError, match="increment"):
        _rate(
            rounding_policy=BillingRoundingPolicy(
                increment=0, mode=RoundingMode.CEILING, minimum_units=1
            )
        )


def test_quote_must_precede_the_trainer_start_and_timestamps_must_be_utc_ordered() -> None:
    with pytest.raises(ValueError, match="quote observed_at"):
        _start(quoted_rate=_rate(observed_at=datetime(2026, 8, 23, 13, 5, tzinfo=UTC)))
    with pytest.raises(ValueError, match="ordering"):
        _start(trainer_started_at=datetime(2026, 8, 23, 12, 59, tzinfo=UTC))
