"""Canonical run-segment provenance mapping and JSON admission."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from entropic_substrate.kernel.evidence_class.resource_provenance import (
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
)
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import (
    ResourceProvenanceCodecGap,
    parse_run_start_line,
    quoted_rate_from_mapping,
    quoted_rate_to_mapping,
    resource_activation_from_mapping,
    resource_activation_to_mapping,
    run_start_from_mapping,
    run_start_to_mapping,
    serialize_run_start_line,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass

pytestmark = pytest.mark.unit


def _start(segment: int = 0) -> RunStartProvenance:
    logical = "d1-both-seed-7"
    created = datetime(2026, 8, 24, 1, segment, 2, 123456, tzinfo=UTC)
    return RunStartProvenance(
        schema_version=RESOURCE_TELEMETRY_SCHEMA_VERSION,
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
        compiler_enabled=False,
        compiler_version="disabled",
        resource_activation=unavailable_resource_activation_registration(),
        quoted_rate=QuotedRate(
            amount=Money(Decimal("1.39"), "USD"),
            billing_unit=BillingUnit.HOUR,
            rounding_policy=BillingRoundingPolicy(1, RoundingMode.CEILING, 1),
            rate_scope=RateScope.PER_ACCELERATOR,
            source="RunPod pricing API",
            quote_id="quote-a100-20260824",
            observed_at=created,
            evidence_class=EvidenceClass.MEASURED,
        ),
        pod_created_at=created,
        trainer_started_at=created + timedelta(minutes=1),
    )


def test_run_start_mapping_and_json_line_round_trip_canonically() -> None:
    start = _start()
    mapping = run_start_to_mapping(start)
    assert mapping["event"] == "resource.segment_start"
    assert mapping["pod_created_at"] == "2026-08-24T01:00:02.123456Z"
    quoted = mapping["quoted_rate"]
    assert isinstance(quoted, dict)
    assert quoted["amount"] == {"amount": "1.39", "currency": "USD"}
    activation = mapping["resource_activation"]
    assert isinstance(activation, dict)
    assert activation["mode"] == "gaps-only"
    assert run_start_from_mapping(mapping) == start
    line = serialize_run_start_line(start)
    assert parse_run_start_line(line) == start
    assert serialize_run_start_line(parse_run_start_line(line)) == line


def test_rate_and_activation_component_codecs_reuse_run_start_rules() -> None:
    start = _start()

    rate = quoted_rate_to_mapping(start.quoted_rate)
    activation = resource_activation_to_mapping(start.resource_activation)

    assert rate["observed_at"] == "2026-08-24T01:00:02.123456Z"
    assert quoted_rate_from_mapping(rate) == start.quoted_rate
    assert resource_activation_from_mapping(activation) == start.resource_activation

    rate["unknown"] = True
    activation["unknown"] = True
    with pytest.raises(ResourceProvenanceCodecGap):
        quoted_rate_from_mapping(rate)
    with pytest.raises(ResourceProvenanceCodecGap):
        resource_activation_from_mapping(activation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_segment_index", True),
        ("accelerator_count", False),
        ("compiler_enabled", 0),
        ("pod_created_at", "2026-08-24T01:00:02+00:00"),
        ("schema_version", "resource-telemetry/v2"),
    ],
)
def test_mapping_refuses_type_schema_and_noncanonical_timestamp(field: str, value: object) -> None:
    mapping = run_start_to_mapping(_start())
    mapping[field] = value
    with pytest.raises(ResourceProvenanceCodecGap):
        run_start_from_mapping(mapping)


def test_mapping_refuses_unknown_missing_and_noncanonical_decimal() -> None:
    unknown = run_start_to_mapping(_start())
    unknown["invented"] = 1
    missing = run_start_to_mapping(_start())
    missing.pop("region")
    decimal = run_start_to_mapping(_start())
    quoted = decimal["quoted_rate"]
    assert isinstance(quoted, dict)
    amount = quoted["amount"]
    assert isinstance(amount, dict)
    amount["amount"] = "1.390"
    activation = run_start_to_mapping(_start())
    activation_value = activation["resource_activation"]
    assert isinstance(activation_value, dict)
    activation_value["invented"] = True
    for mapping in (unknown, missing, decimal, activation):
        with pytest.raises(ResourceProvenanceCodecGap):
            run_start_from_mapping(mapping)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "[]",
        '{"event":"resource.segment_start","x":NaN}',
        '{"event":"resource.segment_start","event":"duplicate"}',
    ],
)
def test_json_line_refuses_blank_nonobject_nonfinite_and_duplicate_keys(line: str) -> None:
    with pytest.raises(ResourceProvenanceCodecGap):
        parse_run_start_line(line)
