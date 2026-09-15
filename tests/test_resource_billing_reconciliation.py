"""Terminal estimates and multi-segment reconciliation preserve evidence classes."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from entropic_substrate.evidence.resource_billing import (
    BillingGap,
    ExecutionSegmentTermination,
    ProviderBilledAmountObservation,
    ProviderBilling,
    ProviderBillingPartial,
)
from entropic_substrate.evidence.resource_billing_reconciliation import (
    LogicalBillingMissingCost,
    LogicalBillingReconciliation,
    SegmentBillingMissingCost,
    SegmentBillingReconciliation,
    estimate_terminal_rate,
    reconcile_logical_billing,
    reconcile_segment_billing,
)
from entropic_substrate.evidence.resource_provenance import (
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
from entropic_substrate.experiments.d1_primitives import EvidenceClass

pytestmark = pytest.mark.unit


def _start(segment: int = 0, *, currency: str = "USD") -> RunStartProvenance:
    logical = "d1-both-seed-7"
    created = datetime(2026, 8, 23, 12 + 2 * segment, tzinfo=UTC)
    return RunStartProvenance(
        RESOURCE_TELEMETRY_SCHEMA_VERSION,
        logical,
        segment,
        execution_segment_id(logical, segment),
        SegmentStartMode.FRESH if segment == 0 else SegmentStartMode.RESUME,
        None if segment == 0 else "b" * 64,
        "NVIDIA A100 PCIe 80GB",
        2,
        "runpod",
        "a100-pcie-80gb",
        "US-TX-3",
        "secure-cloud",
        PurchaseOption.ON_DEMAND,
        "sha256:" + "a" * 64,
        "12.6",
        "560.35.03",
        "2.13.0",
        True,
        "triton-3.4.0",
        unavailable_resource_activation_registration(),
        QuotedRate(
            Money(Decimal("3"), currency),
            BillingUnit.HOUR,
            BillingRoundingPolicy(1, RoundingMode.CEILING, 1),
            RateScope.PER_ACCELERATOR,
            "RunPod pricing API",
            f"quote-{segment}",
            created,
            EvidenceClass.MEASURED,
        ),
        created,
        created + timedelta(minutes=1),
    )


def _termination(segment: int = 0) -> ExecutionSegmentTermination:
    start = _start(segment)
    return ExecutionSegmentTermination(
        start.logical_run_id,
        segment,
        start.execution_segment_id,
        start.pod_created_at + timedelta(hours=1),
        start.pod_created_at + timedelta(hours=1, seconds=1),
    )


def _bill(segment: int = 0, amount: str = "11.7", record: str | None = None) -> ProviderBilling:
    start = _start(segment)
    return ProviderBilling(
        start.logical_run_id,
        segment,
        start.execution_segment_id,
        Decimal("7200"),
        Money(Decimal(amount), "USD"),
        BillingUnit.SECOND,
        record or f"provider-bill-{segment}",
        EvidenceClass.PROVIDER_BILLED,
    )


def _gap(segment: int) -> BillingGap:
    start = _start(segment)
    reason = "provider billing export not yet available"
    return BillingGap(
        start.logical_run_id,
        segment,
        start.execution_segment_id,
        EvidenceGap("provider_billed_duration_seconds", reason),
        EvidenceGap("provider_billed_amount", reason),
        EvidenceGap("provider_billing_unit", reason),
        EvidenceGap("provider_billing_record_id", reason),
    )


def test_terminal_estimate_applies_rounding_and_per_accelerator_scope() -> None:
    estimate = estimate_terminal_rate(_start(), _termination())
    assert estimate.measured_duration_seconds == Decimal("3601")
    assert estimate.rounded_billable_units == Decimal("2")
    assert estimate.rounded_billable_duration_seconds == Decimal("7200")
    assert estimate.scope_multiplier == 2
    assert estimate.amount == Money(Decimal("12"), "USD")
    assert estimate.evidence_class is EvidenceClass.RATE_ESTIMATE


def test_segment_reconciliation_is_exact_and_missing_billing_is_not_zero() -> None:
    reconciled = reconcile_segment_billing(_start(), _termination(), _bill())
    assert reconciled.amount_delta == Decimal("-0.3")
    assert reconciled.duration_delta_seconds == Decimal("0")
    missing = reconcile_segment_billing(_start(), _termination(), _gap(0))
    assert type(missing) is SegmentBillingMissingCost
    assert missing.provider_billed_amount is None
    with pytest.raises(ValueError, match="canonical"):
        replace(reconciled, amount_delta=Decimal("-0.30"))


def test_partial_unit_gap_is_preserved_without_money_reconciliation() -> None:
    measured = reconcile_segment_billing(_start(), _termination(), _bill())
    assert type(measured) is SegmentBillingReconciliation
    assert measured.rate_estimate.quoted_rate.billing_unit is BillingUnit.HOUR
    assert measured.provider_billing.billing_unit is BillingUnit.SECOND

    unit_gap = EvidenceGap("provider_billing_unit", "provider export omits billing unit")
    start = _start()
    billed = ProviderBillingPartial(
        start.logical_run_id,
        start.execution_segment_index,
        start.execution_segment_id,
        Decimal("7200"),
        ProviderBilledAmountObservation(Decimal("11.7"), "USD"),
        unit_gap,
        "provider-bill-0",
        "settled",
        "2026-08-23 13:00:00",
        "a" * 64,
        start.trainer_started_at,
        EvidenceClass.PROVIDER_BILLED,
    )

    reconciled = reconcile_segment_billing(_start(), _termination(), billed)

    assert type(reconciled) is SegmentBillingMissingCost
    assert reconciled.rate_estimate.quoted_rate.billing_unit is BillingUnit.HOUR
    assert reconciled.provider_evidence.billing_unit == unit_gap
    assert reconciled.manifest["provider_evidence"]["billing_unit"] == {
        "field": "provider_billing_unit",
        "kind": "gap",
        "reason": "provider export omits billing unit",
    }


def test_reconciliation_refuses_identity_and_currency_substitution() -> None:
    with pytest.raises(ValueError, match="identity"):
        estimate_terminal_rate(_start(), _termination(1))
    with pytest.raises(ValueError, match="currency"):
        reconcile_segment_billing(_start(currency="EUR"), _termination(), _bill())


def test_logical_reconciliation_sums_exact_segments_and_preserves_ids() -> None:
    result = reconcile_logical_billing(
        (_start(0), _start(1)),
        (_termination(0), _termination(1)),
        (_bill(0), _bill(1, "12.1")),
    )
    assert type(result) is LogicalBillingReconciliation
    assert result.estimated_amount == Money(Decimal("24"), "USD")
    assert result.provider_billed_amount == Money(Decimal("23.8"), "USD")
    assert result.amount_delta == Decimal("-0.2")
    assert result.provider_billed_duration_seconds == Decimal("14400")
    assert [item.execution_segment_id for item in result.segments] == [
        "d1-both-seed-7:segment:0",
        "d1-both-seed-7:segment:1",
    ]


def test_incomplete_logical_bill_is_typed_and_duplicate_records_are_refused() -> None:
    missing = reconcile_logical_billing(
        (_start(0), _start(1)),
        (_termination(0), _termination(1)),
        (_bill(0), _gap(1)),
    )
    assert type(missing) is LogicalBillingMissingCost
    assert missing.provider_billed_amount is None
    assert missing.provider_billed_duration_seconds is None
    assert missing.estimated_amount == Money(Decimal("24"), "USD")
    assert missing.missing_segment_ids == ("d1-both-seed-7:segment:1",)
    assert missing.segments[1].manifest["provider_evidence"]["kind"] == "billing_gap"
    with pytest.raises(ValueError, match="duplicate provider billing_record_id"):
        reconcile_logical_billing(
            (_start(0), _start(1)),
            (_termination(0), _termination(1)),
            (_bill(0, record="same"), _bill(1, record="same")),
        )
    with pytest.raises(ValueError, match="consecutive"):
        reconcile_logical_billing(
            (_start(0), _start(0)),
            (_termination(0), _termination(0)),
            (_bill(0), _bill(0, record="other")),
        )
