"""Terminal provider evidence is complete or explicitly absent."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from entropic_substrate.evidence.resource_billing import (
    BillingGap,
    ExecutionSegmentTermination,
    ProviderBilling,
)
from entropic_substrate.evidence.resource_provenance import BillingUnit, EvidenceGap, Money
from entropic_substrate.experiments.d1_primitives import EvidenceClass

pytestmark = pytest.mark.unit
LOGICAL = "d1-both-seed-7"
SEGMENT = f"{LOGICAL}:segment:0"


def _termination() -> ExecutionSegmentTermination:
    return ExecutionSegmentTermination(
        logical_run_id=LOGICAL,
        execution_segment_index=0,
        execution_segment_id=SEGMENT,
        trainer_ended_at=datetime(2026, 8, 23, 13, tzinfo=UTC),
        pod_terminated_at=datetime(2026, 8, 23, 13, 1, tzinfo=UTC),
    )


def _bill() -> ProviderBilling:
    return ProviderBilling(
        logical_run_id=LOGICAL,
        execution_segment_index=0,
        execution_segment_id=SEGMENT,
        billed_duration_seconds=Decimal("3660"),
        billed_amount=Money(Decimal("6.1"), "USD"),
        billing_unit=BillingUnit.MINUTE,
        billing_record_id="provider-record-0",
        evidence_class=EvidenceClass.PROVIDER_BILLED,
    )


def _gap() -> BillingGap:
    reason = "provider export not yet available"
    return BillingGap(
        logical_run_id=LOGICAL,
        execution_segment_index=0,
        execution_segment_id=SEGMENT,
        billed_duration=EvidenceGap("provider_billed_duration_seconds", reason),
        billed_amount=EvidenceGap("provider_billed_amount", reason),
        billing_unit=EvidenceGap("provider_billing_unit", reason),
        billing_record_id=EvidenceGap("provider_billing_record_id", reason),
    )


def test_termination_is_utc_ordered_and_has_a_strict_manifest() -> None:
    assert _termination().manifest == {
        "execution_segment_id": SEGMENT,
        "execution_segment_index": 0,
        "logical_run_id": LOGICAL,
        "pod_terminated_at": "2026-08-23T13:01:00+00:00",
        "schema_version": "resource-billing/v3",
        "trainer_ended_at": "2026-08-23T13:00:00+00:00",
    }
    with pytest.raises(ValueError, match="ordering"):
        replace(_termination(), pod_terminated_at=datetime(2026, 8, 23, 12, 59, tzinfo=UTC))


def test_provider_bill_is_all_present_exact_and_provider_classed() -> None:
    assert _bill().manifest["evidence_class"] == "provider_billed"
    assert _bill().manifest["billed_amount"] == {"amount": "6.1", "currency": "USD"}
    assert _bill().manifest["billing_unit"] == "minute"
    with pytest.raises(ValueError, match="evidence_class"):
        replace(_bill(), evidence_class=EvidenceClass.MEASURED)
    with pytest.raises(ValueError, match="canonical"):
        replace(_bill(), billed_duration_seconds=Decimal("3660.0"))
    with pytest.raises(ValueError, match="finite"):
        replace(_bill(), billed_duration_seconds=Decimal("NaN"))
    with pytest.raises(TypeError, match="BillingUnit"):
        replace(_bill(), billing_unit="minute")  # type: ignore[arg-type]


def test_complete_provider_bill_refuses_a_unit_gap_without_inference() -> None:
    gap = EvidenceGap("provider_billing_unit", "provider record omits its billing unit")
    with pytest.raises(TypeError, match="BillingUnit"):
        replace(_bill(), billing_unit=gap)


def test_terminal_gap_requires_all_exact_names_without_invalidating_quality() -> None:
    gap = _gap()
    assert gap.quality_evidence_admissible is True
    assert gap.manifest["kind"] == "billing_gap"
    assert gap.manifest["provider_billing_unit"]["field"] == "provider_billing_unit"
    assert gap.manifest["provider_billing_record_id"]["field"] == ("provider_billing_record_id")
    with pytest.raises(ValueError, match="provider_billed_amount"):
        replace(gap, billed_amount=EvidenceGap("wrong", "missing"))
    with pytest.raises(ValueError, match="provider_billing_unit"):
        replace(gap, billing_unit=EvidenceGap("billing_unit", "missing"))


def test_segment_ids_and_schema_are_not_free_form() -> None:
    with pytest.raises(ValueError, match="identity"):
        replace(_termination(), execution_segment_id="wrong")
    with pytest.raises(ValueError, match="schema_version"):
        replace(_bill(), schema_version="resource-billing/v1")
    with pytest.raises(ValueError, match="billing_record_id"):
        replace(_bill(), billing_record_id="")
