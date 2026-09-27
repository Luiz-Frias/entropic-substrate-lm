"""Provider billing preserves measured partial facts without inferring missing fields."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    BillingUnit,
    EvidenceGap,
    Money,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.ladder.capture.resource.resource_billing import (
    BillingGap,
    ProviderBilledAmountObservation,
    ProviderBilling,
    ProviderBillingPartial,
    ProviderBillingPending,
)
from entropic_substrate.ladder.capture.resource.resource_billing_reconciliation import (
    reconcile_segment_billing,
)

from .test_resource_billing_reconciliation import _start, _termination

pytestmark = pytest.mark.unit

_LOGICAL = "d1-both-seed-7"
_SEGMENT = f"{_LOGICAL}:segment:0"
_RECONCILED = datetime(2026, 8, 26, 22, 35, tzinfo=UTC)


def _gap(field: str) -> EvidenceGap:
    return EvidenceGap(field, "RunPod billing rows omit this field")


def _partial(**changes: object) -> ProviderBillingPartial:
    values: dict[str, object] = {
        "logical_run_id": _LOGICAL,
        "execution_segment_index": 0,
        "execution_segment_id": _SEGMENT,
        "billed_duration_seconds": Decimal("24.726"),
        "billed_amount": ProviderBilledAmountObservation(
            Decimal("0.0017170833889394999"),
            _gap("provider_billed_currency"),
        ),
        "billing_unit": _gap("provider_billing_unit"),
        "billing_record_id": _gap("provider_billing_record_id"),
        "provider_status": _gap("provider_billing_status"),
        "provider_timestamp": "2026-08-26 17:00:00",
        "observation_sha256": "a" * 64,
        "reconciled_at": _RECONCILED,
        "evidence_class": EvidenceClass.PROVIDER_BILLED,
    }
    values.update(changes)
    return ProviderBillingPartial(**values)  # type: ignore[arg-type]


def test_partial_provider_billing_preserves_measured_fields_and_exact_gaps() -> None:
    partial = _partial()

    assert partial.is_terminal is True
    assert partial.provider_billing_complete is False
    assert partial.quality_evidence_admissible is True
    assert partial.manifest == {
        "billed_amount": {
            "amount": "0.0017170833889394999",
            "currency": {
                "field": "provider_billed_currency",
                "kind": "gap",
                "reason": "RunPod billing rows omit this field",
            },
        },
        "billed_duration_seconds": "24.726",
        "billing_record_id": {
            "field": "provider_billing_record_id",
            "kind": "gap",
            "reason": "RunPod billing rows omit this field",
        },
        "billing_unit": {
            "field": "provider_billing_unit",
            "kind": "gap",
            "reason": "RunPod billing rows omit this field",
        },
        "evidence_class": "provider_billed",
        "execution_segment_id": _SEGMENT,
        "execution_segment_index": 0,
        "kind": "partial_provider_billing",
        "logical_run_id": _LOGICAL,
        "observation_sha256": "a" * 64,
        "provider_status": {
            "field": "provider_billing_status",
            "kind": "gap",
            "reason": "RunPod billing rows omit this field",
        },
        "provider_timestamp": "2026-08-26 17:00:00",
        "quality_evidence_admissible": True,
        "reconciled_at": "2026-08-26T22:35:00+00:00",
        "schema_version": "resource-billing/v3",
    }


def test_partial_state_must_contain_both_provider_values_and_explicit_gaps() -> None:
    all_gap = {
        "billed_duration_seconds": _gap("provider_billed_duration_seconds"),
        "billed_amount": ProviderBilledAmountObservation(
            _gap("provider_billed_amount"),
            _gap("provider_billed_currency"),
        ),
    }
    with pytest.raises(ValueError, match="BillingGap"):
        _partial(**all_gap)

    with pytest.raises(ValueError, match="ProviderBilling"):
        _partial(
            billed_amount=ProviderBilledAmountObservation(Decimal("0.0017170833889394999"), "USD"),
            billing_unit=BillingUnit.SECOND,
            billing_record_id="provider-record-1",
            provider_status="settled",
        )


def test_only_complete_nominal_provider_billing_advances_paid_campaigns() -> None:
    complete = ProviderBilling(
        _LOGICAL,
        0,
        _SEGMENT,
        Decimal("24.726"),
        billed_amount=Money(Decimal("0.0017170833889394999"), "USD"),
        billing_unit=BillingUnit.SECOND,
        billing_record_id="provider-record-1",
        evidence_class=EvidenceClass.PROVIDER_BILLED,
    )
    gap = BillingGap(
        _LOGICAL,
        0,
        _SEGMENT,
        _gap("provider_billed_duration_seconds"),
        _gap("provider_billed_amount"),
        _gap("provider_billing_unit"),
        _gap("provider_billing_record_id"),
    )

    pending = ProviderBillingPending(
        _LOGICAL,
        0,
        _SEGMENT,
        _RECONCILED,
        "provider billing row not settled",
    )

    assert (complete.is_terminal, complete.provider_billing_complete) == (True, True)
    assert (_partial().is_terminal, _partial().provider_billing_complete) == (True, False)
    assert (pending.is_terminal, pending.provider_billing_complete) == (False, False)
    assert (gap.is_terminal, gap.provider_billing_complete) == (True, False)
    assert pending.manifest["kind"] == "provider_billing_pending"
    with pytest.raises(ValueError, match="pending"):
        reconcile_segment_billing(_start(), _termination(), pending)


def test_complete_pending_and_partial_authority_boundaries_are_nominal() -> None:
    with pytest.raises(TypeError, match="billing_record_id"):
        ProviderBilling(
            _LOGICAL,
            0,
            _SEGMENT,
            Decimal("24.726"),
            Money(Decimal("0.25"), "USD"),
            BillingUnit.SECOND,
            _gap("provider_billing_record_id"),  # type: ignore[arg-type]
            EvidenceClass.PROVIDER_BILLED,
        )
    with pytest.raises(TypeError, match="reason"):
        ProviderBillingPending(  # type: ignore[arg-type]
            _LOGICAL, 0, _SEGMENT, _RECONCILED, _gap("provider_billing_pending_reason")
        )
    with pytest.raises(ValueError, match="ProviderBilling"):
        _partial(
            billed_amount=ProviderBilledAmountObservation(Decimal("0.25"), "USD"),
            billing_unit=BillingUnit.SECOND,
            billing_record_id="provider-record-1",
            provider_status=_gap("provider_billing_status"),
        )
