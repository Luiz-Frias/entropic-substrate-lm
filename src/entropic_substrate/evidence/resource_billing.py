"""Immutable terminal lifecycle and provider billing evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from entropic_substrate.evidence.resource_provenance import (
    BillingUnit,
    EvidenceGap,
    Money,
    execution_segment_id,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass

RESOURCE_BILLING_SCHEMA_VERSION = "resource-billing/v2"
PROVIDER_BILLING_GAP_FIELDS = (
    "provider_billed_duration_seconds",
    "provider_billed_amount",
    "provider_billing_unit",
    "provider_billing_record_id",
)


def canonical_decimal(value: Decimal) -> Decimal:
    """Return a calculation result with the one admitted decimal spelling."""
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError("decimal calculation must be finite Decimal")
    return Decimal("0") if value.is_zero() else Decimal(format(value.normalize(), "f"))


def require_canonical_decimal(value: Decimal, *, label: str, signed: bool = False) -> None:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be a Decimal")
    if not value.is_finite() or (not signed and value < 0):
        qualifier = "finite" if signed else "finite and nonnegative"
        raise ValueError(f"{label} must be {qualifier}")
    if str(value) != str(canonical_decimal(value)):
        raise ValueError(f"{label} must use canonical decimal spelling")


def validate_segment_identity(logical_run_id: str, index: int, segment_id: str) -> None:
    if type(logical_run_id) is not str or not logical_run_id.strip():
        raise ValueError("logical_run_id must be a non-empty string")
    if type(index) is not int or index < 0:
        raise ValueError("execution_segment_index must be a nonnegative integer")
    if segment_id != execution_segment_id(logical_run_id, index):
        raise ValueError("execution segment identity is not derived from logical run and index")


def _utc(value: datetime, *, label: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a UTC timestamp")


def _gap_manifest(gap: EvidenceGap) -> dict[str, str]:
    return {"field": gap.field, "kind": "gap", "reason": gap.reason}


def _provider_billing_unit_manifest(
    value: BillingUnit | EvidenceGap,
) -> str | dict[str, str]:
    return value.value if isinstance(value, BillingUnit) else _gap_manifest(value)


def _validate_provider_billing_unit(value: BillingUnit | EvidenceGap) -> None:
    if type(value) is BillingUnit:
        return
    if type(value) is not EvidenceGap:
        raise TypeError("billing_unit must be BillingUnit or EvidenceGap")
    if value.field != "provider_billing_unit":
        raise ValueError("provider billing unit gap must name exactly 'provider_billing_unit'")


@dataclass(frozen=True, slots=True)
class ExecutionSegmentTermination:
    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    trainer_ended_at: datetime
    pod_terminated_at: datetime
    schema_version: str = RESOURCE_BILLING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_BILLING_SCHEMA_VERSION:
            raise ValueError("unsupported resource billing schema_version")
        validate_segment_identity(
            self.logical_run_id, self.execution_segment_index, self.execution_segment_id
        )
        _utc(self.trainer_ended_at, label="trainer_ended_at")
        _utc(self.pod_terminated_at, label="pod_terminated_at")
        if self.pod_terminated_at < self.trainer_ended_at:
            raise ValueError("trainer-end/pod-termination timestamp ordering is inverted")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "logical_run_id": self.logical_run_id,
            "pod_terminated_at": self.pod_terminated_at.isoformat(),
            "schema_version": self.schema_version,
            "trainer_ended_at": self.trainer_ended_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProviderBilling:
    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    billed_duration_seconds: Decimal
    billed_amount: Money
    billing_unit: BillingUnit | EvidenceGap
    billing_record_id: str
    evidence_class: EvidenceClass
    schema_version: str = RESOURCE_BILLING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_BILLING_SCHEMA_VERSION:
            raise ValueError("unsupported resource billing schema_version")
        validate_segment_identity(
            self.logical_run_id, self.execution_segment_index, self.execution_segment_id
        )
        require_canonical_decimal(self.billed_duration_seconds, label="billed_duration_seconds")
        if type(self.billed_amount) is not Money:
            raise TypeError("billed_amount must be Money")
        _validate_provider_billing_unit(self.billing_unit)
        if type(self.billing_record_id) is not str or not self.billing_record_id.strip():
            raise ValueError("billing_record_id must be a non-empty string")
        if self.evidence_class is not EvidenceClass.PROVIDER_BILLED:
            raise ValueError("provider billing evidence_class must be PROVIDER_BILLED")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "billed_amount": self.billed_amount.manifest,
            "billed_duration_seconds": str(self.billed_duration_seconds),
            "billing_unit": _provider_billing_unit_manifest(self.billing_unit),
            "billing_record_id": self.billing_record_id,
            "evidence_class": self.evidence_class.value,
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "kind": "provider_billing",
            "logical_run_id": self.logical_run_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class BillingGap:
    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    billed_duration: EvidenceGap
    billed_amount: EvidenceGap
    billing_unit: EvidenceGap
    billing_record_id: EvidenceGap
    schema_version: str = RESOURCE_BILLING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_BILLING_SCHEMA_VERSION:
            raise ValueError("unsupported resource billing schema_version")
        validate_segment_identity(
            self.logical_run_id, self.execution_segment_index, self.execution_segment_id
        )
        values = (
            self.billed_duration,
            self.billed_amount,
            self.billing_unit,
            self.billing_record_id,
        )
        for value, field in zip(values, PROVIDER_BILLING_GAP_FIELDS, strict=True):
            if type(value) is not EvidenceGap:
                raise TypeError(f"{field} must be EvidenceGap")
            if value.field != field:
                raise ValueError(f"billing gap must name exactly {field!r}")

    @property
    def quality_evidence_admissible(self) -> bool:
        return True

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "kind": "billing_gap",
            "logical_run_id": self.logical_run_id,
            "provider_billed_amount": _gap_manifest(self.billed_amount),
            "provider_billed_duration_seconds": _gap_manifest(self.billed_duration),
            "provider_billing_unit": _gap_manifest(self.billing_unit),
            "provider_billing_record_id": _gap_manifest(self.billing_record_id),
            "quality_evidence_admissible": True,
            "schema_version": self.schema_version,
        }
