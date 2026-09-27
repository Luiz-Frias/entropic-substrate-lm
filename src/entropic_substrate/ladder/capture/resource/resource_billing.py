"""Immutable terminal lifecycle and provider billing evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import cast

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    BillingUnit,
    EvidenceGap,
    Money,
    execution_segment_id,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass

RESOURCE_BILLING_SCHEMA_VERSION = "resource-billing/v3"
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


def _validate_gap(value: EvidenceGap, *, field: str) -> None:
    if type(value) is not EvidenceGap or value.field != field:
        raise ValueError(f"billing gap must name exactly {field!r}")


def _decimal_or_gap(value: Decimal | EvidenceGap, *, field: str) -> None:
    if type(value) is EvidenceGap:
        _validate_gap(value, field=field)
    else:
        require_canonical_decimal(cast(Decimal, value), label=field)


def _text_or_gap(value: str | EvidenceGap, *, field: str) -> None:
    if type(value) is EvidenceGap:
        _validate_gap(value, field=field)
    elif type(value) is not str or not value.strip() or value.strip() != value:
        raise ValueError(f"{field} must be nonempty trimmed text or EvidenceGap")


def _text(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be text")
    if not value.strip() or value.strip() != value:
        raise ValueError(f"{field} must be nonempty trimmed text")
    return value


def _decimal_or_gap_manifest(value: Decimal | EvidenceGap) -> str | dict[str, str]:
    return _gap_manifest(value) if type(value) is EvidenceGap else str(value)


def _text_or_gap_manifest(value: str | EvidenceGap) -> str | dict[str, str]:
    return _gap_manifest(value) if type(value) is EvidenceGap else cast(str, value)


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
class ProviderBilledAmountObservation:
    """Provider-reported numeric amount whose currency may be unavailable."""

    amount: Decimal | EvidenceGap
    currency: str | EvidenceGap

    def __post_init__(self) -> None:
        _decimal_or_gap(self.amount, field="provider_billed_amount")
        if type(self.currency) is EvidenceGap:
            _validate_gap(self.currency, field="provider_billed_currency")
        elif type(self.currency) is not str:
            raise TypeError("provider billed currency must be text or EvidenceGap")
        else:
            Money(Decimal("0"), self.currency)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "amount": _decimal_or_gap_manifest(self.amount),
            "currency": _text_or_gap_manifest(self.currency),
        }


@dataclass(frozen=True, slots=True)
class ProviderBilling:
    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    billed_duration_seconds: Decimal
    billed_amount: Money
    billing_unit: BillingUnit
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
        if type(self.billing_unit) is not BillingUnit:
            raise TypeError("complete provider billing requires BillingUnit")
        _text(self.billing_record_id, field="billing_record_id")
        if self.evidence_class is not EvidenceClass.PROVIDER_BILLED:
            raise ValueError("provider billing evidence_class must be PROVIDER_BILLED")

    is_terminal = True
    provider_billing_complete = True
    quality_evidence_admissible = True

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "billed_amount": self.billed_amount.manifest,
            "billed_duration_seconds": str(self.billed_duration_seconds),
            "billing_record_id": self.billing_record_id,
            "billing_unit": self.billing_unit.value,
            "evidence_class": self.evidence_class.value,
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "kind": "provider_billing",
            "logical_run_id": self.logical_run_id,
            "quality_evidence_admissible": True,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ProviderBillingPartial:
    """Terminal provider rows with some measured facts and explicit gaps."""

    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    billed_duration_seconds: Decimal | EvidenceGap
    billed_amount: ProviderBilledAmountObservation
    billing_unit: BillingUnit | EvidenceGap
    billing_record_id: str | EvidenceGap
    provider_status: str | EvidenceGap
    provider_timestamp: str | EvidenceGap
    observation_sha256: str
    reconciled_at: datetime
    evidence_class: EvidenceClass
    schema_version: str = RESOURCE_BILLING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_BILLING_SCHEMA_VERSION:
            raise ValueError("unsupported resource billing schema_version")
        validate_segment_identity(
            self.logical_run_id, self.execution_segment_index, self.execution_segment_id
        )
        _decimal_or_gap(
            self.billed_duration_seconds,
            field="provider_billed_duration_seconds",
        )
        if type(self.billed_amount) is not ProviderBilledAmountObservation:
            raise TypeError("partial provider billing requires amount observation")
        _validate_provider_billing_unit(self.billing_unit)
        _text_or_gap(self.billing_record_id, field="provider_billing_record_id")
        _text_or_gap(self.provider_status, field="provider_billing_status")
        _text_or_gap(self.provider_timestamp, field="provider_billing_timestamp")
        if (
            type(self.observation_sha256) is not str
            or len(self.observation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.observation_sha256)
        ):
            raise ValueError("provider billing observation_sha256 must be lowercase SHA-256")
        _utc(self.reconciled_at, label="reconciled_at")
        if self.evidence_class is not EvidenceClass.PROVIDER_BILLED:
            raise ValueError("partial provider billing evidence_class must be PROVIDER_BILLED")
        core = (
            self.billed_duration_seconds,
            self.billed_amount.amount,
            self.billed_amount.currency,
            self.billing_unit,
            self.billing_record_id,
            self.provider_status,
        )
        if all(type(value) is EvidenceGap for value in core):
            raise ValueError("all-missing provider billing must use BillingGap")
        if (
            type(self.billed_duration_seconds) is Decimal
            and type(self.billed_amount.amount) is Decimal
            and type(self.billed_amount.currency) is str
            and type(self.billing_unit) is BillingUnit
            and type(self.billing_record_id) is str
        ):
            raise ValueError("complete provider billing must use ProviderBilling")

    is_terminal = True
    provider_billing_complete = False
    quality_evidence_admissible = True

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "billed_amount": self.billed_amount.manifest,
            "billed_duration_seconds": _decimal_or_gap_manifest(self.billed_duration_seconds),
            "billing_record_id": _text_or_gap_manifest(self.billing_record_id),
            "billing_unit": _provider_billing_unit_manifest(self.billing_unit),
            "evidence_class": self.evidence_class.value,
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "kind": "partial_provider_billing",
            "logical_run_id": self.logical_run_id,
            "observation_sha256": self.observation_sha256,
            "provider_status": _text_or_gap_manifest(self.provider_status),
            "provider_timestamp": _text_or_gap_manifest(self.provider_timestamp),
            "quality_evidence_admissible": True,
            "reconciled_at": self.reconciled_at.isoformat(),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ProviderBillingPending:
    """A successful provider query whose terminal billing row has not settled."""

    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    checked_at: datetime
    reason: str
    schema_version: str = RESOURCE_BILLING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_BILLING_SCHEMA_VERSION:
            raise ValueError("unsupported resource billing schema_version")
        validate_segment_identity(
            self.logical_run_id, self.execution_segment_index, self.execution_segment_id
        )
        _utc(self.checked_at, label="checked_at")
        _text(self.reason, field="provider billing pending reason")

    is_terminal = False
    provider_billing_complete = False
    quality_evidence_admissible = True

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "checked_at": self.checked_at.isoformat(),
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "kind": "provider_billing_pending",
            "logical_run_id": self.logical_run_id,
            "quality_evidence_admissible": True,
            "reason": self.reason,
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
            _validate_gap(value, field=field)

    is_terminal = True
    provider_billing_complete = False
    quality_evidence_admissible = True

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "kind": "billing_gap",
            "logical_run_id": self.logical_run_id,
            "provider_billed_amount": _gap_manifest(self.billed_amount),
            "provider_billed_duration_seconds": _gap_manifest(self.billed_duration),
            "provider_billing_record_id": _gap_manifest(self.billing_record_id),
            "provider_billing_unit": _gap_manifest(self.billing_unit),
            "quality_evidence_admissible": True,
            "schema_version": self.schema_version,
        }


ProviderBillingObservation = (
    ProviderBilling | ProviderBillingPartial | ProviderBillingPending | BillingGap
)
TerminalProviderBillingEvidence = ProviderBilling | ProviderBillingPartial | BillingGap
