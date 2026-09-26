"""Quoted-rate estimates and provider billing reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import cast

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    BillingRoundingPolicy,
    BillingUnit,
    Money,
    QuotedRate,
    RateScope,
    RoundingMode,
    RunStartProvenance,
    validate_execution_segments,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.ladder.capture.resource.resource_billing import (
    BillingGap,
    ExecutionSegmentTermination,
    ProviderBilling,
    ProviderBillingPartial,
    ProviderBillingPending,
    TerminalProviderBillingEvidence,
    canonical_decimal,
    require_canonical_decimal,
    validate_segment_identity,
)

_UNIT_SECONDS = {
    BillingUnit.SECOND: Decimal("1"),
    BillingUnit.MINUTE: Decimal("60"),
    BillingUnit.HOUR: Decimal("3600"),
}


def _rounded_units(raw: Decimal, policy: BillingRoundingPolicy) -> Decimal:
    increment = Decimal(policy.increment)
    subject = max(raw, Decimal(policy.minimum_units)) / increment
    mode = {
        RoundingMode.CEILING: ROUND_CEILING,
        RoundingMode.FLOOR: ROUND_FLOOR,
        RoundingMode.NEAREST: ROUND_HALF_UP,
    }[policy.mode]
    return canonical_decimal(subject.to_integral_value(rounding=mode) * increment)


def _elapsed_seconds(start: RunStartProvenance, end: ExecutionSegmentTermination) -> Decimal:
    delta = end.pod_terminated_at - start.pod_created_at
    value = Decimal(delta.days * 86400 + delta.seconds)
    value += Decimal(delta.microseconds) / Decimal(1_000_000)
    return canonical_decimal(value)


@dataclass(frozen=True, slots=True)
class TerminalRateEstimate:
    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    measured_duration_seconds: Decimal
    rounded_billable_units: Decimal
    rounded_billable_duration_seconds: Decimal
    accelerator_count: int
    scope_multiplier: int
    quoted_rate: QuotedRate
    amount: Money
    evidence_class: EvidenceClass

    def __post_init__(self) -> None:
        validate_segment_identity(
            self.logical_run_id, self.execution_segment_index, self.execution_segment_id
        )
        for label, value in (
            ("measured_duration_seconds", self.measured_duration_seconds),
            ("rounded_billable_units", self.rounded_billable_units),
            ("rounded_billable_duration_seconds", self.rounded_billable_duration_seconds),
        ):
            require_canonical_decimal(value, label=label)
        if type(self.accelerator_count) is not int or self.accelerator_count < 1:
            raise ValueError("accelerator_count must be positive")
        if type(self.quoted_rate) is not QuotedRate or type(self.amount) is not Money:
            raise TypeError("terminal estimate requires QuotedRate and Money")
        expected_scope = (
            self.accelerator_count
            if self.quoted_rate.rate_scope is RateScope.PER_ACCELERATOR
            else 1
        )
        expected_duration = canonical_decimal(
            self.rounded_billable_units * _UNIT_SECONDS[self.quoted_rate.billing_unit]
        )
        expected_amount = canonical_decimal(
            self.quoted_rate.amount.amount * self.rounded_billable_units * expected_scope
        )
        if (
            self.scope_multiplier != expected_scope
            or self.rounded_billable_duration_seconds != expected_duration
        ):
            raise ValueError("terminal estimate scope or rounded duration is inconsistent")
        if self.amount != Money(expected_amount, self.quoted_rate.amount.currency):
            raise ValueError("terminal estimate amount is inconsistent with quoted rate")
        if self.evidence_class is not EvidenceClass.RATE_ESTIMATE:
            raise ValueError("terminal estimate evidence_class must be RATE_ESTIMATE")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "accelerator_count": self.accelerator_count,
            "amount": self.amount.manifest,
            "evidence_class": self.evidence_class.value,
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "logical_run_id": self.logical_run_id,
            "measured_duration_seconds": str(self.measured_duration_seconds),
            "quoted_rate": self.quoted_rate.manifest,
            "rounded_billable_duration_seconds": str(self.rounded_billable_duration_seconds),
            "rounded_billable_units": str(self.rounded_billable_units),
            "scope_multiplier": self.scope_multiplier,
        }


def estimate_terminal_rate(
    start: RunStartProvenance, termination: ExecutionSegmentTermination
) -> TerminalRateEstimate:
    if (
        type(start) is not RunStartProvenance
        or type(termination) is not ExecutionSegmentTermination
    ):
        raise TypeError("terminal estimation requires start and termination evidence")
    if (start.logical_run_id, start.execution_segment_id) != (
        termination.logical_run_id,
        termination.execution_segment_id,
    ):
        raise ValueError("start/termination segment identity mismatch")
    if termination.trainer_ended_at < start.trainer_started_at:
        raise ValueError("trainer start/end timestamp ordering is inverted")
    duration = _elapsed_seconds(start, termination)
    unit_seconds = _UNIT_SECONDS[start.quoted_rate.billing_unit]
    units = _rounded_units(duration / unit_seconds, start.quoted_rate.rounding_policy)
    multiplier = (
        start.accelerator_count if start.quoted_rate.rate_scope is RateScope.PER_ACCELERATOR else 1
    )
    amount = canonical_decimal(start.quoted_rate.amount.amount * units * multiplier)
    return TerminalRateEstimate(
        start.logical_run_id,
        start.execution_segment_index,
        start.execution_segment_id,
        duration,
        units,
        canonical_decimal(units * unit_seconds),
        start.accelerator_count,
        multiplier,
        start.quoted_rate,
        Money(amount, start.quoted_rate.amount.currency),
        EvidenceClass.RATE_ESTIMATE,
    )


@dataclass(frozen=True, slots=True)
class SegmentBillingReconciliation:
    rate_estimate: TerminalRateEstimate
    provider_billing: ProviderBilling
    amount_delta: Decimal
    duration_delta_seconds: Decimal

    def __post_init__(self) -> None:
        if type(self.rate_estimate) is not TerminalRateEstimate:
            raise TypeError("segment reconciliation requires TerminalRateEstimate")
        if type(self.provider_billing) is not ProviderBilling:
            raise TypeError("segment reconciliation requires ProviderBilling")
        if self.rate_estimate.execution_segment_id != self.provider_billing.execution_segment_id:
            raise ValueError("segment reconciliation identities differ")
        if self.rate_estimate.amount.currency != self.provider_billing.billed_amount.currency:
            raise ValueError("segment reconciliation currencies differ")
        require_canonical_decimal(self.amount_delta, label="amount_delta", signed=True)
        require_canonical_decimal(
            self.duration_delta_seconds,
            label="duration_delta_seconds",
            signed=True,
        )
        if self.amount_delta != canonical_decimal(
            self.provider_billing.billed_amount.amount - self.rate_estimate.amount.amount
        ):
            raise ValueError("segment reconciliation amount_delta is inconsistent")

    @property
    def execution_segment_id(self) -> str:
        return self.rate_estimate.execution_segment_id

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "amount_delta": str(self.amount_delta),
            "duration_delta_seconds": str(self.duration_delta_seconds),
            "kind": "reconciled",
            "provider_billing": self.provider_billing.manifest,
            "rate_estimate": self.rate_estimate.manifest,
        }


@dataclass(frozen=True, slots=True)
class SegmentBillingMissingCost:
    rate_estimate: TerminalRateEstimate
    provider_evidence: ProviderBillingPartial | BillingGap

    def __post_init__(self) -> None:
        if type(self.rate_estimate) is not TerminalRateEstimate or type(
            self.provider_evidence
        ) not in (ProviderBillingPartial, BillingGap):
            raise TypeError("missing-cost outcome requires terminal incomplete provider evidence")
        if self.rate_estimate.execution_segment_id != self.provider_evidence.execution_segment_id:
            raise ValueError("missing-cost segment identities differ")

    @property
    def execution_segment_id(self) -> str:
        return self.rate_estimate.execution_segment_id

    @property
    def provider_billed_amount(self) -> None:
        return None

    @property
    def provider_billed_duration_seconds(self) -> None:
        return None

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "provider_evidence": self.provider_evidence.manifest,
            "kind": "missing_cost",
            "rate_estimate": self.rate_estimate.manifest,
        }


SegmentBillingOutcome = SegmentBillingReconciliation | SegmentBillingMissingCost


def reconcile_segment_billing(
    start: RunStartProvenance,
    termination: ExecutionSegmentTermination,
    provider: TerminalProviderBillingEvidence | ProviderBillingPending,
) -> SegmentBillingOutcome:
    estimate = estimate_terminal_rate(start, termination)
    if type(provider) is ProviderBillingPending:
        raise ValueError("pending provider billing cannot enter terminal reconciliation")
    if type(provider) not in (ProviderBilling, ProviderBillingPartial, BillingGap):
        raise TypeError("provider billing must be terminal nominal evidence")
    if (provider.logical_run_id, provider.execution_segment_id) != (
        start.logical_run_id,
        start.execution_segment_id,
    ):
        raise ValueError("provider/start segment identity mismatch")
    if type(provider) in (ProviderBillingPartial, BillingGap):
        return SegmentBillingMissingCost(
            estimate,
            cast(ProviderBillingPartial | BillingGap, provider),
        )
    billed = cast(ProviderBilling, provider)
    if billed.billed_amount.currency != estimate.amount.currency:
        raise ValueError("provider and estimate currency mismatch; FX inference is forbidden")
    return SegmentBillingReconciliation(
        estimate,
        billed,
        canonical_decimal(billed.billed_amount.amount - estimate.amount.amount),
        canonical_decimal(
            billed.billed_duration_seconds - estimate.rounded_billable_duration_seconds
        ),
    )


@dataclass(frozen=True, slots=True)
class LogicalBillingReconciliation:
    logical_run_id: str
    segments: tuple[SegmentBillingReconciliation, ...]
    estimated_amount: Money
    provider_billed_amount: Money
    estimated_billed_duration_seconds: Decimal
    provider_billed_duration_seconds: Decimal
    amount_delta: Decimal
    duration_delta_seconds: Decimal

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "amount_delta": str(self.amount_delta),
            "duration_delta_seconds": str(self.duration_delta_seconds),
            "estimated_amount": self.estimated_amount.manifest,
            "estimated_billed_duration_seconds": str(self.estimated_billed_duration_seconds),
            "kind": "reconciled",
            "logical_run_id": self.logical_run_id,
            "provider_billed_amount": self.provider_billed_amount.manifest,
            "provider_billed_duration_seconds": str(self.provider_billed_duration_seconds),
            "segments": [item.manifest for item in self.segments],
        }


@dataclass(frozen=True, slots=True)
class LogicalBillingMissingCost:
    logical_run_id: str
    segments: tuple[SegmentBillingOutcome, ...]
    estimated_amount: Money
    estimated_billed_duration_seconds: Decimal
    missing_segment_ids: tuple[str, ...]

    @property
    def provider_billed_amount(self) -> None:
        return None

    @property
    def provider_billed_duration_seconds(self) -> None:
        return None

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "estimated_amount": self.estimated_amount.manifest,
            "estimated_billed_duration_seconds": str(self.estimated_billed_duration_seconds),
            "kind": "missing_cost",
            "logical_run_id": self.logical_run_id,
            "missing_segment_ids": list(self.missing_segment_ids),
            "provider_billed_amount": None,
            "provider_billed_duration_seconds": None,
            "segments": [item.manifest for item in self.segments],
        }


LogicalBillingOutcome = LogicalBillingReconciliation | LogicalBillingMissingCost


def _money_sum(values: tuple[Money, ...]) -> Money:
    currencies = {item.currency for item in values}
    if len(currencies) != 1:
        raise ValueError("logical billing cannot aggregate currencies; FX inference is forbidden")
    return Money(
        canonical_decimal(sum((item.amount for item in values), Decimal("0"))), currencies.pop()
    )


def reconcile_logical_billing(
    starts: tuple[RunStartProvenance, ...],
    terminations: tuple[ExecutionSegmentTermination, ...],
    provider_billings: tuple[TerminalProviderBillingEvidence, ...],
) -> LogicalBillingOutcome:
    validate_execution_segments(starts)
    if type(terminations) is not tuple or type(provider_billings) is not tuple:
        raise TypeError("terminations and provider billings must be tuples")
    if len(starts) != len(terminations) or len(starts) != len(provider_billings):
        raise ValueError("logical billing inputs must have equal segment counts")
    outcomes = tuple(
        reconcile_segment_billing(start, termination, provider)
        for start, termination, provider in zip(
            starts, terminations, provider_billings, strict=True
        )
    )
    complete = tuple(item for item in outcomes if type(item) is SegmentBillingReconciliation)
    record_ids = tuple(item.provider_billing.billing_record_id for item in complete)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate provider billing_record_id across execution segments")
    estimated_amount = _money_sum(tuple(item.rate_estimate.amount for item in outcomes))
    estimated_duration = canonical_decimal(
        sum(
            (item.rate_estimate.rounded_billable_duration_seconds for item in outcomes),
            Decimal("0"),
        )
    )
    missing = tuple(
        item.execution_segment_id for item in outcomes if type(item) is SegmentBillingMissingCost
    )
    if missing:
        return LogicalBillingMissingCost(
            starts[0].logical_run_id, outcomes, estimated_amount, estimated_duration, missing
        )
    provider_amount = _money_sum(tuple(item.provider_billing.billed_amount for item in complete))
    provider_duration = canonical_decimal(
        sum((item.provider_billing.billed_duration_seconds for item in complete), Decimal("0"))
    )
    return LogicalBillingReconciliation(
        starts[0].logical_run_id,
        complete,
        estimated_amount,
        provider_amount,
        estimated_duration,
        provider_duration,
        canonical_decimal(provider_amount.amount - estimated_amount.amount),
        canonical_decimal(provider_duration - estimated_duration),
    )
