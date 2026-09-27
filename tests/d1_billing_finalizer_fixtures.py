"""Shared real-trainer fixtures for D1 operator-finalization tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    run_start,
)
from entropic_substrate.kernel.evidence_class.resource_provenance import (
    BillingUnit,
    EvidenceGap,
    Money,
    RunStartProvenance,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, EvidenceClass
from entropic_substrate.ladder.capture.resource.resource_billing import (
    BillingGap,
    ExecutionSegmentTermination,
    ProviderBilledAmountObservation,
    ProviderBilling,
    ProviderBillingPartial,
)
from entropic_substrate.ladder.register.d1_plan import D1RunPlan
from entropic_substrate.ladder.run.trainer.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import D1FreshRunCompletion


@dataclass(frozen=True, slots=True)
class CompletedD1Case:
    plan: D1RunPlan
    start: RunStartProvenance
    journal_path: Path
    completion: D1FreshRunCompletion
    termination: ExecutionSegmentTermination


def completed_case(tmp_path: Path, arm: D1Arm = D1Arm.NONE) -> CompletedD1Case:
    plan = micro_plan(arm)
    start = run_start(plan)
    journal_path = tmp_path / f"{arm.value}.jsonl"
    completion = run_d1_fresh(
        plan=plan,
        start=start,
        origin=D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS),
        locations=D1FreshRunLocations(journal_path, tmp_path / f"{arm.value}.pt"),
        clock=DeterministicTrainerClock(),
    )
    ended = completion.termination.trainer_ended_at
    termination = ExecutionSegmentTermination(
        start.logical_run_id,
        start.execution_segment_index,
        start.execution_segment_id,
        ended,
        ended + timedelta(seconds=1),
    )
    return CompletedD1Case(plan, start, journal_path, completion, termination)


def provider_bill(
    start: RunStartProvenance,
    *,
    amount: str = "0",
    currency: str = "USD",
    record: str | None = None,
) -> ProviderBilling:
    return ProviderBilling(
        start.logical_run_id,
        start.execution_segment_index,
        start.execution_segment_id,
        Decimal("65"),
        Money(Decimal(amount), currency),
        BillingUnit.HOUR,
        record or f"provider-{start.execution_segment_index}",
        EvidenceClass.PROVIDER_BILLED,
    )


def billing_gap(start: RunStartProvenance) -> BillingGap:
    reason = "provider export is not available"
    return BillingGap(
        start.logical_run_id,
        start.execution_segment_index,
        start.execution_segment_id,
        EvidenceGap("provider_billed_duration_seconds", reason),
        EvidenceGap("provider_billed_amount", reason),
        EvidenceGap("provider_billing_unit", reason),
        EvidenceGap("provider_billing_record_id", reason),
    )


def partial_provider_bill(start: RunStartProvenance) -> ProviderBillingPartial:
    payload = b'[{"amount":0.25,"podId":"pod-123","timeBilledMs":24726}]'
    reason = "RunPod billing rows omit this field"
    return ProviderBillingPartial(
        start.logical_run_id,
        start.execution_segment_index,
        start.execution_segment_id,
        Decimal("24.726"),
        ProviderBilledAmountObservation(
            Decimal("0.25"),
            EvidenceGap("provider_billed_currency", reason),
        ),
        EvidenceGap("provider_billing_unit", reason),
        EvidenceGap("provider_billing_record_id", reason),
        EvidenceGap("provider_billing_status", reason),
        "2026-08-26 17:00:00",
        sha256(payload).hexdigest(),
        start.trainer_started_at + timedelta(minutes=2),
        EvidenceClass.PROVIDER_BILLED,
    )
