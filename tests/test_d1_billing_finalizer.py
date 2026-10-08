"""Operator billing finalization closes D1 only after exact reconciliation."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import entropic_substrate.ladder.run.trainer.d1_segment_engine as segment_engine
from d1_billing_finalizer_fixtures import (
    CompletedD1Case,
    billing_gap,
    completed_case,
    partial_provider_bill,
    provider_bill,
)
from d1_resume_fixtures import RecordingObserver, origin, resume_start
from d1_resume_hard_crash_fixture import hard_crash_tail_case
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    run_start,
)
from entropic_substrate.kernel.evidence_class.resource_provenance import (
    Money,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.capture.records.d1_journal import (
    AuthoritativeRunJournal,
    read_journal,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind
from entropic_substrate.ladder.capture.resource.resource_billing import (
    ExecutionSegmentTermination,
    TerminalProviderBillingEvidence,
)
from entropic_substrate.ladder.capture.resource.resource_billing_reconciliation import (
    LogicalBillingMissingCost,
    LogicalBillingReconciliation,
)
from entropic_substrate.ladder.run.closure.d1_billing_finalization_values import (
    D1BillingFinalizationRequest,
)
from entropic_substrate.ladder.run.closure.d1_billing_finalizer import finalize_d1_billing
from entropic_substrate.ladder.run.trainer.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.ladder.run.trainer.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import run_d1_resume
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import (
    D1FreshRunFailure,
)

pytestmark = pytest.mark.unit


def _finalize(
    case: CompletedD1Case,
    provider: TerminalProviderBillingEvidence,
):
    return finalize_d1_billing(
        D1BillingFinalizationRequest(
            plan=case.plan,
            segment_starts=(case.start,),
            segment_terminations=(case.termination,),
            provider_billings=(provider,),
            handoff=case.completion,
            journal_path=case.journal_path,
            expected_journal_state=case.completion.journal_state,
        )
    )


def test_success_reconciles_then_appends_one_run_end_with_export_evidence(
    tmp_path: Path,
) -> None:
    case = completed_case(tmp_path)

    finalized = _finalize(case, provider_bill(case.start))

    assert type(finalized.billing) is LogicalBillingReconciliation
    assert finalized.quality_export_ready is True
    assert finalized.trainer_outcome == "completed"
    events = read_journal(case.journal_path)
    assert events[-1].event_kind is JournalEventKind.RUN_END
    assert sum(event.event_kind is JournalEventKind.RUN_END for event in events) == 1
    assert JournalEventKind.RUN_RESUME not in {event.event_kind for event in events}
    assert finalized.journal_state.last_event_sequence == events[-1].event_seq
    payload = cast(dict[str, object], events[-1].manifest["payload"])
    assert set(payload) == {
        "billing",
        "checkpoint_export",
        "provider_billing_complete",
        "quality_export_ready",
        "schema_version",
        "terminal_resource_evidence",
        "trainer_handoff",
        "trainer_outcome",
    }
    billing = cast(dict[str, object], payload["billing"])
    segment = cast(dict[str, object], cast(list[object], billing["segments"])[0])
    estimate = cast(dict[str, object], segment["rate_estimate"])
    quoted = cast(dict[str, object], estimate["quoted_rate"])
    provider = cast(dict[str, object], segment["provider_billing"])
    assert quoted["billing_unit"] == "second"
    assert provider["billing_unit"] == "hour"
    assert provider["evidence_class"] == "provider_billed"
    checkpoint = cast(dict[str, object], payload["checkpoint_export"])
    assert checkpoint["kind"] == "verified_terminal_checkpoint"
    assert checkpoint["sha256"] == case.completion.final_checkpoint.sha256
    assert checkpoint["size_bytes"] == case.completion.final_checkpoint.size_bytes
    resources = cast(dict[str, object], payload["terminal_resource_evidence"])
    assert resources["non_quality_online_train_token_positions"] == 0
    assert resources["non_quality_counter_evidence"] == "derived_from_measured"


def test_provider_gap_stays_missing_without_blocking_quality_export(tmp_path: Path) -> None:
    case = completed_case(tmp_path)

    finalized = _finalize(case, billing_gap(case.start))

    assert type(finalized.billing) is LogicalBillingMissingCost
    assert finalized.billing.provider_billed_amount is None
    assert finalized.quality_export_ready is True
    payload = cast(dict[str, object], read_journal(case.journal_path)[-1].manifest["payload"])
    billing = cast(dict[str, object], payload["billing"])
    assert billing["provider_billed_amount"] is None
    assert billing["provider_billed_duration_seconds"] is None
    assert payload["provider_billing_complete"] is False
    segment = cast(dict[str, object], cast(list[object], billing["segments"])[0])
    gap = cast(dict[str, object], segment["provider_evidence"])
    assert cast(dict[str, object], gap["provider_billing_unit"])["kind"] == "gap"


def test_partial_provider_rows_survive_without_cost_reconciliation(tmp_path: Path) -> None:
    case = completed_case(tmp_path)

    finalized = _finalize(case, partial_provider_bill(case.start))

    assert type(finalized.billing) is LogicalBillingMissingCost
    assert finalized.quality_export_ready is True
    partial = cast(dict[str, object], finalized.billing.segments[0].manifest["provider_evidence"])
    assert partial["kind"] == "partial_provider_billing"
    assert partial["billed_duration_seconds"] == "24.726"
    assert partial["billed_amount"] == {
        "amount": "0.25",
        "currency": {
            "field": "provider_billed_currency",
            "kind": "gap",
            "reason": "RunPod billing rows omit this field",
        },
    }


def test_failed_trainer_finalizes_billing_but_never_claims_quality_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    journal_path = tmp_path / "failed.jsonl"
    real_step = segment_engine.apply_d1_training_step
    calls = 0

    def fail_second_step(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("operator-finalized defect")
        return real_step(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_second_step)
    with pytest.raises(D1FreshRunFailure) as caught:
        run_d1_fresh(
            plan=plan,
            start=start,
            origin=D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS),
            locations=D1FreshRunLocations(journal_path, tmp_path / "failed.pt"),
            clock=DeterministicTrainerClock(),
        )
    handoff = caught.value.handoff
    ended = handoff.termination.trainer_ended_at
    termination = ExecutionSegmentTermination(
        start.logical_run_id,
        0,
        start.execution_segment_id,
        ended,
        ended + timedelta(seconds=1),
    )

    finalized = finalize_d1_billing(
        D1BillingFinalizationRequest(
            plan=plan,
            segment_starts=(start,),
            segment_terminations=(termination,),
            provider_billings=(provider_bill(start),),
            handoff=handoff,
            journal_path=journal_path,
            expected_journal_state=handoff.termination.journal_state,
        )
    )

    assert finalized.trainer_outcome == "failed"
    assert finalized.quality_export_ready is False
    payload = cast(dict[str, object], read_journal(journal_path)[-1].manifest["payload"])
    checkpoint = cast(dict[str, object], payload["checkpoint_export"])
    assert checkpoint == {
        "field": "terminal_checkpoint",
        "kind": "gap",
        "reason": "trainer failure has no successful terminal checkpoint handoff",
    }
    resources = cast(dict[str, object], payload["terminal_resource_evidence"])
    assert resources["non_quality_online_train_token_positions"] == 4
    handoff_payload = cast(dict[str, object], payload["trainer_handoff"])
    assert handoff_payload["error_type"] == "RuntimeError"
    assert handoff_payload["error_message"] == "operator-finalized defect"
    assert [event.event_kind for event in read_journal(journal_path)[-2:]] == [
        JournalEventKind.TRAINER_ABORTED,
        JournalEventKind.RUN_END,
    ]


def test_resumed_completion_preserves_discarded_tail_work_and_segment_billing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = hard_crash_tail_case(tmp_path, monkeypatch)
    first = run_start(case.plan)
    second = resume_start(case.plan, case.checkpoint_sha256)
    quarantine = tmp_path / "discarded-tail.jsonl"
    completion = run_d1_resume(
        plan=case.plan,
        start=second,
        origin=origin(),
        locations=D1ResumeRunLocations(
            case.journal_path,
            case.checkpoint_path,
            quarantine_path=quarantine,
        ),
        observer=RecordingObserver(),
        clock=DeterministicTrainerClock(),
    )
    final_end = completion.termination.trainer_ended_at
    terminations = (
        ExecutionSegmentTermination(
            first.logical_run_id,
            0,
            first.execution_segment_id,
            STARTED_AT + timedelta(seconds=1),
            STARTED_AT + timedelta(seconds=1),
        ),
        ExecutionSegmentTermination(
            second.logical_run_id,
            1,
            second.execution_segment_id,
            final_end,
            final_end + timedelta(seconds=1),
        ),
    )

    finalized = finalize_d1_billing(
        D1BillingFinalizationRequest(
            plan=case.plan,
            segment_starts=(first, second),
            segment_terminations=terminations,
            provider_billings=(
                provider_bill(first, record="first"),
                provider_bill(second, record="second"),
            ),
            handoff=completion,
            journal_path=case.journal_path,
            expected_journal_state=completion.journal_state,
        )
    )

    assert type(finalized.billing) is LogicalBillingReconciliation
    assert len(finalized.billing.segments) == 2
    assert finalized.billing.provider_billed_amount == Money(Decimal("0"), "USD")
    payload = cast(dict[str, object], read_journal(case.journal_path)[-1].manifest["payload"])
    resources = cast(dict[str, object], payload["terminal_resource_evidence"])
    assert resources["non_quality_online_train_token_positions"] == 4
    quarantine_evidence = cast(dict[str, object], resources["tail_quarantine"])
    assert quarantine_evidence["tail_event_count"] == 2
    assert quarantine_evidence["quarantined_journal_sha256"] == (
        completion.tail_quarantine.quarantined_journal_sha256  # type: ignore[union-attr]
    )


def test_currency_mismatch_never_appends(tmp_path: Path) -> None:
    case = completed_case(tmp_path)
    before = case.journal_path.read_bytes()

    with pytest.raises(ValueError):
        finalize_d1_billing(
            D1BillingFinalizationRequest(
                plan=case.plan,
                segment_starts=(case.start,),
                segment_terminations=(case.termination,),
                provider_billings=(provider_bill(case.start, currency="EUR"),),
                handoff=case.completion,
                journal_path=case.journal_path,
                expected_journal_state=case.completion.journal_state,
            )
        )

    assert case.journal_path.read_bytes() == before


def test_tampered_terminal_checkpoint_refuses_export_before_journal_append(
    tmp_path: Path,
) -> None:
    case = completed_case(tmp_path)
    case.completion.final_checkpoint.path.write_bytes(b"not the receipted checkpoint")
    before = case.journal_path.read_bytes()

    with pytest.raises(Exception, match="checkpoint"):
        _finalize(case, provider_bill(case.start))

    assert case.journal_path.read_bytes() == before


def test_request_without_a_typed_plan_is_refused_before_any_journal_read(
    tmp_path: Path,
) -> None:
    case = completed_case(tmp_path)
    before = case.journal_path.read_bytes()

    with pytest.raises(TypeError, match="typed run plan"):
        finalize_d1_billing(
            D1BillingFinalizationRequest(
                plan=cast(Any, None),
                segment_starts=(case.start,),
                segment_terminations=(case.termination,),
                provider_billings=(provider_bill(case.start),),
                handoff=case.completion,
                journal_path=case.journal_path,
                expected_journal_state=case.completion.journal_state,
            )
        )

    assert case.journal_path.read_bytes() == before


def test_append_failure_never_returns_finalized_or_leaves_run_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = completed_case(tmp_path)
    before = case.journal_path.read_bytes()

    def reject_append(self: AuthoritativeRunJournal, event: object) -> object:
        raise RuntimeError("durable RUN_END append failed")

    monkeypatch.setattr(AuthoritativeRunJournal, "append", reject_append)

    with pytest.raises(RuntimeError, match="durable RUN_END"):
        _finalize(case, provider_bill(case.start))

    assert case.journal_path.read_bytes() == before
    assert all(
        event.event_kind is not JournalEventKind.RUN_END
        for event in read_journal(case.journal_path)
    )
