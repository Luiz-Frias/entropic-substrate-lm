"""TDD contract for explicit D1 operator resume segments."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest

import entropic_substrate.d1_operator_controller as controller_module
from d1_billing_finalizer_fixtures import billing_gap, provider_bill
from entropic_substrate.d1_billing_finalization_values import D1BillingFinalization
from entropic_substrate.d1_finalized_cell_custody import D1FinalizedCellCustodyRecord
from entropic_substrate.d1_operator_controller import (
    D1OperatorQueueHalt,
    D1ResumeOperatorHistory,
    D1SequentialOperatorController,
)
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    verify_d1_trainer_export,
)
from entropic_substrate.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.d1_trainer_handoff import (
    D1FreshRunCompletion,
    D1TrainerFailureHandoff,
)
from entropic_substrate.evidence.resource_billing import BillingGap, ExecutionSegmentTermination
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
    execution_segment_id,
)
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.experiments.d1_primitives import RunClass

from .test_d1_finalized_cell_custody_values import _attest, _finalized_case
from .test_d1_operator_controller import (
    _artifact,
    _case,
    _Clock,
    _Custody,
    _manager,
    _Provider,
    _Trainer,
)

pytestmark = pytest.mark.unit


def _running(case: object, tmp_path: Path):
    plan = case.intent.plan
    subject = plan.logical_run_id
    manager = _manager(tmp_path).admit_launch(
        subject_id=subject,
        actor=_actor_operator(),
        artifact=_artifact(50, "prior-admission"),
        event_id="prior-admission",
    )
    manager = manager.start_running(
        subject_id=subject,
        actor=_actor_automation(),
        artifact=_artifact(51, "prior-running"),
        event_id="prior-running",
    )
    identity = D1LaunchAttemptIdentity(
        RunClass.CONFIRMATORY,
        plan.profile.family.family_digest,
        plan.profile.arm,
        plan.seed,
        subject,
        case.design.design_digest,
    )
    ledger = D1LaunchAttemptLedger().record(
        identity, observed_at=case.start.trainer_started_at - timedelta(minutes=1)
    )
    ledger = ledger.transition(
        identity,
        status=D1LaunchAttemptStatus.TRAINER_STARTED,
        observed_at=case.start.trainer_started_at,
    )
    return manager, ledger


def _actor_operator():
    from entropic_substrate.evidence.lifecycle import ActorIdentity, ActorRole

    return ActorIdentity("operator", "operator-session", ActorRole.OPERATOR)


def _actor_automation():
    from entropic_substrate.evidence.lifecycle import ActorIdentity, ActorRole

    return ActorIdentity("automation", "automation-session", ActorRole.AUTOMATION)


@pytest.mark.parametrize(
    "outcome", ["admissible", "billing_gap", "trainer_failure", "start_failure"]
)
def test_resume_uses_next_segment_without_recording_a_second_scientific_look(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    case = _case(tmp_path)
    fresh = cast(D1FreshRunCompletion, case.handoff)
    start0 = cast(RunStartProvenance, case.start)
    prior_end = start0.trainer_started_at + timedelta(seconds=10)
    prior_termination = ExecutionSegmentTermination(
        start0.logical_run_id,
        0,
        start0.execution_segment_id,
        prior_end,
        prior_end + timedelta(seconds=1),
    )
    start1 = replace(
        start0,
        execution_segment_index=1,
        execution_segment_id=execution_segment_id(start0.logical_run_id, 1),
        segment_start_mode=SegmentStartMode.RESUME,
        resume_checkpoint_sha256=fresh.final_checkpoint.sha256,
        pod_created_at=prior_end + timedelta(seconds=2),
        trainer_started_at=prior_end + timedelta(seconds=3),
    )
    intent = replace(
        case.intent,
        execution_segment_index=1,
        resume_checkpoint_sha256=fresh.final_checkpoint.sha256,
    )
    ledger_state = replace(
        fresh.termination.resource_ledger_state,
        execution_segment_index=1,
    )
    termination = replace(
        fresh.termination,
        execution_segment_index=1,
        execution_segment_id=start1.execution_segment_id,
        trainer_ended_at=start1.trainer_started_at + timedelta(seconds=2),
        resource_ledger_state=ledger_state,
    )
    resume_work = replace(
        fresh.final_resource_work,
        execution_segment_index=1,
        execution_segment_id=start1.execution_segment_id,
    )
    resume = D1ResumeRunCompletion(
        termination,
        fresh.final_checkpoint,
        fresh.final_evaluation,
        resume_work,
        fresh.final_checkpoint.sha256,
        start1,
        "a" * 64,
        1,
        None,
    )
    handoff = (
        D1TrainerFailureHandoff(termination, "RuntimeError", "simulated trainer failure")
        if outcome == "trainer_failure"
        else resume
    )
    candidate = D1ExportedTrainerHandoff(handoff, case.journal_path)
    receipt = replace(
        verify_d1_trainer_export(D1ExportedTrainerHandoff(fresh, case.journal_path)),
        handoff=handoff,
    )
    manager, ledger = _running(case, tmp_path)
    history = D1ResumeOperatorHistory(
        (start0,),
        (prior_termination,),
        (provider_bill(start0),),
    )
    events: list[str] = []
    finalized = cast(D1BillingFinalization, object())
    captured: dict[str, object] = {}
    finalized_cell = _attest(_finalized_case(tmp_path / "finalized-cell"))
    custody_captured: dict[str, object] = {}

    def finalize(**kwargs: object) -> D1BillingFinalization:
        captured.update(kwargs)
        return finalized

    monkeypatch.setattr(controller_module, "verify_d1_trainer_export", lambda _: receipt)
    monkeypatch.setattr(controller_module, "finalize_d1_billing", finalize)
    monkeypatch.setattr(
        controller_module,
        "identify_d1_finalized_journal",
        lambda *_args: finalized_cell.journal,
    )

    def attest(**kwargs: object) -> D1FinalizedCellCustodyRecord:
        custody_captured.update(kwargs)
        return finalized_cell

    monkeypatch.setattr(controller_module, "attest_d1_finalized_cell_custody", attest)
    provider = _Provider(start1, events)
    if outcome == "billing_gap":

        def missing_billing(*_args: object, **_kwargs: object) -> BillingGap:
            events.append("billing")
            return billing_gap(start1)

        monkeypatch.setattr(provider, "billing", missing_billing)
    controller = D1SequentialOperatorController(
        design=case.design,
        process_manager=manager,
        attempt_ledger=ledger,
        runpod=provider,
        trainer=_Trainer(
            candidate, events, start1, start_drift="raise" if outcome == "start_failure" else None
        ),
        custody=_Custody(events),
        operator=_actor_operator(),
        automation=_actor_automation(),
        clock=_Clock(start1.trainer_started_at - timedelta(minutes=1)),
    )

    if outcome == "start_failure":
        with pytest.raises(D1OperatorQueueHalt, match="trainer start failed"):
            controller.execute_resume(intent, history=history)
        assert "cancel" in events
        assert controller.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.FAILED
        assert all(
            event.event_kind.value != "resumed" for event in controller.process_manager.events
        )
        return

    result = controller.execute_resume(intent, history=history)

    expected_tail = (
        ["persist_attempt_ledger", "persist_finalized_cell"]
        if outcome == "admissible"
        else ["persist_attempt_ledger", "persist_finalization"]
    )
    assert events == [
        "authenticate",
        "persist_precreate",
        "create",
        "runtime",
        "persist_run_start",
        "stage_inputs",
        "start_trainer",
        "await_handoff",
        "terminate",
        "billing",
        *expected_tail,
    ]
    assert len(result.attempt_ledger.events) == 3
    expected_status = (
        D1LaunchAttemptStatus.FAILED
        if outcome == "trainer_failure"
        else D1LaunchAttemptStatus.COMPLETED
    )
    assert result.attempt_ledger.events[-1].status is expected_status
    assert captured["segment_starts"] == (start0, start1)
    assert captured["segment_terminations"] == (prior_termination, result.termination)
    assert result.queue_admissible is (outcome == "admissible")
    if outcome == "admissible":
        assert custody_captured["segment_starts"] == (start0, start1)
        assert custody_captured["attempt_ledger"] == result.attempt_ledger
        assert result.finalized_cell is finalized_cell
    else:
        assert custody_captured == {}
        assert result.finalized_cell is None
    assert result.process_manager.events[-1].artifacts == (result.finalized_cell_artifact,)
    assert result.process_manager.finalized_subjects == (intent.plan.logical_run_id,)


def test_resume_refuses_nonconsecutive_segment_before_provider_auth(tmp_path: Path) -> None:
    case = _case(tmp_path)
    manager, ledger = _running(case, tmp_path)
    events: list[str] = []
    controller = D1SequentialOperatorController(
        design=case.design,
        process_manager=manager,
        attempt_ledger=ledger,
        runpod=_Provider(case.start, events),
        trainer=_Trainer(
            D1ExportedTrainerHandoff(case.handoff, case.journal_path),
            events,
            case.start,
        ),
        custody=_Custody(events),
        operator=_actor_operator(),
        automation=_actor_automation(),
        clock=_Clock(case.start.trainer_started_at),
    )
    invalid = replace(
        case.intent,
        execution_segment_index=2,
        resume_checkpoint_sha256=case.handoff.final_checkpoint.sha256,
    )
    start0 = cast(RunStartProvenance, case.start)
    ended = start0.trainer_started_at + timedelta(seconds=1)
    history = D1ResumeOperatorHistory(
        (start0,),
        (
            ExecutionSegmentTermination(
                start0.logical_run_id,
                0,
                start0.execution_segment_id,
                ended,
                ended,
            ),
        ),
        (provider_bill(start0),),
    )

    with pytest.raises(ValueError, match="next execution segment"):
        controller.execute_resume(invalid, history=history)
    assert events == []
