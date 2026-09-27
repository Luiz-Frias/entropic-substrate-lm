"""Typed trainer-termination values at the operator-finalization boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from d1_trainer_fixtures import STARTED_AT, micro_plan, run_start
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalCheckpointState
from entropic_substrate.ladder.capture.resource.resource_ledger import ResourceLedgerState
from entropic_substrate.ladder.run.trainer.d1_dual_failure import (
    D1SecondaryFailureEvidence,
    D1SecondaryFailureStage,
)
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import (
    D1TrainerFailureHandoff,
    capture_d1_trainer_termination,
)

pytestmark = pytest.mark.unit


def _states():
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    journal = JournalCheckpointState(
        logical_run_id=plan.logical_run_id,
        execution_digest=plan.run.execution_digest,
        last_event_sequence=3,
        prefix_sha256="a" * 64,
        successful_step=1,
    )
    ledger = ResourceLedgerState(
        logical_run_id=plan.logical_run_id,
        execution_segment_index=0,
        next_eval_index=2,
        last_training_step=1,
        trainer_wall_seconds=2.0,
        committed_quality_source_bytes=4,
        committed_quality_model_tokens=4,
        online_train_token_positions=4,
        target_train_token_positions=5,
        baseline_calibration_positions=160,
        eval_positions=18,
        last_optional_readings=(),
        last_flop_model_identity=None,
        last_energy_sample_count=None,
    )
    return plan, start, journal, ledger


def test_termination_carries_exact_segment_journal_and_ledger_identity() -> None:
    plan, start, journal, ledger = _states()
    ended = STARTED_AT + timedelta(seconds=3)
    termination = capture_d1_trainer_termination(
        start=start,
        trainer_ended_at=ended,
        successful_step=1,
        model_successful_step=1,
        journal_state=journal,
        resource_ledger_state=ledger,
    )

    assert termination.logical_run_id == plan.logical_run_id
    assert termination.execution_segment_index == 0
    assert termination.execution_segment_id == f"{plan.logical_run_id}:segment:0"
    assert termination.trainer_ended_at == ended
    assert termination.model_successful_step == 1
    assert termination.schema_version == "d1-trainer-termination-v2"
    assert termination.journal_state is journal
    assert termination.resource_ledger_state is ledger
    assert termination.manifest["journal_state"] == journal.manifest
    assert termination.manifest["resource_ledger_state"] == ledger.manifest


def test_termination_refuses_clock_identity_and_cursor_drift() -> None:
    _, start, journal, ledger = _states()
    candidates = (
        {"trainer_ended_at": STARTED_AT - timedelta(seconds=1)},
        {"journal_state": replace(journal, logical_run_id="wrong")},
        {"resource_ledger_state": replace(ledger, execution_segment_index=1)},
        {"successful_step": 0},
        {"model_successful_step": 3},
    )
    for overrides in candidates:
        values = {
            "start": start,
            "trainer_ended_at": STARTED_AT + timedelta(seconds=1),
            "successful_step": 1,
            "model_successful_step": 1,
            "journal_state": journal,
            "resource_ledger_state": ledger,
            **overrides,
        }
        with pytest.raises(ValueError):
            capture_d1_trainer_termination(**values)  # type: ignore[arg-type]


def test_failure_handoff_attests_the_durable_trainer_event() -> None:
    _, start, journal, ledger = _states()
    termination = capture_d1_trainer_termination(
        start=start,
        trainer_ended_at=STARTED_AT + timedelta(seconds=1),
        successful_step=1,
        model_successful_step=1,
        journal_state=journal,
        resource_ledger_state=ledger,
    )
    handoff = D1TrainerFailureHandoff.from_error(termination, RuntimeError("boom"))

    assert handoff.trainer_event_kind == "TRAINER_ABORTED"
    assert handoff.manifest["trainer_event_kind"] == "TRAINER_ABORTED"
    assert handoff.manifest["termination"] == termination.manifest
    assert handoff.rollback_claimed is False
    assert handoff.operator_finalization_required is True
    assert handoff.error_type == "RuntimeError"
    assert handoff.error_message == "boom"


def test_termination_refuses_eval_ledger_ahead_of_durable_journal() -> None:
    _, start, journal, ledger = _states()
    with pytest.raises(ValueError, match="durable successful_step"):
        capture_d1_trainer_termination(
            start=start,
            trainer_ended_at=STARTED_AT + timedelta(seconds=1),
            successful_step=1,
            model_successful_step=2,
            journal_state=journal,
            resource_ledger_state=replace(ledger, last_training_step=2),
        )


def test_failure_handoff_refuses_abort_append_as_durable_secondary() -> None:
    _, start, journal, ledger = _states()
    termination = capture_d1_trainer_termination(
        start=start,
        trainer_ended_at=STARTED_AT + timedelta(seconds=1),
        successful_step=1,
        model_successful_step=1,
        journal_state=journal,
        resource_ledger_state=ledger,
    )
    secondary = D1SecondaryFailureEvidence.from_error(
        D1SecondaryFailureStage.TRAINER_ABORT_APPEND,
        RuntimeError("abort append failed"),
    )

    with pytest.raises(ValueError, match="durable TRAINER_ABORTED"):
        D1TrainerFailureHandoff.from_error(
            termination,
            RuntimeError("primary trainer failure"),
            secondary_failure=secondary,
        )
