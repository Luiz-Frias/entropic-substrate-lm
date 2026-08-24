"""Typed trainer-termination values at the operator-finalization boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from d1_trainer_fixtures import STARTED_AT, micro_plan, run_start
from entropic_substrate.d1_trainer_handoff import (
    D1TrainerFailureHandoff,
    capture_d1_trainer_termination,
)
from entropic_substrate.evidence.d1_journal_values import JournalCheckpointState
from entropic_substrate.evidence.resource_ledger import ResourceLedgerState
from entropic_substrate.experiments.d1_primitives import D1Arm

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
        journal_state=journal,
        resource_ledger_state=ledger,
    )

    assert termination.logical_run_id == plan.logical_run_id
    assert termination.execution_segment_index == 0
    assert termination.execution_segment_id == f"{plan.logical_run_id}:segment:0"
    assert termination.trainer_ended_at == ended
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
    )
    for overrides in candidates:
        values = {
            "start": start,
            "trainer_ended_at": STARTED_AT + timedelta(seconds=1),
            "successful_step": 1,
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
