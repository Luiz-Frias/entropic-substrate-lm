"""D1 continuation restores exact state and preserves discarded hard-crash work."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
import torch

import d1_trainer_fixtures as trainer_fixtures
import entropic_substrate.ladder.run.trainer.d1_fresh_trainer as fresh_module
import entropic_substrate.ladder.run.trainer.d1_resume_runtime as runtime_module
import entropic_substrate.ladder.run.trainer.d1_resume_trainer as resume_module
import entropic_substrate.ladder.run.trainer.d1_segment_engine as segment_engine
import entropic_substrate.ladder.run.trainer.d1_terminal_failure as terminal_failure_module
from d1_resume_assertions import assert_nested_equal
from d1_resume_fixtures import (
    RecordingObserver,
    captured_step_one_case,
    origin,
    resume_start,
)
from d1_resume_hard_crash_fixture import hard_crash_tail_case
from d1_trainer_fixtures import DeterministicTrainerClock
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.capture.records.d1_journal import read_journal
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind
from entropic_substrate.ladder.capture.records.d1_telemetry_query import read_d1_run_evidence
from entropic_substrate.ladder.capture.resource.resource_work_codec import (
    resource_snapshot_to_mapping,
)
from entropic_substrate.ladder.run.trainer.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.ladder.run.trainer.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import (
    D1ResumeRunFailure,
    run_d1_resume,
)
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import D1FreshRunFailure

pytestmark = pytest.mark.unit


class _LaterSegmentClock(DeterministicTrainerClock):
    def utc_now(self) -> datetime:
        return super().utc_now() + timedelta(hours=1)


def test_resume_matches_uninterrupted_and_journals_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    observer = RecordingObserver()
    real_materialize = runtime_module.materialize_d1_model_runtime
    observed: list[object] = []

    def materialize_after_resume(plan: object):
        assert observer.events[-1].event_kind is JournalEventKind.RUN_RESUME
        assert read_journal(case.journal_path)[-1] == observer.events[-1]
        observed.append(plan)
        return real_materialize(plan)  # type: ignore[arg-type]

    monkeypatch.setattr(runtime_module, "materialize_d1_model_runtime", materialize_after_resume)
    completion = run_d1_resume(
        plan=case.plan,
        start=resume_start(case.plan, case.checkpoint_sha256),
        origin=origin(),
        locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
        observer=observer,
        clock=_LaterSegmentClock(),
    )

    assert observed == [case.plan]
    resumed = torch.load(case.checkpoint_path, map_location="cpu", weights_only=True)
    assert case.uninterrupted_terminal is not None
    for field in ("model", "optimizer", "sampler", "rng"):
        assert_nested_equal(resumed[field], case.uninterrupted_terminal[field])
    events = read_journal(case.journal_path)
    assert [event.event_kind for event in events[-4:]] == [
        JournalEventKind.RUN_RESUME,
        JournalEventKind.TRAIN,
        JournalEventKind.EVAL,
        JournalEventKind.CHECKPOINT,
    ]
    assert completion.successful_step == 2
    assert completion.operator_finalization_required is True
    assert completion.termination.execution_segment_index == 1
    assert completion.termination.execution_segment_id.endswith(":segment:1")
    assert completion.termination.trainer_ended_at == origin().started_at + timedelta(
        hours=1, seconds=2
    )
    assert completion.termination.journal_state == completion.journal_state
    assert completion.termination.resource_ledger_state.last_training_step == 2
    admitted = read_d1_run_evidence(
        case.journal_path,
        expected_plan_digest=case.plan.plan_digest,
        allow_incomplete=True,
    )
    assert [item.successful_step for item in admitted.train_steps] == [1, 2]
    observed = admitted.evals[-1].quality
    assert (observed.state_moments, observed.target_moments, observed.predicted_moments) == (
        completion.final_evaluation.state_moments,
        completion.final_evaluation.target_moments,
        completion.final_evaluation.predicted_moments,
    )
    assert {item.quality.latent_width for item in admitted.evals} == {
        completion.final_evaluation.state_moments.width
    }
    resume_event = events[-4]
    assert resume_event.payload["run_start_provenance"] == run_start_to_mapping(
        completion.run_start_provenance
    )
    eval_event = events[-2]
    eval_payload = cast(dict[str, object], eval_event.manifest["payload"])
    assert set(eval_payload) == {"eval_index", "quality", "resource_work"}
    assert eval_payload["eval_index"] == completion.final_resource_work.eval_index
    assert eval_payload["quality"] == completion.final_evaluation.telemetry_manifest
    assert eval_payload["resource_work"] == resource_snapshot_to_mapping(
        completion.final_resource_work
    )
    assert all(event.event_kind is not JournalEventKind.RUN_END for event in events)


def test_resume_completion_refuses_lineage_and_finalization_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    completion = run_d1_resume(
        plan=case.plan,
        start=resume_start(case.plan, case.checkpoint_sha256),
        origin=origin(),
        locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
        observer=RecordingObserver(),
        clock=DeterministicTrainerClock(),
    )

    for field, value in (
        ("resumed_from_sha256", "0" * 64),
        ("segment_data_order_sha256", "not-a-digest"),
        ("segment_data_order_batches_attested", 0),
        ("tail_quarantine", object()),
        ("operator_finalization_required", False),
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(completion, **{field: value})


def test_hard_crash_tail_is_quarantined_and_counted_as_work_not_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = hard_crash_tail_case(tmp_path, monkeypatch)
    original = case.journal_path.read_bytes()
    quarantine = tmp_path / "crashed-tail.jsonl"
    completion = run_d1_resume(
        plan=case.plan,
        start=resume_start(case.plan, case.checkpoint_sha256),
        origin=origin(),
        locations=D1ResumeRunLocations(
            case.journal_path,
            case.checkpoint_path,
            quarantine_path=quarantine,
        ),
        observer=RecordingObserver(),
        clock=DeterministicTrainerClock(),
    )

    assert quarantine.read_bytes() == original
    assert completion.tail_quarantine is not None
    assert [event.event_kind for event in completion.tail_quarantine.tail_events] == [
        JournalEventKind.TRAIN,
        JournalEventKind.EVAL,
    ]
    work = completion.final_resource_work
    assert work.committed_quality_source_bytes == 8
    assert work.online_train_token_positions == 12
    assert work.target_train_token_positions == 15
    assert work.eval_positions == 36


_FAILURE_PAYLOAD_KEYS = sorted(
    (
        "error_message error_type execution_segment_id execution_segment_index journal_cursor "
        "logical_run_id model_successful_step operator_finalization_required "
        "resource_ledger_state rollback_claimed secondary_failure trainer_ended_at"
    ).split()
)


class _TrainSentinel(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("mode", "expected_failure"),
    [("fresh", D1FreshRunFailure), ("resume", D1ResumeRunFailure)],
)
def test_both_trainers_share_one_failure_helper_and_one_abort_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_failure: type[RuntimeError],
) -> None:
    shared = terminal_failure_module._trainer_failure  # type: ignore[attr-defined]
    assert fresh_module._trainer_failure is shared  # type: ignore[attr-defined]
    assert resume_module._trainer_failure is shared  # type: ignore[attr-defined]
    case = captured_step_one_case(tmp_path, monkeypatch) if mode == "resume" else None
    primary = _TrainSentinel("shared failure helper sentinel")

    def fail_step(**_kwargs: object) -> object:
        raise primary

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    if case is None:
        plan = trainer_fixtures.micro_plan(D1Arm.BOTH)
        journal_path, checkpoint_path = trainer_fixtures.output_paths(tmp_path, D1Arm.BOTH)
        with pytest.raises(expected_failure) as caught:
            run_d1_fresh(
                plan=plan,
                start=trainer_fixtures.run_start(plan),
                origin=origin(),
                locations=D1FreshRunLocations(journal_path, checkpoint_path),
                clock=DeterministicTrainerClock(),
            )
    else:
        plan, journal_path = case.plan, case.journal_path
        with pytest.raises(expected_failure) as caught:
            run_d1_resume(
                plan=plan,
                start=resume_start(plan, case.checkpoint_sha256),
                origin=origin(),
                locations=D1ResumeRunLocations(journal_path, case.checkpoint_path),
                observer=RecordingObserver(),
                clock=DeterministicTrainerClock(),
            )

    last = read_journal(journal_path)[-1]
    assert last.event_kind is JournalEventKind.TRAINER_ABORTED
    payload = cast(dict[str, object], last.manifest["payload"])
    assert sorted(payload) == _FAILURE_PAYLOAD_KEYS
    assert payload["error_type"] == "_TrainSentinel"
    assert payload["error_message"] == "shared failure helper sentinel"
    assert payload["logical_run_id"] == plan.logical_run_id
    assert payload["model_successful_step"] == last.successful_step
    assert payload["rollback_claimed"] is False
    assert payload["operator_finalization_required"] is True
    assert payload["secondary_failure"] is None
    assert caught.value.__cause__ is primary
    assert caught.value.handoff.termination.journal_state.last_event_sequence == last.event_seq  # type: ignore[attr-defined]
