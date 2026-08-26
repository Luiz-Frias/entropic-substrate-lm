"""D1 continuation restores exact state and preserves discarded hard-crash work."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
import torch

import entropic_substrate.d1_resume_runtime as runtime_module
from d1_resume_assertions import assert_nested_equal
from d1_resume_fixtures import (
    RecordingObserver,
    captured_step_one_case,
    origin,
    resume_start,
)
from d1_resume_hard_crash_fixture import hard_crash_tail_case
from d1_trainer_fixtures import DeterministicTrainerClock
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.d1_resume_trainer import run_d1_resume
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.evidence.resource_work_codec import resource_snapshot_to_mapping

pytestmark = pytest.mark.unit


def test_resume_matches_uninterrupted_and_journals_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    observer = RecordingObserver()
    real_materialize = runtime_module._materialize  # pyright: ignore[reportPrivateUsage]

    def materialize_after_resume(plan: object):
        assert observer.events[-1].event_kind is JournalEventKind.RUN_RESUME
        assert read_journal(case.journal_path)[-1] == observer.events[-1]
        return real_materialize(plan)  # type: ignore[arg-type]

    monkeypatch.setattr(runtime_module, "_materialize", materialize_after_resume)
    completion = run_d1_resume(
        plan=case.plan,
        start=resume_start(case.plan, case.checkpoint_sha256),
        origin=origin(),
        locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
        observer=observer,
        clock=DeterministicTrainerClock(),
    )

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
    assert completion.termination.trainer_ended_at == origin().started_at + timedelta(seconds=2)
    assert completion.termination.journal_state == completion.journal_state
    assert completion.termination.resource_ledger_state.last_training_step == 2
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
