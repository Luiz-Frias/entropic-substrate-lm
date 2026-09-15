"""End-to-end D1 fresh-run trainer evidence and failure contracts."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
import torch

import entropic_substrate.d1_segment_engine as segment_engine
import entropic_substrate.telemetry as telemetry_api
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    output_paths,
    run_start,
)
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_trainer_handoff import D1FreshRunFailure
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.d1_journal import AuthoritativeRunJournal, read_journal
from entropic_substrate.evidence.d1_journal_values import (
    JournalCheckpointState,
    JournalEventKind,
    canonical_event_line,
)
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.evidence.resource_work_codec import resource_snapshot_to_mapping
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


def _run(tmp_path: Path, arm: D1Arm):
    plan = micro_plan(arm)
    journal, checkpoint = output_paths(tmp_path, arm)
    completion = run_d1_fresh(
        plan=plan,
        start=run_start(plan),
        origin=D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS),
        locations=D1FreshRunLocations(journal_path=journal, checkpoint_path=checkpoint),
        clock=DeterministicTrainerClock(),
    )
    return plan, completion, read_journal(journal)


@pytest.mark.parametrize("arm", list(D1Arm))
def test_four_arm_fresh_run_has_exact_evidence_order_and_final_receipt(
    tmp_path: Path, arm: D1Arm
) -> None:
    plan, completion, events = _run(tmp_path, arm)
    kinds = [event.event_kind for event in events]
    assert [kind.value for kind in kinds] == (
        "run_start baseline_calibrated eval train eval checkpoint train eval checkpoint".split()
    )
    assert [event.successful_step for event in events] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert completion.operator_finalization_required is True
    assert completion.journal_state.last_event_sequence == events[-1].event_seq
    assert completion.termination.logical_run_id == plan.logical_run_id
    assert completion.termination.execution_segment_index == 0
    assert completion.termination.execution_segment_id == f"{plan.logical_run_id}:segment:0"
    assert completion.termination.trainer_ended_at == STARTED_AT + timedelta(seconds=4)
    assert completion.termination.resource_ledger_state.last_training_step == 2
    assert not {JournalEventKind.RUN_END, JournalEventKind.RUN_ABORT} & set(kinds)
    start_payload = cast(dict[str, object], events[0].manifest["payload"])
    assert start_payload["run_start_provenance"] == run_start_to_mapping(run_start(plan))
    receipt = completion.final_checkpoint
    assert receipt.size_bytes == receipt.path.stat().st_size
    assert receipt.sha256 == hashlib.sha256(receipt.path.read_bytes()).hexdigest()
    assert events[-1].payload["checkpoint_sha256"] == receipt.sha256
    eval_events = [event for event in events if event.event_kind is JournalEventKind.EVAL]
    assert [event.successful_step for event in eval_events] == [0, 1, 2]
    final_payload = cast(dict[str, object], eval_events[-1].manifest["payload"])
    assert set(final_payload) == {"eval_index", "quality", "resource_work"}
    assert final_payload["eval_index"] == completion.final_resource_work.eval_index
    assert final_payload["quality"] == completion.final_evaluation.telemetry_manifest
    journal, _ = output_paths(tmp_path, arm)
    readback = telemetry_api.read_d1_run_evidence(journal, allow_incomplete=True)
    observed = readback.evals[-1].quality
    assert readback.status.value == "partial"
    assert readback.evals[-1].eval_index == completion.final_resource_work.eval_index
    assert (observed.state_moments, observed.target_moments, observed.predicted_moments) == (
        completion.final_evaluation.state_moments,
        completion.final_evaluation.target_moments,
        completion.final_evaluation.predicted_moments,
    )
    assert final_payload["resource_work"] == resource_snapshot_to_mapping(
        completion.final_resource_work
    )
    work = cast(dict[str, object], final_payload["resource_work"])
    batching = plan.profile.draft.execution_protocol.batching
    assert (
        work["committed_quality_source_bytes"] == 2 * batching.batch_size * batching.sequence_length
    )
    assert work["committed_quality_model_tokens"] == work["committed_quality_source_bytes"]
    assert work["online_train_token_positions"] == work["committed_quality_model_tokens"]
    assert work["target_train_token_positions"] == (
        2 * batching.batch_size * (batching.sequence_length + batching.horizon)
    )
    state = torch.load(receipt.path, map_location="cpu", weights_only=True)
    model = cast(dict[str, object], state["model"])
    extra = cast(dict[str, object], model["_extra_state"])
    policy = cast(dict[str, object], extra["target_policy"])
    if arm in (D1Arm.SG, D1Arm.BOTH):
        assert policy["update_cursor"] == 2
    else:
        assert "update_cursor" not in policy


def test_paired_arms_retain_identical_observed_data_order(tmp_path: Path) -> None:
    observed: dict[D1Arm, tuple[str, int]] = {}
    for arm in D1Arm:
        _, _, events = _run(tmp_path, arm)
        final_train = [event for event in events if event.event_kind is JournalEventKind.TRAIN][-1]
        order = cast(dict[str, object], final_train.payload["data_order"])
        observed[arm] = cast(str, order["sha256"]), cast(int, order["batches_attested"])
    assert len(set(observed.values())) == 1 and next(iter(observed.values()))[1] == 2


def test_failed_step_returns_nonterminal_typed_handoff_and_chains_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = micro_plan(D1Arm.BOTH)
    journal, checkpoint = output_paths(tmp_path, D1Arm.BOTH)
    real_step = segment_engine.apply_d1_training_step
    calls = 0

    def fail_step(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate train defect")
        return real_step(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    with pytest.raises(D1FreshRunFailure, match="deliberate train defect") as captured:
        run_d1_fresh(
            plan=plan,
            start=run_start(plan),
            origin=D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS),
            locations=D1FreshRunLocations(journal_path=journal, checkpoint_path=checkpoint),
            clock=DeterministicTrainerClock(),
        )
    events = read_journal(journal)
    assert events[-1].event_kind is JournalEventKind.TRAINER_ABORTED
    assert events[-1].successful_step == 1
    assert all(
        event.event_kind not in {JournalEventKind.RUN_ABORT, JournalEventKind.RUN_END}
        for event in events
    )
    failure = captured.value.handoff
    assert type(captured.value.__cause__) is RuntimeError
    assert str(captured.value.__cause__) == "deliberate train defect"
    assert failure.termination.trainer_ended_at == STARTED_AT + timedelta(seconds=3)
    assert failure.termination.journal_state.last_event_sequence == events[-1].event_seq
    payload = cast(dict[str, object], events[-1].manifest["payload"])
    assert (
        sorted(payload)
        == (
            "error_message error_type execution_segment_id execution_segment_index journal_cursor "
            "logical_run_id model_successful_step operator_finalization_required "
            "resource_ledger_state rollback_claimed secondary_failure trainer_ended_at"
        ).split()
    )
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_message"] == "deliberate train defect"
    assert payload["logical_run_id"] == plan.logical_run_id
    assert payload["model_successful_step"] == 1
    assert payload["execution_segment_index"] == 0
    assert payload["execution_segment_id"] == f"{plan.logical_run_id}:segment:0"
    prefix = b"".join(canonical_event_line(event) for event in events[:-1])
    expected_cursor = JournalCheckpointState(
        logical_run_id=plan.logical_run_id,
        execution_digest=plan.run.execution_digest,
        last_event_sequence=events[-2].event_seq,
        prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        successful_step=1,
    )
    assert payload["journal_cursor"] == expected_cursor.manifest
    assert payload["resource_ledger_state"] == failure.termination.resource_ledger_state.manifest
    assert payload["trainer_ended_at"] == "2026-08-23T20:01:03Z"
    assert payload["rollback_claimed"] is False
    assert payload["operator_finalization_required"] is True
    assert payload["secondary_failure"] is None
    assert failure.secondary_failure is None
    ledger = failure.termination.resource_ledger_state
    batching = plan.profile.draft.execution_protocol.batching
    assert ledger.committed_quality_source_bytes == batching.batch_size * batching.sequence_length
    assert ledger.online_train_token_positions == 2 * batching.batch_size * batching.sequence_length
    assert ledger.target_train_token_positions == (
        2 * batching.batch_size * (batching.sequence_length + batching.horizon)
    )
    assert checkpoint.exists()


def test_authoritative_journal_construction_is_injectable_without_observer_coupling(
    tmp_path: Path,
) -> None:
    plan = micro_plan(D1Arm.NONE)
    journal, checkpoint = output_paths(tmp_path, D1Arm.NONE)
    opened: list[tuple[Path, JournalEventKind]] = []

    def journal_factory(path: Path, start_event: object) -> AuthoritativeRunJournal:
        opened.append((path, start_event.event_kind))  # type: ignore[attr-defined]
        return AuthoritativeRunJournal.fresh(path, start_event)  # type: ignore[arg-type]

    run_d1_fresh(
        plan=plan,
        start=run_start(plan),
        origin=D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS),
        locations=D1FreshRunLocations(journal_path=journal, checkpoint_path=checkpoint),
        clock=DeterministicTrainerClock(),
        journal_factory=journal_factory,
    )
    assert opened == [(journal, JournalEventKind.RUN_START)]
