"""Projection of authoritative D1 snapshots into bounded operator read models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.kernel.identity.d1_primitives import D1Arm, canonical_digest
from entropic_substrate.ladder.analyze.d1_inspect_projection import compose_d1_inspect_view
from entropic_substrate.ladder.analyze.d1_inspect_values import (
    D1_INSPECT_MAX_EVALUATIONS,
    D1InspectComplete,
    D1InspectEventRef,
    D1InspectPartial,
    D1InspectRunAborted,
    D1InspectTrainerAborted,
)
from entropic_substrate.ladder.capture.reader.d1_journal_snapshot import (
    D1JournalSnapshot,
    read_d1_local_journal_snapshot,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import (
    D1_REPORT_ONLY_METRICS,
    D1_SCALAR_METRICS,
    D1RunStatus,
)

from .d1_telemetry_fixtures import valid_events, write_events

pytestmark = pytest.mark.unit


def _snapshot(
    tmp_path: Path,
    *,
    events: list[D1JournalEvent] | None = None,
    allow_incomplete: bool = False,
) -> D1JournalSnapshot:
    path = tmp_path / "operator" / "run.jsonl"
    path.parent.mkdir(parents=True)
    write_events(path, valid_events() if events is None else events)
    return read_d1_local_journal_snapshot(
        path,
        max_bytes=path.stat().st_size,
        allow_incomplete=allow_incomplete,
    )


def test_complete_view_binds_snapshot_identity_plan_progress_and_completion(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)

    view = compose_d1_inspect_view(snapshot, evaluation_limit=2)

    assert view.status is D1RunStatus.COMPLETE
    assert type(view.state) is D1InspectComplete
    assert (view.journal.path, view.journal.relative_path) == (
        str(snapshot.path),
        snapshot.relative_path,
    )
    assert (view.journal.byte_count, view.journal.sha256) == (
        snapshot.byte_count,
        snapshot.sha256,
    )
    assert view.run.logical_run_id == snapshot.run.identity.logical_run_id
    assert view.run.authority_digest == snapshot.run.authority_digest
    assert view.run.plan_digest == snapshot.run.plan_digest
    assert (view.plan.profile_name, view.plan.arm, view.plan.seed) == (
        "d1-both",
        D1Arm.BOTH,
        7,
    )
    assert (
        view.progress.segment_count,
        view.progress.data_order_segment_count,
        view.progress.tail_quarantine_count,
        view.progress.baseline_present,
        view.progress.total_evaluation_count,
        view.progress.total_train_step_count,
        view.progress.total_checkpoint_count,
    ) == (1, 1, 0, True, 3, 2, 2)
    assert view.progress.latest_quality is not None
    assert view.progress.latest_checkpoint is not None
    assert view.progress.latest_quality.step == 2
    assert view.progress.latest_checkpoint.successful_step == 2
    assert view.state.completion.final_quality is view.progress.latest_quality
    assert view.state.completion.final_checkpoint is view.progress.latest_checkpoint
    assert snapshot.run.terminal is not None
    assert view.state.completion.run_end == D1InspectEventRef(
        snapshot.run.terminal.event_kind,
        snapshot.run.terminal.event_seq,
        snapshot.run.terminal.successful_step,
        canonical_digest(snapshot.run.terminal.manifest),
    )


def test_projection_uses_only_admitted_snapshot_after_source_path_is_removed(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    snapshot.path.unlink()

    view = compose_d1_inspect_view(snapshot, evaluation_limit=1)

    assert view.run.authority_digest == snapshot.run.authority_digest
    assert view.evaluations.items[-1].step == snapshot.run.evals[-1].step


def test_evaluation_window_is_exact_bounded_suffix_with_metric_partitions(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)

    view = compose_d1_inspect_view(snapshot, evaluation_limit=1)

    assert (
        view.evaluations.limit,
        view.evaluations.total_count,
        view.evaluations.returned_count,
        view.evaluations.omitted_count,
    ) == (1, 3, 1, 2)
    assert [item.eval_index for item in view.evaluations.items] == [2]
    item = view.evaluations.items[0]
    deciding, report_only = item.quality.deciding, item.quality.report_only
    assert set(deciding).isdisjoint(report_only)
    assert set(deciding) | set(report_only) == set(D1_SCALAR_METRICS)
    assert set(report_only) == set(D1_REPORT_ONLY_METRICS)
    assert all(type(value) is float for value in (*deciding.values(), *report_only.values()))
    assert item.latent_width == snapshot.run.evals[-1].quality.latent_width
    assert item.resource_work == snapshot.run.evals[-1].resource_work


@pytest.mark.parametrize("limit", [0, -1, True, D1_INSPECT_MAX_EVALUATIONS + 1])
def test_evaluation_window_limit_is_positive_and_hard_bounded(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="evaluation_limit"):
        compose_d1_inspect_view(_snapshot(tmp_path), evaluation_limit=limit)


def test_zero_eval_partial_prefix_has_an_explicit_empty_window(tmp_path: Path) -> None:
    view = compose_d1_inspect_view(
        _snapshot(tmp_path, events=valid_events()[:2], allow_incomplete=True),
        evaluation_limit=4,
    )

    assert type(view.state) is D1InspectPartial
    assert view.evaluations.total_count == view.evaluations.returned_count == 0
    assert view.progress.latest_quality is None
    assert view.progress.latest_checkpoint is None


@pytest.mark.parametrize("finalized", [False, True])
def test_trainer_abort_retains_bounded_event_ref_and_separate_finalization(
    tmp_path: Path, finalized: bool
) -> None:
    events = valid_events()
    events[-2] = replace(
        events[-2],
        event_kind=JournalEventKind.TRAINER_ABORTED,
        payload={"error": "sensitive arbitrary text is not projected"},
    )
    snapshot = _snapshot(
        tmp_path,
        events=events if finalized else events[:-1],
        allow_incomplete=True,
    )

    view = compose_d1_inspect_view(snapshot, evaluation_limit=4)

    assert type(view.state) is D1InspectTrainerAborted
    assert snapshot.run.trainer_abort is not None
    assert view.state.trainer_abort.event_sha256 == canonical_digest(
        snapshot.run.trainer_abort.event.manifest
    )
    assert view.state.numerical_defect is None
    assert (view.state.run_finalization is not None) is finalized


def test_run_abort_is_not_misrepresented_as_partial_or_trainer_abort(tmp_path: Path) -> None:
    events = valid_events()
    events[-1] = replace(
        events[-1],
        event_kind=JournalEventKind.RUN_ABORT,
        payload={"error": "operator abort"},
    )
    snapshot = _snapshot(tmp_path, events=events, allow_incomplete=True)

    view = compose_d1_inspect_view(snapshot, evaluation_limit=4)

    assert type(view.state) is D1InspectRunAborted
    assert snapshot.run.terminal is not None
    assert view.state.run_abort.event_sha256 == canonical_digest(snapshot.run.terminal.manifest)


def test_view_assembly_refuses_a_drifted_snapshot_capability(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    object.__setattr__(snapshot.run, "terminal", None)

    with pytest.raises(TypeError, match="factory-attested"):
        compose_d1_inspect_view(snapshot, evaluation_limit=1)
