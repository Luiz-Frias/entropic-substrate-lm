"""Terminal-defect prior checkpoints retain a dedicated capture role."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from entropic_substrate.ladder.capture.custody.d1_capture_aggregate import D1TerminalDefectInput
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate_validation import (
    D1CaptureAggregateError,
    build_d1_capture_aggregate,
)
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.ladder.capture.reader.d1_numerical_defect import (
    D1DefectCheckpointReference,
    terminal_numerical_defect_from_mapping,
)
from entropic_substrate.ladder.capture.records.d1_journal import read_journal
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .d1_telemetry_fixtures import write_events

pytestmark = pytest.mark.unit


def _receipt_case(tmp_path: Path):
    roster, complete, defects, attempts, root = heterogeneous_family(tmp_path)
    source = defects[0]
    checkpoint_path = root / "defects" / "prior-checkpoint.pt"
    checkpoint_path.write_bytes(b"durable prior checkpoint\n")
    checkpoint = identify_d1_artifact(
        root,
        checkpoint_path,
        role=D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT,
        dvc_pointer_path=source.journal.dvc.pointer_path,
    )
    journal_path = root / source.journal.normalized_path
    events = list(read_journal(journal_path))
    checkpoint_index = next(
        index
        for index, event in enumerate(events)
        if event.event_kind is JournalEventKind.CHECKPOINT
    )
    events[checkpoint_index] = replace(
        events[checkpoint_index],
        payload={"checkpoint_sha256": checkpoint.sha256},
    )
    abort_index = next(
        index
        for index, event in enumerate(events)
        if event.event_kind is JournalEventKind.TRAINER_ABORTED
    )
    abort_payload = cast(dict[str, object], events[abort_index].manifest["payload"])
    record = terminal_numerical_defect_from_mapping(abort_payload["numerical_defect"])
    record = replace(record, checkpoint=D1DefectCheckpointReference.receipt(checkpoint.sha256))
    events[abort_index] = replace(
        events[abort_index],
        payload={"error": abort_payload["error"], "numerical_defect": record.manifest},
    )
    write_events(journal_path, events)
    journal = identify_d1_artifact(
        root,
        journal_path,
        role=D1ArtifactRole.JOURNAL,
        dvc_pointer_path=source.journal.dvc.pointer_path,
    )
    defect = D1TerminalDefectInput(
        logical_run_id=source.logical_run_id,
        journal=journal,
        supporting_artifacts=(*source.supporting_artifacts, checkpoint),
    )
    return roster, complete, (defect,), attempts, root


def _build(case):
    roster, complete, defects, attempts, root = case
    return build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )


def test_receipt_bearing_defect_requires_dedicated_prior_checkpoint_role(
    tmp_path: Path,
) -> None:
    aggregate = _build(_receipt_case(tmp_path))

    assert len(aggregate.terminal_defects) == 1
    supporting = aggregate.terminal_defects[0].capture_input.supporting_artifacts
    assert any(item.role is D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT for item in supporting)


def test_ordinary_complete_run_checkpoint_role_cannot_substitute(tmp_path: Path) -> None:
    roster, complete, defects, attempts, root = _receipt_case(tmp_path)
    defect = defects[0]
    supporting = tuple(
        replace(item, role=D1ArtifactRole.CHECKPOINT)
        if item.role is D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT
        else item
        for item in defect.supporting_artifacts
    )

    with pytest.raises(D1CaptureAggregateError) as captured:
        _build(
            (roster, complete, (replace(defect, supporting_artifacts=supporting),), attempts, root)
        )

    assert captured.value.code == "defect_checkpoint"


def test_checkpoint_gap_cannot_capture_a_dedicated_prior_checkpoint(tmp_path: Path) -> None:
    roster, complete, defects, attempts, root = _receipt_case(tmp_path)
    defect = defects[0]
    journal_path = root / defect.journal.normalized_path
    events = list(read_journal(journal_path))
    abort_index = next(
        index
        for index, event in enumerate(events)
        if event.event_kind is JournalEventKind.TRAINER_ABORTED
    )
    payload = cast(dict[str, object], events[abort_index].manifest["payload"])
    record = terminal_numerical_defect_from_mapping(payload["numerical_defect"])
    record = replace(record, checkpoint=D1DefectCheckpointReference.gap("no_prior_checkpoint"))
    events[abort_index] = replace(
        events[abort_index],
        payload={"error": payload["error"], "numerical_defect": record.manifest},
    )
    write_events(journal_path, events)
    journal = identify_d1_artifact(
        root,
        journal_path,
        role=D1ArtifactRole.JOURNAL,
        dvc_pointer_path=defect.journal.dvc.pointer_path,
    )

    with pytest.raises(D1CaptureAggregateError) as captured:
        _build(
            (
                roster,
                complete,
                (replace(defect, journal=journal),),
                attempts,
                root,
            )
        )

    assert captured.value.code == "defect_checkpoint"
