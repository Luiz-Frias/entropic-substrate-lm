"""Mixed complete/terminal-defect capture fixture for D1 application tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from entropic_substrate.kernel.evidence_class.resource_provenance import execution_segment_id
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate import D1TerminalDefectInput
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.ladder.capture.custody.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureInput,
)
from entropic_substrate.ladder.capture.custody.d1_capture_roster import (
    D1AttemptLane,
    D1LaunchRoster,
)
from entropic_substrate.ladder.capture.reader.d1_numerical_defect import (
    D1DefectCheckpointReference,
    D1NonfiniteClassification,
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind

from .d1_capture_fixtures import _events_for, capture_family
from .d1_telemetry_fixtures import write_events
from .test_d1_capture_attempt_transport import _attempt
from .test_d1_capture_service import _POINTER_PATH, _artifact_pointer, _capture_pointer


def heterogeneous_family(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (7,),
) -> tuple[
    D1LaunchRoster,
    tuple[D1CaptureInput, ...],
    tuple[D1TerminalDefectInput, ...],
    tuple[D1AttemptEvidence, ...],
    Path,
]:
    roster, raw_inputs, root = capture_family(tmp_path, seeds=seeds)
    expected = next(
        item
        for item in roster.expected_confirmatory
        if item.seed == seeds[0] and item.arm is D1Arm.SG
    )
    source = next(item for item in raw_inputs if item.logical_run_id == expected.logical_run_id)
    events, plan_digest = _events_for(
        expected.arm,
        logical_run_id=expected.logical_run_id,
        construction_digest=expected.construction_digest,
        execution_digest=expected.execution_digest,
        aggregation_digest=expected.aggregation_digest,
        checkpoint_sha256="f" * 64,
        seed=expected.seed,
        pairing_id=expected.pairing_id,
    )
    assert plan_digest == expected.plan_digest
    record = D1TerminalNumericalDefectRecord(
        metric_name="nats_per_byte",
        classification=D1NonfiniteClassification.NAN,
        source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
        successful_step=2,
        max_steps=2,
        eval_index=2,
        arm=expected.arm,
        seed=expected.seed,
        pairing_id=expected.pairing_id,
        logical_run_id=expected.logical_run_id,
        run_class=RunClass.CONFIRMATORY,
        execution_segment_index=0,
        execution_segment_id=execution_segment_id(expected.logical_run_id, 0),
        plan_digest=expected.plan_digest,
        family_digest=expected.family_digest,
        construction_digest=expected.construction_digest,
        execution_digest=expected.execution_digest,
        aggregation_digest=expected.aggregation_digest,
        checkpoint=D1DefectCheckpointReference.gap("no_prior_checkpoint"),
    )
    events[-2] = replace(
        events[-2],
        event_kind=JournalEventKind.TRAINER_ABORTED,
        payload={"error": "terminal nonfinite", "numerical_defect": record.manifest},
    )
    journal_path = root / "defects" / f"{expected.logical_run_id}.jsonl"
    journal_path.parent.mkdir()
    write_events(journal_path, events)
    defect = D1TerminalDefectInput(
        logical_run_id=expected.logical_run_id,
        journal=identify_d1_artifact(
            root,
            journal_path,
            role=D1ArtifactRole.JOURNAL,
            dvc_pointer_path=_POINTER_PATH,
        ),
        supporting_artifacts=tuple(_artifact_pointer(item) for item in source.billing_sidecars),
    )
    attempt, attempt_evidence = _attempt(
        root,
        logical_run_id=f"d1-sg-s{expected.seed}-attempt-0",
        run_class=RunClass.CONFIRMATORY,
        lane=D1AttemptLane.ABORTED,
    )
    roster = D1LaunchRoster(
        preregistration_path=roster.preregistration_path,
        preregistration_commit=roster.preregistration_commit,
        family_digest=roster.family_digest,
        expected_confirmatory=roster.expected_confirmatory,
        attempts=(attempt,),
    )
    complete = tuple(
        _capture_pointer(item)
        for item in raw_inputs
        if item.logical_run_id != expected.logical_run_id
    )
    return roster, complete, (defect,), (attempt_evidence,), root
