"""Dedicated defect capture admits typed aborts and derives whole-pair exclusions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import entropic_substrate.ladder.capture.reader.telemetry as telemetry_api
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass, RunClass
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.ladder.capture.reader.d1_numerical_defect import (
    D1DefectCheckpointReference,
    D1NonfiniteClassification,
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.ladder.capture.records.d1_journal import read_journal
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind
from entropic_substrate.ladder.capture.records.d1_terminal_defect_capture import (
    D1DefectCaptureError,
    D1TerminalDefectCaptureInput,
    derive_d1_pair_exclusions,
    read_d1_terminal_numerical_defect,
    validate_d1_pair_exclusion_claims,
)

from .d1_capture_fixtures import DVC_POINTER, capture_family
from .d1_telemetry_fixtures import write_events

pytestmark = pytest.mark.unit


def _defect_case(tmp_path: Path):
    roster, inputs, root = capture_family(tmp_path)
    expected = roster.expected_confirmatory[0]
    original = inputs[0]
    events = list(read_journal(root / original.journal.normalized_path))
    checkpoint = replace(
        original.checkpoint,
        role=D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT,
    )
    events[5] = replace(
        events[5],
        payload={"checkpoint_sha256": checkpoint.sha256},
    )
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
        execution_segment_id=f"{expected.logical_run_id}:segment:0",
        plan_digest=expected.plan_digest,
        family_digest=expected.family_digest,
        construction_digest=expected.construction_digest,
        execution_digest=expected.execution_digest,
        aggregation_digest=expected.aggregation_digest,
        checkpoint=D1DefectCheckpointReference.receipt(checkpoint.sha256),
        evidence_class=EvidenceClass.MEASURED,
    )
    abort = replace(
        events[-1],
        event_kind=JournalEventKind.TRAINER_ABORTED,
        event_seq=7,
        successful_step=2,
        payload={
            "error_message": "typed terminal defect",
            "error_type": "D1NonfiniteMetricError",
            "numerical_defect": record.manifest,
        },
    )
    defect_events = [*events[:7], abort]
    journal_path = root / original.journal.normalized_path
    write_events(journal_path, defect_events)
    journal = identify_d1_artifact(
        root,
        journal_path,
        role=D1ArtifactRole.JOURNAL,
        dvc_pointer_path=DVC_POINTER,
    )
    capture = D1TerminalDefectCaptureInput(
        expected=expected,
        journal=journal,
        checkpoint=checkpoint,
        billing_sidecars=original.billing_sidecars,
    )
    return roster, root, capture


def test_dedicated_reader_admits_only_typed_incomplete_defect_and_exact_artifacts(
    tmp_path: Path,
) -> None:
    roster, root, capture = _defect_case(tmp_path)
    with pytest.raises(Exception, match="trainer_aborted"):
        telemetry_api.read_d1_run_evidence(root / capture.journal.normalized_path)

    evidence = read_d1_terminal_numerical_defect(root, capture)

    assert evidence.record.logical_run_id == capture.expected.logical_run_id
    assert evidence.record.checkpoint.sha256 == capture.checkpoint.sha256
    assert evidence.journal == capture.journal
    assert evidence.checkpoint == capture.checkpoint
    assert evidence.capture_input_digest == capture.capture_input_digest
    assert len(evidence.capture_input_digest) == 64
    assert evidence.record.evidence_class is EvidenceClass.MEASURED
    assert roster.expected_run_ids > {evidence.record.logical_run_id}


def test_dedicated_reader_retains_defect_after_operator_run_end_finalization(
    tmp_path: Path,
) -> None:
    _, root, capture = _defect_case(tmp_path)
    path = root / capture.journal.normalized_path
    events = list(read_journal(path))
    events.append(
        replace(
            events[-1],
            event_kind=JournalEventKind.RUN_END,
            event_seq=events[-1].event_seq + 1,
            payload={"outcome": "trainer_failed"},
        )
    )
    write_events(path, events)
    capture = replace(
        capture,
        journal=identify_d1_artifact(
            root,
            path,
            role=D1ArtifactRole.JOURNAL,
            dvc_pointer_path=DVC_POINTER,
        ),
    )

    evidence = read_d1_terminal_numerical_defect(root, capture)

    assert evidence.record.logical_run_id == capture.expected.logical_run_id


@pytest.mark.parametrize("mutation", ["gap", "wrong-checkpoint", "wrong-source", "complete"])
def test_defect_reader_rejects_unsubstantiated_or_wrong_lane_evidence(
    tmp_path: Path, mutation: str
) -> None:
    _, root, capture = _defect_case(tmp_path)
    if mutation == "wrong-checkpoint":
        replacement_path = root / "replacement.pt"
        replacement_path.write_bytes(b"replacement\n")
        capture = replace(
            capture,
            checkpoint=identify_d1_artifact(
                root,
                replacement_path,
                role=D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT,
                dvc_pointer_path=DVC_POINTER,
            ),
        )
    else:
        events = list(read_journal(root / capture.journal.normalized_path))
        payload = cast(dict[str, object], events[-1].manifest["payload"])
        defect = cast(dict[str, object], payload["numerical_defect"])
        if mutation == "gap":
            defect["checkpoint"] = {
                "gap_code": "no_prior_checkpoint",
                "kind": "gap",
                "sha256": None,
            }
        elif mutation == "wrong-source":
            defect["source_exception_code"] = "unregistered.source"
        else:
            events[-1] = replace(
                events[-1], event_kind=JournalEventKind.RUN_END, payload={"outcome": "success"}
            )
        if mutation != "complete":
            payload["numerical_defect"] = defect
            events[-1] = replace(events[-1], payload=payload)
        path = root / capture.journal.normalized_path
        write_events(path, events)
        capture = replace(
            capture,
            journal=identify_d1_artifact(
                root,
                path,
                role=D1ArtifactRole.JOURNAL,
                dvc_pointer_path=DVC_POINTER,
            ),
        )
    with pytest.raises(D1DefectCaptureError):
        read_d1_terminal_numerical_defect(root, capture)


def test_analysis_derives_whole_pair_exclusion_and_rejects_inherited_claim(
    tmp_path: Path,
) -> None:
    roster, root, capture = _defect_case(tmp_path)
    evidence = read_d1_terminal_numerical_defect(root, capture)

    exclusions = derive_d1_pair_exclusions(roster, (evidence,))

    assert len(exclusions) == 1
    assert set(exclusions[0].member_run_ids) == roster.expected_run_ids
    assert exclusions[0].evidence_paths == (capture.journal.normalized_path,)
    assert validate_d1_pair_exclusion_claims(roster, exclusions, (evidence,)) == exclusions
    with pytest.raises(D1DefectCaptureError, match="unsupported_exclusion"):
        validate_d1_pair_exclusion_claims(roster, exclusions, ())


def test_reader_calls_public_authority_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root, capture = _defect_case(tmp_path)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise ValueError("authoritative reader refused")

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", refuse)
    with pytest.raises(D1DefectCaptureError, match="authoritative_reader") as caught:
        read_d1_terminal_numerical_defect(root, capture)
    assert isinstance(caught.value.__cause__, ValueError)


def test_reader_consumes_one_authoritative_fold_without_second_journal_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root, capture = _defect_case(tmp_path)
    path = root / capture.journal.normalized_path
    real = telemetry_api.read_d1_run_evidence

    def read_once(*args: object, **kwargs: object):
        run = real(*args, **kwargs)
        path.unlink()
        return run

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", read_once)

    evidence = read_d1_terminal_numerical_defect(root, capture)

    assert evidence.record.logical_run_id == capture.expected.logical_run_id
    assert not path.exists()
