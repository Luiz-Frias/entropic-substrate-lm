"""Authoritative D1 query path accepts only complete journal evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import entropic_substrate.evidence.d1_telemetry_query as query_module
from entropic_substrate.d1_training_step import D1GradientEvidenceKind, D1TrainStepResult
from entropic_substrate.evidence.d1_abort_evidence import D1TrainerAbortEvidence
from entropic_substrate.evidence.d1_journal_values import JournalEventKind, canonical_event_line
from entropic_substrate.evidence.d1_telemetry_query import (
    final_d1_eval_metric,
    read_d1_eval_series,
    read_d1_run_evidence,
)
from entropic_substrate.evidence.d1_telemetry_values import (
    D1EvalEvidence,
    D1RunEvidence,
    D1RunStatus,
    D1TailQuarantineKind,
    D1TelemetryGap,
)
from entropic_substrate.evidence.resource_work_codec import resource_snapshot_to_mapping
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass, canonical_digest

from .d1_telemetry_fixtures import (
    IDENTITY,
    LOGICAL_RUN_ID,
    replace_payload,
    resequence,
    resource,
    valid_events,
    write_events,
)

pytestmark = pytest.mark.unit


def _path(tmp_path: Path, events: list[object]) -> Path:
    target = tmp_path / "d1.jsonl"
    write_events(target, cast(list, events))
    return target


def test_complete_run_returns_immutable_quality_resource_and_identity(tmp_path: Path) -> None:
    path = _path(tmp_path, valid_events())

    run = read_d1_run_evidence(
        path,
        expected_logical_run_id=LOGICAL_RUN_ID,
        expected_family_digest=str(IDENTITY["family_digest"]),
        expected_run_class=RunClass.CONFIRMATORY,
    )

    assert type(run) is D1RunEvidence
    assert run.path == path
    assert run.identity.logical_run_id == LOGICAL_RUN_ID
    assert [item.step for item in run.evals] == [0, 1, 2]
    assert all(type(item) is D1EvalEvidence for item in run.evals)
    assert [item.resource_work.eval_index for item in run.evals] == [0, 1, 2]
    assert [item.quality.target_rms_l2_norm for item in run.evals] == [0.6127805479941412] * 3
    assert run.final_checkpoint_sha256 == "f" * 64
    assert run.data_order_segments == ((0, "e" * 64, 2),)
    assert len(run.segments) == 1
    assert [item.successful_step for item in run.train_steps] == [1, 2]
    assert all(type(item) is D1TrainStepResult for item in run.train_steps)


def test_resumed_run_binds_checkpoint_segment_provenance_and_local_order(tmp_path: Path) -> None:
    run = read_d1_run_evidence(_path(tmp_path, valid_events(resumed=True)))

    assert [segment.execution_segment_index for segment in run.segments] == [0, 1]
    assert run.resume_checkpoint_sha256s == ("d" * 64,)
    assert run.data_order_segments == ((0, "c" * 64, 1), (1, "e" * 64, 1))
    assert run.evals[-1].resource_work.execution_segment_index == 1
    assert len(run.tail_quarantine_evidence) == 1
    assert run.tail_quarantine_evidence[0].kind is D1TailQuarantineKind.NO_TAIL
    assert [item.successful_step for item in run.train_steps] == [1, 2]


def test_online_target_arm_admits_measured_target_gradients(tmp_path: Path) -> None:
    run = read_d1_run_evidence(_path(tmp_path, valid_events(arm=D1Arm.NONE)))

    assert [item.successful_step for item in run.train_steps] == [1, 2]
    assert all(
        item.gradient_reachability.target_branch.kind is D1GradientEvidenceKind.MEASURED_NONZERO
        for item in run.train_steps
    )


@pytest.mark.parametrize(
    ("plan_mutation", "match"),
    [
        (
            lambda plan: cast(dict[str, object], plan["evidence"]).update(
                {"telemetry_schema_version": 1}
            ),
            "telemetry_schema_version",
        ),
        (
            lambda plan: cast(dict[str, object], plan["evidence"]).update(
                {"telemetry_schema_version": 2}
            ),
            "telemetry_schema_version",
        ),
        (lambda plan: plan.update({"arm": "none"}), "target gradient posture"),
    ],
)
def test_run_start_binds_train_schema_and_target_posture(
    tmp_path: Path,
    plan_mutation: Callable[[dict[str, Any]], object],
    match: str,
) -> None:
    events = valid_events()
    start_payload = cast(dict[str, object], events[0].manifest["payload"])
    plan = cast(dict[str, Any], start_payload["plan"])
    plan_mutation(plan)
    start_payload["plan_digest"] = canonical_digest(plan)
    events[0] = replace_payload(events[0], start_payload)

    with pytest.raises(D1TelemetryGap, match=match):
        read_d1_run_evidence(_path(tmp_path, events))


def _quarantined_resume_path(tmp_path: Path) -> tuple[Path, Path]:
    events = valid_events(resumed=True)
    retained = b"".join(canonical_event_line(event) for event in events[:6])
    tail = [replace(events[7], event_seq=6), replace(events[8], event_seq=7)]
    artifact_bytes = retained + b"".join(canonical_event_line(event) for event in tail)
    artifact = tmp_path / "d1.pre-resume.jsonl"
    artifact.write_bytes(artifact_bytes)
    resume_payload = cast(dict[str, object], events[6].manifest["payload"])
    resume_payload["tail_quarantine"] = {
        "artifact": {
            "byte_count": len(artifact_bytes),
            "file_name": artifact.name,
            "role": "pre_resume_authoritative_journal",
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        },
        "kind": "quarantined_tail",
        "retained_prefix": {
            "byte_count": len(retained),
            "sha256": hashlib.sha256(retained).hexdigest(),
        },
        "schema_version": "d1-tail-quarantine/v1",
        "tail": {"event_count": 2, "first_event_sequence": 6, "last_event_sequence": 7},
    }
    events[6] = replace_payload(events[6], resume_payload)
    return _path(tmp_path, events), artifact


def test_quarantined_resume_exposes_verified_concrete_receipt_identity(tmp_path: Path) -> None:
    path, artifact = _quarantined_resume_path(tmp_path)
    run = read_d1_run_evidence(path)
    evidence = run.tail_quarantine_evidence[0]
    assert evidence.kind is D1TailQuarantineKind.QUARANTINED_TAIL
    assert evidence.artifact_file_name == artifact.name
    assert evidence.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert evidence.tail_event_count == 2


def test_quarantine_artifact_drift_fails_through_authoritative_reader(tmp_path: Path) -> None:
    path, artifact = _quarantined_resume_path(tmp_path)
    drifted = bytearray(artifact.read_bytes())
    drifted[0] ^= 1
    artifact.write_bytes(drifted)
    with pytest.raises(D1TelemetryGap, match=r"artifact.*sha256|sha256.*artifact"):
        read_d1_run_evidence(path)


def test_series_and_terminal_metric_share_the_whole_run_admission_gate(tmp_path: Path) -> None:
    path = _path(tmp_path, valid_events())

    assert [row.quality.nats_per_byte for row in read_d1_eval_series(path)] == [0.8, 0.79, 0.78]
    assert final_d1_eval_metric(path, "nats_per_byte") == 0.78
    assert final_d1_eval_metric(path, "numerical_rank") == 2.0
    assert final_d1_eval_metric(path, "target_rms_l2_norm") == 0.6127805479941412
    with pytest.raises(D1TelemetryGap, match="registered scalar quality metric"):
        final_d1_eval_metric(path, "state_variance")


@pytest.mark.parametrize("terminal", [JournalEventKind.RUN_ABORT, JournalEventKind.TRAINER_ABORTED])
def test_abort_evidence_is_typed_invalid_not_an_empty_run(
    tmp_path: Path, terminal: JournalEventKind
) -> None:
    events = valid_events()
    if terminal is JournalEventKind.RUN_ABORT:
        events[-1] = replace(events[-1], event_kind=terminal, payload={"error": "defect"})
    else:
        events[-2] = replace(events[-2], event_kind=terminal, payload={"error": "defect"})
        events[-1] = replace(events[-1], event_seq=events[-2].event_seq + 1)
    with pytest.raises(D1TelemetryGap) as captured:
        read_d1_run_evidence(_path(tmp_path, events))
    assert captured.value.code in {"run_aborted", "trainer_aborted"}


@pytest.mark.parametrize(
    ("terminal", "status"),
    [
        (JournalEventKind.RUN_ABORT, D1RunStatus.RUN_ABORTED),
        (JournalEventKind.TRAINER_ABORTED, D1RunStatus.TRAINER_ABORTED),
    ],
)
def test_abort_state_is_returned_only_when_explicitly_requested(
    tmp_path: Path, terminal: JournalEventKind, status: D1RunStatus
) -> None:
    events = valid_events()
    if terminal is JournalEventKind.RUN_ABORT:
        events[-1] = replace(events[-1], event_kind=terminal, payload={"error": "defect"})
    else:
        events[-2] = replace(events[-2], event_kind=terminal, payload={"error": "defect"})
        events[-1] = replace(events[-1], event_seq=events[-2].event_seq + 1)
    run = read_d1_run_evidence(_path(tmp_path, events), allow_incomplete=True)
    assert run.status is status
    if terminal is JournalEventKind.TRAINER_ABORTED:
        assert type(run.trainer_abort) is D1TrainerAbortEvidence
        assert run.trainer_abort.numerical_defect is None
        assert run.terminal is not None
        assert run.terminal.event_kind is JournalEventKind.RUN_END
    else:
        assert run.trainer_abort is None
        assert run.terminal is not None
        assert run.terminal.event_kind is JournalEventKind.RUN_ABORT


def test_partial_state_is_returned_only_when_explicitly_requested(tmp_path: Path) -> None:
    run = read_d1_run_evidence(_path(tmp_path, valid_events()[:-1]), allow_incomplete=True)
    assert run.status is D1RunStatus.PARTIAL
    assert run.trainer_abort is None
    assert run.terminal is None


def test_unfinalized_trainer_abort_retains_abort_evidence_without_terminal_event(
    tmp_path: Path,
) -> None:
    events = valid_events()
    events[-2] = replace(
        events[-2],
        event_kind=JournalEventKind.TRAINER_ABORTED,
        payload={"error": "trainer stopped before operator finalization"},
    )

    run = read_d1_run_evidence(_path(tmp_path, events[:-1]), allow_incomplete=True)

    assert run.status is D1RunStatus.TRAINER_ABORTED
    assert type(run.trainer_abort) is D1TrainerAbortEvidence
    assert run.trainer_abort.event == events[-2]
    assert run.trainer_abort.numerical_defect is None
    assert run.terminal is None


def test_partial_run_and_missing_final_checkpoint_are_refused(tmp_path: Path) -> None:
    with pytest.raises(D1TelemetryGap) as partial:
        read_d1_run_evidence(_path(tmp_path, valid_events()[:-1]))
    assert partial.value.code == "partial_run"

    events = valid_events()
    del events[-2]
    with pytest.raises(D1TelemetryGap) as missing:
        read_d1_run_evidence(_path(tmp_path, resequence(events)))
    assert missing.value.code == "missing_final_checkpoint"


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong-index", "wrong-width"])
def test_eval_payload_mutations_are_rejected(tmp_path: Path, mutation: str) -> None:
    events = valid_events()
    target = events[2]
    payload = cast(dict[str, object], target.manifest["payload"])
    if mutation == "missing":
        del cast(dict[str, object], payload["quality"])["future_nll"]
    elif mutation == "extra":
        cast(dict[str, object], payload["quality"])["inferred_quality"] = 0.0
    elif mutation == "wrong-index":
        payload["eval_index"] = 1
    else:
        cast(dict[str, object], payload["quality"])["state_variance"] = [1.0]
    events[2] = replace_payload(target, payload)

    with pytest.raises(D1TelemetryGap):
        read_d1_run_evidence(_path(tmp_path, events))


@pytest.mark.parametrize(("resumed", "drift_index"), [(False, 1), (True, 2)])
def test_reader_enforces_cross_eval_width_on_the_complete_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resumed: bool, drift_index: int
) -> None:
    real_parse = query_module.parse_eval_event
    width_one = type("WidthOne", (), {"latent_width": 1})()

    def drift(event: object, index: int) -> D1EvalEvidence:
        evidence = real_parse(event, index)  # type: ignore[arg-type]
        return replace(evidence, quality=cast(Any, width_one)) if index == drift_index else evidence

    monkeypatch.setattr(query_module, "parse_eval_event", drift)
    with pytest.raises(D1TelemetryGap, match="latent width changed"):
        read_d1_run_evidence(_path(tmp_path, valid_events(resumed=resumed)))


def test_nonmonotonic_resource_and_data_order_are_refused(tmp_path: Path) -> None:
    events = valid_events()
    payload = cast(dict[str, object], events[7].manifest["payload"])
    regressed = replace(resource(2, 2), committed_quality_source_bytes=0)
    payload["resource_work"] = resource_snapshot_to_mapping(regressed)
    events[7] = replace_payload(events[7], payload)
    with pytest.raises(D1TelemetryGap, match="resource work series"):
        read_d1_run_evidence(_path(tmp_path, events))

    events = valid_events()
    train_payload = cast(dict[str, object], events[6].manifest["payload"])
    cast(dict[str, object], train_payload["data_order"])["batches_attested"] = 1
    events[6] = replace_payload(events[6], train_payload)
    with pytest.raises(D1TelemetryGap, match="data-order"):
        read_d1_run_evidence(_path(tmp_path, events))


def test_expected_identity_is_not_inferred_from_a_different_capture(tmp_path: Path) -> None:
    path = _path(tmp_path, valid_events())
    with pytest.raises(D1TelemetryGap, match="logical_run_id"):
        read_d1_run_evidence(path, expected_logical_run_id="another-run")
    with pytest.raises(D1TelemetryGap, match="family_digest"):
        read_d1_run_evidence(path, expected_family_digest="0" * 64)
    with pytest.raises(D1TelemetryGap, match="run_class"):
        read_d1_run_evidence(path, expected_run_class=RunClass.PILOT)
