"""Whole-journal D1 state-machine validation and query APIs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from entropic_substrate.d1_training_step import D1TrainStepResult
from entropic_substrate.evidence.d1_journal import read_journal, read_journal_bytes
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalError,
    D1JournalEvent,
    FrozenJson,
    JournalEventKind,
    JournalIdentity,
)
from entropic_substrate.evidence.d1_telemetry_measurements import (
    parse_eval_event,
    parse_train_event,
)
from entropic_substrate.evidence.d1_telemetry_payloads import (
    exact_fields,
    parse_baseline_event,
    parse_run_start_event,
    parse_run_start_manifest,
    parse_wall_origin,
    require_mapping,
    require_sha256,
)
from entropic_substrate.evidence.d1_telemetry_values import (
    D1EvalEvidence,
    D1RunEvidence,
    D1RunStatus,
    D1TailQuarantineEvidence,
    D1TailQuarantineKind,
    gap,
)
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
    validate_execution_segments,
)
from entropic_substrate.evidence.resource_work import validate_resource_work_series
from entropic_substrate.experiments.d1_family import strategies_for_arm
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

__all__ = ["final_d1_eval_metric", "read_d1_eval_series", "read_d1_run_evidence"]


def _tail_quarantine(event: D1JournalEvent) -> D1TailQuarantineEvidence:
    value = event.payload["tail_quarantine"]
    if not isinstance(value, Mapping):  # canonical journal validation proves this first
        raise gap("tail_quarantine", "run_resume.tail_quarantine", "must be a mapping")
    manifest = cast(Mapping[str, FrozenJson], value)
    kind = manifest.get("kind")
    if kind == D1TailQuarantineKind.NO_TAIL.value:
        return D1TailQuarantineEvidence.no_tail(manifest)
    if kind != D1TailQuarantineKind.QUARANTINED_TAIL.value:
        raise gap("tail_quarantine", "run_resume.tail_quarantine.kind", "is unsupported")
    artifact = cast(Mapping[str, FrozenJson], manifest["artifact"])
    retained = cast(Mapping[str, FrozenJson], manifest["retained_prefix"])
    tail = cast(Mapping[str, FrozenJson], manifest["tail"])
    file_name = artifact["file_name"]
    counts = (
        artifact["byte_count"],
        retained["byte_count"],
        tail["event_count"],
        tail["first_event_sequence"],
        tail["last_event_sequence"],
    )
    if type(file_name) is not str or any(type(value) is not int for value in counts):
        raise gap("tail_quarantine", "run_resume.tail_quarantine", "identity is malformed")
    return D1TailQuarantineEvidence.quarantined(
        manifest,
        artifact_file_name=file_name,
        artifact_sha256=require_sha256(
            artifact["sha256"], path="run_resume.tail_quarantine.artifact.sha256"
        ),
        artifact_byte_count=cast(int, counts[0]),
        retained_prefix_sha256=require_sha256(
            retained["sha256"], path="run_resume.tail_quarantine.retained_prefix.sha256"
        ),
        retained_prefix_byte_count=cast(int, counts[1]),
        tail_event_count=cast(int, counts[2]),
        first_tail_event_sequence=cast(int, counts[3]),
        last_tail_event_sequence=cast(int, counts[4]),
    )


def _parse_resume_event(
    event: D1JournalEvent,
    *,
    previous_checkpoint_sha256: str,
    expected_segment: int,
    previous_segment: RunStartProvenance,
) -> tuple[RunStartProvenance, D1TailQuarantineEvidence]:
    payload = require_mapping(event.manifest["payload"], path="run_resume.payload")
    exact_fields(
        payload,
        {
            "checkpoint_sha256",
            "run_start_provenance",
            "tail_quarantine",
            "trainer_wall_origin",
        },
        path="run_resume.payload",
    )
    checkpoint = require_sha256(payload["checkpoint_sha256"], path="run_resume.checkpoint_sha256")
    if checkpoint != previous_checkpoint_sha256:
        raise gap("resume_checkpoint", "run_resume.checkpoint_sha256", "differs from receipt")
    start = parse_run_start_manifest(
        payload["run_start_provenance"], path="run_resume.run_start_provenance"
    )
    if (
        start.logical_run_id != event.logical_run_id
        or start.execution_segment_index != expected_segment
        or start.segment_start_mode is not SegmentStartMode.RESUME
        or start.resume_checkpoint_sha256 != checkpoint
    ):
        raise gap("segment_chain", "run_resume.run_start_provenance", "is not the next segment")
    if start.trainer_started_at < previous_segment.trainer_started_at:
        raise gap("segment_time", "run_resume.run_start_provenance", "time moved backwards")
    parse_wall_origin(
        payload["trainer_wall_origin"], path="run_resume.trainer_wall_origin", start=start
    )
    return start, _tail_quarantine(event)


def _check_expected_identity(
    identity: JournalIdentity,
    *,
    expected_logical_run_id: str | None,
    expected_family_digest: str | None,
    expected_run_class: RunClass | None,
) -> None:
    if expected_logical_run_id is not None and identity.logical_run_id != expected_logical_run_id:
        raise gap("identity_mismatch", "logical_run_id", "differs from the requested run")
    if expected_family_digest is not None:
        require_sha256(expected_family_digest, path="expected_family_digest")
        if identity.family_digest != expected_family_digest:
            raise gap("identity_mismatch", "family_digest", "differs from the requested family")
    if expected_run_class is not None:
        if type(expected_run_class) is not RunClass:
            raise TypeError("expected_run_class must be a RunClass")
        if identity.run_class is not expected_run_class:
            raise gap("identity_mismatch", "run_class", "differs from the requested class")


def _status(events: tuple[D1JournalEvent, ...], *, allow_incomplete: bool) -> D1RunStatus:
    if type(allow_incomplete) is not bool:
        raise TypeError("allow_incomplete must be a bool")
    if any(event.event_kind is JournalEventKind.TRAINER_ABORTED for event in events):
        status = D1RunStatus.TRAINER_ABORTED
    elif events[-1].event_kind is JournalEventKind.RUN_ABORT:
        status = D1RunStatus.RUN_ABORTED
    elif events[-1].event_kind is JournalEventKind.RUN_END:
        status = D1RunStatus.COMPLETE
    else:
        status = D1RunStatus.PARTIAL
    if not allow_incomplete and status is not D1RunStatus.COMPLETE:
        code = {
            D1RunStatus.PARTIAL: "partial_run",
            D1RunStatus.TRAINER_ABORTED: "trainer_aborted",
            D1RunStatus.RUN_ABORTED: "run_aborted",
        }[status]
        raise gap(code, "event_kind", f"{status.value} is not complete run evidence")
    return status


def _read_d1_run_evidence(
    path: str | Path,
    *,
    expected_logical_run_id: str | None = None,
    expected_family_digest: str | None = None,
    expected_run_class: RunClass | None = None,
    expected_plan_digest: str | None = None,
    allow_incomplete: bool = False,
    journal_payload: bytes | None = None,
) -> D1RunEvidence:
    """Admit one D1 journal; incomplete states require explicit opt-in."""
    from entropic_substrate.data.batching import DATA_ORDER_DIGEST_BATCHES

    target = Path(path)
    try:
        events = (
            read_journal(target)
            if journal_payload is None
            else read_journal_bytes(journal_payload, journal_path=target)
        )
    except D1JournalError as error:
        raise gap(f"journal_{error.code}", error.path, str(error)) from error
    start_event = events[0]
    identity = start_event.identity
    _check_expected_identity(
        identity,
        expected_logical_run_id=expected_logical_run_id,
        expected_family_digest=expected_family_digest,
        expected_run_class=expected_run_class,
    )
    plan, plan_digest, first_segment = parse_run_start_event(start_event)
    try:
        target_update = strategies_for_arm(D1Arm(plan["arm"]))[0]
    except (TypeError, ValueError) as error:
        raise gap("plan_arm", "run_start.plan.arm", "is not a registered D1 arm") from error
    if expected_plan_digest is not None:
        require_sha256(expected_plan_digest, path="expected_plan_digest")
        if plan_digest != expected_plan_digest:
            raise gap("identity_mismatch", "plan_digest", "differs from the requested plan")
    status = _status(events, allow_incomplete=allow_incomplete)
    terminal = (
        events[-1]
        if events[-1].event_kind in {JournalEventKind.RUN_END, JournalEventKind.RUN_ABORT}
        else None
    )
    body = events[1:-1] if terminal is not None else events[1:]
    baselines: list[D1JournalEvent] = []
    evaluations: list[D1EvalEvidence] = []
    train_steps: list[D1TrainStepResult] = []
    checkpoints: list[tuple[int, str]] = []
    segments = [first_segment]
    tail_quarantine: list[D1TailQuarantineEvidence] = []
    current_segment = 0
    order_state: dict[int, tuple[str, int]] = {}
    last_eval_step = -1
    for event in body:
        kind = event.event_kind
        if kind is JournalEventKind.BASELINE_CALIBRATED:
            parse_baseline_event(event)
            baselines.append(event)
        elif kind is JournalEventKind.TRAIN:
            digest, count, result = parse_train_event(
                event,
                expected_target_update=target_update,
            )
            train_steps.append(result)
            prior = order_state.get(current_segment)
            expected_count = 1 if prior is None else min(prior[1] + 1, DATA_ORDER_DIGEST_BATCHES)
            if count != expected_count or (
                prior is not None and count == prior[1] and digest != prior[0]
            ):
                raise gap("data_order", "train.data-order", "segment-local chain is inconsistent")
            order_state[current_segment] = (digest, count)
        elif kind is JournalEventKind.EVAL:
            evidence = parse_eval_event(event, len(evaluations))
            if evidence.step <= last_eval_step:
                raise gap("eval_step", "eval.successful_step", "must be strictly increasing")
            if evaluations and evidence.quality.latent_width != evaluations[0].quality.latent_width:
                raise gap("vector_width", "eval.quality", "latent width changed across evaluations")
            segment = segments[current_segment]
            if evidence.resource_work.execution_segment_index != current_segment:
                raise gap("resource_segment", "eval.resource_work", "differs from active segment")
            if evidence.resource_work.execution_segment_id != segment.execution_segment_id:
                raise gap("resource_segment", "eval.resource_work", "segment identity differs")
            if evidence.resource_work.observed_at < segment.trainer_started_at:
                raise gap("resource_time", "eval.resource_work.observed_at", "precedes segment")
            evaluations.append(evidence)
            last_eval_step = evidence.step
        elif kind is JournalEventKind.CHECKPOINT:
            checkpoint = require_sha256(
                event.payload["checkpoint_sha256"], path="checkpoint_sha256"
            )
            if checkpoints and event.successful_step <= checkpoints[-1][0]:
                raise gap("checkpoint_order", "checkpoint", "receipt steps must increase")
            if any(checkpoint == prior for _, prior in checkpoints):
                raise gap("checkpoint_hash", "checkpoint_sha256", "receipt hash was reused")
            checkpoints.append((event.successful_step, checkpoint))
        elif kind is JournalEventKind.RUN_RESUME:
            if not checkpoints:
                raise gap("resume_checkpoint", "run_resume", "has no prior receipt")
            segment, quarantine = _parse_resume_event(
                event,
                previous_checkpoint_sha256=checkpoints[-1][1],
                expected_segment=len(segments),
                previous_segment=segments[-1],
            )
            segments.append(segment)
            tail_quarantine.append(quarantine)
            current_segment += 1
        elif kind is JournalEventKind.TRAINER_ABORTED:
            continue
        elif kind is JournalEventKind.RUN_ABORT:
            raise gap("journal_abort_order", "event_kind", "run_abort must be terminal")
        else:
            raise gap("unexpected_event", "event_kind", f"unexpected {kind.value!r}")

    if len(baselines) > 1 or (status is D1RunStatus.COMPLETE and len(baselines) != 1):
        raise gap("baseline_count", "baseline", "run requires exactly one calibration")
    if evaluations and evaluations[0].step != 0:
        raise gap("missing_step_zero_eval", "eval", "first evaluation must be step zero")
    if status is D1RunStatus.COMPLETE:
        if not evaluations or evaluations[0].step != 0:
            raise gap("missing_step_zero_eval", "eval", "complete run requires step-zero eval")
        if not checkpoints or checkpoints[-1][0] != evaluations[-1].step:
            raise gap("missing_final_checkpoint", "checkpoint", "does not receipt final evaluation")
        if len(events) < 3 or events[-2].event_kind is not JournalEventKind.CHECKPOINT:
            raise gap("missing_final_checkpoint", "event_kind", "run_end lacks final receipt")
        if terminal is None or terminal.successful_step != evaluations[-1].step:
            raise gap("terminal_step", "run_end.successful_step", "differs from final evaluation")
        if not order_state:
            raise gap("data_order", "train.data_order", "run has no measured training order")
    try:
        validate_execution_segments(tuple(segments))
    except (TypeError, ValueError) as error:
        raise gap("series_validation", "segment chain", str(error)) from error
    if evaluations:
        try:
            validate_resource_work_series(tuple(item.resource_work for item in evaluations))
        except (TypeError, ValueError) as error:
            raise gap("series_validation", "resource work series", str(error)) from error
    frozen_plan = start_event.payload["plan"]
    if not isinstance(frozen_plan, Mapping):  # pragma: no cover - parsed above
        raise AssertionError("validated plan lost its mapping representation")
    data_order = tuple((index, *order_state[index]) for index in sorted(order_state))
    return D1RunEvidence(
        path=target,
        status=status,
        identity=identity,
        plan_digest=plan_digest,
        plan_manifest=cast(Mapping[str, FrozenJson], frozen_plan),
        segments=tuple(segments),
        baseline=baselines[0] if baselines else None,
        evals=tuple(evaluations),
        train_steps=tuple(train_steps),
        data_order_segments=data_order,
        checkpoint_receipts=tuple(checkpoints),
        tail_quarantine_evidence=tuple(tail_quarantine),
        terminal=terminal,
    )


def read_d1_run_evidence(
    path: str | Path,
    *,
    expected_logical_run_id: str | None = None,
    expected_family_digest: str | None = None,
    expected_run_class: RunClass | None = None,
    expected_plan_digest: str | None = None,
    allow_incomplete: bool = False,
) -> D1RunEvidence:
    """Admit one D1 journal from bytes read by the authoritative reader."""
    return _read_d1_run_evidence(
        path,
        expected_logical_run_id=expected_logical_run_id,
        expected_family_digest=expected_family_digest,
        expected_run_class=expected_run_class,
        expected_plan_digest=expected_plan_digest,
        allow_incomplete=allow_incomplete,
        journal_payload=None,
    )


def read_d1_eval_series(path: str | Path, **expected: object) -> tuple[D1EvalEvidence, ...]:
    return read_d1_run_evidence(path, **expected).evals  # type: ignore[arg-type]


def final_d1_eval_metric(path: str | Path, metric: str, **expected: object) -> float:
    run = read_d1_run_evidence(path, **expected)  # type: ignore[arg-type]
    return run.evals[-1].quality.scalar_metric(metric)
