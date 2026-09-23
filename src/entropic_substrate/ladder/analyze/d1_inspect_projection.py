"""Projection from one authoritative journal snapshot to bounded inspect values."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from entropic_substrate.__legacy.d1_journal_snapshot import D1JournalSnapshot
from entropic_substrate.__legacy.evidence.d1_journal_values import D1JournalEvent
from entropic_substrate.__legacy.evidence.d1_telemetry_values import (
    D1_REPORT_ONLY_METRICS,
    D1_SCALAR_METRICS,
    D1EvalEvidence,
    D1QualityEvidence,
    D1RunStatus,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, canonical_digest
from entropic_substrate.ladder.analyze.d1_inspect_values import (
    D1_INSPECT_MAX_EVALUATIONS,
    D1InspectCheckpoint,
    D1InspectComplete,
    D1InspectCompletion,
    D1InspectEvaluation,
    D1InspectEvaluationWindow,
    D1InspectEventRef,
    D1InspectJournal,
    D1InspectPartial,
    D1InspectPlan,
    D1InspectProgress,
    D1InspectQualityScalars,
    D1InspectRun,
    D1InspectRunAborted,
    D1InspectState,
    D1InspectTrainerAborted,
    D1InspectView,
)

__all__ = ["compose_d1_inspect_view"]

_DECIDING_METRICS = D1_SCALAR_METRICS - D1_REPORT_ONLY_METRICS


def _quality(value: D1QualityEvidence) -> D1InspectQualityScalars:
    return D1InspectQualityScalars(
        deciding=MappingProxyType(
            {name: value.scalar_metric(name) for name in sorted(_DECIDING_METRICS)}
        ),
        report_only=MappingProxyType(
            {name: value.scalar_metric(name) for name in sorted(D1_REPORT_ONLY_METRICS)}
        ),
    )


def _evaluation(value: D1EvalEvidence) -> D1InspectEvaluation:
    quality = value.quality
    return D1InspectEvaluation(
        value.step,
        value.eval_index,
        quality.latent_width,
        _quality(quality),
        quality.pairwise_cosine_semantic,
        quality.target_predicted_exact_match_semantic,
        value.resource_work,
    )


def _event_ref(value: D1JournalEvent) -> D1InspectEventRef:
    return D1InspectEventRef(
        value.event_kind,
        value.event_seq,
        value.successful_step,
        canonical_digest(value.manifest),
    )


def _plan(value: Mapping[str, object]) -> D1InspectPlan:
    return D1InspectPlan(
        cast(str, value["profile_name"]),
        D1Arm(cast(str, value["arm"])),
        cast(int, value["seed"]),
        cast(str, value["pairing_id"]),
    )


def _state(snapshot: D1JournalSnapshot, progress: D1InspectProgress) -> D1InspectState:
    run = snapshot.run
    if run.status is D1RunStatus.COMPLETE:
        if progress.latest_quality is None or progress.latest_checkpoint is None:
            raise ValueError("complete D1 inspection lost completion evidence")
        if run.terminal is None:
            raise ValueError("complete D1 inspection lost RUN_END evidence")
        return D1InspectComplete(
            D1InspectCompletion(
                progress.latest_quality,
                progress.latest_checkpoint,
                _event_ref(run.terminal),
            )
        )
    if run.status is D1RunStatus.PARTIAL:
        return D1InspectPartial()
    if run.status is D1RunStatus.TRAINER_ABORTED:
        if run.trainer_abort is None:
            raise ValueError("trainer-aborted D1 inspection lost abort evidence")
        return D1InspectTrainerAborted(
            _event_ref(run.trainer_abort.event),
            run.trainer_abort.numerical_defect,
            None if run.terminal is None else _event_ref(run.terminal),
        )
    if run.terminal is None:
        raise ValueError("run-aborted D1 inspection lost RUN_ABORT evidence")
    return D1InspectRunAborted(_event_ref(run.terminal))


def compose_d1_inspect_view(
    snapshot: D1JournalSnapshot,
    *,
    evaluation_limit: int,
) -> D1InspectView:
    """Project one attested journal snapshot without rereading its source path."""
    if type(snapshot) is not D1JournalSnapshot:
        raise TypeError("D1 inspection requires a nominal journal snapshot")
    snapshot.require_factory_attested()
    if type(evaluation_limit) is not int or not 1 <= evaluation_limit <= D1_INSPECT_MAX_EVALUATIONS:
        raise ValueError("D1 inspect evaluation_limit is outside its hard bound")
    run = snapshot.run
    items = tuple(_evaluation(item) for item in run.evals[-evaluation_limit:])
    window = D1InspectEvaluationWindow(len(run.evals), evaluation_limit, items)
    checkpoint = (
        None if not run.checkpoint_receipts else D1InspectCheckpoint(*run.checkpoint_receipts[-1])
    )
    progress = D1InspectProgress(
        len(run.segments),
        len(run.data_order_segments),
        len(run.tail_quarantine_evidence),
        run.baseline is not None,
        len(run.evals),
        len(run.train_steps),
        len(run.checkpoint_receipts),
        run.segments[-1],
        items[-1] if items else None,
        checkpoint,
    )
    identity = run.identity
    return D1InspectView(
        D1InspectJournal(
            str(snapshot.path), snapshot.relative_path, snapshot.byte_count, snapshot.sha256
        ),
        D1InspectRun(
            identity.logical_run_id,
            identity.run_class,
            identity.execution_digest,
            identity.aggregation_digest,
            identity.construction_digest,
            identity.family_digest,
            run.plan_digest,
            run.authority_digest,
        ),
        _plan(cast(Mapping[str, object], run.plan_manifest)),
        progress,
        window,
        _state(snapshot, progress),
    )
