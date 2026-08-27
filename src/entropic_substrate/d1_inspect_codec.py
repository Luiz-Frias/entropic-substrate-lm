"""Exact canonical wire codec for bounded D1 operator inspection results."""

from __future__ import annotations

import json
from typing import Final

from entropic_substrate.d1_inspect_values import (
    D1InspectCheckpoint,
    D1InspectComplete,
    D1InspectEvaluation,
    D1InspectEventRef,
    D1InspectPartial,
    D1InspectRunAborted,
    D1InspectTrainerAborted,
    D1InspectView,
)
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.evidence.resource_work_codec import resource_snapshot_to_mapping

__all__ = [
    "D1_INSPECT_MAX_RESULT_BYTES",
    "D1_INSPECT_RESULT_SCHEMA",
    "D1InspectCodecError",
    "d1_inspect_result_to_mapping",
    "serialize_d1_inspect_result",
]

D1_INSPECT_RESULT_SCHEMA: Final = "d1-inspect-result-v1"
D1_INSPECT_MAX_RESULT_BYTES: Final = 1_048_576


class D1InspectCodecError(ValueError):
    """The inspect result is noncanonical, incomplete, or outside its byte bound."""


def _event(value: D1InspectEventRef) -> dict[str, object]:
    if type(value) is not D1InspectEventRef:
        raise TypeError("D1 inspect event reference must be nominal")
    return {
        "event_kind": value.event_kind.value,
        "event_seq": value.event_seq,
        "event_sha256": value.event_sha256,
        "successful_step": value.successful_step,
    }


def _checkpoint(value: D1InspectCheckpoint | None) -> dict[str, object] | None:
    if value is None:
        return None
    if type(value) is not D1InspectCheckpoint:
        raise TypeError("D1 inspect checkpoint must be nominal")
    return {"sha256": value.sha256, "successful_step": value.successful_step}


def _cursor(value: D1InspectEvaluation | None) -> dict[str, int] | None:
    if value is None:
        return None
    if type(value) is not D1InspectEvaluation:
        raise TypeError("D1 inspect evaluation cursor must be nominal")
    return {"eval_index": value.eval_index, "successful_step": value.step}


def _evaluation(value: D1InspectEvaluation) -> dict[str, object]:
    if type(value) is not D1InspectEvaluation:
        raise TypeError("D1 inspect evaluation must be nominal")
    return {
        "deciding_quality_scalars": dict(value.quality.deciding),
        "eval_index": value.eval_index,
        "latent_width": value.latent_width,
        "quality_semantics": {
            "pairwise_cosine": value.pairwise_cosine_semantic,
            "target_predicted_exact_match": value.target_predicted_exact_match_semantic,
        },
        "report_only_quality_scalars": dict(value.quality.report_only),
        "resource_work": resource_snapshot_to_mapping(value.resource_work),
        "successful_step": value.step,
    }


def _state(view: D1InspectView) -> dict[str, object]:
    state = view.state
    if type(state) is D1InspectComplete:
        completion = state.completion
        return {
            "completion": {
                "final_checkpoint": _checkpoint(completion.final_checkpoint),
                "final_evaluation": _cursor(completion.final_quality),
                "run_end": _event(completion.run_end),
            },
            "status": "complete",
        }
    if type(state) is D1InspectPartial:
        return {"status": "partial"}
    if type(state) is D1InspectTrainerAborted:
        return {
            "numerical_defect": (
                None if state.numerical_defect is None else state.numerical_defect.manifest
            ),
            "run_finalization": (
                None if state.run_finalization is None else _event(state.run_finalization)
            ),
            "status": "trainer_aborted",
            "trainer_abort": _event(state.trainer_abort),
        }
    if type(state) is D1InspectRunAborted:
        return {"run_abort": _event(state.run_abort), "status": "run_aborted"}
    raise TypeError("D1 inspect state must be nominal")


def _mapping(view: D1InspectView) -> dict[str, object]:
    if type(view) is not D1InspectView:
        raise TypeError("D1 inspect result codec requires D1InspectView")
    journal, run, plan, progress, evaluations = (
        view.journal,
        view.run,
        view.plan,
        view.progress,
        view.evaluations,
    )
    return {
        "evaluations": {
            "items": [_evaluation(item) for item in evaluations.items],
            "limit": evaluations.limit,
            "omitted_count": evaluations.omitted_count,
            "returned_count": evaluations.returned_count,
            "total_count": evaluations.total_count,
        },
        "journal": {
            "byte_count": journal.byte_count,
            "path": journal.path,
            "relative_path": journal.relative_path,
            "sha256": journal.sha256,
        },
        "plan": {
            "arm": plan.arm.value,
            "pairing_id": plan.pairing_id,
            "profile_name": plan.profile_name,
            "seed": plan.seed,
        },
        "progress": {
            "baseline_present": progress.baseline_present,
            "checkpoint_count": progress.total_checkpoint_count,
            "data_order_segment_count": progress.data_order_segment_count,
            "evaluation_count": progress.total_evaluation_count,
            "latest_checkpoint": _checkpoint(progress.latest_checkpoint),
            "latest_evaluation": _cursor(progress.latest_quality),
            "latest_segment": run_start_to_mapping(progress.latest_segment),
            "segment_count": progress.segment_count,
            "tail_quarantine_count": progress.tail_quarantine_count,
            "train_step_count": progress.total_train_step_count,
        },
        "run": {
            "aggregation_digest": run.aggregation_digest,
            "authority_digest": run.authority_digest,
            "construction_digest": run.construction_digest,
            "execution_digest": run.execution_digest,
            "family_digest": run.family_digest,
            "logical_run_id": run.logical_run_id,
            "plan_digest": run.plan_digest,
            "run_class": run.run_class.value,
        },
        "schema_version": D1_INSPECT_RESULT_SCHEMA,
        "state": _state(view),
    }


def _canonical(mapping: dict[str, object]) -> bytes:
    try:
        payload = json.dumps(
            mapping,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:  # pragma: no cover - nominal values prove this
        raise D1InspectCodecError("D1 inspect result is not finite canonical JSON") from error
    if len(payload) > D1_INSPECT_MAX_RESULT_BYTES:
        raise D1InspectCodecError("D1 inspect result exceeds its canonical byte limit")
    return payload


def d1_inspect_result_to_mapping(view: D1InspectView) -> dict[str, object]:
    """Return one exact mapping only after enforcing the wire-size envelope."""
    mapping = _mapping(view)
    _canonical(mapping)
    return mapping


def serialize_d1_inspect_result(view: D1InspectView) -> bytes:
    """Serialize one exact finite result under the independent 1 MiB bound."""
    return _canonical(_mapping(view))
