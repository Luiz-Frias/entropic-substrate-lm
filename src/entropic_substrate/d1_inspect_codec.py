"""Exact canonical wire codec for bounded D1 operator inspection results."""

from __future__ import annotations

import json
from typing import Final, NoReturn, cast

from entropic_substrate.d1_inspect_values import (
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
from entropic_substrate.d1_numerical_defect import terminal_numerical_defect_from_mapping
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.resource_provenance_codec import (
    ResourceProvenanceCodecGap,
    run_start_from_mapping,
    run_start_to_mapping,
)
from entropic_substrate.evidence.resource_work_codec import (
    ResourceWorkCodecGap,
    resource_snapshot_from_mapping,
    resource_snapshot_to_mapping,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

__all__ = [
    "D1_INSPECT_MAX_RESULT_BYTES",
    "D1_INSPECT_RESULT_SCHEMA",
    "D1InspectCodecError",
    "d1_inspect_result_from_mapping",
    "d1_inspect_result_to_mapping",
    "parse_d1_inspect_result",
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


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1InspectCodecError(message)
    raise D1InspectCodecError(message) from error


def _object(value: object, *, label: str, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"D1 inspect {label} must be an object with string keys")
    result = cast(dict[str, object], value)
    if set(result) != keys:
        _fail(f"D1 inspect {label} keys must be exact")
    return result


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"D1 inspect {label} must be a nonempty string")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"D1 inspect {label} must be a nonnegative integer")
    return value


def _floating_map(value: object, *, label: str) -> dict[str, float]:
    if type(value) is not dict or any(
        type(key) is not str or type(item) is not float
        for key, item in cast(dict[object, object], value).items()
    ):
        _fail(f"D1 inspect {label} must contain string-to-float entries")
    return cast(dict[str, float], value)


def _event_from_mapping(value: object, *, label: str) -> D1InspectEventRef:
    item = _object(
        value,
        label=label,
        keys={"event_kind", "event_seq", "event_sha256", "successful_step"},
    )
    try:
        kind = JournalEventKind(item["event_kind"])
    except (TypeError, ValueError) as error:
        _fail(f"D1 inspect {label} event kind is unsupported", error)
    return D1InspectEventRef(
        kind,
        _integer(item["event_seq"], label=f"{label}.event_seq"),
        _integer(item["successful_step"], label=f"{label}.successful_step"),
        _text(item["event_sha256"], label=f"{label}.event_sha256"),
    )


def _checkpoint_from_mapping(
    value: object, *, label: str, optional: bool
) -> D1InspectCheckpoint | None:
    if value is None and optional:
        return None
    item = _object(value, label=label, keys={"sha256", "successful_step"})
    return D1InspectCheckpoint(
        _integer(item["successful_step"], label=f"{label}.successful_step"),
        _text(item["sha256"], label=f"{label}.sha256"),
    )


def _cursor_from_mapping(value: object, *, label: str, optional: bool) -> tuple[int, int] | None:
    if value is None and optional:
        return None
    item = _object(value, label=label, keys={"eval_index", "successful_step"})
    return (
        _integer(item["successful_step"], label=f"{label}.successful_step"),
        _integer(item["eval_index"], label=f"{label}.eval_index"),
    )


def _evaluation_from_mapping(value: object, *, label: str) -> D1InspectEvaluation:
    item = _object(
        value,
        label=label,
        keys={
            "deciding_quality_scalars",
            "eval_index",
            "latent_width",
            "quality_semantics",
            "report_only_quality_scalars",
            "resource_work",
            "successful_step",
        },
    )
    semantics = _object(
        item["quality_semantics"],
        label=f"{label}.quality_semantics",
        keys={"pairwise_cosine", "target_predicted_exact_match"},
    )
    quality = D1InspectQualityScalars(
        _floating_map(item["deciding_quality_scalars"], label=f"{label}.deciding"),
        _floating_map(item["report_only_quality_scalars"], label=f"{label}.report_only"),
    )
    return D1InspectEvaluation(
        _integer(item["successful_step"], label=f"{label}.successful_step"),
        _integer(item["eval_index"], label=f"{label}.eval_index"),
        _integer(item["latent_width"], label=f"{label}.latent_width"),
        quality,
        _text(semantics["pairwise_cosine"], label=f"{label}.pairwise semantic"),
        _text(
            semantics["target_predicted_exact_match"],
            label=f"{label}.target-predicted semantic",
        ),
        resource_snapshot_from_mapping(item["resource_work"]),
    )


def _window_from_mapping(value: object) -> D1InspectEvaluationWindow:
    item = _object(
        value,
        label="evaluations",
        keys={"items", "limit", "omitted_count", "returned_count", "total_count"},
    )
    raw_items = item["items"]
    if type(raw_items) is not list:
        _fail("D1 inspect evaluations.items must be an array")
    values = tuple(
        _evaluation_from_mapping(entry, label=f"evaluations.items[{index}]")
        for index, entry in enumerate(cast(list[object], raw_items))
    )
    window = D1InspectEvaluationWindow(
        _integer(item["total_count"], label="evaluations.total_count"),
        _integer(item["limit"], label="evaluations.limit"),
        values,
    )
    if (
        _integer(item["returned_count"], label="evaluations.returned_count")
        != window.returned_count
        or _integer(item["omitted_count"], label="evaluations.omitted_count")
        != window.omitted_count
    ):
        _fail("D1 inspect evaluation window counts differ from its items")
    return window


def _progress_from_mapping(value: object, window: D1InspectEvaluationWindow) -> D1InspectProgress:
    item = _object(
        value,
        label="progress",
        keys={
            "baseline_present",
            "checkpoint_count",
            "data_order_segment_count",
            "evaluation_count",
            "latest_checkpoint",
            "latest_evaluation",
            "latest_segment",
            "segment_count",
            "tail_quarantine_count",
            "train_step_count",
        },
    )
    latest = window.items[-1] if window.items else None
    cursor = _cursor_from_mapping(
        item["latest_evaluation"], label="latest_evaluation", optional=True
    )
    expected = None if latest is None else (latest.step, latest.eval_index)
    if cursor != expected:
        _fail("D1 inspect latest evaluation cursor differs from the terminal window item")
    baseline = item["baseline_present"]
    if type(baseline) is not bool:
        _fail("D1 inspect baseline_present must be a bool")
    return D1InspectProgress(
        _integer(item["segment_count"], label="progress.segment_count"),
        _integer(item["data_order_segment_count"], label="progress.data_order_segment_count"),
        _integer(item["tail_quarantine_count"], label="progress.tail_quarantine_count"),
        baseline,
        _integer(item["evaluation_count"], label="progress.evaluation_count"),
        _integer(item["train_step_count"], label="progress.train_step_count"),
        _integer(item["checkpoint_count"], label="progress.checkpoint_count"),
        run_start_from_mapping(item["latest_segment"]),
        latest,
        _checkpoint_from_mapping(
            item["latest_checkpoint"], label="latest_checkpoint", optional=True
        ),
    )


def _state_from_mapping(value: object, progress: D1InspectProgress) -> D1InspectState:
    if type(value) is not dict:
        _fail("D1 inspect state must be an object")
    status = _text(cast(dict[str, object], value).get("status"), label="state status")
    if status == "partial":
        _object(value, label="state", keys={"status"})
        return D1InspectPartial()
    if status == "complete":
        item = _object(value, label="state", keys={"completion", "status"})
        completion = _object(
            item["completion"],
            label="state.completion",
            keys={"final_checkpoint", "final_evaluation", "run_end"},
        )
        cursor = _cursor_from_mapping(
            completion["final_evaluation"], label="final_evaluation", optional=False
        )
        latest = progress.latest_quality
        if latest is None or cursor != (latest.step, latest.eval_index):
            _fail("D1 inspect final evaluation differs from latest progress")
        checkpoint = _checkpoint_from_mapping(
            completion["final_checkpoint"], label="final_checkpoint", optional=False
        )
        if checkpoint is None:  # pragma: no cover - optional=False proves this
            raise AssertionError("required checkpoint decoded as absent")
        return D1InspectComplete(
            D1InspectCompletion(
                latest,
                checkpoint,
                _event_from_mapping(completion["run_end"], label="run_end"),
            )
        )
    if status == "trainer_aborted":
        item = _object(
            value,
            label="state",
            keys={"numerical_defect", "run_finalization", "status", "trainer_abort"},
        )
        raw_defect = item["numerical_defect"]
        return D1InspectTrainerAborted(
            _event_from_mapping(item["trainer_abort"], label="trainer_abort"),
            None if raw_defect is None else terminal_numerical_defect_from_mapping(raw_defect),
            (
                None
                if item["run_finalization"] is None
                else _event_from_mapping(item["run_finalization"], label="run_finalization")
            ),
        )
    if status == "run_aborted":
        item = _object(value, label="state", keys={"run_abort", "status"})
        return D1InspectRunAborted(_event_from_mapping(item["run_abort"], label="run_abort"))
    _fail("D1 inspect state status is unsupported")


def _from_mapping(value: object) -> D1InspectView:
    root = _object(
        value,
        label="root",
        keys={"evaluations", "journal", "plan", "progress", "run", "schema_version", "state"},
    )
    if root["schema_version"] != D1_INSPECT_RESULT_SCHEMA:
        _fail("D1 inspect result schema is unsupported")
    journal = _object(
        root["journal"],
        label="journal",
        keys={"byte_count", "path", "relative_path", "sha256"},
    )
    run = _object(
        root["run"],
        label="run",
        keys={
            "aggregation_digest",
            "authority_digest",
            "construction_digest",
            "execution_digest",
            "family_digest",
            "logical_run_id",
            "plan_digest",
            "run_class",
        },
    )
    plan = _object(root["plan"], label="plan", keys={"arm", "pairing_id", "profile_name", "seed"})
    window = _window_from_mapping(root["evaluations"])
    progress = _progress_from_mapping(root["progress"], window)
    return D1InspectView(
        D1InspectJournal(
            _text(journal["path"], label="journal.path"),
            _text(journal["relative_path"], label="journal.relative_path"),
            _integer(journal["byte_count"], label="journal.byte_count"),
            _text(journal["sha256"], label="journal.sha256"),
        ),
        D1InspectRun(
            _text(run["logical_run_id"], label="run.logical_run_id"),
            RunClass(_text(run["run_class"], label="run.run_class")),
            _text(run["execution_digest"], label="run.execution_digest"),
            _text(run["aggregation_digest"], label="run.aggregation_digest"),
            _text(run["construction_digest"], label="run.construction_digest"),
            _text(run["family_digest"], label="run.family_digest"),
            _text(run["plan_digest"], label="run.plan_digest"),
            _text(run["authority_digest"], label="run.authority_digest"),
        ),
        D1InspectPlan(
            _text(plan["profile_name"], label="plan.profile_name"),
            D1Arm(_text(plan["arm"], label="plan.arm")),
            _integer(plan["seed"], label="plan.seed"),
            _text(plan["pairing_id"], label="plan.pairing_id"),
        ),
        progress,
        window,
        _state_from_mapping(root["state"], progress),
    )


def d1_inspect_result_from_mapping(value: object) -> D1InspectView:
    """Reconstruct one nominal result only after exact nested admission."""
    try:
        return _from_mapping(value)
    except D1InspectCodecError:
        raise
    except (ResourceProvenanceCodecGap, ResourceWorkCodecGap, TypeError, ValueError) as error:
        _fail(f"D1 inspect result is invalid: {error}", error)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"D1 inspect result contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_d1_inspect_result(payload: bytes) -> D1InspectView:
    """Admit one canonical UTF-8 JSON result under the same byte envelope."""
    if type(payload) is not bytes or not payload:
        _fail("D1 inspect result payload must be nonempty bytes")
    if len(payload) > D1_INSPECT_MAX_RESULT_BYTES:
        _fail("D1 inspect result payload exceeds its byte limit")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(
                f"D1 inspect result contains non-finite JSON constant {constant}"
            ),
        )
    except D1InspectCodecError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("D1 inspect result payload must be UTF-8 JSON", error)
    result = d1_inspect_result_from_mapping(value)
    if serialize_d1_inspect_result(result) != payload:
        _fail("D1 inspect result payload must use canonical JSON")
    return result
