"""H lane's additive MetricSink routing (RUNG 2b, conductor CONTINUATION #7).

The D1 journal (``ladder.run.journal``) stays the authoritative,
content-addressed record; this module tests the SEPARATE, additive path
that lets a :class:`MetricSink` observe the same committed training-step
rows for an operational mirror, without changing what the journal writes
or how the sha256 chain resolves. Omitting ``sink`` must reproduce exactly
today's behaviour -- that is the guard-teeth companion for "additive,
never a substitution."
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.test_sentence_execution import engine

from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor
from entropic_substrate.ladder.run.runner import run_probe_training

pytestmark = pytest.mark.unit

LIMIT = 10_000_000


class _FakeSink:
    """A minimal MetricSink double -- records exactly what it is given."""

    def __init__(self) -> None:
        self.records: list[Mapping[str, Any]] = []
        # DN-1643-PORT: annotate rows are kept apart from `records`, which stays the
        # schema-bearing stream every assertion below indexes by ``row["schema"]``.
        self.annotations: list[dict[str, str]] = []
        self.finished = False

    def log(self, record: Mapping[str, Any]) -> None:
        self.records.append(record)

    def annotate(self, tags: Mapping[str, str]) -> None:
        self.annotations.append(dict(tags))

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        raise AssertionError("H training-step routing must never call log_artifact")

    def finish(self, *, status: str = "FINISHED") -> None:
        self.finished = True


def test_committed_training_steps_reach_the_sink_unmodified(tmp_path: Path) -> None:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    sink = _FakeSink()

    result = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        sink=sink,
    )

    assert result.status == "training_complete"
    # 2 codec_warmup steps + 4 prediction steps, per
    # test_sentence_runner.py's own pinned schedule for this profile. RESTATED
    # 2026-09-26 (exit-gate lane): the sink now also receives the sealed
    # evaluation reports (tested below), so the training-row assertions filter
    # on the training schema instead of assuming the sink sees nothing else.
    training = [row for row in sink.records if row["schema"] == "h-training-step-v2"]
    assert [row["step"] for row in training] == [1, 2, 1, 2, 3, 4]
    assert {row["schema"] for row in sink.records} == {
        "h-training-step-v2",
        "h-predictive-evaluation-v1",
    }
    assert all(row["status"] == "committed" for row in training)
    assert {"objective", "aggregate_norm", "decoder_ce"} <= training[0].keys()
    assert not sink.finished, "run_probe_training does not own the sink's lifecycle"


def test_no_sink_is_the_default_and_training_still_completes(tmp_path: Path) -> None:
    """Guard-teeth companion: omitting sink reproduces pre-2b behaviour exactly."""
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()

    result = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )

    assert result.status == "training_complete"


def test_extractor_projects_only_finite_scalars_the_row_already_carries() -> None:
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    row = {
        "schema": "h-training-step-v2",
        "status": "committed",
        "phase": "codec_warmup",
        "step": 3,
        "objective": 1.25,
        "aggregate_norm": 0.5,
        "decoder_ce": 1.1,
        "future_nll_per_dimension": None,
        "wall_seconds": 0.02,
        "source_batch": "abc",
    }

    extracted = extractor.extract(row)

    assert extracted is not None
    assert extracted.step == 3
    assert extracted.metrics == {
        "phase_index": 0.0,
        "objective": 1.25,
        "aggregate_norm": 0.5,
        "decoder_ce": 1.1,
        "wall_seconds": 0.02,
    }


def test_extractor_ignores_non_h_and_uncommitted_and_nonfinite_events() -> None:
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    assert extractor.extract("not-a-mapping") is None
    assert extractor.extract({}) is None
    assert extractor.extract({"schema": "other", "status": "committed", "step": 1}) is None
    assert (
        extractor.extract({"schema": "h-training-step-v2", "status": "started", "step": 1}) is None
    )
    assert (
        extractor.extract({"schema": "h-training-step-v2", "status": "committed", "step": "x"})
        is None
    )
    non_finite = extractor.extract(
        {
            "schema": "h-training-step-v2",
            "status": "committed",
            "phase": "prediction",
            "step": 1,
            "objective": float("nan"),
            "aggregate_norm": float("inf"),
        }
    )
    assert non_finite is not None
    assert non_finite.step == 1001
    assert non_finite.metrics == {"phase_index": 1.0}


# Appended to tests/test_h_metric_routing.py (RED-first for commit 2 of the exit-gate lane).


def test_extractor_reports_the_global_optimizer_step_across_phases() -> None:
    """EXIT-GATE LANE (operator-cleared 2026-09-26): the engine resets ``step`` to 0
    at ``begin_prediction`` (ladder/run/execution.py), so under the v1 extractor
    warmup step 3 and prediction step 3 landed on the same MLflow x-coordinate.
    The extractor now takes the profile's warmup length and reports the GLOBAL
    optimizer step: codec_warmup rows keep their step, prediction rows are offset
    by ``warmup_steps``. Mutation arm: an extractor built without the offset (or
    a prediction row without ``phase``) must not report step 3 for both rows."""
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    warm = extractor.extract(
        {
            "schema": "h-training-step-v2",
            "status": "committed",
            "phase": "codec_warmup",
            "step": 3,
            "objective": 1.5,
        }
    )
    pred = extractor.extract(
        {
            "schema": "h-training-step-v2",
            "status": "committed",
            "phase": "prediction",
            "step": 3,
            "objective": 1.25,
        }
    )
    assert warm is not None and pred is not None
    assert warm.step == 3 and pred.step == 1003
    assert warm.metrics["phase_index"] == 0.0 and pred.metrics["phase_index"] == 1.0
    # A committed row without a phase cannot be placed on the global axis: refused.
    assert (
        extractor.extract(
            {"schema": "h-training-step-v2", "status": "committed", "step": 3, "objective": 1.0}
        )
        is None
    )


def test_extractor_projects_evaluation_reports_per_sample_count() -> None:
    """Evaluation reports (``h-predictive-evaluation-v1``) reach MLflow as scalar
    summaries per importance sample count: the mean finite log-marginal estimate,
    the mean effective sample size, the count of zero-mass gap rows, plus the
    evaluation wall clock, all at the global step."""
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    report = {
        "schema": "h-predictive-evaluation-v1",
        "meaning": "importance_weighted_log_marginal_estimate",
        "run_identity": "x",
        "step": 2000,
        "wall_seconds": 4.5,
        "cuda_seconds": None,
        "examples": [
            {
                "by_sample_count": {
                    "2": [
                        {
                            "log_estimate": -10.0,
                            "effective_sample_size": 1.5,
                            "likelihood_relative_mcse": 0.1,
                            "gap": None,
                        },
                        {
                            "log_estimate": -12.0,
                            "effective_sample_size": 1.9,
                            "likelihood_relative_mcse": 0.2,
                            "gap": None,
                        },
                    ],
                    "4": [
                        {
                            "log_estimate": None,
                            "effective_sample_size": None,
                            "likelihood_relative_mcse": None,
                            "gap": "zero_importance_weight_mass",
                        },
                        {
                            "log_estimate": -9.0,
                            "effective_sample_size": 3.0,
                            "likelihood_relative_mcse": 0.05,
                            "gap": None,
                        },
                    ],
                }
            }
        ],
    }
    out = extractor.extract(report)
    assert out is not None and out.step == 3000
    assert out.metrics["eval/log_marginal_k2"] == pytest.approx(-11.0)
    assert out.metrics["eval/ess_k2"] == pytest.approx(1.7)
    assert out.metrics["eval/gap_rows_k2"] == 0.0
    assert out.metrics["eval/log_marginal_k4"] == pytest.approx(-9.0)
    assert out.metrics["eval/gap_rows_k4"] == 1.0
    assert out.metrics["eval/wall_seconds"] == pytest.approx(4.5)
    assert "eval/cuda_seconds" not in out.metrics
    # Non-finite or non-evaluation rows are still refused.
    assert extractor.extract({"schema": "h-predictive-evaluation-v1", "step": "x"}) is None


def test_evaluation_reports_reach_the_sink_after_their_receipt(tmp_path: Path) -> None:
    """Guard teeth: ``run_probe_training`` logs each evaluation report to the sink
    right after ``read_evaluation`` admits it (never before, never a report the
    reader refused). Mutation arm: dropping the ``sink.log(report)`` call leaves
    the sink with training rows only and this test RED."""
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    sink = _FakeSink()
    result = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        sink=sink,
    )
    assert result.status == "training_complete"
    kinds = [row["schema"] for row in sink.records]
    evaluations = [row for row in sink.records if row["schema"] == "h-predictive-evaluation-v1"]
    assert evaluations, kinds
    # The pinned test profile evaluates at every evaluation_interval of its 4
    # prediction steps and at the final step; each report follows its own step's
    # training row in the sink's order.
    steps_seen = [row["step"] for row in sink.records if row["schema"] == "h-training-step-v2"]
    for report in evaluations:
        assert report["step"] in steps_seen
        assert report["meaning"] == "importance_weighted_log_marginal_estimate"


def test_lifecycle_records_never_reach_the_mlflow_metric_surface() -> None:
    """FOR-1639: h-run-lifecycle-v1 records are stream rows only. The extractor
    projects exactly two schemas and returns None for anything else, and the
    MlflowSink drops a None projection, so the lifecycle records ride the
    authoritative JSONL without becoming MLflow metrics. Characterization pin
    over hand-built records (GREEN before the emitter exists); its teeth are
    mutation M5, which teaches the extractor the schema."""
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    start = {
        "schema": "h-run-lifecycle-v1",
        "event": "run_start",
        "request_sha256": "a" * 64,
        "runtime": {"kind": "local_tiny"},
    }
    end = {
        "schema": "h-run-lifecycle-v1",
        "event": "run_end",
        "status": "ok",
        "result_status": "probe_workload_complete",
        "wall_seconds": 1.5,
        "successful_optimizer_steps": 6,
    }
    assert extractor.extract(start) is None
    assert extractor.extract(end) is None
    # a step row with the same payload fields DOES project: only the schema
    # keeps the lifecycle records off the metric surface
    step = {
        "schema": "h-training-step-v2",
        "status": "committed",
        "phase": "prediction",
        "step": 1,
        "objective": 1.0,
        "wall_seconds": 1.5,
    }
    assert extractor.extract(step) is not None
    assert extractor.extract({**step, "schema": "h-run-lifecycle-v1"}) is None


def test_the_workers_lifecycle_records_are_not_projected_either(tmp_path: Path) -> None:
    """Emitter-sourced companion: the records the REAL worker writes (not a
    hand-built dict) yield None from the extractor, while its step rows still
    project. RED before the emitter exists: there is no lifecycle record."""
    from tests.test_d1_trainer_object_transport import _ObjectStore
    from tests.test_sentence_worker_execution import prepared

    from entropic_substrate.ladder.run.object_worker import run_object_worker
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    root = tmp_path / "object-worker"
    root.mkdir()
    sink = _FakeSink()

    run_object_worker(objects, marker, root, max_request_bytes=1_000_000, sink=sink)

    extractor = HTrainingStepExtractor(warmup_steps=request.profile.schedule.warmup_steps)
    lifecycle = [r for r in sink.records if r.get("schema") == "h-run-lifecycle-v1"]
    assert [r["event"] for r in lifecycle] == ["run_start", "run_end"]
    assert all(extractor.extract(r) is None for r in lifecycle)
    steps = [r for r in sink.records if r.get("schema") == "h-training-step-v2"]
    assert steps and all(extractor.extract(r) is not None for r in steps)
    # FOR-1642: the checkpoint evidence the worker annotates yields nothing either, in
    # the shape JsonlSink writes it (`{"event": "annotate", **tags}`, no schema key).
    assert sink.annotations
    assert all(extractor.extract({"event": "annotate", **t}) is None for t in sink.annotations)


def test_an_annotate_record_yields_no_metrics_from_the_h_extractor() -> None:
    """FOR-1642 (DN-1643-PORT): annotate rows share the authoritative JSONL with the
    step rows, but they carry no ``schema``; the extractor must project nothing from
    them, so a checkpoint annotation can never become a metric point."""
    extractor = HTrainingStepExtractor(warmup_steps=2)
    annotate = {
        "event": "annotate",
        "checkpoint/00000002/key": "run/snapshots/checkpoint-00000002.json",
        "checkpoint/00000002/size_bytes": "812",
    }
    assert extractor.extract(annotate) is None
    # Control: the same extractor does project a committed step row.
    step = {"schema": "h-training-step-v2", "status": "committed", "phase": "prediction", "step": 1}
    assert extractor.extract(step) is not None


# ---------------------------------------------------------------------------
# FOR-1669: final-leg stage/admission measurements, projected onto the LAST
# global step a training row placed on the axis (never a guessed step).
# ---------------------------------------------------------------------------
def _training_row(phase: str, step: int, *, status: str = "committed") -> dict[str, object]:
    return {"schema": "h-training-step-v2", "status": status, "phase": phase, "step": step}


def _stage_end(stage: str = "generation", seconds: object = 12.5) -> dict[str, object]:
    return {
        "schema": "h-worker-stage-v1",
        "event": "stage.end",
        "stage": stage,
        "seconds": seconds,
    }


def _admission_end(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "h-worker-stage-v1",
        "event": "admission.end",
        "caller": "generation",
        "root": "run",
        "memo": "shared",
        "seals_walked": 0,
        "evaluations": 3,
        "journals_read": 0,
        "seconds": 155.25,
    }
    record.update(overrides)
    return record


def test_a_stage_end_projects_its_seconds_onto_the_last_training_global_step() -> None:
    """The axis assertion carries the teeth for mutation M4: a prediction row at raw step 2
    with a 1000-step warmup sits at global step 1002, which is neither 0 nor the raw 2."""
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    placed = extractor.extract(_training_row("prediction", 2))
    assert placed is not None and placed.step == 1002

    out = extractor.extract(_stage_end("generation", 12.5))

    assert out is not None
    assert out.step == 1002
    assert out.metrics == {"stage/generation/seconds": 12.5}


def test_an_admission_end_projects_its_four_measurements_onto_the_same_step() -> None:
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    extractor.extract(_training_row("prediction", 2))

    out = extractor.extract(_admission_end())

    assert out is not None
    assert out.step == 1002
    assert out.metrics == {
        "admission/generation/seconds": 155.25,
        "admission/generation/seals_walked": 0.0,
        "admission/generation/evaluations": 3.0,
        "admission/generation/journals_read": 0.0,
    }


def test_a_stage_record_before_any_training_row_is_refused_not_guessed() -> None:
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    assert extractor.extract(_stage_end()) is None
    assert extractor.extract(_admission_end()) is None
    # Control: the same extractor projects the same record once a training row placed the axis.
    extractor.extract(_training_row("codec_warmup", 3))
    out = extractor.extract(_stage_end())
    assert out is not None and out.step == 3


def test_only_a_committed_training_row_moves_the_stage_axis() -> None:
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    extractor.extract(_training_row("prediction", 2))
    # An uncommitted row, a row with no phase and a sealed evaluation report all carry a
    # step of their own; none of them is a training row the axis may follow.
    assert extractor.extract(_training_row("prediction", 9, status="pending")) is None
    assert (
        extractor.extract({"schema": "h-training-step-v2", "status": "committed", "step": 9})
        is None
    )
    report = {
        "schema": "h-predictive-evaluation-v1",
        "step": 5,
        "examples": [
            {"by_sample_count": {"2": [{"log_estimate": -1.0, "effective_sample_size": 1.0}]}}
        ],
    }
    evaluated = extractor.extract(report)
    assert evaluated is not None and evaluated.step == 1005

    out = extractor.extract(_stage_end())
    assert out is not None and out.step == 1002

    extractor.extract(_training_row("prediction", 3))
    moved = extractor.extract(_stage_end())
    assert moved is not None and moved.step == 1003


@pytest.mark.parametrize(
    "record",
    [
        _stage_end("not_a_stage"),
        _stage_end("generation", float("nan")),
        _stage_end("generation", float("inf")),
        _stage_end("generation", -1.0),
        _stage_end("generation", True),
        _stage_end("generation", "12.5"),
        _stage_end("generation", None),
        {**_stage_end(), "event": "stage.start"},
        {**_stage_end(), "event": "evaluation.end"},
        {k: v for k, v in _stage_end().items() if k != "seconds"},
        _admission_end(caller="bogus"),
        _admission_end(root="/home/operator/run"),
        _admission_end(memo="stale"),
        _admission_end(seconds=float("nan")),
        _admission_end(seals_walked=-1),
        _admission_end(evaluations=True),
        _admission_end(journals_read=1.5),
        {**_admission_end(), "event": "admission.start"},
        {k: v for k, v in _admission_end().items() if k != "journals_read"},
    ],
)
def test_an_out_of_vocabulary_or_malformed_stage_record_is_refused(
    record: dict[str, object],
) -> None:
    extractor = HTrainingStepExtractor(warmup_steps=1000)
    extractor.extract(_training_row("prediction", 2))
    assert extractor.extract(record) is None
