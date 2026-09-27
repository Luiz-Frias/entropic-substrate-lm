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
        self.finished = False

    def log(self, record: Mapping[str, Any]) -> None:
        self.records.append(record)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        raise AssertionError("H training-step routing must never call log_artifact")

    def finish(self) -> None:
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
