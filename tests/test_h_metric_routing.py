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
    # test_sentence_runner.py's own pinned schedule for this profile.
    assert [row["step"] for row in sink.records] == [1, 2, 1, 2, 3, 4]
    assert all(row["schema"] == "h-training-step-v2" for row in sink.records)
    assert all(row["status"] == "committed" for row in sink.records)
    assert {"objective", "aggregate_norm", "decoder_ce"} <= sink.records[0].keys()
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
    extractor = HTrainingStepExtractor()
    row = {
        "schema": "h-training-step-v2",
        "status": "committed",
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
        "objective": 1.25,
        "aggregate_norm": 0.5,
        "decoder_ce": 1.1,
        "wall_seconds": 0.02,
    }


def test_extractor_ignores_non_h_and_uncommitted_and_nonfinite_events() -> None:
    extractor = HTrainingStepExtractor()
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
            "step": 1,
            "objective": float("nan"),
            "aggregate_norm": float("inf"),
        }
    )
    assert non_finite is not None
    assert non_finite.metrics == {}
