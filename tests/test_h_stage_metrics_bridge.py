"""FOR-1669: the final leg's stage and admission measurements reach MLflow.

MEASURED 2026-10-19 on the third paired 1K probe (wave ``h0-plumbing-1k-v3.HEpP7g``,
MLflow queried live 12 minutes into the final leg): training metrics, eval rows and the
checkpoint evidence tags were in MLflow; the stages ``training_readback``, ``generation``,
``span_diagnostics``, ``trace_capture``, ``native_verification`` and their admission walks
were in NEITHER a metric nor a tag, only in the ``stage.*`` / ``admission.*`` log lines. A
host with no pod-log access sees RUNNING plus ``system/*`` and cannot tell a stage from a
hang (the 2026-10-12 and 2026-10-17 final-leg wedges were diagnosable only from pod stdout).

``build_sink_for_run`` installs ONE loguru handler beside the sink when it is given a
``metric_extractor``; the handler mirrors ``stage.end`` and ``admission.end`` into the sink,
and ``finish()`` removes it before the legs finish. ``worker_execution.narrated_stage`` and
``seals.narrated_admission`` (reader-tuple members) are never edited.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.kernel.artifacts import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.adapters.composite import CompositeSink, build_sink_for_run
from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
from entropic_substrate.ladder.capture.ports.run_descriptor import (
    ExtractedMetrics,
    TrackingRunDescriptor,
)
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor
from entropic_substrate.ladder.run.seals import SealedStep, SealMemo, narrated_admission
from entropic_substrate.ladder.run.worker_execution import narrated_stage
from test_sinks import FakeMlflowClient

WARMUP = 1000
TRAINING_ROW = {
    "schema": "h-training-step-v2",
    "status": "committed",
    "phase": "prediction",
    "step": 2,
    "objective": 1.5,
}
GLOBAL_STEP = 1002  # prediction step 2 after a 1000-step warmup: neither 0 nor the raw 2


class _SteppedClient(FakeMlflowClient):
    """``FakeMlflowClient`` keeps only the last value per key; the axis needs the step."""

    def __init__(self) -> None:
        super().__init__()
        self.points: list[tuple[str, float, int]] = []

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        super().log_metric(run_id, key, value, step=step)
        self.points.append((key, float(value), step))


def _descriptor() -> TrackingRunDescriptor:
    return TrackingRunDescriptor(
        run_name="stage-bridge",
        tiny=False,
        params={"exp": "stage-bridge"},
        tags={"exp": "stage-bridge"},
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _stage_rows(path: Path) -> list[dict[str, Any]]:
    return [r for r in _rows(path) if r.get("schema") == "h-worker-stage-v1"]


@pytest.fixture
def write_failed() -> Iterator[list[dict[str, Any]]]:
    """Every ``sink.write_failed`` record the logger emits while the test runs."""
    seen: list[dict[str, Any]] = []
    handler = logger.add(
        lambda message: seen.append(dict(message.record["extra"])),
        level="DEBUG",
        filter=lambda record: record["extra"].get("event") == "sink.write_failed",
    )
    try:
        yield seen
    finally:
        logger.remove(handler)


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[tuple[CompositeSink, _SteppedClient, Path]]:
    """A composite with the H extractor and a recording client; ALWAYS finished.

    A failing test must not leave the bridge handler on the process-wide logger.
    """
    client = _SteppedClient()
    results = tmp_path / "metrics.jsonl"
    sink = build_sink_for_run(
        _descriptor(),
        results_path=results,
        client=client,
        metric_extractor=HTrainingStepExtractor(warmup_steps=WARMUP),
    )
    assert type(sink) is CompositeSink
    try:
        yield sink, client, results
    finally:
        sink.finish()


def _points(client: _SteppedClient, prefix: str) -> dict[str, tuple[float, int]]:
    return {k: (v, s) for k, v, s in client.points if k.startswith(prefix)}


def _populated_memo() -> SealMemo:
    memo = SealMemo()
    identity = ArtifactIdentity("a.json", ContentHash("sha256", "a" * 64))
    memo.remember(identity, SealedStep("prediction", 1, identity, identity, 1, ("a" * 64,)))
    return memo


@pytest.mark.unit
def test_a_stage_end_lands_on_the_training_global_step_in_both_legs(
    rig: tuple[CompositeSink, _SteppedClient, Path],
) -> None:
    sink, client, results = rig
    sink.log(TRAINING_ROW)

    with narrated_stage("generation"):
        pass
    sink.finish()

    (row,) = _stage_rows(results)
    assert row["event"] == "stage.end" and row["stage"] == "generation"
    assert row["seconds"] >= 0.0
    stage = _points(client, "stage/")
    assert list(stage) == ["stage/generation/seconds"]
    assert stage["stage/generation/seconds"] == (row["seconds"], GLOBAL_STEP)
    # Control: the training row itself is on the same axis, so the stage is aligned with it.
    assert _points(client, "objective") == {"objective": (1.5, GLOBAL_STEP)}


@pytest.mark.unit
def test_an_admission_end_lands_its_measurements_and_tags_the_memo_provenance(
    rig: tuple[CompositeSink, _SteppedClient, Path],
) -> None:
    sink, client, results = rig
    sink.log(TRAINING_ROW)

    fresh = SealMemo()
    with narrated_admission("completion", "run", fresh):
        fresh.seals_walked += 4
        fresh.evaluations_read += 3
        fresh.evaluations_memo_hits += 5  # log-only: it must NOT reach the sink projection
        fresh.journals_read += 4
        fresh.admission_seconds += 0.25
    with narrated_admission("generation", "run", _populated_memo()):
        pass
    sink.finish()

    assert _points(client, "admission/completion/") == {
        "admission/completion/seconds": (0.25, GLOBAL_STEP),
        "admission/completion/seals_walked": (4.0, GLOBAL_STEP),
        "admission/completion/evaluations": (3.0, GLOBAL_STEP),
        "admission/completion/journals_read": (4.0, GLOBAL_STEP),
    }
    # The populated arm still reports its measured zeros: a shared walk that read nothing
    # is a measurement, not an absence.
    assert _points(client, "admission/generation/") == {
        "admission/generation/seconds": (0.0, GLOBAL_STEP),
        "admission/generation/seals_walked": (0.0, GLOBAL_STEP),
        "admission/generation/evaluations": (0.0, GLOBAL_STEP),
        "admission/generation/journals_read": (0.0, GLOBAL_STEP),
    }
    (run,) = client._runs.values()
    assert run.data.tags["admission/completion/memo"] == "fresh"
    assert run.data.tags["admission/generation/memo"] == "shared"
    # The authoritative leg carries the same two admission rows and the two annotate rows.
    assert [(r["event"], r["caller"], r["memo"]) for r in _stage_rows(results)] == [
        ("admission.end", "completion", "fresh"),
        ("admission.end", "generation", "shared"),
    ]


@pytest.mark.unit
def test_a_stage_end_before_any_training_row_lands_no_metric(
    rig: tuple[CompositeSink, _SteppedClient, Path],
) -> None:
    sink, client, _results = rig
    with narrated_stage("runtime_admission"):
        pass
    sink.finish()
    assert _points(client, "stage/") == {}
    assert client.points == []


@pytest.mark.unit
def test_the_stage_axis_starts_at_the_first_training_row_not_before(
    rig: tuple[CompositeSink, _SteppedClient, Path],
) -> None:
    """Control for the test above: the same sink mirrors the next stage once a row placed the axis."""
    sink, client, _results = rig
    with narrated_stage("runtime_admission"):
        pass
    sink.log(TRAINING_ROW)
    with narrated_stage("training_readback"):
        pass
    sink.finish()
    assert list(_points(client, "stage/")) == ["stage/training_readback/seconds"]


@pytest.mark.unit
def test_a_stage_end_after_finish_lands_nothing_and_counts_nothing(
    rig: tuple[CompositeSink, _SteppedClient, Path],
    write_failed: list[dict[str, Any]],
) -> None:
    sink, client, results = rig
    sink.log(TRAINING_ROW)
    with narrated_stage("training_readback"):
        pass  # control: before finish() the same stage lands
    sink.finish()
    assert [r["stage"] for r in _stage_rows(results)] == ["training_readback"]
    points_at_finish = list(client.points)
    rows_at_finish = results.read_text()

    with narrated_stage("generation"):
        pass
    with narrated_admission("generation", "run", SealMemo()):
        pass

    assert client.points == points_at_finish
    assert results.read_text() == rows_at_finish
    assert write_failed == []
    assert sink.best_effort[0].failed_writes == 0  # type: ignore[attr-defined]


@pytest.mark.unit
def test_a_sink_built_without_an_extractor_or_without_tracking_installs_no_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("ENTROPIC_MLFLOW_TINY", raising=False)
    extractor = HTrainingStepExtractor(warmup_steps=WARMUP)
    client = _SteppedClient()
    controlled = build_sink_for_run(
        _descriptor(),
        results_path=tmp_path / "control.jsonl",
        client=_SteppedClient(),
        metric_extractor=extractor,
    )
    cases = {
        "composite, no extractor": build_sink_for_run(
            _descriptor(), results_path=tmp_path / "a.jsonl", client=client
        ),
        "bare jsonl, no extractor": build_sink_for_run(
            _descriptor(), results_path=tmp_path / "b.jsonl"
        ),
        "bare jsonl, extractor given": build_sink_for_run(
            _descriptor(), results_path=tmp_path / "c.jsonl", metric_extractor=extractor
        ),
    }
    assert type(cases["bare jsonl, no extractor"]) is JsonlSink
    assert type(cases["bare jsonl, extractor given"]) is JsonlSink
    try:
        for sink in (controlled, *cases.values()):
            sink.log(TRAINING_ROW)
        with narrated_stage("generation"):
            pass
        for name in ("a", "b", "c"):
            assert _stage_rows(tmp_path / f"{name}.jsonl") == [], name
        # Control: the extractor-bearing composite built in the same process mirrored it.
        assert len(_stage_rows(tmp_path / "control.jsonl")) == 1
        assert _points(client, "stage/") == {}
    finally:
        controlled.finish()
        for sink in cases.values():
            sink.finish()


@pytest.mark.unit
def test_an_unfinished_sink_that_is_collected_leaves_no_handler_behind(tmp_path: Path) -> None:
    """The bridge holds the composite weakly: a test or tool that builds one and never
    finishes it must not leave a process-wide handler writing to its file forever."""
    results = tmp_path / "metrics.jsonl"
    sink = build_sink_for_run(
        _descriptor(),
        results_path=results,
        client=_SteppedClient(),
        metric_extractor=HTrainingStepExtractor(warmup_steps=WARMUP),
    )
    assert type(sink) is CompositeSink
    # Only the composite is abandoned; the MLflow leg is finished so its writer thread ends.
    mlflow_leg = sink.best_effort[0]
    try:
        sink.log(TRAINING_ROW)
        with narrated_stage("generation"):
            pass
        assert len(_stage_rows(results)) == 1  # control: the live bridge does write

        del sink
        gc.collect()
        with narrated_stage("generation"):
            pass

        assert len(_stage_rows(results)) == 1
    finally:
        mlflow_leg.finish()


class _FailsOnStage:
    """The H extractor, except that it raises on a worker-stage record."""

    def __init__(self) -> None:
        self._inner = HTrainingStepExtractor(warmup_steps=WARMUP)

    def extract(self, event: dict[str, object]) -> ExtractedMetrics | None:
        if event.get("schema") == "h-worker-stage-v1":
            raise RuntimeError("simulated extractor failure")
        return self._inner.extract(event)


@pytest.mark.unit
def test_a_failing_mirror_is_counted_and_never_reaches_the_logging_call(
    tmp_path: Path, write_failed: list[dict[str, Any]]
) -> None:
    """The MLflow leg's extractor failure is counted by its own ``write_failed`` path, and the
    warning it logs from INSIDE the bridge handler must neither re-enter the handler nor raise."""
    sink = build_sink_for_run(
        _descriptor(),
        results_path=tmp_path / "metrics.jsonl",
        client=_SteppedClient(),
        metric_extractor=_FailsOnStage(),
    )
    try:
        sink.log(TRAINING_ROW)
        with narrated_stage("generation"):
            pass  # must complete: the worker never dies because the mirror did
        assert sink.best_effort[0].failed_writes == 1  # type: ignore[attr-defined]
        assert [row["sink"] for row in write_failed] == ["mlflow"]
    finally:
        sink.finish()


@pytest.mark.unit
def test_a_failing_authoritative_write_is_counted_by_the_bridge_and_never_raises(
    rig: tuple[CompositeSink, _SteppedClient, Path],
    write_failed: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink, client, _results = rig
    sink.log(TRAINING_ROW)

    def refuse(self: JsonlSink, record: object) -> None:
        raise OSError("simulated full disk")

    monkeypatch.setattr(JsonlSink, "log", refuse)
    with narrated_stage("generation"):
        pass  # JsonlSink has no counter of its own: the bridge reports its failed mirror

    assert [(row["sink"], row["count"]) for row in write_failed] == [("stage_bridge", 1)]
    assert _points(client, "stage/") == {}


@pytest.mark.unit
def test_the_bridge_row_keys_are_exactly_the_projection_vocabulary(
    rig: tuple[CompositeSink, _SteppedClient, Path],
) -> None:
    """The capture layer may not import ``ladder.run`` (C10), so the bridge copies a literal
    field list. Pin that list against the projection's own output, both ways."""
    from entropic_substrate.ladder.run.worker_log_events import project_worker_stage_record

    sink, _client, results = rig
    sink.log(TRAINING_ROW)
    with narrated_stage("generation"):
        pass
    with narrated_admission("generation", "run", SealMemo()):
        pass
    sink.finish()
    for row in _stage_rows(results):
        assert project_worker_stage_record(row) == row


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("MLFLOW_TRACKING_URI", "").strip(),
    reason="$MLFLOW_TRACKING_URI not set -- run via `just container-test`",
)
def test_live_worker_command_mirrors_its_final_leg_stages_to_the_tracking_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LIVE arm, on the production call path (``run_worker_command --local-tiny``).

    MEASURED 2026-10-19 (the same command with tracking off): the tiny path ends
    ``materialization`` and ``runtime_admission`` BEFORE any training row exists, so those
    two carry no axis and are correctly not mirrored; the first stage that ends after
    training is ``training_readback``, which is therefore the key this arm asserts.
    """
    import mlflow.tracking
    from tests.test_d1_trainer_object_transport import _ObjectStore
    from tests.test_sentence_worker_execution import prepared

    from entropic_substrate.ladder.run.worker_command import run_worker_command
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    monkeypatch.setenv("ENTROPIC_MLFLOW_TINY", "1")
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "bridge-worker"
    workspace.mkdir()
    err = StringIO()
    status = run_worker_command(
        [
            "run",
            "--repo-root",
            str(tmp_path),
            "--max-object-bytes",
            "10000000",
            "--max-objects",
            "100",
            "--workspace-root",
            str(workspace),
            "--request-key",
            marker.key,
            "--request-sha256",
            marker.sha256,
            "--request-size",
            str(marker.size_bytes),
            "--local-tiny",
        ],
        environ={},
        objects=objects,
        stdout=StringIO(),
        stderr=err,
    )
    assert status == 0, err.getvalue()

    observability = workspace.parent / f"{workspace.name}.observability"
    log = _rows(observability / "worker-log.jsonl")
    (mlflow_row,) = [row for row in log if row.get("event") == "mlflow.run"]
    run_id: str = mlflow_row["run_id"]
    client = mlflow.tracking.MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    run = client.get_run(run_id)
    mirrored = {key: value for key, value in run.data.metrics.items() if key.startswith("stage/")}
    history = client.get_metric_history(run_id, "stage/training_readback/seconds")
    evidence = (
        f"run id: {run_id}\nstage metrics: {sorted(mirrored)}\n"
        f"training_readback history: {[(m.step, m.value) for m in history]}\n"
        f"admission tags: {sorted(k for k in run.data.tags if k.startswith('admission/'))}\n"
    )
    print(evidence)
    evidence_file = Path(__file__).resolve().parents[1] / "results" / "h-stage-bridge-live.txt"
    evidence_file.parent.mkdir(parents=True, exist_ok=True)
    evidence_file.write_text(evidence)

    assert "stage/training_readback/seconds" in mirrored, sorted(run.data.metrics)
    assert "stage/runtime_admission/seconds" not in mirrored  # ended before any training row
    (point,) = history
    training_steps = {m.step for m in client.get_metric_history(run_id, "phase_index")}
    assert point.step == max(training_steps)  # the axis is the last training row's global step
    assert "admission/completion/memo" in run.data.tags
