"""The H aggregate (FOR-1648): captured H runs paired by global step.

Two evidence classes, kept apart on purpose. The ``unit`` tests build tiny synthetic runs
with :func:`write_h_run` (low-entropy synthetic identities) and need no fixture. The
``integration`` tests read the two real captured HEpP7g runs under
``results/h0-plumbing-1k-v3-HEpP7g/`` (``dvc pull results/h0-plumbing-1k-v3-HEpP7g.dvc``)
and RAISE when the directory is absent: a skip here would be the false-green this repo
already paid for. CI runs ``-m unit`` only and has no DVC.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.ladder.analyze import h_aggregate
from entropic_substrate.ladder.analyze.h_aggregate import (
    CodecHandoffEvidence,
    HRun,
    aggregate_h_runs,
    load_h_run,
    parse_arm_labels,
)
from entropic_substrate.ladder.analyze.h_report import build_h_report
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor

IDENTITY_A = "0" * 63 + "1"
IDENTITY_B = "0" * 63 + "2"
REQUEST_A = "1" * 63 + "a"
REQUEST_B = "1" * 63 + "b"
RUN_ID_A = "a" * 32
RUN_ID_B = "b" * 32

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "results" / "h0-plumbing-1k-v3-HEpP7g"


# ------------------------------------------------------------------ synthetic builder
def _step_row(
    identity: str, phase: str, step: int, scale: float, *, committed: bool = True
) -> dict[str, Any]:
    glob = step if phase == "codec_warmup" else step + 2  # warm-up of 2 in the default builder
    return {
        "aggregate_norm": 1.0 + 0.01 * glob,
        "decoder_ce": scale * (6.0 - 0.1 * glob),
        "future_nll_per_dimension": None if phase == "codec_warmup" else scale * (1.0 - 0.4 * step),
        "objective": scale * (7.0 - 0.1 * glob),
        "phase": phase,
        "run_identity": identity,
        "schema": "h-training-step-v2",
        "status": "committed" if committed else "failed",
        "step": step,
        "wall_seconds": 0.5,
    }


def _eval_row(identity: str, step: int, scale: float) -> dict[str, Any]:
    def entry(estimate: float, ess: float) -> dict[str, Any]:
        return {
            "effective_sample_size": ess,
            "gap": None,
            "likelihood_relative_mcse": 0.5,
            "log_estimate": estimate,
        }

    return {
        "checkpoint_seal": {"path": f"seal-{step}.json", "sha256": "2" * 64},
        "cuda_seconds": 0.25,
        "examples": [
            {
                "boundary_symbols": [1, 1],
                "source_bytes": [100, 200],
                "by_sample_count": {
                    "2": [entry(-100.0 * scale - step, 1.0), entry(-110.0 * scale - step, 1.5)],
                    "4": [entry(-99.0 * scale - step, 2.0), entry(-98.0 * scale - step, 2.5)],
                },
            }
        ],
        "meaning": "importance_weighted_log_marginal_estimate",
        "run_identity": identity,
        "schema": "h-predictive-evaluation-v1",
        "step": step,
        "wall_seconds": 0.75,
    }


def _annotate(global_step: int, seconds: str) -> dict[str, Any]:
    prefix = f"checkpoint/{global_step:08d}"
    return {
        "event": "annotate",
        f"{prefix}/key": f"snapshots/checkpoint-{global_step:08d}.json",
        f"{prefix}/publication_seconds": seconds,
        f"{prefix}/sha256": "3" * 64,
        f"{prefix}/size_bytes": "100",
    }


def _log(event: str, **fields: Any) -> dict[str, Any]:
    return {"event": event, "level": "INFO", "message": event, **fields}


def build_h_run_rows(
    identity: str,
    request: str,
    *,
    run_id: str | None = RUN_ID_A,
    warmup: int = 2,
    steps: int = 4,
    interval: int = 2,
    scale: float = 1.0,
    status: str = "ok",
    result_status: str = "probe_workload_complete",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(metrics rows, worker-log rows)`` of a tiny admissible run (warm-up 2, 4 steps)."""
    assert warmup == 2, "the step-row scalar formulas assume a warm-up of 2"
    metrics: list[dict[str, Any]] = [
        {
            "code_commit": "c" * 40,
            "event": "run_start",
            "image": "example.invalid/image@sha256:" + "4" * 64,
            "profile_identity": identity,
            "request_sha256": request,
            "run_class": "remote_tiny",
            "runtime": {"accelerator": {"count": 1, "model": "synthetic"}, "pod_id": "pod"},
            "schema": "h-run-lifecycle-v1",
        }
    ]
    log: list[dict[str, Any]] = [_log("worker.start", request_sha256=request)]
    if run_id is not None:
        log.append(_log("mlflow.run", run_id=run_id))
    metrics.extend(_step_row(identity, "codec_warmup", s, scale) for s in range(1, warmup + 1))
    metrics.append(_annotate(warmup, "1.5"))
    log.append(_log("checkpoint.seal", phase="codec_warmup", step=warmup, global_step=warmup))
    for p in range(1, steps + 1):
        metrics.append(_step_row(identity, "prediction", p, scale))
        if p % interval == 0:
            metrics.append(_eval_row(identity, p, scale))
            metrics.append(_annotate(warmup + p, "2.5"))
            log.append(_log("checkpoint.seal", phase="prediction", step=p, global_step=warmup + p))
            log.append(_log("evaluation.start", global_step=warmup + p))
    log.append(_log("stage.end", stage="training_readback", seconds=2.0))
    log.append(
        _log(
            "admission.end",
            caller="completion",
            seconds=3.0,
            seals_walked=2,
            evaluations=2,
            journals_read=1,
        )
    )
    metrics.append(
        {
            "event": "run_end",
            "result_status": result_status,
            "schema": "h-run-lifecycle-v1",
            "status": status,
            "successful_optimizer_steps": warmup + steps,
            "wall_seconds": 12.5,
        }
    )
    return metrics, log


def _dump(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def write_h_run(
    directory: Path,
    identity: str = IDENTITY_A,
    request: str = REQUEST_A,
    *,
    tag: str = "a",
    mutate_metrics: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_log: Callable[[list[dict[str, Any]]], None] | None = None,
    **kwargs: Any,
) -> tuple[Path, Path]:
    metrics, log = build_h_run_rows(identity, request, **kwargs)
    if mutate_metrics is not None:
        mutate_metrics(metrics)
    if mutate_log is not None:
        mutate_log(log)
    directory.mkdir(parents=True, exist_ok=True)
    return (
        _dump(directory / f"{tag}-metrics.jsonl", metrics),
        _dump(directory / f"{tag}-worker-log.jsonl", log),
    )


def _two_arms(tmp_path: Path) -> tuple[list[HRun], dict[str, str]]:
    a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a", run_id=RUN_ID_A, scale=1.0)
    b = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b", run_id=RUN_ID_B, scale=2.0)
    runs = [load_h_run(*a), load_h_run(*b)]
    return runs, {IDENTITY_A: "magnitude", IDENTITY_B: "sphere"}


# ------------------------------------------------------------------ unit: the happy path
@pytest.mark.unit
class TestPairedAggregate:
    def test_one_summary_per_labelled_arm_and_values_stay_with_their_label(
        self, tmp_path: Path
    ) -> None:
        runs, labels = _two_arms(tmp_path)
        doc = aggregate_h_runs(runs, labels)
        assert sorted(doc["arms"]) == ["magnitude", "sphere"]
        assert doc["arms"]["magnitude"]["profile_identity"] == IDENTITY_A
        assert doc["arms"]["sphere"]["profile_identity"] == IDENTITY_B
        assert doc["arms"]["magnitude"]["mlflow_run_id"] == RUN_ID_A
        # the scale differs per arm, so the pairing is visible in the values themselves
        first = doc["training"]["metrics"]["objective"][0]
        assert first["step"] == 1
        assert first["values"] == {"magnitude": 6.9, "sphere": 13.8}
        step3 = doc["training"]["metrics"]["objective"][2]
        assert step3["step"] == 3
        assert step3["values"]["sphere"] == pytest.approx(2 * step3["values"]["magnitude"])

    def test_labels_decide_the_grouping_not_input_order(self, tmp_path: Path) -> None:
        runs, labels = _two_arms(tmp_path)
        swapped = {IDENTITY_A: "sphere", IDENTITY_B: "magnitude"}
        doc = aggregate_h_runs(runs, swapped)
        assert doc["arms"]["sphere"]["profile_identity"] == IDENTITY_A
        assert doc["training"]["metrics"]["objective"][0]["values"] == {
            "magnitude": 13.8,
            "sphere": 6.9,
        }
        assert labels != swapped

    def test_arm_labels_bind_by_identity_not_by_insertion_or_input_order(
        self, tmp_path: Path
    ) -> None:
        runs, _ = _two_arms(tmp_path)  # input order is [A, B]
        reversed_labels = {IDENTITY_B: "sphere", IDENTITY_A: "magnitude"}  # inserted [B, A]
        doc = aggregate_h_runs(runs, reversed_labels)
        assert doc["arms"]["magnitude"]["profile_identity"] == IDENTITY_A
        assert doc["arms"]["sphere"]["profile_identity"] == IDENTITY_B
        doc = aggregate_h_runs(
            list(reversed(runs)), {IDENTITY_A: "magnitude", IDENTITY_B: "sphere"}
        )
        assert doc["arms"]["magnitude"]["profile_identity"] == IDENTITY_A

    def test_global_step_axis_is_rebuilt_from_the_worker_log(self, tmp_path: Path) -> None:
        runs, labels = _two_arms(tmp_path)
        doc = aggregate_h_runs(runs, labels)
        assert doc["training"]["paired_steps"] == [1, 2, 3, 4, 5, 6]
        assert doc["phase_index"][0]["values"] == {
            "magnitude": "codec_warmup",
            "sphere": "codec_warmup",
        }
        assert doc["phase_index"][2]["values"]["magnitude"] == "prediction"
        assert doc["arms"]["magnitude"]["first_prediction_step"] == 3
        assert doc["arms"]["magnitude"]["warmup_steps"] == 2
        assert doc["arms"]["magnitude"]["step_range"] == [1, 6]

    def test_evaluation_rows_sit_on_global_steps_not_phase_local_ones(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A)
        raw_steps = [
            json.loads(line)["step"]
            for line in a[0].read_text().splitlines()
            if json.loads(line).get("schema") == "h-predictive-evaluation-v1"
        ]
        assert raw_steps == [2, 4]  # what a reader keying on the bare step would report
        run = load_h_run(*a)
        assert [step for step, _ in run.evaluations] == [4, 6]
        assert sorted(run.publication_seconds) == [2, 4, 6]  # annotate keys are already global

    def test_evaluation_scalars_equal_the_extractors_projection(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A)
        run = load_h_run(*a)
        eval_rows = [
            json.loads(line)
            for line in a[0].read_text().splitlines()
            if json.loads(line).get("schema") == "h-predictive-evaluation-v1"
        ]
        extractor = HTrainingStepExtractor(warmup_steps=2)
        for row, (step, metrics) in zip(eval_rows, run.evaluations, strict=True):
            projected = extractor.extract(row)
            assert projected is not None
            assert projected.step == step
            assert projected.metrics == metrics

    def test_training_scalar_names_equal_the_extractors_projection_keys(
        self, tmp_path: Path
    ) -> None:
        run = load_h_run(*write_h_run(tmp_path))
        _, _, prediction_scalars = run.training[-1]
        assert set(prediction_scalars) == set(h_aggregate._STEP_SCALARS)  # pyright: ignore[reportPrivateUsage]
        _, _, warmup_scalars = run.training[0]
        assert set(warmup_scalars) == set(h_aggregate._STEP_SCALARS) - {"future_nll_per_dimension"}  # pyright: ignore[reportPrivateUsage]

    def test_negative_future_nll_is_reported_as_captured_and_warmup_null_stays_null(
        self, tmp_path: Path
    ) -> None:
        runs, labels = _two_arms(tmp_path)
        doc = aggregate_h_runs(runs, labels)
        rows = doc["training"]["metrics"]["future_nll_per_dimension"]
        assert rows[0]["values"] == {"magnitude": None, "sphere": None}  # codec warm-up
        last = rows[-1]["values"]
        assert last["magnitude"] == pytest.approx(1.0 - 0.4 * 4)  # -0.6, never clamped
        assert last["sphere"] == pytest.approx(2.0 * (1.0 - 0.4 * 4))
        summary = doc["arms"]["magnitude"]["future_nll_per_dimension"]
        assert summary["null_rows"] == 2
        assert summary["negative_rows"] == 2
        assert summary["min"] == pytest.approx(-0.6)

    def test_publication_timing_is_taken_from_the_annotate_rows(self, tmp_path: Path) -> None:
        runs, labels = _two_arms(tmp_path)
        doc = aggregate_h_runs(runs, labels)
        assert doc["publication_seconds"]["paired_steps"] == [2, 4, 6]
        assert doc["publication_seconds"]["values"][0] == {
            "step": 2,
            "values": {"magnitude": 1.5, "sphere": 1.5},
        }

    def test_document_is_strict_json_and_deterministic(self, tmp_path: Path) -> None:
        runs, labels = _two_arms(tmp_path)
        first = json.dumps(aggregate_h_runs(runs, labels), sort_keys=True, allow_nan=False)
        second = json.dumps(aggregate_h_runs(runs, labels), sort_keys=True, allow_nan=False)
        assert first == second
        assert math.isfinite(json.loads(first)["arms"]["magnitude"]["run_end"]["wall_seconds"])

    def test_steps_held_by_one_arm_only_are_reported_never_truncated_silently(
        self, tmp_path: Path
    ) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a", steps=4)
        b = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b", steps=2, interval=2)
        doc = aggregate_h_runs(
            [load_h_run(*a), load_h_run(*b)], {IDENTITY_A: "long", IDENTITY_B: "short"}
        )
        assert doc["training"]["paired_steps"] == [1, 2, 3, 4]
        assert doc["training"]["unpaired"] == {"long": [5, 6], "short": []}
        assert doc["evaluation"]["paired_steps"] == [4]
        assert doc["evaluation"]["unpaired"] == {"long": [6], "short": []}

    def test_a_failed_run_is_not_presented_as_complete(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, status="failed")
        doc = aggregate_h_runs([load_h_run(*a)], {IDENTITY_A: "only"})
        assert doc["arms"]["only"]["completed"] is False
        assert doc["arms"]["only"]["run_end"]["status"] == "failed"

    @pytest.mark.parametrize(
        ("result_status", "completed"),
        [
            ("probe_workload_complete", True),
            ("paused", False),
            ("stopped_by_operator", False),
        ],
    )
    def test_only_a_complete_terminal_result_is_presented_as_complete(
        self, tmp_path: Path, result_status: str, completed: bool
    ) -> None:
        # the worker writes status "ok" for every non-failure end, so status alone lies here
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, result_status=result_status)
        doc = aggregate_h_runs([load_h_run(*a)], {IDENTITY_A: "only"})
        assert doc["arms"]["only"]["run_end"]["status"] == "ok"
        assert doc["arms"]["only"]["completed"] is completed

    def test_uncommitted_rows_are_counted_not_dropped_silently(self, tmp_path: Path) -> None:
        def add_uncommitted(rows: list[dict[str, Any]]) -> None:
            rows.insert(-1, _step_row(IDENTITY_A, "prediction", 5, 1.0, committed=False))

        paths = write_h_run(tmp_path, mutate_metrics=add_uncommitted)
        run = load_h_run(*paths)
        assert run.uncommitted_step_rows == 1
        assert len(run.training) == 6
        doc = aggregate_h_runs([run], {IDENTITY_A: "only"})
        assert doc["arms"]["only"]["uncommitted_step_rows"] == 1
        assert build_h_report(*paths)["steps"]["uncommitted_rows"] == 1

    def test_a_run_without_a_mlflow_run_event_carries_no_run_id(self, tmp_path: Path) -> None:
        run = load_h_run(*write_h_run(tmp_path, run_id=None))
        assert run.mlflow_run_id is None


# ------------------------------------------------------------------ unit: arm labels
@pytest.mark.unit
class TestArmLabels:
    def test_a_valid_pair_parses_with_the_full_identity(self) -> None:
        assert parse_arm_labels([f"{IDENTITY_A}=magnitude"]) == {IDENTITY_A: "magnitude"}

    @pytest.mark.parametrize(
        "pair",
        [
            "deadbeef=magnitude",  # a short id is never a prefix match
            f"{IDENTITY_A}",  # no label
            f"{IDENTITY_A}=",  # empty label
            "0" * 63 + "A=x",  # identities are lowercase
            f"{IDENTITY_A}=../escape",  # no path separators in a label
            f"{IDENTITY_A}=a b",
            f"{IDENTITY_A}=x\n",  # fullmatch, not match: a trailing newline is refused
            f"{IDENTITY_A}\n=x",
        ],
    )
    def test_malformed_pairs_are_refused(self, pair: str) -> None:
        with pytest.raises(ValueError, match="--arm"):
            parse_arm_labels([pair])

    def test_a_duplicate_identity_and_a_shared_label_are_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            parse_arm_labels([f"{IDENTITY_A}=x", f"{IDENTITY_A}=y"])
        with pytest.raises(ValueError, match="already the label"):
            parse_arm_labels([f"{IDENTITY_A}=x", f"{IDENTITY_B}=x"])

    def test_an_unknown_identity_fails_loud_and_names_it(self, tmp_path: Path) -> None:
        runs, labels = _two_arms(tmp_path)
        unknown = "f" * 64
        with pytest.raises(ValueError, match=unknown):
            aggregate_h_runs(runs, {**labels, unknown: "ghost"})

    def test_an_unlabelled_run_fails_loud_and_names_it(self, tmp_path: Path) -> None:
        runs, _ = _two_arms(tmp_path)
        with pytest.raises(ValueError, match=IDENTITY_B):
            aggregate_h_runs(runs, {IDENTITY_A: "magnitude"})

    def test_the_same_run_given_twice_is_refused(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path)
        with pytest.raises(ValueError, match="more than once"):
            aggregate_h_runs([load_h_run(*a), load_h_run(*a)], {IDENTITY_A: "x"})

    def test_no_runs_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            aggregate_h_runs([], {})


# ------------------------------------------------------------------ unit: fail loud
def _drop(predicate: Callable[[dict[str, Any]], bool]) -> Callable[[list[dict[str, Any]]], None]:
    def mutate(rows: list[dict[str, Any]]) -> None:
        rows[:] = [r for r in rows if not predicate(r)]

    return mutate


def _set(
    predicate: Callable[[dict[str, Any]], bool], **changes: Any
) -> Callable[[list[dict[str, Any]]], None]:
    def mutate(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if predicate(row):
                row.update(changes)

    return mutate


def _is_event(name: str) -> Callable[[dict[str, Any]], bool]:
    return lambda r: r.get("event") == name


def _prediction_step(n: int) -> Callable[[dict[str, Any]], bool]:
    return lambda r: (
        r.get("schema") == "h-training-step-v2"
        and r.get("phase") == "prediction"
        and r.get("step") == n
    )


def _without(
    predicate: Callable[[dict[str, Any]], bool], key: str
) -> Callable[[list[dict[str, Any]]], None]:
    def mutate(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if predicate(row):
                row.pop(key, None)

    return mutate


def _duplicate_first(
    predicate: Callable[[dict[str, Any]], bool],
) -> Callable[[list[dict[str, Any]]], None]:
    def mutate(rows: list[dict[str, Any]]) -> None:
        index = next(i for i, r in enumerate(rows) if predicate(r))
        rows.insert(index + 1, dict(rows[index]))

    return mutate


def _first_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(r for r in rows if r.get("schema") == "h-predictive-evaluation-v1")


def _empty_sample_count_entries(rows: list[dict[str, Any]]) -> None:
    _first_evaluation(rows)["examples"][0]["by_sample_count"]["2"] = []


def _non_object_entry(rows: list[dict[str, Any]]) -> None:
    _first_evaluation(rows)["examples"][0]["by_sample_count"]["2"] = ["not an object"]


def _non_object_example(rows: list[dict[str, Any]]) -> None:
    _first_evaluation(rows)["examples"] = ["not an object"]


@pytest.mark.unit
class TestInadmissibleTelemetryRaises:
    """Each case is one distinct way an H file can be inadmissible; none may return a value."""

    @pytest.mark.parametrize(
        ("name", "mutate"),
        [
            ("no run_end", _drop(_is_event("run_end"))),
            ("no run_start", _drop(_is_event("run_start"))),
            ("no committed step", _drop(lambda r: r.get("schema") == "h-training-step-v2")),
            (
                "nan objective",
                _set(_prediction_step(1), objective=float("nan")),
            ),
            ("null objective", _set(_prediction_step(1), objective=None)),
            (
                "null future_nll on a prediction row",
                _set(_prediction_step(2), future_nll_per_dimension=None),
            ),
            ("infinite decoder_ce", _set(_prediction_step(1), decoder_ce=float("inf"))),
            ("non-monotonic step", _set(_prediction_step(2), step=1)),
            ("foreign row identity", _set(_prediction_step(2), run_identity="9" * 64)),
            ("row without identity", _set(_prediction_step(2), run_identity=None)),
            ("unknown phase", _set(_prediction_step(2), phase="mystery")),
            ("string objective", _set(_prediction_step(1), objective="7.0")),
            ("bool objective", _set(_prediction_step(1), objective=True)),
            ("integer beyond float range", _set(_prediction_step(1), objective=10**400)),
            (
                "malformed run_start profile_identity",
                _set(_is_event("run_start"), profile_identity="not-hex"),
            ),
            (
                "step row without a run_identity key",
                _without(_prediction_step(2), "run_identity"),
            ),
            (
                "evaluation row without a run_identity key",
                _without(lambda r: r.get("schema") == "h-predictive-evaluation-v1", "run_identity"),
            ),
            ("empty sample-count entries", _empty_sample_count_entries),
            ("non-object sample-count entry", _non_object_entry),
            ("non-object example", _non_object_example),
        ],
    )
    def test_metrics_defect(
        self, tmp_path: Path, name: str, mutate: Callable[[list[dict[str, Any]]], None]
    ) -> None:
        metrics, log = write_h_run(tmp_path, mutate_metrics=mutate)
        with pytest.raises(TelemetryGap) as excinfo:
            load_h_run(metrics, log)
        assert str(metrics) in str(excinfo.value), name

    def test_missing_run_end_never_returns_an_empty_aggregate(self, tmp_path: Path) -> None:
        metrics, log = write_h_run(tmp_path, mutate_metrics=_drop(_is_event("run_end")))
        with pytest.raises(TelemetryGap):
            aggregate_h_runs([load_h_run(metrics, log)], {IDENTITY_A: "x"})

    def test_two_run_starts_are_refused(self, tmp_path: Path) -> None:
        def duplicate(rows: list[dict[str, Any]]) -> None:
            rows.insert(1, dict(rows[0]))

        with pytest.raises(TelemetryGap, match="exactly one run_start"):
            load_h_run(*write_h_run(tmp_path, mutate_metrics=duplicate))

    def test_a_missing_or_corrupt_file_is_a_telemetry_gap(self, tmp_path: Path) -> None:
        metrics, log = write_h_run(tmp_path)
        with pytest.raises(TelemetryGap):
            load_h_run(tmp_path / "absent-metrics.jsonl", log)
        with pytest.raises(TelemetryGap):
            load_h_run(metrics, tmp_path / "absent-worker-log.jsonl")
        log.write_text(log.read_text() + "{not json\n", encoding="utf-8")
        with pytest.raises(TelemetryGap, match="not valid JSON"):
            load_h_run(metrics, log)

    def test_a_duplicate_key_in_a_row_is_refused(self, tmp_path: Path) -> None:
        metrics, log = write_h_run(tmp_path)
        lines = metrics.read_text().splitlines()
        lines[2] = lines[2][:-1] + ', "step": 99}'
        metrics.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(TelemetryGap, match="duplicate key"):
            load_h_run(metrics, log)

    def test_a_worker_log_of_another_run_is_refused(self, tmp_path: Path) -> None:
        metrics, _ = write_h_run(tmp_path, tag="a")
        _, other_log = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b")
        with pytest.raises(TelemetryGap, match="different run"):
            load_h_run(metrics, other_log)

    def test_a_swapped_worker_log_is_refused_even_when_the_schedules_match(
        self, tmp_path: Path
    ) -> None:
        a_metrics, a_log = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        b_metrics, b_log = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b")
        with pytest.raises(TelemetryGap):
            load_h_run(a_metrics, b_log)
        with pytest.raises(TelemetryGap):
            load_h_run(b_metrics, a_log)

    @pytest.mark.parametrize(
        ("name", "mutate", "match"),
        [
            ("no worker.start", _drop(_is_event("worker.start")), "worker.start"),
            (
                "no prediction seal",
                _drop(
                    lambda r: r.get("event") == "checkpoint.seal" and r.get("phase") == "prediction"
                ),
                "warm-up length",
            ),
            (
                "disagreeing prediction seals",
                _set(
                    lambda r: r.get("event") == "checkpoint.seal" and r.get("step") == 4,
                    global_step=99,
                ),
                "warm-up length",
            ),
            (
                "evaluation.start missing",
                lambda rows: rows.remove(
                    next(r for r in rows if r.get("event") == "evaluation.start")
                ),
                "evaluation.start",
            ),
            (
                "evaluation.start at the wrong step",
                _set(
                    lambda r: r.get("event") == "evaluation.start" and r["global_step"] == 4,
                    global_step=5,
                ),
                "evaluation.start",
            ),
            (
                "bad mlflow run id",
                _set(_is_event("mlflow.run"), run_id="a163161f"),
                "32 lowercase hex",
            ),
            (
                "two worker.start events",
                _duplicate_first(_is_event("worker.start")),
                "exactly one worker.start",
            ),
            (
                "two mlflow.run events",
                _duplicate_first(_is_event("mlflow.run")),
                "at most one mlflow.run",
            ),
            (
                "stage.end without seconds",
                _without(_is_event("stage.end"), "seconds"),
                r"stage\.end",
            ),
            (
                "stage.end with a string seconds",
                _set(_is_event("stage.end"), seconds="2.0"),
                r"stage\.end",
            ),
            (
                "stage.end without a string stage",
                _set(_is_event("stage.end"), stage=7),
                r"stage\.end",
            ),
            (
                "stage.end with an integer seconds beyond float range",
                _set(_is_event("stage.end"), seconds=10**400),
                r"stage\.end",
            ),
            (
                "admission.end without seconds",
                _without(_is_event("admission.end"), "seconds"),
                r"admission\.end",
            ),
            (
                "admission.end without a string caller",
                _set(_is_event("admission.end"), caller=None),
                r"admission\.end",
            ),
            (
                "admission.end with a string count",
                _set(_is_event("admission.end"), evaluations="2"),
                r"admission\.end",
            ),
            (
                "admission.end with a null count",
                _set(_is_event("admission.end"), journals_read=None),
                r"admission\.end",
            ),
            (
                "admission.end with a bool count",
                _set(_is_event("admission.end"), seals_walked=True),
                r"admission\.end",
            ),
        ],
    )
    def test_worker_log_defect(
        self,
        tmp_path: Path,
        name: str,
        mutate: Callable[[list[dict[str, Any]]], None],
        match: str,
    ) -> None:
        metrics, log = write_h_run(tmp_path, mutate_log=mutate)
        with pytest.raises(TelemetryGap, match=match) as excinfo:
            load_h_run(metrics, log)
        assert str(log) in str(excinfo.value), name

    def test_a_warmup_that_disagrees_with_the_worker_log_is_refused(self, tmp_path: Path) -> None:
        metrics, log = write_h_run(
            tmp_path,
            mutate_metrics=_drop(
                lambda r: (
                    r.get("schema") == "h-training-step-v2"
                    and r.get("phase") == "codec_warmup"
                    and r["step"] == 2
                )
            ),
        )
        with pytest.raises(TelemetryGap, match="warm-up"):
            load_h_run(metrics, log)

    def test_publication_timing_must_be_a_finite_number_string_at_a_sealed_checkpoint(
        self, tmp_path: Path
    ) -> None:
        def poison(rows: list[dict[str, Any]]) -> None:
            row = next(r for r in rows if "checkpoint/00000004/publication_seconds" in r)
            row["checkpoint/00000004/publication_seconds"] = "fast"

        with pytest.raises(TelemetryGap, match="publication_seconds"):
            load_h_run(*write_h_run(tmp_path, mutate_metrics=poison))

        def unsealed(rows: list[dict[str, Any]]) -> None:
            rows.append(_annotate(9, "1.0"))

        with pytest.raises(TelemetryGap, match=r"no matching checkpoint\.seal"):
            load_h_run(*write_h_run(tmp_path / "u", mutate_metrics=unsealed))

        def duplicate(rows: list[dict[str, Any]]) -> None:
            rows.append(_annotate(4, "1.0"))

        with pytest.raises(TelemetryGap, match="duplicate publication"):
            load_h_run(*write_h_run(tmp_path / "d", mutate_metrics=duplicate))

    def test_repeated_evaluation_global_steps_are_refused_not_collapsed(
        self, tmp_path: Path
    ) -> None:
        metrics, log = write_h_run(
            tmp_path,
            mutate_metrics=_duplicate_first(
                lambda r: r.get("schema") == "h-predictive-evaluation-v1"
            ),
            mutate_log=_duplicate_first(_is_event("evaluation.start")),
        )
        with pytest.raises(TelemetryGap, match="not strictly increasing") as excinfo:
            load_h_run(metrics, log)
        assert str(metrics) in str(excinfo.value)

    def test_a_non_finite_evaluation_number_is_refused_not_dropped(self, tmp_path: Path) -> None:
        def poison(rows: list[dict[str, Any]]) -> None:
            row = next(r for r in rows if r.get("schema") == "h-predictive-evaluation-v1")
            row["examples"][0]["by_sample_count"]["2"][0]["log_estimate"] = float("nan")

        with pytest.raises(TelemetryGap):
            load_h_run(*write_h_run(tmp_path, mutate_metrics=poison))

        def empty(rows: list[dict[str, Any]]) -> None:
            row = next(r for r in rows if r.get("schema") == "h-predictive-evaluation-v1")
            row["examples"] = []

        with pytest.raises(TelemetryGap, match="no examples"):
            load_h_run(*write_h_run(tmp_path / "e", mutate_metrics=empty))


def _gap_entry(rows: list[dict[str, Any]], **changes: Any) -> None:
    entry = {
        "effective_sample_size": None,
        "gap": "zero_importance_weight_mass",
        "likelihood_relative_mcse": None,
        "log_estimate": None,
    }
    entry.update(changes)
    _first_evaluation(rows)["examples"][0]["by_sample_count"]["2"][1] = entry


@pytest.mark.unit
class TestZeroImportanceMassGapEntries:
    def test_a_gap_entry_loads_and_is_counted_on_its_global_step(self, tmp_path: Path) -> None:
        run = load_h_run(*write_h_run(tmp_path, mutate_metrics=_gap_entry))
        assert [step for step, _ in run.evaluations] == [4, 6]
        assert dict(run.evaluations)[4]["eval/gap_rows_k2"] == 1.0
        doc = aggregate_h_runs([run], {IDENTITY_A: "only"})
        assert doc["evaluation"]["metrics"]["eval/gap_rows_k2"][0]["values"] == {"only": 1.0}

    @pytest.mark.parametrize(
        "entry",
        [
            {"gap": None, "log_estimate": None},
            {"gap": None, "likelihood_relative_mcse": None},
            {"gap": "zero_importance_weight_mass", "log_estimate": -1.0},
            {"gap": "unknown_gap_kind"},
        ],
    )
    def test_any_other_gap_shape_is_refused_and_names_the_global_step(
        self, tmp_path: Path, entry: dict[str, Any]
    ) -> None:
        metrics, log = write_h_run(tmp_path, mutate_metrics=lambda rows: _gap_entry(rows, **entry))
        with pytest.raises(TelemetryGap, match="global step 4") as excinfo:
            load_h_run(metrics, log)
        assert str(metrics) in str(excinfo.value)


# ------------------------------------------------------------------ unit: static contracts
_ANALYZE = Path(h_aggregate.__file__).parent
_H_MODULES = ("h_aggregate.py", "h_charts.py", "h_report.py", "h_attach.py")


@pytest.mark.unit
class TestModuleContracts:
    @pytest.mark.parametrize("name", _H_MODULES)
    def test_no_print_no_composition_import_no_private_cross_import(self, name: str) -> None:
        tree = ast.parse((_ANALYZE / name).read_text(encoding="utf-8"))
        prints = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
        ]
        assert prints == [], "machine-consumed stdout comes only from the CLI"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("entropic_substrate.composition"), node.module
                assert not node.module.startswith("entropic_substrate.historical"), node.module
                for alias in node.names:
                    assert not alias.name.startswith("_"), f"{node.module}.{alias.name}"
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith("mlflow") for a in node.names)

    @pytest.mark.parametrize("name", _H_MODULES)
    def test_no_broad_except_and_no_local_strict_json_hook(self, name: str) -> None:
        text = (_ANALYZE / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        broad = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ExceptHandler)
            and isinstance(n.type, ast.Name)
            and n.type.id in {"Exception", "BaseException"}
        ]
        if name == "h_attach.py":
            # the one sanctioned collect-and-reraise site, pinned by the attach tests
            assert len(broad) == 1
        else:
            assert broad == []
        for banned in ("_pairs", "_strict_pairs", "_nonfinite", "_reject_constant"):
            assert banned not in text

    def test_the_reader_is_reused_not_forked(self) -> None:
        text = (_ANALYZE / "h_aggregate.py").read_text(encoding="utf-8")
        assert "final_h_run_end(" in text
        assert "require_json_object_lines(" in text
        assert "_json_object_lines" not in text.replace("require_json_object_lines", "")


# ------------------------------------------------------------------ unit: hand-off recipient
def _evidence(identity: str = IDENTITY_B, warmup: int = 2) -> CodecHandoffEvidence:
    return CodecHandoffEvidence(identity, IDENTITY_A, warmup, "5" * 64, "6" * 64)


def _recipient_run(directory: Path, **kwargs: Any) -> tuple[Path, Path]:
    """A recipient commits no codec_warmup row: its codec arrived through the hand-off."""
    return write_h_run(
        directory,
        IDENTITY_B,
        REQUEST_B,
        tag="r",
        mutate_metrics=_drop(lambda r: r.get("phase") == "codec_warmup"),
        **kwargs,
    )


@pytest.mark.unit
class TestCodecHandoffRecipient:
    """D12 (FOR-1692): a recipient is admitted only with hand-off evidence."""

    def test_a_recipient_is_admitted_with_evidence_and_keeps_the_global_axis(
        self, tmp_path: Path
    ) -> None:
        run = load_h_run(*_recipient_run(tmp_path), handoff=_evidence())
        control = load_h_run(*write_h_run(tmp_path / "control", IDENTITY_B, REQUEST_B))
        assert run.handoff == _evidence() and run.warmup_steps == 2
        assert [step for step, phase, _ in run.training] == [3, 4, 5, 6]
        assert {phase for _, phase, _ in run.training} == {"prediction"}
        assert run.training == tuple(row for row in control.training if row[1] == "prediction")
        assert run.evaluations == control.evaluations and control.handoff is None

    def test_the_same_files_without_evidence_are_still_a_gap(self, tmp_path: Path) -> None:
        paths = _recipient_run(tmp_path)
        with pytest.raises(TelemetryGap, match="committed codec_warmup rows number 0"):
            load_h_run(*paths)

    def test_evidence_never_combines_with_the_warmup_override(self, tmp_path: Path) -> None:
        paths = _recipient_run(tmp_path)
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_h_run(*paths, warmup_steps=2, handoff=_evidence())

    def test_evidence_must_be_the_typed_value(self, tmp_path: Path) -> None:
        paths = _recipient_run(tmp_path)
        with pytest.raises(TypeError, match="CodecHandoffEvidence"):
            load_h_run(*paths, handoff=True)  # pyright: ignore[reportArgumentType]

    def test_a_recipient_with_a_warmup_row_is_refused(self, tmp_path: Path) -> None:
        paths = write_h_run(tmp_path, IDENTITY_B, REQUEST_B)
        with pytest.raises(TelemetryGap, match="commits no codec_warmup row, but 2 are present"):
            load_h_run(*paths, handoff=_evidence())

    def test_a_truncated_warmup_is_not_a_recipient(self, tmp_path: Path) -> None:
        paths = write_h_run(
            tmp_path,
            IDENTITY_B,
            REQUEST_B,
            mutate_metrics=_drop(lambda r: r.get("phase") == "codec_warmup" and r.get("step") == 1),
        )
        with pytest.raises(TelemetryGap, match="commits no codec_warmup row, but 1 are present"):
            load_h_run(*paths, handoff=_evidence())

    def test_the_inherited_length_must_equal_the_worker_logs(self, tmp_path: Path) -> None:
        paths = _recipient_run(tmp_path)
        with pytest.raises(TelemetryGap, match="inherited 3"):
            load_h_run(*paths, handoff=_evidence(warmup=3))

    def test_the_evidence_must_name_this_runs_profile(self, tmp_path: Path) -> None:
        paths = _recipient_run(tmp_path)
        with pytest.raises(TelemetryGap, match="is not the hand-off recipient"):
            load_h_run(*paths, handoff=_evidence(identity=IDENTITY_A))

    @pytest.mark.parametrize(
        "changes",
        [
            {"recipient_identity": "xyz"},
            {"donor_identity": "A" * 64},
            {"record_digest": "7" * 63},
            {"donor_seal_digest": ""},
            {"warmup_steps": 0},
            {"warmup_steps": True},
        ],
    )
    def test_evidence_is_validated_at_construction(self, changes: dict[str, Any]) -> None:
        fields = {
            "recipient_identity": IDENTITY_B,
            "donor_identity": IDENTITY_A,
            "warmup_steps": 2,
            "record_digest": "5" * 64,
            "donor_seal_digest": "6" * 64,
            **changes,
        }
        with pytest.raises(ValueError, match="hand-off evidence"):
            CodecHandoffEvidence(**fields)

    def test_non_recipient_aggregates_gain_no_key(self, tmp_path: Path) -> None:
        runs, labels = _two_arms(tmp_path)
        recipient = load_h_run(*_recipient_run(tmp_path / "r"), handoff=_evidence())
        for run in (*runs, recipient):
            assert "handoff" not in json.dumps(aggregate_h_runs([run], {run.profile_identity: "x"}))
        assert [run.handoff for run in runs] == [None, None] and labels


# ------------------------------------------------------------------ integration: real capture
def _fixture_runs() -> dict[str, tuple[Path, Path, str]]:
    """``{8-hex run suffix: (metrics, worker log, profile identity)}`` for the HEpP7g capture."""
    if not FIXTURE_DIR.is_dir():
        raise AssertionError(
            f"{FIXTURE_DIR} is absent: run `dvc pull results/h0-plumbing-1k-v3-HEpP7g.dvc` "
            "(this test fails rather than skips: a skip is a false green)"
        )
    runs: dict[str, tuple[Path, Path, str]] = {}
    for metrics in sorted(FIXTURE_DIR.glob("*-metrics.jsonl")):
        worker_log = metrics.with_name(metrics.name.replace("-metrics.jsonl", "-worker-log.jsonl"))
        assert worker_log.is_file(), worker_log
        suffix = metrics.name.split("-")[-2]
        first = json.loads(metrics.read_text().splitlines()[0])
        runs[suffix] = (metrics, worker_log, first["profile_identity"])
    assert set(runs) == {"a163161f", "988d8388"}, sorted(runs)
    return runs


def fixture_arguments() -> tuple[list[tuple[Path, Path]], dict[str, str]]:
    runs = _fixture_runs()
    pairs = [(runs[s][0], runs[s][1]) for s in ("a163161f", "988d8388")]
    labels = {runs["a163161f"][2]: "magnitude", runs["988d8388"][2]: "sphere"}
    return pairs, labels


@pytest.mark.integration
class TestCapturedHepP7gRuns:
    def test_global_axis_reaches_1000_and_evaluations_sit_at_500_750_1000(self) -> None:
        pairs, labels = fixture_arguments()
        doc = aggregate_h_runs([load_h_run(*p) for p in pairs], labels)
        assert doc["training"]["paired_steps"] == list(range(1, 1001))
        assert doc["training"]["unpaired"] == {"magnitude": [], "sphere": []}
        assert doc["evaluation"]["paired_steps"] == [500, 750, 1000]
        assert doc["publication_seconds"]["paired_steps"] == [250, 500, 750, 1000]
        for arm in doc["arms"].values():
            assert arm["warmup_steps"] == 250
            assert arm["first_prediction_step"] == 251
            assert arm["step_range"] == [1, 1000]
            assert arm["completed"] is True
            assert arm["run_end"]["result_status"] == "probe_workload_complete"

    def test_a_reader_keying_on_the_bare_step_would_see_250_500_750(self) -> None:
        pairs, _ = fixture_arguments()
        for metrics, _ in pairs:
            raw = [
                json.loads(line)["step"]
                for line in metrics.read_text().splitlines()
                if json.loads(line).get("schema") == "h-predictive-evaluation-v1"
            ]
            assert raw == [250, 500, 750]

    def test_arm_identity_is_the_profile_identity_and_the_log_names_the_mlflow_run(self) -> None:
        pairs, labels = fixture_arguments()
        doc = aggregate_h_runs([load_h_run(*p) for p in pairs], labels)
        by_label = dict(doc["arms"].items())
        assert by_label["magnitude"]["profile_identity"] in labels
        assert by_label["magnitude"]["mlflow_run_id"].startswith("a163161f")
        assert by_label["sphere"]["mlflow_run_id"].startswith("988d8388")
        assert len(by_label["magnitude"]["mlflow_run_id"]) == 32

    def test_future_nll_is_reported_as_captured_negative_and_null_in_warmup(self) -> None:
        pairs, labels = fixture_arguments()
        doc = aggregate_h_runs([load_h_run(*p) for p in pairs], labels)
        for arm in doc["arms"].values():
            nll = arm["future_nll_per_dimension"]
            assert nll["null_rows"] == 250
            assert nll["rows"] == 750
            assert nll["min"] < 0  # reported, not clamped
            assert nll["negative_rows"] > 0
        rows = doc["training"]["metrics"]["future_nll_per_dimension"]
        assert rows[0]["values"] == {"magnitude": None, "sphere": None}

    def test_evaluation_gap_is_reported_as_captured(self) -> None:
        pairs, labels = fixture_arguments()
        doc = aggregate_h_runs([load_h_run(*p) for p in pairs], labels)
        for key in ("eval/gap_rows_k2", "eval/gap_rows_k4"):
            for row in doc["evaluation"]["metrics"][key]:
                assert row["values"] == {"magnitude": 0.0, "sphere": 0.0}

    def test_swapping_the_worker_logs_of_the_real_runs_is_refused(self) -> None:
        pairs, _ = fixture_arguments()
        (m_a, l_a), (m_b, l_b) = pairs
        with pytest.raises(TelemetryGap, match="different run"):
            load_h_run(m_a, l_b)
        with pytest.raises(TelemetryGap, match="different run"):
            load_h_run(m_b, l_a)

    def test_a_label_for_a_run_that_is_not_an_input_fails_loud(self) -> None:
        pairs, labels = fixture_arguments()
        runs = [load_h_run(*pairs[0])]
        with pytest.raises(ValueError, match=r"no --arm label|names an identity"):
            aggregate_h_runs(runs, labels)


# ------------------------------------------------ DH-46 (FOR-1693): the scientifically stopped stream
def stopped_rows(
    identity: str = IDENTITY_A,
    request: str = REQUEST_A,
    *,
    committed: int = 0,
    warmup: int = 2,
    result_status: str = "scientific_gap",
    status: str = "failed",
    phase_starts: int = 1,
    extra_log: tuple[dict[str, Any], ...] = (),
    extra_metrics: tuple[dict[str, Any], ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A donor stream that stopped INSIDE its warm-up: ``committed`` rows, then the run_end."""
    metrics, log = build_h_run_rows(identity, request)
    start, end = metrics[0], metrics[-1]
    rows = [_step_row(identity, "codec_warmup", s, 1.0) for s in range(1, committed + 1)]
    end = end | {"status": status, "result_status": result_status}
    end["successful_optimizer_steps"] = None
    worker_start = log[0]
    run = log[1]
    phase = [
        _log("phase.start", phase="codec_warmup", step_from=0, step_to=warmup)
        for _ in range(phase_starts)
    ]
    return [start, *rows, *extra_metrics, end], [worker_start, run, *phase, *extra_log]


def write_stopped_run(directory: Path, **kwargs: Any) -> tuple[Path, Path]:
    metrics, log = stopped_rows(**kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    return _dump(directory / "s-metrics.jsonl", metrics), _dump(
        directory / "s-worker-log.jsonl", log
    )


@pytest.mark.unit
class TestScientificallyStoppedStream:
    @pytest.mark.parametrize("committed", [0, 1])
    def test_a_donor_stopped_in_its_warmup_loads_on_the_default_path(
        self, tmp_path: Path, committed: int
    ) -> None:
        """Warm-up step 1 (zero rows) and step 2 of 2 (one row) both load with no override."""
        run = load_h_run(*write_stopped_run(tmp_path, committed=committed))
        assert run.warmup_steps == 2
        assert [step for step, _, _ in run.training] == list(range(1, committed + 1))
        assert {phase for _, phase, _ in run.training} <= {"codec_warmup"}
        assert run.evaluations == () and run.elm == ()
        assert run.run_end["result_status"] == "scientific_gap"

    def test_the_parity_pin_the_adapter_derives_missing_endpoint_with_no_override(
        self, tmp_path: Path
    ) -> None:
        from entropic_substrate.ladder.analyze.h1_observations import project_h1_cell
        from entropic_substrate.ladder.register.h_preservation import H1ExclusionReason

        for committed in (0, 1):
            run = load_h_run(*write_stopped_run(tmp_path / str(committed), committed=committed))
            cell = project_h1_cell(run)
            assert type(cell.endpoint_reason) is H1ExclusionReason
            assert cell.endpoint_reason == H1ExclusionReason.MISSING_ENDPOINT

    def test_every_downstream_consumer_takes_the_stopped_run_without_an_untyped_crash(
        self, tmp_path: Path
    ) -> None:
        """P5-12: run_summary, aggregate_h_runs, h_report, K3 and K7 each run once on both
        admitted donor streams; their outcomes are asserted, not assumed."""
        from entropic_substrate.ladder.analyze.h_aggregate import run_summary
        from entropic_substrate.ladder.analyze.h_final_probe_checks import (
            check_analysis_surface,
            check_capture,
        )
        from entropic_substrate.ladder.analyze.h_report import h_report_from_run

        files = {
            "magnitude": write_stopped_run(tmp_path / "a", committed=0),
            "sphere": write_stopped_run(
                tmp_path / "b", identity=IDENTITY_B, request=REQUEST_B, committed=1
            ),
        }
        stopped = [load_h_run(*pair) for pair in files.values()]
        summary = run_summary(stopped[0])
        assert summary["committed_step_rows"] == 0 and summary["completed"] is False
        assert summary["step_range"] is None and summary["evaluation_steps"] == []
        document = aggregate_h_runs(stopped, {IDENTITY_A: "magnitude", IDENTITY_B: "sphere"})
        assert document["training"]["paired_steps"] == []
        assert document["evaluation"]["paired_steps"] == []
        report = h_report_from_run(stopped[1])
        assert report["status"]["completed"] is False and report["steps"]["committed_rows"] == 1
        assert report["evaluation"]["steps"] == []
        capture = check_capture([m for m, _ in files.values()], [REQUEST_A, REQUEST_B])
        assert capture.verdict == "PASS", capture.reasons
        surface = check_analysis_surface({label: [pair] for label, pair in files.items()})
        assert surface.verdict == "PASS", surface.reasons

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("execution_defect", {"result_status": "execution_defect"}),
            ("diagnostic_gap", {"result_status": "diagnostic_gap"}),
            ("ok status", {"status": "ok"}),
            ("no phase.start", {"phase_starts": 0}),
            ("two phase.start", {"phase_starts": 2}),
            (
                "warm-up-final seal present",
                {
                    "committed": 2,
                    "extra_log": (
                        _log("checkpoint.seal", phase="codec_warmup", step=2, global_step=2),
                    ),
                },
            ),
            (
                "warm-up-final seal present beside fewer committed rows",
                {
                    "committed": 1,
                    "extra_log": (
                        _log("checkpoint.seal", phase="codec_warmup", step=2, global_step=2),
                    ),
                },
            ),
            (
                "evaluation.start present",
                {"extra_log": (_log("evaluation.start", global_step=3),)},
            ),
            (
                "prediction step row present",
                {"extra_metrics": (_step_row(IDENTITY_A, "prediction", 1, 1.0),)},
            ),
            (
                "complete warm-up rows",
                {"committed": 2},
            ),
            (
                "a prediction seal already gives the warm-up length",
                {
                    "committed": 1,
                    "extra_log": (
                        _log("checkpoint.seal", phase="prediction", step=1, global_step=4),
                    ),
                },
            ),
        ],
    )
    def test_everything_else_keeps_the_existing_refusals(
        self, tmp_path: Path, name: str, kwargs: dict[str, Any]
    ) -> None:
        """An execution-truncated stream, a stop that HAS its warm-up-final seal and every
        other shape the record does not describe raise with the messages they always had."""
        with pytest.raises(TelemetryGap) as raised:
            load_h_run(*write_stopped_run(tmp_path, **kwargs))
        message = str(raised.value)
        assert (
            "warm-up length cannot be derived" in message
            or "committed codec_warmup" in message
            or "at least one committed" in message
        ), (name, message)

    @pytest.mark.parametrize("declared", [0, -1])
    def test_a_non_positive_warmup_declaration_is_never_a_derived_length(
        self, tmp_path: Path, declared: int
    ) -> None:
        """``_scientifically_stopped_warmup`` accepts only a ``step_to`` >= 1. The refusal must be
        the existing derivation message (a mutated bound lets W=0 through to a later, different
        message), so the text is matched here rather than in the shared any-of check above."""
        with pytest.raises(TelemetryGap, match="warm-up length cannot be derived"):
            load_h_run(*write_stopped_run(tmp_path, warmup=declared))

    def test_the_unchanged_message_for_a_truncated_stream_is_byte_identical(
        self, tmp_path: Path
    ) -> None:
        metrics, log = stopped_rows(committed=1, result_status="execution_defect")
        directory = tmp_path
        m = _dump(directory / "m.jsonl", metrics)
        w = _dump(directory / "w.jsonl", log)
        with pytest.raises(TelemetryGap) as raised:
            load_h_run(m, w)
        assert str(raised.value) == (
            f"{w}: the warm-up length cannot be derived: prediction checkpoint.seal events "
            "must exist and agree on global_step - step, found offsets []"
        )

    def test_an_override_with_committed_rows_still_loads(self, tmp_path: Path) -> None:
        """The widened branch never combines with ``warmup_steps=`` or ``handoff=``: with an
        override and a committed row the existing check still decides."""
        run = load_h_run(*write_stopped_run(tmp_path, committed=1), warmup_steps=2)
        assert run.warmup_steps == 2

    @pytest.mark.parametrize("override", ["warmup_steps", "handoff"])
    def test_a_zero_row_pod_is_refused_under_either_override(
        self, tmp_path: Path, override: str
    ) -> None:
        """A stopped pod with no committed row, whose worker log holds one prediction seal, never
        trained. Both overrides derive the warm-up from that seal and would load it with an
        empty training axis; each refuses it with the reader's never-trained text."""
        seal = (_log("checkpoint.seal", phase="prediction", step=1, global_step=3),)
        if override == "warmup_steps":
            paths = write_stopped_run(tmp_path, committed=0, extra_log=seal)
            with pytest.raises(TelemetryGap) as raised:
                load_h_run(*paths, warmup_steps=2)
        else:
            paths = write_stopped_run(
                tmp_path, identity=IDENTITY_B, request=REQUEST_B, committed=0, extra_log=seal
            )
            with pytest.raises(TelemetryGap) as raised:
                load_h_run(*paths, handoff=_evidence())
        assert str(raised.value) == (
            f"{paths[0]}: an admissible H run holds at least one committed "
            "h-training-step-v2 row, found none — the run never trained"
        )

    def test_a_zero_row_recipient_without_a_prediction_seal_keeps_the_derivation_refusal(
        self, tmp_path: Path
    ) -> None:
        """Control: with no prediction seal the hand-off path refuses earlier, at the warm-up
        derivation, with the message it always had."""
        paths = write_stopped_run(tmp_path, identity=IDENTITY_B, request=REQUEST_B, committed=0)
        with pytest.raises(TelemetryGap, match="the warm-up length cannot be derived"):
            load_h_run(*paths, handoff=_evidence())


@pytest.mark.unit
@pytest.mark.parametrize("failing_call", [1, 2])
def test_the_real_worker_emits_the_scientific_stop_the_loader_keys_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_call: int
) -> None:
    """Emitter-to-reader pin (P5-10): the ``failed``/``scientific_gap`` run_end and the one
    ``codec_warmup`` ``phase.start`` come from the REAL ``run_object_worker`` on a donor whose
    warm-up diverged at step 1 (zero rows) or step 2 of 2 (one row); the default-path loader
    admits both and the L3 adapter derives MISSING_ENDPOINT, with no override anywhere. The
    fault is injected at the numerical primitive below the runner."""
    from tests.test_d1_trainer_object_transport import _ObjectStore
    from tests.test_sentence_worker_execution import prepared

    from entropic_substrate.core.sentence.codec import NeuralSentenceCodec
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.analyze.h1_observations import project_h1_cell
    from entropic_substrate.ladder.capture.reader.h_telemetry import is_scientific_stop
    from entropic_substrate.ladder.register.h_preservation import H1ExclusionReason
    from entropic_substrate.ladder.run.worker_command import run_worker_command
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    real = NeuralSentenceCodec.reconstruct
    calls = {"n": 0}

    def diverge(self: NeuralSentenceCodec, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == failing_call:
            raise UndefinedDirection(2)
        return real(self, *args, **kwargs)

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    monkeypatch.setattr(NeuralSentenceCodec, "reconstruct", diverge)
    workspace = tmp_path / "command-worker"
    workspace.mkdir()
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
        stderr=StringIO(),
    )
    assert status == 2
    observability = tmp_path / "command-worker.observability"
    run = load_h_run(observability / "metrics.jsonl", observability / "worker-log.jsonl")
    assert is_scientific_stop(run.run_end)
    assert (run.run_end["status"], run.run_end["result_status"]) == ("failed", "scientific_gap")
    assert run.warmup_steps == 2 and len(run.training) == failing_call - 1
    cell = project_h1_cell(run)
    assert type(cell.endpoint_reason) is H1ExclusionReason
    assert cell.endpoint_reason == H1ExclusionReason.MISSING_ENDPOINT
