"""H analysis aggregate: paired arms over captured H runs, on one global step axis.

Reads the two files an H worker leaves behind -- ``metrics.jsonl`` and
``worker-log.jsonl`` -- and nothing else: no network, no MLflow, a pure function of
the files. Admissibility goes through the existing H reader
(:func:`~entropic_substrate.ladder.capture.reader.h_telemetry.final_h_run_end` and
:func:`~entropic_substrate.ladder.capture.reader.h_telemetry.require_json_object_lines`),
reused and never forked; an inadmissible or self-contradicting file raises
:class:`~entropic_substrate.ladder.capture.reader.telemetry.TelemetryGap` naming the
file, and no empty or partial aggregate is ever returned (measurement path, fail LOUD).

MEASURED at 501c0447 on the HEpP7g capture (four files, two runs), each of which this
module pins rather than assumes:

* The captured rows carry no carrier, magnitude, sphere or ``config_hash`` marker.
  A run's arm identity is ``run_start.profile_identity`` and every step and
  evaluation row repeats it as ``run_identity``. Arm LABELS therefore come only from
  the caller (``identity=label`` pairs), never from a file name and never inferred.
* The file-name prefix is the profile identity, not the request digest; a worker log
  is bound to its metrics file by ``worker.start.request_sha256 ==
  run_start.request_sha256``, never by name.
* Step and evaluation rows carry a PHASE-LOCAL ``step`` (warm-up ``1..W``, then
  prediction ``1..N``); the evaluation rows are phase-local too. The global axis is
  rebuilt from the worker log: ``W = checkpoint.seal.global_step - step`` over the
  prediction seals, which must be constant and must equal both the count and the
  maximum step of the committed ``codec_warmup`` rows. Evaluation rows map in stream
  order, one to one, onto ``evaluation.start`` events. The ``annotate`` rows'
  ``checkpoint/<8-digit>/...`` keys are ALREADY global and are never offset.
* ``future_nll_per_dimension`` is null on every ``codec_warmup`` row (the future law
  is structurally absent there) and negative on most prediction rows; both are
  reported as captured, never clamped, abs-ed or filtered.

Every scalar the aggregate reports for a training or evaluation row equals
:class:`~entropic_substrate.ladder.run.h_metric_extractor.HTrainingStepExtractor`'s
projection of the same row (one definition, imported). That projection silently
omits non-finite values, so this module checks finiteness itself first and refuses
rather than drops.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from entropic_substrate.kernel.codec.strict_json import strict_loads
from entropic_substrate.ladder.capture.reader.h_telemetry import (
    final_h_run_end,
    require_json_object_lines,
)
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor

__all__ = [
    "AGGREGATE_SCHEMA",
    "HRun",
    "aggregate_h_runs",
    "load_h_run",
    "parse_arm_labels",
    "run_summary",
]

AGGREGATE_SCHEMA = "h-aggregate-v1"

_LIFECYCLE_SCHEMA = "h-run-lifecycle-v1"
_STEP_SCHEMA = "h-training-step-v2"
_EVALUATION_SCHEMA = "h-predictive-evaluation-v1"
_COMMITTED = "committed"
_WARMUP = "codec_warmup"
_PREDICTION = "prediction"
# The scalar names of the step schema; pinned equal to the extractor's projection keys
# by tests/test_h_aggregate.py (a schema fact, not a second formula).
_STEP_SCALARS = (
    "objective",
    "aggregate_norm",
    "decoder_ce",
    "future_nll_per_dimension",
    "wall_seconds",
)
_FUTURE_NLL = "future_nll_per_dimension"

_IDENTITY = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_PUBLICATION_KEY = re.compile(r"checkpoint/([0-9]{8})/publication_seconds")


@dataclass(frozen=True, slots=True)
class HRun:
    """One admitted H run: its lineage, its global-axis series and its worker timings."""

    metrics_path: Path
    worker_log_path: Path
    profile_identity: str
    request_sha256: str
    run_start: dict[str, Any]
    run_end: dict[str, Any]
    runtime: dict[str, Any]
    warmup_steps: int
    # (global step, phase, projected scalars); a null future_nll is absent from the dict
    training: tuple[tuple[int, str, dict[str, float]], ...]
    uncommitted_step_rows: int
    # (global step, projected eval/* metrics), in stream order
    evaluations: tuple[tuple[int, dict[str, float]], ...]
    # global checkpoint step -> publication seconds (annotate rows)
    publication_seconds: dict[int, float]
    seal_steps: tuple[int, ...]
    mlflow_run_id: str | None
    stages: tuple[dict[str, Any], ...]
    admissions: tuple[dict[str, Any], ...]


def _gap(message: str) -> NoReturn:
    raise TelemetryGap(message)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Strictly decode an already admitted JSON-lines file (duplicate keys and
    non-finite constants refused through the codec's own fail channel)."""
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue

            def fail(message: str, *, _number: int = number) -> NoReturn:
                _gap(f"{path}: line {_number}: {message}")

            value = strict_loads(raw, fail=fail)
            if not isinstance(value, dict):
                _gap(f"{path}: line {number} is not a JSON object")
            rows.append(value)  # pyright: ignore[reportUnknownArgumentType]
    return rows


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))  # pyright: ignore[reportArgumentType]


def _require_identity(value: object, *, where: str, path: Path) -> str:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        _gap(f"{path}: {where} must be 64 lowercase hex characters, got {value!r}")
    return value


def _worker_log_facts(
    path: Path, request_sha256: str
) -> tuple[
    int, list[int], list[int], str | None, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]
]:
    """Bind the worker log to the run and read the global-axis facts out of it.

    Returns ``(warmup_steps, seal_steps, evaluation_steps, mlflow_run_id, stages,
    admissions)``.
    """
    rows = _read_rows(path)
    starts = [r for r in rows if r.get("event") == "worker.start"]
    if len(starts) != 1:
        _gap(f"{path}: exactly one worker.start event is required, found {len(starts)}")
    if starts[0].get("request_sha256") != request_sha256:
        _gap(
            f"{path}: worker.start.request_sha256 {starts[0].get('request_sha256')!r} does not "
            f"equal the metrics run_start.request_sha256 {request_sha256!r} -- this worker log "
            "belongs to a different run"
        )
    runs = [r for r in rows if r.get("event") == "mlflow.run"]
    if len(runs) > 1:
        _gap(f"{path}: at most one mlflow.run event is allowed, found {len(runs)}")
    mlflow_run_id: str | None = None
    if runs:
        run_id = runs[0].get("run_id")
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            _gap(f"{path}: mlflow.run.run_id must be 32 lowercase hex characters, got {run_id!r}")
        mlflow_run_id = run_id

    seals = [r for r in rows if r.get("event") == "checkpoint.seal"]
    seal_steps: list[int] = []
    offsets: set[int] = set()
    for seal in seals:
        global_step, step = seal.get("global_step"), seal.get("step")
        if type(global_step) is not int or type(step) is not int:
            _gap(f"{path}: checkpoint.seal without integer step/global_step: {seal!r}")
        seal_steps.append(global_step)
        if seal.get("phase") == _PREDICTION:
            offsets.add(global_step - step)
    if len(offsets) != 1:
        _gap(
            f"{path}: the warm-up length cannot be derived: prediction checkpoint.seal events "
            f"must exist and agree on global_step - step, found offsets {sorted(offsets)}"
        )
    warmup_steps = offsets.pop()
    if warmup_steps < 0:
        _gap(f"{path}: derived warm-up length {warmup_steps} is negative")

    evaluation_steps: list[int] = []
    for event in rows:
        if event.get("event") == "evaluation.start":
            global_step = event.get("global_step")
            if type(global_step) is not int:
                _gap(f"{path}: evaluation.start without an integer global_step: {event!r}")
            evaluation_steps.append(global_step)

    stages = tuple(
        {"seconds": float(r["seconds"]), "stage": str(r["stage"])}
        for r in rows
        if r.get("event") == "stage.end"
        and isinstance(r.get("stage"), str)
        and _finite_number(r.get("seconds"))
    )
    admissions = tuple(
        {
            "caller": str(r["caller"]),
            "evaluations": r.get("evaluations"),
            "journals_read": r.get("journals_read"),
            "seals_walked": r.get("seals_walked"),
            "seconds": float(r["seconds"]),
        }
        for r in rows
        if r.get("event") == "admission.end"
        and isinstance(r.get("caller"), str)
        and _finite_number(r.get("seconds"))
    )
    return warmup_steps, seal_steps, evaluation_steps, mlflow_run_id, stages, admissions


def _check_step_scalars(row: dict[str, Any], phase: str, path: Path) -> None:
    for key in _STEP_SCALARS:
        value = row.get(key)
        if value is None:
            if key == _FUTURE_NLL and phase == _WARMUP:
                continue  # structurally absent in the codec warm-up: admissible only here
            _gap(f"{path}: step {row.get('step')} ({phase}) has a null {key}")
        if not _finite_number(value):
            _gap(f"{path}: step {row.get('step')} ({phase}) has a non-finite {key}: {value!r}")


def _check_evaluation_numbers(row: dict[str, Any], path: Path) -> None:
    examples = row.get("examples")
    if not isinstance(examples, list) or not examples:
        _gap(f"{path}: evaluation at step {row.get('step')} carries no examples")
    for example in examples:  # pyright: ignore[reportUnknownVariableType]
        by_count = example.get("by_sample_count") if isinstance(example, dict) else None
        if not isinstance(by_count, dict) or not by_count:
            _gap(
                f"{path}: evaluation at step {row.get('step')} has an example without by_sample_count"
            )
        for entries in by_count.values():  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(entries, list) or not entries:
                _gap(
                    f"{path}: evaluation at step {row.get('step')} has an empty sample-count entry"
                )
            for entry in entries:  # pyright: ignore[reportUnknownVariableType]
                if not isinstance(entry, dict):
                    _gap(f"{path}: evaluation at step {row.get('step')} has a non-object entry")
                for key in ("log_estimate", "effective_sample_size"):
                    if not _finite_number(entry.get(key)):  # pyright: ignore[reportUnknownMemberType]
                        _gap(f"{path}: evaluation at step {row.get('step')} has a non-finite {key}")


def load_h_run(metrics_path: str | Path, worker_log_path: str | Path) -> HRun:
    """Admit one captured H run and rebuild its global step axis.

    Raises :class:`TelemetryGap` naming the file for every inadmissible or
    self-contradicting input; never returns a partial run.
    """
    metrics = Path(metrics_path)
    worker_log = Path(worker_log_path)
    run_end = final_h_run_end(metrics)
    require_json_object_lines(worker_log)
    rows = _read_rows(metrics)

    start = next(
        r for r in rows if r.get("schema") == _LIFECYCLE_SCHEMA and r.get("event") == "run_start"
    )
    identity = _require_identity(
        start.get("profile_identity"), where="run_start.profile_identity", path=metrics
    )
    runtime: object = start.get("runtime")
    request_sha256 = _require_identity(
        start.get("request_sha256"), where="run_start.request_sha256", path=metrics
    )
    for row in rows:
        if "run_identity" in row and row["run_identity"] != identity:
            _gap(
                f"{metrics}: a row carries run_identity {row['run_identity']!r}, but "
                f"run_start.profile_identity is {identity!r}"
            )

    warmup_steps, seal_steps, evaluation_starts, mlflow_run_id, stages, admissions = (
        _worker_log_facts(worker_log, request_sha256)
    )

    extractor = HTrainingStepExtractor(warmup_steps=warmup_steps)
    training: list[tuple[int, str, dict[str, float]]] = []
    evaluations: list[tuple[int, dict[str, float]]] = []
    publication: dict[int, float] = {}
    uncommitted = 0
    warmup_rows = 0
    warmup_max = 0
    previous = 0
    for row in rows:
        schema = row.get("schema")
        if schema == _STEP_SCHEMA:
            if row.get("run_identity") != identity:
                _gap(f"{metrics}: step row {row.get('step')} carries no run_identity of this run")
            if row.get("status") != _COMMITTED:
                uncommitted += 1
                continue
            phase = row.get("phase")
            if phase not in (_WARMUP, _PREDICTION):
                _gap(f"{metrics}: committed row {row.get('step')} has no usable phase: {phase!r}")
            _check_step_scalars(row, phase, metrics)
            projected = extractor.extract(row)
            if projected is None:
                _gap(f"{metrics}: committed row {row.get('step')} ({phase}) has no projection")
            step = projected.step
            if step <= previous:
                _gap(
                    f"{metrics}: the reconstructed global step axis is not strictly increasing "
                    f"at {phase} step {row.get('step')} (global {step} after {previous})"
                )
            previous = step
            if phase == _WARMUP:
                warmup_rows += 1
                warmup_max = max(warmup_max, step)
            scalars = {k: v for k, v in projected.metrics.items() if k != "phase_index"}
            training.append((step, phase, scalars))
        elif schema == _EVALUATION_SCHEMA:
            if row.get("run_identity") != identity:
                _gap(f"{metrics}: evaluation row carries no run_identity of this run")
            _check_evaluation_numbers(row, metrics)
            projected = extractor.extract(row)
            if projected is None or not projected.metrics:
                _gap(f"{metrics}: evaluation at step {row.get('step')} has no projection")
            evaluations.append((projected.step, dict(projected.metrics)))
        elif row.get("event") == "annotate":
            for key, value in row.items():
                match = _PUBLICATION_KEY.fullmatch(key)
                if match is None:
                    continue
                step = int(match.group(1))
                if step in publication:
                    _gap(f"{metrics}: duplicate publication timing for checkpoint {step}")
                try:
                    seconds = float(value) if isinstance(value, str) else math.nan
                except ValueError:
                    seconds = math.nan
                if not math.isfinite(seconds):
                    _gap(
                        f"{metrics}: checkpoint {step} publication_seconds {value!r} is not a "
                        "finite number string"
                    )
                publication[step] = seconds

    if warmup_rows != warmup_steps or warmup_max != warmup_steps:
        _gap(
            f"{metrics}: the worker log puts the warm-up at {warmup_steps} steps, but the "
            f"committed codec_warmup rows number {warmup_rows} and end at {warmup_max}"
        )
    if [step for step, _ in evaluations] != evaluation_starts:
        _gap(
            f"{metrics}: evaluation rows at global steps {[s for s, _ in evaluations]} do not "
            f"match the evaluation.start events {evaluation_starts} of {worker_log} one to one"
        )
    unsealed = sorted(set(publication) - set(seal_steps))
    if unsealed:
        _gap(
            f"{metrics}: publication timing for checkpoints {unsealed} has no matching "
            "checkpoint.seal in the worker log"
        )
    return HRun(
        metrics_path=metrics,
        worker_log_path=worker_log,
        profile_identity=identity,
        request_sha256=request_sha256,
        run_start={k: v for k, v in start.items() if k != "runtime"},
        run_end=dict(run_end),
        runtime=dict(runtime) if isinstance(runtime, dict) else {},
        warmup_steps=warmup_steps,
        training=tuple(training),
        uncommitted_step_rows=uncommitted,
        evaluations=tuple(evaluations),
        publication_seconds=publication,
        seal_steps=tuple(seal_steps),
        mlflow_run_id=mlflow_run_id,
        stages=stages,
        admissions=admissions,
    )


def parse_arm_labels(pairs: Sequence[str]) -> dict[str, str]:
    """``identity=label`` strings to ``{identity: label}``; any malformed, duplicate or
    ambiguous pair raises :class:`ValueError` naming it."""
    labels: dict[str, str] = {}
    used: dict[str, str] = {}
    for pair in pairs:
        identity, separator, label = pair.partition("=")
        if not separator or not _IDENTITY.fullmatch(identity):
            raise ValueError(
                f"--arm {pair!r}: expected <64 lowercase hex profile identity>=<label>"
            )
        if not _LABEL.fullmatch(label):
            raise ValueError(
                f"--arm {identity}: label {label!r} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}"
            )
        if identity in labels:
            raise ValueError(f"--arm {identity}: identity given more than once")
        if label in used:
            raise ValueError(
                f"--arm {identity}: label {label!r} is already the label of {used[label]}"
            )
        labels[identity] = label
        used[label] = identity
    return labels


def _bind_labels(runs: Sequence[HRun], labels: Mapping[str, str]) -> dict[str, HRun]:
    by_identity: dict[str, HRun] = {}
    for run in runs:
        if run.profile_identity in by_identity:
            raise ValueError(f"run {run.profile_identity} was given more than once")
        by_identity[run.profile_identity] = run
    unknown = sorted(set(labels) - set(by_identity))
    if unknown:
        raise ValueError(f"--arm names an identity no input run carries: {unknown[0]}")
    unlabelled = sorted(set(by_identity) - set(labels))
    if unlabelled:
        raise ValueError(f"input run {unlabelled[0]} has no --arm label")
    return {labels[identity]: run for identity, run in by_identity.items()}


def _nll_summary(run: HRun) -> dict[str, Any]:
    values = [s[_FUTURE_NLL] for _, _, s in run.training if _FUTURE_NLL in s]
    return {
        "max": max(values) if values else None,
        "min": min(values) if values else None,
        "negative_rows": sum(1 for v in values if v < 0),
        "null_rows": sum(1 for _, _, s in run.training if _FUTURE_NLL not in s),
        "rows": len(values),
    }


def run_summary(run: HRun) -> dict[str, Any]:
    """One run's lineage, status and axis summary; every value is captured, none inferred."""
    steps = [step for step, _, _ in run.training]
    first_prediction = next((s for s, phase, _ in run.training if phase == _PREDICTION), None)
    return {
        "checkpoint_steps": sorted(run.publication_seconds),
        "committed_step_rows": len(run.training),
        "completed": run.run_end.get("status") == "ok",
        "evaluation_steps": [step for step, _ in run.evaluations],
        "first_prediction_step": first_prediction,
        "future_nll_per_dimension": _nll_summary(run),
        "lineage": {
            "code_commit": run.run_start.get("code_commit"),
            "image": run.run_start.get("image"),
            "run_class": run.run_start.get("run_class"),
        },
        "mlflow_run_id": run.mlflow_run_id,
        "profile_identity": run.profile_identity,
        "request_sha256": run.request_sha256,
        "run_end": {
            "result_status": run.run_end.get("result_status"),
            "status": run.run_end.get("status"),
            "successful_optimizer_steps": run.run_end.get("successful_optimizer_steps"),
            "wall_seconds": run.run_end.get("wall_seconds"),
        },
        "step_range": [steps[0], steps[-1]] if steps else None,
        "uncommitted_step_rows": run.uncommitted_step_rows,
        "warmup_steps": run.warmup_steps,
    }


def _paired(
    series: Mapping[str, Mapping[int, object]],
) -> tuple[list[int], dict[str, list[int]]]:
    """Steps present on every arm, and per arm the steps it holds that others lack."""
    common = sorted(set.intersection(*(set(s) for s in series.values())))
    unpaired = {label: sorted(set(s) - set(common)) for label, s in series.items()}
    return common, unpaired


def aggregate_h_runs(runs: Sequence[HRun], labels: Mapping[str, str]) -> dict[str, Any]:
    """Pair the labelled arms by global step.

    ``labels`` maps each run's full profile identity to its arm label; an unknown
    identity, an unlabelled run or a duplicate raises :class:`ValueError`. Steps held
    by only some arms are listed under ``unpaired``, never silently truncated.
    """
    if not runs:
        raise ValueError("aggregate needs at least one captured run")
    arms = _bind_labels(runs, labels)
    order = sorted(arms)

    train: dict[str, dict[int, dict[str, float]]] = {
        label: {step: scalars for step, _, scalars in arm.training} for label, arm in arms.items()
    }
    train_steps, train_unpaired = _paired(dict(train.items()))
    metrics = sorted(
        {m for arm in arms.values() for _, _, s in arm.training for m in s} | {_FUTURE_NLL}
    )
    training_doc: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        training_doc[metric] = [
            {"step": step, "values": {label: train[label][step].get(metric) for label in order}}
            for step in train_steps
        ]
    phase_index = [
        {
            "step": step,
            "values": {
                label: next(p for s, p, _ in arms[label].training if s == step) for label in order
            },
        }
        for step in train_steps
    ]

    evals: dict[str, dict[int, dict[str, float]]] = {
        label: dict(arm.evaluations) for label, arm in arms.items()
    }
    eval_steps, eval_unpaired = _paired(dict(evals.items()))
    eval_metrics = sorted({m for e in evals.values() for s in e.values() for m in s})
    evaluation_doc = {
        metric: [
            {"step": step, "values": {label: evals[label][step].get(metric) for label in order}}
            for step in eval_steps
        ]
        for metric in eval_metrics
    }

    pub: dict[str, dict[int, float]] = {
        label: dict(arm.publication_seconds) for label, arm in arms.items()
    }
    pub_steps, pub_unpaired = _paired(dict(pub.items()))

    return {
        "arms": {label: run_summary(arms[label]) for label in order},
        "evaluation": {
            "metrics": evaluation_doc,
            "paired_steps": eval_steps,
            "unpaired": eval_unpaired,
        },
        "phase_index": phase_index,
        "publication_seconds": {
            "paired_steps": pub_steps,
            "unpaired": pub_unpaired,
            "values": [
                {"step": step, "values": {label: pub[label][step] for label in order}}
                for step in pub_steps
            ],
        },
        "schema": AGGREGATE_SCHEMA,
        "training": {
            "metrics": training_doc,
            "paired_steps": train_steps,
            "unpaired": train_unpaired,
        },
    }
