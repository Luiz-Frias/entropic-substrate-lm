"""A stopped H run plus its resumed continuation, admitted as one run on one global axis.

An operator stop (FOR-1617) ends pod A at a sealed boundary; pod B resumes the same
lineage under a fresh request (its own nonce, so its own request digest and its own
captured ``metrics.jsonl`` / ``worker-log.jsonl``). The single-pod reader
:func:`~entropic_substrate.ladder.analyze.h_aggregate.load_h_run` refuses pod B on its
own: B starts in prediction and commits no ``codec_warmup`` row, which for one pod is a
gap. For a resumed lineage it is expected (ruling D3, 2026-10-22), so this module admits
the pods TOGETHER:

* every pod is admitted by ``load_h_run`` itself (no second predicate), given the
  lineage's warm-up length so that only the per-file warm-up coverage check is lifted;
* the warm-up length is derived once, from the prediction ``checkpoint.seal`` events of
  every pod's worker log (``global_step - step``), and must agree across pods;
* one profile identity, pairwise-distinct request digests;
* every pod but the last ended ``run_end status=ok`` with ``result_status`` in
  ``stopped_by_operator`` / ``paused`` and ``successful_optimizer_steps`` equal to its
  last committed global step;
* the committed global steps are continuous ACROSS the seam and within each pod
  (pod 1 starts at 1, pod k at pod k-1's last step + 1), and the warm-up rows over the
  union are exactly ``1..W``.

The result's ``run`` is an ordinary :class:`~entropic_substrate.ladder.analyze.h_aggregate.HRun`
on the global axis, so the aggregate, the report and the charts consume it unchanged:
``run_start`` is the first pod's, and ``run_end``, ``request_sha256``, ``runtime`` and
``mlflow_run_id`` are the final pod's; each pod's own lineage stays in ``segments``.
Every refusal raises :class:`~entropic_substrate.ladder.capture.reader.telemetry.TelemetryGap`
naming a file (measurement path, fail LOUD); nothing partial is returned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import NoReturn

from entropic_substrate.ladder.analyze.h_aggregate import HRun, load_h_run, read_h_rows
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

__all__ = ["HResumedRun", "load_h_resumed_run"]

_PREDICTION = "prediction"
_WARMUP = "codec_warmup"
# The worker's result statuses for a pod whose lineage continues elsewhere
# (object_worker.py run_end ``result_status = result["status"]``).
_RESUMABLE = ("stopped_by_operator", "paused")


@dataclass(frozen=True, slots=True)
class HResumedRun:
    """One lineage across pods: the stitched global-axis run and each pod's own run."""

    run: HRun
    segments: tuple[HRun, ...]


def _gap(message: str) -> NoReturn:
    raise TelemetryGap(message)


def _lineage_warmup(segments: Sequence[tuple[Path, Path]]) -> int:
    """The warm-up length every pod's prediction seals agree on (at least one must exist)."""
    offsets: dict[int, Path] = {}
    for _, worker_log in segments:
        for row in read_h_rows(worker_log):
            if row.get("event") != "checkpoint.seal" or row.get("phase") != _PREDICTION:
                continue
            global_step, step = row.get("global_step"), row.get("step")
            if type(global_step) is not int or type(step) is not int:
                _gap(f"{worker_log}: checkpoint.seal without integer step/global_step: {row!r}")
            offsets.setdefault(global_step - step, worker_log)
    if len(offsets) != 1:
        _gap(
            "the resumed lineage's warm-up length cannot be derived: the pods' prediction "
            f"checkpoint.seal events give offsets {sorted(offsets)} "
            f"({[str(p) for p in offsets.values()]})"
        )
    return next(iter(offsets))


def _require_one_lineage(pods: Sequence[HRun]) -> None:
    first = pods[0]
    for pod in pods[1:]:
        if pod.profile_identity != first.profile_identity:
            _gap(
                f"{pod.metrics_path}: profile identity {pod.profile_identity} differs from "
                f"{first.profile_identity} of {first.metrics_path}; one lineage has one profile"
            )
    digests = [pod.request_sha256 for pod in pods]
    if len(set(digests)) != len(digests):
        _gap(
            f"{pods[-1].metrics_path}: the pods' request digests {digests} repeat; every "
            "resumed pod runs its own request (a fresh nonce)"
        )
    for pod in pods[:-1]:
        end = pod.run_end
        last = pod.training[-1][0] if pod.training else None
        if end.get("status") != "ok" or end.get("result_status") not in _RESUMABLE:
            _gap(
                f"{pod.metrics_path}: run_end status {end.get('status')!r} / result_status "
                f"{end.get('result_status')!r} is not a stop or pause; only a stopped or "
                "paused pod is continued by a later one"
            )
        if end.get("successful_optimizer_steps") != last:
            _gap(
                f"{pod.metrics_path}: run_end.successful_optimizer_steps "
                f"{end.get('successful_optimizer_steps')!r} differs from the last committed "
                f"global step {last}"
            )


def _require_continuous(pods: Sequence[HRun], warmup_steps: int) -> None:
    expected = 1
    for pod in pods:
        steps = [step for step, _, _ in pod.training]
        if not steps:
            _gap(f"{pod.metrics_path}: the pod committed no training step")
        if steps[0] != expected or steps != list(range(steps[0], steps[-1] + 1)):
            _gap(
                f"{pod.metrics_path}: the global step axis is not continuous across the "
                f"resume seam: this pod commits {steps[0]}..{steps[-1]} "
                f"({len(steps)} rows) where the lineage continues at {expected}"
            )
        expected = steps[-1] + 1
    warmup = [step for pod in pods for step, phase, _ in pod.training if phase == _WARMUP]
    if warmup != list(range(1, warmup_steps + 1)):
        _gap(
            f"{pods[-1].metrics_path}: the lineage's committed codec_warmup rows "
            f"{warmup[:3]}..{warmup[-3:]} are not exactly 1..{warmup_steps}"
        )


def _stitch(pods: Sequence[HRun], warmup_steps: int) -> HRun:
    first, final = pods[0], pods[-1]
    publication: dict[int, float] = {}
    for pod in pods:
        for step, seconds in pod.publication_seconds.items():
            if step in publication:
                _gap(f"{pod.metrics_path}: checkpoint {step} is published by two pods")
            publication[step] = seconds
    evaluation_steps = [step for pod in pods for step, _ in pod.evaluations]
    for earlier, later in pairwise(evaluation_steps):
        if later <= earlier:
            owners = [
                pod.metrics_path for pod in pods if any(s == later for s, _ in pod.evaluations)
            ]
            _gap(
                f"{owners[-1]}: evaluation at global step {later} follows {earlier} across "
                f"the lineage ({' and '.join(str(o) for o in owners)}); evaluation steps are "
                "not strictly increasing, and a repeated step would be silently collapsed"
            )
    return HRun(
        metrics_path=final.metrics_path,
        worker_log_path=final.worker_log_path,
        profile_identity=final.profile_identity,
        request_sha256=final.request_sha256,
        run_start=dict(first.run_start),
        run_end=dict(final.run_end),
        runtime=dict(final.runtime),
        warmup_steps=warmup_steps,
        training=tuple(row for pod in pods for row in pod.training),
        uncommitted_step_rows=sum(pod.uncommitted_step_rows for pod in pods),
        evaluations=tuple(row for pod in pods for row in pod.evaluations),
        publication_seconds=publication,
        seal_steps=tuple(step for pod in pods for step in pod.seal_steps),
        mlflow_run_id=final.mlflow_run_id,
        stages=tuple(stage for pod in pods for stage in pod.stages),
        admissions=tuple(admission for pod in pods for admission in pod.admissions),
        elm=tuple(row for pod in pods for row in pod.elm),
    )


def load_h_resumed_run(segments: Sequence[tuple[str | Path, str | Path]]) -> HResumedRun:
    """Admit pods ``[(metrics, worker_log), ...]`` (in lineage order) as one resumed run.

    Raises :class:`TelemetryGap` naming a file for every inadmissible pod, a broken
    lineage, a discontinuous global axis or fewer than two pods.
    """
    pairs = [(Path(metrics), Path(worker_log)) for metrics, worker_log in segments]
    if len(pairs) < 2:
        _gap(f"a resumed run needs at least two pods, got {len(pairs)}: {pairs}")
    warmup_steps = _lineage_warmup(pairs)
    pods = tuple(load_h_run(m, w, warmup_steps=warmup_steps) for m, w in pairs)
    _require_one_lineage(pods)
    _require_continuous(pods, warmup_steps)
    return HResumedRun(run=_stitch(pods, warmup_steps), segments=pods)
