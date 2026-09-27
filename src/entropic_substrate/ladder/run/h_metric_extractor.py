"""H lane's :class:`MetricExtractor` -- projects committed ``h-training-step-v2``
rows and sealed ``h-predictive-evaluation-v1`` reports into the scalar metrics
an MLflow mirror can carry, on one global optimizer-step axis (exit-gate lane,
2026-09-26; see the class docstring for the measured motivation).

Landed at p4 (RUNG 2b, conductor CONTINUATION #7 ruling 2026-09-22): additive
observability only. This module never constructs a :class:`MetricSink`
implementation (that stays the sole responsibility of
``ladder.capture.adapters.composite.build_sink_for_run``, the run-descriptor
door's single construction seam) and never touches the D1 journal's
authoritative bytes -- :mod:`entropic_substrate.ladder.run.journal` and the
``sha256`` chain it feeds are untouched by this module or by anything that
calls it. It emits exactly the finite scalar fields a committed step row
already carries; it never invents a value the numerical engine did not
measure.
"""

from __future__ import annotations

import math
from typing import cast

from entropic_substrate.ladder.capture.ports.run_descriptor import ExtractedMetrics

__all__ = ["HTrainingStepExtractor"]

# The h-training-step-v2 schema's own finite-or-None numeric fields
# (execution.py:advance) -- see that module's docstring for the schema.
# Deliberately excludes source/history/encoder/boundary byte COUNTS: those
# are exposure accounting, not training signal, and are not currently
# projected to keep the metric surface to what a reviewer expects from a
# loss/gradient dashboard. A later phase may widen this tuple; doing so is
# additive and does not change this module's contract.
_SCALAR_FIELDS = (
    "objective",
    "aggregate_norm",
    "decoder_ce",
    "future_nll_per_dimension",
    "wall_seconds",
)


_PHASE_INDEX = {"codec_warmup": 0.0, "prediction": 1.0}


def _finite(value: object) -> float | None:
    if type(value) in (int, float) and math.isfinite(cast(float, value)):
        return float(cast(float, value))
    return None


class HTrainingStepExtractor:
    """Projects committed H training-step rows AND sealed evaluation reports into
    ``ExtractedMetrics`` on ONE global step axis.

    EXIT-GATE LANE (operator-cleared 2026-09-26). MEASURED at bbfbceb: the
    engine resets ``step`` to 0 at ``begin_prediction`` (ladder/run/execution.py),
    so the v1 projection put warmup step *n* and prediction step *n* on the same
    MLflow x-coordinate for every key. The extractor now takes the profile's
    ``warmup_steps`` and reports the global optimizer step: codec_warmup rows keep
    their step, prediction rows are offset by ``warmup_steps``. A committed row
    without a ``phase`` cannot be placed on that axis and is refused (``None``),
    never guessed. ``phase_index`` (0/1) marks the transition.

    Evaluation reports (``h-predictive-evaluation-v1``, always prediction-phase)
    are projected as scalar summaries per importance sample count *k*:
    ``eval/log_marginal_k{k}`` (mean of the finite ``log_estimate`` rows),
    ``eval/ess_k{k}`` (mean finite effective sample size), ``eval/gap_rows_k{k}``
    (rows whose importance mass was zero), plus ``eval/wall_seconds`` and
    ``eval/cuda_seconds`` when present. Nothing is invented: a count with no
    finite row contributes only its gap count. The authoritative journal and the
    evaluation artifacts are untouched; this remains the additive mirror.

    Returns ``None`` for every other event, exactly like
    :class:`~entropic_substrate.ladder.capture.ports.run_descriptor.LegacyMetricExtractor`
    returns ``None`` for a non-``eval`` E-series event.
    """

    def __init__(self, *, warmup_steps: int) -> None:
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("the global step axis needs the profile's warmup length")
        self._warmup_steps = warmup_steps

    def extract(self, event: object) -> ExtractedMetrics | None:
        if not isinstance(event, dict):
            return None
        schema = event.get("schema")
        if schema == "h-training-step-v2":
            return self._training(event)
        if schema == "h-predictive-evaluation-v1":
            return self._evaluation(event)
        return None

    def _training(self, event: dict[str, object]) -> ExtractedMetrics | None:
        if event.get("status") != "committed":
            return None
        step = event.get("step")
        phase = event.get("phase")
        if type(step) is not int or phase not in _PHASE_INDEX:
            return None
        metrics: dict[str, float] = {"phase_index": _PHASE_INDEX[cast(str, phase)]}
        for key in _SCALAR_FIELDS:
            value = _finite(event.get(key))
            if value is not None:
                metrics[key] = value
        offset = 0 if phase == "codec_warmup" else self._warmup_steps
        return ExtractedMetrics(step=step + offset, metrics=metrics)

    def _evaluation(self, event: dict[str, object]) -> ExtractedMetrics | None:
        step = event.get("step")
        examples = event.get("examples")
        if type(step) is not int or not isinstance(examples, list):
            return None
        estimates: dict[str, list[float]] = {}
        sizes: dict[str, list[float]] = {}
        gaps: dict[str, int] = {}
        for example in examples:
            if not isinstance(example, dict):
                return None
            by_count = example.get("by_sample_count")
            if not isinstance(by_count, dict):
                return None
            for count, rows in cast(dict[object, object], by_count).items():
                key = str(count)
                if not isinstance(rows, list):
                    return None
                for row in cast(list[object], rows):
                    if not isinstance(row, dict):
                        return None
                    r = cast(dict[str, object], row)
                    if r.get("gap") is not None:
                        gaps[key] = gaps.get(key, 0) + 1
                    estimate = _finite(r.get("log_estimate"))
                    if estimate is not None:
                        estimates.setdefault(key, []).append(estimate)
                    size = _finite(r.get("effective_sample_size"))
                    if size is not None:
                        sizes.setdefault(key, []).append(size)
                    gaps.setdefault(key, gaps.get(key, 0))
        metrics: dict[str, float] = {}
        for key in sorted(gaps, key=lambda k: (len(k), k)):
            if key in estimates:
                metrics[f"eval/log_marginal_k{key}"] = sum(estimates[key]) / len(estimates[key])
            if key in sizes:
                metrics[f"eval/ess_k{key}"] = sum(sizes[key]) / len(sizes[key])
            metrics[f"eval/gap_rows_k{key}"] = float(gaps[key])
        for field in ("wall_seconds", "cuda_seconds"):
            value = _finite(event.get(field))
            if value is not None:
                metrics[f"eval/{field}"] = value
        return ExtractedMetrics(step=step + self._warmup_steps, metrics=metrics)
