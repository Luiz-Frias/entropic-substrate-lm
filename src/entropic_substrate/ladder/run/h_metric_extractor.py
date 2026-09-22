"""H lane's :class:`MetricExtractor` -- projects a committed ``h-training-step-v2``
row into the scalar metrics an MLflow mirror can carry.

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


class HTrainingStepExtractor:
    """Projects one committed H training-step row into ``ExtractedMetrics``.

    Returns ``None`` for anything that is not a committed
    ``h-training-step-v2`` event, exactly like
    :class:`~entropic_substrate.ladder.capture.ports.run_descriptor.LegacyMetricExtractor`
    returns ``None`` for a non-``eval`` E-series event -- the same
    "opt in per event" shape the :class:`MetricExtractor` protocol expects.
    """

    def extract(self, event: object) -> ExtractedMetrics | None:
        if not isinstance(event, dict):
            return None
        if event.get("schema") != "h-training-step-v2" or event.get("status") != "committed":
            return None
        step = event.get("step")
        if type(step) is not int:
            return None
        metrics: dict[str, float] = {}
        for key in _SCALAR_FIELDS:
            value = event.get(key)
            if type(value) in (int, float) and math.isfinite(cast(float, value)):
                metrics[key] = float(cast(float, value))
        return ExtractedMetrics(step=step, metrics=metrics)
