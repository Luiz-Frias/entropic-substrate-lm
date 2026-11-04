"""H lane's :class:`MetricExtractor` -- projects committed ``h-training-step-v2``
rows and sealed ``h-predictive-evaluation-v1``/``-v2`` reports into the scalar metrics
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
from entropic_substrate.ladder.run.worker_log_events import (
    ADMISSION_END,
    WORKER_STAGE_SCHEMA,
    project_worker_stage_record,
)

__all__ = [
    "EVALUATION_SCHEMAS",
    "LATENT_OPERANDS",
    "PROPOSALS",
    "HTrainingStepExtractor",
    "project_evaluation",
]

# The evaluation schemas this mirror reads: the v1 report (the H0 captures) and the v2 report
# (FOR-1704: prior and mixture proposals, diagnostics, latent usage). The analysis side keeps its
# own literal tuple and never imports a checkpoint-reader member for these strings; a test pins
# them equal to the writer's admitted set.
EVALUATION_SCHEMAS = ("h-predictive-evaluation-v1", "h-predictive-evaluation-v2")
# The closed proposal vocabulary of the v2 record and the closed latent-usage operand vocabulary,
# each pinned equal to its writer-side Literal/StrEnum by tests/test_h_metric_extractor_l4d.py.
PROPOSALS = ("q", "prior", "mixture")
LATENT_OPERANDS = ("posterior_mean", "draw_average")

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

    Evaluation reports (``h-predictive-evaluation-v1`` and ``-v2``, always prediction-phase)
    are projected as scalar summaries per importance sample count *k*:
    ``eval/log_marginal_k{k}`` (mean of the finite ``log_estimate`` rows),
    ``eval/ess_k{k}`` (mean finite effective sample size), ``eval/gap_rows_k{k}``
    (rows whose importance mass was zero), plus ``eval/wall_seconds`` and
    ``eval/cuda_seconds`` when present. Nothing is invented: a count with no
    finite row contributes only its gap count. The authoritative journal and the
    evaluation artifacts are untouched; this remains the additive mirror.

    FOR-1704 (L4d): a v2 report adds the prior and mixture proposals, per-proposal ratio of sums
    (``sum(log_estimate) / sum(source_bytes)`` over the SAME finite rows; gap rows' bytes are
    excluded and the key is absent, never 0, when no row is finite), log-weight spread, the
    log-radius and cosine diagnostics, and signed latent usage. The q keys keep their bare names;
    every other proposal's k-suffixed key carries the proposal segment before ``_k``. See
    :func:`project_evaluation`.

    Keys are ALWAYS k-suffixed (FOR-1664): a bare ``eval/log_marginal`` or ``eval/ess``
    is never emitted, because an estimate without its k reads as converged when it is
    not. The k set is the profile's whole evaluated population: the importance-sample
    curve AND the fixed diagnostic count (``RunSchedule.evaluated_samples``), so the
    diagnostic rows arrive under their own k (for example ``eval/ess_k256``) exactly as
    the curve rows do. An effective sample size is bounded by its k, so the smoke
    setting's ``eval/ess_k2``/``eval/ess_k4`` carry no convergence evidence.

    Final-leg stage and admission measurements (``h-worker-stage-v1``, FOR-1669) are
    projected as ``stage/<stage>/seconds`` and ``admission/<caller>/{seconds,seals_walked,
    evaluations,journals_read}`` onto the LAST global step a committed training row
    placed on the axis. They carry no step of their own, so a record seen before any
    training row is refused (``None``), never given a guessed step; evaluation reports
    and uncommitted rows do not move that anchor. MEASURED 2026-10-19 on the local tiny
    worker: ``materialization`` and ``runtime_admission`` end before the first training
    row and are therefore not mirrored; every later stage is.

    Returns ``None`` for every other event, exactly like
    :class:`~entropic_substrate.ladder.capture.ports.run_descriptor.LegacyMetricExtractor`
    returns ``None`` for a non-``eval`` E-series event.
    """

    def __init__(self, *, warmup_steps: int) -> None:
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("the global step axis needs the profile's warmup length")
        self._warmup_steps = warmup_steps
        # The global step of the last COMMITTED training row projected; ``None`` until
        # one was. The only state this extractor keeps (FOR-1669).
        self._last_training_step: int | None = None

    def extract(self, event: dict[str, object]) -> ExtractedMetrics | None:
        return self._project(event)

    def _project(self, event: object) -> ExtractedMetrics | None:
        # Typed ``object`` so the runtime refusal of a non-dict survives the
        # narrowed public signature (a checker would call that isinstance dead).
        if not isinstance(event, dict):
            return None
        schema = event.get("schema")
        if schema == "h-training-step-v2":
            return self._training(event)
        if schema in EVALUATION_SCHEMAS:
            return self._evaluation(event)
        if schema == WORKER_STAGE_SCHEMA:
            return self._worker_stage(event)
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
        self._last_training_step = step + offset
        return ExtractedMetrics(step=step + offset, metrics=metrics)

    def _worker_stage(self, event: dict[str, object]) -> ExtractedMetrics | None:
        if self._last_training_step is None:
            return None
        record = project_worker_stage_record(event)
        if record is None:
            return None
        seconds = cast(float, record["seconds"])
        if record["event"] == ADMISSION_END:
            prefix = f"admission/{record['caller']}"
            metrics = {
                f"{prefix}/seconds": seconds,
                f"{prefix}/seals_walked": float(cast(int, record["seals_walked"])),
                f"{prefix}/evaluations": float(cast(int, record["evaluations"])),
                f"{prefix}/journals_read": float(cast(int, record["journals_read"])),
            }
        else:
            metrics = {f"stage/{record['stage']}/seconds": seconds}
        return ExtractedMetrics(step=self._last_training_step, metrics=metrics)

    def _evaluation(self, event: dict[str, object]) -> ExtractedMetrics | None:
        step = event.get("step")
        if type(step) is not int:
            return None
        metrics = project_evaluation(event)
        if metrics is None:
            return None
        return ExtractedMetrics(step=step + self._warmup_steps, metrics=metrics)


def _proposal_key(base: str, proposal: str, k: str) -> str:
    """``eval/<base>_k<k>`` for q, ``eval/<base>_<proposal>_k<k>`` otherwise.

    The q keys keep their bare names; the proposal segment sits immediately before ``_k`` so the
    trailing ``_k<int>`` that ``h_aggregate._metric_order`` parses is unchanged. The flat metrics
    dict keeps only the last writer of a name, so a shared name would overwrite q.
    """
    return f"eval/{base}_k{k}" if proposal == "q" else f"eval/{base}_{proposal}_k{k}"


def _rows(by_count: object) -> dict[str, list[dict[str, object]]] | None:
    """The importance rows of one proposal keyed by k, or ``None`` when malformed."""
    if not isinstance(by_count, dict):
        return None
    out: dict[str, list[dict[str, object]]] = {}
    for count, rows in cast(dict[object, object], by_count).items():
        if not isinstance(rows, list):
            return None
        checked: list[dict[str, object]] = []
        for row in cast(list[object], rows):
            if not isinstance(row, dict):
                return None
            checked.append(cast(dict[str, object], row))
        out[str(count)] = checked
    return out


def _bytes(value: object, rows: int) -> list[int] | None:
    """Canonical source-byte counts aligned with ``rows``; ``None`` when unusable."""
    if not isinstance(value, list) or len(cast(list[object], value)) != rows:
        return None
    counts: list[int] = []
    for n in cast(list[object], value):
        if type(n) is not int or n < 1:
            return None
        counts.append(n)
    return counts


class _Tally:
    """Per-proposal accumulation of the importance rows, per k, over every example."""

    def __init__(self) -> None:
        self.estimates: dict[str, list[float]] = {}
        self.sizes: dict[str, list[float]] = {}
        self.gaps: dict[str, int] = {}
        # ratio of sums: log_estimate and source_bytes summed over the SAME finite rows
        self.numerator: dict[str, float] = {}
        self.denominator: dict[str, int] = {}
        self.ratio_available = True

    def add(self, by_count: dict[str, list[dict[str, object]]], source_bytes: object) -> None:
        for key, rows in by_count.items():
            counts = _bytes(source_bytes, len(rows))
            if counts is None:
                self.ratio_available = False
            for index, r in enumerate(rows):
                if r.get("gap") is not None:
                    self.gaps[key] = self.gaps.get(key, 0) + 1
                estimate = _finite(r.get("log_estimate"))
                if estimate is not None:
                    self.estimates.setdefault(key, []).append(estimate)
                    if counts is not None:
                        self.numerator[key] = self.numerator.get(key, 0.0) + estimate
                        self.denominator[key] = self.denominator.get(key, 0) + counts[index]
                size = _finite(r.get("effective_sample_size"))
                if size is not None:
                    self.sizes.setdefault(key, []).append(size)
                self.gaps.setdefault(key, self.gaps.get(key, 0))

    def emit(self, proposal: str, metrics: dict[str, float]) -> None:
        for key in sorted(self.gaps, key=lambda k: (len(k), k)):
            if key in self.estimates:
                metrics[_proposal_key("log_marginal", proposal, key)] = sum(
                    self.estimates[key]
                ) / len(self.estimates[key])
            if key in self.sizes:
                metrics[_proposal_key("ess", proposal, key)] = sum(self.sizes[key]) / len(
                    self.sizes[key]
                )
            metrics[_proposal_key("gap_rows", proposal, key)] = float(self.gaps[key])
            # Absent, never 0, when no row of THIS proposal is finite (or the bytes are unusable).
            if self.ratio_available and self.denominator.get(key, 0) > 0:
                metrics[_proposal_key("log_marginal_per_byte_ratio_of_sums", proposal, key)] = (
                    self.numerator[key] / self.denominator[key]
                )


def _finite_list(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in cast(list[object], value):
        number = _finite(item)
        if number is None:
            return None
        out.append(number)
    return out


def project_evaluation(event: dict[str, object]) -> dict[str, float] | None:
    """The scalar metrics of one sealed evaluation report (v1 or v2); ``None`` when malformed.

    This is the one projection both the MLflow/JSONL mirror and the worker's ``evaluation.end``
    telemetry read, so no second formula exists in a reader member (FOR-1704, C11/C12). A key is
    emitted if and only if its source fields are present in the record, so the schema version is
    not the switch and nothing is invented: a v1 record carries no prior or mixture block and so
    emits no such key. A MALFORMED v2 block (a proposal, diagnostics or latent-usage field that is
    present but unusable) returns ``None`` for the whole record, never the q keys alone (A8c).
    """
    examples = event.get("examples")
    if not isinstance(examples, list):
        return None
    tallies = {name: _Tally() for name in PROPOSALS}
    seen: dict[str, int] = dict.fromkeys(PROPOSALS, 0)
    prior_radius: list[float] = []
    q_radius: list[float] = []
    cosines: list[float] = []
    radius_seen = [0, 0]
    sd_values: dict[str, dict[str, list[float]]] = {name: {} for name in PROPOSALS}
    sd_seen = 0
    usage_gain = 0.0
    usage_bytes = 0
    usage_seen = 0
    count = len(cast(list[object], examples))
    for example in cast(list[object], examples):
        if not isinstance(example, dict):
            return None
        ex = cast(dict[str, object], example)
        q_rows = _rows(ex.get("by_sample_count"))
        if q_rows is None:
            return None
        tallies["q"].add(q_rows, ex.get("source_bytes"))
        seen["q"] += 1
        proposals = ex.get("proposals")
        if proposals is not None:
            if not isinstance(proposals, dict):
                return None
            blocks = cast(dict[str, object], proposals)
            if any(name not in ("prior", "mixture") for name in blocks):
                return None
            for name, block in blocks.items():
                if not isinstance(block, dict):
                    return None
                block_rows = _rows(cast(dict[str, object], block).get("by_sample_count"))
                if block_rows is None or cast(dict[str, object], block).get("proposal") != name:
                    return None
                tallies[name].add(block_rows, ex.get("source_bytes"))
                seen[name] += 1
        diagnostics = ex.get("diagnostics")
        if diagnostics is not None:
            if not isinstance(diagnostics, dict):
                return None
            d = cast(dict[str, object], diagnostics)
            for index, field, bucket in (
                (0, "prior_log_radius_mean", prior_radius),
                (1, "q_log_radius_mean", q_radius),
            ):
                if d.get(field) is not None:
                    values = _finite_list(d.get(field))
                    if values is None:
                        return None
                    bucket.extend(values)
                    radius_seen[index] += 1
            if d.get("prior_q_cosine") is not None:
                values = _finite_list(d.get("prior_q_cosine"))
                if values is None:
                    return None
                cosines.extend(values)
            spread = d.get("log_weight_sd")
            if spread is not None:
                if not isinstance(spread, dict):
                    return None
                for name, by_k in cast(dict[str, object], spread).items():
                    if name not in PROPOSALS or not isinstance(by_k, dict):
                        return None
                    for k, sds in cast(dict[object, object], by_k).items():
                        if not isinstance(sds, list):
                            return None
                        bucket_k = sd_values[name].setdefault(str(k), [])
                        for item in cast(list[object], sds):
                            if item is None:
                                continue
                            number = _finite(item)
                            if number is None:
                                return None
                            bucket_k.append(number)
                sd_seen += 1
        usage = ex.get("latent_usage")
        if usage is not None:
            if not isinstance(usage, dict):
                return None
            u = cast(dict[str, object], usage)
            q_lp = _finite_list(u.get("q_decoder_log_prob"))
            null_lp = _finite_list(u.get("null_decoder_log_prob"))
            counts = _bytes(ex.get("source_bytes"), len(q_lp) if q_lp is not None else -1)
            if q_lp is None or null_lp is None or len(null_lp) != len(q_lp) or counts is None:
                return None
            # signed, never clamped: a negative sum is the measurement
            usage_gain += sum(q - n for q, n in zip(q_lp, null_lp, strict=True))
            usage_bytes += sum(counts)
            usage_seen += 1
    # a block present on some examples and not others is a malformed record, not a partial one
    for name in ("prior", "mixture"):
        if seen[name] not in (0, count):
            return None
    for flag in (*radius_seen, sd_seen, usage_seen):
        if flag not in (0, count):
            return None
    metrics: dict[str, float] = {}
    for name in PROPOSALS:
        if seen[name]:
            tallies[name].emit(name, metrics)
    if sd_seen:
        for name in PROPOSALS:
            for k in sorted(sd_values[name], key=lambda key: (len(key), key)):
                if sd_values[name][k]:
                    metrics[_proposal_key("log_weight_sd", name, k)] = sum(
                        sd_values[name][k]
                    ) / len(sd_values[name][k])
    if radius_seen[0] and prior_radius:
        metrics["eval/prior_log_radius_mean"] = sum(prior_radius) / len(prior_radius)
    if radius_seen[1] and q_radius:
        metrics["eval/q_log_radius_mean"] = sum(q_radius) / len(q_radius)
    if cosines:
        metrics["eval/prior_q_cosine_mean"] = sum(cosines) / len(cosines)
    if usage_seen:
        operand = event.get("latent_operand")
        if operand not in LATENT_OPERANDS or usage_bytes <= 0:
            return None
        metrics[f"eval/latent_usage_{operand}_nats_per_byte"] = usage_gain / usage_bytes
    for field in ("wall_seconds", "cuda_seconds"):
        value = _finite(event.get(field))
        if value is not None:
            metrics[f"eval/{field}"] = value
    return metrics
