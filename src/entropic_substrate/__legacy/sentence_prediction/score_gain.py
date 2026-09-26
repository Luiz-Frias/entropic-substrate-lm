"""Span-aware realized predictive score differences, never mutual information."""

import math
import statistics
from dataclasses import dataclass
from typing import ClassVar, Literal

from entropic_substrate.analysis.predictive_score_gain import ScoreBinding


@dataclass(frozen=True, slots=True)
class SpanPredictiveScore:
    binding: ScoreBinding
    event_index: int
    observed_through: int
    target_position: int
    target_byte: int
    remaining_horizon: int
    log_probability: float

    def __post_init__(self) -> None:
        if type(self.binding) is not ScoreBinding:
            raise ValueError("span score requires exact checkpoint/probe/source bindings")
        if any(
            type(v) is not int or v < 0
            for v in (self.event_index, self.target_position, self.target_byte)
        ):
            raise ValueError("span score coordinates must be nonnegative integers")
        if type(self.observed_through) is not int or self.observed_through < -1:
            raise ValueError("observed frontier permits only -1 as the document BOS sentinel")
        if (
            self.target_byte > 255
            or type(self.remaining_horizon) is not int
            or self.remaining_horizon < 1
        ):
            raise ValueError("span score requires a canonical future byte and positive horizon")
        if self.target_position - self.observed_through != self.remaining_horizon:
            raise ValueError("span score horizon does not match source coordinates")
        if (
            type(self.log_probability) not in (int, float)
            or not math.isfinite(self.log_probability)
            or self.log_probability > 0
        ):
            raise ValueError("span score log probability must be finite and nonpositive")


@dataclass(frozen=True, slots=True)
class SpanScoreGain:
    before: SpanPredictiveScore
    after: SpanPredictiveScore
    span_start: int
    span_end: int
    schema: ClassVar[str] = "span-predictive-score-gain-v1"

    def __post_init__(self) -> None:
        if (
            type(self.before) is not SpanPredictiveScore
            or type(self.after) is not SpanPredictiveScore
        ):
            raise ValueError("span gain requires two measured span scores")
        if self.before.binding != self.after.binding:
            raise ValueError("span score binding mismatch")
        if (self.before.target_position, self.before.target_byte) != (
            self.after.target_position,
            self.after.target_byte,
        ):
            raise ValueError("span scores must reference the same future observable byte")
        if (
            type(self.span_start) is not int
            or type(self.span_end) is not int
            or not 0 <= self.span_start < self.span_end
        ):
            raise ValueError("span boundaries must be a nonempty source interval")
        if self.after.event_index != self.before.event_index + 1:
            raise ValueError("span scores require adjacent sentence events")
        if (
            self.before.observed_through != self.span_start - 1
            or self.after.observed_through != self.span_end - 1
        ):
            raise ValueError("score frontiers do not delimit the completed span")
        if self.after.remaining_horizon < 2:
            raise ValueError("score-gain target must remain beyond the next source byte")

    @property
    def gain(self) -> float:
        return self.after.log_probability - self.before.log_probability


@dataclass(frozen=True, slots=True)
class SpanGainSummary:
    count: int
    mean_gain: float
    sample_std_gain: float | None
    negative_fraction: float
    before_nats_per_byte: float
    after_nats_per_byte: float
    inference_unit: Literal["nested_source_positions_not_model_replicates"] = (
        "nested_source_positions_not_model_replicates"
    )


def summarize_span_gains(records: tuple[SpanScoreGain, ...]) -> SpanGainSummary:
    if not records:
        raise ValueError("cannot summarize empty span score evidence")
    first = records[0].before.binding
    shared = (first.checkpoint_id, first.probe_id, first.codec_id, first.split_id)
    keys = []
    for item in records:
        binding = item.before.binding
        if (binding.checkpoint_id, binding.probe_id, binding.codec_id, binding.split_id) != shared:
            raise ValueError("cannot mix model, probe, codec or split bindings")
        keys.append((binding.source_id, item.span_start, item.before.target_position))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate span score positions")
    gains = [item.gain for item in records]
    return SpanGainSummary(
        len(records),
        statistics.fmean(gains),
        statistics.stdev(gains) if len(gains) > 1 else None,
        sum(value < 0 for value in gains) / len(gains),
        -statistics.fmean(item.before.log_probability for item in records),
        -statistics.fmean(item.after.log_probability for item in records),
    )
