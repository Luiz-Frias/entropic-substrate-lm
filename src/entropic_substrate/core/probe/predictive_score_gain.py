"""Report-only realized score gain on one fixed, canonical future byte.

These quantities are neither mutual information nor inference over model seeds.
Per-example dispersion describes held-out positions within one model artifact.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol


@dataclass(frozen=True, slots=True)
class ScoreBinding:
    """Immutable identities that must agree on both sides of a score pair."""

    checkpoint_id: str
    probe_id: str
    codec_id: str
    split_id: str
    source_id: str

    def __post_init__(self) -> None:
        for value in (
            self.checkpoint_id,
            self.probe_id,
            self.codec_id,
            self.split_id,
            self.source_id,
        ):
            if type(value) is not str or not value.strip():
                raise ValueError("score binding identities must be nonblank strings")


@dataclass(frozen=True, slots=True)
class PredictiveScore:
    """Measured log probability with explicit causal source coordinates."""

    binding: ScoreBinding
    state_position: int
    target_position: int
    target_byte: int
    remaining_horizon: int
    log_probability: float

    def __post_init__(self) -> None:
        for value in (self.state_position, self.target_position, self.remaining_horizon):
            if type(value) is not int or value < 0:
                raise ValueError("score positions and horizon must be nonnegative integers")
        if self.target_position <= self.state_position:
            raise ValueError("score target must remain in the future")
        if self.target_position - self.state_position != self.remaining_horizon:
            raise ValueError("score remaining horizon does not match positions")
        if type(self.target_byte) is not int or not 0 <= self.target_byte < 256:
            raise ValueError("score target must be a canonical byte")
        if not math.isfinite(self.log_probability) or self.log_probability > 0:
            raise ValueError("score log probability must be finite and nonpositive")


@dataclass(frozen=True, slots=True)
class ScoreGain:
    before: PredictiveScore
    after: PredictiveScore

    def __post_init__(self) -> None:
        if self.before.binding != self.after.binding:
            raise ValueError("score pair binding mismatch")
        if (self.before.target_position, self.before.target_byte) != (
            self.after.target_position,
            self.after.target_byte,
        ):
            raise ValueError("score pair must score the same future observation")
        if self.after.state_position != self.before.state_position + 1:
            raise ValueError("score pair requires adjacent state positions")

    @property
    def gain(self) -> float:
        return self.after.log_probability - self.before.log_probability


@dataclass(frozen=True, slots=True)
class ScoreGainSummary:
    count: int
    mean_gain: float
    sample_std_gain: float | None
    negative_fraction: float
    before_nats_per_byte: float
    after_nats_per_byte: float


def pair_score_gain(before: PredictiveScore, after: PredictiveScore) -> ScoreGain:
    """Join adjacent states only when their scoring and target identities agree."""
    return ScoreGain(before, after)


class LogProbabilityScore(Protocol):
    """One measured log probability, read only."""

    @property
    def log_probability(self) -> float: ...


class GainRecord(Protocol):
    """A before/after score pair with its signed log-probability difference."""

    @property
    def before(self) -> LogProbabilityScore: ...

    @property
    def after(self) -> LogProbabilityScore: ...

    @property
    def gain(self) -> float: ...


class GainStatistics(NamedTuple):
    """Positional order is the shared field order of both summary dataclasses."""

    size: int
    mean_gain: float
    sample_std_gain: float | None
    negative_fraction: float
    before_nats_per_byte: float
    after_nats_per_byte: float


def gain_statistics(records: Sequence[GainRecord]) -> GainStatistics:
    """Reduce already validated, nonempty pairs; callers own every refusal."""
    gains = [item.gain for item in records]
    return GainStatistics(
        len(records),
        statistics.fmean(gains),
        statistics.stdev(gains) if len(gains) > 1 else None,
        sum(value < 0 for value in gains) / len(gains),
        -statistics.fmean(item.before.log_probability for item in records),
        -statistics.fmean(item.after.log_probability for item in records),
    )


def summarize_score_gains(records: tuple[ScoreGain, ...]) -> ScoreGainSummary:
    """Summarize one bound sample; singleton sample deviation is undefined."""
    if not records:
        raise ValueError("cannot summarize empty score-gain evidence")
    if any(item.before.binding != records[0].before.binding for item in records):
        raise ValueError("cannot mix score bindings in one summary")
    keys = [(item.before.state_position, item.before.target_position) for item in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate score-gain positions")
    return ScoreGainSummary(*gain_statistics(records))
