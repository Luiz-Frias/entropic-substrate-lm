"""Immutable values returned by the authoritative D1 telemetry query path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from entropic_substrate.d1_evaluation_values import (
    TARGET_PREDICTED_EXACT_MATCH_SEMANTIC,
    D1ChannelMoments,
)
from entropic_substrate.d1_training_step import D1TrainStepResult
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalIdentity,
)
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.evidence.resource_work import ResourceWorkSnapshot
from entropic_substrate.experiments.d1_execution import PAIRWISE_COSINE_SEMANTIC

__all__ = [
    "D1_REPORT_ONLY_METRICS",
    "D1EvalEvidence",
    "D1QualityEvidence",
    "D1RunEvidence",
    "D1RunStatus",
    "D1TailQuarantineEvidence",
    "D1TailQuarantineKind",
    "D1TelemetryGap",
]

D1_QUALITY_FIELDS: Final = frozenset(
    {
        "batch_count",
        "bits_per_byte",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "effective_rank",
        "future_nll",
        "language_loss",
        "mean_baseline_improvement",
        "mean_displacement_nll",
        "mean_pairwise_cosine_distance",
        "nats_per_byte",
        "numerical_rank",
        "online_positions",
        "pairwise_cosine_semantic",
        "perplexity",
        "predicted_covariance_eigenvalues",
        "predicted_effective_rank",
        "predicted_mean",
        "predicted_numerical_rank",
        "predicted_rms_l2_norm",
        "predicted_variance",
        "raw_covariance_penalty",
        "raw_variance_penalty",
        "residual_elements",
        "residual_mean",
        "residual_mean_l2_norm",
        "residual_mean_vector",
        "residual_rms",
        "residual_variance",
        "state_covariance_eigenvalues",
        "state_mean",
        "state_rms_l2_norm",
        "state_variance",
        "target_covariance_eigenvalues",
        "target_effective_rank",
        "target_mean",
        "target_numerical_rank",
        "target_predicted_exact_match_count",
        "target_predicted_exact_match_fraction",
        "target_predicted_exact_match_semantic",
        "target_rms_l2_norm",
        "target_variance",
        "weighted_covariance_penalty",
        "weighted_variance_penalty",
        "zero_baseline_improvement",
        "zero_displacement_nll",
    }
)
D1_SCALAR_METRICS: Final = D1_QUALITY_FIELDS - {
    "pairwise_cosine_semantic",
    "predicted_covariance_eigenvalues",
    "predicted_mean",
    "predicted_variance",
    "residual_mean_vector",
    "state_covariance_eigenvalues",
    "state_mean",
    "state_variance",
    "target_covariance_eigenvalues",
    "target_mean",
    "target_predicted_exact_match_semantic",
    "target_variance",
}
D1_REPORT_ONLY_METRICS: Final = frozenset(
    "state_rms_l2_norm target_effective_rank target_numerical_rank target_rms_l2_norm "
    "predicted_effective_rank predicted_numerical_rank predicted_rms_l2_norm "
    "residual_mean_l2_norm target_predicted_exact_match_count "
    "target_predicted_exact_match_fraction".split()
)


class D1TelemetryGap(ValueError):
    """Stable typed refusal; missing evidence is never represented as zero."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = MappingProxyType({"code": code, "path": path})
        super().__init__(f"D1 telemetry {code} at {path}: {detail}")


def gap(code: str, path: str, detail: str) -> D1TelemetryGap:
    return D1TelemetryGap(code=code, path=path, detail=detail)


class D1RunStatus(StrEnum):
    """Admission posture of a canonical journal prefix."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    TRAINER_ABORTED = "trainer_aborted"
    RUN_ABORTED = "run_aborted"


class D1TailQuarantineKind(StrEnum):
    """Whether a resume retained no tail or persisted a discarded one."""

    NO_TAIL = "no_tail"
    QUARANTINED_TAIL = "quarantined_tail"


@dataclass(frozen=True, slots=True)
class D1TailQuarantineEvidence:
    """Immutable identity already verified against disk by ``read_journal``."""

    kind: D1TailQuarantineKind
    manifest: Mapping[str, FrozenJson]
    artifact_file_name: str | None
    artifact_sha256: str | None
    artifact_byte_count: int | None
    retained_prefix_sha256: str | None
    retained_prefix_byte_count: int | None
    tail_event_count: int
    first_tail_event_sequence: int | None
    last_tail_event_sequence: int | None

    def __post_init__(self) -> None:
        if type(self.kind) is not D1TailQuarantineKind:
            raise TypeError("tail quarantine kind must be D1TailQuarantineKind")
        if (
            not isinstance(cast(object, self.manifest), Mapping)
            or self.manifest.get("kind") != self.kind.value
        ):
            raise ValueError("tail quarantine manifest and kind differ")
        optional = (
            self.artifact_file_name,
            self.artifact_sha256,
            self.artifact_byte_count,
            self.retained_prefix_sha256,
            self.retained_prefix_byte_count,
            self.first_tail_event_sequence,
            self.last_tail_event_sequence,
        )
        if self.kind is D1TailQuarantineKind.NO_TAIL:
            if any(value is not None for value in optional) or self.tail_event_count != 0:
                raise ValueError("no-tail evidence cannot claim a quarantine artifact")
        elif any(value is None for value in optional) or self.tail_event_count < 1:
            raise ValueError("quarantined-tail evidence requires complete artifact identity")

    @classmethod
    def no_tail(cls, manifest: Mapping[str, FrozenJson]) -> D1TailQuarantineEvidence:
        return cls(
            D1TailQuarantineKind.NO_TAIL,
            manifest,
            None,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
        )

    @classmethod
    def quarantined(
        cls,
        manifest: Mapping[str, FrozenJson],
        *,
        artifact_file_name: str,
        artifact_sha256: str,
        artifact_byte_count: int,
        retained_prefix_sha256: str,
        retained_prefix_byte_count: int,
        tail_event_count: int,
        first_tail_event_sequence: int,
        last_tail_event_sequence: int,
    ) -> D1TailQuarantineEvidence:
        return cls(
            D1TailQuarantineKind.QUARANTINED_TAIL,
            manifest,
            artifact_file_name,
            artifact_sha256,
            artifact_byte_count,
            retained_prefix_sha256,
            retained_prefix_byte_count,
            tail_event_count,
            first_tail_event_sequence,
            last_tail_event_sequence,
        )


def _finite(value: float, field: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise gap("invalid_metric", field, "must be a finite float")


def _nonnegative(value: float, field: str) -> None:
    _finite(value, field)
    if value < 0.0:
        raise gap("metric_domain", field, "must be nonnegative")


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)


@dataclass(frozen=True, slots=True)
class D1QualityEvidence:
    """The exact registered scalar/vector payload from one D1 evaluation."""

    language_loss: float
    nats_per_byte: float
    bits_per_byte: float
    perplexity: float
    future_nll: float
    zero_displacement_nll: float
    mean_displacement_nll: float
    zero_baseline_improvement: float
    mean_baseline_improvement: float
    residual_mean: float
    residual_rms: float
    residual_variance: float
    coverage_50: float
    coverage_80: float
    coverage_95: float
    state_moments: D1ChannelMoments
    target_moments: D1ChannelMoments
    predicted_moments: D1ChannelMoments
    target_predicted_exact_match_count: int
    mean_pairwise_cosine_distance: float
    pairwise_cosine_semantic: str
    raw_variance_penalty: float
    weighted_variance_penalty: float
    raw_covariance_penalty: float
    weighted_covariance_penalty: float
    batch_count: int
    online_positions: int
    residual_elements: int

    def __post_init__(self) -> None:
        channels = (self.state_moments, self.target_moments, self.predicted_moments)
        if any(type(channel) is not D1ChannelMoments for channel in channels):
            raise gap("invalid_channel", "channels", "must be D1ChannelMoments")
        for field in (
            "language_loss",
            "nats_per_byte",
            "bits_per_byte",
            "perplexity",
            "future_nll",
            "zero_displacement_nll",
            "mean_displacement_nll",
            "residual_rms",
            "residual_variance",
            "effective_rank",
            "raw_variance_penalty",
            "weighted_variance_penalty",
            "raw_covariance_penalty",
            "weighted_covariance_penalty",
        ):
            _nonnegative(getattr(self, field), field)
        for field in ("zero_baseline_improvement", "mean_baseline_improvement", "residual_mean"):
            _finite(getattr(self, field), field)
        coverages = (self.coverage_50, self.coverage_80, self.coverage_95)
        for field, value in zip(
            ("coverage_50", "coverage_80", "coverage_95"), coverages, strict=True
        ):
            _finite(value, field)
            if not 0.0 <= value <= 1.0:
                raise gap("metric_domain", field, "must be in [0, 1]")
        if tuple(sorted(coverages)) != coverages:
            raise gap("coverage_order", "coverage", "coverage intervals must be nested")
        _finite(self.mean_pairwise_cosine_distance, "mean_pairwise_cosine_distance")
        if not 0.0 <= self.mean_pairwise_cosine_distance <= 2.0:
            raise gap("metric_domain", "mean_pairwise_cosine_distance", "must be in [0, 2]")
        if self.pairwise_cosine_semantic != PAIRWISE_COSINE_SEMANTIC:
            raise gap("semantic_mismatch", "pairwise_cosine_semantic", "registered meaning changed")
        for field in ("batch_count", "online_positions", "residual_elements"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise gap("invalid_count", field, "must be a positive integer")
        width = self.state_moments.width
        if any(channel.width != width for channel in channels[1:]):
            raise gap("vector_width", "channels", "must share one latent width")
        if self.residual_elements != self.online_positions * width:
            raise gap("position_width", "residual_elements", "must equal positions times width")
        if (
            type(self.target_predicted_exact_match_count) is not int
            or not 0 <= self.target_predicted_exact_match_count <= self.residual_elements
        ):
            raise gap("invalid_count", "target_predicted_exact_match_count", "is out of range")
        for channel in channels:
            expected_square_norm = sum(value * value for value in channel.mean) + (
                (self.online_positions - 1) / self.online_positions
            ) * sum(channel.variance)
            if not _close(channel.rms_l2_norm * channel.rms_l2_norm, expected_square_norm):
                raise gap("metric_identity", "channel.rms_l2_norm", "differs from moments")
        if not _close(self.residual_mean, sum(self.residual_mean_vector) / width):
            raise gap("metric_identity", "residual_mean", "differs from channel means")
        if not _close(
            self.residual_variance,
            self.residual_rms * self.residual_rms - self.residual_mean * self.residual_mean,
        ):
            raise gap("metric_identity", "residual_variance", "differs from RMS and mean")
        identities = (
            (self.nats_per_byte, self.language_loss, "nats_per_byte"),
            (self.bits_per_byte, self.language_loss / math.log(2.0), "bits_per_byte"),
            (self.perplexity, math.exp(self.language_loss), "perplexity"),
            (
                self.zero_baseline_improvement,
                self.zero_displacement_nll - self.future_nll,
                "zero_baseline_improvement",
            ),
            (
                self.mean_baseline_improvement,
                self.mean_displacement_nll - self.future_nll,
                "mean_baseline_improvement",
            ),
        )
        for actual, expected, field in identities:
            if actual != expected:
                raise gap("metric_identity", field, "does not match its registered derivation")

    @property
    def latent_width(self) -> int:
        return self.state_moments.width

    @property
    def state_mean(self) -> tuple[float, ...]:
        return self.state_moments.mean

    @property
    def state_variance(self) -> tuple[float, ...]:
        return self.state_moments.variance

    @property
    def state_covariance_eigenvalues(self) -> tuple[float, ...]:
        return self.state_moments.covariance_eigenvalues

    @property
    def effective_rank(self) -> float:
        return self.state_moments.effective_rank

    @property
    def numerical_rank(self) -> int:
        return self.state_moments.numerical_rank

    @property
    def state_rms_l2_norm(self) -> float:
        return self.state_moments.rms_l2_norm

    @property
    def target_mean(self) -> tuple[float, ...]:
        return self.target_moments.mean

    @property
    def target_variance(self) -> tuple[float, ...]:
        return self.target_moments.variance

    @property
    def target_covariance_eigenvalues(self) -> tuple[float, ...]:
        return self.target_moments.covariance_eigenvalues

    @property
    def target_effective_rank(self) -> float:
        return self.target_moments.effective_rank

    @property
    def target_numerical_rank(self) -> int:
        return self.target_moments.numerical_rank

    @property
    def target_rms_l2_norm(self) -> float:
        return self.target_moments.rms_l2_norm

    @property
    def predicted_mean(self) -> tuple[float, ...]:
        return self.predicted_moments.mean

    @property
    def predicted_variance(self) -> tuple[float, ...]:
        return self.predicted_moments.variance

    @property
    def predicted_covariance_eigenvalues(self) -> tuple[float, ...]:
        return self.predicted_moments.covariance_eigenvalues

    @property
    def predicted_effective_rank(self) -> float:
        return self.predicted_moments.effective_rank

    @property
    def predicted_numerical_rank(self) -> int:
        return self.predicted_moments.numerical_rank

    @property
    def predicted_rms_l2_norm(self) -> float:
        return self.predicted_moments.rms_l2_norm

    @property
    def residual_mean_vector(self) -> tuple[float, ...]:
        return tuple(
            target - predicted
            for target, predicted in zip(self.target_mean, self.predicted_mean, strict=True)
        )

    @property
    def residual_mean_l2_norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.residual_mean_vector))

    @property
    def target_predicted_exact_match_fraction(self) -> float:
        return self.target_predicted_exact_match_count / self.residual_elements

    @property
    def target_predicted_exact_match_semantic(self) -> str:
        return TARGET_PREDICTED_EXACT_MATCH_SEMANTIC

    def scalar_metric(self, name: str) -> float:
        if name not in D1_SCALAR_METRICS:
            raise gap("unknown_metric", name, "is not a registered scalar quality metric")
        value = getattr(self, name)
        if type(value) not in (int, float):  # pragma: no cover - whitelist is pinned above
            raise AssertionError("registered scalar metric has a non-scalar representation")
        return float(value)


@dataclass(frozen=True, slots=True)
class D1EvalEvidence:
    """One quality measurement and its same-cursor cumulative work evidence."""

    step: int
    eval_index: int
    quality: D1QualityEvidence
    resource_work: ResourceWorkSnapshot


@dataclass(frozen=True, slots=True)
class D1RunEvidence:
    """One complete logical run admitted from a terminal authoritative journal."""

    path: Path
    status: D1RunStatus
    identity: JournalIdentity
    plan_digest: str
    plan_manifest: Mapping[str, FrozenJson]
    segments: tuple[RunStartProvenance, ...]
    baseline: D1JournalEvent | None
    evals: tuple[D1EvalEvidence, ...]
    train_steps: tuple[D1TrainStepResult, ...]
    data_order_segments: tuple[tuple[int, str, int], ...]
    checkpoint_receipts: tuple[tuple[int, str], ...]
    tail_quarantine_evidence: tuple[D1TailQuarantineEvidence, ...]
    terminal: D1JournalEvent | None

    @property
    def resources(self) -> tuple[ResourceWorkSnapshot, ...]:
        return tuple(item.resource_work for item in self.evals)

    @property
    def final_checkpoint_sha256(self) -> str:
        if not self.checkpoint_receipts:
            raise gap("missing_final_checkpoint", "checkpoint", "run has no receipt")
        return self.checkpoint_receipts[-1][1]

    @property
    def resume_checkpoint_sha256s(self) -> tuple[str, ...]:
        return tuple(
            segment.resume_checkpoint_sha256
            for segment in self.segments[1:]
            if segment.resume_checkpoint_sha256 is not None
        )
