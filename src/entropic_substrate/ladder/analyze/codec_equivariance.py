"""Measured D5 paired-codec diagnostics; scientific thresholds live in registration.

Snapshots contain canonicalized logits and matching held-out source positions.
The bounded state sample is declared by the caller, never selected for agreement.
Exact coupled controls and independently trained pairs retain distinct labels.
"""

from __future__ import annotations

import math
import string
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import torch

from entropic_substrate.core.observation.contracts import CodecIdentity
from entropic_substrate.ladder.analyze.embedding_geometry import procrustes_align
from entropic_substrate.ladder.analyze.field_head_audit import realized_spectrum


class CodecComparisonMode(StrEnum):
    COUPLED_CONTROL = "coupled_control"
    INDEPENDENT_TRAINING = "independent_training"


@dataclass(frozen=True, slots=True)
class DiagnosticGap:
    """A declared unavailable measurement, never represented by numeric zero."""

    reason: str
    affected_positions: int


@dataclass(frozen=True, slots=True)
class ScalePolicyObservation:
    """Sampled causal closing events; duration is zero at every nonclosing position."""

    boundaries: tuple[bool, ...]
    durations: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.boundaries or len(self.boundaries) != len(self.durations):
            raise ValueError("scale boundaries and durations must be nonempty and aligned")
        for boundary, duration in zip(self.boundaries, self.durations, strict=True):
            if type(boundary) is not bool or type(duration) is not int:
                raise TypeError("scale boundaries require bool and durations require int")
            if duration < 0 or boundary != (duration > 0):
                raise ValueError("duration must be positive exactly at a completed boundary")


def _digest(value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in string.hexdigits for c in value):
        raise ValueError("snapshot digest must be hexadecimal SHA-256")


@dataclass(frozen=True, slots=True)
class CodecSnapshot:
    """A checkpoint's actual outputs on a declared canonical source sample.

    ``sample_digest`` binds complete canonical input contexts, sampling recipe,
    split and source artifact. ``source_positions`` identify the state positions,
    and ``canonical_targets`` their next-byte labels. All outputs must already
    use canonical byte label order. State tensors are copied and detached here.
    """

    seed: int
    sample_digest: str
    checkpoint_digest: str
    codec_identity: CodecIdentity
    context_starts: tuple[int, ...]
    source_positions: tuple[int, ...]
    canonical_targets: tuple[int, ...]
    canonical_support: tuple[int, ...]
    canonical_logits: torch.Tensor
    states: torch.Tensor
    scale_policy: ScalePolicyObservation | None

    def __post_init__(self) -> None:
        _digest(self.sample_digest)
        _digest(self.checkpoint_digest)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("snapshot seed must be a nonnegative integer")
        if type(self.codec_identity) is not CodecIdentity or self.codec_identity.cardinality != 256:
            raise ValueError("snapshot codec must have 256 boundary labels")
        if self.canonical_support != tuple(range(256)):
            raise ValueError("snapshot target support must be canonical bytes in order")
        if (
            self.states.ndim != 2
            or not 2 <= self.states.shape[0] <= 256
            or self.states.shape[1] < 1
        ):
            raise ValueError("state sample requires 2 through 256 positions and positive width")
        count = self.states.shape[0]
        if self.canonical_logits.shape != (count, 256):
            raise ValueError("snapshot requires one 256-label readout per sampled state")
        for value in (self.states, self.canonical_logits):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError("snapshot outputs must be finite floating tensors")
        if any(
            len(x) != count
            for x in (self.context_starts, self.source_positions, self.canonical_targets)
        ):
            raise ValueError("snapshot source positions, contexts, and targets must align")
        for start, position, target in zip(
            self.context_starts, self.source_positions, self.canonical_targets, strict=True
        ):
            if any(type(x) is not int for x in (start, position, target)):
                raise TypeError("snapshot source positions and canonical targets must be integers")
            if start < 0 or position < start or not 0 <= target < 256:
                raise ValueError("snapshot source offsets or canonical targets are invalid")
        if len(set(zip(self.context_starts, self.source_positions, strict=True))) != count:
            raise ValueError("snapshot sample positions must not duplicate the same context")
        if self.scale_policy is not None and len(self.scale_policy.boundaries) != count:
            raise ValueError("scale policy sample must match state positions")
        object.__setattr__(self, "states", self.states.detach().double().cpu().clone())
        object.__setattr__(
            self, "canonical_logits", self.canonical_logits.detach().double().cpu().clone()
        )


@dataclass(frozen=True, slots=True)
class CodecEquivarianceReport:
    """Per-seed measured diagnostics; no equivalence claim is generated here."""

    seed: int
    mode: CodecComparisonMode
    sample_digest: str
    left_checkpoint: str
    right_checkpoint: str
    left_codec: CodecIdentity
    right_codec: CodecIdentity
    metrics: Mapping[str, float]
    gaps: Mapping[str, DiagnosticGap]
    left_singular_values: tuple[float, ...]
    right_singular_values: tuple[float, ...]
    evidence_class: str = "DERIVED_FROM_MEASURED"


def _scale_metrics(
    left: ScalePolicyObservation | None,
    right: ScalePolicyObservation | None,
    count: int,
) -> tuple[dict[str, float], dict[str, DiagnosticGap]]:
    if left is None and right is None:
        return {}, {"scale_policy": DiagnosticGap("not_chunked", count)}
    if left is None or right is None:
        raise ValueError("paired scale policies must both exist or both be nonchunked")
    metrics = {
        "boundary_disagreement": sum(
            a != b for a, b in zip(left.boundaries, right.boundaries, strict=True)
        )
        / count,
        "boundary_rate_gap": (sum(right.boundaries) - sum(left.boundaries)) / count,
        "duration_event_mae": sum(
            abs(a - b) for a, b in zip(left.durations, right.durations, strict=True)
        )
        / count,
    }
    left_durations = [d for d in left.durations if d > 0]
    right_durations = [d for d in right.durations if d > 0]
    if not left_durations or not right_durations:
        return metrics, {
            "mean_completed_duration_gap": DiagnosticGap("no_completed_segments", count)
        }
    metrics["mean_completed_duration_gap"] = sum(right_durations) / len(right_durations) - sum(
        left_durations
    ) / len(left_durations)
    return metrics, {}


def compare_codec_pair(
    left: CodecSnapshot,
    right: CodecSnapshot,
    *,
    mode: CodecComparisonMode,
) -> CodecEquivarianceReport:
    """Compare one paired seed on matched contexts, canonical targets and widths.

    Rank/spectrum are uncentered, matching the existing realized-spectrum
    observer. Gram RMSE averages all declared sample pairs. Procrustes RMS is
    the raw orthogonal-alignment residual divided by sqrt(position * width).
    Gram is undefined if any state has zero norm; rank and raw alignment still
    remain measured. Each result is one seed observation, not N independent runs.
    """
    if type(left) is not CodecSnapshot or type(right) is not CodecSnapshot:
        raise TypeError("codec comparison requires typed snapshots")
    if type(mode) is not CodecComparisonMode:
        raise TypeError("codec comparison mode must distinguish coupling from trained pairs")
    for name in (
        "seed",
        "sample_digest",
        "context_starts",
        "source_positions",
        "canonical_targets",
        "canonical_support",
    ):
        if getattr(left, name) != getattr(right, name):
            raise ValueError(f"paired codec snapshots disagree on {name}")
    if left.states.shape != right.states.shape:
        raise ValueError("paired codec state shapes and widths must match")
    count, width = left.states.shape
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        log_a = left.canonical_logits.log_softmax(-1)
        log_b = right.canonical_logits.log_softmax(-1)
        prob_a, prob_b = log_a.exp(), log_b.exp()
        log_mix = torch.logaddexp(log_a, log_b) - math.log(2)
        labels = torch.tensor(left.canonical_targets).unsqueeze(-1)
        metrics = {
            "kl_left_to_right_nats": float((prob_a * (log_a - log_b)).sum(-1).mean()),
            "kl_right_to_left_nats": float((prob_b * (log_b - log_a)).sum(-1).mean()),
            "js_nats": float(
                0.5
                * (
                    (prob_a * (log_a - log_mix)).sum(-1) + (prob_b * (log_b - log_mix)).sum(-1)
                ).mean()
            ),
            "left_nats_per_byte": float(-log_a.gather(-1, labels).mean()),
            "right_nats_per_byte": float(-log_b.gather(-1, labels).mean()),
            "procrustes_rms": procrustes_align(left.states, right.states).residual
            / math.sqrt(count * width),
        }
        metrics["nats_per_byte_gap"] = (
            metrics["right_nats_per_byte"] - metrics["left_nats_per_byte"]
        )
        left_spectrum, right_spectrum = (
            realized_spectrum(left.states),
            realized_spectrum(right.states),
        )
        for prefix, spectrum in (("left", left_spectrum), ("right", right_spectrum)):
            metrics[f"{prefix}_state_rank"] = float(spectrum.rank)
            metrics[f"{prefix}_effective_rank"] = spectrum.effective_rank
        norms_a, norms_b = left.states.norm(dim=-1), right.states.norm(dim=-1)
        undefined = (norms_a == 0) | (norms_b == 0)
        gaps: dict[str, DiagnosticGap] = {}
        if bool(undefined.any()):
            gaps["gram"] = DiagnosticGap("undefined_zero_norm", int(undefined.sum()))
        else:
            a, b = left.states / norms_a[:, None], right.states / norms_b[:, None]
            delta = a @ a.T - b @ b.T
            metrics["gram_rmse"] = float(delta.square().mean().sqrt())
            metrics["gram_max_abs"] = float(delta.abs().max())
        scale_metrics, scale_gaps = _scale_metrics(left.scale_policy, right.scale_policy, count)
        metrics.update(scale_metrics)
        gaps.update(scale_gaps)
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("codec diagnostic calculation produced nonfinite metrics")
    return CodecEquivarianceReport(
        left.seed,
        mode,
        left.sample_digest,
        left.checkpoint_digest,
        right.checkpoint_digest,
        left.codec_identity,
        right.codec_identity,
        MappingProxyType(metrics),
        MappingProxyType(gaps),
        tuple(float(s) for s in left_spectrum.singular_values),
        tuple(float(s) for s in right_spectrum.singular_values),
    )
