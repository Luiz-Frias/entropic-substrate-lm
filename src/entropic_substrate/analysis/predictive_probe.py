"""Detached, reproducible observable-byte probes for report-only score gain.

Callers supply causal states from a pinned checkpoint and canonicalized bytes.
This module never invokes or updates that model. Sequence boundaries must be
preserved: a StateSequence is one contiguous source span, never packed documents.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor

from entropic_substrate.analysis.predictive_score_gain import (
    PredictiveScore,
    ScoreBinding,
    ScoreGain,
    ScoreGainSummary,
    pair_score_gain,
    summarize_score_gains,
)


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    lags: tuple[int, ...] = (2, 4, 16, 64)
    train_positions: int = 65536
    eval_positions: int = 8192
    updates: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.lags or len(set(self.lags)) != len(self.lags):
            raise ValueError("probe lags must be nonempty and unique")
        if any(type(lag) is not int or lag < 2 for lag in self.lags):
            raise ValueError("probe lags must keep target future to both states")
        for value in (self.train_positions, self.eval_positions, self.updates, self.batch_size):
            if type(value) is not int or value < 1:
                raise ValueError("probe position and fitting budgets must be positive integers")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("probe seed must be a nonnegative 63-bit integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("probe learning rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("probe weight decay must be nonnegative and finite")

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted({horizon for lag in self.lags for horizon in (lag - 1, lag)}))


@dataclass(frozen=True, slots=True)
class StateSequence:
    states: Tensor
    canonical_bytes: Tensor
    checkpoint_id: str
    codec_id: str
    split_id: str
    source_id: str
    source_offset: int
    split: str
    window_starts: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.states.ndim != 2 or min(self.states.shape) < 1:
            raise ValueError("probe states must be nonempty [position, width]")
        if not self.states.is_floating_point() or not torch.isfinite(self.states).all():
            raise ValueError("probe states must be finite floating point")
        if self.canonical_bytes.shape != (self.states.shape[0],):
            raise ValueError("canonical bytes must be aligned with state positions")
        if self.canonical_bytes.dtype != torch.long:
            raise ValueError("canonical bytes must be int64")
        if torch.any((self.canonical_bytes < 0) | (self.canonical_bytes > 255)):
            raise ValueError("canonical byte outside [0,255]")
        if self.split not in ("train", "heldout"):
            raise ValueError("probe split must be train or heldout")
        if type(self.source_offset) is not int or self.source_offset < 0:
            raise ValueError("probe source offset must be nonnegative")
        if (
            not self.window_starts
            or self.window_starts[0] != 0
            or any(
                type(start) is not int or not 0 <= start < self.states.shape[0]
                for start in self.window_starts
            )
            or tuple(sorted(set(self.window_starts))) != self.window_starts
        ):
            raise ValueError(
                "probe window starts must be unique increasing offsets starting at zero"
            )
        for value in (self.checkpoint_id, self.codec_id, self.split_id, self.source_id):
            if type(value) is not str or not value.strip():
                raise ValueError("probe source identities must be nonblank")


@dataclass(frozen=True, slots=True)
class ProbeManifest:
    probe_id: str
    checkpoint_id: str
    codec_id: str
    training_source_id: str
    training_split_id: str
    training_span: tuple[int, int]
    training_positions: tuple[int, ...]
    config: ProbeConfig
    state_width: int
    fit_seconds: float
    training_content_digest: str
    training_window_starts: tuple[int, ...]
    schema_version: str = "predictive-score-probe-v2"


def _digest(metadata: object, *tensors: Tensor) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FrozenPredictiveProbe:
    """Frozen tensors with cloned export; verify identity before every evaluation."""

    def __init__(self, manifest: ProbeManifest, weight: Tensor, bias: Tensor) -> None:
        self.manifest = manifest
        self._weight = weight.detach().cpu().float().clone()
        self._bias = bias.detach().cpu().float().clone()
        self.verify()

    def verify(self) -> None:
        metadata = asdict(self.manifest)
        del metadata["probe_id"]
        del metadata["fit_seconds"]
        if _digest(metadata, self._weight, self._bias) != self.manifest.probe_id:
            raise ValueError("probe artifact identity mismatch")

    def state_dict(self) -> dict[str, Tensor]:
        """Export exact CPU FP32 weights for the caller's authoritative capture."""
        return {"weight": self._weight.clone(), "bias": self._bias.clone()}

    def log_probabilities(self, states: Tensor, horizon: int) -> Tensor:
        index = self.manifest.config.horizons.index(horizon)
        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            return F.log_softmax(
                F.linear(states.detach().cpu().float(), self._weight[index], self._bias[index]),
                dim=-1,
            )


def _positions(sequence: StateSequence, config: ProbeConfig, count: int, seed: int) -> Tensor:
    ends = (*sequence.window_starts[1:], sequence.states.shape[0])
    candidates = torch.cat(
        [
            torch.arange(start, max(start, end - max(config.lags)))
            for start, end in zip(sequence.window_starts, ends, strict=True)
        ]
    )
    available = len(candidates)
    if available < count:
        raise ValueError(f"insufficient causal probe positions: need {count}, have {available}")
    generator = torch.Generator().manual_seed(seed)
    return candidates[torch.randperm(available, generator=generator)[:count]].sort().values


def fit_predictive_probe(
    train: StateSequence, config: ProbeConfig | None = None
) -> FrozenPredictiveProbe:
    """Fit CPU FP32 linear heads on training states only, with an isolated RNG."""
    train.__post_init__()
    config = ProbeConfig() if config is None else config
    if train.split != "train":
        raise ValueError("probe fitting requires the training split")
    start = time.perf_counter()
    positions = _positions(train, config, config.train_positions, config.seed)
    states = train.states.detach().cpu().float()
    targets = train.canonical_bytes.detach().cpu()
    width = states.shape[1]
    weight = torch.zeros(len(config.horizons), 256, width, requires_grad=True)
    bias = torch.zeros(len(config.horizons), 256, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [weight, bias], lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    with torch.enable_grad(), torch.autocast(device_type="cpu", enabled=False):
        for _ in range(config.updates):
            indices = positions[
                torch.randint(len(positions), (config.batch_size,), generator=generator)
            ]
            logits = torch.einsum("bd,kvd->kbv", states[indices], weight) + bias[:, None, :]
            labels = torch.stack([targets[indices + horizon] for horizon in config.horizons])
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            if not torch.isfinite(loss):
                raise ValueError("nonfinite probe fitting loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                tensor.grad is None or not torch.isfinite(tensor.grad).all()
                for tensor in (weight, bias)
            ):
                raise ValueError("missing or nonfinite probe fitting gradient")
            optimizer.step()
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("nonfinite fitted probe")
    manifest = ProbeManifest(
        probe_id="pending",
        checkpoint_id=train.checkpoint_id,
        codec_id=train.codec_id,
        training_source_id=train.source_id,
        training_split_id=train.split_id,
        training_span=(train.source_offset, train.source_offset + states.shape[0]),
        training_positions=tuple(int(value) + train.source_offset for value in positions.tolist()),
        config=config,
        state_width=width,
        fit_seconds=time.perf_counter() - start,
        training_content_digest=_digest({}, states, targets),
        training_window_starts=train.window_starts,
    )
    metadata = asdict(manifest)
    del metadata["probe_id"]
    del metadata["fit_seconds"]
    manifest = replace(manifest, probe_id=_digest(metadata, weight, bias))
    return FrozenPredictiveProbe(manifest, weight, bias)


@dataclass(frozen=True, slots=True)
class ProbeReport:
    primary: tuple[ScoreGain, ...]
    shuffled: tuple[ScoreGain, ...]
    uniform: tuple[ScoreGain, ...]
    primary_summary: ScoreGainSummary
    shuffled_summary: ScoreGainSummary
    uniform_summary: ScoreGainSummary
    score_seconds: float
    evaluation_content_digest: str
    shuffled_source_positions: tuple[tuple[int, int], ...]
    evaluation_window_starts: tuple[int, ...]


def score_predictive_probe(probe: FrozenPredictiveProbe, heldout: StateSequence) -> ProbeReport:
    """Score exact future targets with actual, independently shuffled, and uniform states."""
    heldout.__post_init__()
    probe.verify()
    manifest = probe.manifest
    if heldout.split != "heldout" or heldout.split_id == manifest.training_split_id:
        raise ValueError("probe scoring requires a distinct held-out split")
    if (heldout.checkpoint_id, heldout.codec_id) != (manifest.checkpoint_id, manifest.codec_id):
        raise ValueError("probe evaluation checkpoint/codec identity mismatch")
    if heldout.states.shape[1] != manifest.state_width:
        raise ValueError("probe evaluation width mismatch")
    low, high = manifest.training_span
    if heldout.source_id == manifest.training_source_id and (
        heldout.source_offset < high and heldout.source_offset + heldout.states.shape[0] > low
    ):
        raise ValueError("probe training and held-out source spans overlap")
    start = time.perf_counter()
    config = manifest.config
    positions = _positions(heldout, config, config.eval_positions, config.seed + 1)
    states = heldout.states.detach().cpu().float()
    targets = heldout.canonical_bytes.detach().cpu()
    binding = ScoreBinding(
        heldout.checkpoint_id,
        manifest.probe_id,
        heldout.codec_id,
        heldout.split_id,
        heldout.source_id,
    )
    generator = torch.Generator().manual_seed(config.seed + 2)
    primary: list[ScoreGain] = []
    shuffled: list[ScoreGain] = []
    uniform: list[ScoreGain] = []
    shuffled_sources: list[tuple[int, int]] = []
    for lag in config.lags:
        labels = targets[positions + lag]
        before = probe.log_probabilities(states[positions], lag)
        after = probe.log_probabilities(states[positions + 1], lag - 1)
        # Separate permutations break context/target correspondence independently.
        before_permutation = torch.randperm(len(positions), generator=generator)
        after_permutation = torch.randperm(len(positions), generator=generator)
        before_shuffle = before[before_permutation]
        after_shuffle = after[after_permutation]
        for index, position in enumerate(positions.tolist()):
            target = int(labels[index])
            absolute = int(position) + heldout.source_offset
            shuffled_sources.append(
                (
                    int(positions[before_permutation[index]]) + heldout.source_offset,
                    int(positions[after_permutation[index]]) + heldout.source_offset + 1,
                )
            )
            for destination, previous, current in (
                (primary, float(before[index, target]), float(after[index, target])),
                (
                    shuffled,
                    float(before_shuffle[index, target]),
                    float(after_shuffle[index, target]),
                ),
                (uniform, -math.log(256), -math.log(256)),
            ):
                destination.append(
                    pair_score_gain(
                        PredictiveScore(binding, absolute, absolute + lag, target, lag, previous),
                        PredictiveScore(
                            binding, absolute + 1, absolute + lag, target, lag - 1, current
                        ),
                    )
                )
    return ProbeReport(
        tuple(primary),
        tuple(shuffled),
        tuple(uniform),
        summarize_score_gains(tuple(primary)),
        summarize_score_gains(tuple(shuffled)),
        summarize_score_gains(tuple(uniform)),
        time.perf_counter() - start,
        _digest({}, states, targets),
        tuple(shuffled_sources),
        heldout.window_starts,
    )
