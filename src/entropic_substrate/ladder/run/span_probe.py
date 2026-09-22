"""Evaluation-only linear byte probes conditioned on remaining source horizon.

The fixed-capacity development feature map is affine in the unnormalized state
and log1p(horizon). This deliberately limited readout is not the generator's
likelihood and its scores are not mutual information.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from entropic_substrate.__legacy.analysis.predictive_probe import ProbeConfig
from entropic_substrate.__legacy.analysis.predictive_score_gain import ScoreBinding
from entropic_substrate.core.probe.score_gain import (
    SpanGainSummary,
    SpanPredictiveScore,
    SpanScoreGain,
    summarize_span_gains,
)
from entropic_substrate.ladder.run.span_states import SpanProbePosition, SpanStateSet


def _digest(metadata: object, *tensors: Tensor) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_span_states(data: SpanStateSet) -> str:
    """Revalidate mutable tensors and return the full selected-state content identity."""
    count = len(data.positions)
    if not count or data.split not in ("train", "validation", "test"):
        raise ValueError("probe states require a nonempty declared split")
    if (
        any(
            type(v) is not str or not v.strip()
            for v in (data.checkpoint_id, data.codec_id, data.source_digest)
        )
        or data.selection_policy != "stratified-reservoir-by-lag-v1"
    ):
        raise ValueError("probe state identities or selection policy invalid")
    if type(data.selection_seed) is not int or not 0 <= data.selection_seed < 2**63:
        raise ValueError("probe selection seed invalid")
    for states in (data.before, data.after):
        if (
            states.ndim != 2
            or states.shape[0] != count
            or states.shape[1] < 1
            or states.dtype != torch.float32
            or states.device.type != "cpu"
            or states.requires_grad
            or not torch.isfinite(states).all()
        ):
            raise ValueError("probe states must be detached finite CPU FP32 matrices")
    if data.before.shape != data.after.shape:
        raise ValueError("before and after state shapes disagree")
    for values in (data.before_horizons, data.after_horizons, data.labels):
        if values.shape != (count,) or values.dtype != torch.long or values.device.type != "cpu":
            raise ValueError("probe labels and horizons require aligned CPU int64 vectors")
    if torch.any((data.labels < 0) | (data.labels > 255)):
        raise ValueError("probe target is not a canonical byte")
    keys = []
    for index, position in enumerate(data.positions):
        if (
            type(position) is not SpanProbePosition
            or type(position.document_id) is not str
            or not position.document_id.strip()
        ):
            raise ValueError("invalid probe document position")
        coordinates = (
            position.window_start,
            position.event_index,
            position.span_start,
            position.span_end,
            position.target_position,
            position.lag,
        )
        if any(type(v) is not int or v < 0 for v in coordinates):
            raise ValueError("invalid probe coordinates")
        if not (
            position.window_start <= position.span_start < position.span_end
            and position.lag >= 2
            and position.target_position == position.span_end - 1 + position.lag
            and int(data.before_horizons[index])
            == position.target_position - position.span_start + 1
            and int(data.after_horizons[index]) == position.lag
        ):
            raise ValueError("probe source coordinates and horizons disagree")
        keys.append((position.document_id, position.span_start, position.target_position))
    if len(set(keys)) != count:
        raise ValueError("duplicate probe positions")
    metadata = {key: value for key, value in vars(data).items() if not isinstance(value, Tensor)}
    metadata["positions"] = [asdict(p) for p in data.positions]
    return _digest(
        metadata, data.before, data.after, data.before_horizons, data.after_horizons, data.labels
    )


@dataclass(frozen=True, slots=True)
class SpanProbeManifest:
    probe_id: str
    checkpoint_id: str
    codec_id: str
    source_digest: str
    training_content_digest: str
    training_positions: tuple[SpanProbePosition, ...]
    training_permutation: tuple[int, ...]
    config: ProbeConfig
    width: int
    control: Literal["observed", "shuffled"]
    fit_seconds: float
    feature_map: str = "state-plus-log1p-horizon-v1"
    schema: str = "span-predictive-probe-v1"


def _metadata(manifest: SpanProbeManifest) -> dict[str, object]:
    row = asdict(manifest)
    del row["probe_id"], row["fit_seconds"]
    return row


def _features(states: Tensor, horizons: Tensor) -> Tensor:
    with torch.autocast("cpu", enabled=False):
        return torch.cat((states, torch.log1p(horizons.float()).unsqueeze(-1)), dim=-1)


def _permutation(horizons: Tensor, seed: int, shuffled: bool) -> Tensor:
    result = torch.arange(len(horizons))
    if shuffled:
        generator = torch.Generator().manual_seed(seed)
        count = len(horizons) // 2
        for offset in (0, count):
            side = horizons[offset : offset + count]
            for horizon in side.unique():
                group = (side == horizon).nonzero().flatten() + offset
                result[group] = group[torch.randperm(len(group), generator=generator)]
    return result


class FrozenSpanProbe:
    """Content-bound frozen weights; exported tensors cannot mutate the artifact."""

    def __init__(self, manifest: SpanProbeManifest, weight: Tensor, bias: Tensor) -> None:
        self.manifest = manifest
        self._weight = weight.detach().cpu().clone()
        self._bias = bias.detach().cpu().clone()
        self.verify()

    def verify(self) -> None:
        self.manifest.config.__post_init__()
        if (
            self._weight.shape != (256, self.manifest.width + 1)
            or self._bias.shape != (256,)
            or any(
                t.dtype != torch.float32 or not torch.isfinite(t).all()
                for t in (self._weight, self._bias)
            )
            or self.manifest.feature_map != "state-plus-log1p-horizon-v1"
            or self.manifest.schema != "span-predictive-probe-v1"
            or self.manifest.control not in ("observed", "shuffled")
            or not math.isfinite(self.manifest.fit_seconds)
            or self.manifest.fit_seconds < 0
            or _digest(_metadata(self.manifest), self._weight, self._bias) != self.manifest.probe_id
        ):
            raise ValueError("frozen span probe identity mismatch")

    def state_dict(self) -> dict[str, Tensor]:
        return {"weight": self._weight.clone(), "bias": self._bias.clone()}

    def verify_training(self, train: SpanStateSet) -> None:
        """Join a frozen artifact to the exact selected fitting tensors and policy."""
        self.verify()
        content = validate_span_states(train)
        manifest = self.manifest
        if (
            train.split != "train"
            or content != manifest.training_content_digest
            or train.positions != manifest.training_positions
            or (train.checkpoint_id, train.codec_id, train.source_digest)
            != (manifest.checkpoint_id, manifest.codec_id, manifest.source_digest)
            or train.before.shape[1] != manifest.width
        ):
            raise ValueError("probe training binding mismatch")
        _envelope(train, manifest.config, manifest.config.train_positions)
        horizons = torch.cat((train.before_horizons, train.after_horizons))
        expected = _permutation(
            horizons, (manifest.config.seed + 1) % 2**63, manifest.control == "shuffled"
        )
        if tuple(expected.tolist()) != manifest.training_permutation:
            raise ValueError("probe training permutation differs from fitting policy")

    def log_probabilities(self, states: Tensor, horizons: Tensor) -> Tensor:
        self.verify()
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            result = F.log_softmax(
                F.linear(_features(states, horizons), self._weight, self._bias), -1
            )
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite span probe scores")
        return result


def _envelope(data: SpanStateSet, config: ProbeConfig, count: int) -> None:
    if (
        len(data.positions) != count
        or tuple(sorted({p.lag for p in data.positions})) != config.lags
    ):
        raise ValueError("probe sampling envelope mismatch")


def fit_span_probe(
    train: SpanStateSet, config: ProbeConfig, *, control: Literal["observed", "shuffled"]
) -> FrozenSpanProbe:
    """Fit detached training states only; the model is not an argument or optimizer owner."""
    started = time.perf_counter()
    content = validate_span_states(train)
    config.__post_init__()
    if train.split != "train":
        raise ValueError("probe fitting requires the training split")
    if control not in ("observed", "shuffled"):
        raise ValueError("unknown probe control")
    _envelope(train, config, config.train_positions)
    states, horizons = (
        torch.cat((train.before, train.after)),
        torch.cat((train.before_horizons, train.after_horizons)),
    )
    permutation = _permutation(horizons, (config.seed + 1) % 2**63, control == "shuffled")
    features = _features(states[permutation], horizons)
    labels = train.labels.repeat(2)
    weight = torch.zeros(256, features.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(256, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [weight, bias], lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    with torch.enable_grad(), torch.autocast("cpu", enabled=False):
        for _ in range(config.updates):
            indices = torch.randint(len(labels), (config.batch_size,), generator=generator)
            loss = F.cross_entropy(F.linear(features[indices], weight, bias), labels[indices])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite probe fitting loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in (weight, bias)):
                raise ValueError("missing or nonfinite probe gradient")
            optimizer.step()
    manifest = SpanProbeManifest(
        "pending",
        train.checkpoint_id,
        train.codec_id,
        train.source_digest,
        content,
        train.positions,
        tuple(permutation.tolist()),
        config,
        train.before.shape[1],
        control,
        time.perf_counter() - started,
    )
    manifest = replace(manifest, probe_id=_digest(_metadata(manifest), weight, bias))
    return FrozenSpanProbe(manifest, weight, bias)


@dataclass(frozen=True, slots=True)
class SpanProbeReport:
    pairs: tuple[SpanScoreGain, ...]
    uniform: tuple[SpanScoreGain, ...]
    summary: SpanGainSummary
    uniform_summary: SpanGainSummary
    before_accuracy: float
    after_accuracy: float
    evaluation_content_digest: str
    evaluation_positions: tuple[SpanProbePosition, ...]
    state_permutation: tuple[int, ...]
    score_seconds: float


def score_span_probe(probe: FrozenSpanProbe, heldout: SpanStateSet) -> SpanProbeReport:
    started = time.perf_counter()
    content = validate_span_states(heldout)
    probe.verify()
    manifest = probe.manifest
    if heldout.split not in ("validation", "test"):
        raise ValueError("probe scoring requires held-out states")
    if (heldout.checkpoint_id, heldout.codec_id, heldout.source_digest) != (
        manifest.checkpoint_id,
        manifest.codec_id,
        manifest.source_digest,
    ):
        raise ValueError("probe scoring binding mismatch")
    if heldout.before.shape[1] != manifest.width:
        raise ValueError("probe scoring width mismatch")
    if {p.document_id for p in heldout.positions} & {
        p.document_id for p in manifest.training_positions
    }:
        raise ValueError("probe training and held-out documents overlap")
    _envelope(heldout, manifest.config, manifest.config.eval_positions)
    states, horizons = (
        torch.cat((heldout.before, heldout.after)),
        torch.cat((heldout.before_horizons, heldout.after_horizons)),
    )
    permutation = _permutation(
        horizons, (manifest.config.seed + 2) % 2**63, manifest.control == "shuffled"
    )
    logits = probe.log_probabilities(states[permutation], horizons)
    count = len(heldout.positions)
    selected = logits.gather(1, heldout.labels.repeat(2).unsqueeze(-1)).squeeze(-1)
    pairs, uniform = [], []
    for index, position in enumerate(heldout.positions):
        for destination, previous, current, identity in (
            (pairs, float(selected[index]), float(selected[index + count]), manifest.probe_id),
            (uniform, -math.log(256), -math.log(256), "uniform-canonical-byte-v1"),
        ):
            binding = ScoreBinding(
                heldout.checkpoint_id,
                identity,
                heldout.codec_id,
                f"{heldout.source_digest}:{heldout.split}",
                position.document_id,
            )
            common = (position.target_position, int(heldout.labels[index]))
            destination.append(
                SpanScoreGain(
                    SpanPredictiveScore(
                        binding,
                        position.event_index,
                        position.span_start - 1,
                        *common,
                        int(heldout.before_horizons[index]),
                        previous,
                    ),
                    SpanPredictiveScore(
                        binding,
                        position.event_index + 1,
                        position.span_end - 1,
                        *common,
                        int(heldout.after_horizons[index]),
                        current,
                    ),
                    position.span_start,
                    position.span_end,
                )
            )
    accuracy = (logits.argmax(-1) == heldout.labels.repeat(2)).float()
    return SpanProbeReport(
        tuple(pairs),
        tuple(uniform),
        summarize_span_gains(tuple(pairs)),
        summarize_span_gains(tuple(uniform)),
        float(accuracy[:count].mean()),
        float(accuracy[count:].mean()),
        content,
        heldout.positions,
        tuple(permutation.tolist()),
        time.perf_counter() - started,
    )
