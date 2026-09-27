"""Production-model score probes over verified D-ladder checkpoints.

Only checkpoint identity admission occurs here. This diagnostic confers no pilot,
confirmatory, capture, verdict, or successor authority.
"""

from __future__ import annotations

import io
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import torch

from entropic_substrate.core.probe.predictive_probe import (
    ProbeConfig,
    StateSequence,
    fit_predictive_probe,
    score_predictive_probe,
)
from entropic_substrate.historical.d_ladder.d_ladder_execution import SmokeRequest
from entropic_substrate.historical.d_ladder.d_ladder_request import ScientificRequest
from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
    snapshot_d1_artifact,
)
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.register.ladder import RungSpec
from entropic_substrate.ladder.run.construction.ladder_runtime import LadderPredictiveSystem


@dataclass(frozen=True, slots=True)
class ProbeSource:
    """One canonical-byte contiguous document/split span with a verified digest."""

    root: Path
    relative_path: str
    content_hash: ContentHash
    split: str
    split_id: str
    source_id: str
    source_offset: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "heldout"}:
            raise ValueError("probe source split must be train or heldout")
        if self.content_hash.algorithm != "sha256":
            raise ValueError("probe source digest must be sha256")
        if not self.split_id or not self.source_id:
            raise ValueError("probe source requires split and source identities")
        if type(self.source_offset) is not int or self.source_offset < 0:
            raise ValueError("probe source offset must be a nonnegative integer")

    def read(self) -> bytes:
        """Read exactly the bytes validated through a no-follow descriptor."""
        _, payload, digest = read_d1_artifact_bytes(
            self.root, self.relative_path, label="probe source"
        )
        if digest != self.content_hash.digest:
            raise ValueError("probe source digest mismatch")
        return payload


@dataclass(frozen=True, slots=True)
class CheckpointProbeCapture:
    weights: ArtifactIdentity
    report: ArtifactIdentity
    receipt: ArtifactIdentity


class ProbeStateEncoder(Protocol):
    """Observation-only adapter shared by the D1 and later-rung model paths."""

    @property
    def block_size(self) -> int: ...

    @property
    def codec_digest(self) -> str: ...

    @property
    def construction_identity(self) -> str: ...

    def encode_observations(self, canonical_ids: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class _LadderEncoder:
    system: LadderPredictiveSystem

    @property
    def block_size(self) -> int:
        return self.system.spec.base.encoder_pair.online.block_size

    @property
    def codec_digest(self) -> str:
        return self.system.codec_identity.digest

    @property
    def construction_identity(self) -> str:
        return self.system.construction_identity

    def encode_observations(self, canonical_ids: torch.Tensor) -> torch.Tensor:
        return self.system.encode_observations(canonical_ids).values


def _extract(
    model: ProbeStateEncoder,
    source: ProbeSource,
    payload: bytes,
    checkpoint_digest: str,
    config: ProbeConfig,
    count: int,
) -> StateSequence:
    window = model.block_size
    eligible = window - max(config.lags)
    if eligible <= 0:
        raise ValueError("probe lags exceed model context window")
    size = min(len(payload), math.ceil(count / eligible) * window)
    if size < 1:
        raise ValueError("probe source is empty")
    canonical = torch.tensor(list(payload[:size]), dtype=torch.long)
    starts = tuple(range(0, size, window))
    available = sum(
        max(0, min(size, start + window) - start - max(config.lags)) for start in starts
    )
    if available < count:
        raise ValueError("insufficient within-window probe positions")
    chunks: list[torch.Tensor] = []
    # G requires differentiation internally; do not wrap model extraction in no_grad.
    for start in starts:
        with torch.enable_grad(), torch.autocast(device_type="cpu", enabled=False):
            trajectory = model.encode_observations(canonical[start : start + window][None])
        chunks.append(trajectory[0].detach().cpu().float())
        del trajectory
    return StateSequence(
        torch.cat(chunks),
        canonical,
        checkpoint_digest,
        model.codec_digest,
        source.split_id,
        source.source_id,
        source.source_offset,
        source.split,
        starts,
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def run_checkpoint_probe(
    request: SmokeRequest | ScientificRequest,
    checkpoint: Path,
    training: ProbeSource,
    heldout: ProbeSource,
    output: Path,
    config: ProbeConfig | None = None,
) -> CheckpointProbeCapture:
    """Fit/score detached probes on the exact production observation-state path.

    Inputs are canonical bytes even for D5: the runtime applies its declared
    permutation internally, while probe targets remain canonical. Model forward
    target construction and objectives are never invoked during extraction.
    """
    checkpoint_root = normalize_d1_artifact_root(checkpoint.absolute().parent)
    _, _, digest = snapshot_d1_artifact(checkpoint_root, checkpoint.name, label="probe checkpoint")
    return run_verified_checkpoint_probe(
        request.spec, request.manifest, checkpoint, digest, training, heldout, output, config
    )


def run_verified_checkpoint_probe(
    spec: RungSpec,
    manifest: Mapping[str, object],
    checkpoint: Path,
    expected_sha256: str,
    training: ProbeSource,
    heldout: ProbeSource,
    output: Path,
    config: ProbeConfig | None = None,
) -> CheckpointProbeCapture:
    """Diagnostic CLI entry: exact profile, manifest and checkpoint, without relaunch authority."""
    from entropic_substrate.historical.d_ladder.d_ladder_artifacts import load_bound_ladder_model

    config = ProbeConfig() if config is None else config
    model = load_bound_ladder_model(spec, manifest, checkpoint, expected_sha256=expected_sha256)
    model.eval()
    return run_model_probe(
        _LadderEncoder(model),
        expected_sha256,
        cast(str, manifest["run_class"]),
        canonical_digest(manifest),
        training,
        heldout,
        output,
        config,
    )


def run_model_probe(
    model: ProbeStateEncoder,
    digest: str,
    run_class: str,
    execution_digest: str,
    training: ProbeSource,
    heldout: ProbeSource,
    output: Path,
    config: ProbeConfig,
) -> CheckpointProbeCapture:
    """Shared diagnostic execution after a checkpoint adapter has admitted its model."""
    if run_class not in {"smoke", "pilot", "confirmatory"}:
        raise ValueError("unsupported source checkpoint run class")
    if training.split != "train" or heldout.split != "heldout":
        raise ValueError("checkpoint probe requires training and held-out source roles")
    training_bytes, heldout_bytes = training.read(), heldout.read()
    start = time.perf_counter()
    train_states = _extract(model, training, training_bytes, digest, config, config.train_positions)
    eval_states = _extract(model, heldout, heldout_bytes, digest, config, config.eval_positions)
    extraction_seconds = time.perf_counter() - start
    probe = fit_predictive_probe(train_states, config)
    report = score_predictive_probe(probe, eval_states)
    output.mkdir(parents=True, exist_ok=True)
    normalize_d1_artifact_root(output)
    buffer = io.BytesIO()
    torch.save({"manifest": asdict(probe.manifest), "state_dict": probe.state_dict()}, buffer)
    weights = publish_d1_atomic_bytes(output, "probe.pt", buffer.getvalue(), label="probe weights")
    evidence = publish_d1_atomic_bytes(
        output, "report.json", _json_bytes(asdict(report)), label="probe report"
    )
    sources = {
        name: {
            "relative_path": source.relative_path,
            "content_hash": source.content_hash.digest,
            "source_id": source.source_id,
            "split_id": source.split_id,
            "source_offset": source.source_offset,
            "source_bytes": len(payload),
        }
        for name, source, payload in (
            ("training_source", training, training_bytes),
            ("heldout_source", heldout, heldout_bytes),
        )
    }
    receipt = {
        "schema": "d-predictive-score-probe-v1",
        "run_class": run_class,
        "execution_digest": execution_digest,
        "scientific_authority": "none",
        "diagnostic_only": True,
        "checkpoint_sha256": digest,
        "construction_identity": model.construction_identity,
        "weights": asdict(weights),
        "report": asdict(evidence),
        **sources,
        "extraction_seconds": extraction_seconds,
        "extracted_online_positions": train_states.states.shape[0] + eval_states.states.shape[0],
        "fit_seconds": probe.manifest.fit_seconds,
        "score_seconds": report.score_seconds,
        "diagnostic_device": "cpu",
        "diagnostic_dtype": "float32",
        "resource_scope": "diagnostic_only_excludes_model_training",
    }
    published = publish_d1_atomic_bytes(
        output, "receipt.json", _json_bytes(receipt), label="probe receipt"
    )
    return CheckpointProbeCapture(weights, evidence, published)
