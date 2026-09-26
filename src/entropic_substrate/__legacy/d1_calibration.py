"""Dedicated pre-optimization calibration for D1's frozen mean comparator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from entropic_substrate.data.batching import PredictiveBatch, SequenceSampler, TokenSplit
from entropic_substrate.experiments.d1_primitives import canonical_digest, require_sha256
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

__all__ = ["D1BaselineCalibrationAttestation", "D1BaselineCalibrator"]


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _split_digest(split: TokenSplit) -> str:
    values = np.asarray(split.tokens, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class D1BaselineCalibrationAttestation:
    """Identity and accounted work of one successful calibration transition."""

    protocol_digest: str
    spec_digest: str
    split_digest: str
    source_order_digest: str
    mean_parameter_digest: str
    codec_digest: str
    initialization_seed: int
    num_batches: int
    online_positions: int
    source_span_positions: int
    successful_step_at_fit: int = 0
    schema_version: str = "d1-baseline-calibration-v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("protocol_digest", self.protocol_digest),
            ("spec_digest", self.spec_digest),
            ("split_digest", self.split_digest),
            ("source_order_digest", self.source_order_digest),
            ("mean_parameter_digest", self.mean_parameter_digest),
            ("codec_digest", self.codec_digest),
        ):
            require_sha256(value, label=label)
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            msg = "calibration initialization_seed must be nonnegative"
            raise ValueError(msg)
        for label, value in (
            ("num_batches", self.num_batches),
            ("online_positions", self.online_positions),
            ("source_span_positions", self.source_span_positions),
        ):
            if type(value) is not int or value < 1:
                msg = f"calibration {label} must be a positive integer"
                raise ValueError(msg)
        if self.successful_step_at_fit != 0:
            msg = "D1 calibration attestation must be fitted at step zero"
            raise ValueError(msg)
        if self.schema_version != "d1-baseline-calibration-v1":
            msg = "unsupported D1 baseline calibration schema"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "codec_digest": self.codec_digest,
            "initialization_seed": self.initialization_seed,
            "mean_parameter_digest": self.mean_parameter_digest,
            "num_batches": self.num_batches,
            "online_positions": self.online_positions,
            "protocol_digest": self.protocol_digest,
            "schema_version": self.schema_version,
            "source_order_digest": self.source_order_digest,
            "source_span_positions": self.source_span_positions,
            "spec_digest": self.spec_digest,
            "split_digest": self.split_digest,
            "successful_step_at_fit": self.successful_step_at_fit,
        }


class D1BaselineCalibrator:
    """Fit from a private deterministic train-grid sampler without consuming train RNG."""

    def calibrate(
        self,
        *,
        system: D1PredictiveSystem,
        split: TokenSplit,
        successful_step: int,
    ) -> D1BaselineCalibrationAttestation:
        if type(system) is not D1PredictiveSystem:
            msg = "D1 baseline calibrator requires a D1PredictiveSystem"
            raise TypeError(msg)
        if type(split) is not TokenSplit:
            msg = "D1 baseline calibrator requires a TokenSplit"
            raise TypeError(msg)
        if type(successful_step) is not int or successful_step != 0:
            msg = "D1 baseline calibration must occur at successful step zero"
            raise ValueError(msg)
        protocol = system.draft.execution_protocol
        spec = protocol.baseline
        batching = protocol.batching
        sampler = SequenceSampler(
            split,
            batch_size=batching.batch_size,
            seq_len=batching.sequence_length,
            train=False,
            seed=0,
            device=next(system.parameters()).device,
        )
        order_digest = hashlib.sha256()
        observed_batches = 0

        def batches() -> tuple[PredictiveBatch, ...]:
            nonlocal observed_batches
            values: list[PredictiveBatch] = []
            for _ in range(spec.num_batches):
                batch = sampler.predictive_batch(batching.horizon)
                order_digest.update(batch.source_starts.numpy().astype("<i8", copy=False).tobytes())
                observed_batches += 1
                values.append(batch)
            return tuple(values)

        system.fit_mean_displacement_baseline(batches())
        if observed_batches != spec.num_batches:
            msg = "D1 baseline calibration did not consume its exact batch count"
            raise RuntimeError(msg)
        return D1BaselineCalibrationAttestation(
            protocol_digest=protocol.protocol_digest,
            spec_digest=canonical_digest(spec.manifest),
            split_digest=_split_digest(split),
            source_order_digest=order_digest.hexdigest(),
            mean_parameter_digest=_tensor_digest(system.mean_displacement_baseline.mean),
            codec_digest=system.draft.codec.construction_digest,
            initialization_seed=system.initialization_attestation.seed,
            num_batches=observed_batches,
            online_positions=(observed_batches * batching.batch_size * batching.sequence_length),
            source_span_positions=(
                observed_batches
                * batching.batch_size
                * (batching.sequence_length + batching.horizon)
            ),
        )
