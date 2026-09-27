"""Dedicated pre-optimization calibration for D1's frozen mean comparator.

RED-first split (ruling R-parallel; contract C17-p5): the immutable
attestation type moved to the record half, ``d1_calibration_records.py``,
now landed at ``ladder.capture.records`` (p8). This module RE-EXPORTS it
(never redefines it) for the test-file importers still on this module's
old dotted name; p13 rewrites those.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from entropic_substrate.core.corpus.batching import PredictiveBatch, SequenceSampler, TokenSplit
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.ladder.capture.records.d1_calibration_records import (
    D1BaselineCalibrationAttestation,
)
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem

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
