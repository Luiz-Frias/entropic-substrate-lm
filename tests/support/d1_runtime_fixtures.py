"""Shared, deterministic fixtures for D1 runtime contract tests."""

from __future__ import annotations

import torch

from entropic_substrate.core.corpus.batching import PredictiveBatch
from entropic_substrate.core.latent.initialization import SeededD1BranchInitializer
from entropic_substrate.core.observation.contracts import CodecIdentity, ObservationAlignment
from entropic_substrate.kernel.identity.d1_primitives import (
    CausalEncoderSpec,
    D1Arm,
    EncoderInitializationIdentity,
    MatchedEncoderPairSpec,
    default_d1_encoder_initialization_identity,
)
from entropic_substrate.ladder.register.d1_family import (
    D1ExperimentFamily,
    build_d1_experiment_family,
)
from entropic_substrate.ladder.register.d1_stabilization import EMAParameters
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem


def runtime_codec() -> CodecIdentity:
    return CodecIdentity(name="runtime-test", construction_digest="c" * 64, cardinality=11)


def runtime_family(
    *,
    ema: EMAParameters | None = None,
    initialization: EncoderInitializationIdentity | None = None,
) -> D1ExperimentFamily:
    encoder = CausalEncoderSpec(
        latent_width=8,
        n_layers=1,
        n_heads=2,
        mlp_ratio=2.0,
        block_size=8,
        dropout=0.0,
        bias=False,
        rms_epsilon=1e-5,
        initialization=initialization or default_d1_encoder_initialization_identity(),
    )
    return build_d1_experiment_family(
        codec=runtime_codec(),
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
        ema=ema,
    )


def runtime_batch(*, horizon: int = 1, future_last_id: int = 7) -> PredictiveBatch:
    online = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    language = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.int64)
    context = torch.tensor(
        [[1, 2, 3, future_last_id], [4, 5, 6, future_last_id]], dtype=torch.int64
    )
    if horizon == 2:
        context = torch.cat((context, torch.full((2, 1), future_last_id, dtype=torch.int64)), dim=1)
    starts = torch.tensor([0, 10], dtype=torch.int64)
    return PredictiveBatch(
        online_observations=online,
        language_targets=language,
        target_encoder_context=context,
        source_starts=starts,
        alignments=tuple(
            ObservationAlignment.predictive(start=start, sequence_length=3, horizon=horizon)
            for start in starts.tolist()
        ),
        horizon=horizon,
    )


def runtime_system(arm: D1Arm, *, seed: int = 7) -> D1PredictiveSystem:
    return D1PredictiveSystem.from_family(
        runtime_family(), arm, initializer=SeededD1BranchInitializer(seed=seed)
    )


def fit_runtime_baseline(system: D1PredictiveSystem) -> None:
    system.fit_mean_displacement_baseline([runtime_batch(), runtime_batch(future_last_id=8)])
