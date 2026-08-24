"""End-to-end composition contracts for the D1 predictive-state tracer."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from entropic_substrate.data.batching import PredictiveBatch
from entropic_substrate.experiments.d1_family import build_d1_experiment_family
from entropic_substrate.experiments.d1_primitives import (
    CausalEncoderSpec,
    D1Arm,
    EncoderInitializationIdentity,
    MatchedEncoderPairSpec,
    default_d1_encoder_initialization_identity,
)
from entropic_substrate.observation import CodecIdentity, ObservationAlignment
from entropic_substrate.predictive_state import D1PredictiveSystem
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer

pytestmark = pytest.mark.unit


def _codec() -> CodecIdentity:
    return CodecIdentity(name="runtime-test", construction_digest="c" * 64, cardinality=11)


def _family(
    *,
    initialization: EncoderInitializationIdentity | None = None,
):
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
        codec=_codec(),
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
    )


def _batch(*, horizon: int = 1, future_last_id: int = 7) -> PredictiveBatch:
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


def _system(arm: D1Arm) -> D1PredictiveSystem:
    return D1PredictiveSystem.from_family(
        _family(), arm, initializer=SeededD1BranchInitializer(seed=7)
    )


def _fit(system: D1PredictiveSystem) -> None:
    system.fit_mean_displacement_baseline([_batch(), _batch(future_last_id=8)])


@pytest.mark.parametrize("arm", list(D1Arm))
def test_four_arms_build_full_capacity_and_differ_only_in_optimizer_ownership(arm: D1Arm) -> None:
    system = _system(arm)
    attestation = system.parameter_attestation()

    assert system.online_branch.parameter_signature() == system.target_branch.parameter_signature()
    assert (
        attestation["trainable_param_count"] + attestation["frozen_param_count"]
        == _system(D1Arm.NONE).parameter_attestation()["trainable_param_count"]
        + _system(D1Arm.NONE).parameter_attestation()["frozen_param_count"]
    )
    for (online_name, online), (target_name, target) in zip(
        system.online_branch.named_parameters(),
        system.target_branch.named_parameters(),
        strict=True,
    ):
        assert online_name == target_name and online.data_ptr() != target.data_ptr()
        torch.testing.assert_close(online, target, atol=0.0, rtol=0.0)
    if arm in (D1Arm.SG, D1Arm.BOTH):
        assert not any(parameter.requires_grad for parameter in system.target_branch.parameters())
        assert not set(system.target_branch.parameters()) & set(system.optimizer_parameters())
    else:
        assert all(parameter.requires_grad for parameter in system.target_branch.parameters())
        assert set(system.target_branch.parameters()) <= set(system.optimizer_parameters())


def test_runtime_materializes_the_declared_initialization_and_attests_the_seed() -> None:
    first = _system(D1Arm.NONE)
    second = _system(D1Arm.BOTH)

    assert first.initialization_attestation == second.initialization_attestation
    assert first.initialization_attestation.seed == 7
    assert (
        first.initialization_attestation.identity == first.draft.encoder_pair.online.initialization
    )


def test_runtime_refuses_an_initializer_that_does_not_match_the_declared_construction() -> None:
    family = _family(
        initialization=replace(
            default_d1_encoder_initialization_identity(), construction_digest="f" * 64
        )
    )

    with pytest.raises(ValueError, match="does not materialize"):
        D1PredictiveSystem.from_family(
            family, D1Arm.NONE, initializer=SeededD1BranchInitializer(seed=7)
        )


def test_runtime_validates_exact_optimizer_ownership_for_stop_gradient() -> None:
    system = _system(D1Arm.SG)
    safe = torch.optim.AdamW(system.optimizer_parameters(), lr=1e-3)
    system.validate_optimizer_ownership(safe)

    unsafe = torch.optim.AdamW(system.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="target encoder parameters must be absent"):
        system.validate_optimizer_ownership(unsafe)


@pytest.mark.parametrize("arm", list(D1Arm))
def test_all_four_arms_run_tiny_forward_backward_with_explicit_terms(arm: D1Arm) -> None:
    system = _system(arm)
    _fit(system)

    result = system(_batch())
    result.objective.total.backward()

    assert result.language_logits.shape == (2, 3, _codec().cardinality)
    assert result.target.values.shape == (2, 3, 8)
    assert result.future_law_loss.mean.shape == (2, 3, 8)
    assert result.collapse.diagnostics is not None
    assert not hasattr(result, "aux_loss")
    assert any(parameter.grad is not None for parameter in system.online_branch.parameters())
    if arm in (D1Arm.AC, D1Arm.BOTH):
        assert result.objective.variance.item() >= 0.0
        assert result.objective.covariance.item() >= 0.0
    else:
        assert result.objective.variance.item() == 0.0
        assert result.objective.covariance.item() == 0.0
