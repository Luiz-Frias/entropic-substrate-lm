"""End-to-end composition contracts for the D1 predictive-state tracer."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from d1_runtime_fixtures import (
    fit_runtime_baseline,
    runtime_batch,
    runtime_codec,
    runtime_family,
    runtime_system,
)
from entropic_substrate.core.latent.initialization import SeededD1BranchInitializer
from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    default_d1_encoder_initialization_identity,
)
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("arm", list(D1Arm))
def test_four_arms_build_full_capacity_and_differ_only_in_optimizer_ownership(arm: D1Arm) -> None:
    system = runtime_system(arm)
    attestation = system.parameter_attestation()

    assert system.online_branch.parameter_signature() == system.target_branch.parameter_signature()
    assert (
        attestation["trainable_param_count"] + attestation["frozen_param_count"]
        == runtime_system(D1Arm.NONE).parameter_attestation()["trainable_param_count"]
        + runtime_system(D1Arm.NONE).parameter_attestation()["frozen_param_count"]
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
    first = runtime_system(D1Arm.NONE)
    second = runtime_system(D1Arm.BOTH)

    assert first.initialization_attestation == second.initialization_attestation
    assert first.initialization_attestation.seed == 7
    assert (
        first.initialization_attestation.identity == first.draft.encoder_pair.online.initialization
    )


def test_runtime_refuses_an_initializer_that_does_not_match_the_declared_construction() -> None:
    family = runtime_family(
        initialization=replace(
            default_d1_encoder_initialization_identity(), construction_digest="f" * 64
        )
    )

    with pytest.raises(ValueError, match="does not materialize"):
        D1PredictiveSystem.from_family(
            family, D1Arm.NONE, initializer=SeededD1BranchInitializer(seed=7)
        )


def test_runtime_validates_exact_optimizer_ownership_for_stop_gradient() -> None:
    system = runtime_system(D1Arm.SG)
    safe = torch.optim.AdamW(system.optimizer_parameters(), lr=1e-3)
    system.validate_optimizer_ownership(safe)

    unsafe = torch.optim.AdamW(system.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="target encoder parameters must be absent"):
        system.validate_optimizer_ownership(unsafe)


@pytest.mark.parametrize("arm", list(D1Arm))
def test_all_four_arms_run_tiny_forward_backward_with_explicit_terms(arm: D1Arm) -> None:
    system = runtime_system(arm)
    fit_runtime_baseline(system)

    result = system(runtime_batch())
    result.objective.total.backward()

    assert result.language_logits.shape == (2, 3, runtime_codec().cardinality)
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
