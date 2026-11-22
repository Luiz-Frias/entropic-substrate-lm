"""D1 runtime refusal, causality, and checkpoint guard teeth."""

from __future__ import annotations

import pytest
import torch

from d1_runtime_fixtures import (
    fit_runtime_baseline,
    runtime_batch,
    runtime_system,
)
from entropic_substrate.core.latent.target_updates import OptimizationStepCommitted
from entropic_substrate.kernel.identity.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


def test_baseline_is_explicit_training_calibration_not_a_forward_side_effect() -> None:
    system = runtime_system(D1Arm.NONE)
    with pytest.raises(ValueError, match="fitted"):
        system(runtime_batch())

    fit_runtime_baseline(system)
    assert system.mean_displacement_baseline.is_fitted
    system(runtime_batch())
    with pytest.raises(RuntimeError, match="already fitted"):
        system.fit_mean_displacement_baseline([runtime_batch()])


def test_target_future_context_predictor_and_readout_cannot_change_online_state() -> None:
    system = runtime_system(D1Arm.AC)
    fit_runtime_baseline(system)
    first = system(runtime_batch())
    with torch.no_grad():
        system.target_branch.adapter.embedding.weight.add_(3.0)
        for parameter in system.future_law.parameters():
            parameter.add_(2.0)
        system.language_readout.projection.weight.add_(1.0)
    second = system(runtime_batch(future_last_id=9))

    torch.testing.assert_close(first.online_states.values, second.online_states.values)


def test_stop_gradient_runs_the_entire_target_branch_without_autograd() -> None:
    system = runtime_system(D1Arm.SG)
    fit_runtime_baseline(system)
    observed_grad_modes: list[bool] = []
    handle = system.target_branch.register_forward_hook(
        lambda _module, _args, _output: observed_grad_modes.append(torch.is_grad_enabled())
    )
    try:
        result = system(runtime_batch())
    finally:
        handle.remove()

    assert observed_grad_modes == [False]
    assert result.target.detached is True
    assert not result.target.values.requires_grad


def test_horizon_one_and_alignments_are_revalidated_at_the_runtime_boundary() -> None:
    system = runtime_system(D1Arm.NONE)
    fit_runtime_baseline(system)
    with pytest.raises(ValueError, match="horizon=1"):
        system(runtime_batch(horizon=2))


def test_ema_post_commit_and_state_dict_roundtrip_preserve_policy_and_baseline() -> None:
    system = runtime_system(D1Arm.BOTH)
    fit_runtime_baseline(system)
    with torch.no_grad():
        for parameter in system.online_branch.parameters():
            parameter.add_(1.0)
    system.after_successful_optimizer_step(OptimizationStepCommitted(step=1))
    state = system.state_dict()

    restored = runtime_system(D1Arm.BOTH)
    restored.load_state_dict(state)

    assert restored.target_policy_state() == system.target_policy_state()
    assert restored.mean_displacement_baseline.is_fitted
    assert restored.construction_identity == system.construction_identity
    for name, value in system.target_branch.state_dict().items():
        torch.testing.assert_close(value, restored.target_branch.state_dict()[name])


def test_checkpoint_refuses_a_different_paired_initialization_seed() -> None:
    source = runtime_system(D1Arm.BOTH, seed=7)
    fit_runtime_baseline(source)
    incompatible = runtime_system(D1Arm.BOTH, seed=8)

    with pytest.raises(ValueError, match="initialization identity"):
        incompatible.load_state_dict(source.state_dict())
