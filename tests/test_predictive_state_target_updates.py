"""Target-update and successful-commit EMA contracts for D1."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch

from entropic_substrate.core.latent.state import CausalStateEncoder, TargetEncoder
from entropic_substrate.core.latent.targets import (
    CosineEmaDecaySchedule,
    FixedEmaDecaySchedule,
    OnlineTargetUpdate,
    OptimizationStepCommitted,
    StopGradientEmaUpdate,
)
from predictive_state_helpers import CompositeTargetBranch, model_config

pytestmark = pytest.mark.unit


def test_online_target_update_preserves_target_gradient_path() -> None:
    target_branch, source, policy = (
        CompositeTargetBranch(),
        torch.randn(2, 3, 3, requires_grad=True),
        OnlineTargetUpdate(),
    )
    policy.configure_target_encoder(target_branch)
    policy.run_target_branch(lambda: target_branch(source)).values.square().mean().backward()
    assert target_branch.forward_grad_enabled is True
    assert any(parameter.grad is not None for parameter in target_branch.parameters())


def test_stop_gradient_ema_detaches_targets_freezes_teacher_and_excludes_optimizer() -> None:
    online_branch, target_branch = CompositeTargetBranch(), copy.deepcopy(CompositeTargetBranch())
    target_branch.load_state_dict(online_branch.state_dict())
    policy = StopGradientEmaUpdate(tau=0.75)
    policy.configure_target_encoder(target_branch)
    optimizer = torch.optim.AdamW(online_branch.parameters(), lr=0.01)
    stabilized = policy.run_target_branch(
        lambda: target_branch(torch.randn(2, 3, 3, requires_grad=True))
    )
    policy.validate_optimizer_excludes_target(optimizer, target_branch)
    assert not stabilized.values.requires_grad
    assert target_branch.forward_grad_enabled is False
    assert not any(parameter.requires_grad for parameter in target_branch.parameters())
    with pytest.raises(ValueError, match="target"):
        policy.validate_optimizer_excludes_target(
            torch.optim.AdamW([*online_branch.parameters(), *target_branch.parameters()], lr=0.01),
            target_branch,
        )


def test_stop_gradient_ema_updates_only_after_monotonic_commit_and_persists_cursor() -> None:
    online_branch, target_branch = CompositeTargetBranch(), copy.deepcopy(CompositeTargetBranch())
    target_branch.load_state_dict(online_branch.state_dict())
    policy = StopGradientEmaUpdate(tau=0.75)
    policy.configure_target_encoder(target_branch)
    before = {name: value.detach().clone() for name, value in target_branch.named_parameters()}
    with torch.no_grad():
        for parameter in online_branch.parameters():
            parameter.add_(2.0)
    policy.after_optimization_commit(
        online_branch, target_branch, OptimizationStepCommitted(step=1)
    )
    for name, target in target_branch.named_parameters():
        torch.testing.assert_close(
            target,
            before[name] * 0.75 + dict(online_branch.named_parameters())[name] * 0.25,
            atol=0.0,
            rtol=0.0,
        )
    assert policy.update_cursor == 1
    with pytest.raises(ValueError, match="monotonic"):
        policy.after_optimization_commit(
            online_branch, target_branch, OptimizationStepCommitted(step=1)
        )
    restored = StopGradientEmaUpdate(tau=0.1)
    restored.load_state_dict(copy.deepcopy(policy.state_dict()))
    assert restored.tau == 0.75 and restored.update_cursor == 1


def test_fixed_ema_decay_schedule_has_exact_constant_decay() -> None:
    schedule = FixedEmaDecaySchedule(tau=0.75)
    assert schedule.decay_for_successful_commit(1) == 0.75
    assert schedule.decay_for_successful_commit(123) == 0.75
    assert schedule.state_dict() == {"kind": "fixed", "tau": 0.75}


def test_cosine_ema_decay_schedule_has_registered_endpoints_and_monotonic_intermediate() -> None:
    schedule = CosineEmaDecaySchedule(start_decay=0.5, final_decay=0.9, total_successful_commits=5)
    decays = [schedule.decay_for_successful_commit(step) for step in range(1, 6)]
    assert (
        decays[0] == pytest.approx(0.5)
        and decays[2] == pytest.approx(0.7)
        and decays[-1] == pytest.approx(0.9)
    )
    assert decays == sorted(decays)
    with pytest.raises(ValueError, match="domain"):
        schedule.decay_for_successful_commit(0)
    with pytest.raises(ValueError, match="domain"):
        schedule.decay_for_successful_commit(6)


def test_cosine_ema_schedule_advances_only_on_contiguous_successful_commit() -> None:
    online_branch, target_branch = CompositeTargetBranch(), copy.deepcopy(CompositeTargetBranch())
    target_branch.load_state_dict(online_branch.state_dict())
    policy = StopGradientEmaUpdate(
        schedule=CosineEmaDecaySchedule(
            start_decay=0.5, final_decay=0.9, total_successful_commits=3
        )
    )
    policy.configure_target_encoder(target_branch)
    with pytest.raises(ValueError, match="monotonic"):
        policy.after_optimization_commit(
            online_branch, target_branch, OptimizationStepCommitted(step=2)
        )
    assert policy.update_cursor == 0
    policy.after_optimization_commit(
        online_branch, target_branch, OptimizationStepCommitted(step=1)
    )
    with pytest.raises(ValueError, match="monotonic"):
        policy.after_optimization_commit(
            online_branch, target_branch, OptimizationStepCommitted(step=3)
        )
    assert policy.update_cursor == 1


def test_cosine_ema_schedule_state_roundtrips_complete_identity_and_cursor() -> None:
    online_branch, target_branch = CompositeTargetBranch(), copy.deepcopy(CompositeTargetBranch())
    target_branch.load_state_dict(online_branch.state_dict())
    schedule = CosineEmaDecaySchedule(start_decay=0.4, final_decay=0.8, total_successful_commits=4)
    policy = StopGradientEmaUpdate(schedule=schedule)
    policy.configure_target_encoder(target_branch)
    policy.after_optimization_commit(
        online_branch, target_branch, OptimizationStepCommitted(step=1)
    )
    restored = StopGradientEmaUpdate(tau=0.1)
    restored.load_state_dict(copy.deepcopy(policy.state_dict()))
    assert isinstance(restored.schedule, CosineEmaDecaySchedule)
    assert restored.schedule.state_dict() == schedule.state_dict() and restored.update_cursor == 1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FixedEmaDecaySchedule(tau=-0.1),
        lambda: FixedEmaDecaySchedule(tau=1.0),
        lambda: CosineEmaDecaySchedule(
            start_decay=0.8, final_decay=0.7, total_successful_commits=4
        ),
        lambda: CosineEmaDecaySchedule(
            start_decay=0.5, final_decay=1.0, total_successful_commits=4
        ),
        lambda: CosineEmaDecaySchedule(
            start_decay=0.5, final_decay=0.9, total_successful_commits=0
        ),
        lambda: CosineEmaDecaySchedule(
            start_decay=0.5, final_decay=0.9, total_successful_commits=1
        ),
    ],
)
def test_ema_decay_schedules_reject_invalid_registered_domains(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("tau", [-0.1, 1.0, 1.1, float("nan")])
def test_stop_gradient_ema_rejects_invalid_decay(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        StopGradientEmaUpdate(tau=tau)


def test_stop_gradient_ema_rejects_parameter_map_shape_or_dtype_mismatch() -> None:
    policy = StopGradientEmaUpdate(tau=0.9)
    target = TargetEncoder(model_config(d_model=10, n_heads=2))
    policy.configure_target_encoder(target)
    with pytest.raises(ValueError, match="parameter"):
        policy.after_optimization_commit(
            CausalStateEncoder(model_config(d_model=8, n_heads=2)),
            target,
            OptimizationStepCommitted(step=1),
        )
    matching_target = TargetEncoder(model_config()).double()
    policy.configure_target_encoder(matching_target)
    with pytest.raises(ValueError, match="dtype"):
        policy.after_optimization_commit(
            CausalStateEncoder(model_config()), matching_target, OptimizationStepCommitted(step=1)
        )
