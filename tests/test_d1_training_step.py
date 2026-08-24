"""Transactional optimization-step contracts for D1."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
import torch
from torch import nn

import entropic_substrate.d1_training_step as step_module
from d1_runtime_fixtures import fit_runtime_baseline, runtime_batch, runtime_family, runtime_system
from entropic_substrate.d1_optimizer import D1OptimizerBundle, build_d1_optimizer
from entropic_substrate.d1_training_step import (
    D1TrainStepResult,
    apply_d1_training_step,
    d1_learning_rate,
)
from entropic_substrate.experiments.d1_execution import D1AmpPolicy
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

pytestmark = pytest.mark.unit


def _runtime(
    arm: D1Arm = D1Arm.BOTH, *, amp: D1AmpPolicy | None = None
) -> tuple[D1PredictiveSystem, D1OptimizerBundle]:
    if amp is None:
        system = runtime_system(arm)
    else:
        draft = runtime_family().draft_for(arm)
        protocol = replace(
            draft.execution_protocol,
            optimization=replace(draft.execution_protocol.optimization, amp=amp),
        )
        system = D1PredictiveSystem(
            replace(draft, execution_protocol=protocol),
            initializer=SeededD1BranchInitializer(seed=7),
        )
    fit_runtime_baseline(system)
    spec = system.draft.execution_protocol.optimization
    return system, build_d1_optimizer(system, spec)


def _cursor(system: D1PredictiveSystem) -> int | None:
    value = system.target_policy_state().get("update_cursor")
    return value if type(value) is int else None


def _snap(system: D1PredictiveSystem) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in system.parameters())


def _unchanged(system: D1PredictiveSystem, before: tuple[torch.Tensor, ...]) -> None:
    for actual, expected in zip(system.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _apply(
    system: D1PredictiveSystem,
    bundle: D1OptimizerBundle,
    *,
    successful_step: int = 0,
) -> D1TrainStepResult:
    return apply_d1_training_step(
        system=system,
        optimizer_bundle=bundle,
        batch=runtime_batch(),
        spec=system.draft.execution_protocol.optimization,
        successful_step=successful_step,
    )


def test_learning_rate_has_exact_registered_endpoints() -> None:
    spec = replace(
        runtime_family().drafts[0].execution_protocol.optimization,
        learning_rate=2.0,
        min_learning_rate_ratio=0.1,
        warmup_steps=2,
        max_steps=6,
    )
    assert d1_learning_rate(spec, successful_step=1) == 1.0
    assert d1_learning_rate(spec, successful_step=2) == 2.0
    assert d1_learning_rate(spec, successful_step=4) == pytest.approx(1.1)
    assert d1_learning_rate(spec, successful_step=6) == 0.2
    no_warmup = replace(spec, warmup_steps=0)
    assert d1_learning_rate(no_warmup, successful_step=1) == 2.0
    assert d1_learning_rate(no_warmup, successful_step=6) == 0.2
    for invalid in (True, 0, -1, 7):
        with pytest.raises(ValueError, match="successful step"):
            d1_learning_rate(spec, successful_step=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("arm", list(D1Arm))
def test_one_transaction_path_optimizes_every_arm(arm: D1Arm) -> None:
    system, bundle = _runtime(arm)
    before = _snap(system)
    result = _apply(system, bundle)

    assert result.successful_step == 1
    assert math.isfinite(result.objective)
    assert math.isfinite(result.gradient_norm)
    assert any(
        not torch.equal(actual.detach(), expected)
        for actual, expected in zip(system.parameters(), before, strict=True)
        if actual.requires_grad
    )
    assert _cursor(system) == (1 if arm in (D1Arm.SG, D1Arm.BOTH) else None)


@pytest.mark.parametrize(("amp", "enabled"), [(D1AmpPolicy.OFF, False), (D1AmpPolicy.BF16, True)])
def test_forward_honors_amp_and_restores_entry_mode(amp: D1AmpPolicy, enabled: bool) -> None:
    system, bundle = _runtime(amp=amp)
    system.eval()
    observed: list[tuple[bool, bool]] = []
    handle = system.register_forward_pre_hook(
        lambda module, _args: observed.append((module.training, torch.is_autocast_enabled("cpu")))
    )
    try:
        _apply(system, bundle)
    finally:
        handle.remove()
    assert observed == [(True, enabled)]
    assert system.training is False


def test_ema_cursor_is_preflighted_and_result_is_immutable() -> None:
    system, bundle = _runtime(D1Arm.SG)
    result = _apply(system, bundle)
    assert _cursor(system) == 1
    with pytest.raises(ValueError, match="cursor"):
        _apply(system, bundle, successful_step=0)
    assert _cursor(system) == 1
    with pytest.raises(FrozenInstanceError):
        result.objective = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        replace(result, gradient_norm=float("nan"))


def test_nan_objective_is_caught_before_gradients_or_commit() -> None:
    system, bundle = _runtime()
    before = _snap(system)

    def poison(_module: nn.Module, _args: tuple[object, ...], output: Any) -> Any:
        with torch.no_grad():
            output.objective.language.fill_(float("nan"))
        return output

    handle = system.register_forward_hook(poison)
    try:
        with pytest.raises(ValueError, match=r"objective.language.*finite"):
            _apply(system, bundle)
    finally:
        handle.remove()
    _unchanged(system, before)
    assert all(parameter.grad is None for parameter in system.parameters())
    assert not bundle.optimizer.state
    assert _cursor(system) == 0


def test_backward_and_optimizer_failures_do_not_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    system, bundle = _runtime()
    before = _snap(system)

    def fail_backward(*_args: object) -> None:
        raise RuntimeError("deliberate backward failure")

    handle = system.future_law.predictor[2].register_full_backward_hook(fail_backward)
    try:
        with pytest.raises(RuntimeError, match="deliberate backward failure"):
            _apply(system, bundle)
    finally:
        handle.remove()
    _unchanged(system, before)
    assert not bundle.optimizer.state
    assert _cursor(system) == 0

    monkeypatch.setattr(
        bundle.optimizer,
        "step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("deliberate optimizer failure")
        ),
    )
    with pytest.raises(RuntimeError, match="deliberate optimizer failure"):
        _apply(system, bundle)
    _unchanged(system, before)
    assert _cursor(system) == 0


def test_explicit_gradient_guard_has_teeth_when_norm_guard_is_neutralized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system, bundle = _runtime()
    before = _snap(system)
    victim = next(parameter for parameter in system.optimizer_parameters() if parameter.ndim >= 2)
    handle = victim.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    monkeypatch.setattr(
        step_module.nn.utils,
        "clip_grad_norm_",
        lambda *_args, **_kwargs: torch.tensor(1.0),
    )
    try:
        with pytest.raises(ValueError, match=r"gradient.*finite"):
            _apply(system, bundle)
    finally:
        handle.remove()
    _unchanged(system, before)
    assert not bundle.optimizer.state
    assert _cursor(system) == 0


def test_post_step_nan_and_ownership_drift_are_caught_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system, bundle = _runtime()
    victim = next(iter(system.optimizer_parameters()))
    real_step = bundle.optimizer.step

    def poison_step(*args: object, **kwargs: object) -> object:
        result = real_step(*args, **kwargs)
        with torch.no_grad():
            victim.fill_(float("nan"))
        return result

    monkeypatch.setattr(bundle.optimizer, "step", poison_step)
    with pytest.raises(ValueError, match=r"optimizer parameter.*finite"):
        _apply(system, bundle)
    assert _cursor(system) == 0

    clean_system, clean_bundle = _runtime()
    clean_bundle.optimizer.param_groups[0]["params"].append(nn.Parameter(torch.zeros(())))
    with pytest.raises(ValueError, match="every and only"):
        _apply(clean_system, clean_bundle)
    assert _cursor(clean_system) == 0


def test_registered_spec_and_step_domain_are_preflight_requirements() -> None:
    system, bundle = _runtime()
    changed = replace(system.draft.execution_protocol.optimization, grad_clip_norm=2.0)
    with pytest.raises(ValueError, match="registered optimization spec"):
        apply_d1_training_step(
            system=system,
            optimizer_bundle=bundle,
            batch=runtime_batch(),
            spec=changed,
            successful_step=0,
        )
    with pytest.raises(ValueError, match="successful step"):
        _apply(
            system, bundle, successful_step=system.draft.execution_protocol.optimization.max_steps
        )
