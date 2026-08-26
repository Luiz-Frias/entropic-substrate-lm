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
    D1GradientEvidenceKind,
    D1GradientReachability,
    D1TrainStepResult,
    apply_d1_training_step,
    d1_learning_rate,
)
from entropic_substrate.experiments.d1_execution import D1AmpPolicy
from entropic_substrate.experiments.d1_primitives import D1Arm, EMAKind
from entropic_substrate.experiments.d1_stabilization import EMAParameters
from entropic_substrate.predictive_state.ema_schedules import CosineEmaDecaySchedule
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem
from entropic_substrate.predictive_state.target_updates import (
    OnlineTargetUpdate,
    StopGradientEmaUpdate,
)

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


def _capture_reachability(
    monkeypatch: pytest.MonkeyPatch,
) -> list[D1GradientReachability]:
    captured: list[D1GradientReachability] = []
    measure = step_module._gradient_reachability  # pyright: ignore[reportPrivateUsage]

    def record(
        system: D1PredictiveSystem,
        owned: tuple[nn.Parameter, ...],
    ) -> D1GradientReachability:
        evidence = measure(system, owned)
        captured.append(evidence)
        return evidence

    monkeypatch.setattr(step_module, "_gradient_reachability", record)
    return captured


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
def test_one_transaction_path_optimizes_every_arm(
    arm: D1Arm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system, bundle = _runtime(arm)
    before = _snap(system)
    captured = _capture_reachability(monkeypatch)
    result = _apply(system, bundle)

    assert result.successful_step == 1
    assert result.manifest["successful_step"] == 1
    assert result.manifest["objective"] == result.objective
    assert set(result.manifest) == {
        "covariance_penalty",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "effective_rank",
        "future_nll",
        "gradient_norm",
        "language_loss",
        "learning_rate",
        "mean_displacement_nll",
        "numerical_rank",
        "objective",
        "raw_covariance_penalty",
        "raw_variance_penalty",
        "residual_mean",
        "residual_rms",
        "residual_variance",
        "successful_step",
        "variance_penalty",
        "zero_displacement_nll",
    }
    assert math.isfinite(result.objective)
    assert math.isfinite(result.gradient_norm)
    assert len(captured) == 1
    reachability = captured[0]
    assert set(reachability.manifest) == {
        "future_law",
        "language_readout",
        "online_branch",
        "target_branch",
    }
    for name in ("online_branch", "language_readout", "future_law"):
        evidence = getattr(reachability, name)
        assert evidence.kind is D1GradientEvidenceKind.MEASURED_NONZERO
        assert evidence.pre_clip_l2_norm is not None and evidence.pre_clip_l2_norm > 0.0
    target = reachability.target_branch
    if arm in (D1Arm.SG, D1Arm.BOTH):
        assert target.kind is D1GradientEvidenceKind.STRUCTURALLY_ABSENT_STOP_GRADIENT
        assert target.pre_clip_l2_norm is None
    else:
        assert target.kind is D1GradientEvidenceKind.MEASURED_NONZERO
        assert target.pre_clip_l2_norm is not None and target.pre_clip_l2_norm > 0.0
    component_squared_norm = sum(
        evidence.pre_clip_l2_norm**2
        for evidence in (
            reachability.online_branch,
            reachability.target_branch,
            reachability.language_readout,
            reachability.future_law,
        )
        if evidence.pre_clip_l2_norm is not None
    )
    assert result.gradient_norm**2 == pytest.approx(component_squared_norm, rel=1e-5, abs=1e-12)
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
        step_module,
        "_clip_grad_norm",
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


def test_component_gradient_guard_refuses_zero_and_unpartitioned_parameters() -> None:
    system, bundle = _runtime(D1Arm.NONE)
    before = _snap(system)
    handles = [
        parameter.register_hook(torch.zeros_like)
        for parameter in system.language_readout.parameters()
    ]
    try:
        with pytest.raises(ValueError, match=r"language_readout.*nonzero"):
            _apply(system, bundle)
    finally:
        for handle in handles:
            handle.remove()
    _unchanged(system, before)
    assert not bundle.optimizer.state
    assert _cursor(system) is None

    drifted = runtime_system(D1Arm.NONE)
    fit_runtime_baseline(drifted)
    drifted.register_parameter("unpartitioned_probe", nn.Parameter(torch.ones(())))
    drifted_bundle = build_d1_optimizer(drifted, drifted.draft.execution_protocol.optimization)
    drifted_before = _snap(drifted)
    with pytest.raises(ValueError, match=r"cover every system parameter"):
        _apply(drifted, drifted_bundle)
    _unchanged(drifted, drifted_before)
    assert not drifted_bundle.optimizer.state
    assert _cursor(drifted) is None

    disconnected = runtime_system(D1Arm.NONE)
    fit_runtime_baseline(disconnected)
    disconnected.language_readout.register_parameter("unused_probe", nn.Parameter(torch.ones(())))
    disconnected_bundle = build_d1_optimizer(
        disconnected, disconnected.draft.execution_protocol.optimization
    )
    disconnected_before = _snap(disconnected)
    with pytest.raises(ValueError, match=r"language_readout.*every trainable parameter"):
        _apply(disconnected, disconnected_bundle)
    _unchanged(disconnected, disconnected_before)
    assert not disconnected_bundle.optimizer.state
    assert _cursor(disconnected) is None

    duplicated = runtime_system(D1Arm.NONE)
    fit_runtime_baseline(duplicated)
    duplicated.language_readout.register_parameter(
        "duplicate_probe", next(duplicated.online_branch.parameters())
    )
    duplicated_bundle = build_d1_optimizer(
        duplicated, duplicated.draft.execution_protocol.optimization
    )
    duplicated_before = _snap(duplicated)
    with pytest.raises(ValueError, match=r"must not share"):
        _apply(duplicated, duplicated_bundle)
    _unchanged(duplicated, duplicated_before)
    assert not duplicated_bundle.optimizer.state
    assert _cursor(duplicated) is None

    laundered = runtime_system(D1Arm.NONE)
    fit_runtime_baseline(laundered)
    for parameter in laundered.target_branch.parameters():
        parameter.requires_grad_(False)
    laundered_bundle = build_d1_optimizer(
        laundered, laundered.draft.execution_protocol.optimization
    )
    laundered_before = _snap(laundered)
    with pytest.raises(ValueError, match=r"registered online target_branch.*trainable"):
        _apply(laundered, laundered_bundle)
    _unchanged(laundered, laundered_before)
    assert not laundered_bundle.optimizer.state
    assert _cursor(laundered) is None


def test_stop_gradient_absence_and_pre_clip_measurement_guards_have_teeth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped, stopped_bundle = _runtime(D1Arm.SG)
    stopped_before = _snap(stopped)
    target_parameter = next(stopped.target_branch.parameters())
    target_parameter.grad = torch.ones_like(target_parameter)
    with pytest.raises(ValueError, match=r"target_branch.*structurally absent"):
        _apply(stopped, stopped_bundle)
    _unchanged(stopped, stopped_before)
    assert not stopped_bundle.optimizer.state
    assert _cursor(stopped) == 0

    policy_drift, policy_drift_bundle = _runtime(D1Arm.SG)
    policy_drift_before = _snap(policy_drift)
    policy_drift.target_policy = OnlineTargetUpdate()
    with pytest.raises(ValueError, match=r"active target policy.*stop-gradient"):
        _apply(policy_drift, policy_drift_bundle)
    _unchanged(policy_drift, policy_drift_before)
    assert not policy_drift_bundle.optimizer.state

    fixed_drift, fixed_drift_bundle = _runtime(D1Arm.SG)
    fixed_drift_before = _snap(fixed_drift)
    fixed_drift.target_policy = StopGradientEmaUpdate(tau=0.5)
    with pytest.raises(ValueError, match=r"active target policy.*stop-gradient"):
        _apply(fixed_drift, fixed_drift_bundle)
    _unchanged(fixed_drift, fixed_drift_before)
    assert not fixed_drift_bundle.optimizer.state
    assert _cursor(fixed_drift) == 0

    cosine_family = runtime_family(
        ema=EMAParameters(
            kind=EMAKind.COSINE_SCHEDULE,
            decay=0.5,
            final_decay=0.9,
            schedule_steps=20_000,
        )
    )
    cosine_drift = D1PredictiveSystem.from_family(
        cosine_family,
        D1Arm.SG,
        initializer=SeededD1BranchInitializer(seed=7),
    )
    fit_runtime_baseline(cosine_drift)
    cosine_drift_bundle = build_d1_optimizer(
        cosine_drift,
        cosine_drift.draft.execution_protocol.optimization,
    )
    cosine_drift_before = _snap(cosine_drift)
    cosine_drift.target_policy = StopGradientEmaUpdate(
        schedule=CosineEmaDecaySchedule(
            start_decay=0.5,
            final_decay=0.8,
            total_successful_commits=20_000,
        )
    )
    with pytest.raises(ValueError, match=r"active target policy.*stop-gradient"):
        _apply(cosine_drift, cosine_drift_bundle)
    _unchanged(cosine_drift, cosine_drift_before)
    assert not cosine_drift_bundle.optimizer.state
    assert _cursor(cosine_drift) == 0

    system, bundle = _runtime(D1Arm.NONE)
    real_clip = step_module._clip_grad_norm  # pyright: ignore[reportPrivateUsage]
    captured = _capture_reachability(monkeypatch)

    def erase_after_clip(parameters: object, *args: object, **kwargs: object) -> torch.Tensor:
        owned = tuple(parameters)  # type: ignore[arg-type]
        norm = real_clip(owned, *args, **kwargs)
        for parameter in owned:
            if parameter.grad is not None:
                parameter.grad.zero_()
        return norm

    monkeypatch.setattr(step_module, "_clip_grad_norm", erase_after_clip)
    _apply(system, bundle)
    reachability = captured[0]
    assert all(
        evidence.pre_clip_l2_norm is not None and evidence.pre_clip_l2_norm > 0.0
        for evidence in (
            reachability.online_branch,
            reachability.target_branch,
            reachability.language_readout,
            reachability.future_law,
        )
    )

    mismatch, mismatch_bundle = _runtime(D1Arm.SG)
    mismatch_before = _snap(mismatch)
    mismatch_target_before = tuple(
        parameter.detach().clone() for parameter in mismatch.target_branch.parameters()
    )
    monkeypatch.setattr(
        step_module,
        "_clip_grad_norm",
        lambda *_args, **_kwargs: torch.tensor(123.0),
    )
    with pytest.raises(ValueError, match="reconstruct"):
        _apply(mismatch, mismatch_bundle)
    _unchanged(mismatch, mismatch_before)
    assert not mismatch_bundle.optimizer.state
    assert _cursor(mismatch) == 0
    for parameter, expected in zip(
        mismatch.target_branch.parameters(), mismatch_target_before, strict=True
    ):
        torch.testing.assert_close(parameter, expected)


@pytest.mark.parametrize(("control", "moment"), [(D1Arm.NONE, D1Arm.AC), (D1Arm.SG, D1Arm.BOTH)])
def test_matched_moment_arm_changes_only_the_observed_online_component_norm(
    control: D1Arm,
    moment: D1Arm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_system, control_bundle = _runtime(control)
    moment_system, moment_bundle = _runtime(moment)
    for name in (
        "online_branch",
        "target_branch",
        "language_readout",
        "future_law",
        "mean_displacement_baseline",
    ):
        getattr(moment_system, name).load_state_dict(getattr(control_system, name).state_dict())
    captured = _capture_reachability(monkeypatch)
    control_result = _apply(control_system, control_bundle)
    moment_result = _apply(moment_system, moment_bundle)
    control_reachability, moment_reachability = captured

    assert control_result.variance_penalty == control_result.covariance_penalty == 0.0
    assert moment_result.variance_penalty + moment_result.covariance_penalty > 0.0
    assert moment_reachability.online_branch.pre_clip_l2_norm != pytest.approx(
        control_reachability.online_branch.pre_clip_l2_norm
    )
    for name in ("target_branch", "language_readout", "future_law"):
        assert getattr(moment_reachability, name) == getattr(control_reachability, name)


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
