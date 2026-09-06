"""Transactional application service for one registered D1 training step."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

from entropic_substrate.d1_optimizer import D1OptimizerBundle
from entropic_substrate.data.batching import PredictiveBatch
from entropic_substrate.experiments.d1_execution import D1AmpPolicy, D1OptimizationSpec
from entropic_substrate.experiments.d1_primitives import TargetUpdateStrategy, canonical_digest
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem
from entropic_substrate.predictive_state.runtime_components import D1PredictiveResult
from entropic_substrate.predictive_state.runtime_construction import target_policy_from_draft
from entropic_substrate.predictive_state.target_updates import (
    OnlineTargetUpdate,
    OptimizationStepCommitted,
    StopGradientEmaUpdate,
)

__all__ = [
    "D1ComponentGradientEvidence",
    "D1GradientEvidenceKind",
    "D1GradientReachability",
    "D1TrainStepResult",
    "apply_d1_training_step",
    "d1_learning_rate",
]

_clip_grad_norm = nn.utils.clip_grad_norm_


class D1GradientEvidenceKind(StrEnum):
    """The only admissible interpretations of a D1 component gradient."""

    MEASURED_NONZERO = "measured_nonzero"
    MEASURED_ZERO = "measured_zero"
    STRUCTURALLY_ABSENT_STOP_GRADIENT = "structurally_absent_stop_gradient"


@dataclass(frozen=True, slots=True)
class D1ComponentGradientEvidence:
    """Pre-clip gradient norm or the registered stop-gradient absence."""

    kind: D1GradientEvidenceKind
    pre_clip_l2_norm: float | None

    def __post_init__(self) -> None:
        if type(self.kind) is not D1GradientEvidenceKind:
            raise TypeError("D1 component gradient kind must be D1GradientEvidenceKind")
        if self.kind is D1GradientEvidenceKind.MEASURED_NONZERO:
            if (
                type(self.pre_clip_l2_norm) is not float
                or not math.isfinite(self.pre_clip_l2_norm)
                or self.pre_clip_l2_norm <= 0.0
            ):
                raise ValueError("D1 measured component gradient must be a finite nonzero float")
        elif self.kind is D1GradientEvidenceKind.MEASURED_ZERO:
            if type(self.pre_clip_l2_norm) is not float or self.pre_clip_l2_norm != 0.0:
                raise ValueError("D1 measured zero gradient must have a zero float norm")
        elif self.pre_clip_l2_norm is not None:
            raise ValueError("D1 structurally absent gradient must have a null norm")

    @property
    def manifest(self) -> dict[str, object]:
        return {"kind": self.kind.value, "pre_clip_l2_norm": self.pre_clip_l2_norm}


@dataclass(frozen=True, slots=True)
class D1GradientReachability:
    """Closed four-component gradient evidence for one D1 backward pass."""

    online_branch: D1ComponentGradientEvidence
    target_branch: D1ComponentGradientEvidence
    language_readout: D1ComponentGradientEvidence
    future_law: D1ComponentGradientEvidence

    def __post_init__(self) -> None:
        components = (
            self.online_branch,
            self.target_branch,
            self.language_readout,
            self.future_law,
        )
        if any(type(item) is not D1ComponentGradientEvidence for item in components):
            raise TypeError("D1 gradient reachability requires four component evidence values")
        required = (self.online_branch, self.language_readout, self.future_law)
        measured = {D1GradientEvidenceKind.MEASURED_NONZERO, D1GradientEvidenceKind.MEASURED_ZERO}
        if any(item.kind not in measured for item in required):
            raise ValueError("D1 non-target component gradients must be measured")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "future_law": self.future_law.manifest,
            "language_readout": self.language_readout.manifest,
            "online_branch": self.online_branch.manifest,
            "target_branch": self.target_branch.manifest,
        }


@dataclass(frozen=True, slots=True)
class D1TrainStepResult:
    """Immutable, unrounded scalar evidence from one committed optimizer step."""

    successful_step: int
    learning_rate: float
    objective: float
    language_loss: float
    future_nll: float
    variance_penalty: float
    covariance_penalty: float
    raw_variance_penalty: float
    raw_covariance_penalty: float
    zero_displacement_nll: float
    mean_displacement_nll: float
    residual_mean: float
    residual_rms: float
    residual_variance: float
    coverage_50: float
    coverage_80: float
    coverage_95: float
    effective_rank: float
    numerical_rank: int
    gradient_norm: float
    gradient_reachability: D1GradientReachability

    def __post_init__(self) -> None:
        if type(self.successful_step) is not int or self.successful_step < 1:
            msg = "D1 train result successful_step must be a positive integer"
            raise ValueError(msg)
        float_fields = (
            "learning_rate",
            "objective",
            "language_loss",
            "future_nll",
            "variance_penalty",
            "covariance_penalty",
            "raw_variance_penalty",
            "raw_covariance_penalty",
            "zero_displacement_nll",
            "mean_displacement_nll",
            "residual_mean",
            "residual_rms",
            "residual_variance",
            "coverage_50",
            "coverage_80",
            "coverage_95",
            "effective_rank",
            "gradient_norm",
        )
        for name in float_fields:
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                msg = f"D1 train result {name} must be a finite float"
                raise ValueError(msg)
        if self.learning_rate <= 0.0 or self.gradient_norm < 0.0:
            msg = "D1 train result learning rate must be positive and gradient norm nonnegative"
            raise ValueError(msg)
        if any(not 0.0 <= value <= 1.0 for value in self.coverages):
            msg = "D1 train result coverage must be in [0, 1]"
            raise ValueError(msg)
        if type(self.numerical_rank) is not int or self.numerical_rank < 0:
            msg = "D1 train result numerical_rank must be a nonnegative integer"
            raise ValueError(msg)
        if type(self.gradient_reachability) is not D1GradientReachability:
            raise TypeError("D1 train result requires typed gradient reachability")
        _require_aggregate_gradient_norm(self.gradient_reachability, self.gradient_norm)

    @property
    def coverages(self) -> tuple[float, float, float]:
        return self.coverage_50, self.coverage_80, self.coverage_95

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "covariance_penalty": self.covariance_penalty,
            "coverage_50": self.coverage_50,
            "coverage_80": self.coverage_80,
            "coverage_95": self.coverage_95,
            "effective_rank": self.effective_rank,
            "future_nll": self.future_nll,
            "gradient_norm": self.gradient_norm,
            "gradient_reachability": self.gradient_reachability.manifest,
            "language_loss": self.language_loss,
            "learning_rate": self.learning_rate,
            "mean_displacement_nll": self.mean_displacement_nll,
            "numerical_rank": self.numerical_rank,
            "objective": self.objective,
            "raw_covariance_penalty": self.raw_covariance_penalty,
            "raw_variance_penalty": self.raw_variance_penalty,
            "residual_mean": self.residual_mean,
            "residual_rms": self.residual_rms,
            "residual_variance": self.residual_variance,
            "successful_step": self.successful_step,
            "variance_penalty": self.variance_penalty,
            "zero_displacement_nll": self.zero_displacement_nll,
        }


def d1_learning_rate(spec: D1OptimizationSpec, *, successful_step: int) -> float:
    """Registered 1-based warmup/cosine schedule with an exact terminal floor."""
    if type(spec) is not D1OptimizationSpec:
        msg = "D1 learning-rate schedule requires a D1OptimizationSpec"
        raise TypeError(msg)
    if type(successful_step) is not int or successful_step < 1 or successful_step > spec.max_steps:
        msg = (
            "D1 learning-rate successful step must be in "
            f"[1, {spec.max_steps}], got {successful_step!r}"
        )
        raise ValueError(msg)
    base = float(spec.learning_rate)
    floor = base * float(spec.min_learning_rate_ratio)
    if successful_step == spec.max_steps:
        return floor
    if spec.warmup_steps > 0 and successful_step <= spec.warmup_steps:
        return base * successful_step / spec.warmup_steps
    decay_origin = max(spec.warmup_steps, 1)
    decay_span = spec.max_steps - decay_origin
    if decay_span <= 0:
        return floor
    progress = (successful_step - decay_origin) / decay_span
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (
        float(spec.min_learning_rate_ratio)
        + (1.0 - float(spec.min_learning_rate_ratio)) * coefficient
    )


def apply_d1_training_step(
    *,
    system: D1PredictiveSystem,
    optimizer_bundle: D1OptimizerBundle,
    batch: PredictiveBatch,
    spec: D1OptimizationSpec,
    successful_step: int,
) -> D1TrainStepResult:
    """Apply one D1 update and expose a commit only after every finite guard."""
    owned, next_step = _preflight(
        system=system,
        optimizer_bundle=optimizer_bundle,
        batch=batch,
        spec=spec,
        successful_step=successful_step,
    )
    learning_rate = d1_learning_rate(spec, successful_step=next_step)
    optimizer_bundle.set_learning_rate(learning_rate)
    device = owned[0].device
    was_training = system.training
    system.train()
    try:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=spec.amp is D1AmpPolicy.BF16,
        ):
            output = system(batch)
        _require_finite_output(output)
        optimizer_bundle.optimizer.zero_grad(set_to_none=True)
        objective = output.objective.total
        _require_finite_tensor("objective.total", objective)
        objective.backward()
        gradient_reachability = _gradient_reachability(system, owned)
        gradient_norm = _clip_grad_norm(
            owned,
            float(spec.grad_clip_norm),
            error_if_nonfinite=True,
        )
        gradient_norm_value = _as_finite_float("gradient norm", gradient_norm)
        _require_aggregate_gradient_norm(gradient_reachability, gradient_norm_value)
        _require_finite_gradients(owned)
        result = _result(
            output,
            successful_step=next_step,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm_value,
            gradient_reachability=gradient_reachability,
        )
        optimizer_bundle.optimizer.step()
        _require_finite_parameters(owned)
        system.after_successful_optimizer_step(OptimizationStepCommitted(step=next_step))
        _require_target_policy_binding(system, expected_cursor=next_step)
        return result
    finally:
        system.train(was_training)


def _preflight(
    *,
    system: D1PredictiveSystem,
    optimizer_bundle: D1OptimizerBundle,
    batch: PredictiveBatch,
    spec: D1OptimizationSpec,
    successful_step: int,
) -> tuple[tuple[nn.Parameter, ...], int]:
    if type(system) is not D1PredictiveSystem:
        raise TypeError("D1 training step requires a D1PredictiveSystem")
    if type(optimizer_bundle) is not D1OptimizerBundle:
        raise TypeError("D1 training step requires a D1OptimizerBundle")
    if type(batch) is not PredictiveBatch:
        raise TypeError("D1 training step requires a PredictiveBatch")
    if type(spec) is not D1OptimizationSpec:
        raise TypeError("D1 training step requires a D1OptimizationSpec")
    if type(successful_step) is not int or successful_step < 0:
        msg = "D1 current successful step must be a nonnegative integer"
        raise ValueError(msg)
    if spec != system.draft.execution_protocol.optimization:
        raise ValueError("D1 training step requires the runtime's registered optimization spec")
    if optimizer_bundle.spec_digest != canonical_digest(spec.manifest):
        raise ValueError("D1 optimizer bundle does not attest the registered optimization spec")
    if type(optimizer_bundle.optimizer) is not torch.optim.AdamW:
        raise TypeError("D1 optimizer bundle must contain the registered AdamW optimizer")
    system.validate_optimizer_ownership(optimizer_bundle.optimizer)
    owned = _owned_parameters(optimizer_bundle.optimizer)
    _require_finite_parameters(owned)
    _require_target_policy_binding(system, expected_cursor=successful_step)
    next_step = successful_step + 1
    d1_learning_rate(spec, successful_step=next_step)
    return owned, next_step


def _owned_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    owned: list[nn.Parameter] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not isinstance(parameter, nn.Parameter):
                raise TypeError("D1 optimizer groups may contain only Parameters")
            owned.append(parameter)
    if not owned:
        raise ValueError("D1 optimizer owns no parameters")
    return tuple(owned)


def _require_target_policy_binding(
    system: D1PredictiveSystem,
    *,
    expected_cursor: int,
) -> None:
    strategy = system.draft.target_update
    policy = system.target_policy
    state = system.target_policy_state()
    expected_policy = target_policy_from_draft(system.draft)
    expected_state = expected_policy.state_dict()
    if strategy is TargetUpdateStrategy.ONLINE:
        if type(policy) is not OnlineTargetUpdate or state != expected_state:
            raise ValueError("D1 active target policy differs from registered online strategy")
        return
    if type(policy) is not StopGradientEmaUpdate:
        raise ValueError("D1 active target policy differs from registered stop-gradient strategy")
    cursor = state.get("update_cursor")
    if type(cursor) is not int or cursor != expected_cursor:
        raise ValueError(
            f"D1 target-policy cursor must equal successful step {expected_cursor}, got {cursor!r}"
        )
    expected_state["update_cursor"] = expected_cursor
    if state != expected_state:
        raise ValueError("D1 active target policy differs from registered stop-gradient strategy")


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if type(value) is not torch.Tensor or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"D1 {name} must be a finite tensor")


def _require_finite_output(output: D1PredictiveResult) -> None:
    if type(output) is not D1PredictiveResult:
        raise TypeError("D1 runtime must return a D1PredictiveResult")
    diagnostics = output.collapse.diagnostics
    if diagnostics is None:
        raise ValueError("D1 training step requires collapse diagnostics")
    tensors = {
        "language_logits": output.language_logits,
        "online_states": output.online_states.values,
        "target": output.target.values,
        "future.mean": output.future_law_loss.mean,
        "objective.language": output.objective.language,
        "objective.future": output.objective.future,
        "objective.variance": output.objective.variance,
        "objective.covariance": output.objective.covariance,
        "collapse.raw_variance": output.collapse.raw_variance_penalty,
        "collapse.weighted_variance": output.collapse.weighted_variance_penalty,
        "collapse.raw_covariance": output.collapse.raw_covariance_penalty,
        "collapse.weighted_covariance": output.collapse.weighted_covariance_penalty,
        "collapse.total": output.collapse.total,
        "collapse.variance": diagnostics.variance,
        "collapse.std": diagnostics.std,
        "collapse.covariance": diagnostics.covariance,
        "collapse.eigenvalues": diagnostics.covariance_eigenvalues,
    }
    law = output.future_law_loss
    for name in (
        "nll",
        "zero_displacement_nll",
        "mean_displacement_nll",
        "residual_mean",
        "residual_rms",
        "residual_variance",
        "coverage_50",
        "coverage_80",
        "coverage_95",
    ):
        tensors[f"future.{name}"] = getattr(law, name)
    for name, tensor in tensors.items():
        _require_finite_tensor(name, tensor)
    if not math.isfinite(diagnostics.effective_rank):
        raise ValueError("D1 collapse effective-rank diagnostic must be finite")


def _require_finite_gradients(owned: tuple[nn.Parameter, ...]) -> None:
    for parameter in owned:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            raise ValueError("D1 optimizer gradient must be finite")


def _gradient_reachability(
    system: D1PredictiveSystem, owned: tuple[nn.Parameter, ...]
) -> D1GradientReachability:
    components = {
        "online_branch": tuple(system.online_branch.parameters()),
        "target_branch": tuple(system.target_branch.parameters()),
        "language_readout": tuple(system.language_readout.parameters()),
        "future_law": tuple(system.future_law.parameters()),
    }
    component_ids = [id(parameter) for values in components.values() for parameter in values]
    system_ids = {id(parameter) for parameter in system.parameters()}
    owned_ids = {id(parameter) for parameter in owned}
    trainable_ids = {
        id(parameter)
        for values in components.values()
        for parameter in values
        if parameter.requires_grad
    }
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("D1 gradient components must not share a parameter")
    if set(component_ids) != system_ids:
        raise ValueError("D1 gradient components must cover every system parameter exactly once")
    if trainable_ids != owned_ids:
        raise ValueError("D1 trainable component parameters must equal optimizer ownership")

    target = components["target_branch"]
    if system.draft.target_update is TargetUpdateStrategy.STOP_GRADIENT_EMA:
        if any(parameter.requires_grad for parameter in target):
            raise ValueError("D1 registered stop-gradient target_branch must be frozen")
        if any(parameter.grad is not None for parameter in target):
            raise ValueError("D1 target_branch gradient must be structurally absent")
        target_evidence = D1ComponentGradientEvidence(
            D1GradientEvidenceKind.STRUCTURALLY_ABSENT_STOP_GRADIENT, None
        )
    else:
        if any(not parameter.requires_grad for parameter in target):
            raise ValueError("D1 registered online target_branch must remain trainable")
        target_evidence = _measured_component_gradient("target_branch", target)
    return D1GradientReachability(
        online_branch=_measured_component_gradient("online_branch", components["online_branch"]),
        target_branch=target_evidence,
        language_readout=_measured_component_gradient(
            "language_readout", components["language_readout"]
        ),
        future_law=_measured_component_gradient("future_law", components["future_law"]),
    )


def _measured_component_gradient(
    name: str, parameters: tuple[nn.Parameter, ...]
) -> D1ComponentGradientEvidence:
    squared_norm = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        if not parameter.requires_grad or gradient is None:
            raise ValueError(f"D1 {name} gradient must reach every trainable parameter")
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError(f"D1 {name} gradient must be finite")
        tensor_norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
        squared_norm += tensor_norm * tensor_norm
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm):
        raise ValueError(f"D1 {name} gradient norm must be finite")
    # A connected stationary/collapsed model has finite zero gradients. The
    # scientific clauses decide collapse; graph reachability is checked above.
    kind = (
        D1GradientEvidenceKind.MEASURED_ZERO
        if norm == 0.0
        else D1GradientEvidenceKind.MEASURED_NONZERO
    )
    return D1ComponentGradientEvidence(kind, norm)


def _require_aggregate_gradient_norm(
    reachability: D1GradientReachability,
    aggregate_norm: float,
) -> None:
    """Bind component norms to the pre-clip total returned by ``clip_grad_norm_``."""
    component_norms = tuple(
        item.pre_clip_l2_norm
        for item in (
            reachability.online_branch,
            reachability.target_branch,
            reachability.language_readout,
            reachability.future_law,
        )
        if item.pre_clip_l2_norm is not None
    )
    if not math.isclose(
        aggregate_norm,
        math.hypot(*component_norms),
        rel_tol=1e-5,
        abs_tol=1e-12,
    ):
        raise ValueError("D1 component gradients do not reconstruct the aggregate norm")


def _require_finite_parameters(owned: tuple[nn.Parameter, ...]) -> None:
    for parameter in owned:
        if not bool(torch.isfinite(parameter).all().item()):
            raise ValueError("D1 optimizer parameter must be finite")


def _as_finite_float(name: str, value: torch.Tensor | float) -> float:
    scalar = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"D1 {name} must be finite")
    return scalar


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().item())


def _result(
    output: D1PredictiveResult,
    *,
    successful_step: int,
    learning_rate: float,
    gradient_norm: float,
    gradient_reachability: D1GradientReachability,
) -> D1TrainStepResult:
    law, collapse = output.future_law_loss, output.collapse
    diagnostics = collapse.diagnostics
    if diagnostics is None:  # already rejected before mutation; retains static narrowing
        raise ValueError("D1 training step requires collapse diagnostics")
    return D1TrainStepResult(
        successful_step=successful_step,
        learning_rate=float(learning_rate),
        objective=_scalar(output.objective.total),
        language_loss=_scalar(output.objective.language),
        future_nll=_scalar(law.nll),
        variance_penalty=_scalar(output.objective.variance),
        covariance_penalty=_scalar(output.objective.covariance),
        raw_variance_penalty=_scalar(collapse.raw_variance_penalty),
        raw_covariance_penalty=_scalar(collapse.raw_covariance_penalty),
        zero_displacement_nll=_scalar(law.zero_displacement_nll),
        mean_displacement_nll=_scalar(law.mean_displacement_nll),
        residual_mean=_scalar(law.residual_mean),
        residual_rms=_scalar(law.residual_rms),
        residual_variance=_scalar(law.residual_variance),
        coverage_50=_scalar(law.coverage_50),
        coverage_80=_scalar(law.coverage_80),
        coverage_95=_scalar(law.coverage_95),
        effective_rank=float(diagnostics.effective_rank),
        numerical_rank=diagnostics.numerical_rank,
        gradient_norm=gradient_norm,
        gradient_reachability=gradient_reachability,
    )
