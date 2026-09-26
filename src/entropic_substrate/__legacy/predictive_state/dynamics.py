"""Euclidean and fully differentiated inverse-Fisher residual consumers."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from entropic_substrate.predictive_state.metric import (
    FisherNumericalError,
    MeanMap,
    finite_tensor,
    fisher_action,
    numerical_pair,
)


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Registered numerical settings; positive damping makes the operator SPD."""

    damping: float = 1.0
    max_iterations: int = 32
    relative_tolerance: float = 1e-5
    absolute_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("damping", "relative_tolerance", "absolute_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")


DEFAULT_SOLVER_CONFIG = SolverConfig()
type ParameterizedMeanMap = Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor]


class _ZeroContext(Protocol):
    mean_map: ParameterizedMeanMap
    config: SolverConfig
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *values: torch.Tensor) -> None: ...


class _ZeroInverse(torch.autograd.Function):
    """Exact-zero solution with its mathematical inverse tangent and adjoint.

    Saving the output (rather than a detached zero) preserves coefficient /
    residual mixed derivatives. Coefficients are explicit functional inputs
    so their VJP holds the saved solution fixed only within that transform.
    """

    @staticmethod
    def forward(
        ctx: _ZeroContext,
        mean_map: ParameterizedMeanMap,
        config: SolverConfig,
        states: torch.Tensor,
        residual: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        update = residual.clone()
        ctx.mean_map, ctx.config = mean_map, config
        ctx.save_for_backward(states, update, *parameters)
        return update

    @staticmethod
    def backward(
        ctx: _ZeroContext, *grad_outputs: torch.Tensor
    ) -> tuple[None, None, torch.Tensor, torch.Tensor, *tuple[torch.Tensor, ...]]:
        output_gradient = grad_outputs[0]
        states, update, *parameters = ctx.saved_tensors
        parameter_tuple = tuple(parameters)

        def mean(x: torch.Tensor) -> torch.Tensor:
            return ctx.mean_map(x, parameter_tuple)

        with torch.enable_grad():
            tangent = inverse_fisher_residual(
                mean,
                states,
                output_gradient,
                ctx.config,
                parameterized_mean_map=ctx.mean_map,
                mean_parameters=parameter_tuple,
            ).update

            def coefficients(h: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
                return fisher_action(lambda x: ctx.mean_map(x, params), h, update)

            coefficient_vjp = torch.func.vjp(coefficients, states, *parameters)[1]
            gradients = coefficient_vjp(-tangent / (1 + ctx.config.damping))
        return None, None, gradients[0], tangent, *gradients[1:]


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    """Per-position measured values; tensors are detached from the solve graph."""

    iterations: torch.Tensor
    residual_norm: torch.Tensor
    tolerance: torch.Tensor
    metric_action_norm: torch.Tensor
    damping_action_norm: torch.Tensor
    update_norm: torch.Tensor


@dataclass(frozen=True, slots=True)
class ResidualSolution:
    """The consumed residual and its numerical evidence; caller adds the state."""

    update: torch.Tensor
    diagnostics: SolverDiagnostics


def _norm(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values, dim=-1)


def inverse_fisher_residual(
    mean_map: MeanMap,
    states: torch.Tensor,
    residual: torch.Tensor,
    config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    *,
    parameterized_mean_map: ParameterizedMeanMap | None = None,
    mean_parameters: tuple[torch.Tensor, ...] = (),
) -> ResidualSolution:
    """Solve ``(JᵀJ + λI)u = (1 + λ)v`` by differentiable unrolled CG.

    Each leading position is an independent solve. Convergence is measured
    against the true residual before return. No detached metric, inverse,
    Euclidean fallback, or cross-call cache enters the update graph.

    For learned coefficients, supply ``parameterized_mean_map`` and the
    ordered ``mean_parameters``. That functional map is authoritative for
    both forward and backward, including mixed derivatives at exactly zero.
    The plain mean map suffices for parameter-free laws.
    """
    if mean_parameters and parameterized_mean_map is None:
        raise ValueError("mean_parameters require a parameterized_mean_map")
    if parameterized_mean_map is not None:

        def resolved_mean(state: torch.Tensor) -> torch.Tensor:
            return parameterized_mean_map(state, mean_parameters)

        mean_map = resolved_mean

    states, residual = numerical_pair(states, residual)
    with torch.autocast(device_type=states.device.type, enabled=False):
        rhs = (1 + config.damping) * residual
        tolerance = config.absolute_tolerance + config.relative_tolerance * _norm(rhs)
        update = residual * 0
        compensation = torch.zeros_like(update)
        remainder = rhs
        direction = rhs
        squared = remainder.square().sum(dim=-1)
        counts = torch.zeros_like(squared, dtype=torch.int64)
        active = _norm(remainder) > tolerance
        for _ in range(config.max_iterations):
            if not bool(active.any()):
                break
            applied = fisher_action(mean_map, states, direction) + config.damping * direction
            denominator = (direction * applied).sum(dim=-1)
            finite_tensor(denominator, "cg_denominator")
            if bool((active & (denominator <= 0)).any()):
                raise FisherNumericalError("nonpositive_curvature", "conjugate gradient")
            safe_denominator = torch.where(active, denominator, torch.ones_like(denominator))
            alpha = torch.where(active, squared / safe_denominator, torch.zeros_like(squared))
            increment = torch.where(
                active.unsqueeze(-1),
                alpha.unsqueeze(-1) * direction - compensation,
                torch.zeros_like(direction),
            )
            next_update = update + increment
            compensation = (next_update - update) - increment
            update = next_update
            remainder = remainder - alpha.unsqueeze(-1) * applied
            # FP32 recurrence can understate the equation residual. A candidate
            # stop must measure it; restart only affected rows, in this budget.
            candidate = active & (_norm(remainder) <= tolerance)
            if bool(candidate.any()):
                measured = rhs - fisher_action(mean_map, states, update) - config.damping * update
                remainder = torch.where(candidate.unsqueeze(-1), measured, remainder)
            next_squared = remainder.square().sum(dim=-1)
            finite_tensor(next_squared, "cg_residual")
            counts = counts + active.to(torch.int64)
            next_active = active & (_norm(remainder) > tolerance)
            safe_squared = torch.where(active, squared, torch.ones_like(squared))
            beta = torch.where(
                next_active & ~candidate,
                next_squared / safe_squared,
                torch.zeros_like(squared),
            )
            direction = torch.where(
                next_active.unsqueeze(-1),
                remainder + beta.unsqueeze(-1) * direction,
                torch.zeros_like(direction),
            )
            squared, active = next_squared, next_active
        metric_update = fisher_action(mean_map, states, update)
        true_residual = _norm(rhs - metric_update - config.damping * update)
        finite_tensor(update, "update")
        finite_tensor(true_residual, "solver_residual")
        if bool((true_residual > tolerance).any()):
            raise FisherNumericalError("solver_nonconvergence", "true residual exceeds tolerance")
        zero = (residual == 0).all(dim=-1, keepdim=True)
        if bool(zero.any()):
            functional_mean = parameterized_mean_map or (lambda x, _params: mean_map(x))
            zero_rhs = torch.where(zero, residual, torch.zeros_like(residual))
            zero_update = cast(
                torch.Tensor,
                _ZeroInverse.apply(functional_mean, config, states, zero_rhs, *mean_parameters),
            )
            update = torch.where(zero, zero_update, update)
        diagnostics = SolverDiagnostics(
            iterations=counts.detach(),
            residual_norm=true_residual.detach(),
            tolerance=tolerance.detach(),
            metric_action_norm=_norm(metric_update).detach(),
            damping_action_norm=(config.damping * _norm(update)).detach(),
            update_norm=_norm(update).detach(),
        )
        return ResidualSolution(update, diagnostics)


def euclidean_residual(
    mean_map: MeanMap,
    states: torch.Tensor,
    residual: torch.Tensor,
    config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    *,
    parameterized_mean_map: ParameterizedMeanMap | None = None,
    mean_parameters: tuple[torch.Tensor, ...] = (),
) -> ResidualSolution:
    """Same consumer contract with ``G=I``; the exact update is ``v``."""
    states, validated = numerical_pair(states, residual)
    del mean_map, parameterized_mean_map, mean_parameters
    norm = _norm(validated).detach()
    diagnostics = SolverDiagnostics(
        iterations=torch.zeros_like(norm, dtype=torch.int64),
        residual_norm=torch.zeros_like(norm),
        tolerance=config.absolute_tolerance
        + config.relative_tolerance * (1 + config.damping) * norm,
        metric_action_norm=norm,
        damping_action_norm=config.damping * norm,
        update_norm=norm,
    )
    return ResidualSolution(validated, diagnostics)
