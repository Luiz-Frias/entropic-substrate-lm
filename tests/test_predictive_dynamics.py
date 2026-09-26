"""Inverse-Fisher residual solve and full-differentiation contracts."""

import pytest
import torch

from entropic_substrate.core.latent.dynamics import (
    SolverConfig,
    euclidean_residual,
    inverse_fisher_residual,
)
from entropic_substrate.core.latent.metric import FisherNumericalError

pytestmark = pytest.mark.unit


def test_nonlinear_solve_matches_dense_update_and_all_derivatives() -> None:
    torch.manual_seed(12)
    states = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    residual = torch.randn_like(states, requires_grad=True)
    weights = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)

    def mean(x: torch.Tensor) -> torch.Tensor:
        return (weights @ (x * (x.square().mean() + 0.2).rsqrt())).tanh()

    config = SolverConfig(relative_tolerance=1e-12, absolute_tolerance=1e-14)
    solved = inverse_fisher_residual(mean, states, residual, config)
    jac = torch.vmap(torch.func.jacrev(mean))(states)
    matrix = jac.transpose(-1, -2) @ jac + torch.eye(3, dtype=torch.float64)
    oracle = torch.linalg.solve(matrix, (2 * residual).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(solved.update, oracle)
    actual_grad = torch.autograd.grad(solved.update.square().sum(), (states, residual, weights))
    oracle_grad = torch.autograd.grad(oracle.square().sum(), (states, residual, weights))
    for actual, expected in zip(actual_grad, oracle_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
        assert actual.abs().sum() > 0
    assert bool((solved.diagnostics.residual_norm <= solved.diagnostics.tolerance).all())


def test_zero_rhs_and_mixed_zero_rows_are_valid() -> None:
    states = torch.ones(2, 2)
    residual = torch.tensor([[0.0, 0.0], [1.0, 2.0]], requires_grad=True)
    result = inverse_fisher_residual(lambda x: 2 * x, states, residual)
    torch.testing.assert_close(result.update, 0.4 * residual)
    assert result.diagnostics.iterations.tolist() == [0, 1]
    assert torch.isfinite(torch.autograd.grad(result.update.sum(), residual)[0]).all()
    zero = inverse_fisher_residual(lambda x: x, states, torch.zeros_like(states))
    assert zero.diagnostics.iterations.tolist() == [0, 0]


def test_euclidean_returns_original_residual_and_identity_fisher_agrees() -> None:
    residual = torch.randn(2, 3, 4)
    states = torch.randn_like(residual)
    assert euclidean_residual(lambda x: x, states, residual).update is residual
    fisher = inverse_fisher_residual(lambda x: x, states, residual)
    torch.testing.assert_close(fisher.update, residual)


def test_iteration_exhaustion_is_structured_defect() -> None:
    diagonal = torch.tensor([1.0, 3.0, 9.0])
    with pytest.raises(FisherNumericalError, match="solver_nonconvergence"):
        inverse_fisher_residual(
            lambda x: diagonal * x,
            torch.ones(1, 3),
            torch.ones(1, 3),
            SolverConfig(max_iterations=1),
        )


def test_conditioned_fp32_law_still_refuses_unmet_true_residual() -> None:
    torch.manual_seed(0)
    states = torch.randn(2, 8, 8) * 0.02
    residual = torch.randn_like(states) * 0.005
    weights = torch.randn(8, 8) * 0.2

    def mean(point: torch.Tensor) -> torch.Tensor:
        return torch.tanh(weights @ (point * (point.square().mean() + 1e-5).rsqrt()))

    # Measured fixture condition max 3987.57; even the dense FP64 solve of its
    # FP32 Jacobian, returned to FP32, misses the matrix-free residual tolerance.
    # Replacement/refinement must not promote this bounded failure to success.
    with pytest.raises(FisherNumericalError, match="solver_nonconvergence"):
        inverse_fisher_residual(mean, states, residual)


@pytest.mark.parametrize("damping", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_damping_refused(damping: float) -> None:
    with pytest.raises(ValueError, match="damping"):
        SolverConfig(damping=damping)


def test_full_solve_passes_finite_difference_gradcheck() -> None:
    states = torch.tensor([[0.3, 0.8]], dtype=torch.float64, requires_grad=True)
    residual = torch.tensor([[0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[1.2, 0.1], [0.2, 0.7]], dtype=torch.float64, requires_grad=True)

    def solve(h: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return inverse_fisher_residual(lambda x: torch.sin(w @ x), h, v).update

    assert torch.autograd.gradcheck(solve, (states, residual, weights))

    def detached_metric_mutant(h: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return inverse_fisher_residual(lambda x: torch.sin(w.detach() @ x), h.detach(), v).update

    with pytest.raises(RuntimeError, match="Jacobian mismatch"):
        torch.autograd.gradcheck(detached_metric_mutant, (states, residual, weights))


def test_nonfinite_direction_and_predictor_fail_without_fallback() -> None:
    states = torch.ones(1, 2)
    with pytest.raises(FisherNumericalError, match="nonfinite_vector"):
        inverse_fisher_residual(lambda x: x, states, states * float("inf"))
    with pytest.raises(FisherNumericalError, match="nonfinite_mean"):
        inverse_fisher_residual(lambda x: x * float("nan"), states, states)


def test_zero_metric_and_bfloat_autocast() -> None:
    states = torch.ones(2, 3, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        solved = inverse_fisher_residual(lambda x: 0 * x, states, states)
    assert solved.update.dtype == torch.float32
    torch.testing.assert_close(solved.update, torch.full((2, 3), 2.0))
    assert solved.diagnostics.metric_action_norm.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("mixed", [False, True])
def test_zero_rhs_has_inverse_tangent_and_coefficient_crossgradients(mixed: bool) -> None:
    states = torch.tensor([[0.3, 0.8], [0.2, 0.6]], dtype=torch.float64, requires_grad=True)
    residual = torch.zeros_like(states, requires_grad=True)
    if mixed:
        with torch.no_grad():
            residual[1] = torch.tensor([0.2, -0.1])
    weights = torch.tensor([[1.2, 0.1], [0.2, 0.7]], dtype=torch.float64, requires_grad=True)

    def solve(h: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return inverse_fisher_residual(
            lambda x: torch.sin(w @ x),
            h,
            v,
            parameterized_mean_map=lambda x, p: torch.sin(p[0] @ x),
            mean_parameters=(w,),
        ).update

    assert torch.autograd.gradcheck(solve, (states, residual, weights))
    assert torch.autograd.gradgradcheck(solve, (states, residual, weights))
    actual = solve(states, residual, weights)
    jac = torch.vmap(torch.func.jacrev(lambda x: torch.sin(weights @ x)))(states)
    matrix = jac.transpose(-1, -2) @ jac + torch.eye(2, dtype=torch.float64)
    oracle = torch.linalg.solve(matrix, (2 * residual).unsqueeze(-1)).squeeze(-1)
    actual_first = torch.autograd.grad(actual.sum(), (states, residual, weights), create_graph=True)
    oracle_first = torch.autograd.grad(oracle.sum(), (states, residual, weights), create_graph=True)
    for index in (0, 2):
        actual_grad, oracle_grad = actual_first[index], oracle_first[index]
        actual_cross = torch.autograd.grad(actual_grad.sum(), residual, retain_graph=True)[0]
        oracle_cross = torch.autograd.grad(oracle_grad.sum(), residual, retain_graph=True)[0]
        torch.testing.assert_close(actual_cross, oracle_cross)


def test_euclidean_and_fisher_controls_share_the_numerical_dtype() -> None:
    state = torch.ones(2, 3, dtype=torch.bfloat16)
    residual = torch.full_like(state, 0.25)
    euclidean = euclidean_residual(lambda x: x, state, residual)
    fisher = inverse_fisher_residual(lambda x: x, state, residual)
    assert euclidean.update.dtype == fisher.update.dtype == torch.float32
    torch.testing.assert_close(euclidean.update, fisher.update)
