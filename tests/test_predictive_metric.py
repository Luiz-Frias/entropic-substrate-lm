"""Matrix-free Gaussian Fisher pullback contract."""

import pytest
import torch

from entropic_substrate.core.latent.metric import FisherNumericalError, fisher_action

pytestmark = pytest.mark.unit


def test_action_matches_dense_pointwise_oracle_and_retains_derivatives() -> None:
    torch.manual_seed(3)
    states = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    vector = torch.randn_like(states, requires_grad=True)
    weights = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)

    def mean(x: torch.Tensor) -> torch.Tensor:
        normalized = x * (x.square().mean() + 0.1).rsqrt()
        return torch.tanh(weights @ normalized)

    actual = fisher_action(mean, states, vector)
    jac = torch.vmap(torch.func.jacrev(mean))(states.reshape(-1, 4))
    expected = torch.einsum("noi,noj,nj->ni", jac, jac, vector.reshape(-1, 4))
    torch.testing.assert_close(actual, expected.reshape_as(states))
    gradients = torch.autograd.grad(actual.square().sum(), (states, vector, weights))
    expected_gradients = torch.autograd.grad(expected.square().sum(), (states, vector, weights))
    for gradient, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, reference)
        assert gradient.abs().sum() > 0


def test_action_rebuilds_after_weight_update() -> None:
    weights = torch.eye(2, requires_grad=True)
    states = torch.ones(1, 2)
    vector = torch.ones_like(states)
    before = fisher_action(lambda x: weights @ x, states, vector)
    before.sum().backward()
    with torch.no_grad():
        weights.mul_(2)
    after = fisher_action(lambda x: weights @ x, states, vector)
    after.sum().backward()
    torch.testing.assert_close(after, 4 * before.detach())


def test_bfloat_input_uses_fp32_even_inside_autocast() -> None:
    states = torch.ones(2, 3, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fisher_action(lambda x: x.square(), states, states)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.full((2, 3), 4.0))


@pytest.mark.parametrize("shape", [(0, 3), ()])
def test_empty_or_scalar_state_refused(shape: tuple[int, ...]) -> None:
    values = torch.empty(shape)
    with pytest.raises(FisherNumericalError, match="invalid_shape"):
        fisher_action(lambda x: x, values, values)


def test_nonfinite_mean_and_shape_mismatch_refused() -> None:
    values = torch.ones(2, 3)
    with pytest.raises(FisherNumericalError, match="nonfinite_mean"):
        fisher_action(lambda x: x * float("nan"), values, values)
    with pytest.raises(FisherNumericalError, match="shape_mismatch"):
        fisher_action(lambda x: x, values, torch.ones(3))


def test_nonlinear_metric_detach_mutant_loses_state_gradient() -> None:
    states = torch.tensor([[0.3, 0.7]], requires_grad=True)
    vector = torch.ones_like(states)
    actual = fisher_action(torch.sin, states, vector)
    derivative = torch.autograd.grad(actual.sum(), states)[0]
    torch.testing.assert_close(derivative, -2 * states.sin() * states.cos())
    mutant = fisher_action(torch.sin, states.detach(), vector)
    assert not mutant.requires_grad
