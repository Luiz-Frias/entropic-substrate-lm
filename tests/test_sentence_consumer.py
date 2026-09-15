"""Law-correct consumers retain full derivatives and independent carrier modes."""

from dataclasses import replace

import pytest
import torch

from entropic_substrate.predictive_state.dynamics import SolverConfig
from entropic_substrate.predictive_state.metric import FisherNumericalError
from entropic_substrate.sentence_prediction.consumer import SentenceConsumer
from entropic_substrate.sentence_prediction.head import SentenceLawHead
from entropic_substrate.sentence_prediction.law import LawSpec
from entropic_substrate.sentence_prediction.numerics import coordinate_map

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("carrier", ["sphere", "magnitude"])
@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_law_consumer_matches_dense_value_and_all_active_derivatives(
    carrier: str, interface: str
) -> None:
    torch.manual_seed(42)
    head = SentenceLawHead(3, LawSpec(4, interface)).double()
    current = torch.randn(2, 3, dtype=torch.float64)
    proposed = current + 0.1 * torch.randn_like(current)
    current.requires_grad_()
    proposed.requires_grad_()
    source, proposal_source = current, proposed
    if carrier == "sphere":
        # Differentiate on the declared manifold, through the same retraction
        # that produces the spine's unit carriers, not off-sphere leaf inputs.
        current = coordinate_map(current).direction
        proposed = coordinate_map(proposed).direction
    config = SolverConfig(relative_tolerance=1e-11, absolute_tolerance=1e-13)
    consumer = SentenceConsumer(head, carrier, "fisher", config)
    actual = consumer(current, proposed)
    parameters = head.metric_parameters()

    def features(x: torch.Tensor) -> torch.Tensor:
        if carrier == "sphere":
            x = coordinate_map(x).direction
        return head.features_with_parameters(x, parameters)

    jac = torch.vmap(torch.func.jacrev(features))(current)
    metric = jac.transpose(-1, -2) @ jac
    delta = proposed - current
    radial = (current * delta).sum(dim=-1, keepdim=True)
    rhs = delta - radial * current if carrier == "sphere" else delta
    update = torch.linalg.solve(metric + torch.eye(3, dtype=torch.float64), 2 * rhs.unsqueeze(-1))
    expected = current + update.squeeze(-1)
    if carrier == "sphere":
        expected = coordinate_map(expected + radial * current).direction
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    # A non-radial observable avoids the constant squared norm of unit states.
    weights = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
    inputs = (source, proposal_source, *parameters)
    actual_grads = torch.autograd.grad(
        (actual * weights).sum(), inputs, allow_unused=True, retain_graph=True
    )
    oracle_grads = torch.autograd.grad((expected * weights).sum(), inputs, allow_unused=True)
    for actual_grad, oracle_grad in zip(actual_grads, oracle_grads, strict=True):
        if oracle_grad is None:
            assert actual_grad is None or bool((actual_grad == 0).all())
        else:
            assert actual_grad is not None
            torch.testing.assert_close(actual_grad, oracle_grad, atol=1e-8, rtol=1e-7)
    assert consumer.diagnostics is not None
    assert bool((consumer.diagnostics.residual_norm <= consumer.diagnostics.tolerance).all())
    assert not consumer.diagnostics.update_norm.requires_grad


@pytest.mark.parametrize("carrier", ["sphere", "magnitude"])
def test_isotropic_substitution_reproduces_original_proposal(carrier: str) -> None:
    torch.manual_seed(42)
    current, proposed = torch.randn(2, 4), torch.randn(2, 4)
    if carrier == "sphere":
        current, proposed = coordinate_map(current).direction, coordinate_map(proposed).direction
    consumer = SentenceConsumer(SentenceLawHead(4, LawSpec(8, "sphere")), carrier, "isotropic")
    torch.testing.assert_close(consumer(current, proposed), proposed, atol=0, rtol=0)
    assert consumer.diagnostics.iterations.tolist() == [0, 0]


def test_zero_residual_retains_mixed_metric_derivatives() -> None:
    torch.manual_seed(3)
    head = SentenceLawHead(2, LawSpec(3, "sphere")).double()
    current = torch.tensor([0.3, 0.8], dtype=torch.float64, requires_grad=True)
    proposal = current.detach().clone().requires_grad_()
    result = SentenceConsumer(head, "magnitude", "fisher")(current, proposal)
    torch.testing.assert_close(result, current, atol=0, rtol=0)
    first = torch.autograd.grad(result.sum(), proposal, create_graph=True)[0]
    mixed = torch.autograd.grad(first.sum(), head.direction.weight)[0]
    assert bool(torch.isfinite(mixed).all()) and mixed.abs().sum() > 0


@pytest.mark.parametrize("carrier", ["sphere", "magnitude"])
def test_isotropic_metric_inside_real_solver_reproduces_control(
    carrier: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def identity_features(
        self: SentenceLawHead, state: torch.Tensor, parameters: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return state

    monkeypatch.setattr(SentenceLawHead, "features_with_parameters", identity_features)
    torch.manual_seed(9)
    current, proposed = torch.randn(2, 4), torch.randn(2, 4)
    if carrier == "sphere":
        current, proposed = coordinate_map(current).direction, coordinate_map(proposed).direction
    consumer = SentenceConsumer(SentenceLawHead(4, LawSpec(8, "sphere")), carrier, "fisher")
    torch.testing.assert_close(consumer(current, proposed), proposed, atol=1e-6, rtol=1e-6)


def test_metric_detachment_breaks_the_derivative_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    original = SentenceLawHead.features_with_parameters

    def detached(
        self: SentenceLawHead, state: torch.Tensor, parameters: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return original(self, state, tuple(p.detach() for p in parameters))

    # Detach the production metric only; restore it for the independent oracle.
    torch.manual_seed(8)
    head = SentenceLawHead(3, LawSpec(4, "sphere")).double()
    state = torch.randn(3, dtype=torch.float64, requires_grad=True)
    proposal = torch.randn_like(state, requires_grad=True)
    with monkeypatch.context() as patch:
        patch.setattr(SentenceLawHead, "features_with_parameters", detached)
        broken = SentenceConsumer(head, "magnitude", "fisher")(state, proposal)
    correct = SentenceConsumer(head, "magnitude", "fisher")(state, proposal)
    broken_gradient = torch.autograd.grad(broken.sum(), head.direction.weight, allow_unused=True)[0]
    correct_gradient = torch.autograd.grad(correct.sum(), head.direction.weight)[0]
    assert broken_gradient is None
    assert correct_gradient.abs().sum() > 0


def test_failed_solve_clears_old_diagnostics_and_never_falls_back() -> None:
    torch.manual_seed(7)
    head = SentenceLawHead(4, LawSpec(8, "magnitude"))
    consumer = SentenceConsumer(head, "magnitude", "fisher")
    state, proposal = torch.randn(2, 4), torch.randn(2, 4)
    consumer(state, proposal)
    assert consumer.diagnostics is not None
    consumer.config = replace(consumer.config, max_iterations=1)
    with pytest.raises(FisherNumericalError, match="solver_nonconvergence"):
        consumer(state, proposal)
    assert consumer.diagnostics is None


def test_consumer_promotes_math_inside_ambient_autocast() -> None:
    consumer = SentenceConsumer(SentenceLawHead(4, LawSpec(8, "sphere")), "magnitude", "fisher")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = consumer(torch.ones(2, 4), torch.ones(2, 4) * 1.1)
    assert result.dtype == torch.float32


def test_invalid_modes_and_nonunit_spherical_proposals_are_rejected() -> None:
    head = SentenceLawHead(4, LawSpec(8, "sphere"))
    with pytest.raises(ValueError, match="consumer"):
        SentenceConsumer(head, "magnitude", "unknown")
    with pytest.raises(ValueError, match="carrier"):
        SentenceConsumer(head, "unknown", "fisher")
    with pytest.raises(ValueError, match="unit"):
        SentenceConsumer(head, "sphere", "isotropic")(torch.ones(4), torch.ones(4))
