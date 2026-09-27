"""A future law reads the carrier; its Fisher factors are not arbitrary scales."""

import math

import pytest
import torch

from entropic_substrate.core.latent.metric import fisher_action
from entropic_substrate.core.sentence.head import SentenceLawHead
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.numerics import UndefinedDirection

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["sphere", "magnitude"])
def test_head_parameters_and_features_describe_the_same_effective_law(mode: str) -> None:
    torch.manual_seed(42)
    head = SentenceLawHead(4, LawSpec(8, mode)).double()
    states = torch.randn(2, 3, 4, dtype=torch.float64)
    original = states.clone()
    law = head(states)
    features = head.features_with_parameters(states, head.metric_parameters())
    torch.testing.assert_close(
        features[..., :8], law.direction * math.sqrt(head.spec.fisher_factor)
    )
    torch.testing.assert_close(states, original, atol=0, rtol=0)
    if mode == "sphere":
        assert law.log_radius_mean is None
        assert features.shape == (2, 3, 8)
        assert len(head.metric_parameters()) == 2
        assert not any("radius" in name for name, _ in head.named_parameters())
    else:
        assert features.shape == (2, 3, 9)
        assert len(head.metric_parameters()) == 4
        torch.testing.assert_close(
            features[..., -1], law.log_radius_mean / head.spec.log_radius_std
        )


@pytest.mark.parametrize("mode", ["sphere", "magnitude"])
def test_matrix_free_action_matches_separately_weighted_density_metric(mode: str) -> None:
    torch.manual_seed(17)
    head = SentenceLawHead(4, LawSpec(8, mode)).double()
    state = torch.randn(4, dtype=torch.float64, requires_grad=True)
    vector = torch.randn_like(state)
    parameters = head.metric_parameters()

    def feature_map(h: torch.Tensor) -> torch.Tensor:
        return head.features_with_parameters(h, parameters)

    def direction(h: torch.Tensor) -> torch.Tensor:
        return head.parameters_with_values(h, parameters)[0]

    j = torch.func.jacrev(direction)(state)
    dense = head.spec.fisher_factor * j.T @ j
    if mode == "magnitude":

        def radius(h: torch.Tensor) -> torch.Tensor:
            value = head.parameters_with_values(h, parameters)[1]
            assert value is not None
            return value

        g = torch.func.jacrev(radius)(state)
        dense = dense + torch.outer(g, g) / head.spec.log_radius_std**2
    actual = fisher_action(feature_map, state, vector)
    torch.testing.assert_close(actual, dense @ vector, atol=1e-10, rtol=1e-10)
    gradients = torch.autograd.grad(actual.square().sum(), (state, *parameters), allow_unused=True)
    # The scalar mean bias does not change a fixed-variance Fisher metric.
    required = gradients[:-1] if mode == "magnitude" else gradients
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in required)
    assert required[0].abs().sum() > 0


def test_head_is_trainable_through_observable_reparameterized_samples() -> None:
    torch.manual_seed(42)
    head = SentenceLawHead(4, LawSpec(8, "magnitude"))
    state = torch.randn(2, 4, requires_grad=True)
    head(state).rsample().square().sum().backward()
    assert state.grad is not None and bool(torch.isfinite(state.grad).all())
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())


@pytest.mark.parametrize("mode", ["sphere", "magnitude"])
def test_fisher_pullback_matches_observed_local_kl_curvature(mode: str) -> None:
    torch.manual_seed(52)
    head = SentenceLawHead(3, LawSpec(4, mode)).double()
    state = torch.randn(3, dtype=torch.float64)
    vector = torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)
    base = head(state)
    samples = base.rsample((32768,)).detach()
    step = 1e-3
    plus, minus = head(state + step * vector), head(state - step * vector)
    # Symmetric perturbations cancel the noisy first-order score term.
    curvature = (
        2 * base.log_prob(samples) - plus.log_prob(samples) - minus.log_prob(samples)
    ) / step**2
    action = fisher_action(
        lambda x: head.features_with_parameters(x, head.metric_parameters()), state, vector
    )
    expected = (vector * action).sum()
    standard_error = curvature.std() / math.sqrt(curvature.numel())
    assert abs(curvature.mean() - expected) < 6 * standard_error + 1e-4


def test_density_parameters_own_an_fp32_island() -> None:
    head = SentenceLawHead(4, LawSpec(8, "magnitude"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        law = head(torch.ones(2, 4, dtype=torch.bfloat16))
    assert law.direction.dtype == torch.float32
    assert law.log_radius_mean.dtype == torch.float32


def test_zero_mean_is_an_explicit_gap_not_an_invented_direction() -> None:
    head = SentenceLawHead(4, LawSpec(8, "sphere"))
    with torch.no_grad():
        head.direction.weight.zero_()
        head.direction.bias.zero_()
    with pytest.raises(UndefinedDirection) as caught:
        head(torch.zeros(2, 4))
    assert caught.value.positions == 2


@pytest.mark.parametrize("width", [0, 1, True, 2.5])
def test_invalid_head_width_is_rejected(width: int) -> None:
    with pytest.raises(ValueError):
        SentenceLawHead(width, LawSpec(8, "sphere"))


def test_malformed_state_and_parameter_roster_fail_loudly() -> None:
    head = SentenceLawHead(4, LawSpec(8, "sphere"))
    with pytest.raises(ValueError, match="width"):
        head.parameters_with_values(torch.tensor(1.0), head.metric_parameters())
    with pytest.raises(TypeError, match="floating"):
        head.parameters_with_values(torch.ones(4, dtype=torch.int64), head.metric_parameters())
    with pytest.raises(ValueError, match="width"):
        head(torch.ones(2, 5))
    with pytest.raises(ValueError, match="parameter"):
        head.features_with_parameters(torch.ones(4), ())
    with pytest.raises(ValueError, match="nonfinite"):
        head(torch.full((4,), float("nan")))
