"""Directional/log-magnitude density, sampling, and active-law contracts."""

import math
from dataclasses import replace

import pytest
import torch

from entropic_substrate.predictive_state.metric import FisherNumericalError
from entropic_substrate.sentence_prediction.law import DirectionalMagnitudeLaw, LawSpec

pytestmark = pytest.mark.unit


def test_spec_derives_the_directional_fisher_factor() -> None:
    spec = LawSpec(8, "sphere")
    assert spec.concentration == pytest.approx(63.0)
    assert spec.fisher_factor == pytest.approx(3969 / 131)
    assert spec.identity != replace(spec, mode="magnitude").identity


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 1},
        {"width": True},
        {"mode": "unknown"},
        {"mean_resultant": 0.0},
        {"mean_resultant": 1.0},
        {"mean_resultant": float("nan")},
        {"log_radius_std": 0.0},
        {"log_radius_std": float("inf")},
        {"width": 2, "mean_resultant": 0.1},
    ],
)
def test_invalid_or_nonregular_law_is_refused(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(LawSpec(4, "sphere"), **changes)


def test_spherical_law_has_no_active_radius() -> None:
    law = DirectionalMagnitudeLaw.from_vector(LawSpec(2, "sphere"), torch.tensor([3.0, 4.0]))
    assert law.log_radius_mean is None
    with pytest.raises(ValueError, match="inactive"):
        DirectionalMagnitudeLaw(law.spec, law.direction, torch.tensor(1.0))
    with pytest.raises(ValueError, match="unit"):
        law.log_prob(torch.tensor([3.0, 4.0]))


def test_raw_density_includes_cartesian_change_of_measure() -> None:
    spec = LawSpec(2, "magnitude", log_radius_std=1.0)
    base = torch.tensor([1.0, 0.0], dtype=torch.float64)
    law = DirectionalMagnitudeLaw.from_vector(spec, base)
    difference = law.log_prob(math.e * base) - law.log_prob(base)
    # -rho²/(2 sigma²) - d*rho, with rho=1; direction is unchanged.
    torch.testing.assert_close(difference, torch.tensor(-2.5, dtype=torch.float64))


@pytest.mark.parametrize("mode", ["sphere", "magnitude"])
def test_sampling_support_shape_and_reparameterization(mode: str) -> None:
    torch.manual_seed(42)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    law = DirectionalMagnitudeLaw.from_vector(LawSpec(4, mode), vector)
    samples = law.rsample((512,))
    assert samples.shape == (512, 4)
    assert bool(torch.isfinite(samples).all())
    if mode == "sphere":
        torch.testing.assert_close(samples.norm(dim=-1), torch.ones(512), atol=2e-6, rtol=2e-6)
    else:
        assert float(samples.detach().norm(dim=-1).log().std()) == pytest.approx(0.1, abs=0.02)
    samples.sum().backward()
    assert vector.grad is not None and bool(torch.isfinite(vector.grad).all())
    assert float(vector.grad.norm()) > 0
    assert bool(torch.isfinite(law.log_prob(samples.detach())).all())


def test_same_rng_state_reproduces_sampling() -> None:
    law = DirectionalMagnitudeLaw.from_vector(
        LawSpec(3, "magnitude"), torch.tensor([1.0, 2.0, 3.0])
    )
    state = torch.random.get_rng_state()
    first = law.rsample((5,))
    torch.random.set_rng_state(state)
    torch.testing.assert_close(first, law.rsample((5,)), atol=0, rtol=0)


def test_antipodal_zero_density_is_not_replaced_with_an_epsilon() -> None:
    vector = torch.tensor([1.0, 0.0, 0.0])
    law = DirectionalMagnitudeLaw.from_vector(LawSpec(3, "sphere"), vector)
    assert bool(torch.isneginf(law.log_prob(-vector)))
    nearby = torch.tensor([-1.0, 1e-10, 0.0])
    assert bool(torch.isfinite(law.log_prob(nearby)))


def test_log_density_retains_parameter_derivatives() -> None:
    vector = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64, requires_grad=True)
    observation = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)

    def score(x: torch.Tensor) -> torch.Tensor:
        return DirectionalMagnitudeLaw.from_vector(LawSpec(3, "magnitude"), x).log_prob(observation)

    assert torch.autograd.gradcheck(score, (vector,))
    assert torch.autograd.gradgradcheck(score, (vector,))


def test_law_precision_is_not_lowered_by_autocast() -> None:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        law = DirectionalMagnitudeLaw.from_vector(
            LawSpec(3, "sphere"), torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        )
        sample = law.rsample()
        score = law.log_prob(sample)
    assert sample.dtype == score.dtype == torch.float32


@pytest.mark.parametrize("shape", [(0,), (-1,), (True,), [2]])
def test_sample_shape_is_explicit_and_valid(shape: object) -> None:
    law = DirectionalMagnitudeLaw.from_vector(LawSpec(2, "sphere"), torch.ones(2))
    with pytest.raises(ValueError):
        law.rsample(shape)


def test_degenerate_sampling_noise_is_not_reported_as_latent_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    law = DirectionalMagnitudeLaw.from_vector(LawSpec(2, "sphere"), torch.tensor([1.0, 0.0]))
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: law.direction.clone())
    with pytest.raises(FisherNumericalError, match="tangent_noise"):
        law.rsample()


def test_law_refuses_incompatible_batch_shapes() -> None:
    law = DirectionalMagnitudeLaw.from_vector(LawSpec(2, "sphere"), torch.eye(2))
    observations = torch.tensor([[1.0, 0.0]]).repeat(3, 1)
    with pytest.raises(ValueError, match="batch"):
        law.log_prob(observations)


def test_missing_spec_and_nontensor_radius_are_explicit_errors() -> None:
    with pytest.raises(TypeError, match="LawSpec"):
        DirectionalMagnitudeLaw.from_vector(None, torch.ones(2))
    with pytest.raises(ValueError, match="scalar"):
        DirectionalMagnitudeLaw(LawSpec(2, "magnitude"), torch.tensor([1.0, 0.0]), 1.0)


def test_unrepresentable_concentration_is_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        LawSpec(10**400, "sphere")


@pytest.mark.parametrize("width", [2, 3, 8])
def test_surface_density_matches_independent_beta_marginal(width: int) -> None:
    spec = LawSpec(width, "sphere")
    pole = torch.zeros(width, dtype=torch.float64)
    pole[0] = 1
    law = DirectionalMagnitudeLaw.from_vector(spec, pole)
    t = torch.tensor([-0.8, -0.3, 0.0, 0.4, 0.8], dtype=torch.float64)
    points = torch.zeros(len(t), width, dtype=torch.float64)
    points[:, 0], points[:, 1] = t, (1 - t.square()).sqrt()
    beta = (width - 1) / 2
    alpha = beta + spec.concentration
    marginal = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float64), torch.tensor(beta, dtype=torch.float64)
    )
    expected = marginal.log_prob((t + 1) / 2) - math.log(2)
    shell_area = math.log(2) + beta * math.log(math.pi) - math.lgamma(beta)
    actual = law.log_prob(points) + shell_area + ((width - 3) / 2) * (1 - t.square()).log()
    torch.testing.assert_close(actual, expected)


def test_directional_fisher_factor_matches_sampled_score_second_moment() -> None:
    torch.manual_seed(42)
    spec = LawSpec(8, "sphere")
    pole = torch.zeros(8, dtype=torch.float64)
    pole[0] = 1
    sample = DirectionalMagnitudeLaw.from_vector(spec, pole).rsample((32768,))
    score = spec.concentration * sample[:, 1] / (1 + sample[:, 0])
    second_moment = score.square()
    standard_error = second_moment.std() / math.sqrt(len(sample))
    assert abs(float(second_moment.mean()) - spec.fisher_factor) < 5 * float(standard_error)


def test_sampling_derivatives_match_fixed_noise_finite_differences() -> None:
    vector = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64, requires_grad=True)

    def draw(x: torch.Tensor) -> torch.Tensor:
        with torch.random.fork_rng():
            torch.manual_seed(3)
            return DirectionalMagnitudeLaw.from_vector(LawSpec(3, "magnitude"), x).rsample((2,))

    assert torch.autograd.gradcheck(draw, (vector,))
    assert torch.autograd.gradgradcheck(draw, (vector,))


def test_sample_underflow_is_a_numerical_defect_not_a_zero_latent() -> None:
    law = DirectionalMagnitudeLaw(
        LawSpec(2, "magnitude"), torch.tensor([1.0, 0.0]), torch.tensor(-1000.0)
    )
    with pytest.raises(FisherNumericalError, match="sample_underflow"):
        law.rsample()


def test_equal_numeric_scales_share_one_law_identity() -> None:
    assert (
        LawSpec(2, "magnitude", log_radius_std=1).identity
        == LawSpec(2, "magnitude", log_radius_std=1.0).identity
    )
    assert (
        LawSpec(2, "sphere", log_radius_std=1).identity
        == LawSpec(2, "sphere", log_radius_std=2).identity
    )


@pytest.mark.parametrize(
    "radius,message",
    [(torch.ones(2), "aligned"), (torch.tensor(1.0, dtype=torch.float64), "dtype/device")],
)
def test_radius_tensor_shape_and_precision_are_validated(
    radius: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DirectionalMagnitudeLaw(LawSpec(2, "magnitude"), torch.tensor([1.0, 0.0]), radius)


def test_partial_radial_underflow_is_not_hidden_by_a_surviving_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction = torch.tensor([1.0, 1e-8])
    radius = torch.tensor(-90.0).exp()
    assert float(radius) > 0
    assert float((radius * direction)[0]) > 0
    assert float((radius * direction)[1]) == 0

    def fixed_beta(self: torch.distributions.Beta, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.ones(shape)

    monkeypatch.setattr(torch.distributions.Beta, "sample", fixed_beta)

    def fixed_noise(shape: tuple[int, ...], **kwargs: object) -> torch.Tensor:
        return torch.tensor([0.0, 1.0], **kwargs) if shape == (2,) else torch.zeros(shape, **kwargs)

    monkeypatch.setattr(torch, "randn", fixed_noise)
    law = DirectionalMagnitudeLaw(LawSpec(2, "magnitude"), direction, torch.tensor(-90.0))
    with pytest.raises(FisherNumericalError, match="sample_underflow"):
        law.rsample()
