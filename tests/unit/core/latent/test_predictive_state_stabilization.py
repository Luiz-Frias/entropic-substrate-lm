"""Online-latent VICReg stabilization contracts for D1."""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.core.latent.stabilization import (
    MomentCollapseRegularizer,
    NoCollapseRegularizer,
)
from entropic_substrate.core.latent.state import LatentTrajectory
from predictive_state_helpers import codec_identity

pytestmark = pytest.mark.unit


def _trajectory(values: torch.Tensor) -> LatentTrajectory:
    return LatentTrajectory(values=values, codec_identity=codec_identity())


def test_no_collapse_regularizer_is_an_exact_off_identity() -> None:
    outcome = NoCollapseRegularizer()(_trajectory(torch.randn(2, 4, 5, requires_grad=True)))
    assert outcome.total.item() == 0.0
    assert (
        outcome.raw_variance_penalty.item() == 0.0
        and outcome.weighted_variance_penalty.item() == 0.0
    )
    assert (
        outcome.raw_covariance_penalty.item() == 0.0
        and outcome.weighted_covariance_penalty.item() == 0.0
    )
    assert outcome.diagnostics is not None
    assert (
        outcome.diagnostics.variance.dtype == torch.float32
        and not outcome.diagnostics.variance.requires_grad
    )
    assert outcome.total.dtype == torch.float32


def test_moment_regularizer_reports_expected_collapse_ranks_and_covariance() -> None:
    regularizer = MomentCollapseRegularizer(
        gamma=0.5, variance_weight=0.7, covariance_weight=0.25, variance_eps=1e-4
    )
    constant = _trajectory(torch.ones(2, 4, 4))
    source = torch.arange(8, dtype=torch.float32).view(2, 4, 1)
    basis = torch.tensor([1.0, -2.0, 3.0, -4.0])
    near_constant = _trajectory(torch.ones(2, 4, 4) + 1e-6 * source * basis)
    rank_one = _trajectory(source * basis)
    duplicate = _trajectory(source.expand(-1, -1, 4).clone())
    full_rank = _trajectory(torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(43)))
    constant_outcome, near_outcome, rank_one_outcome, duplicate_outcome, full_outcome = map(
        regularizer, (constant, near_constant, rank_one, duplicate, full_rank)
    )
    assert (
        constant_outcome.diagnostics is not None
        and constant_outcome.diagnostics.numerical_rank == 0
    )
    assert near_outcome.diagnostics is not None and near_outcome.diagnostics.effective_rank < 1.1
    assert (
        rank_one_outcome.diagnostics is not None
        and rank_one_outcome.diagnostics.numerical_rank == 1
    )
    assert (
        duplicate_outcome.diagnostics is not None
        and duplicate_outcome.diagnostics.numerical_rank == 1
    )
    assert full_outcome.diagnostics is not None and full_outcome.diagnostics.numerical_rank == 4
    assert (
        full_outcome.total.isfinite()
        and constant_outcome.raw_variance_penalty > 0.0
        and rank_one_outcome.raw_covariance_penalty > 0.0
    )
    torch.testing.assert_close(
        NoCollapseRegularizer()(rank_one).diagnostics.covariance,
        rank_one_outcome.diagnostics.covariance,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        NoCollapseRegularizer()(rank_one).diagnostics.std,
        rank_one_outcome.diagnostics.std,
        atol=0.0,
        rtol=0.0,
    )


def test_moment_regularizer_matches_registered_vicreg_sample_std_and_off_diagonal_formula() -> None:
    values = _trajectory(torch.tensor([[[0.0, 0.0], [2.0, 0.0], [4.0, 2.0]]]))
    outcome = MomentCollapseRegularizer(
        gamma=1.5, variance_weight=0.4, covariance_weight=0.3, variance_eps=0.25
    )(values)
    flattened = values.values.reshape(3, 2)
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / 2.0
    std = (covariance.diagonal() + 0.25).sqrt()
    raw_variance = torch.relu(torch.tensor(1.5) - std).mean()
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    raw_covariance = off_diagonal.square().sum() / 2.0
    torch.testing.assert_close(outcome.diagnostics.covariance, covariance, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(outcome.diagnostics.std, std, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(outcome.raw_variance_penalty, raw_variance, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(outcome.raw_covariance_penalty, raw_covariance, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(
        outcome.weighted_variance_penalty, raw_variance * 0.4, atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(
        outcome.weighted_covariance_penalty, raw_covariance * 0.3, atol=1e-7, rtol=0.0
    )
    torch.testing.assert_close(
        outcome.total, raw_variance * 0.4 + raw_covariance * 0.3, atol=1e-7, rtol=0.0
    )


def test_moment_regularizer_operates_on_online_latent_with_finite_gradients() -> None:
    values = (1e-3 * torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)).requires_grad_()
    outcome = MomentCollapseRegularizer(
        gamma=1.0, variance_weight=1.0, covariance_weight=0.2, variance_eps=1e-4
    )(_trajectory(values))
    outcome.total.backward()
    assert (
        values.grad is not None
        and torch.isfinite(values.grad).all()
        and values.grad.abs().sum() > 0.0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.0, "variance_weight": 1.0, "covariance_weight": 0.1, "variance_eps": 1e-4},
        {"gamma": 0.1, "variance_weight": -0.1, "covariance_weight": 0.1, "variance_eps": 1e-4},
        {"gamma": 0.1, "variance_weight": 1.0, "covariance_weight": 0.1, "variance_eps": 0.0},
    ],
)
def test_moment_regularizer_rejects_invalid_constants(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        MomentCollapseRegularizer(**kwargs)
