"""Future-law grounding, baselines, and calibration contracts for D1."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from entropic_substrate.core.latent.future_law import (
    DiagonalGaussianFutureLaw,
    MeanDisplacementBaseline,
)
from entropic_substrate.core.latent.state import LatentTrajectory
from entropic_substrate.core.latent.target_values import FutureTarget
from entropic_substrate.core.observation.contracts import CodecIdentity
from predictive_state_helpers import codec_identity, fit_mean_baseline

pytestmark = pytest.mark.unit


def test_diagonal_gaussian_future_law_uses_nonlinear_mean_and_full_unit_variance_nll() -> None:
    law = DiagonalGaussianFutureLaw(latent_width=2, hidden_width=5)
    assert any(isinstance(module, nn.SiLU) for module in law.modules())
    with torch.no_grad():
        for parameter in law.parameters():
            parameter.zero_()
    states = LatentTrajectory(values=torch.zeros(1, 2, 2), codec_identity=codec_identity())
    target = FutureTarget(
        values=torch.tensor([[[1.0, -2.0], [3.0, -4.0]]]), codec_identity=codec_identity()
    )
    baseline = fit_mean_baseline(width=2, values=torch.tensor([[[10.0, -10.0], [10.0, -10.0]]]))
    outcome = law.loss(states, target, mean_baseline=baseline)
    expected = 0.5 * (target.values.square() + math.log(2.0 * math.pi)).mean()
    mean_expected = (
        0.5
        * ((target.values - baseline.mean.view(1, 1, -1)).square() + math.log(2.0 * math.pi)).mean()
    )
    torch.testing.assert_close(outcome.nll, expected, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(outcome.zero_displacement_nll, expected, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(outcome.mean_displacement_nll, mean_expected, atol=1e-7, rtol=0.0)
    assert outcome.residual_rms > 0.0
    assert outcome.residual_variance > 0.0
    assert outcome.mean.dtype == torch.float32


def test_future_law_refuses_mismatched_state_target_codec_or_shape() -> None:
    law = DiagonalGaussianFutureLaw(latent_width=8, hidden_width=12)
    states = LatentTrajectory(values=torch.zeros(1, 3, 8), codec_identity=codec_identity())
    wrong_width = FutureTarget(values=torch.zeros(1, 3, 7), codec_identity=codec_identity())
    with pytest.raises(ValueError, match="width"):
        law.loss(
            states,
            wrong_width,
            mean_baseline=fit_mean_baseline(width=8, values=torch.zeros(1, 2, 8)),
        )


def test_future_law_nll_has_a_gradient_path_to_online_states_and_predictor() -> None:
    law = DiagonalGaussianFutureLaw(latent_width=8, hidden_width=12)
    state_values = torch.randn(2, 3, 8, requires_grad=True)
    states = LatentTrajectory(values=state_values, codec_identity=codec_identity())
    target = FutureTarget(values=torch.randn(2, 3, 8), codec_identity=codec_identity())
    law.loss(
        states, target, mean_baseline=fit_mean_baseline(width=8, values=torch.randn(2, 3, 8))
    ).nll.backward()
    assert state_values.grad is not None and torch.isfinite(state_values.grad).all()
    assert all(parameter.grad is not None for parameter in law.parameters())


def test_mean_displacement_baseline_is_training_fitted_frozen_and_state_serializable() -> None:
    training_target = FutureTarget(
        values=torch.tensor([[[2.0, -4.0], [6.0, -8.0]]]), codec_identity=codec_identity()
    )
    baseline = MeanDisplacementBaseline(latent_width=2)
    baseline.fit(training_target)
    original_mean = baseline.mean.detach().clone()
    law = DiagonalGaussianFutureLaw(latent_width=2, hidden_width=5)
    states = LatentTrajectory(values=torch.zeros(1, 2, 2), codec_identity=codec_identity())
    eval_target = FutureTarget(
        values=torch.tensor([[[100.0, 200.0], [-100.0, -200.0]]]), codec_identity=codec_identity()
    )
    _ = law.loss(states, eval_target, mean_baseline=baseline)
    restored = MeanDisplacementBaseline(latent_width=2)
    restored.load_state_dict(copy.deepcopy(baseline.state_dict()))
    torch.testing.assert_close(baseline.mean, original_mean, atol=0.0, rtol=0.0)
    torch.testing.assert_close(restored.mean, original_mean, atol=0.0, rtol=0.0)
    assert restored.is_fitted
    assert restored.codec_identity == codec_identity()
    with pytest.raises(RuntimeError, match="already fitted"):
        baseline.fit(eval_target)


def test_future_law_refuses_unfitted_or_codec_mismatched_mean_baseline() -> None:
    law = DiagonalGaussianFutureLaw(latent_width=2, hidden_width=5)
    states = LatentTrajectory(values=torch.zeros(1, 2, 2), codec_identity=codec_identity())
    target = FutureTarget(values=torch.zeros(1, 2, 2), codec_identity=codec_identity())
    with pytest.raises(ValueError, match="fitted"):
        law.loss(states, target, mean_baseline=MeanDisplacementBaseline(latent_width=2))
    other_codec = CodecIdentity(name="other", construction_digest="c" * 64, cardinality=17)
    mismatched = MeanDisplacementBaseline(latent_width=2)
    mismatched.fit(FutureTarget(values=torch.zeros(1, 2, 2), codec_identity=other_codec))
    with pytest.raises(ValueError, match="codec"):
        law.loss(states, target, mean_baseline=mismatched)


def test_future_law_reports_residual_variance_and_fixed_marginal_gaussian_coverage() -> None:
    law = DiagonalGaussianFutureLaw(latent_width=1, hidden_width=3)
    with torch.no_grad():
        for parameter in law.parameters():
            parameter.zero_()
    states = LatentTrajectory(values=torch.zeros(1, 4, 1), codec_identity=codec_identity())
    target = FutureTarget(
        values=torch.tensor([[[0.0], [0.7], [1.3], [2.0]]]), codec_identity=codec_identity()
    )
    outcome = law.loss(
        states, target, mean_baseline=fit_mean_baseline(width=1, values=torch.zeros(1, 2, 1))
    )
    assert outcome.coverage_50.item() == pytest.approx(0.25)
    assert outcome.coverage_80.item() == pytest.approx(0.5)
    assert outcome.coverage_95.item() == pytest.approx(0.75)
    assert outcome.residual_variance.item() == pytest.approx(
        target.values.var(unbiased=False).item()
    )
