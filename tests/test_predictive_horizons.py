"""Source-aligned D3 horizon targets and Gaussian head bank contracts."""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.core.latent.future_law import MeanDisplacementBaseline
from entropic_substrate.core.latent.horizons import (
    HorizonFutureLaw,
    horizon_displacement,
)
from entropic_substrate.core.latent.state import LatentTrajectory, TargetEncoder
from entropic_substrate.core.latent.target_values import one_step_displacement
from entropic_substrate.core.observation.contracts import EncodedObservations, ObservationAlignment
from predictive_state_helpers import codec_identity, encoded, model_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("horizon", [1, 4, 16, 64])
def test_displacement_is_aligned_and_preserves_target_gradients(horizon: int) -> None:
    values = torch.arange((3 + horizon) * 8, dtype=torch.float32).reshape(1, -1, 8)
    values.requires_grad_()
    trajectory = LatentTrajectory(values, codec_identity())
    alignment = ObservationAlignment.predictive(start=23, sequence_length=3, horizon=horizon)
    result = horizon_displacement(trajectory, alignment)
    assert result.alignment.online_observations.start == 23
    torch.testing.assert_close(result.target.values, values[:, horizon:] - values[:, :3])
    result.target.values.square().sum().backward()
    assert values.grad is not None and bool(values.grad.abs().sum() > 0)
    if horizon == 1:
        torch.testing.assert_close(result.target.values, one_step_displacement(trajectory).values)


def test_horizon_refuses_short_context_and_unregistered_horizon() -> None:
    trajectory = LatentTrajectory(torch.zeros(1, 5, 8), codec_identity())
    with pytest.raises(ValueError, match="context length"):
        horizon_displacement(
            trajectory, ObservationAlignment.predictive(start=0, sequence_length=3, horizon=4)
        )
    with pytest.raises(ValueError, match="registered"):
        horizon_displacement(
            trajectory, ObservationAlignment.predictive(start=0, sequence_length=3, horizon=2)
        )


def test_target_context_suffix_cannot_change_prior_encoder_states() -> None:
    encoder = TargetEncoder(model_config()).eval()
    original = encoded(time=8)
    changed = original.values.clone()
    changed[:, 4:] += 3.0
    before = encoder(original)
    after = encoder(EncodedObservations(changed, original.codec_identity))
    torch.testing.assert_close(before.values[:, :4], after.values[:, :4], atol=0.0, rtol=0.0)
    assert not torch.equal(before.values[:, 4:], after.values[:, 4:])


def test_composite_is_equal_average_and_all_heads_receive_gradients() -> None:
    horizons = (1, 4, 16, 64)
    values = torch.randn(2, 67, 8, requires_grad=True)
    trajectory = LatentTrajectory(values, codec_identity())
    states = LatentTrajectory(torch.randn(2, 3, 8, requires_grad=True), codec_identity())
    targets = {
        h: horizon_displacement(
            trajectory if h == 64 else LatentTrajectory(values[:, : 3 + h], codec_identity()),
            ObservationAlignment.predictive(start=11, sequence_length=3, horizon=h),
        )
        for h in horizons
    }
    baselines = {h: MeanDisplacementBaseline(latent_width=8) for h in horizons}
    for h in horizons:
        baselines[h].fit(targets[h].target)
    bank = HorizonFutureLaw(latent_width=8, hidden_width=12, horizons=horizons)
    result = bank.loss(states, targets, mean_baselines=baselines)
    expected = torch.stack([result.per_horizon[h].nll for h in horizons]).mean()
    torch.testing.assert_close(result.nll, expected)
    result.nll.backward()
    assert all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in bank.parameters())
    assert states.values.grad is not None
    assert values.grad is not None
    with pytest.raises(ValueError, match="roster"):
        bank.loss(states, {1: targets[1]}, mean_baselines=baselines)
    mismatched = dict(targets)
    mismatched[4] = horizon_displacement(
        LatentTrajectory(values[:, :7], codec_identity()),
        ObservationAlignment.predictive(start=12, sequence_length=3, horizon=4),
    )
    with pytest.raises(ValueError, match="source span"):
        bank.loss(states, mismatched, mean_baselines=baselines)


@pytest.mark.parametrize("horizons", [(), (1, 1), (2,), (True,)])
def test_head_bank_rejects_invalid_rosters(horizons: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="horizon"):
        HorizonFutureLaw(latent_width=8, hidden_width=12, horizons=horizons)


def test_single_horizon_reuses_d1_loss_exactly() -> None:
    trajectory = LatentTrajectory(torch.randn(2, 5, 8), codec_identity())
    states = LatentTrajectory(torch.randn(2, 4, 8), codec_identity())
    target = horizon_displacement(
        trajectory, ObservationAlignment.predictive(start=4, sequence_length=4, horizon=1)
    )
    baseline = MeanDisplacementBaseline(latent_width=8)
    baseline.fit(target.target)
    bank = HorizonFutureLaw(latent_width=8, hidden_width=12, horizons=(1,))
    result = bank.loss(states, {1: target}, mean_baselines={1: baseline})
    torch.testing.assert_close(result.nll, result.per_horizon[1].nll, atol=0.0, rtol=0.0)
    assert set(bank(states)) == {1}
