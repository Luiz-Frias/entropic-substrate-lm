"""Future-target value and horizon-one alignment contracts for D1."""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.core.latent.state import LatentTrajectory, TargetEncoder
from entropic_substrate.core.latent.targets import (
    FutureTarget,
    one_step_displacement,
    one_step_target_from_context,
)
from entropic_substrate.core.observation.contracts import ObservationAlignment
from predictive_state_helpers import codec_identity, encoded, model_config

pytestmark = pytest.mark.unit


def test_one_step_target_displacement_is_exact_from_t_plus_one_target_context() -> None:
    values = torch.tensor([[[1.0, 2.0], [4.0, 8.0], [10.0, 20.0], [22.0, 44.0]]])
    target = one_step_displacement(LatentTrajectory(values=values, codec_identity=codec_identity()))
    assert isinstance(target, FutureTarget)
    assert target.values.shape == (1, 3, 2)
    torch.testing.assert_close(
        target.values,
        torch.tensor([[[3.0, 6.0], [6.0, 12.0], [12.0, 24.0]]]),
        atol=0.0,
        rtol=0.0,
    )


def test_target_displacement_requires_target_context_with_at_least_two_positions() -> None:
    trajectory = LatentTrajectory(values=torch.zeros(2, 1, 8), codec_identity=codec_identity())
    with pytest.raises(ValueError, match="at least two"):
        one_step_displacement(trajectory)


def test_target_encoder_constructs_one_step_target_only_from_legal_t_plus_one_context() -> None:
    target, context = TargetEncoder(model_config()).eval(), encoded(time=5)
    alignment = ObservationAlignment.predictive(start=9, sequence_length=4, horizon=1)
    assert one_step_target_from_context(target, context, alignment).values.shape == (2, 4, 8)
    with pytest.raises(ValueError, match="horizon 1"):
        one_step_target_from_context(
            target,
            context,
            ObservationAlignment.predictive(start=9, sequence_length=3, horizon=2),
        )
