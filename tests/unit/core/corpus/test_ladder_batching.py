"""Paired horizon arms consume identical source offsets, not just RNG seeds."""

import numpy as np
import pytest
import torch

from entropic_substrate.core.corpus.batching import SequenceSampler, TokenSplit
from entropic_substrate.core.corpus.ladder_batching import paired_predictive_batch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("train", [True, False])
def test_horizon_pairing_preserves_observations_offsets_and_digest(train: bool) -> None:
    split = TokenSplit(np.arange(512, dtype=np.int64) % 256)
    samplers = [SequenceSampler(split, 2, 8, seed=7, train=train) for _ in range(4)]
    for _ in range(40):
        batches = [
            paired_predictive_batch(s, h, reservation=64)
            for s, h in zip(samplers, (1, 4, 16, 64), strict=True)
        ]
        for horizon, batch in zip((1, 4, 16, 64), batches, strict=True):
            assert batch.horizon == horizon
            assert batch.target_encoder_context.shape == (2, 8 + horizon)
            torch.testing.assert_close(batch.source_starts, batches[0].source_starts)
            torch.testing.assert_close(batch.online_observations, batches[0].online_observations)
            torch.testing.assert_close(batch.language_targets, batches[0].language_targets)
    assert len({sampler.data_order_digest() for sampler in samplers}) == 1


@pytest.mark.parametrize("horizon,reservation", [(64, 1), (0, 64), (True, 64)])
def test_invalid_reservation_is_refused(horizon: int, reservation: int) -> None:
    sampler = SequenceSampler(TokenSplit(np.arange(128, dtype=np.int64)), 1, 8, seed=0)
    with pytest.raises(ValueError):
        paired_predictive_batch(sampler, horizon, reservation=reservation)
