"""Fixed-future score comparisons refuse incompatible evidence."""

from dataclasses import replace

import pytest

from entropic_substrate.analysis.predictive_score_gain import (
    PredictiveScore,
    ScoreBinding,
    pair_score_gain,
    summarize_score_gains,
)

pytestmark = pytest.mark.unit


def scores() -> tuple[PredictiveScore, PredictiveScore]:
    binding = ScoreBinding("checkpoint", "probe", "codec", "heldout", "source")
    return (
        PredictiveScore(binding, 10, 14, 65, 4, -2.0),
        PredictiveScore(binding, 11, 14, 65, 3, -1.0),
    )


def test_gain_and_summary_preserve_sign_and_quality() -> None:
    before, after = scores()
    gain = pair_score_gain(before, after)
    assert gain.gain == 1.0
    negative = pair_score_gain(
        replace(before, state_position=20, target_position=24),
        replace(after, state_position=21, target_position=24, log_probability=-3.0),
    )
    summary = summarize_score_gains((gain, negative))
    assert summary.count == 2
    assert summary.mean_gain == 0.0
    assert summary.sample_std_gain == pytest.approx(2**0.5)
    assert summary.negative_fraction == 0.5
    assert summary.before_nats_per_byte == summary.after_nats_per_byte == 2.0


@pytest.mark.parametrize(
    "field", ["checkpoint_id", "probe_id", "codec_id", "split_id", "source_id"]
)
def test_pair_refuses_identity_mismatch(field: str) -> None:
    before, after = scores()
    with pytest.raises(ValueError, match="binding"):
        pair_score_gain(before, replace(after, binding=replace(after.binding, **{field: "other"})))


def test_pair_refuses_different_target_or_nonadjacent_states() -> None:
    before, after = scores()
    with pytest.raises(ValueError, match="same future"):
        pair_score_gain(before, replace(after, target_byte=66))
    with pytest.raises(ValueError, match="adjacent"):
        pair_score_gain(before, replace(after, state_position=12, remaining_horizon=2))
    with pytest.raises(ValueError, match="same future"):
        pair_score_gain(before, replace(after, target_position=15, remaining_horizon=4))


def test_invalid_and_empty_evidence_fails_loudly() -> None:
    before, _ = scores()
    for value in (float("nan"), float("inf"), 0.1):
        with pytest.raises(ValueError, match="log probability"):
            replace(before, log_probability=value)
    with pytest.raises(ValueError, match="horizon"):
        replace(before, remaining_horizon=3)
    with pytest.raises(ValueError, match="future"):
        replace(before, target_position=10, remaining_horizon=0)
    with pytest.raises(ValueError, match="empty"):
        summarize_score_gains(())


def test_uniform_gain_is_zero_and_singleton_dispersion_is_explicit() -> None:
    before, after = scores()
    pair = pair_score_gain(before, replace(after, log_probability=-2.0))
    summary = summarize_score_gains((pair,))
    assert summary.mean_gain == summary.negative_fraction == 0.0
    assert summary.sample_std_gain is None
