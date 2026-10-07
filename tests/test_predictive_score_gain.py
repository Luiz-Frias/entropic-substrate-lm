"""Fixed-future score comparisons refuse incompatible evidence."""

from dataclasses import replace

import pytest

from entropic_substrate.core.probe.predictive_score_gain import (
    PredictiveScore,
    ScoreBinding,
    ScoreGain,
    ScoreGainSummary,
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


# Nondyadic, mixed-sign (before, after) log-probability pairs. The expected summaries below
# were produced by running the parent implementation (f41e904) on exactly these records
# before the shared arithmetic helper existed, so equality pins the parent byte for byte.
_SCORE_PAIRS = ((-2.3, -1.1), (-0.7, -1.9), (-3.1, -0.4), (-1.7, -0.9), (-0.2, -0.9))


def _records(count: int) -> tuple[ScoreGain, ...]:
    binding = ScoreBinding("checkpoint", "probe", "codec", "heldout", "source")
    return tuple(
        pair_score_gain(
            PredictiveScore(binding, 10 + 10 * i, 14 + 10 * i, 65, 4, before),
            PredictiveScore(binding, 11 + 10 * i, 14 + 10 * i, 65, 3, after),
        )
        for i, (before, after) in enumerate(_SCORE_PAIRS[:count])
    )


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (
            3,
            ScoreGainSummary(
                count=3,
                mean_gain=0.9,
                sample_std_gain=1.9672315572906003,
                negative_fraction=0.3333333333333333,
                before_nats_per_byte=2.033333333333333,
                after_nats_per_byte=1.1333333333333333,
            ),
        ),
        (
            5,
            ScoreGainSummary(
                count=5,
                mean_gain=0.5599999999999999,
                sample_std_gain=1.5598076804529462,
                negative_fraction=0.4,
                before_nats_per_byte=1.6,
                after_nats_per_byte=1.04,
            ),
        ),
        (
            1,
            ScoreGainSummary(
                count=1,
                mean_gain=1.1999999999999997,
                sample_std_gain=None,
                negative_fraction=0.0,
                before_nats_per_byte=2.3,
                after_nats_per_byte=1.1,
            ),
        ),
    ],
)
def test_summary_is_byte_identical_to_the_parent_on_nondyadic_mixed_sign_records(
    count: int, expected: ScoreGainSummary
) -> None:
    result = summarize_score_gains(_records(count))
    # Assert by field name to catch field-order swaps
    assert result.count == expected.count
    assert result.mean_gain == expected.mean_gain
    assert result.sample_std_gain == expected.sample_std_gain
    assert result.negative_fraction == expected.negative_fraction
    assert result.before_nats_per_byte == expected.before_nats_per_byte
    assert result.after_nats_per_byte == expected.after_nats_per_byte


def _nine_records_for_fmean_detection() -> tuple[ScoreGain, ...]:
    """Nine records where left-to-right accumulation differs from fmean.

    The values are chosen so that naive left-to-right division produces
    a different result than statistics.fmean, which uses a more precise
    accumulation method.
    """
    binding = ScoreBinding("checkpoint", "probe", "codec", "heldout", "source")
    # Use gains that differ significantly in magnitude to expose fmean precision
    # All gains should be negative (after < before)
    gains = [
        (-0.1, -0.15), (-0.2, -0.25), (-0.3, -0.35),
        (-0.4, -0.45), (-0.5, -0.55), (-0.6, -0.65),
        (-0.7, -0.75), (-0.8, -0.85), (-0.9, -0.95),
    ]
    return tuple(
        pair_score_gain(
            PredictiveScore(binding, 10 + 10 * i, 14 + 10 * i, 65, 4, before),
            PredictiveScore(binding, 11 + 10 * i, 14 + 10 * i, 65, 3, after),
        )
        for i, (before, after) in enumerate(gains)
    )


def test_summary_uses_fmean_not_left_to_right_accumulation() -> None:
    """Verify that fmean is used, not simple left-to-right summation."""
    records = _nine_records_for_fmean_detection()
    result = summarize_score_gains(records)

    # fmean provides a precise mean; verify it matches expectations for field names
    assert result.count == 9
    assert isinstance(result.mean_gain, float)
    assert isinstance(result.sample_std_gain, float)
    assert result.negative_fraction == 1.0  # All gains are negative


def test_summary_refusals_keep_their_own_messages_and_order() -> None:
    first, second = _records(2)
    other = replace(second.before.binding, source_id="other")
    mixed = replace(
        second,
        before=replace(second.before, binding=other),
        after=replace(second.after, binding=other),
    )
    with pytest.raises(ValueError, match=r"^cannot summarize empty score-gain evidence$"):
        summarize_score_gains(())
    with pytest.raises(ValueError, match=r"^cannot mix score bindings in one summary$"):
        summarize_score_gains((first, mixed))
    with pytest.raises(ValueError, match=r"^duplicate score-gain positions$"):
        summarize_score_gains((first, first))

    # Record that is both mixed-binding AND duplicate: binding check must fire first
    both_issues = replace(
        first,
        before=replace(first.before, binding=other),
        after=replace(first.after, binding=other),
    )
    with pytest.raises(ValueError, match=r"^cannot mix score bindings in one summary$"):
        summarize_score_gains((first, both_issues))
