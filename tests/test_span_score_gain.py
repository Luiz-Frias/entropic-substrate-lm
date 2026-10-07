"""Sentence-event score pairs never weaken historical byte-adjacency contracts."""

import math
import statistics
from dataclasses import replace

import pytest

from entropic_substrate.core.probe.predictive_score_gain import (
    PredictiveScore,
    ScoreBinding,
    pair_score_gain,
)
from entropic_substrate.core.probe.score_gain import (
    SpanGainSummary,
    SpanPredictiveScore,
    SpanScoreGain,
    summarize_span_gains,
)

pytestmark = pytest.mark.unit


def pair() -> SpanScoreGain:
    binding = ScoreBinding("checkpoint", "probe", "codec", "heldout", "document")
    return SpanScoreGain(
        SpanPredictiveScore(binding, 0, -1, 13, 65, 14, -3.0),
        SpanPredictiveScore(binding, 1, 9, 13, 65, 4, -2.0),
        0,
        10,
    )


def test_sentence_update_keeps_the_target_and_explicit_horizons() -> None:
    value = pair()
    assert value.gain == 1.0
    assert value.before.remaining_horizon - value.after.remaining_horizon == 10
    assert value.schema == "span-predictive-score-gain-v1"


@pytest.mark.parametrize(
    "field", ["checkpoint_id", "probe_id", "codec_id", "split_id", "source_id"]
)
def test_score_pair_rejects_mismatched_bindings(field: str) -> None:
    value = pair()
    changed = replace(value.after, binding=replace(value.after.binding, **{field: "different"}))
    with pytest.raises(ValueError, match="binding"):
        replace(value, after=changed)


@pytest.mark.parametrize(
    "change",
    [
        {"event_index": 3},
        {"target_byte": 66},
        {"target_position": 14, "remaining_horizon": 5},
        {"observed_through": 8, "remaining_horizon": 5},
    ],
)
def test_wrong_event_span_or_target_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(pair(), after=replace(pair().after, **change))


@pytest.mark.parametrize(
    "change",
    [
        {"event_index": True},
        {"observed_through": -2},
        {"log_probability": math.nan},
        {"log_probability": 1.0},
        {"target_byte": 256},
        {"remaining_horizon": 0},
    ],
)
def test_invalid_score_coordinates_or_values_are_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(pair().before, **change)


def test_document_samples_are_nested_not_independent_seed_replicates() -> None:
    first = pair()
    binding = replace(first.before.binding, source_id="second-document")
    second = replace(
        first,
        before=replace(first.before, binding=binding),
        after=replace(first.after, binding=binding, log_probability=-4.0),
    )
    summary = summarize_span_gains((first, second))
    assert summary.count == 2 and summary.mean_gain == 0
    assert summary.negative_fraction == 0.5
    assert summary.inference_unit == "nested_source_positions_not_model_replicates"
    with pytest.raises(ValueError, match="duplicate"):
        summarize_span_gains((first, first))


def test_historical_reader_still_refuses_nonadjacent_byte_states() -> None:
    binding = pair().before.binding
    before = PredictiveScore(binding, 1, 13, 65, 12, -3.0)
    after = PredictiveScore(binding, 9, 13, 65, 4, -2.0)
    with pytest.raises(ValueError, match="adjacent"):
        pair_score_gain(before, after)


# Nondyadic, mixed-sign (before, after) log-probability pairs, one document each. The expected
# summaries below were produced by running the parent implementation (f41e904) on exactly these
# records before the shared arithmetic helper existed, so equality pins the parent byte for byte.
_SPAN_PAIRS = ((-2.3, -1.1), (-0.7, -1.9), (-3.1, -0.4), (-1.7, -0.9), (-0.2, -0.9))


def _records(count: int) -> tuple[SpanScoreGain, ...]:
    records = []
    for i, (before, after) in enumerate(_SPAN_PAIRS[:count]):
        binding = ScoreBinding("checkpoint", "probe", "codec", "heldout", f"doc{i}")
        records.append(
            SpanScoreGain(
                SpanPredictiveScore(binding, 0, -1, 13, 65, 14, before),
                SpanPredictiveScore(binding, 1, 9, 13, 65, 4, after),
                0,
                10,
            )
        )
    return tuple(records)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (
            3,
            SpanGainSummary(
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
            SpanGainSummary(
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
            SpanGainSummary(
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
    count: int, expected: SpanGainSummary
) -> None:
    result = summarize_span_gains(_records(count))
    # Assert by field name to catch field-order swaps
    assert result.count == expected.count
    assert result.mean_gain == expected.mean_gain
    assert result.sample_std_gain == expected.sample_std_gain
    assert result.negative_fraction == expected.negative_fraction
    assert result.before_nats_per_byte == expected.before_nats_per_byte
    assert result.after_nats_per_byte == expected.after_nats_per_byte


# Measured on the pinned CPython 3.12.13: for these three pairs the gain mean from
# statistics.fmean is -1.3666666666666665 while a manual left-to-right loop gives
# -1.366666666666667. sum()/len() agrees with fmean on 3.12 (sum is compensated there), so
# only the manual loop separates the implementations; the precondition test pins that.
_SEPARATING_PAIRS = ((-0.6, -2.0), (-0.8, -2.6), (-1.2, -2.1))


def _separating_records() -> tuple[SpanScoreGain, ...]:
    records = []
    for i, (before, after) in enumerate(_SEPARATING_PAIRS):
        binding = ScoreBinding("checkpoint", "probe", "codec", "heldout", f"doc{i}")
        records.append(
            SpanScoreGain(
                SpanPredictiveScore(binding, 0, -1, 13, 65, 14, before),
                SpanPredictiveScore(binding, 1, 9, 13, 65, 4, after),
                0,
                10,
            )
        )
    return tuple(records)


def _left_to_right_mean(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def test_separating_fixture_distinguishes_fmean_from_left_to_right_loop() -> None:
    gains = [item.gain for item in _separating_records()]
    assert statistics.fmean(gains) == -1.3666666666666665
    assert _left_to_right_mean(gains) == -1.366666666666667


def test_summary_means_are_exactly_fmean_not_left_to_right_accumulation() -> None:
    records = _separating_records()
    result = summarize_span_gains(records)
    assert result.mean_gain == statistics.fmean(item.gain for item in records)
    assert result.mean_gain == -1.3666666666666665
    assert result.before_nats_per_byte == -statistics.fmean(
        item.before.log_probability for item in records
    )
    assert result.before_nats_per_byte == 0.8666666666666667
    assert result.after_nats_per_byte == -statistics.fmean(
        item.after.log_probability for item in records
    )
    assert result.after_nats_per_byte == 2.2333333333333334


def test_summary_refusals_keep_their_own_messages_and_order() -> None:
    first, second = _records(2)
    with pytest.raises(ValueError, match=r"^cannot summarize empty span score evidence$"):
        summarize_span_gains(())
    other = replace(second.before.binding, checkpoint_id="other")
    mixed = replace(
        second,
        before=replace(second.before, binding=other),
        after=replace(second.after, binding=other),
    )
    with pytest.raises(ValueError, match=r"^cannot mix model, probe, codec or split bindings$"):
        summarize_span_gains((first, mixed))
    with pytest.raises(ValueError, match=r"^duplicate span score positions$"):
        summarize_span_gains((first, first))

    # Record that is both mixed-binding AND duplicate: binding check must fire first
    # Built from first's binding: same source_id, span_start and target_position as `first`,
    # so the record is a duplicate key AND a mixed binding (checkpoint_id differs).
    mixed_dup = replace(first.before.binding, checkpoint_id="other")
    both_issues = replace(
        first,
        before=replace(first.before, binding=mixed_dup),
        after=replace(first.after, binding=mixed_dup),
    )
    with pytest.raises(ValueError, match=r"^cannot mix model, probe, codec or split bindings$"):
        summarize_span_gains((first, both_issues))
