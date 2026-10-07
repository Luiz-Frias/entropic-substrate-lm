"""Sentence-event score pairs never weaken historical byte-adjacency contracts."""

import math
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
                3,
                0.9,
                1.9672315572906003,
                0.3333333333333333,
                2.033333333333333,
                1.1333333333333333,
            ),
        ),
        (
            5,
            SpanGainSummary(
                5,
                0.5599999999999999,
                1.5598076804529462,
                0.4,
                1.6,
                1.04,
            ),
        ),
        (1, SpanGainSummary(1, 1.1999999999999997, None, 0.0, 2.3, 1.1)),
    ],
)
def test_summary_is_byte_identical_to_the_parent_on_nondyadic_mixed_sign_records(
    count: int, expected: SpanGainSummary
) -> None:
    assert summarize_span_gains(_records(count)) == expected


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
