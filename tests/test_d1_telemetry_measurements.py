"""D1 TRAIN/EVAL payload parsing preserves exact cursors and schemas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from entropic_substrate.kernel.identity.d1_primitives import TargetUpdateStrategy
from entropic_substrate.ladder.capture.records.d1_telemetry_measurements import (
    parse_eval_event,
    parse_quality,
    parse_train_event,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1TelemetryGap

from .d1_telemetry_fixtures import replace_payload, valid_events

pytestmark = pytest.mark.unit


def test_eval_and_train_payloads_return_typed_same_cursor_evidence() -> None:
    events = valid_events()
    evaluation = parse_eval_event(events[2], 0)
    digest, count, result = parse_train_event(
        events[3], expected_target_update=TargetUpdateStrategy.STOP_GRADIENT_EMA
    )
    assert evaluation.step == evaluation.resource_work.training_step == 0
    assert evaluation.quality.nats_per_byte == 0.8
    assert evaluation.quality.state_mean == (0.1, -0.2)
    assert evaluation.quality.target_mean == (0.02, 0.01)
    assert evaluation.quality.predicted_mean == (0.01, -0.01)
    assert evaluation.quality.residual_mean_vector == (0.01, 0.02)
    assert evaluation.quality.target_predicted_exact_match_fraction == 0.0
    assert len(digest) == 64
    assert count == 1
    assert result.successful_step == 1


def test_measurement_parsers_refuse_partial_extra_and_wrong_cursor_payloads() -> None:
    events = valid_events()
    eval_payload = cast(dict[str, object], events[2].manifest["payload"])
    quality = cast(dict[str, object], eval_payload["quality"])
    del quality["future_nll"]
    with pytest.raises(D1TelemetryGap, match="fields"):
        parse_quality(quality)
    eval_payload = cast(dict[str, object], events[2].manifest["payload"])
    eval_payload["eval_index"] = 1
    with pytest.raises(D1TelemetryGap, match="consecutive"):
        parse_eval_event(replace_payload(events[2], eval_payload), 0)
    train_payload = cast(dict[str, object], events[3].manifest["payload"])
    train_payload["extra"] = 0
    with pytest.raises(D1TelemetryGap, match="fields"):
        parse_train_event(
            replace_payload(events[3], train_payload),
            expected_target_update=TargetUpdateStrategy.STOP_GRADIENT_EMA,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("predicted_mean", [0.01], "width"),
        ("residual_mean_vector", [0.02, 0.02], "derivation"),
        ("residual_mean_l2_norm", 0.03, "derivation"),
        ("target_predicted_exact_match_fraction", 0.25, "derivation"),
        ("target_predicted_exact_match_semantic", "different", "derivation"),
        ("target_variance", [float("nan"), 0.3], "finite"),
    ],
)
def test_quality_parser_refuses_direct_channel_width_and_assertion_drift(
    field: str, value: object, match: str
) -> None:
    quality = cast(dict[str, object], valid_events()[2].manifest["payload"])["quality"]
    drifted = cast(dict[str, object], quality)
    drifted[field] = value
    with pytest.raises(D1TelemetryGap, match=match):
        parse_quality(drifted)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update({"extra": 1}), "fields"),
        (lambda value: value.pop("future_law"), "fields"),
        (lambda value: value["online_branch"].update({"pre_clip_l2_norm": 0.0}), "nonzero"),
        (lambda value: value["target_branch"].update({"kind": "unknown"}), "target_branch"),
        (lambda value: value["target_branch"].update({"pre_clip_l2_norm": 0.25}), "null"),
    ],
)
def test_train_parser_strictly_admits_gradient_reachability(
    mutation: Callable[[dict[str, Any]], object],
    match: str,
) -> None:
    events = valid_events()
    train_payload = cast(dict[str, object], events[3].manifest["payload"])
    train = cast(dict[str, object], train_payload["train"])
    reachability = cast(dict[str, Any], train["gradient_reachability"])
    mutation(reachability)
    with pytest.raises(D1TelemetryGap, match=match):
        parse_train_event(
            replace_payload(events[3], train_payload),
            expected_target_update=TargetUpdateStrategy.STOP_GRADIENT_EMA,
        )


def test_train_parser_refuses_inconsistent_aggregate_gradient_norm() -> None:
    events = valid_events()
    train_payload = cast(dict[str, object], events[3].manifest["payload"])
    train = cast(dict[str, object], train_payload["train"])
    train["gradient_norm"] = 7.0

    with pytest.raises(D1TelemetryGap, match="reconstruct"):
        parse_train_event(
            replace_payload(events[3], train_payload),
            expected_target_update=TargetUpdateStrategy.STOP_GRADIENT_EMA,
        )
