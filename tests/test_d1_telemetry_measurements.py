"""D1 TRAIN/EVAL payload parsing preserves exact cursors and schemas."""

from __future__ import annotations

from typing import cast

import pytest

from entropic_substrate.evidence.d1_telemetry_measurements import (
    parse_eval_event,
    parse_quality,
    parse_train_event,
)
from entropic_substrate.evidence.d1_telemetry_values import D1TelemetryGap

from .d1_telemetry_fixtures import replace_payload, valid_events

pytestmark = pytest.mark.unit


def test_eval_and_train_payloads_return_typed_same_cursor_evidence() -> None:
    events = valid_events()
    evaluation = parse_eval_event(events[2], 0)
    digest, count = parse_train_event(events[3])
    assert evaluation.step == evaluation.resource_work.training_step == 0
    assert evaluation.quality.nats_per_byte == 0.8
    assert len(digest) == 64
    assert count == 1


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
        parse_train_event(replace_payload(events[3], train_payload))
