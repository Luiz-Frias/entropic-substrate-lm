"""Run-start and baseline payload parsing is exact and typed."""

from __future__ import annotations

from typing import cast

import pytest

from entropic_substrate.evidence.d1_telemetry_payloads import (
    parse_baseline_event,
    parse_run_start_event,
)
from entropic_substrate.evidence.d1_telemetry_values import D1TelemetryGap
from entropic_substrate.experiments.d1_primitives import canonical_digest

from .d1_telemetry_fixtures import replace_payload, valid_events

pytestmark = pytest.mark.unit


def test_start_plan_provenance_and_baseline_are_bound_to_journal_identity() -> None:
    events = valid_events()
    plan, digest, start = parse_run_start_event(events[0])
    baseline = parse_baseline_event(events[1])
    assert plan["logical_run_id"] == events[0].logical_run_id
    assert len(digest) == 64
    assert start.execution_segment_index == 0
    assert baseline.source_span_positions == 8


def test_start_and_baseline_refuse_extra_or_unbound_fields() -> None:
    events = valid_events()
    start_payload = cast(dict[str, object], events[0].manifest["payload"])
    start_payload["plan_digest"] = "0" * 64
    with pytest.raises(D1TelemetryGap, match="plan_digest"):
        parse_run_start_event(replace_payload(events[0], start_payload))
    start_payload = cast(dict[str, object], events[0].manifest["payload"])
    plan = cast(dict[str, object], start_payload["plan"])
    evidence = cast(dict[str, object], plan["evidence"])
    evidence["unregistered"] = True
    start_payload["plan_digest"] = canonical_digest(plan)
    with pytest.raises(D1TelemetryGap, match="evidence"):
        parse_run_start_event(replace_payload(events[0], start_payload))
    baseline_payload = cast(dict[str, object], events[1].manifest["payload"])
    baseline_payload["unregistered"] = 0
    with pytest.raises(D1TelemetryGap, match="fields"):
        parse_baseline_event(replace_payload(events[1], baseline_payload))
