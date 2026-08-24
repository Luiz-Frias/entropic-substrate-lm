"""TDD contract for canonical D1 launch-attempt ledger persistence."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from entropic_substrate.experiments.d1_launch_attempt_codec import (
    D1LaunchAttemptCodecError,
    launch_attempt_ledger_from_mapping,
    launch_attempt_ledger_to_mapping,
    parse_launch_attempt_ledger,
    serialize_launch_attempt_ledger,
)
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)

from .test_d1_launch_attempt_ledger import NOW, identity

pytestmark = pytest.mark.unit


def _ledger() -> D1LaunchAttemptLedger:
    first = identity()
    second = identity(logical_run_id="d1-confirmatory-sg-s103", seed=103)
    ledger = D1LaunchAttemptLedger().record(first, observed_at=NOW)
    ledger = ledger.transition(
        first,
        status=D1LaunchAttemptStatus.TRAINER_STARTED,
        observed_at=NOW + timedelta(seconds=1),
    )
    ledger = ledger.transition(
        first,
        status=D1LaunchAttemptStatus.COMPLETED,
        observed_at=NOW + timedelta(seconds=2),
    )
    return ledger.record(second, observed_at=NOW + timedelta(seconds=3))


@pytest.mark.parametrize("ledger", [D1LaunchAttemptLedger(), _ledger()])
def test_canonical_round_trip_preserves_events_and_ledger_digest(
    ledger: D1LaunchAttemptLedger,
) -> None:
    payload = serialize_launch_attempt_ledger(ledger)
    restored = parse_launch_attempt_ledger(payload)

    assert restored == ledger
    assert restored.ledger_digest == ledger.ledger_digest
    assert launch_attempt_ledger_from_mapping(launch_attempt_ledger_to_mapping(ledger)) == ledger
    assert payload == json.dumps(
        ledger.manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"invented": True}),
        lambda value: value.pop("schema_version"),
        lambda value: value["events"][0].update({"status": "invented"}),
        lambda value: value["events"][0]["identity"].update({"optional_stopping_digest": "0" * 64}),
        lambda value: value["events"][1].update({"event_seq": 99}),
    ],
)
def test_mapping_parser_refuses_unknown_missing_forged_or_illegal_history(
    mutation: object,
) -> None:
    value = launch_attempt_ledger_to_mapping(_ledger())
    mutation(value)  # type: ignore[operator]

    with pytest.raises(D1LaunchAttemptCodecError):
        launch_attempt_ledger_from_mapping(value)


@pytest.mark.parametrize(
    "payload",
    [
        '{"events":[],"events":[],"schema_version":"d1-launch-ledger-v1"}',
        '{"events":[],"schema_version":NaN}',
        ' {"events":[],"schema_version":"d1-launch-ledger-v1"}',
        '{"schema_version":"d1-launch-ledger-v1","events":[]}',
    ],
)
def test_text_parser_refuses_duplicates_nonfinite_and_noncanonical_json(payload: str) -> None:
    with pytest.raises(D1LaunchAttemptCodecError):
        parse_launch_attempt_ledger(payload)


@pytest.mark.parametrize(
    "timestamp",
    ["2026-08-24T11:00:00-04:00", "2026-08-24T15:00:00Z", "2026-08-24 15:00:00+00:00"],
)
def test_timestamp_must_use_datetime_isoformat_utc_spelling(timestamp: str) -> None:
    value = launch_attempt_ledger_to_mapping(_ledger())
    value["events"][0]["observed_at"] = timestamp

    with pytest.raises(D1LaunchAttemptCodecError, match="observed_at"):
        launch_attempt_ledger_from_mapping(value)


def test_codec_requires_nominal_ledger_and_top_level_mapping() -> None:
    with pytest.raises(TypeError, match="D1LaunchAttemptLedger"):
        serialize_launch_attempt_ledger(object())  # type: ignore[arg-type]
    with pytest.raises(D1LaunchAttemptCodecError, match="object"):
        launch_attempt_ledger_from_mapping([])
