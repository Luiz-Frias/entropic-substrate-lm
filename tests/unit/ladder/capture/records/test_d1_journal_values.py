"""D1 authoritative-journal values are strict, canonical, and immutable."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.kernel.identity.d1_primitives import RunClass
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalError,
    D1JournalEvent,
    JournalCheckpointState,
    JournalEventKind,
    canonical_event_line,
    parse_event_line,
)

pytestmark = pytest.mark.unit


def _event(**overrides: object) -> D1JournalEvent:
    values: dict[str, object] = {
        "event_kind": JournalEventKind.RUN_START,
        "event_seq": 0,
        "logical_run_id": "d1-confirmatory-both-seed-7",
        "execution_digest": "1" * 64,
        "aggregation_digest": "2" * 64,
        "construction_digest": "3" * 64,
        "family_digest": "4" * 64,
        "run_class": RunClass.CONFIRMATORY,
        "successful_step": 0,
        "payload": {
            "exact_float": 0.12345678901234568,
            "nested": {"enabled": True, "values": [1, None, "evidence"]},
        },
    }
    values.update(overrides)
    return D1JournalEvent(**values)  # type: ignore[arg-type]


def test_event_envelope_is_canonical_round_trippable_and_full_precision() -> None:
    event = _event()
    line = canonical_event_line(event)
    assert line.endswith(b"\n")
    assert b'"exact_float":0.12345678901234568' in line
    assert line.startswith(b'{"aggregation_digest":')
    assert parse_event_line(line) == event
    assert event.schema_version == "d1-authoritative-event-v1"
    assert event.payload["nested"] == {"enabled": True, "values": (1, None, "evidence")}
    with pytest.raises(TypeError):
        event.payload["later"] = 1  # type: ignore[index]


def test_event_kind_vocabulary_is_exact() -> None:
    assert tuple(kind.value for kind in JournalEventKind) == (
        "run_start",
        "baseline_calibrated",
        "train",
        "eval",
        "checkpoint",
        "run_resume",
        "trainer_aborted",
        "run_end",
        "run_abort",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("event_seq", True, "event_seq"),
        ("event_seq", -1, "event_seq"),
        ("successful_step", False, "successful_step"),
        ("successful_step", -1, "successful_step"),
        ("logical_run_id", " ", "logical_run_id"),
        ("execution_digest", "A" * 64, "execution_digest"),
        ("aggregation_digest", "2" * 63, "aggregation_digest"),
        ("construction_digest", "g" * 64, "construction_digest"),
        ("family_digest", "", "family_digest"),
        ("payload", {"loss": float("nan")}, "payload.loss"),
        ("payload", {"loss": float("inf")}, "payload.loss"),
        ("payload", {1: "not-json"}, "payload"),
        ("payload", {"set": {1, 2}}, "payload.set"),
    ],
)
def test_event_refuses_ambiguous_or_non_json_values(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, D1JournalError), match=message):
        _event(**{field: value})


def test_event_requires_real_enum_values_at_the_programmatic_boundary() -> None:
    with pytest.raises(TypeError, match="event_kind"):
        _event(event_kind="run_start")
    with pytest.raises(TypeError, match="run_class"):
        _event(run_class="confirmatory")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda text: text.replace('"event_seq":0,', '"event_seq":0,"event_seq":1,', 1),
            "duplicate",
        ),
        (lambda text: text.replace("0.12345678901234568", "NaN"), "finite"),
        (lambda text: text.replace('"event_seq":0', '"event_seq":true'), "event_seq"),
        (
            lambda text: text.replace(
                '"schema_version":"d1-authoritative-event-v1",',
                '"schema_version":"d1-authoritative-event-v1","unknown":1,',
            ),
            "unknown",
        ),
        (lambda text: text.replace(',"successful_step":0', ""), "missing"),
        (
            lambda text: text.replace('"run_class":"confirmatory"', '"run_class":"other"'),
            "run_class",
        ),
        (
            lambda text: text.replace(
                '"schema_version":"d1-authoritative-event-v1"',
                '"schema_version":"legacy"',
            ),
            "schema_version",
        ),
    ],
)
def test_parser_rejects_duplicate_nonfinite_boolean_unknown_and_missing_fields(
    mutation: object, message: str
) -> None:
    text = canonical_event_line(_event()).decode("ascii").rstrip("\n")
    mutate = mutation
    assert callable(mutate)
    with pytest.raises(D1JournalError, match=message):
        parse_event_line(mutate(text).encode("ascii") + b"\n")  # type: ignore[operator]


def test_parser_requires_the_exact_canonical_jsonl_representation() -> None:
    canonical = canonical_event_line(_event())
    with pytest.raises(D1JournalError, match="newline"):
        parse_event_line(canonical.rstrip(b"\n"))
    with pytest.raises(D1JournalError, match="canonical"):
        parse_event_line(b" " + canonical)
    with pytest.raises(D1JournalError, match="one event"):
        parse_event_line(canonical + canonical)
    with pytest.raises(D1JournalError, match="UTF-8"):
        parse_event_line(b"\xff\n")


def test_checkpoint_state_is_strict_immutable_and_checkpoint_friendly() -> None:
    state = JournalCheckpointState(
        logical_run_id="d1-confirmatory-both-seed-7",
        execution_digest="1" * 64,
        last_event_sequence=8,
        prefix_sha256="a" * 64,
        successful_step=1500,
    )
    expected = {
        "schema_version": "d1-authoritative-journal-v1",
        "logical_run_id": "d1-confirmatory-both-seed-7",
        "execution_digest": "1" * 64,
        "last_event_sequence": 8,
        "prefix_sha256": "a" * 64,
        "successful_step": 1500,
    }
    assert state.manifest == expected
    assert JournalCheckpointState.from_mapping(expected) == state
    with pytest.raises(D1JournalError, match="unknown"):
        JournalCheckpointState.from_mapping({**expected, "run_id": "1" * 64})
    missing = dict(expected)
    del missing["prefix_sha256"]
    with pytest.raises(D1JournalError, match="missing"):
        JournalCheckpointState.from_mapping(missing)
    with pytest.raises(D1JournalError, match="last_event_sequence"):
        replace(state, last_event_sequence=True)  # type: ignore[arg-type]
