"""Strict decoding and canonical byte admission for D1 inspect results."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from entropic_substrate.d1_inspect_codec import (
    D1_INSPECT_MAX_RESULT_BYTES,
    D1InspectCodecError,
    d1_inspect_result_from_mapping,
    d1_inspect_result_to_mapping,
    parse_d1_inspect_result,
    serialize_d1_inspect_result,
)
from entropic_substrate.evidence.d1_journal_values import JournalEventKind

from .d1_telemetry_fixtures import valid_events
from .test_d1_inspect_codec import _view

pytestmark = pytest.mark.unit


def test_complete_mapping_and_canonical_bytes_round_trip_nominal_view(tmp_path: Path) -> None:
    view, _ = _view(tmp_path)
    mapping = d1_inspect_result_to_mapping(view)
    payload = serialize_d1_inspect_result(view)

    assert d1_inspect_result_from_mapping(mapping) == view
    assert parse_d1_inspect_result(payload) == view
    assert serialize_d1_inspect_result(parse_d1_inspect_result(payload)) == payload


@pytest.mark.parametrize("kind", ["partial", "trainer_aborted", "run_aborted"])
def test_every_noncomplete_state_round_trips_without_compatibility_guessing(
    tmp_path: Path, kind: str
) -> None:
    events = valid_events()
    if kind == "partial":
        events = events[:-1]
    elif kind == "trainer_aborted":
        events[-2] = replace(
            events[-2],
            event_kind=JournalEventKind.TRAINER_ABORTED,
            payload={"error": "trainer abort"},
        )
    else:
        events[-1] = replace(
            events[-1],
            event_kind=JournalEventKind.RUN_ABORT,
            payload={"error": "run abort"},
        )
    view, _ = _view(tmp_path, events=events, allow_incomplete=True)

    decoded = d1_inspect_result_from_mapping(d1_inspect_result_to_mapping(view))

    assert decoded == view
    assert decoded.status.value == kind


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("schema_version"), "root keys"),
        (lambda value: value.update({"extra": True}), "root keys"),
        (lambda value: value.update({"schema_version": "unknown"}), "schema"),
        (lambda value: value["state"].update({"status": "unknown"}), "status"),
        (lambda value: value["evaluations"].update({"limit": True}), "integer"),
        (
            lambda value: value["evaluations"]["items"][0]["quality_semantics"].update(
                {"pairwise_cosine": "changed"}
            ),
            "semantic",
        ),
        (
            lambda value: value["evaluations"]["items"][0]["resource_work"].update(
                {"gpu_train_device_seconds": None}
            ),
            "gpu_train_device_seconds",
        ),
    ],
)
def test_mapping_decoder_refuses_missing_unknown_and_nested_invalid_values(
    tmp_path: Path, mutation: Any, match: str
) -> None:
    view, _ = _view(tmp_path)
    value = copy.deepcopy(d1_inspect_result_to_mapping(view))
    mutation(cast(dict[str, Any], value))

    with pytest.raises(D1InspectCodecError, match=match):
        d1_inspect_result_from_mapping(value)


def test_state_discriminator_controls_its_exact_shape(tmp_path: Path) -> None:
    view, _ = _view(tmp_path)
    value = copy.deepcopy(d1_inspect_result_to_mapping(view))
    state = cast(dict[str, object], value["state"])
    state["status"] = "partial"

    with pytest.raises(D1InspectCodecError, match="state keys"):
        d1_inspect_result_from_mapping(value)


def test_byte_parser_rejects_duplicate_nonfinite_noncanonical_and_oversized_json(
    tmp_path: Path,
) -> None:
    view, _ = _view(tmp_path)
    payload = serialize_d1_inspect_result(view)
    duplicate = b'{"schema_version":"duplicate",' + payload[1:]

    for invalid, match in (
        (duplicate, "duplicate"),
        (b'{"metric":NaN}', "non-finite"),
        (payload + b"\n", "canonical"),
        (b"x" * (D1_INSPECT_MAX_RESULT_BYTES + 1), "byte limit"),
    ):
        with pytest.raises(D1InspectCodecError, match=match):
            parse_d1_inspect_result(invalid)
