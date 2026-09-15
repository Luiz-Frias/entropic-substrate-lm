"""Strict command-specific request contract for D1 journal inspection."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.d1_inspect_request import (
    D1_INSPECT_MAX_JOURNAL_BYTES,
    D1_INSPECT_REQUEST_SCHEMA,
    D1InspectRequest,
    d1_inspect_request_from_mapping,
)
from entropic_substrate.d1_inspect_values import D1_INSPECT_MAX_EVALUATIONS
from entropic_substrate.experiments.d1_primitives import RunClass

pytestmark = pytest.mark.unit


def _mapping() -> dict[str, object]:
    return {
        "allow_incomplete": True,
        "evaluation_limit": 8,
        "expected_family_digest": "a" * 64,
        "expected_logical_run_id": "d1-both-s7",
        "expected_plan_digest": "b" * 64,
        "expected_run_class": "pilot",
        "journal_path": "artifacts/run.jsonl",
        "max_journal_bytes": 4096,
        "schema_version": D1_INSPECT_REQUEST_SCHEMA,
    }


def test_request_round_trips_exact_identity_filters_and_relative_path() -> None:
    mapping = _mapping()

    request = d1_inspect_request_from_mapping(mapping)

    assert request == D1InspectRequest(
        Path("artifacts/run.jsonl"),
        4096,
        True,
        8,
        "d1-both-s7",
        "a" * 64,
        RunClass.PILOT,
        "b" * 64,
    )
    assert request.manifest == mapping
    assert d1_inspect_request_from_mapping(request.manifest) == request


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("schema_version"), "keys"),
        (lambda value: value.update({"extra": True}), "keys"),
        (lambda value: value.update({"schema_version": "unknown"}), "schema"),
        (lambda value: value.update({"journal_path": ""}), "journal_path"),
        (lambda value: value.update({"journal_path": "bad\x00path"}), "journal_path"),
        (lambda value: value.update({"max_journal_bytes": True}), "max_journal_bytes"),
        (lambda value: value.update({"max_journal_bytes": 0}), "max_journal_bytes"),
        (
            lambda value: value.update({"max_journal_bytes": D1_INSPECT_MAX_JOURNAL_BYTES + 1}),
            "max_journal_bytes",
        ),
        (lambda value: value.update({"evaluation_limit": 0}), "evaluation_limit"),
        (
            lambda value: value.update({"evaluation_limit": D1_INSPECT_MAX_EVALUATIONS + 1}),
            "evaluation_limit",
        ),
        (lambda value: value.update({"allow_incomplete": 1}), "allow_incomplete"),
        (lambda value: value.update({"expected_family_digest": "bad"}), "family"),
        (lambda value: value.update({"expected_plan_digest": "bad"}), "plan"),
        (lambda value: value.update({"expected_run_class": "unknown"}), "run_class"),
        (lambda value: value.update({"expected_logical_run_id": "  "}), "logical_run_id"),
    ],
)
def test_request_refuses_missing_unknown_invalid_and_over_budget_fields(
    mutation: Any, match: str
) -> None:
    value = copy.deepcopy(_mapping())
    mutation(value)

    with pytest.raises((TypeError, ValueError), match=match):
        d1_inspect_request_from_mapping(value)


def test_optional_identity_filters_are_explicit_nulls_not_inferred_defaults() -> None:
    value = _mapping()
    for field in (
        "expected_family_digest",
        "expected_logical_run_id",
        "expected_plan_digest",
        "expected_run_class",
    ):
        value[field] = None

    request = d1_inspect_request_from_mapping(value)

    assert request.expected_family_digest is None
    assert request.expected_logical_run_id is None
    assert request.expected_plan_digest is None
    assert request.expected_run_class is None
    assert request.manifest == value
