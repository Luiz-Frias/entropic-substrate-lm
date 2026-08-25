"""Canonical trainer requests rehydrate the real fresh/resume inputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as launch
from d1_trainer_fixtures import micro_plan, run_start
from entropic_substrate.d1_trainer_request_codec import (
    D1FreshTrainerRequest,
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
    parse_d1_trainer_request,
    serialize_d1_trainer_request,
)
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.resource_provenance import (
    SegmentStartMode,
    execution_segment_id,
)
from entropic_substrate.experiments.d1_plan import SmokeD1RunPlan
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


def _fresh() -> tuple[D1FreshTrainerRequest, tuple[object, ...]]:
    catalog = launch.profiles()
    plan = launch.plan(catalog[0], 101)
    start = launch.start(plan)
    request = D1FreshTrainerRequest(
        plan,
        start,
        D1TrainerWallOrigin(start.trainer_started_at, 123),
        D1TrainerLocationSpec("runs/journal.jsonl", "runs/checkpoint.pt"),
    )
    return request, catalog


def test_fresh_request_round_trips_and_resolves_safe_output_locations(tmp_path: Path) -> None:
    request, catalog = _fresh()
    (tmp_path / "runs").mkdir()

    parsed = parse_d1_trainer_request(
        serialize_d1_trainer_request(request),
        catalog=catalog,
    )
    locations = parsed.resolve_locations(tmp_path)

    assert type(parsed) is D1FreshTrainerRequest
    assert parsed == request
    assert locations.journal_path == tmp_path / "runs/journal.jsonl"
    assert locations.checkpoint_path == tmp_path / "runs/checkpoint.pt"


def test_tiny_smoke_request_round_trips_without_confirmatory_reinterpretation() -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    request = D1FreshTrainerRequest(
        plan,
        start,
        D1TrainerWallOrigin(start.trainer_started_at, 123),
        D1TrainerLocationSpec("runs/journal.jsonl", "runs/checkpoint.pt"),
    )

    parsed = parse_d1_trainer_request(
        serialize_d1_trainer_request(request),
        catalog=(plan.profile,),
    )

    assert type(parsed.plan) is SmokeD1RunPlan
    assert parsed == request


def test_resume_request_round_trips_with_checkpoint_source_and_quarantine(
    tmp_path: Path,
) -> None:
    fresh, catalog = _fresh()
    checkpoint = tmp_path / "runs/checkpoint.pt"
    journal = tmp_path / "runs/journal.jsonl"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    journal.write_bytes(b"journal\n")
    digest = "9" * 64
    start = replace(
        fresh.start,
        execution_segment_index=1,
        execution_segment_id=execution_segment_id(fresh.plan.logical_run_id, 1),
        segment_start_mode=SegmentStartMode.RESUME,
        resume_checkpoint_sha256=digest,
    )
    request = D1ResumeTrainerRequest(
        fresh.plan,
        start,
        D1TrainerWallOrigin(start.trainer_started_at, 456),
        D1TrainerLocationSpec(
            "runs/journal.jsonl",
            "runs/checkpoint.pt",
            "runs/quarantined-tail.jsonl",
        ),
    )

    parsed = parse_d1_trainer_request(
        serialize_d1_trainer_request(request),
        catalog=catalog,
    )
    locations = parsed.resolve_locations(tmp_path)

    assert type(parsed) is D1ResumeTrainerRequest
    assert parsed == request
    assert locations.checkpoint_path == checkpoint
    assert locations.journal_path == journal
    assert locations.quarantine_path == tmp_path / "runs/quarantined-tail.jsonl"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"command": "d1-run-resume"}), "segment mode"),
        (lambda value: value["plan"].update({"profile_name": "unknown"}), "profile"),
        (lambda value: value["locations"].update({"journal_path": "../escape"}), "path"),
        (lambda value: value.update({"extra": True}), "keys"),
    ],
)
def test_request_refuses_command_profile_path_and_schema_drift(
    mutation: object,
    message: str,
) -> None:
    request, catalog = _fresh()
    value = json.loads(serialize_d1_trainer_request(request))
    mutation(value)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        parse_d1_trainer_request(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode(),
            catalog=catalog,
        )


def test_request_refuses_duplicate_and_noncanonical_json() -> None:
    request, catalog = _fresh()
    duplicate = serialize_d1_trainer_request(request).replace(
        b'{"command":',
        b'{"command":"d1-run-fresh","command":',
        1,
    )
    indented = json.dumps(json.loads(serialize_d1_trainer_request(request)), indent=2).encode()

    with pytest.raises(ValueError, match="duplicate"):
        parse_d1_trainer_request(duplicate, catalog=catalog)
    with pytest.raises(ValueError, match="canonical"):
        parse_d1_trainer_request(indented, catalog=catalog)
