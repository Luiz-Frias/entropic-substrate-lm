"""Canonical trainer requests rehydrate the real fresh/resume inputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

import d1_launch_admission_fixtures as launch
from d1_trainer_fixtures import micro_plan, run_start
from entropic_substrate.d1_trainer_request_codec import (
    D1FreshTrainerRequest,
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
    build_d1_trainer_request,
    parse_d1_trainer_request,
    serialize_d1_trainer_request,
)
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.resource_provenance import (
    SegmentStartMode,
    execution_segment_id,
)
from entropic_substrate.experiments.d1_config import (
    D1LaunchScale,
    load_d1_profile_catalog,
)
from entropic_substrate.experiments.d1_plan import (
    SmokeD1RunPlan,
    create_d1_smoke_plan,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, EvidenceClass
from entropic_substrate.experiments.d1_provenance import EvidenceEnvelope

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


def test_request_builder_owns_the_deterministic_per_run_location_policy() -> None:
    request, _ = _fresh()
    built = build_d1_trainer_request(request.plan, request.start, request.origin)
    assert type(built) is D1FreshTrainerRequest
    other_execution = replace(request.plan.run.execution, seed=request.plan.seed + 1)
    other_run = replace(request.plan.run, execution=other_execution)
    other = build_d1_trainer_request(
        replace(request.plan, run=other_run), request.start, request.origin
    )
    run_root = PurePosixPath(built.locations.checkpoint_path).parent
    assert run_root.parent == PurePosixPath("results")
    assert run_root.name.startswith("d1-run-")
    assert built.locations.journal_path == f"{run_root.as_posix()}/journal.jsonl"
    assert built.locations.checkpoint_path == f"{run_root.as_posix()}/checkpoint.pt"
    assert PurePosixPath(f"{run_root.as_posix()}.dvc").parent == PurePosixPath("results")
    assert other.locations.journal_path != built.locations.journal_path


def test_non_smoke_plan_rejects_missing_container_identity() -> None:
    request, _ = _fresh()
    execution = replace(request.plan.run.execution, container=None)
    with pytest.raises(ValueError, match="container identity"):
        replace(request.plan.run, execution=execution)


def test_request_directly_binds_plan_to_start_identity_and_runtime_posture() -> None:
    request, _ = _fresh()
    with pytest.raises(ValueError, match="logical identities"):
        replace(
            request,
            start=replace(
                request.start,
                logical_run_id="another-run",
                execution_segment_id=execution_segment_id("another-run", 0),
            ),
        )
    with pytest.raises(ValueError, match="image"):
        replace(request, start=replace(request.start, image_digest="sha256:" + "9" * 64))
    with pytest.raises(ValueError, match="compiler posture"):
        replace(request, start=replace(request.start, compiler_enabled=True))


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


def test_catalog_resolves_reused_profile_name_by_confirmatory_scale() -> None:
    request, full_catalog = _fresh()
    tiny_catalog = load_d1_profile_catalog(tiny=True)

    parsed = parse_d1_trainer_request(
        serialize_d1_trainer_request(request),
        catalog=(*full_catalog, *tiny_catalog),
    )

    assert parsed.plan.profile.scale is D1LaunchScale.FULL


@pytest.mark.parametrize("tiny", [False, True])
def test_catalog_resolves_full_and_tiny_smoke_by_construction_identity(tiny: bool) -> None:
    full_catalog = load_d1_profile_catalog()
    tiny_catalog = load_d1_profile_catalog(tiny=True)
    profile = (tiny_catalog if tiny else full_catalog)[-1]
    plan = create_d1_smoke_plan(
        profile,
        logical_run_id=f"d1-smoke-catalog-{profile.scale.value}",
        seed=41,
        pairing_id="d1-smoke-catalog-pair-41",
        evidence=EvidenceEnvelope(
            schema_version="d1-evidence-v1",
            required_classes=frozenset({EvidenceClass.MEASURED}),
        ),
    )
    start = run_start(plan)
    request = D1FreshTrainerRequest(
        plan,
        start,
        D1TrainerWallOrigin(start.trainer_started_at, 123),
        D1TrainerLocationSpec("runs/journal.jsonl", "runs/checkpoint.pt"),
    )

    parsed = parse_d1_trainer_request(
        serialize_d1_trainer_request(request),
        catalog=(*full_catalog, *tiny_catalog),
    )

    expected = D1LaunchScale.TINY if tiny else D1LaunchScale.FULL
    assert parsed.plan.profile.scale is expected


def test_catalog_refuses_two_profiles_with_the_same_serialized_construction() -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    request = D1FreshTrainerRequest(
        plan,
        start,
        D1TrainerWallOrigin(start.trainer_started_at, 123),
        D1TrainerLocationSpec("runs/journal.jsonl", "runs/checkpoint.pt"),
    )
    same_construction_at_another_scale = replace(
        plan.profile,
        scale=D1LaunchScale.FULL,
    )

    with pytest.raises(ValueError, match="exactly one installed construction"):
        parse_d1_trainer_request(
            serialize_d1_trainer_request(request),
            catalog=(plan.profile, same_construction_at_another_scale),
        )


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
    built1 = build_d1_trainer_request(fresh.plan, start, request.origin)
    start2 = replace(
        start,
        execution_segment_index=2,
        execution_segment_id=execution_segment_id(fresh.plan.logical_run_id, 2),
    )
    built2 = build_d1_trainer_request(fresh.plan, start2, request.origin)
    fresh_built = build_d1_trainer_request(fresh.plan, fresh.start, fresh.origin)
    assert built1.locations.journal_path == built2.locations.journal_path
    assert built1.locations.journal_path == fresh_built.locations.journal_path
    assert built1.locations.checkpoint_path == fresh_built.locations.checkpoint_path
    assert built1.locations.quarantine_path.endswith("-000001.jsonl")
    assert built2.locations.quarantine_path.endswith("-000002.jsonl")


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
