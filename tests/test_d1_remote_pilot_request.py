"""Excluded pilots retain their identity across the trainer process boundary."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import d1_launch_admission_fixtures as launch
from entropic_substrate.d1_trainer_request_codec import (
    D1ResumeTrainerRequest,
    build_d1_trainer_request,
    d1_trainer_request_sha256,
    parse_d1_trainer_request,
    serialize_d1_trainer_request,
)
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.resource_provenance import SegmentStartMode, execution_segment_id
from entropic_substrate.experiments.d1_config import load_d1_profile_catalog
from entropic_substrate.experiments.d1_plan import (
    ExcludedPilotD1RunPlan,
    create_d1_excluded_pilot_plan,
    create_d1_smoke_plan,
    parse_d1_confirmatory_plan,
)
from entropic_substrate.experiments.d1_primitives import RunClass

pytestmark = pytest.mark.unit


def _pilot() -> ExcludedPilotD1RunPlan:
    reference = launch.plan(launch.profiles()[0], 101)
    return create_d1_excluded_pilot_plan(
        reference.profile,
        logical_run_id="d1-excluded-pilot-101",
        seed=reference.seed,
        pairing_id="excluded-pilot-101",
        evidence=reference.run.evidence,
        data=reference.run.data,
        code=reference.run.execution.code,
        container=reference.run.execution.container,
    )


@pytest.mark.parametrize("resume", [False, True])
def test_pilot_request_roundtrip_preserves_exclusion_and_all_digests(resume: bool) -> None:
    pilot = _pilot()
    start = launch.start(pilot)
    if resume:
        start = replace(
            start,
            execution_segment_index=1,
            execution_segment_id=execution_segment_id(pilot.logical_run_id, 1),
            segment_start_mode=SegmentStartMode.RESUME,
            resume_checkpoint_sha256="9" * 64,
        )
    request = build_d1_trainer_request(
        pilot, start, D1TrainerWallOrigin(start.trainer_started_at, 123)
    )
    parsed = parse_d1_trainer_request(
        serialize_d1_trainer_request(request), catalog=(pilot.profile,)
    )
    assert isinstance(parsed, D1ResumeTrainerRequest) is resume
    assert type(parsed.plan) is ExcludedPilotD1RunPlan
    assert parsed.plan.run.run_class is RunClass.PILOT
    assert parsed.plan.run.preregistration is parsed.plan.run.family is None
    assert parsed.plan.plan_digest == pilot.plan_digest
    assert parsed.plan.run.execution_digest == pilot.run.execution_digest
    assert d1_trainer_request_sha256(parsed) == d1_trainer_request_sha256(request)
    with pytest.raises(ValueError, match="run_class"):
        parse_d1_confirmatory_plan(parsed.plan.profile, parsed.plan.plan_manifest)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("preregistration", {"path": "forbidden"}, "forbids preregistration"),
        ("family_digest", "0" * 64, "installed construction"),
        ("construction_digest", "0" * 64, "installed construction"),
        ("run_class", "confirmatory", "preregistration"),
        ("seed", True, "seed"),
    ],
)
def test_pilot_request_rejects_identity_and_registration_drift(
    key: str, value: object, message: str
) -> None:
    pilot = _pilot()
    start = launch.start(pilot)
    request = build_d1_trainer_request(
        pilot, start, D1TrainerWallOrigin(start.trainer_started_at, 123)
    )
    payload = json.loads(serialize_d1_trainer_request(request))
    payload["plan"][key] = value
    with pytest.raises(ValueError, match=message):
        parse_d1_trainer_request(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(),
            catalog=(pilot.profile,),
        )


def test_request_selects_remote_tiny_from_full_cpu_and_gpu_catalogs() -> None:
    reference = _pilot()
    gpu = load_d1_profile_catalog(tiny=True, tiny_target="remote_gpu")
    plan = create_d1_smoke_plan(
        gpu[0],
        logical_run_id="gpu-smoke",
        seed=101,
        pairing_id="gpu-pair",
        evidence=reference.run.evidence,
        container=reference.run.execution.container,
    )
    start = launch.start(plan)
    request = build_d1_trainer_request(
        plan, start, D1TrainerWallOrigin(start.trainer_started_at, 123)
    )
    parsed = parse_d1_trainer_request(
        serialize_d1_trainer_request(request),
        catalog=(*load_d1_profile_catalog(), *load_d1_profile_catalog(tiny=True), *gpu),
    )
    assert parsed.plan.profile.scale.value == "gpu_tiny"
    assert parsed.plan.plan_digest == plan.plan_digest
