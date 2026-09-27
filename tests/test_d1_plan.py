"""D1 run-plan typestate and strict confirmatory manifest tests."""

from __future__ import annotations

from typing import cast

import pytest

from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass, RunClass
from entropic_substrate.ladder.register.d1_config import (
    D1LaunchProfile,
    load_d1_profile_catalog,
)
from entropic_substrate.ladder.register.d1_plan import (
    ConfirmatoryD1RunPlan,
    D1PlanManifestError,
    ExcludedPilotD1RunPlan,
    SmokeD1RunPlan,
    create_d1_excluded_pilot_plan,
    create_d1_smoke_plan,
    parse_d1_confirmatory_plan,
)
from entropic_substrate.ladder.register.d1_provenance import (
    D1_TELEMETRY_SCHEMA_VERSION,
    DataIdentity,
    EvidenceEnvelope,
)

pytestmark = pytest.mark.unit


def _profile(name: str = "d1-both", *, tiny: bool = False) -> D1LaunchProfile:
    return next(profile for profile in load_d1_profile_catalog(tiny=tiny) if profile.name == name)


def _evidence() -> EvidenceEnvelope:
    return EvidenceEnvelope(
        schema_version="d1-evidence-v1",
        required_classes=frozenset({EvidenceClass.MEASURED, EvidenceClass.DERIVED_FROM_MEASURED}),
    )


def _data() -> DataIdentity:
    return DataIdentity(
        dataset_name="enwik8",
        dataset_digest="1" * 64,
        split_digest="2" * 64,
        sampler_construction_digest="3" * 64,
    )


def test_new_evidence_envelopes_select_direct_channel_telemetry_schema_v3() -> None:
    assert _evidence().telemetry_schema_version == D1_TELEMETRY_SCHEMA_VERSION == 3
    with pytest.raises(ValueError, match="telemetry_schema_version must be 3"):
        EvidenceEnvelope(
            schema_version="d1-evidence-v1",
            required_classes=frozenset({EvidenceClass.MEASURED}),
            telemetry_schema_version=1,
        )
    with pytest.raises(ValueError, match="telemetry_schema_version must be 3"):
        float_version = cast(int, 3.0)
        EvidenceEnvelope(
            schema_version="d1-evidence-v1",
            required_classes=frozenset({EvidenceClass.MEASURED}),
            telemetry_schema_version=float_version,
        )


def _confirmatory_manifest(profile: D1LaunchProfile) -> dict[str, object]:
    return {
        "schema_version": "d1-run-plan-v1",
        "run_class": "confirmatory",
        "profile_name": profile.name,
        "arm": profile.arm.value,
        "family_digest": profile.family.family_digest,
        "construction_digest": profile.draft.construction_digest,
        "logical_run_id": "d1-confirmatory-both-seed-7",
        "seed": 7,
        "pairing_id": "d1-stabilization-pair-7",
        "code": {"commit": "a" * 40, "repository_digest": "b" * 64},
        "container": {"image_digest": "sha256:" + "c" * 64},
        "data": _data().manifest,
        "evidence": _evidence().manifest,
        "preregistration": {
            "path": "docs/d1-preregistration.md",
            "commit": "d" * 40,
            "family_digest": profile.family.family_digest,
        },
    }


def _set_path(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target = payload
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value


def test_smoke_and_pilot_factories_create_excluded_nominal_states() -> None:
    profile = _profile()
    smoke = create_d1_smoke_plan(
        profile,
        logical_run_id="d1-smoke-both-seed-3",
        seed=3,
        pairing_id="smoke-3",
        evidence=_evidence(),
    )
    pilot = create_d1_excluded_pilot_plan(
        profile,
        logical_run_id="d1-pilot-both-seed-3",
        seed=3,
        pairing_id="pilot-3",
        evidence=_evidence(),
        data=_data(),
    )

    assert type(smoke) is SmokeD1RunPlan
    assert type(pilot) is ExcludedPilotD1RunPlan
    assert smoke.run.run_class is RunClass.SMOKE
    assert pilot.run.run_class is RunClass.PILOT
    assert smoke.run.preregistration is pilot.run.preregistration is None
    assert smoke.run.family is pilot.run.family is None


def test_plan_seed_overrides_profile_adapter_but_not_family_identity() -> None:
    profile = _profile()
    first = create_d1_smoke_plan(
        profile,
        logical_run_id="d1-smoke-both-seed-7",
        seed=7,
        pairing_id="smoke-7",
        evidence=_evidence(),
    )
    second = create_d1_smoke_plan(
        profile,
        logical_run_id="d1-smoke-both-seed-8",
        seed=8,
        pairing_id="smoke-8",
        evidence=_evidence(),
    )

    assert profile.train.seed != first.seed
    assert first.train.seed == 7
    assert second.train.seed == 8
    assert profile.train.seed != first.train.seed
    assert first.profile.family.family_digest == second.profile.family.family_digest
    assert first.plan_digest != second.plan_digest
    assert "seed" not in first.profile.family.family_manifest


def test_confirmatory_manifest_round_trips_every_bound_identity() -> None:
    profile = _profile()
    payload = _confirmatory_manifest(profile)
    plan = parse_d1_confirmatory_plan(profile, payload)

    assert type(plan) is ConfirmatoryD1RunPlan
    assert plan.run.run_class is RunClass.CONFIRMATORY
    assert plan.run.family == profile.family
    assert plan.plan_manifest == payload
    assert len(plan.plan_digest) == 64
    assert plan.train.seed == payload["seed"]
    returned = plan.plan_manifest
    code = returned["code"]
    assert isinstance(code, dict)
    code["commit"] = "f" * 40
    assert plan.plan_manifest == payload


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), "d1-run-plan-v2", "schema_version"),
        (("run_class",), "pilot", "run_class"),
        (("profile_name",), "d1-sg", "profile_name"),
        (("arm",), "sg", "arm"),
        (("family_digest",), "f" * 64, "family_digest"),
        (("construction_digest",), "e" * 64, "construction_digest"),
        (("seed",), True, "seed"),
        (("code", "commit"), True, "code.commit"),
        (("evidence", "checkpoint_schema_version"), True, "checkpoint_schema_version"),
        (("preregistration", "family_digest"), "0" * 64, "family_digest"),
    ],
)
def test_confirmatory_parser_rejects_type_and_identity_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    profile = _profile()
    payload = _confirmatory_manifest(profile)
    _set_path(payload, path, value)

    with pytest.raises(D1PlanManifestError, match=message):
        parse_d1_confirmatory_plan(profile, payload)


def test_confirmatory_parser_rejects_unknown_missing_and_nested_extra_keys() -> None:
    profile = _profile()
    unknown = _confirmatory_manifest(profile)
    unknown["provider"] = "runpod"
    missing = _confirmatory_manifest(profile)
    missing.pop("pairing_id")
    nested = _confirmatory_manifest(profile)
    code = nested["code"]
    assert isinstance(code, dict)
    code["branch"] = "main"

    for payload, message in (
        (unknown, "unknown keys"),
        (missing, "missing keys"),
        (nested, "unknown keys"),
    ):
        with pytest.raises(D1PlanManifestError, match=message):
            parse_d1_confirmatory_plan(profile, payload)


def test_confirmatory_requires_full_profile_and_manifest_has_no_placement_state() -> None:
    full = _profile()
    tiny = _profile(tiny=True)
    tiny_payload = _confirmatory_manifest(tiny)
    with pytest.raises(D1PlanManifestError, match="full launch profile"):
        parse_d1_confirmatory_plan(tiny, tiny_payload)

    manifest_text = repr(
        parse_d1_confirmatory_plan(full, _confirmatory_manifest(full)).plan_manifest
    )
    for forbidden in ("results_dir", "models_dir", "device", "provider", "region"):
        assert forbidden not in manifest_text
