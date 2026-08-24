"""Canonical declaration codecs for D1 artifacts, rosters, and attempts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    D1DvcBinding,
)
from entropic_substrate.evidence.d1_capture_declaration_codec import (
    D1CaptureDeclarationCodecError,
    artifact_identity_from_mapping,
    artifact_identity_to_mapping,
    launch_roster_from_mapping,
    launch_roster_to_mapping,
    parse_artifact_identity,
    parse_launch_roster,
    serialize_artifact_identity,
    serialize_launch_roster,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1LaunchRoster,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

from .test_d1_capture_roster import _family, _run

pytestmark = pytest.mark.unit


def _artifact(*, bound: bool = True) -> D1ArtifactIdentity:
    dvc = D1DvcBinding.pending("results/captured.dvc")
    if bound:
        dvc = dvc.bind(pointer_sha256="1" * 64, object_id="2" * 32 + ".dir")
    return D1ArtifactIdentity(
        normalized_path="journals/d1-none-s7.jsonl",
        role=D1ArtifactRole.JOURNAL,
        size_bytes=123,
        md5="3" * 32,
        sha256="4" * 64,
        dvc=dvc,
    )


def _attempt() -> D1DeclaredAttempt:
    return D1DeclaredAttempt(
        logical_run_id="d1-sg-s7-attempt-0",
        arm=D1Arm.SG,
        seed=7,
        pairing_id="pair-7",
        run_class=RunClass.CONFIRMATORY,
        lane=D1AttemptLane.ABORTED,
        disposition="trainer recorded a terminal defect",
    )


def _roster(*, two_seeds: bool = False) -> D1LaunchRoster:
    expected = _family()
    if two_seeds:
        expected += tuple(_run(arm, seed=8, pairing_id="pair-8") for arm in D1Arm)
    return D1LaunchRoster(
        preregistration_path="docs/d1-preregistration.md",
        preregistration_commit="abcdef0",
        family_digest="a" * 64,
        expected_confirmatory=expected,
        attempts=(_attempt(),),
    )


@pytest.mark.parametrize("artifact", [_artifact(bound=False), _artifact(bound=True)])
def test_artifact_and_dvc_binding_round_trip_exact_canonical_manifest(
    artifact: D1ArtifactIdentity,
) -> None:
    mapping = artifact_identity_to_mapping(artifact)
    payload = serialize_artifact_identity(artifact)

    assert artifact_identity_from_mapping(mapping) == artifact
    assert parse_artifact_identity(payload) == artifact
    assert mapping == artifact.manifest
    assert payload == json.dumps(mapping, separators=(",", ":"), sort_keys=True)


def test_roster_round_trip_preserves_expected_blocks_and_attempt_declarations() -> None:
    roster = _roster(two_seeds=True)
    mapping = launch_roster_to_mapping(roster)
    payload = serialize_launch_roster(roster)
    restored = parse_launch_roster(payload)

    assert restored == roster
    assert launch_roster_from_mapping(mapping) == roster
    assert mapping == roster.manifest
    assert payload == json.dumps(mapping, separators=(",", ":"), sort_keys=True)


def _unknown_artifact(value: dict[str, object]) -> None:
    value["invented"] = True


def _missing_artifact(value: dict[str, object]) -> None:
    value.pop("sha256")


def _bad_role(value: dict[str, object]) -> None:
    value["role"] = "invented"


def _bad_dvc(value: dict[str, object]) -> None:
    value["dvc"]["pointer_sha256"] = None  # type: ignore[index]


def _bad_size(value: dict[str, object]) -> None:
    value["size_bytes"] = True


@pytest.mark.parametrize(
    "mutate", [_unknown_artifact, _missing_artifact, _bad_role, _bad_dvc, _bad_size]
)
def test_artifact_decoder_rejects_unknown_missing_role_binding_or_type_drift(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    mapping = artifact_identity_to_mapping(_artifact())
    mutate(mapping)
    with pytest.raises(D1CaptureDeclarationCodecError):
        artifact_identity_from_mapping(mapping)


def test_roster_decoder_rejects_nested_unknown_missing_and_cross_record_drift() -> None:
    mapping = launch_roster_to_mapping(_roster(two_seeds=True))
    mapping["expected_confirmatory"][0]["invented"] = True  # type: ignore[index]
    with pytest.raises(D1CaptureDeclarationCodecError):
        launch_roster_from_mapping(mapping)

    mapping = launch_roster_to_mapping(_roster(two_seeds=True))
    mapping["attempts"][0].pop("disposition")  # type: ignore[index,union-attr]
    with pytest.raises(D1CaptureDeclarationCodecError):
        launch_roster_from_mapping(mapping)

    mapping = launch_roster_to_mapping(_roster(two_seeds=True))
    mapping["expected_confirmatory"][4]["construction_digest"] = "9" * 64  # type: ignore[index]
    with pytest.raises(D1CaptureDeclarationCodecError, match="construction"):
        launch_roster_from_mapping(mapping)

    mapping = launch_roster_to_mapping(_roster(two_seeds=True))
    mapping["expected_confirmatory"][1]["image_digest"] = "sha256:" + "8" * 64  # type: ignore[index]
    with pytest.raises(D1CaptureDeclarationCodecError, match="paired"):
        launch_roster_from_mapping(mapping)


def test_confirmatory_attempt_may_remain_off_roster_nondecision_evidence() -> None:
    roster = _roster()
    off_roster = replace(
        roster.attempts[0],
        seed=41,
        pairing_id="attempt-41",
    )
    roster = replace(roster, attempts=(off_roster,))

    assert launch_roster_from_mapping(launch_roster_to_mapping(roster)) == roster


@pytest.mark.parametrize(
    ("parse", "payload"),
    [
        (parse_artifact_identity, '{"path":"a","path":"b"}'),
        (parse_artifact_identity, '{"size_bytes":NaN}'),
        (parse_launch_roster, " {}"),
        (parse_launch_roster, "{}\n"),
        (parse_launch_roster, "[]"),
    ],
)
def test_text_decoders_reject_duplicate_nonfinite_noncanonical_or_nonobject_json(
    parse: Callable[[str], object], payload: str
) -> None:
    with pytest.raises(D1CaptureDeclarationCodecError):
        parse(payload)


def test_codecs_require_nominal_values_and_mapping_roots() -> None:
    with pytest.raises(TypeError, match="D1ArtifactIdentity"):
        serialize_artifact_identity(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="D1LaunchRoster"):
        serialize_launch_roster(object())  # type: ignore[arg-type]
    with pytest.raises(D1CaptureDeclarationCodecError, match="object"):
        artifact_identity_from_mapping([])
    with pytest.raises(D1CaptureDeclarationCodecError, match="object"):
        launch_roster_from_mapping([])
