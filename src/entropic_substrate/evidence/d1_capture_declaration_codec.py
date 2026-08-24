"""Strict canonical codecs for D1 capture artifacts and launch declarations."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import NoReturn, cast

from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    D1DvcBinding,
    D1DvcBindingState,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1ExpectedRun,
    D1LaunchRoster,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

__all__ = [
    "D1CaptureDeclarationCodecError",
    "artifact_identity_from_mapping",
    "artifact_identity_to_mapping",
    "launch_roster_from_mapping",
    "launch_roster_to_mapping",
    "parse_artifact_identity",
    "parse_launch_roster",
    "serialize_artifact_identity",
    "serialize_launch_roster",
]

_ARTIFACT_KEYS = frozenset({"dvc", "md5", "path", "role", "sha256", "size_bytes"})
_DVC_KEYS = frozenset({"object_id", "pointer_path", "pointer_sha256", "state"})
_EXPECTED_KEYS = frozenset(
    {
        "aggregation_digest",
        "arm",
        "code_commit",
        "construction_digest",
        "dataset_digest",
        "execution_digest",
        "family_digest",
        "image_digest",
        "logical_run_id",
        "pairing_id",
        "plan_digest",
        "repository_digest",
        "sampler_construction_digest",
        "seed",
        "split_digest",
    }
)
_ATTEMPT_KEYS = frozenset(
    {"arm", "disposition", "lane", "logical_run_id", "pairing_id", "run_class", "seed"}
)
_ROSTER_KEYS = frozenset(
    {
        "attempts",
        "expected_confirmatory",
        "family_digest",
        "preregistration_commit",
        "preregistration_path",
        "schema_version",
    }
)
_ROSTER_SCHEMA = "d1-launch-roster-v1"
_PAIRED_FIELDS = (
    "code_commit",
    "repository_digest",
    "image_digest",
    "dataset_digest",
    "split_digest",
    "sampler_construction_digest",
)


class D1CaptureDeclarationCodecError(ValueError):
    """Persisted capture declarations are malformed, permissive, or drifted."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1CaptureDeclarationCodecError(message)
    raise D1CaptureDeclarationCodecError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _exact(value: Mapping[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"{label} keys must be exact")


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be a nonempty trimmed string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _array(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{label} must be a JSON array")
    return cast(list[object], value)


def _canonical(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _dvc_binding(value: object) -> D1DvcBinding:
    item = _mapping(value, label="artifact.dvc")
    _exact(item, _DVC_KEYS, label="artifact.dvc")
    try:
        return D1DvcBinding(
            state=D1DvcBindingState(_text(item["state"], label="artifact.dvc.state")),
            pointer_path=_text(item["pointer_path"], label="artifact.dvc.pointer_path"),
            pointer_sha256=_optional_text(
                item["pointer_sha256"], label="artifact.dvc.pointer_sha256"
            ),
            object_id=_optional_text(item["object_id"], label="artifact.dvc.object_id"),
        )
    except (TypeError, ValueError) as error:
        _fail(f"artifact DVC binding is invalid: {error}", error)


def artifact_identity_from_mapping(value: object) -> D1ArtifactIdentity:
    root = _mapping(value, label="artifact identity")
    _exact(root, _ARTIFACT_KEYS, label="artifact identity")
    try:
        artifact = D1ArtifactIdentity(
            normalized_path=_text(root["path"], label="artifact.path"),
            role=D1ArtifactRole(_text(root["role"], label="artifact.role")),
            size_bytes=_integer(root["size_bytes"], label="artifact.size_bytes", positive=True),
            md5=_text(root["md5"], label="artifact.md5"),
            sha256=_text(root["sha256"], label="artifact.sha256"),
            dvc=_dvc_binding(root["dvc"]),
        )
    except D1CaptureDeclarationCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"artifact identity is invalid: {error}", error)
    if _canonical(root) != _canonical(artifact.manifest):
        _fail("artifact identity mapping is not canonical")
    return artifact


def artifact_identity_to_mapping(artifact: D1ArtifactIdentity) -> dict[str, object]:
    if type(artifact) is not D1ArtifactIdentity:
        raise TypeError("artifact declaration codec requires D1ArtifactIdentity")
    return copy.deepcopy(artifact.manifest)


def serialize_artifact_identity(artifact: D1ArtifactIdentity) -> str:
    return _canonical(artifact_identity_to_mapping(artifact))


def _expected_run(value: object, *, path: str) -> D1ExpectedRun:
    item = _mapping(value, label=path)
    _exact(item, _EXPECTED_KEYS, label=path)
    try:
        return D1ExpectedRun(
            logical_run_id=_text(item["logical_run_id"], label=f"{path}.logical_run_id"),
            arm=D1Arm(_text(item["arm"], label=f"{path}.arm")),
            seed=_integer(item["seed"], label=f"{path}.seed"),
            pairing_id=_text(item["pairing_id"], label=f"{path}.pairing_id"),
            family_digest=_text(item["family_digest"], label=f"{path}.family_digest"),
            construction_digest=_text(
                item["construction_digest"], label=f"{path}.construction_digest"
            ),
            execution_digest=_text(item["execution_digest"], label=f"{path}.execution_digest"),
            aggregation_digest=_text(
                item["aggregation_digest"], label=f"{path}.aggregation_digest"
            ),
            plan_digest=_text(item["plan_digest"], label=f"{path}.plan_digest"),
            code_commit=_text(item["code_commit"], label=f"{path}.code_commit"),
            repository_digest=_text(item["repository_digest"], label=f"{path}.repository_digest"),
            image_digest=_text(item["image_digest"], label=f"{path}.image_digest"),
            dataset_digest=_text(item["dataset_digest"], label=f"{path}.dataset_digest"),
            split_digest=_text(item["split_digest"], label=f"{path}.split_digest"),
            sampler_construction_digest=_text(
                item["sampler_construction_digest"],
                label=f"{path}.sampler_construction_digest",
            ),
        )
    except (TypeError, ValueError) as error:
        _fail(f"{path} is invalid: {error}", error)


def _attempt(value: object, *, path: str) -> D1DeclaredAttempt:
    item = _mapping(value, label=path)
    _exact(item, _ATTEMPT_KEYS, label=path)
    try:
        return D1DeclaredAttempt(
            logical_run_id=_text(item["logical_run_id"], label=f"{path}.logical_run_id"),
            arm=D1Arm(_text(item["arm"], label=f"{path}.arm")),
            seed=_integer(item["seed"], label=f"{path}.seed"),
            pairing_id=_text(item["pairing_id"], label=f"{path}.pairing_id"),
            run_class=RunClass(_text(item["run_class"], label=f"{path}.run_class")),
            lane=D1AttemptLane(_text(item["lane"], label=f"{path}.lane")),
            disposition=_text(item["disposition"], label=f"{path}.disposition"),
        )
    except (TypeError, ValueError) as error:
        _fail(f"{path} is invalid: {error}", error)


def _validate_cross_records(roster: D1LaunchRoster) -> None:
    by_pair: dict[str, list[D1ExpectedRun]] = {}
    by_arm: dict[D1Arm, set[str]] = {}
    for run in roster.expected_confirmatory:
        by_pair.setdefault(run.pairing_id, []).append(run)
        by_arm.setdefault(run.arm, set()).add(run.construction_digest)
    if any(len(digests) != 1 for digests in by_arm.values()):
        _fail("one arm carries construction identity drift across seed blocks")
    for pairing_id, block in by_pair.items():
        for field in _PAIRED_FIELDS:
            if len({getattr(run, field) for run in block}) != 1:
                _fail(f"paired block {pairing_id!r} has {field} drift")
    expected = {(run.arm, run.seed, run.pairing_id) for run in roster.expected_confirmatory}
    for attempt in roster.attempts:
        if attempt.lane is D1AttemptLane.EXCLUDED_PILOT:
            continue
        if (attempt.arm, attempt.seed, attempt.pairing_id) not in expected:
            _fail("confirmatory attempt identity does not match an expected roster cell")


def launch_roster_from_mapping(value: object) -> D1LaunchRoster:
    root = _mapping(value, label="launch roster")
    _exact(root, _ROSTER_KEYS, label="launch roster")
    if root["schema_version"] != _ROSTER_SCHEMA:
        _fail("launch roster schema_version is unsupported")
    expected = tuple(
        _expected_run(item, path=f"expected_confirmatory[{index}]")
        for index, item in enumerate(
            _array(root["expected_confirmatory"], label="expected_confirmatory")
        )
    )
    attempts = tuple(
        _attempt(item, path=f"attempts[{index}]")
        for index, item in enumerate(_array(root["attempts"], label="attempts"))
    )
    try:
        roster = D1LaunchRoster(
            preregistration_path=_text(root["preregistration_path"], label="preregistration_path"),
            preregistration_commit=_text(
                root["preregistration_commit"], label="preregistration_commit"
            ),
            family_digest=_text(root["family_digest"], label="family_digest"),
            expected_confirmatory=expected,
            attempts=attempts,
        )
        _validate_cross_records(roster)
    except D1CaptureDeclarationCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"launch roster is invalid: {error}", error)
    if _canonical(root) != _canonical(roster.manifest):
        _fail("launch roster mapping is not canonical")
    return roster


def launch_roster_to_mapping(roster: D1LaunchRoster) -> dict[str, object]:
    if type(roster) is not D1LaunchRoster:
        raise TypeError("roster declaration codec requires D1LaunchRoster")
    return copy.deepcopy(roster.manifest)


def serialize_launch_roster(roster: D1LaunchRoster) -> str:
    return _canonical(launch_roster_to_mapping(roster))


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate capture-declaration JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite capture-declaration JSON constant {value!r}")


def _decode(payload: str, *, label: str) -> object:
    if type(payload) is not str or not payload:
        _fail(f"{label} payload must be nonempty text")
    try:
        return json.loads(payload, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except D1CaptureDeclarationCodecError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        _fail(f"{label} payload is invalid JSON", error)


def parse_artifact_identity(payload: str) -> D1ArtifactIdentity:
    artifact = artifact_identity_from_mapping(_decode(payload, label="artifact identity"))
    if serialize_artifact_identity(artifact) != payload:
        _fail("artifact identity JSON is not canonical")
    return artifact


def parse_launch_roster(payload: str) -> D1LaunchRoster:
    roster = launch_roster_from_mapping(_decode(payload, label="launch roster"))
    if serialize_launch_roster(roster) != payload:
        _fail("launch roster JSON is not canonical")
    return roster
