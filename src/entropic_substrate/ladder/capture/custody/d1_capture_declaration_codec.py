"""Strict canonical codecs for D1 capture artifacts and launch declarations."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import NoReturn, cast

from entropic_substrate.kernel.codec.strict_json import strict_loads
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate import D1TerminalDefectInput
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    D1DvcBinding,
    D1DvcBindingState,
)
from entropic_substrate.ladder.capture.custody.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureInput,
)
from entropic_substrate.ladder.capture.custody.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1ExpectedRun,
    D1LaunchRoster,
)

__all__ = [
    "D1CaptureDeclarationCodecError",
    "artifact_identity_from_mapping",
    "artifact_identity_to_mapping",
    "attempt_evidence_from_mapping",
    "attempt_evidence_to_mapping",
    "capture_input_from_mapping",
    "capture_input_to_mapping",
    "launch_roster_from_mapping",
    "launch_roster_to_mapping",
    "parse_artifact_identity",
    "parse_attempt_evidence",
    "parse_capture_input",
    "parse_launch_roster",
    "parse_terminal_defect_input",
    "serialize_artifact_identity",
    "serialize_attempt_evidence",
    "serialize_capture_input",
    "serialize_launch_roster",
    "serialize_terminal_defect_input",
    "terminal_defect_input_from_mapping",
    "terminal_defect_input_to_mapping",
    "validate_capture_lane_path_partition",
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
_CAPTURE_INPUT_KEYS = frozenset(
    {"billing_sidecars", "checkpoint", "journal", "logical_run_id", "quarantined_siblings"}
)
_TERMINAL_INPUT_KEYS = frozenset({"journal", "logical_run_id", "supporting_artifacts"})
_ATTEMPT_EVIDENCE_KEYS = frozenset({"artifacts", "attempt", "evidence_gap"})
_TERMINAL_SUPPORT_ROLES = frozenset(
    {
        D1ArtifactRole.BILLING_SIDECAR,
        D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT,
        D1ArtifactRole.QUARANTINED_SIBLING,
    }
)
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


def _decode(payload: str, *, label: str) -> object:
    if type(payload) is not str or not payload:
        _fail(f"{label} payload must be nonempty text")
    try:
        return strict_loads(payload, fail=_fail)
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


def _artifacts(value: object, *, label: str) -> tuple[D1ArtifactIdentity, ...]:
    return tuple(artifact_identity_from_mapping(item) for item in _array(value, label=label))


def capture_input_from_mapping(
    value: object,
    *,
    expected: D1ExpectedRun,
) -> D1CaptureInput:
    if type(expected) is not D1ExpectedRun:
        raise TypeError("complete capture declaration requires D1ExpectedRun")
    root = _mapping(value, label="complete capture input")
    _exact(root, _CAPTURE_INPUT_KEYS, label="complete capture input")
    try:
        capture = D1CaptureInput(
            logical_run_id=_text(root["logical_run_id"], label="complete.logical_run_id"),
            journal=artifact_identity_from_mapping(root["journal"]),
            checkpoint=artifact_identity_from_mapping(root["checkpoint"]),
            billing_sidecars=_artifacts(root["billing_sidecars"], label="billing_sidecars"),
            quarantined_siblings=_artifacts(
                root["quarantined_siblings"], label="quarantined_siblings"
            ),
        )
    except D1CaptureDeclarationCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"complete capture input is invalid: {error}", error)
    if capture.logical_run_id != expected.logical_run_id:
        _fail("complete capture logical_run_id differs from supplied expected run")
    if _canonical(root) != _canonical(capture.manifest):
        _fail("complete capture input mapping is not canonical")
    return capture


def capture_input_to_mapping(capture: D1CaptureInput) -> dict[str, object]:
    if type(capture) is not D1CaptureInput:
        raise TypeError("complete capture codec requires D1CaptureInput")
    return copy.deepcopy(capture.manifest)


def serialize_capture_input(capture: D1CaptureInput) -> str:
    return _canonical(capture_input_to_mapping(capture))


def terminal_defect_input_from_mapping(
    value: object,
    *,
    expected: D1ExpectedRun,
) -> D1TerminalDefectInput:
    if type(expected) is not D1ExpectedRun:
        raise TypeError("terminal defect declaration requires D1ExpectedRun")
    root = _mapping(value, label="terminal defect input")
    _exact(root, _TERMINAL_INPUT_KEYS, label="terminal defect input")
    supporting = _artifacts(root["supporting_artifacts"], label="supporting_artifacts")
    if any(artifact.role not in _TERMINAL_SUPPORT_ROLES for artifact in supporting):
        _fail("terminal defect supporting artifact has an unsupported role")
    try:
        capture = D1TerminalDefectInput(
            logical_run_id=_text(root["logical_run_id"], label="terminal.logical_run_id"),
            journal=artifact_identity_from_mapping(root["journal"]),
            supporting_artifacts=supporting,
        )
    except D1CaptureDeclarationCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"terminal defect input is invalid: {error}", error)
    if capture.logical_run_id != expected.logical_run_id:
        _fail("terminal defect logical_run_id differs from supplied expected run")
    if _canonical(root) != _canonical(capture.manifest):
        _fail("terminal defect input mapping is not canonical")
    return capture


def terminal_defect_input_to_mapping(capture: D1TerminalDefectInput) -> dict[str, object]:
    if type(capture) is not D1TerminalDefectInput:
        raise TypeError("terminal defect codec requires D1TerminalDefectInput")
    return copy.deepcopy(capture.manifest)


def serialize_terminal_defect_input(capture: D1TerminalDefectInput) -> str:
    return _canonical(terminal_defect_input_to_mapping(capture))


def attempt_evidence_from_mapping(
    value: object,
    *,
    declared: D1DeclaredAttempt,
) -> D1AttemptEvidence:
    if type(declared) is not D1DeclaredAttempt:
        raise TypeError("attempt evidence declaration requires D1DeclaredAttempt")
    root = _mapping(value, label="attempt evidence")
    _exact(root, _ATTEMPT_EVIDENCE_KEYS, label="attempt evidence")
    parsed_attempt = _attempt(root["attempt"], path="attempt evidence.attempt")
    if parsed_attempt != declared:
        _fail("attempt evidence embedded declaration differs from supplied declaration")
    gap = root["evidence_gap"]
    if gap is not None:
        gap = _text(gap, label="attempt evidence.evidence_gap")
    try:
        evidence = D1AttemptEvidence(
            attempt=parsed_attempt,
            artifacts=_artifacts(root["artifacts"], label="attempt evidence.artifacts"),
            evidence_gap=gap,
        )
    except D1CaptureDeclarationCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"attempt evidence is invalid: {error}", error)
    if _canonical(root) != _canonical(evidence.manifest):
        _fail("attempt evidence mapping is not canonical")
    return evidence


def attempt_evidence_to_mapping(evidence: D1AttemptEvidence) -> dict[str, object]:
    if type(evidence) is not D1AttemptEvidence:
        raise TypeError("attempt evidence codec requires D1AttemptEvidence")
    return copy.deepcopy(evidence.manifest)


def serialize_attempt_evidence(evidence: D1AttemptEvidence) -> str:
    return _canonical(attempt_evidence_to_mapping(evidence))


def parse_capture_input(payload: str, *, expected: D1ExpectedRun) -> D1CaptureInput:
    capture = capture_input_from_mapping(
        _decode(payload, label="complete capture input"), expected=expected
    )
    if serialize_capture_input(capture) != payload:
        _fail("complete capture input JSON is not canonical")
    return capture


def parse_terminal_defect_input(payload: str, *, expected: D1ExpectedRun) -> D1TerminalDefectInput:
    capture = terminal_defect_input_from_mapping(
        _decode(payload, label="terminal defect input"), expected=expected
    )
    if serialize_terminal_defect_input(capture) != payload:
        _fail("terminal defect input JSON is not canonical")
    return capture


def parse_attempt_evidence(payload: str, *, declared: D1DeclaredAttempt) -> D1AttemptEvidence:
    evidence = attempt_evidence_from_mapping(
        _decode(payload, label="attempt evidence"), declared=declared
    )
    if serialize_attempt_evidence(evidence) != payload:
        _fail("attempt evidence JSON is not canonical")
    return evidence


def validate_capture_lane_path_partition(
    complete: tuple[D1CaptureInput, ...],
    terminal_defects: tuple[D1TerminalDefectInput, ...],
    attempts: tuple[D1AttemptEvidence, ...],
) -> None:
    if type(complete) is not tuple or any(type(item) is not D1CaptureInput for item in complete):
        raise TypeError("complete lane partition requires D1CaptureInput tuple")
    if type(terminal_defects) is not tuple or any(
        type(item) is not D1TerminalDefectInput for item in terminal_defects
    ):
        raise TypeError("terminal lane partition requires D1TerminalDefectInput tuple")
    if type(attempts) is not tuple or any(type(item) is not D1AttemptEvidence for item in attempts):
        raise TypeError("attempt lane partition requires D1AttemptEvidence tuple")
    paths = [artifact.normalized_path for item in complete for artifact in item.artifacts]
    paths.extend(
        artifact.normalized_path for item in terminal_defects for artifact in item.artifacts
    )
    paths.extend(artifact.normalized_path for item in attempts for artifact in item.artifacts)
    if len(paths) != len(set(paths)):
        _fail("capture lane artifact path is reused across declarations")
