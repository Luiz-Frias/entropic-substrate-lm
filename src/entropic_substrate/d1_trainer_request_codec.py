"""Strict process-boundary codec for real D1 fresh and resume trainers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations
from entropic_substrate.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
)
from entropic_substrate.evidence.resource_provenance_codec import (
    ResourceProvenanceCodecGap,
    run_start_from_mapping,
    run_start_to_mapping,
)
from entropic_substrate.experiments.d1_config import D1LaunchProfile
from entropic_substrate.experiments.d1_plan import (
    ConfirmatoryD1RunPlan,
    D1PlanManifestError,
    D1RunPlan,
    SmokeD1RunPlan,
    parse_d1_confirmatory_plan,
    parse_d1_smoke_plan,
)
from entropic_substrate.experiments.d1_primitives import (
    RunClass,
    canonical_digest,
    require_nonempty_string,
)

__all__ = [
    "D1FreshTrainerRequest",
    "D1ResumeTrainerRequest",
    "D1TrainerLocationSpec",
    "D1TrainerRequest",
    "D1TrainerRequestError",
    "build_d1_trainer_request",
    "d1_trainer_bootstrap_attempt_sha256",
    "d1_trainer_command_sha256",
    "d1_trainer_request_sha256",
    "parse_d1_trainer_request",
    "serialize_d1_trainer_request",
]

_SCHEMA = "d1-trainer-command-request-v1"
_COMMANDS = {"d1-run-fresh", "d1-run-resume"}
_TOP_KEYS = frozenset(
    {"command", "locations", "plan", "run_start", "schema_version", "wall_origin"}
)
_ORIGIN_KEYS = frozenset({"monotonic_ns", "started_at"})
_LOCATION_KEYS = frozenset({"checkpoint_path", "journal_path", "quarantine_path"})
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


class D1TrainerRequestError(ValueError):
    pass


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1TrainerRequestError(message)
    raise D1TrainerRequestError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _exact(value: Mapping[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"{label} keys must be exact")


def _relative(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        _fail(f"{label} path must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        _fail(f"{label} path must remain normalized below the artifact root")
    return value


@dataclass(frozen=True, slots=True)
class D1TrainerLocationSpec:
    journal_path: str
    checkpoint_path: str
    quarantine_path: str | None = None

    def __post_init__(self) -> None:
        paths = {
            _relative(self.journal_path, label="journal"),
            _relative(self.checkpoint_path, label="checkpoint"),
        }
        if len(paths) != 2:
            raise ValueError("D1 trainer journal and checkpoint paths must differ")
        if self.quarantine_path is not None:
            quarantine = _relative(self.quarantine_path, label="quarantine")
            if quarantine in paths:
                raise ValueError("D1 trainer quarantine path must be distinct")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "journal_path": self.journal_path,
            "quarantine_path": self.quarantine_path,
        }

    def _resolve(self, root: Path, value: str) -> Path:
        if not isinstance(cast(object, root), Path) or root.is_symlink() or not root.is_dir():
            raise ValueError("D1 trainer artifact root must be a real directory")
        candidate = root / value
        if candidate.is_symlink():
            raise ValueError("D1 trainer artifact path must not be a symlink")
        try:
            candidate.parent.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("D1 trainer artifact path escapes its root") from error
        return candidate

    def fresh(self, root: Path) -> D1FreshRunLocations:
        if self.quarantine_path is not None:
            raise ValueError("D1 fresh request forbids a quarantine path")
        return D1FreshRunLocations(
            self._resolve(root, self.journal_path),
            self._resolve(root, self.checkpoint_path),
        )

    def resume(self, root: Path) -> D1ResumeRunLocations:
        return D1ResumeRunLocations(
            self._resolve(root, self.journal_path),
            self._resolve(root, self.checkpoint_path),
            (None if self.quarantine_path is None else self._resolve(root, self.quarantine_path)),
        )


def _validate_common(
    plan: ConfirmatoryD1RunPlan | SmokeD1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
) -> None:
    if (
        type(plan) not in (ConfirmatoryD1RunPlan, SmokeD1RunPlan)
        or type(start) is not RunStartProvenance
        or type(origin) is not D1TrainerWallOrigin
    ):
        raise TypeError("D1 trainer request requires exact plan, start, and wall-origin types")
    if start.logical_run_id != plan.logical_run_id:
        raise ValueError("D1 trainer request plan and start logical identities differ")
    container = plan.run.execution.container
    if container is None and plan.run.run_class is not RunClass.SMOKE:
        raise ValueError("D1 trainer request run plan is missing its container identity")
    if container is not None and start.image_digest != container.image_digest:
        raise ValueError("D1 trainer request start image differs from the run plan")
    compile_expected = plan.profile.draft.execution_protocol.optimization.compile
    if start.compiler_enabled is not compile_expected:
        raise ValueError("D1 trainer request compiler posture differs from the run plan")
    if origin.started_at != start.trainer_started_at:
        raise ValueError("D1 trainer wall origin differs from run start")


@dataclass(frozen=True, slots=True)
class D1FreshTrainerRequest:
    plan: ConfirmatoryD1RunPlan | SmokeD1RunPlan
    start: RunStartProvenance
    origin: D1TrainerWallOrigin
    locations: D1TrainerLocationSpec

    def __post_init__(self) -> None:
        _validate_common(self.plan, self.start, self.origin)
        if self.start.segment_start_mode is not SegmentStartMode.FRESH:
            raise ValueError("D1 fresh request segment mode differs")
        if self.locations.quarantine_path is not None:
            raise ValueError("D1 fresh request forbids a quarantine path")

    def resolve_locations(self, root: Path) -> D1FreshRunLocations:
        return self.locations.fresh(root)


@dataclass(frozen=True, slots=True)
class D1ResumeTrainerRequest:
    plan: ConfirmatoryD1RunPlan | SmokeD1RunPlan
    start: RunStartProvenance
    origin: D1TrainerWallOrigin
    locations: D1TrainerLocationSpec

    def __post_init__(self) -> None:
        _validate_common(self.plan, self.start, self.origin)
        if self.start.segment_start_mode is not SegmentStartMode.RESUME:
            raise ValueError("D1 resume request segment mode differs")

    def resolve_locations(self, root: Path) -> D1ResumeRunLocations:
        return self.locations.resume(root)


type D1TrainerRequest = D1FreshTrainerRequest | D1ResumeTrainerRequest


def d1_trainer_bootstrap_attempt_sha256(pod_id: str, intent: D1OperatorLaunchIntent) -> str:
    require_nonempty_string(pod_id, label="D1 trainer bootstrap pod_id")
    if type(intent) is not D1OperatorLaunchIntent:
        raise TypeError("D1 trainer bootstrap requires a launch intent")
    return canonical_digest({"intent": intent.manifest, "pod_id": pod_id})


def build_d1_trainer_request(
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
) -> D1TrainerRequest:
    if type(plan) not in (ConfirmatoryD1RunPlan, SmokeD1RunPlan):
        raise TypeError("D1 trainer request builder requires a runnable nominal plan")
    plan = cast(ConfirmatoryD1RunPlan | SmokeD1RunPlan, plan)
    run_identity = {"logical_run_id": plan.logical_run_id, "plan_digest": plan.plan_digest}
    root = f"runs/{canonical_digest(run_identity)}"
    locations = D1TrainerLocationSpec(
        f"{root}/journal.jsonl",
        f"{root}/checkpoint.pt",
        f"{root}/quarantined-tail-{start.execution_segment_index:06d}.jsonl"
        if start.segment_start_mode is SegmentStartMode.RESUME
        else None,
    )
    request_type = (
        D1FreshTrainerRequest
        if start.segment_start_mode is SegmentStartMode.FRESH
        else D1ResumeTrainerRequest
    )
    return request_type(plan, start, origin, locations)


def _z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _origin(value: object) -> D1TrainerWallOrigin:
    item = _mapping(value, label="wall_origin")
    _exact(item, _ORIGIN_KEYS, label="wall_origin")
    timestamp = item["started_at"]
    if type(timestamp) is not str or _UTC.fullmatch(timestamp) is None:
        _fail("wall_origin.started_at must be canonical UTC Z time")
    monotonic_ns = item["monotonic_ns"]
    if type(monotonic_ns) is not int or monotonic_ns < 0:
        _fail("wall_origin.monotonic_ns must be nonnegative")
    try:
        return D1TrainerWallOrigin(
            datetime.fromisoformat(timestamp[:-1] + "+00:00"),
            monotonic_ns,
        )
    except ValueError as error:
        _fail("wall_origin is invalid", error)


def _locations(value: object) -> D1TrainerLocationSpec:
    item = _mapping(value, label="locations")
    _exact(item, _LOCATION_KEYS, label="locations")
    quarantine = item["quarantine_path"]
    if quarantine is not None and type(quarantine) is not str:
        _fail("locations.quarantine_path must be a string or null")
    return D1TrainerLocationSpec(
        cast(str, item["journal_path"]),
        cast(str, item["checkpoint_path"]),
        quarantine,
    )


def trainer_request_to_mapping(request: D1TrainerRequest) -> dict[str, object]:
    command = "d1-run-fresh" if type(request) is D1FreshTrainerRequest else "d1-run-resume"
    return {
        "command": command,
        "locations": request.locations.manifest,
        "plan": request.plan.plan_manifest,
        "run_start": run_start_to_mapping(request.start),
        "schema_version": _SCHEMA,
        "wall_origin": {
            "monotonic_ns": request.origin.monotonic_ns,
            "started_at": _z(request.origin.started_at),
        },
    }


def serialize_d1_trainer_request(request: D1TrainerRequest) -> bytes:
    if type(request) not in (D1FreshTrainerRequest, D1ResumeTrainerRequest):
        raise TypeError("D1 trainer request serializer requires a nominal request")
    return json.dumps(
        trainer_request_to_mapping(request),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def d1_trainer_request_sha256(request: D1TrainerRequest) -> str:
    return hashlib.sha256(serialize_d1_trainer_request(request)).hexdigest()


def d1_trainer_command_sha256(request: D1TrainerRequest) -> str:
    if type(request) not in (D1FreshTrainerRequest, D1ResumeTrainerRequest):
        raise TypeError("D1 trainer command identity requires an exact nominal request")
    command = trainer_request_to_mapping(request)["command"]
    return hashlib.sha256(str(command).encode()).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"trainer request contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_d1_trainer_request(
    payload: bytes,
    *,
    catalog: Iterable[D1LaunchProfile],
) -> D1TrainerRequest:
    if type(payload) is not bytes or not payload:
        _fail("trainer request must be nonempty bytes")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(f"non-finite JSON constant {constant}"),
        )
    except D1TrainerRequestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("trainer request must be UTF-8 JSON", error)
    root = _mapping(value, label="trainer request")
    _exact(root, _TOP_KEYS, label="trainer request")
    if root["schema_version"] != _SCHEMA:
        _fail(f"trainer request schema_version must be {_SCHEMA}")
    command = root["command"]
    if command not in _COMMANDS:
        _fail("trainer request command is unsupported")
    plan_mapping = _mapping(root["plan"], label="plan")
    profile_name = plan_mapping.get("profile_name")
    if type(profile_name) is not str or not profile_name:
        _fail("trainer request profile_name must be a nonempty string")
    run_class = plan_mapping.get("run_class")
    if run_class not in {"confirmatory", "smoke"}:
        _fail("trainer request plan run_class is unsupported")
    profiles = tuple(catalog)
    if any(type(profile) is not D1LaunchProfile for profile in profiles):
        _fail("trainer request catalog contains an invalid profile")
    by_identity = {(profile.name, profile.scale): profile for profile in profiles}
    if len(by_identity) != len(profiles):
        _fail("trainer request catalog repeats a profile name and scale")
    matching_profiles = tuple(
        profile
        for profile in profiles
        if profile.name == profile_name
        and profile.arm.value == plan_mapping.get("arm")
        and profile.family.family_digest == plan_mapping.get("family_digest")
        and profile.draft.construction_digest == plan_mapping.get("construction_digest")
    )
    if len(matching_profiles) != 1:
        _fail("trainer request profile does not match exactly one installed construction")
    profile = matching_profiles[0]
    try:
        if run_class == "confirmatory":
            plan = parse_d1_confirmatory_plan(profile, plan_mapping)
        else:
            plan = parse_d1_smoke_plan(profile, plan_mapping)
        start = run_start_from_mapping(root["run_start"])
        origin = _origin(root["wall_origin"])
        locations = _locations(root["locations"])
        request: D1TrainerRequest = (
            D1FreshTrainerRequest(plan, start, origin, locations)
            if command == "d1-run-fresh"
            else D1ResumeTrainerRequest(plan, start, origin, locations)
        )
    except (D1PlanManifestError, ResourceProvenanceCodecGap, TypeError, ValueError) as error:
        _fail(str(error), error)
    if serialize_d1_trainer_request(request) != payload:
        _fail("trainer request bytes must use canonical JSON")
    return request
