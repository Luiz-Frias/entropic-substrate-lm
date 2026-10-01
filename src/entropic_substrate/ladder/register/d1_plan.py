"""Typed D1 run plans and strict confirmatory manifest admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn, cast

from entropic_substrate.kernel.codec.strict_json import (
    exact_keys,
    require_lowercase_sha256,
    require_mapping,
)
from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    EvidenceClass,
    RunClass,
    canonical_digest,
    require_commit,
    require_nonempty_string,
)
from entropic_substrate.kernel.runtime.config import TrainConfig
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile, D1LaunchScale
from entropic_substrate.ladder.register.d1_provenance import (
    CodeIdentity,
    ContainerIdentity,
    DataIdentity,
    EvidenceEnvelope,
    ExecutionIdentity,
    PreregistrationIdentity,
)
from entropic_substrate.ladder.register.d1_runs import (
    ConfirmatoryD1RunSpec,
    D1RunSpec,
    PilotD1RunSpec,
    SmokeD1RunSpec,
)

D1_RUN_PLAN_SCHEMA: Final[str] = "d1-run-plan-v1"
_PLAN_KEYS: Final = frozenset(
    {
        "arm",
        "code",
        "construction_digest",
        "container",
        "data",
        "evidence",
        "family_digest",
        "logical_run_id",
        "pairing_id",
        "preregistration",
        "profile_name",
        "run_class",
        "schema_version",
        "seed",
    }
)


class D1PlanManifestError(ValueError):
    pass


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1PlanManifestError(message)
    raise D1PlanManifestError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    return require_mapping(
        value, fail=lambda _reason: _fail(f"{label} must be a JSON object with string keys")
    )


def _exact_keys(value: dict[str, object], expected: frozenset[str], *, label: str) -> None:
    missing, unknown = exact_keys(value, expected)
    if unknown:
        _fail(f"unknown keys for {label}: {sorted(unknown)}")
    if missing:
        _fail(f"missing keys for {label}: {sorted(missing)}")


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    return require_lowercase_sha256(
        text,
        fail=lambda _reason: _fail(f"{label} must be a 64-character hexadecimal SHA-256 digest"),
    )


def _commit(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    try:
        require_commit(text, label=label)
    except ValueError as error:
        _fail(str(error), error)
    return text


@dataclass(frozen=True, slots=True)
class D1RunPlan:
    profile: D1LaunchProfile
    run: D1RunSpec
    logical_run_id: str

    def __post_init__(self) -> None:
        if type(self) is D1RunPlan:
            raise TypeError("D1RunPlan is factory-only; use a nominal plan route")
        require_nonempty_string(self.logical_run_id, label="logical_run_id")
        if self.run.draft != self.profile.draft:
            raise ValueError("D1 plan run is not the selected launch profile cell")

    @property
    def seed(self) -> int:
        return self.run.execution.seed

    @property
    def train(self) -> TrainConfig:
        value = self.profile.train
        value.seed = self.seed
        return value

    @property
    def plan_manifest(self) -> dict[str, object]:
        execution = self.run.execution
        return {
            "arm": self.profile.arm.value,
            "code": None if execution.code is None else execution.code.manifest,
            "construction_digest": self.profile.draft.construction_digest,
            "container": None if execution.container is None else execution.container.manifest,
            "data": None if self.run.data is None else self.run.data.manifest,
            "evidence": self.run.evidence.manifest,
            "family_digest": self.profile.family.family_digest,
            "logical_run_id": self.logical_run_id,
            "pairing_id": execution.pairing_id,
            "preregistration": (
                None if self.run.preregistration is None else self.run.preregistration.manifest
            ),
            "profile_name": self.profile.name,
            "run_class": self.run.run_class.value,
            "schema_version": D1_RUN_PLAN_SCHEMA,
            "seed": execution.seed,
        }

    @property
    def plan_digest(self) -> str:
        return canonical_digest(self.plan_manifest)


@dataclass(frozen=True, slots=True)
class SmokeD1RunPlan(D1RunPlan):
    def __post_init__(self) -> None:
        D1RunPlan.__post_init__(self)
        if type(self.run) is not SmokeD1RunSpec:
            raise TypeError("smoke plan requires a nominal SmokeD1RunSpec")


@dataclass(frozen=True, slots=True)
class ExcludedPilotD1RunPlan(D1RunPlan):
    def __post_init__(self) -> None:
        D1RunPlan.__post_init__(self)
        if type(self.run) is not PilotD1RunSpec:
            raise TypeError("excluded pilot plan requires a nominal PilotD1RunSpec")


@dataclass(frozen=True, slots=True)
class ConfirmatoryD1RunPlan(D1RunPlan):
    def __post_init__(self) -> None:
        D1RunPlan.__post_init__(self)
        if type(self.run) is not ConfirmatoryD1RunSpec:
            raise TypeError("confirmatory plan requires a nominal ConfirmatoryD1RunSpec")
        if self.profile.scale is not D1LaunchScale.FULL:
            raise ValueError("confirmatory D1 plan requires a full launch profile")
        if self.run.family != self.profile.family:
            raise ValueError("confirmatory D1 plan requires the profile family")


def create_d1_smoke_plan(
    profile: D1LaunchProfile,
    *,
    logical_run_id: str,
    seed: int,
    pairing_id: str,
    evidence: EvidenceEnvelope,
    data: DataIdentity | None = None,
    code: CodeIdentity | None = None,
    container: ContainerIdentity | None = None,
) -> SmokeD1RunPlan:
    run = profile.draft.validate_smoke(
        execution=ExecutionIdentity(
            run_class=RunClass.SMOKE,
            seed=seed,
            pairing_id=pairing_id,
            code=code,
            container=container,
        ),
        evidence=evidence,
        data=data,
    )
    return SmokeD1RunPlan(profile=profile, run=run, logical_run_id=logical_run_id)


def create_d1_excluded_pilot_plan(
    profile: D1LaunchProfile,
    *,
    logical_run_id: str,
    seed: int,
    pairing_id: str,
    evidence: EvidenceEnvelope,
    data: DataIdentity | None = None,
    code: CodeIdentity | None = None,
    container: ContainerIdentity | None = None,
) -> ExcludedPilotD1RunPlan:
    run = profile.draft.validate_pilot(
        execution=ExecutionIdentity(
            run_class=RunClass.PILOT,
            seed=seed,
            pairing_id=pairing_id,
            code=code,
            container=container,
        ),
        evidence=evidence,
        data=data,
    )
    return ExcludedPilotD1RunPlan(profile=profile, run=run, logical_run_id=logical_run_id)


def _code(value: object) -> CodeIdentity:
    item = _mapping(value, label="code")
    _exact_keys(item, frozenset({"commit", "repository_digest"}), label="code")
    return CodeIdentity(
        commit=_commit(item["commit"], label="code.commit"),
        repository_digest=_sha256(item["repository_digest"], label="code.repository_digest"),
    )


def _container(value: object) -> ContainerIdentity:
    item = _mapping(value, label="container")
    _exact_keys(item, frozenset({"image_digest"}), label="container")
    digest = _string(item["image_digest"], label="container.image_digest")
    try:
        return ContainerIdentity(image_digest=digest)
    except ValueError as error:
        _fail(f"container.image_digest is invalid: {error}", error)


def _data(value: object) -> DataIdentity:
    item = _mapping(value, label="data")
    keys = frozenset(
        {"dataset_name", "dataset_digest", "split_digest", "sampler_construction_digest"}
    )
    _exact_keys(item, keys, label="data")
    return DataIdentity(
        dataset_name=_string(item["dataset_name"], label="data.dataset_name"),
        dataset_digest=_sha256(item["dataset_digest"], label="data.dataset_digest"),
        split_digest=_sha256(item["split_digest"], label="data.split_digest"),
        sampler_construction_digest=_sha256(
            item["sampler_construction_digest"], label="data.sampler_construction_digest"
        ),
    )


def _evidence(value: object) -> EvidenceEnvelope:
    item = _mapping(value, label="evidence")
    keys = frozenset(
        {
            "checkpoint_schema_version",
            "dvc_capture_required",
            "required_classes",
            "schema_version",
            "telemetry_schema_version",
        }
    )
    _exact_keys(item, keys, label="evidence")
    raw_classes = item["required_classes"]
    if type(raw_classes) is not list or not raw_classes:
        _fail("evidence.required_classes must be a non-empty list")
    if any(type(value) is not str for value in raw_classes):
        _fail("evidence.required_classes must contain only strings")
    try:
        classes = tuple(EvidenceClass(value) for value in raw_classes)
    except ValueError as error:
        _fail("evidence.required_classes contains an unsupported class", error)
    if len(classes) != len(set(classes)):
        _fail("evidence.required_classes refuses duplicates")
    dvc_required = item["dvc_capture_required"]
    if type(dvc_required) is not bool:
        _fail("evidence.dvc_capture_required must be a bool")
    try:
        return EvidenceEnvelope(
            schema_version=_string(item["schema_version"], label="evidence.schema_version"),
            required_classes=frozenset(classes),
            checkpoint_schema_version=_positive_int(
                item["checkpoint_schema_version"], label="evidence.checkpoint_schema_version"
            ),
            telemetry_schema_version=_positive_int(
                item["telemetry_schema_version"], label="evidence.telemetry_schema_version"
            ),
            dvc_capture_required=dvc_required,
        )
    except ValueError as error:
        _fail(f"evidence is invalid: {error}", error)


def _preregistration(value: object) -> PreregistrationIdentity:
    item = _mapping(value, label="preregistration")
    _exact_keys(item, frozenset({"path", "commit", "family_digest"}), label="preregistration")
    return PreregistrationIdentity(
        path=_string(item["path"], label="preregistration.path"),
        commit=_commit(item["commit"], label="preregistration.commit"),
        family_digest=_sha256(item["family_digest"], label="preregistration.family_digest"),
    )


def parse_d1_confirmatory_plan(profile: D1LaunchProfile, value: object) -> ConfirmatoryD1RunPlan:
    if type(profile) is not D1LaunchProfile:
        _fail("confirmatory D1 plan requires a D1LaunchProfile")
    if profile.scale is not D1LaunchScale.FULL:
        _fail("confirmatory D1 plan requires a full launch profile")
    item = _mapping(value, label="D1 confirmatory plan")
    _exact_keys(item, _PLAN_KEYS, label="D1 confirmatory plan")
    if item["schema_version"] != D1_RUN_PLAN_SCHEMA:
        _fail(f"schema_version must be {D1_RUN_PLAN_SCHEMA!r}")
    if item["run_class"] != RunClass.CONFIRMATORY.value:
        _fail("run_class must be the fixed 'confirmatory' literal")
    if _string(item["profile_name"], label="profile_name") != profile.name:
        _fail("profile_name does not match the supplied launch profile")
    try:
        arm = D1Arm(_string(item["arm"], label="arm"))
    except ValueError as error:
        _fail("arm is not a registered D1 arm", error)
    if arm is not profile.arm:
        _fail("arm does not match the supplied launch profile")
    family_digest = _sha256(item["family_digest"], label="family_digest")
    if family_digest != profile.family.family_digest:
        _fail("family_digest does not match the supplied profile family")
    construction_digest = _sha256(item["construction_digest"], label="construction_digest")
    if construction_digest != profile.draft.construction_digest:
        _fail("construction_digest does not match the supplied profile cell")

    preregistration = _preregistration(item["preregistration"])
    if preregistration.family_digest != family_digest:
        _fail("preregistration.family_digest does not match family_digest")
    execution = ExecutionIdentity(
        run_class=RunClass.CONFIRMATORY,
        seed=_nonnegative_int(item["seed"], label="seed"),
        pairing_id=_string(item["pairing_id"], label="pairing_id"),
        code=_code(item["code"]),
        container=_container(item["container"]),
    )
    try:
        run = profile.family.validate_confirmatory_arm(
            arm=arm,
            execution=execution,
            data=_data(item["data"]),
            evidence=_evidence(item["evidence"]),
            preregistration=preregistration,
        )
        return ConfirmatoryD1RunPlan(
            profile=profile,
            run=run,
            logical_run_id=_string(item["logical_run_id"], label="logical_run_id"),
        )
    except (TypeError, ValueError) as error:
        _fail(f"confirmatory D1 plan is invalid: {error}", error)


def parse_d1_smoke_plan(profile: D1LaunchProfile, value: object) -> SmokeD1RunPlan:
    return cast(SmokeD1RunPlan, _parse_excluded_plan(profile, value, RunClass.SMOKE))


def parse_d1_pilot_plan(profile: D1LaunchProfile, value: object) -> ExcludedPilotD1RunPlan:
    """Rehydrate an excluded pilot without assigning confirmatory authority."""
    return cast(ExcludedPilotD1RunPlan, _parse_excluded_plan(profile, value, RunClass.PILOT))


def _parse_excluded_plan(
    profile: D1LaunchProfile, value: object, run_class: RunClass
) -> SmokeD1RunPlan | ExcludedPilotD1RunPlan:
    label = run_class.value
    if type(profile) is not D1LaunchProfile:
        _fail(f"{label} D1 plan requires a D1LaunchProfile")
    item = _mapping(value, label=f"D1 {label} plan")
    _exact_keys(item, _PLAN_KEYS, label=f"D1 {label} plan")
    if item["schema_version"] != D1_RUN_PLAN_SCHEMA:
        _fail(f"schema_version must be {D1_RUN_PLAN_SCHEMA!r}")
    if item["run_class"] != label:
        _fail(f"run_class must be the fixed {label!r} literal")
    if _string(item["profile_name"], label="profile_name") != profile.name:
        _fail("profile_name does not match the supplied launch profile")
    try:
        arm = D1Arm(_string(item["arm"], label="arm"))
    except ValueError as error:
        _fail("arm is not a registered D1 arm", error)
    if arm is not profile.arm:
        _fail("arm does not match the supplied launch profile")
    if _sha256(item["family_digest"], label="family_digest") != profile.family.family_digest:
        _fail("family_digest does not match the supplied profile family")
    if (
        _sha256(item["construction_digest"], label="construction_digest")
        != profile.draft.construction_digest
    ):
        _fail("construction_digest does not match the supplied profile cell")
    if item["preregistration"] is not None:
        _fail(f"{label} D1 plan forbids preregistration identity")
    factory = create_d1_smoke_plan if run_class is RunClass.SMOKE else create_d1_excluded_pilot_plan
    try:
        return factory(
            profile,
            logical_run_id=_string(item["logical_run_id"], label="logical_run_id"),
            seed=_nonnegative_int(item["seed"], label="seed"),
            pairing_id=_string(item["pairing_id"], label="pairing_id"),
            evidence=_evidence(item["evidence"]),
            data=None if item["data"] is None else _data(item["data"]),
            code=None if item["code"] is None else _code(item["code"]),
            container=(None if item["container"] is None else _container(item["container"])),
        )
    except (TypeError, ValueError) as error:
        _fail(f"{label} D1 plan is invalid: {error}", error)
