"""Immutable values at the D1 Build Cloud and RunPod operator boundary."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from entropic_substrate.evidence.resource_provenance import (
    RESOURCE_TELEMETRY_SCHEMA_VERSION,
    PurchaseOption,
    QuotedRate,
    ResourceActivationRegistration,
    RunStartProvenance,
    SegmentStartMode,
    execution_segment_id,
)
from entropic_substrate.experiments.d1_plan import D1RunPlan
from entropic_substrate.experiments.d1_primitives import (
    RunClass,
    require_nonempty_string,
    require_sha256,
)

_OCI_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_FULL_COMMIT: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_PLATFORM: Final = "linux/amd64"


def _repository(value: str) -> None:
    require_nonempty_string(value, label="image repository")
    tail = value.rsplit("/", 1)[-1]
    if "@" in value or ":" in tail:
        raise ValueError("image repository must be tag-free and digest-free")
    if value.strip() != value or any(character.isspace() for character in value):
        raise ValueError("image repository must not contain whitespace")


def _commit(value: str) -> None:
    if type(value) is not str or _FULL_COMMIT.fullmatch(value) is None:
        raise ValueError("source_commit must be a full lowercase Git identity")


def _utc(value: datetime, *, label: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a UTC timestamp")


def _plain_path(value: object, *, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be a Path")
    path = Path(os.path.abspath(value))
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    return path


class D1A100FormFactor(StrEnum):
    """The two accelerator constructions admitted by the D1 execution lane."""

    PCIE = "pcie"
    SXM = "sxm"


@dataclass(frozen=True, slots=True)
class D1ImageBuildRequest:
    """One content-tagged linux/amd64 Build Cloud request."""

    builder: str
    repository: str
    source_commit: str
    context: Path
    dockerfile: Path
    platform: str = _PLATFORM

    def __post_init__(self) -> None:
        require_nonempty_string(self.builder, label="Build Cloud builder")
        _repository(self.repository)
        _commit(self.source_commit)
        context = _plain_path(cast(object, self.context), label="build context")
        dockerfile = _plain_path(cast(object, self.dockerfile), label="dockerfile")
        if not context.is_dir() or not dockerfile.is_file():
            raise ValueError("build context and dockerfile must exist")
        try:
            dockerfile.relative_to(context)
        except ValueError as error:
            raise ValueError("dockerfile must be inside the build context") from error
        if self.platform != _PLATFORM:
            raise ValueError("D1 Build Cloud platform must be linux/amd64")
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "dockerfile", dockerfile)

    @property
    def build_tag(self) -> str:
        return f"{self.repository}:git-{self.source_commit}"


@dataclass(frozen=True, slots=True)
class D1ImmutableImage:
    """A pushed image addressed only by its OCI manifest digest."""

    repository: str
    digest: str
    source_commit: str
    platform: str = _PLATFORM

    def __post_init__(self) -> None:
        _repository(self.repository)
        if type(self.digest) is not str or _OCI_DIGEST.fullmatch(self.digest) is None:
            raise ValueError("image digest must be lowercase sha256 OCI identity")
        _commit(self.source_commit)
        if self.platform != _PLATFORM:
            raise ValueError("D1 immutable image platform must be linux/amd64")

    @property
    def reference(self) -> str:
        return f"{self.repository}@{self.digest}"


@dataclass(frozen=True, slots=True)
class D1A100ResourceRequest:
    """Exact provider placement requested for one physical segment."""

    accelerator_form_factor: D1A100FormFactor
    provider_gpu_type_id: str
    accelerator_count: int
    provider: str
    provider_sku: str
    region: str
    market: str
    purchase_option: PurchaseOption

    def __post_init__(self) -> None:
        if type(self.accelerator_form_factor) is not D1A100FormFactor:
            raise TypeError("accelerator_form_factor must be D1A100FormFactor")
        for label, value in (
            ("provider_gpu_type_id", self.provider_gpu_type_id),
            ("provider", self.provider),
            ("provider_sku", self.provider_sku),
            ("region", self.region),
            ("market", self.market),
        ):
            require_nonempty_string(value, label=label)
        normalized = self.provider_gpu_type_id.upper().replace("-", " ")
        if "A100" not in normalized:
            raise ValueError("provider GPU type must identify an A100 accelerator")
        marker = "PCIE" if self.accelerator_form_factor is D1A100FormFactor.PCIE else "SXM"
        if marker not in normalized:
            raise ValueError("provider GPU type differs from declared A100 form factor")
        if type(self.accelerator_count) is not int or self.accelerator_count < 1:
            raise ValueError("accelerator_count must be a positive integer")
        if type(self.purchase_option) is not PurchaseOption:
            raise TypeError("purchase_option must be PurchaseOption")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "accelerator_count": self.accelerator_count,
            "accelerator_form_factor": self.accelerator_form_factor.value,
            "market": self.market,
            "provider": self.provider,
            "provider_gpu_type_id": self.provider_gpu_type_id,
            "provider_sku": self.provider_sku,
            "purchase_option": self.purchase_option.value,
            "region": self.region,
        }


@dataclass(frozen=True, slots=True)
class D1OperatorLaunchIntent:
    """Durable pre-create identity for a fresh segment or explicit resume."""

    plan: D1RunPlan
    registered_design_digest: str | None
    image: D1ImmutableImage
    resource: D1A100ResourceRequest
    quoted_rate: QuotedRate
    resource_activation: ResourceActivationRegistration
    execution_segment_index: int
    resume_checkpoint_sha256: str | None
    terminate_after_seconds: int
    schema_version: str = "d1-operator-launch-intent-v2"

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.plan), D1RunPlan):
            raise TypeError("D1 launch intent requires a typed run plan")
        if type(self.image) is not D1ImmutableImage:
            raise TypeError("D1 launch intent requires an immutable image")
        if type(self.resource) is not D1A100ResourceRequest:
            raise TypeError("D1 launch intent requires an A100 resource request")
        if type(self.quoted_rate) is not QuotedRate:
            raise TypeError("D1 launch intent requires measured quoted rate")
        if type(self.resource_activation) is not ResourceActivationRegistration:
            raise TypeError("D1 launch intent requires resource activation registration")
        if type(self.execution_segment_index) is not int or self.execution_segment_index < 0:
            raise ValueError("execution_segment_index must be nonnegative")
        if self.execution_segment_index == 0:
            if self.resume_checkpoint_sha256 is not None:
                raise ValueError("fresh segment forbids a resume checkpoint")
        elif self.resume_checkpoint_sha256 is None:
            raise ValueError("resume checkpoint is required for a later segment")
        else:
            require_sha256(self.resume_checkpoint_sha256, label="resume checkpoint")
        if type(self.terminate_after_seconds) is not int or self.terminate_after_seconds < 1:
            raise ValueError("terminate_after_seconds must be a positive integer")
        execution = self.plan.run.execution
        if execution.container is None or execution.container.image_digest != self.image.digest:
            raise ValueError("D1 plan container differs from immutable launch image")
        if execution.code is not None and execution.code.commit != self.image.source_commit:
            raise ValueError("D1 plan code commit differs from immutable launch image")
        if self.plan.run.run_class is RunClass.CONFIRMATORY:
            if self.registered_design_digest is None:
                raise ValueError("confirmatory launch requires registered_design_digest")
            require_sha256(
                self.registered_design_digest,
                label="registered_design_digest",
            )
        elif self.registered_design_digest is not None:
            raise ValueError("non-confirmatory launch forbids registered_design_digest")
        if self.schema_version != "d1-operator-launch-intent-v2":
            raise ValueError("unsupported D1 operator launch-intent schema")

    @property
    def segment_start_mode(self) -> SegmentStartMode:
        return (
            SegmentStartMode.FRESH if self.execution_segment_index == 0 else SegmentStartMode.RESUME
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "execution_segment_id": execution_segment_id(
                self.plan.logical_run_id, self.execution_segment_index
            ),
            "execution_segment_index": self.execution_segment_index,
            "image_reference": self.image.reference,
            "plan": self.plan.plan_manifest,
            "plan_digest": self.plan.plan_digest,
            "quoted_rate": self.quoted_rate.manifest,
            "registered_design_digest": self.registered_design_digest,
            "resource": self.resource.manifest,
            "resource_activation": self.resource_activation.manifest,
            "resume_checkpoint_sha256": self.resume_checkpoint_sha256,
            "schema_version": self.schema_version,
            "segment_start_mode": self.segment_start_mode.value,
            "terminate_after_seconds": self.terminate_after_seconds,
        }


@dataclass(frozen=True, slots=True)
class D1PodPlacement:
    """Provider-measured allocation facts returned after pod creation."""

    pod_id: str
    accelerator_form_factor: D1A100FormFactor
    accelerator_model: str
    accelerator_count: int
    provider: str
    provider_sku: str
    region: str
    market: str
    purchase_option: PurchaseOption
    pod_created_at: datetime

    def __post_init__(self) -> None:
        require_nonempty_string(self.pod_id, label="pod_id")
        D1A100ResourceRequest(
            self.accelerator_form_factor,
            self.accelerator_model,
            self.accelerator_count,
            self.provider,
            self.provider_sku,
            self.region,
            self.market,
            self.purchase_option,
        )
        _utc(self.pod_created_at, label="pod_created_at")

    @property
    def resource_identity(self) -> tuple[object, ...]:
        return (
            self.accelerator_form_factor,
            self.accelerator_model,
            self.accelerator_count,
            self.provider,
            self.provider_sku,
            self.region,
            self.market,
            self.purchase_option,
        )


@dataclass(frozen=True, slots=True)
class D1RuntimeObservation:
    """Trainer-emitted runtime facts measured before training starts."""

    pod_id: str
    trainer_started_at: datetime
    cuda_version: str
    driver_version: str
    pytorch_version: str
    compiler_enabled: bool
    compiler_version: str
    trainer_monotonic_origin_ns: int

    def __post_init__(self) -> None:
        require_nonempty_string(self.pod_id, label="runtime pod_id")
        _utc(self.trainer_started_at, label="trainer_started_at")
        for label, value in (
            ("cuda_version", self.cuda_version),
            ("driver_version", self.driver_version),
            ("pytorch_version", self.pytorch_version),
            ("compiler_version", self.compiler_version),
        ):
            require_nonempty_string(value, label=label)
        if type(self.compiler_enabled) is not bool:
            raise TypeError("compiler_enabled must be bool")
        if not self.compiler_enabled and self.compiler_version != "disabled":
            raise ValueError("disabled compiler requires compiler_version='disabled'")
        if (
            type(self.trainer_monotonic_origin_ns) is not int
            or self.trainer_monotonic_origin_ns < 0
        ):
            raise ValueError("trainer_monotonic_origin_ns must be nonnegative")


def attest_d1_run_start(
    *,
    intent: D1OperatorLaunchIntent,
    placement: D1PodPlacement,
    runtime: D1RuntimeObservation,
) -> RunStartProvenance:
    """Join requested placement and trainer facts only when they match exactly."""
    if type(intent) is not D1OperatorLaunchIntent:
        raise TypeError("run-start attestation requires D1OperatorLaunchIntent")
    if type(placement) is not D1PodPlacement or type(runtime) is not D1RuntimeObservation:
        raise TypeError("run-start attestation requires typed provider observations")
    requested = intent.resource
    expected_resource = (
        requested.accelerator_form_factor,
        requested.provider_gpu_type_id,
        requested.accelerator_count,
        requested.provider,
        requested.provider_sku,
        requested.region,
        requested.market,
        requested.purchase_option,
    )
    if placement.resource_identity != expected_resource:
        raise ValueError("provider resource placement differs from the launch intent")
    if runtime.pod_id != placement.pod_id:
        raise ValueError("runtime pod identity differs from provider placement")
    compile_expected = intent.plan.profile.draft.execution_protocol.optimization.compile
    if runtime.compiler_enabled is not compile_expected:
        raise ValueError("runtime compiler posture differs from the run plan")
    return RunStartProvenance(
        schema_version=RESOURCE_TELEMETRY_SCHEMA_VERSION,
        logical_run_id=intent.plan.logical_run_id,
        execution_segment_index=intent.execution_segment_index,
        execution_segment_id=execution_segment_id(
            intent.plan.logical_run_id, intent.execution_segment_index
        ),
        segment_start_mode=intent.segment_start_mode,
        resume_checkpoint_sha256=intent.resume_checkpoint_sha256,
        accelerator_model=placement.accelerator_model,
        accelerator_count=placement.accelerator_count,
        provider=placement.provider,
        provider_sku=placement.provider_sku,
        region=placement.region,
        market=placement.market,
        purchase_option=placement.purchase_option,
        image_digest=intent.image.digest,
        cuda_version=runtime.cuda_version,
        driver_version=runtime.driver_version,
        pytorch_version=runtime.pytorch_version,
        compiler_enabled=runtime.compiler_enabled,
        compiler_version=runtime.compiler_version,
        resource_activation=intent.resource_activation,
        quoted_rate=intent.quoted_rate,
        pod_created_at=placement.pod_created_at,
        trainer_started_at=runtime.trainer_started_at,
    )
