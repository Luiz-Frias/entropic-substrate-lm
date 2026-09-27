"""Pre-launch and execution provenance identities for D1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from entropic_substrate.kernel.identity.d1_primitives import (
    EvidenceClass,
    RunClass,
    require_commit,
    require_nonempty_string,
    require_sha256,
)

__all__ = [
    "D1_TELEMETRY_SCHEMA_VERSION",
    "CodeIdentity",
    "ContainerIdentity",
    "DataIdentity",
    "DataOrderAttestation",
    "EvidenceEnvelope",
    "ExecutionIdentity",
    "PreregistrationIdentity",
]

D1_TELEMETRY_SCHEMA_VERSION: Final[int] = 3


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    """Versioned evidence requirements attached to every launched D1 run."""

    schema_version: str
    required_classes: frozenset[EvidenceClass]
    checkpoint_schema_version: int = 1
    telemetry_schema_version: int = D1_TELEMETRY_SCHEMA_VERSION
    dvc_capture_required: bool = True

    def __post_init__(self) -> None:
        require_nonempty_string(self.schema_version, label="evidence schema_version")
        if not self.required_classes:
            msg = "evidence required_classes must be non-empty"
            raise ValueError(msg)
        if type(self.checkpoint_schema_version) is not int or self.checkpoint_schema_version < 1:
            msg = "checkpoint_schema_version must be a positive integer"
            raise ValueError(msg)
        if (
            type(self.telemetry_schema_version) is not int
            or self.telemetry_schema_version != D1_TELEMETRY_SCHEMA_VERSION
        ):
            msg = f"telemetry_schema_version must be {D1_TELEMETRY_SCHEMA_VERSION}"
            raise ValueError(msg)
        if self.dvc_capture_required is not True:
            msg = "D1 evidence requires DVC capture"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "dvc_capture_required": self.dvc_capture_required,
            "required_classes": sorted(item.value for item in self.required_classes),
            "schema_version": self.schema_version,
            "telemetry_schema_version": self.telemetry_schema_version,
        }


@dataclass(frozen=True, slots=True)
class DataIdentity:
    """Pre-launch dataset, split, and sampler construction identity.

    The observed data-order digest is deliberately not a preflight fact. It is
    emitted later by the capture path as :class:`DataOrderAttestation`.
    """

    dataset_name: str
    dataset_digest: str
    split_digest: str
    sampler_construction_digest: str

    def __post_init__(self) -> None:
        require_nonempty_string(self.dataset_name, label="dataset_name")
        require_sha256(self.dataset_digest, label="dataset_digest")
        require_sha256(self.split_digest, label="split_digest")
        require_sha256(self.sampler_construction_digest, label="sampler_construction_digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "dataset_digest": self.dataset_digest,
            "dataset_name": self.dataset_name,
            "sampler_construction_digest": self.sampler_construction_digest,
            "split_digest": self.split_digest,
        }


@dataclass(frozen=True, slots=True)
class DataOrderAttestation:
    """Post-consumption data-order identity, emitted only with runtime evidence."""

    data_order_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.data_order_digest, label="data_order_digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {"data_order_digest": self.data_order_digest}


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    """Repository source identity for an executable run."""

    commit: str
    repository_digest: str

    def __post_init__(self) -> None:
        require_commit(self.commit, label="code commit")
        require_sha256(self.repository_digest, label="repository_digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {"commit": self.commit, "repository_digest": self.repository_digest}


@dataclass(frozen=True, slots=True)
class ContainerIdentity:
    """Immutable OCI image identity, never a mutable image tag."""

    image_digest: str

    def __post_init__(self) -> None:
        if not self.image_digest.startswith("sha256:"):
            msg = "container image_digest must start with 'sha256:'"
            raise ValueError(msg)
        require_sha256(self.image_digest.removeprefix("sha256:"), label="container image digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {"image_digest": self.image_digest}


@dataclass(frozen=True, slots=True)
class PreregistrationIdentity:
    """One committed preregistration binding the complete D1 family."""

    path: str
    commit: str
    family_digest: str

    def __post_init__(self) -> None:
        require_nonempty_string(self.path, label="preregistration path")
        require_commit(self.commit, label="preregistration commit")
        require_sha256(self.family_digest, label="preregistration family_digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "family_digest": self.family_digest,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """Per-run identity; seed stays out of construction and aggregation digests."""

    run_class: RunClass
    seed: int
    pairing_id: str
    code: CodeIdentity | None = None
    container: ContainerIdentity | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            msg = "seed must be a nonnegative integer"
            raise ValueError(msg)
        require_nonempty_string(self.pairing_id, label="pairing_id")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "code": None if self.code is None else self.code.manifest,
            "container": None if self.container is None else self.container.manifest,
            "pairing_id": self.pairing_id,
            "run_class": self.run_class.value,
            "seed": self.seed,
        }
