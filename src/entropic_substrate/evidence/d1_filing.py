"""Detached D1 v2 filing manifests and real-file integrity validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict
from entropic_substrate.evidence.lifecycle import (
    ArtifactIdentity,
    ContentHash,
)

REQUIRED_D1_FILING_ARTIFACTS = (
    "narrative",
    "machine_verdict",
    "preregistration",
    "raw_capture_manifest",
    "blinded_analysis",
    "analysis_receipt",
    "analysis_reader",
    "analysis_isolation_attestation",
    "review_packet",
    "reviewer_prompt",
    "reviewer_report",
    "reviewer_execution_receipt",
    "alias_commitment",
    "alias_reveal",
)


class D1FilingValidationError(ValueError):
    """A malformed, unsafe, or drifted detached D1 filing manifest."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"code": code, "path": path}
        super().__init__(f"D1 filing validation {code} at {path}: {detail}")


def _error(code: str, path: str, detail: str) -> D1FilingValidationError:
    return D1FilingValidationError(code=code, path=path, detail=detail)


def canonical_machine_verdict_bytes(payload: Mapping[str, object]) -> bytes:
    """The canonical byte encoding whose digest the detached manifest records."""

    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error("noncanonical_machine_verdict", "machine_verdict", str(exc)) from exc


def _object(value: object, *, path: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error("invalid_type", path, "must be an object")
    actual: set[str] = set()
    for key in value:
        if not isinstance(key, str):
            raise _error("invalid_field_name", path, "object field names must be strings")
        actual.add(key)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
    if missing:
        raise _error("missing_field", path, f"missing required field(s): {', '.join(missing)}")
    if unknown:
        raise _error("unknown_field", path, f"unknown field(s): {', '.join(unknown)}")
    return value


def _digest(value: object, *, algorithm: str, path: str) -> ContentHash:
    if not isinstance(value, str):
        raise _error("invalid_digest", path, "must be a string")
    try:
        return ContentHash(algorithm=algorithm, digest=value)
    except ValueError as exc:
        raise _error("invalid_digest", path, str(exc)) from exc


def _relative_path(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error("invalid_path", path, "must be a nonempty relative path")
    if "\\" in value:
        raise _error("unsafe_path", path, "must use POSIX separators")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise _error("unsafe_path", path, "must remain below the artifact root")
    if str(candidate) != value:
        raise _error("unsafe_path", path, "must be normalized")
    return value


@dataclass(frozen=True)
class D1FiledArtifact:
    path: str
    sha256: ContentHash
    md5: ContentHash
    size: int

    @classmethod
    def from_mapping(cls, value: object, *, path: str) -> D1FiledArtifact:
        item = _object(value, path=path, keys=frozenset({"path", "sha256", "md5", "size"}))
        size = item["size"]
        if type(size) is not int or size < 0:
            raise _error("invalid_size", f"{path}.size", "must be a nonnegative integer")
        return cls(
            path=_relative_path(item["path"], path=f"{path}.path"),
            sha256=_digest(item["sha256"], algorithm="sha256", path=f"{path}.sha256"),
            md5=_digest(item["md5"], algorithm="md5", path=f"{path}.md5"),
            size=size,
        )

    def verify_at(self, artifact_root: Path) -> None:
        root = artifact_root.resolve()
        candidate = (root / self.path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise _error("unsafe_path", self.path, "resolves outside artifact root") from exc
        if not candidate.is_file():
            raise _error("missing_artifact", self.path, "file is absent")
        payload = candidate.read_bytes()
        if len(payload) != self.size:
            raise _error("size_mismatch", self.path, "actual byte size differs from manifest")
        if hashlib.sha256(payload).hexdigest() != self.sha256.digest:
            raise _error("sha256_mismatch", self.path, "actual SHA-256 differs from manifest")
        if hashlib.md5(payload, usedforsecurity=False).hexdigest() != self.md5.digest:
            raise _error("md5_mismatch", self.path, "actual MD5 differs from manifest")

    def artifact_identity(self) -> ArtifactIdentity:
        return ArtifactIdentity(path=self.path, content_hash=self.sha256)


@dataclass(frozen=True)
class D1FilingManifest:
    rung: str
    artifacts: Mapping[str, D1FiledArtifact]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> D1FilingManifest:
        root = _object(payload, path="$", keys=frozenset({"schema_version", "rung", "artifacts"}))
        if root["schema_version"] != "d1-filing-manifest/v2":
            raise _error(
                "unsupported_schema_version", "schema_version", "expected d1-filing-manifest/v2"
            )
        if root["rung"] != "d1":
            raise _error("invalid_rung", "rung", "expected d1")
        raw_artifacts = root["artifacts"]
        if not isinstance(raw_artifacts, Mapping):
            raise _error("invalid_type", "artifacts", "must be an object")
        if set(raw_artifacts) != set(REQUIRED_D1_FILING_ARTIFACTS):
            raise _error(
                "artifact_role_set",
                "artifacts",
                "must contain exactly every required filing artifact",
            )
        parsed: dict[str, D1FiledArtifact] = {}
        seen_paths: set[str] = set()
        for role in REQUIRED_D1_FILING_ARTIFACTS:
            artifact = D1FiledArtifact.from_mapping(raw_artifacts[role], path=f"artifacts.{role}")
            if artifact.path in seen_paths:
                raise _error(
                    "duplicate_path",
                    f"artifacts.{role}.path",
                    "each filing artifact needs a unique path",
                )
            seen_paths.add(artifact.path)
            parsed[role] = artifact
        return cls(rung="d1", artifacts=MappingProxyType(parsed))

    def verify_artifact_root(self, artifact_root: Path) -> None:
        if not artifact_root.is_dir():
            raise _error(
                "missing_artifact_root", str(artifact_root), "must be an existing directory"
            )
        for artifact in self.artifacts.values():
            artifact.verify_at(artifact_root)


@dataclass(frozen=True, slots=True)
class D1DetachedFiling:
    """Byte-validated filing material with deliberately no gate authority."""

    verdict: ValidatedD1Verdict
    manifest: D1FilingManifest
    artifact_root_verified: bool


def validate_d1_filing_manifest(
    machine_verdict: Mapping[str, object],
    filing_manifest: Mapping[str, object],
    *,
    decision_contract: D1DecisionContract,
    artifact_root: Path | None = None,
) -> D1DetachedFiling:
    """Validate detached bytes without conferring successor-gate authority."""

    verdict = ValidatedD1Verdict.from_mapping(machine_verdict, decision_contract=decision_contract)
    manifest = D1FilingManifest.from_mapping(filing_manifest)
    machine_record = manifest.artifacts["machine_verdict"]
    canonical = canonical_machine_verdict_bytes(machine_verdict)
    if len(canonical) != machine_record.size:
        raise _error(
            "machine_size_mismatch", "artifacts.machine_verdict", "does not bind canonical v2 JSON"
        )
    if hashlib.sha256(canonical).hexdigest() != machine_record.sha256.digest:
        raise _error(
            "machine_sha256_mismatch",
            "artifacts.machine_verdict",
            "does not bind canonical v2 JSON",
        )
    if hashlib.md5(canonical, usedforsecurity=False).hexdigest() != machine_record.md5.digest:
        raise _error(
            "machine_md5_mismatch", "artifacts.machine_verdict", "does not bind canonical v2 JSON"
        )
    if artifact_root is not None:
        manifest.verify_artifact_root(artifact_root)
    return D1DetachedFiling(
        verdict=verdict,
        manifest=manifest,
        artifact_root_verified=artifact_root is not None,
    )
