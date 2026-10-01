"""Remote construction/custody envelope; authority is recomputed by the worker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import NoReturn, cast

from entropic_substrate.kernel.artifacts import normalize_d1_relative_artifact_path
from entropic_substrate.kernel.codec.strict_json import require_lowercase_sha256
from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import (
    run_start_from_mapping,
    run_start_to_mapping,
)
from entropic_substrate.kernel.identity.d1_primitives import (
    canonical_json,
    require_commit,
)
from entropic_substrate.kernel.lifecycle import ActorIdentity, ActorRole, ArtifactIdentity
from entropic_substrate.ladder.gate.ladder_campaign import (
    actor_from_mapping,
    artifact_from_mapping,
    artifact_mapping,
    object_fields,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id


def digest_refusal(label: str) -> Callable[[str], NoReturn]:
    """The ``fail`` channel of the remote boundary digests: a labelled ``ValueError``."""

    def fail(_reason: str) -> NoReturn:
        msg = f"{label} must be a 64-character hexadecimal SHA-256 digest"
        raise ValueError(msg)

    return fail


@dataclass(frozen=True, slots=True)
class RemoteLadderRequest:
    """Full hash-bound input graph and expected authority, never a launch permission.

    The input tuple contains profile predecessors, design/statistics/registration,
    filing evidence and corpus bytes. Fresh output lives under ``run/``; resume
    captures include that prefix and bind the exact checkpoint in run provenance.
    """

    profile: ArtifactIdentity
    campaign: ArtifactIdentity
    source: ArtifactIdentity
    inputs: tuple[ArtifactIdentity, ...]
    operator: ActorIdentity
    authorization_digest: str
    start: RunStartProvenance
    code_commit: str
    pod_id: str
    attempt_sha256: str
    predecessor_declaration: ArtifactIdentity
    resume: ArtifactIdentity | None = None
    registration_git_bundle: ArtifactIdentity | None = None

    def __post_init__(self) -> None:
        require_d1_pod_id(self.pod_id)
        require_commit(self.code_commit, label="remote source commit")
        if len(self.code_commit) not in (40, 64):
            raise ValueError("remote source commit requires a full Git object identity")
        require_lowercase_sha256(self.attempt_sha256, fail=digest_refusal("remote attempt"))
        require_lowercase_sha256(
            self.authorization_digest, fail=digest_refusal("remote expected authorization")
        )
        if type(self.start) is not RunStartProvenance:
            raise TypeError("remote request requires complete resource provenance")
        if type(self.operator) is not ActorIdentity or self.operator.role is not ActorRole.OPERATOR:
            raise ValueError("remote request requires the registered operator")
        if type(self.inputs) is not tuple or not self.inputs:
            raise ValueError("remote inputs must be an explicit nonempty artifact tuple")
        names: set[str] = set()
        for item in self.inputs:
            if type(item) is not ArtifactIdentity or item.content_hash.algorithm != "sha256":
                raise ValueError("remote inputs require SHA-256 artifact identities")
            normalized = normalize_d1_relative_artifact_path(item.path, label="remote input")
            if normalized != item.path or item.path in names:
                raise ValueError("remote inputs must have unique canonical relative paths")
            names.add(item.path)
        for item in (self.profile, self.campaign, self.source, self.predecessor_declaration):
            if item not in self.inputs or PurePosixPath(item.path).parts[0] == "run":
                raise ValueError("remote profile/campaign/source must be bound outside run output")
        if (
            self.registration_git_bundle is not None
            and self.registration_git_bundle not in self.inputs
        ):
            raise ValueError("registration Git bundle must be a bound input")
        if self.resume is None:
            if self.start.resume_checkpoint_sha256 is not None or any(
                PurePosixPath(name).parts[0] == "run" for name in names
            ):
                raise ValueError("fresh remote request cannot import run output")
        elif (
            self.resume not in self.inputs
            or PurePosixPath(self.resume.path).parent != PurePosixPath("run")
            or self.start.resume_checkpoint_sha256 != self.resume.content_hash.digest
        ):
            raise ValueError("remote resume must bind its checkpoint and resource segment")

    @property
    def prefix(self) -> str:
        return f"d-ladder/{self.pod_id}/{self.attempt_sha256}"

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema": "d-ladder-remote-request-v1",
            "profile": artifact_mapping(self.profile),
            "campaign": artifact_mapping(self.campaign),
            "source": artifact_mapping(self.source),
            "inputs": [artifact_mapping(item) for item in self.inputs],
            "operator": {
                "actor_id": self.operator.actor_id,
                "session_id": self.operator.session_id,
                "role": self.operator.role.value,
            },
            "authorization_digest": self.authorization_digest,
            "start": run_start_to_mapping(self.start),
            "code_commit": self.code_commit,
            "pod_id": self.pod_id,
            "attempt_sha256": self.attempt_sha256,
            "resume": None if self.resume is None else artifact_mapping(self.resume),
            "predecessor_declaration": artifact_mapping(self.predecessor_declaration),
            "registration_git_bundle": None
            if self.registration_git_bundle is None
            else artifact_mapping(self.registration_git_bundle),
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(serialize_remote_request(self)).hexdigest()


def serialize_remote_request(request: RemoteLadderRequest) -> bytes:
    return canonical_json(request.manifest).encode()


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    row: dict[str, object] = {}
    for name, value in pairs:
        if name in row:
            raise ValueError("duplicate remote JSON key")
        row[name] = value
    return row


def parse_remote_request(payload: bytes) -> RemoteLadderRequest:
    row = object_fields(
        json.loads(payload, object_pairs_hook=_unique),
        "schema profile campaign source inputs operator authorization_digest start code_commit pod_id attempt_sha256 resume predecessor_declaration registration_git_bundle",
    )
    if row["schema"] != "d-ladder-remote-request-v1" or type(row["inputs"]) is not list:
        raise ValueError("unknown remote request schema or inputs")
    for name in ("authorization_digest", "code_commit", "pod_id", "attempt_sha256"):
        if type(row[name]) is not str:
            raise ValueError("remote identity fields must be strings")
    return RemoteLadderRequest(
        profile=artifact_from_mapping(row["profile"]),
        campaign=artifact_from_mapping(row["campaign"]),
        source=artifact_from_mapping(row["source"]),
        inputs=tuple(artifact_from_mapping(item) for item in row["inputs"]),
        operator=actor_from_mapping(row["operator"]),
        authorization_digest=cast(str, row["authorization_digest"]),
        start=run_start_from_mapping(row["start"]),
        code_commit=cast(str, row["code_commit"]),
        pod_id=cast(str, row["pod_id"]),
        attempt_sha256=cast(str, row["attempt_sha256"]),
        resume=None if row["resume"] is None else artifact_from_mapping(row["resume"]),
        predecessor_declaration=artifact_from_mapping(row["predecessor_declaration"]),
        registration_git_bundle=None
        if row["registration_git_bundle"] is None
        else artifact_from_mapping(row["registration_git_bundle"]),
    )
