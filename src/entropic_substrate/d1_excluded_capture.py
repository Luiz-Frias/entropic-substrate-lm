"""DVC custody for excluded pilots, separate from confirmatory capture admission."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
    relative_d1_artifact_path,
    snapshot_d1_artifact,
)
from entropic_substrate.d1_excluded_operator import D1ExcludedRunResult
from entropic_substrate.evidence.d1_capture_artifacts import D1ArtifactRole, identify_d1_artifact
from entropic_substrate.evidence.d1_capture_filesystem import LocalD1EvidenceExportSource
from entropic_substrate.evidence.d1_capture_ports import (
    D1CloudObjectVerifierPort,
    D1DvcCommandPort,
    D1DvcPointerIdentity,
    D1GitPointerStatusVerifierPort,
)
from entropic_substrate.evidence.d1_capture_service import read_d1_dvc_pointer
from entropic_substrate.evidence.d1_telemetry_query import read_d1_run_evidence_bytes
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.experiments.d1_config import D1LaunchProfile
from entropic_substrate.experiments.d1_launch_attempt_codec import (
    launch_attempt_ledger_from_mapping,
)
from entropic_substrate.experiments.d1_plan import ExcludedPilotD1RunPlan, parse_d1_pilot_plan
from entropic_substrate.experiments.d1_primitives import RunClass, canonical_digest, canonical_json

_RECEIPT_TOKEN = object()


@dataclass(frozen=True, slots=True)
class D1ExcludedCapturePreparation:
    plan: ExcludedPilotD1RunPlan
    repo_root: Path
    tracked_path: Path
    pointer: D1DvcPointerIdentity
    pointer_bytes: bytes
    artifacts: tuple[ArtifactIdentity, ...]
    journal: ArtifactIdentity

    def verify(self) -> bytes:
        observed, pointer_bytes = read_d1_dvc_pointer(
            repo_root=self.repo_root,
            tracked_path=self.tracked_path,
            pointer_path=self.repo_root / self.pointer.pointer_path,
        )
        if observed != self.pointer or pointer_bytes != self.pointer_bytes:
            raise ValueError("excluded capture pointer hash changed")
        journal_bytes = b""
        for artifact in self.artifacts:
            relative = relative_d1_artifact_path(
                self.repo_root, Path(artifact.path), label="capture"
            )
            if artifact == self.journal:
                _, journal_bytes, digest = read_d1_artifact_bytes(
                    self.repo_root, relative, label="capture"
                )
            else:
                _, _, digest = snapshot_d1_artifact(self.repo_root, relative, label="capture")
            if artifact.content_hash != ContentHash("sha256", digest):
                raise ValueError("excluded capture artifact hash changed")
        if not journal_bytes:
            raise ValueError("excluded capture journal is absent")
        read_d1_run_evidence_bytes(
            journal_bytes,
            journal_path=Path(self.journal.path),
            expected_run_class=RunClass.PILOT,
            expected_logical_run_id=self.plan.logical_run_id,
            expected_family_digest=self.plan.profile.family.family_digest,
            expected_plan_digest=self.plan.plan_digest,
        )
        return journal_bytes


@dataclass(frozen=True, slots=True)
class D1ExcludedCaptureReceipt:
    """Committed pilot pointer plus re-verifiable local bytes; no successor authority."""

    preparation: D1ExcludedCapturePreparation
    pointer_commit: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _RECEIPT_TOKEN:
            raise TypeError("excluded capture receipt requires verified finalization")

    @property
    def plan(self) -> ExcludedPilotD1RunPlan:
        return self.preparation.plan

    @property
    def repo_root(self) -> Path:
        return self.preparation.repo_root

    @property
    def pointer(self) -> D1DvcPointerIdentity:
        return self.preparation.pointer

    @property
    def journal(self) -> ArtifactIdentity:
        return self.preparation.journal

    @property
    def artifacts(self) -> tuple[ArtifactIdentity, ...]:
        return self.preparation.artifacts

    @property
    def journal_sha256(self) -> str:
        return self.journal.content_hash.digest

    @property
    def capture_digest(self) -> str:
        return canonical_digest(
            {
                "plan_digest": self.plan.plan_digest,
                "journal_sha256": self.journal_sha256,
                "pointer_sha256": self.pointer.pointer_sha256,
                "pointer_commit": self.pointer_commit,
            }
        )

    def read_journal_bytes(self) -> bytes:
        return self.preparation.verify()


class D1ExcludedCaptureService:
    def __init__(
        self,
        *,
        repo_root: Path,
        source_root: Path,
        dvc: D1DvcCommandPort,
        cloud: D1CloudObjectVerifierPort,
        git: D1GitPointerStatusVerifierPort,
    ) -> None:
        self.repo_root = normalize_d1_artifact_root(repo_root)
        self.source_root = normalize_d1_artifact_root(source_root)
        self._dvc, self._cloud, self._git = dvc, cloud, git

    def load(
        self,
        tracked_path: Path,
        *,
        catalog: Iterable[D1LaunchProfile],
    ) -> D1ExcludedCaptureReceipt:
        """Reverify a persisted pilot capture before a fresh analysis session."""
        relative = relative_d1_artifact_path(self.repo_root, tracked_path, label="excluded capture")
        _, raw, manifest_digest = read_d1_artifact_bytes(
            self.repo_root,
            f"{relative}/excluded-capture.json",
            label="excluded manifest",
        )
        value: object = json.loads(raw)
        if type(value) is not dict or set(value) != {
            "schema_version",
            "plan",
            "attempt_ledger",
            "artifacts",
        }:
            raise ValueError("excluded capture manifest fields differ")
        payload = cast(dict[str, object], value)
        if (
            payload["schema_version"] != "d1-excluded-capture-v1"
            or canonical_json(payload).encode() != raw
        ):
            raise ValueError("excluded capture manifest is not canonical")
        plan_value = payload["plan"]
        if type(plan_value) is not dict:
            raise ValueError("excluded capture plan is missing")
        plan_mapping = cast(dict[str, object], plan_value)
        matches = tuple(
            profile
            for profile in catalog
            if (
                profile.name == plan_mapping.get("profile_name")
                and profile.draft.construction_digest == plan_mapping.get("construction_digest")
                and profile.family.family_digest == plan_mapping.get("family_digest")
            )
        )
        if len(matches) != 1:
            raise ValueError("excluded capture requires one installed construction")
        plan = parse_d1_pilot_plan(matches[0], plan_mapping)
        if relative != f"results/d1-excluded/{plan.plan_digest}":
            raise ValueError("excluded capture path differs from its plan identity")
        launch_attempt_ledger_from_mapping(payload["attempt_ledger"])
        pointer, pointer_bytes = read_d1_dvc_pointer(
            repo_root=self.repo_root,
            tracked_path=tracked_path,
            pointer_path=tracked_path.with_suffix(".dvc"),
        )
        raw_artifacts = payload["artifacts"]
        if type(raw_artifacts) is not list or len(raw_artifacts) < 2:
            raise ValueError("excluded capture requires journal and checkpoint artifacts")
        artifacts: list[ArtifactIdentity] = []
        for index, raw_artifact in enumerate(cast(list[object], raw_artifacts)):
            if type(raw_artifact) is not dict:
                raise ValueError("excluded capture artifact must be an object")
            item = cast(dict[str, object], raw_artifact)
            path = normalize_d1_relative_artifact_path(item.get("path"), label="excluded artifact")
            role = (
                D1ArtifactRole.JOURNAL
                if index == 0
                else D1ArtifactRole.CHECKPOINT
                if index == 1
                else D1ArtifactRole.BILLING_SIDECAR
            )
            observed = identify_d1_artifact(
                tracked_path, tracked_path / path, role=role, dvc_pointer_path=pointer.pointer_path
            )
            if canonical_json(observed.manifest) != canonical_json(item):
                raise ValueError("excluded capture artifact hash or role changed")
            artifacts.append(
                ArtifactIdentity(str(tracked_path / path), ContentHash("sha256", observed.sha256))
            )
        artifacts.append(
            ArtifactIdentity(
                str(tracked_path / "excluded-capture.json"), ContentHash("sha256", manifest_digest)
            )
        )
        if (
            self._cloud.object_is_present(
                pointer_path=self.repo_root / pointer.pointer_path, object_id=pointer.object_id
            )
            is not True
        ):
            raise ValueError("excluded capture remote object presence is not proven")
        return self.finalize(
            D1ExcludedCapturePreparation(
                plan,
                self.repo_root,
                tracked_path,
                pointer,
                pointer_bytes,
                tuple(artifacts),
                artifacts[0],
            )
        )

    def prepare(self, result: D1ExcludedRunResult) -> D1ExcludedCapturePreparation:
        if (
            type(result) is not D1ExcludedRunResult
            or type(result.plan) is not ExcludedPilotD1RunPlan
        ):
            raise TypeError("excluded capture requires a nominal pilot result")
        if type(result.export.checkpoint) is not ArtifactIdentity:
            raise ValueError("excluded capture needs a completed pilot checkpoint")
        if not result.finalization.quality_export_ready:
            raise ValueError("excluded capture requires complete pilot quality evidence")
        destination = self.repo_root / "results" / "d1-excluded" / result.plan.plan_digest
        parent = self.repo_root
        for part in destination.relative_to(self.repo_root).parts:
            parent /= part
            if parent.is_symlink():
                raise ValueError("excluded capture destination traverses a symlink")
            parent.mkdir(exist_ok=True)
        normalize_d1_artifact_root(destination)
        pointer_path = destination.with_suffix(".dvc")
        pointer_relative = pointer_path.relative_to(self.repo_root).as_posix()
        originals = (result.journal, result.export.checkpoint, *result.custody_artifacts)
        identities = []
        for index, artifact in enumerate(originals):
            source = Path(artifact.path)
            if not source.is_absolute():
                source = self.source_root / source
            role = (
                D1ArtifactRole.JOURNAL
                if index == 0
                else D1ArtifactRole.CHECKPOINT
                if index == 1
                else D1ArtifactRole.BILLING_SIDECAR
            )
            identity = identify_d1_artifact(
                self.source_root,
                source,
                role=role,
                dvc_pointer_path=pointer_relative,
            )
            if artifact.content_hash != ContentHash("sha256", identity.sha256):
                raise ValueError("excluded source artifact hash differs from operator receipt")
            identities.append(identity)
        LocalD1EvidenceExportSource(self.source_root).export(
            artifacts=tuple(identities),
            destination_root=destination,
        )
        artifacts = tuple(
            ArtifactIdentity(
                str(destination / item.normalized_path), ContentHash("sha256", item.sha256)
            )
            for item in identities
        )
        manifest = publish_d1_atomic_bytes(
            destination,
            "excluded-capture.json",
            canonical_json(
                {
                    "schema_version": "d1-excluded-capture-v1",
                    "plan": result.plan.plan_manifest,
                    "attempt_ledger": result.attempt_ledger.manifest,
                    "artifacts": [item.manifest for item in identities],
                }
            ).encode(),
            label="excluded capture manifest",
        )
        self._dvc.add(tracked_path=destination, pointer_path=pointer_path)
        self._dvc.push(pointer_path=pointer_path)
        pointer, pointer_bytes = read_d1_dvc_pointer(
            repo_root=self.repo_root,
            tracked_path=destination,
            pointer_path=pointer_path,
        )
        if (
            self._cloud.object_is_present(pointer_path=pointer_path, object_id=pointer.object_id)
            is not True
        ):
            raise ValueError("excluded capture remote object presence is not proven")
        prepared = D1ExcludedCapturePreparation(
            result.plan,
            self.repo_root,
            destination,
            pointer,
            pointer_bytes,
            (*artifacts, manifest),
            artifacts[0],
        )
        prepared.verify()
        return prepared

    def finalize(self, preparation: D1ExcludedCapturePreparation) -> D1ExcludedCaptureReceipt:
        if (
            type(preparation) is not D1ExcludedCapturePreparation
            or preparation.repo_root != self.repo_root
        ):
            raise ValueError("excluded capture preparation belongs to another repository")
        preparation.verify()
        pointer = preparation.pointer
        evidence = self._git.committed_pointer(
            pointer_path=self.repo_root / pointer.pointer_path,
            pointer_bytes=preparation.pointer_bytes,
            pointer_sha256=pointer.pointer_sha256,
            object_id=pointer.object_id,
        )
        if evidence is None:
            raise ValueError("excluded capture pointer is not committed")
        if (evidence.pointer_path, evidence.pointer_sha256, evidence.object_id) != (
            pointer.pointer_path,
            pointer.pointer_sha256,
            pointer.object_id,
        ):
            raise ValueError("excluded committed pointer evidence differs")
        return D1ExcludedCaptureReceipt(preparation, evidence.commit, _RECEIPT_TOKEN)
