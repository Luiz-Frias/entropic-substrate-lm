"""CLI-neutral application service for authoritative D1 evidence capture."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, cast

import yaml

from entropic_substrate.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
    D1TerminalDefectInput,
)
from entropic_substrate.evidence.d1_capture_aggregate_validation import (
    build_d1_capture_aggregate,
)
from entropic_substrate.evidence.d1_capture_artifacts import D1ArtifactIdentity
from entropic_substrate.evidence.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureBundle,
    D1CaptureInput,
    D1DvcCaptureHandoff,
)
from entropic_substrate.evidence.d1_capture_ports import (
    D1CaptureTransportError,
    D1CloudObjectVerifierPort,
    D1DvcCommandPort,
    D1DvcPointerIdentity,
    D1EvidenceExportSourcePort,
    D1GitPointerStatusVerifierPort,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1LaunchRoster,
    D1PairExclusion,
)
from entropic_substrate.evidence.d1_capture_validation import build_d1_capture_bundle

__all__ = ["D1CaptureReceipt", "D1CaptureService", "D1HeterogeneousCaptureReceipt"]

_CAPTURE_RELATIVE: Final = Path("results/captured")
_POINTER_RELATIVE: Final = Path("results/captured.dvc")


def _error(code: str, path: str, detail: str) -> D1CaptureTransportError:
    return D1CaptureTransportError(code=code, path=path, detail=detail)


def _commit(value: str) -> None:
    if (
        type(value) is not str
        or not 7 <= len(value) <= 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("pointer_commit must be a 7-to-64-character lowercase Git identity")


@dataclass(frozen=True, slots=True)
class D1CaptureReceipt:
    """Analysis-ready bundle plus its durable DVC and Git identities."""

    bundle: D1CaptureBundle
    pointer: D1DvcPointerIdentity
    pointer_commit: str

    def __post_init__(self) -> None:
        if type(self.bundle) is not D1CaptureBundle:
            raise TypeError("capture receipt requires a D1CaptureBundle")
        if type(self.pointer) is not D1DvcPointerIdentity:
            raise TypeError("capture receipt requires a D1DvcPointerIdentity")
        _commit(self.pointer_commit)
        if not self.bundle.dvc_handoff.analysis_ready:
            raise ValueError("capture receipt cannot contain a pending DVC handoff")
        if any(
            artifact.dvc.pointer_path != self.pointer.pointer_path
            or artifact.dvc.pointer_sha256 != self.pointer.pointer_sha256
            or artifact.dvc.object_id != self.pointer.object_id
            for artifact in self.bundle.dvc_handoff.artifacts
        ):
            raise ValueError("capture receipt bundle differs from its DVC pointer identity")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_ready": True,
            "bundle_digest": self.bundle.manifest_digest,
            "pointer_commit": self.pointer_commit,
            "pointer_identity": {
                "object_id": self.pointer.object_id,
                "pointer_path": self.pointer.pointer_path,
                "pointer_sha256": self.pointer.pointer_sha256,
            },
            "schema_version": "d1-capture-receipt-v1",
        }


@dataclass(frozen=True, slots=True)
class D1HeterogeneousCaptureReceipt:
    """Sole analysis handoff for a complete/terminal-defect capture aggregate."""

    aggregate: D1HeterogeneousCaptureAggregate
    pointer: D1DvcPointerIdentity
    pointer_commit: str

    def __post_init__(self) -> None:
        if type(self.aggregate) is not D1HeterogeneousCaptureAggregate:
            raise TypeError("heterogeneous receipt requires a D1 capture aggregate")
        if type(self.pointer) is not D1DvcPointerIdentity:
            raise TypeError("heterogeneous receipt requires a D1DvcPointerIdentity")
        _commit(self.pointer_commit)
        if not self.aggregate.dvc_handoff.analysis_ready:
            raise ValueError("heterogeneous receipt cannot contain a pending DVC handoff")
        if any(
            artifact.dvc.pointer_path != self.pointer.pointer_path
            or artifact.dvc.pointer_sha256 != self.pointer.pointer_sha256
            or artifact.dvc.object_id != self.pointer.object_id
            for artifact in self.aggregate.dvc_handoff.artifacts
        ):
            raise ValueError("heterogeneous aggregate differs from its DVC pointer identity")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "aggregate_digest": self.aggregate.manifest_digest,
            "analysis_ready": True,
            "pointer_commit": self.pointer_commit,
            "pointer_identity": {
                "object_id": self.pointer.object_id,
                "pointer_path": self.pointer.pointer_path,
                "pointer_sha256": self.pointer.pointer_sha256,
            },
            "schema_version": "d1-heterogeneous-capture-receipt-v1",
        }


def _read_pointer(repo_root: Path) -> D1DvcPointerIdentity:
    pointer = repo_root / _POINTER_RELATIVE
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(pointer, flags)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("DVC pointer is not a regular file")
            payload_bytes = handle.read()
        payload = yaml.safe_load(payload_bytes)
        if not isinstance(payload, dict):
            raise ValueError("DVC pointer must contain a mapping")
        outs = payload.get("outs")
        if not isinstance(outs, list) or len(outs) != 1 or not isinstance(outs[0], dict):
            raise ValueError("DVC pointer must contain exactly one output mapping")
        output = cast(dict[object, object], outs[0])
        if output.get("path") != _CAPTURE_RELATIVE.name:
            raise ValueError("DVC pointer output path differs from results/captured")
        object_id = output.get("md5")
        if type(object_id) is not str:
            raise ValueError("DVC pointer output requires an md5 object identity")
        return D1DvcPointerIdentity(
            pointer_path=_POINTER_RELATIVE.as_posix(),
            pointer_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            object_id=object_id,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise _error("pointer_identity", _POINTER_RELATIVE.as_posix(), str(error)) from error


def _bind_input(capture: D1CaptureInput, by_path: dict[str, D1ArtifactIdentity]) -> D1CaptureInput:
    return replace(
        capture,
        journal=by_path[capture.journal.normalized_path],
        checkpoint=by_path[capture.checkpoint.normalized_path],
        billing_sidecars=tuple(by_path[item.normalized_path] for item in capture.billing_sidecars),
        quarantined_siblings=tuple(
            by_path[item.normalized_path] for item in capture.quarantined_siblings
        ),
    )


def _bind_bundle(bundle: D1CaptureBundle, pointer: D1DvcPointerIdentity) -> D1CaptureBundle:
    by_path = {
        artifact.normalized_path: replace(artifact, dvc=pointer.binding)
        for artifact in bundle.dvc_handoff.artifacts
    }
    confirmatory = tuple(
        replace(item, capture_input=_bind_input(item.capture_input, by_path))
        for item in bundle.confirmatory
    )
    attempts = tuple(
        replace(
            item,
            artifacts=tuple(by_path[artifact.normalized_path] for artifact in item.artifacts),
        )
        for item in bundle.attempts
    )
    return D1CaptureBundle.from_validated_evidence(
        roster=bundle.roster,
        set_proof=bundle.set_proof,
        confirmatory=confirmatory,
        attempts=attempts,
        pair_exclusions=bundle.pair_exclusions,
        dvc_handoff=D1DvcCaptureHandoff(tuple(by_path.values())),
    )


def _bind_aggregate(
    aggregate: D1HeterogeneousCaptureAggregate,
    pointer: D1DvcPointerIdentity,
) -> D1HeterogeneousCaptureAggregate:
    by_path = {
        artifact.normalized_path: replace(artifact, dvc=pointer.binding)
        for artifact in aggregate.dvc_handoff.artifacts
    }
    complete = tuple(
        replace(item, capture_input=_bind_input(item.capture_input, by_path))
        for item in aggregate.complete_runs
    )
    defects = tuple(
        replace(
            item,
            capture_input=replace(
                item.capture_input,
                journal=by_path[item.capture_input.journal.normalized_path],
                supporting_artifacts=tuple(
                    by_path[artifact.normalized_path]
                    for artifact in item.capture_input.supporting_artifacts
                ),
            ),
        )
        for item in aggregate.terminal_defects
    )
    attempts = tuple(
        replace(
            item,
            artifacts=tuple(by_path[artifact.normalized_path] for artifact in item.artifacts),
        )
        for item in aggregate.attempts
    )
    return D1HeterogeneousCaptureAggregate.from_validated_evidence(
        roster=aggregate.roster,
        partition_proof=aggregate.partition_proof,
        complete_runs=complete,
        terminal_defects=defects,
        attempts=attempts,
        pair_exclusions=aggregate.pair_exclusions,
        dvc_handoff=D1DvcCaptureHandoff(
            tuple(by_path[item.normalized_path] for item in aggregate.dvc_handoff.artifacts)
        ),
    )


class D1CaptureService:
    """Run the complete D1 local-to-DVC custody transaction through ports."""

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str],
        evidence_source: D1EvidenceExportSourcePort,
        dvc: D1DvcCommandPort,
        cloud: D1CloudObjectVerifierPort,
        git: D1GitPointerStatusVerifierPort,
    ) -> None:
        root = Path(os.path.abspath(repo_root))
        if root.is_symlink() or not root.is_dir():
            raise _error("repo_root", str(root), "must be an existing real directory")
        self._repo_root = root
        self._evidence_source = evidence_source
        self._dvc = dvc
        self._cloud = cloud
        self._git = git

    def _export(self, artifacts: tuple[D1ArtifactIdentity, ...]) -> Path:
        if not artifacts or any(
            item.dvc.pointer_path != _POINTER_RELATIVE.as_posix() for item in artifacts
        ):
            raise _error(
                "pointer_path",
                _POINTER_RELATIVE.as_posix(),
                "every artifact must predeclare the canonical capture pointer",
            )
        capture_root = self._repo_root / _CAPTURE_RELATIVE
        try:
            exported = self._evidence_source.export(
                artifacts=artifacts,
                destination_root=capture_root,
            )
        except D1CaptureTransportError:
            raise
        except Exception as error:
            raise _error("source_export", _CAPTURE_RELATIVE.as_posix(), str(error)) from error
        if exported != artifacts:
            raise _error(
                "source_contract",
                _CAPTURE_RELATIVE.as_posix(),
                "evidence source returned identities other than those requested",
            )
        return capture_root

    def _durable_identity(self, capture_root: Path) -> tuple[D1DvcPointerIdentity, str]:
        pointer_path = self._repo_root / _POINTER_RELATIVE
        try:
            self._dvc.add(tracked_path=capture_root, pointer_path=pointer_path)
            self._dvc.push(pointer_path=pointer_path)
        except Exception as error:
            raise _error("dvc_failure", _POINTER_RELATIVE.as_posix(), str(error)) from error
        pointer = _read_pointer(self._repo_root)
        try:
            cloud_present = self._cloud.object_is_present(
                pointer_path=pointer_path,
                object_id=pointer.object_id,
            )
        except Exception as error:
            raise _error("cloud_mismatch", pointer.pointer_path, str(error)) from error
        if cloud_present is not True:
            raise _error(
                "cloud_mismatch", pointer.pointer_path, "remote object presence was not proven"
            )
        try:
            commit = self._git.committed_revision(
                pointer_path=pointer_path,
                pointer_sha256=pointer.pointer_sha256,
            )
        except Exception as error:
            raise _error("pointer_uncommitted", pointer.pointer_path, str(error)) from error
        if commit is None:
            raise _error(
                "pointer_uncommitted",
                pointer.pointer_path,
                "exact DVC pointer bytes are not present in the verified Git revision",
            )
        return pointer, commit

    def capture(
        self,
        *,
        roster: D1LaunchRoster,
        capture_inputs: tuple[D1CaptureInput, ...],
        attempt_evidence: tuple[D1AttemptEvidence, ...] = (),
        pair_exclusions: tuple[D1PairExclusion, ...] = (),
    ) -> D1CaptureReceipt:
        if type(roster) is not D1LaunchRoster or type(capture_inputs) is not tuple:
            raise TypeError("D1 capture requires a nominal roster and capture-input tuple")
        input_ids = [item.logical_run_id for item in capture_inputs]
        if len(input_ids) != len(set(input_ids)) or frozenset(input_ids) != roster.expected_run_ids:
            raise _error(
                "capture_input_set",
                "confirmatory",
                "does not equal the admitted launch roster",
            )
        artifacts = tuple(item for capture in capture_inputs for item in capture.artifacts) + tuple(
            item for evidence in attempt_evidence for item in evidence.artifacts
        )
        capture_root = self._export(artifacts)
        try:
            bundle = build_d1_capture_bundle(
                capture_root,
                roster=roster,
                capture_inputs=capture_inputs,
                attempt_evidence=attempt_evidence,
                pair_exclusions=pair_exclusions,
            )
        except Exception as error:
            raise _error("local_admission", _CAPTURE_RELATIVE.as_posix(), str(error)) from error
        pointer, commit = self._durable_identity(capture_root)
        return D1CaptureReceipt(
            bundle=_bind_bundle(bundle, pointer),
            pointer=pointer,
            pointer_commit=commit,
        )

    def capture_heterogeneous(
        self,
        *,
        roster: D1LaunchRoster,
        complete_inputs: tuple[D1CaptureInput, ...],
        terminal_defect_inputs: tuple[D1TerminalDefectInput, ...],
        attempt_evidence: tuple[D1AttemptEvidence, ...] = (),
    ) -> D1HeterogeneousCaptureReceipt:
        """Capture an exact roster partition and expose no loose defect lane."""
        if type(roster) is not D1LaunchRoster:
            raise TypeError("heterogeneous capture roster must be D1LaunchRoster")
        if type(complete_inputs) is not tuple or type(terminal_defect_inputs) is not tuple:
            raise TypeError("heterogeneous capture lanes must be tuples")
        complete_ids = [item.logical_run_id for item in complete_inputs]
        defect_ids = [item.logical_run_id for item in terminal_defect_inputs]
        if (
            len(complete_ids) != len(set(complete_ids))
            or len(defect_ids) != len(set(defect_ids))
            or set(complete_ids) & set(defect_ids)
            or frozenset((*complete_ids, *defect_ids)) != roster.expected_run_ids
        ):
            raise _error(
                "capture_partition",
                "roster",
                "complete and terminal-defect lanes must be an exact disjoint roster union",
            )
        artifacts = (
            tuple(item for capture in complete_inputs for item in capture.artifacts)
            + tuple(item for capture in terminal_defect_inputs for item in capture.artifacts)
            + tuple(item for evidence in attempt_evidence for item in evidence.artifacts)
        )
        capture_root = self._export(artifacts)
        try:
            aggregate = build_d1_capture_aggregate(
                capture_root,
                roster=roster,
                complete_inputs=complete_inputs,
                terminal_defect_inputs=terminal_defect_inputs,
                attempt_evidence=attempt_evidence,
            )
        except Exception as error:
            raise _error("local_admission", _CAPTURE_RELATIVE.as_posix(), str(error)) from error
        pointer, commit = self._durable_identity(capture_root)
        return D1HeterogeneousCaptureReceipt(
            aggregate=_bind_aggregate(aggregate, pointer),
            pointer=pointer,
            pointer_commit=commit,
        )
