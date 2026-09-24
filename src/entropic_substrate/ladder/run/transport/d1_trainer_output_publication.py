"""Atomic handoff-first, completion-marker-last publication for D1 trainers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NoReturn, cast

from entropic_substrate.__legacy.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1TrainerExportReceipt,
    verify_d1_trainer_export,
)
from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    publish_d1_atomic_bytes,
    relative_d1_artifact_path,
    snapshot_d1_artifact,
)
from entropic_substrate.kernel.evidence_class.resource_provenance import EvidenceGap
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.ladder.run.transport.d1_trainer_handoff_codec import (
    parse_d1_trainer_handoff,
    serialize_d1_trainer_handoff,
)

__all__ = [
    "D1TrainerOutputPublicationError",
    "load_d1_published_trainer_output",
    "publish_d1_trainer_output",
]

_SCHEMA = "d1-trainer-export-marker-v1"
_MARKER_KEYS = frozenset(
    {
        "artifacts",
        "execution_segment_index",
        "handoff_kind",
        "logical_run_id",
        "schema_version",
    }
)
_ARTIFACT_KEYS = frozenset({"checkpoint", "handoff", "journal", "quarantine"})
_REFERENCE_KEYS = frozenset({"kind", "path", "sha256", "size_bytes"})


class D1TrainerOutputPublicationError(ValueError):
    """A trainer export bundle is partial, drifted, or noncanonical."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1TrainerOutputPublicationError(message)
    raise D1TrainerOutputPublicationError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"{label} keys must be exact")


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be a nonempty trimmed string")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a nonnegative integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    digest = _text(value, label=label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _publication_paths(prefix: object) -> tuple[str, str]:
    value = normalize_d1_relative_artifact_path(prefix, label="trainer publication prefix")
    return f"{value}.handoff.json", f"{value}.complete.json"


def _read_stable_artifact(
    root: Path,
    relative: object,
    *,
    label: str,
) -> tuple[Path, bytes, int, str]:
    path, size, digest = snapshot_d1_artifact(root, relative, label=label)
    try:
        data = path.read_bytes()
    except OSError as error:
        _fail(f"{label} could not be read", error)
    if (len(data), hashlib.sha256(data).hexdigest()) != (size, digest):
        _fail(f"{label} changed after its descriptor-bound snapshot")
    return path, data, size, digest


def _reference(
    root: Path,
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    relative = relative_d1_artifact_path(root, path, label=label)
    _, size, digest = snapshot_d1_artifact(root, relative, label=label)
    if expected_sha256 is not None and digest != expected_sha256:
        _fail(f"{label} changed after trainer export verification")
    return {
        "kind": "artifact",
        "path": relative,
        "sha256": digest,
        "size_bytes": size,
    }


def _gap(value: EvidenceGap) -> dict[str, object]:
    return {"field": value.field, "kind": "gap", "reason": value.reason}


def _checkpoint_marker(receipt: D1TrainerExportReceipt, root: Path) -> dict[str, object]:
    checkpoint = receipt.checkpoint
    if isinstance(checkpoint, ArtifactIdentity):
        return _reference(
            root,
            Path(checkpoint.path),
            label="trainer checkpoint",
            expected_sha256=checkpoint.content_hash.digest,
        )
    return _gap(checkpoint)


def _quarantine_marker(receipt: D1TrainerExportReceipt, root: Path) -> dict[str, object]:
    handoff = receipt.handoff
    if type(handoff) is not D1ResumeRunCompletion or handoff.tail_quarantine is None:
        return {"kind": "none"}
    return _reference(
        root,
        handoff.tail_quarantine.quarantine_path,
        label="trainer quarantine",
        expected_sha256=handoff.tail_quarantine.quarantined_journal_sha256,
    )


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def publish_d1_trainer_output(
    candidate: D1ExportedTrainerHandoff,
    *,
    artifact_root: Path,
    publication_prefix: str,
) -> ArtifactIdentity:
    """Verify exports, publish the handoff, then commit the bundle marker."""
    if type(candidate) is not D1ExportedTrainerHandoff:
        raise TypeError("trainer output publication requires D1ExportedTrainerHandoff")
    root = normalize_d1_artifact_root(artifact_root)
    handoff_relative, marker_relative = _publication_paths(publication_prefix)
    receipt = verify_d1_trainer_export(candidate)
    handoff_bytes = serialize_d1_trainer_handoff(receipt.handoff, artifact_root=root)
    handoff_artifact = publish_d1_atomic_bytes(
        root,
        handoff_relative,
        handoff_bytes,
        label="trainer handoff",
    )
    marker: dict[str, object] = {
        "artifacts": {
            "checkpoint": _checkpoint_marker(receipt, root),
            "handoff": _reference(
                root,
                root / handoff_relative,
                label="trainer handoff",
                expected_sha256=handoff_artifact.content_hash.digest,
            ),
            "journal": _reference(
                root,
                Path(receipt.journal.path),
                label="trainer journal",
                expected_sha256=receipt.journal.content_hash.digest,
            ),
            "quarantine": _quarantine_marker(receipt, root),
        },
        "execution_segment_index": receipt.handoff.termination.execution_segment_index,
        "handoff_kind": receipt.handoff_kind,
        "logical_run_id": receipt.handoff.termination.logical_run_id,
        "schema_version": _SCHEMA,
    }
    return publish_d1_atomic_bytes(
        root,
        marker_relative,
        _canonical(marker),
        label="trainer completion marker",
    )


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"trainer completion marker contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_marker(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(f"non-finite JSON constant {constant}"),
        )
    except D1TrainerOutputPublicationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("trainer completion marker must be UTF-8 JSON", error)
    marker = _mapping(value, label="trainer completion marker")
    if _canonical(marker) != payload:
        _fail("trainer completion marker must use canonical JSON")
    _exact(marker, _MARKER_KEYS, label="trainer completion marker")
    if marker["schema_version"] != _SCHEMA:
        _fail(f"trainer completion marker schema_version must be {_SCHEMA}")
    return marker


def _admit_reference(value: object, root: Path, *, label: str) -> tuple[Path, str]:
    item = _mapping(value, label=label)
    _exact(item, _REFERENCE_KEYS, label=label)
    if item["kind"] != "artifact":
        _fail(f"{label} must be an artifact reference")
    relative = normalize_d1_relative_artifact_path(item["path"], label=label)
    path, size, digest = snapshot_d1_artifact(root, relative, label=label)
    expected_digest = _sha256(item["sha256"], label=f"{label} sha256")
    if size != _integer(item["size_bytes"], label=f"{label} size_bytes") or digest != (
        expected_digest
    ):
        _fail(f"{label} bytes differ from marker identity")
    return path, expected_digest


def _compare_checkpoint(value: object, receipt: D1TrainerExportReceipt, root: Path) -> None:
    checkpoint = receipt.checkpoint
    if isinstance(checkpoint, ArtifactIdentity):
        path, _ = _admit_reference(value, root, label="trainer checkpoint")
        if str(path) != checkpoint.path:
            _fail("trainer checkpoint marker path differs from verified export")
        return
    item = _mapping(value, label="trainer checkpoint gap")
    if item != _gap(checkpoint):
        _fail("trainer checkpoint gap differs from verified export")


def _compare_quarantine(value: object, receipt: D1TrainerExportReceipt, root: Path) -> None:
    handoff = receipt.handoff
    if type(handoff) is D1ResumeRunCompletion and handoff.tail_quarantine is not None:
        path, _ = _admit_reference(value, root, label="trainer quarantine")
        if path != handoff.tail_quarantine.quarantine_path:
            _fail("trainer quarantine marker path differs from verified handoff")
        return
    if value != {"kind": "none"}:
        _fail("trainer quarantine marker differs from no-tail handoff")


def load_d1_published_trainer_output(
    *,
    artifact_root: Path,
    publication_prefix: str,
) -> D1ExportedTrainerHandoff:
    """Admit marker-committed bytes and reconstruct the existing handoff candidate."""
    root = normalize_d1_artifact_root(artifact_root)
    handoff_relative, marker_relative = _publication_paths(publication_prefix)
    try:
        _, marker_bytes, _, _ = _read_stable_artifact(
            root,
            marker_relative,
            label="trainer completion marker",
        )
    except ValueError as error:
        _fail(f"trainer completion marker is unavailable: {error}", error)
    marker = _decode_marker(marker_bytes)
    artifacts = _mapping(marker["artifacts"], label="trainer completion artifacts")
    _exact(artifacts, _ARTIFACT_KEYS, label="trainer completion artifacts")
    handoff_path, marker_handoff_digest = _admit_reference(
        artifacts["handoff"], root, label="trainer handoff"
    )
    if handoff_path != root / handoff_relative:
        _fail("trainer handoff marker path differs from publication prefix")
    _, handoff_bytes, _, observed_handoff_digest = _read_stable_artifact(
        root,
        handoff_relative,
        label="trainer handoff",
    )
    if observed_handoff_digest != marker_handoff_digest:
        _fail("trainer handoff changed after marker admission")
    handoff = parse_d1_trainer_handoff(handoff_bytes, artifact_root=root)
    journal_path, marker_journal_digest = _admit_reference(
        artifacts["journal"], root, label="trainer journal"
    )
    candidate = D1ExportedTrainerHandoff(handoff, journal_path)
    receipt = verify_d1_trainer_export(candidate)
    _, _, observed_journal_digest = snapshot_d1_artifact(
        root,
        relative_d1_artifact_path(root, journal_path, label="trainer journal"),
        label="trainer journal",
    )
    if (
        receipt.journal.content_hash.digest != marker_journal_digest
        or observed_journal_digest != marker_journal_digest
    ):
        _fail("trainer journal changed after marker admission")
    marker_logical_run_id = _text(marker["logical_run_id"], label="marker logical_run_id")
    marker_segment = _integer(
        marker["execution_segment_index"], label="marker execution_segment_index"
    )
    marker_kind = _text(marker["handoff_kind"], label="marker handoff_kind")
    if (
        marker_logical_run_id != handoff.termination.logical_run_id
        or marker_segment != handoff.termination.execution_segment_index
        or marker_kind != receipt.handoff_kind
    ):
        _fail("trainer completion marker run identity differs from handoff")
    _compare_checkpoint(artifacts["checkpoint"], receipt, root)
    _compare_quarantine(artifacts["quarantine"], receipt, root)
    return candidate
