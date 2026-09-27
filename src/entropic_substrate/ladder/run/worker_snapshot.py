"""Immutable output snapshots for checkpoint recovery and external byte custody."""

import json
import re
from pathlib import Path, PurePosixPath

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest
from entropic_substrate.ladder.run.worker_transport import publish_verified_object


def _path(value: object) -> str:
    name = normalize_d1_relative_artifact_path(value, label="worker output")
    path = PurePosixPath(name)
    regular = (
        len(path.parts) == 2
        and path.parts[0] in ("run", "capture", "verification", "worker")
        and path.suffix in (".json", ".pt")
    )
    if not regular and name != "capture/source/prepared-corpus.json":
        raise ValueError("worker output is outside its owned namespace")
    return name


def publish_worker_snapshot(
    objects: D1TrainerMailboxObjectPort,
    request: ProbeRequest,
    root: Path,
    paths: tuple[str, ...],
    *,
    name: str,
) -> D1TrainerMailboxObjectIdentity:
    """Snapshot explicit producer files; the marker grants no completion claim.

    Output paths are immutable across snapshots within one request. A changed
    path is a producer defect, not a new version silently hidden by snapshot names.
    """
    if type(name) is not str or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) is None:
        raise ValueError("worker snapshot name is unsafe")
    if type(paths) is not tuple or not paths or len(set(paths)) != len(paths):
        raise ValueError("worker snapshot requires unique explicit paths")
    limits = request.limits
    if len(paths) + 1 > limits.max_objects:
        raise ValueError("worker snapshot object budget exceeded")
    records: list[dict[str, object]] = []
    total = 0
    for path in sorted(paths):
        path = _path(path)
        _, payload, digest = read_d1_artifact_bytes(
            root, path, label="worker output", max_bytes=limits.max_object_bytes
        )
        if not payload:
            raise ValueError("worker snapshot cannot contain empty artifacts")
        total += len(payload)
        records.append(
            {
                "path": path,
                "sha256": digest,
                "size_bytes": len(payload),
                "key": f"{request.prefix}/outputs/{path}",
            }
        )
    manifest = {
        "schema": "h-worker-snapshot-v1",
        "authority": "byte_custody_only",
        "request_sha256": request.digest,
        "name": name,
        "files": records,
        "artifact_bytes": total,
        "artifact_count": len(records),
    }
    marker = json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if total + len(marker) > limits.max_total_bytes or len(marker) > limits.max_object_bytes:
        raise ValueError("worker snapshot byte budget exceeded")
    for record in records:
        path = _path(record["path"])
        _, payload, digest = read_d1_artifact_bytes(
            root, path, label="worker output", max_bytes=limits.max_object_bytes
        )
        if digest != record["sha256"] or len(payload) != record["size_bytes"]:
            raise ValueError("worker output changed during snapshot publication")
        publish_verified_object(objects, f"{request.prefix}/outputs/{path}", payload)
    return publish_verified_object(objects, f"{request.prefix}/snapshots/{name}.json", marker)


def materialize_worker_snapshot(
    objects: D1TrainerMailboxObjectPort,
    request: ProbeRequest,
    marker: D1TrainerMailboxObjectIdentity,
    root: Path,
    *,
    max_marker_bytes: int,
) -> dict[str, object]:
    """Refuse incomplete/mismatched byte sets; native readers remain separate.

    A late failure retains unadmitted partial files without snapshot.json. Recovery
    uses a new empty workspace; this function never deletes forensic partial data.
    """
    if (
        type(max_marker_bytes) is not int
        or max_marker_bytes < 1
        or marker.size_bytes > max_marker_bytes
    ):
        raise ValueError("worker snapshot marker byte budget exceeded")
    found = objects.read(marker.key)
    if type(found) is not D1TrainerMailboxObjectRead or found.identity != marker:
        raise ValueError("worker snapshot marker identity mismatch")
    row = recovery_record(
        json.loads(found.payload),
        "schema authority request_sha256 name files artifact_bytes artifact_count",
    )
    name, files = row["name"], row["files"]
    if (
        row["schema"] != "h-worker-snapshot-v1"
        or row["authority"] != "byte_custody_only"
        or row["request_sha256"] != request.digest
        or type(name) is not str
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) is None
        or marker.key != f"{request.prefix}/snapshots/{name}.json"
        or type(files) is not list
        or not files
        or json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        != found.payload
    ):
        raise ValueError("worker snapshot identity or schema mismatch")
    limits = request.limits
    names: list[str] = []
    identities: list[D1TrainerMailboxObjectIdentity] = []
    for item in files:
        entry = recovery_record(item, "path key sha256 size_bytes")
        path = _path(entry["path"])
        size, digest = entry["size_bytes"], entry["sha256"]
        if (
            type(size) is not int
            or not 1 <= size <= limits.max_object_bytes
            or type(digest) is not str
            or entry["key"] != f"{request.prefix}/outputs/{path}"
        ):
            raise ValueError("worker snapshot entry budget or identity mismatch")
        ContentHash("sha256", digest)
        names.append(path)
        identities.append(
            D1TrainerMailboxObjectIdentity(f"{request.prefix}/outputs/{path}", size, digest)
        )
    total = sum(item.size_bytes for item in identities)
    if (
        names != sorted(set(names))
        or type(row["artifact_count"]) is not int
        or row["artifact_count"] != len(names)
        or type(row["artifact_bytes"]) is not int
        or row["artifact_bytes"] != total
        or len(names) + 1 > limits.max_objects
        or total + marker.size_bytes > limits.max_total_bytes
        or marker.size_bytes > limits.max_object_bytes
    ):
        raise ValueError("worker snapshot inventory or budget mismatch")
    target = normalize_d1_artifact_root(root)
    if any(target.iterdir()):
        raise ValueError("worker snapshot requires an empty destination")
    for directory in sorted(
        {PurePosixPath(path).parent for path in names}, key=lambda p: len(p.parts)
    ):
        (target / directory).mkdir(parents=True, exist_ok=True)
    for path, expected in zip(names, identities, strict=True):
        current = objects.read(expected.key)
        if type(current) is not D1TrainerMailboxObjectRead or current.identity != expected:
            raise ValueError("worker snapshot object identity mismatch")
        publish_d1_atomic_bytes(target, path, current.payload, label="worker recovered output")
    publish_d1_atomic_bytes(target, "snapshot.json", found.payload, label="worker snapshot marker")
    return row
