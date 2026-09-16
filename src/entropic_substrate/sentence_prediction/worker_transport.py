"""Bounded H input transfer through the existing immutable object port.

Transport verifies bytes, not native corpus/model semantics. Production object
adapters must enforce the request's object-byte cap before streaming any body.
"""

import hashlib
from pathlib import Path

from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.d1_trainer_object_transport import D1TrainerMailboxObjectPort
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.sentence_prediction.worker_request import ProbeRequest, parse_probe_request


def publish_verified_object(
    objects: D1TrainerMailboxObjectPort, key: str, payload: bytes
) -> D1TrainerMailboxObjectIdentity:
    expected = D1TrainerMailboxObjectIdentity(
        key, len(payload), hashlib.sha256(payload).hexdigest()
    )
    actual = objects.publish(key, payload, expected_sha256=expected.sha256)
    if type(actual) is not D1TrainerMailboxObjectIdentity or actual != expected:
        raise ValueError("worker publication identity mismatch")
    received = objects.read(key)
    if (
        type(received) is not D1TrainerMailboxObjectRead
        or received.identity != expected
        or received.payload != payload
    ):
        raise ValueError("worker publication readback identity mismatch")
    return expected


def _input(root: Path, artifact: ArtifactIdentity, maximum: int) -> bytes:
    _, data, digest = read_d1_artifact_bytes(
        root, artifact.path, label="worker input", max_bytes=maximum
    )
    if digest != artifact.content_hash.digest:
        raise ValueError("worker local input identity mismatch")
    return data


def publish_probe_request(
    objects: D1TrainerMailboxObjectPort, request: ProbeRequest, root: Path
) -> D1TrainerMailboxObjectIdentity:
    """Preflight the whole input budget, then publish the request commit marker last."""
    request.__post_init__()
    limits = request.limits
    payload = request.payload
    if len(payload) > limits.max_object_bytes or len(request.inputs) + 1 > limits.max_objects:
        raise ValueError("worker request exceeds transfer budget")
    total = len(payload)
    for artifact in request.inputs:
        total += len(_input(root, artifact, limits.max_object_bytes))
        if total > limits.max_total_bytes:
            raise ValueError("worker input total byte budget exceeded")
    for artifact in request.inputs:
        publish_verified_object(
            objects,
            f"{request.prefix}/inputs/{artifact.path}",
            _input(root, artifact, limits.max_object_bytes),
        )
    return publish_verified_object(objects, f"{request.prefix}/request.json", payload)


def materialize_probe_request(
    objects: D1TrainerMailboxObjectPort,
    marker: D1TrainerMailboxObjectIdentity,
    root: Path,
    *,
    max_request_bytes: int,
) -> ProbeRequest:
    """Use an empty owned workspace; a partial transfer has no local request marker."""
    if (
        type(max_request_bytes) is not int
        or max_request_bytes < 1
        or marker.size_bytes > max_request_bytes
    ):
        raise ValueError("worker request exceeds its admission byte limit")
    received = objects.read(marker.key)
    if type(received) is not D1TrainerMailboxObjectRead or received.identity != marker:
        raise ValueError("worker request object identity mismatch")
    request = parse_probe_request(received.payload, max_bytes=max_request_bytes)
    if marker.key != f"{request.prefix}/request.json" or marker.sha256 != request.digest:
        raise ValueError("worker request prefix or content identity mismatch")
    limits = request.limits
    if marker.size_bytes > limits.max_object_bytes or len(request.inputs) + 1 > limits.max_objects:
        raise ValueError("worker materialization exceeds object budget")
    target = normalize_d1_artifact_root(root)
    if any(target.iterdir()):
        raise ValueError("worker materialization requires an empty workspace")
    (target / "source").mkdir()
    (target / "run").mkdir()
    total = marker.size_bytes
    for artifact in request.inputs:
        key = f"{request.prefix}/inputs/{artifact.path}"
        result = objects.read(key)
        if (
            type(result) is not D1TrainerMailboxObjectRead
            or result.identity.key != key
            or result.identity.sha256 != artifact.content_hash.digest
        ):
            raise ValueError("worker input object identity mismatch")
        total += result.identity.size_bytes
        if result.identity.size_bytes > limits.max_object_bytes or total > limits.max_total_bytes:
            raise ValueError("worker input materialization byte budget exceeded")
        publish_d1_atomic_bytes(target, artifact.path, result.payload, label="worker input")
    publish_d1_atomic_bytes(target, "request.json", received.payload, label="worker request marker")
    return request
