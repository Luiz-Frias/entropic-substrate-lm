"""Bounded H input transfer through the existing immutable object port.

Transport verifies bytes, not native corpus/model semantics. Production object
adapters must enforce the request's object-byte cap before streaming any body.
"""

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest, parse_probe_request


@dataclass
class VerificationReadRetry:
    """Bounded retry for a checkpoint-snapshot R2 VERIFICATION READ only.

    Never wraps ``objects.publish``: no PUT is ever retried (house rule 3's
    carve-out, operator-ruled 2026-09-28, FOR-1567 -- measured
    ``D1TrainerS3ObjectError provider_failure`` on a checkpoint-snapshot
    verification READ at 15:01:55Z). ``attempts`` (3 total: the first try plus
    up to 2 retries) is the RULED count. ``backoff_seconds`` is a small,
    fixed, non-random schedule and is UNMEASURED -- this comment makes no
    claim that it saves wall-clock time; the wall-clock effect of this retry
    is unmeasured offline, full stop. ``failures`` is mutated in place by
    :func:`publish_verified_object` to accumulate every failed verification
    read attempt observed while publishing one checkpoint snapshot, so the
    caller can record the count in that event's ``publication`` block; it is
    bookkeeping only, never itself part of the retry predicate.
    """

    attempts: int = 3
    backoff_seconds: tuple[float, ...] = (0.05, 0.1)
    sleep: Callable[[float], None] = time.sleep
    failures: int = 0

    def __post_init__(self) -> None:
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("verification-read retry requires a positive attempt budget")
        if type(self.backoff_seconds) is not tuple or not all(
            type(value) in (int, float) and not isinstance(value, bool) and value >= 0
            for value in self.backoff_seconds
        ):
            raise ValueError("verification-read backoff schedule must be nonnegative numbers")
        if type(self.failures) is not int or self.failures < 0:
            raise ValueError("verification-read failure count must be a nonnegative integer")
        if not callable(self.sleep):
            raise TypeError("verification-read retry requires a callable sleep")


def _verified_read(
    objects: D1TrainerMailboxObjectPort,
    key: str,
    expected: D1TrainerMailboxObjectIdentity,
    payload: bytes,
    retry: VerificationReadRetry,
    *,
    budget: int,
) -> D1TrainerMailboxObjectRead | None:
    """Bounded, backed-off retry on ``objects.read`` alone.

    A raised exception counts as a failed attempt and is retried up to
    ``budget`` times total, then re-raised UNCHANGED (no wrapper type). A
    read that returns cleanly but does not match ``expected``/``payload`` is
    never retried -- it returns ``None`` on its first occurrence so the
    caller raises its own contextual exception.
    """
    if budget < 1:
        return None
    for index in range(budget):
        try:
            received = objects.read(key)
        except Exception:
            retry.failures += 1
            if index == budget - 1:
                raise
            retry.sleep(retry.backoff_seconds[min(index, len(retry.backoff_seconds) - 1)])
            continue
        if (
            type(received) is D1TrainerMailboxObjectRead
            and received.identity == expected
            and received.payload == payload
        ):
            return received
        return None
    return None  # pragma: no cover - the loop above always returns or raises


def publish_verified_object(
    objects: D1TrainerMailboxObjectPort,
    key: str,
    payload: bytes,
    *,
    retry: VerificationReadRetry | None = None,
) -> D1TrainerMailboxObjectIdentity:
    expected = D1TrainerMailboxObjectIdentity(
        key, len(payload), hashlib.sha256(payload).hexdigest()
    )
    try:
        actual = objects.publish(key, payload, expected_sha256=expected.sha256)
    except Exception as publish_error:
        if retry is None:
            raise
        # The adapter's own post-PUT readback (or collision-confirm read) can
        # raise from inside objects.publish, before this function ever makes
        # its own objects.read call. That failure already spent one
        # verification-read attempt; recover it through the SAME counted,
        # backed-off read loop used below -- never by re-issuing put_object
        # or re-calling objects.publish.
        retry.failures += 1
        recovered = _verified_read(
            objects, key, expected, payload, retry, budget=retry.attempts - 1
        )
        if recovered is None:
            raise publish_error
        return expected
    if type(actual) is not D1TrainerMailboxObjectIdentity or actual != expected:
        raise ValueError("worker publication identity mismatch")
    if retry is None:
        received = objects.read(key)
        if (
            type(received) is not D1TrainerMailboxObjectRead
            or received.identity != expected
            or received.payload != payload
        ):
            raise ValueError("worker publication readback identity mismatch")
        return expected
    received = _verified_read(objects, key, expected, payload, retry, budget=retry.attempts)
    if received is None:
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


def read_published_probe_request(
    objects: D1TrainerMailboxObjectPort,
    marker: D1TrainerMailboxObjectIdentity,
    *,
    max_request_bytes: int,
) -> ProbeRequest:
    """Admit the canonical request marker before acting on its input graph."""
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
    return request


def materialize_probe_request(
    objects: D1TrainerMailboxObjectPort,
    marker: D1TrainerMailboxObjectIdentity,
    root: Path,
    *,
    max_request_bytes: int,
) -> ProbeRequest:
    """Use an empty owned workspace; a partial transfer has no local request marker."""
    request = read_published_probe_request(objects, marker, max_request_bytes=max_request_bytes)
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
    publish_d1_atomic_bytes(target, "request.json", request.payload, label="worker request marker")
    return request
