"""Provider-neutral immutable-object transport for D1 trainer mailboxes."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from entropic_substrate.__legacy.d1_operator_values import require_d1_pod_id
from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    publish_d1_atomic_bytes,
)
from entropic_substrate.kernel.identity.d1_primitives import require_sha256
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxCommand,
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
    d1_trainer_mailbox_attempt_prefix,
    d1_trainer_mailbox_command_path,
    d1_trainer_mailbox_error_path,
    parse_d1_trainer_mailbox_error,
    serialize_d1_trainer_mailbox_command,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
)
from entropic_substrate.ladder.run.transport.d1_trainer_process_contracts import (
    D1TrainerArtifactReadback,
    d1_trainer_request_path,
    d1_trainer_runtime_report_path,
    d1_trainer_start_report_path,
)

__all__ = [
    "D1ObjectMailboxTrainerTransport",
    "D1ObjectMailboxTransportError",
    "D1TrainerMailboxObjectPort",
]


class D1TrainerMailboxObjectPort(Protocol):
    def publish(
        self,
        key: str,
        payload: bytes,
        *,
        expected_sha256: str,
    ) -> D1TrainerMailboxObjectIdentity: ...

    def read(self, key: str) -> D1TrainerMailboxObjectRead | None: ...

    def list(self, prefix: str) -> tuple[D1TrainerMailboxObjectIdentity, ...]: ...


class D1ObjectMailboxTransportError(RuntimeError):
    """A bounded object-store, ordering, or materialization failure."""

    def __init__(self, *, stage: str, code: str) -> None:
        self.stage = stage
        self.code = code
        super().__init__(f"D1 trainer object transport {stage}: {code}")


def _error(stage: str, code: str) -> D1ObjectMailboxTransportError:
    return D1ObjectMailboxTransportError(stage=stage, code=code)


def _attempt(value: str) -> str:
    require_sha256(value, label="D1 trainer object attempt")
    if value != value.lower():
        raise _error("identity", "attempt_noncanonical")
    return value


@dataclass(slots=True)
class _Session:
    pod_id: str
    attempt_sha256: str
    request_key: str | None = None
    request_sha256: str | None = None
    started: bool = False
    materialized: bool = False


class D1ObjectMailboxTrainerTransport:
    """Publish commands and materialize marker-committed output through one object port."""

    def __init__(
        self,
        *,
        objects: D1TrainerMailboxObjectPort,
        materialization_root: Path,
        wait_policy: D1TrainerMailboxWaitPolicy,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        methods = ("publish", "read", "list")
        if not all(callable(getattr(objects, method, None)) for method in methods):
            raise TypeError("D1 trainer transport requires the exact object port")
        try:
            self._materialization = normalize_d1_artifact_root(materialization_root)
        except (TypeError, ValueError) as error:
            raise ValueError("D1 trainer materialization root is invalid") from error
        if type(wait_policy) is not D1TrainerMailboxWaitPolicy:
            raise TypeError("D1 trainer object transport requires a nominal wait policy")
        if not callable(monotonic) or not callable(sleep):
            raise TypeError("D1 trainer object transport requires clock and sleep callables")
        self._objects = objects
        self._wait_policy = wait_policy
        self._monotonic = monotonic
        self._sleep = sleep
        self._sessions: dict[str, _Session] = {}

    @staticmethod
    def _prefix(pod_id: str, attempt_sha256: str) -> str:
        return d1_trainer_mailbox_attempt_prefix(pod_id, attempt_sha256)

    @classmethod
    def _key(cls, pod_id: str, attempt_sha256: str, relative: str) -> str:
        normalized = normalize_d1_relative_artifact_path(
            relative,
            label="D1 trainer mailbox relative object",
        )
        return normalize_d1_relative_artifact_path(
            f"{cls._prefix(pod_id, attempt_sha256)}/{normalized}",
            label="D1 trainer mailbox object",
        )

    def _session(self, pod_id: str, attempt_sha256: str) -> _Session:
        pod, attempt = require_d1_pod_id(pod_id), _attempt(attempt_sha256)
        session = self._sessions.get(pod)
        if session is None:
            raise _error("ordering", "session_missing")
        if session.attempt_sha256 != attempt:
            raise _error("ordering", "attempt_mismatch")
        return session

    def _publish(self, *, stage: str, key: str, payload: bytes) -> D1TrainerMailboxObjectIdentity:
        if type(payload) is not bytes or not payload:
            raise _error(stage, "payload_invalid")
        expected = D1TrainerMailboxObjectIdentity(
            key,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        try:
            observed = self._objects.publish(
                key,
                payload,
                expected_sha256=expected.sha256,
            )
        except Exception as cause:
            raise _error(stage, "object_publish") from cause
        if type(observed) is not D1TrainerMailboxObjectIdentity or observed != expected:
            raise _error(stage, "object_publish_identity")
        return observed

    def _read(self, *, stage: str, key: str) -> D1TrainerMailboxObjectRead | None:
        try:
            observed = self._objects.read(key)
        except Exception as cause:
            raise _error(stage, "object_read") from cause
        if observed is None:
            return None
        if type(observed) is not D1TrainerMailboxObjectRead or observed.identity.key != key:
            raise _error(stage, "object_read_identity")
        return observed

    def _wait(self, *, session: _Session, stage: str, key: str) -> D1TrainerMailboxObjectRead:
        started = self._monotonic()
        if type(started) not in (int, float) or not math.isfinite(started):
            raise _error(stage, "clock_invalid")
        deadline = float(started) + self._wait_policy.maximum_seconds
        error_key = d1_trainer_mailbox_error_path(session.pod_id, session.attempt_sha256)
        while True:
            target = self._read(stage=stage, key=key)
            if target is not None:
                return target
            failed = self._read(stage=stage, key=error_key)
            if failed is not None:
                try:
                    report = parse_d1_trainer_mailbox_error(failed.payload)
                except (TypeError, ValueError) as cause:
                    raise _error(stage, "worker_error_invalid") from cause
                if (
                    report.pod_id != session.pod_id
                    or report.attempt_sha256 != session.attempt_sha256
                ):
                    raise _error(stage, "worker_error_identity")
                raise _error(stage, f"worker_{report.stage}_{report.error_type}")
            current = self._monotonic()
            if type(current) not in (int, float) or not math.isfinite(current):
                raise _error(stage, "clock_invalid")
            remaining = deadline - float(current)
            if remaining <= 0:
                raise _error(stage, "timed_out")
            self._sleep(min(self._wait_policy.poll_interval_seconds, remaining))

    def bootstrap(self, pod_id: str, attempt_sha256: str) -> None:
        pod, attempt = require_d1_pod_id(pod_id), _attempt(attempt_sha256)
        if pod in self._sessions:
            raise _error("bootstrap", "session_exists")
        command = D1TrainerMailboxCommand.bootstrap(pod_id=pod, attempt_sha256=attempt)
        self._publish(
            stage="bootstrap",
            key=d1_trainer_mailbox_command_path(command),
            payload=serialize_d1_trainer_mailbox_command(command),
        )
        self._sessions[pod] = _Session(pod, attempt)

    def read_runtime_report(self, pod_id: str, attempt_sha256: str) -> bytes:
        session = self._session(pod_id, attempt_sha256)
        key = self._key(pod_id, attempt_sha256, d1_trainer_runtime_report_path(attempt_sha256))
        return self._wait(session=session, stage="runtime", key=key).payload

    def write_request(
        self,
        pod_id: str,
        attempt_sha256: str,
        path: str,
        payload: bytes,
    ) -> None:
        session = self._session(pod_id, attempt_sha256)
        expected_path = d1_trainer_request_path(attempt_sha256)
        if path != expected_path:
            raise _error("request", "request_path")
        if session.request_key is not None:
            raise _error("request", "request_exists")
        key = self._key(pod_id, attempt_sha256, path)
        identity = self._publish(stage="request", key=key, payload=payload)
        session.request_key = key
        session.request_sha256 = identity.sha256

    def read_request(
        self,
        pod_id: str,
        attempt_sha256: str,
        path: str,
    ) -> D1TrainerArtifactReadback:
        session = self._session(pod_id, attempt_sha256)
        if path != d1_trainer_request_path(attempt_sha256) or session.request_key is None:
            raise _error("request", "request_missing")
        observed = self._read(stage="request", key=session.request_key)
        if observed is None:
            raise _error("request", "request_unavailable")
        return D1TrainerArtifactReadback(path, observed.payload)

    def start_trainer(self, pod_id: str, attempt_sha256: str, request_sha256: str) -> None:
        session = self._session(pod_id, attempt_sha256)
        require_sha256(request_sha256, label="D1 trainer object request")
        if session.request_sha256 is None:
            raise _error("start", "request_missing")
        if request_sha256 != session.request_sha256:
            raise _error("start", "request_mismatch")
        if session.started:
            raise _error("start", "start_exists")
        command = D1TrainerMailboxCommand.start(
            pod_id=pod_id,
            attempt_sha256=attempt_sha256,
            request_sha256=request_sha256,
        )
        self._publish(
            stage="start",
            key=d1_trainer_mailbox_command_path(command),
            payload=serialize_d1_trainer_mailbox_command(command),
        )
        session.started = True

    def read_start_report(
        self,
        pod_id: str,
        attempt_sha256: str,
        request_sha256: str,
    ) -> bytes:
        session = self._session(pod_id, attempt_sha256)
        if not session.started or session.request_sha256 != request_sha256:
            raise _error("start_report", "start_missing")
        relative = d1_trainer_start_report_path(attempt_sha256, request_sha256)
        key = self._key(pod_id, attempt_sha256, relative)
        return self._wait(session=session, stage="start_report", key=key).payload

    def _listing(
        self,
        session: _Session,
    ) -> tuple[D1TrainerMailboxObjectIdentity, ...]:
        prefix = f"{self._prefix(session.pod_id, session.attempt_sha256)}/"
        try:
            values = self._objects.list(prefix)
        except Exception as cause:
            raise _error("export", "object_list") from cause
        if type(values) is not tuple or any(
            type(item) is not D1TrainerMailboxObjectIdentity for item in values
        ):
            raise _error("export", "object_list_identity")
        keys = tuple(item.key for item in values)
        if not values or len(set(keys)) != len(keys):
            raise _error("export", "object_list_duplicate")
        if any(not key.startswith(prefix) for key in keys):
            raise _error("export", "foreign_object")
        return values

    def _materialize(
        self,
        session: _Session,
        values: tuple[D1TrainerMailboxObjectIdentity, ...],
        *,
        marker_key: str,
    ) -> Path:
        pod_root = self._materialization / session.pod_id
        if pod_root.exists() and (pod_root.is_symlink() or not pod_root.is_dir()):
            raise _error("export", "materialization_pod_root")
        pod_root.mkdir(exist_ok=True)
        destination = pod_root / session.attempt_sha256
        try:
            destination.mkdir()
        except FileExistsError as cause:
            raise _error("export", "materialization_exists") from cause
        prefix = PurePosixPath(self._prefix(session.pod_id, session.attempt_sha256))
        ordered = sorted(values, key=lambda item: (item.key == marker_key, item.key))
        for identity in ordered:
            observed = self._read(stage="export", key=identity.key)
            if observed is None or observed.identity != identity:
                raise _error("export", "materialization_read")
            try:
                relative = PurePosixPath(identity.key).relative_to(prefix).as_posix()
            except ValueError as cause:  # pragma: no cover - listing prefix pins this
                raise _error("export", "foreign_object") from cause
            target = destination.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_d1_atomic_bytes(
                destination,
                relative,
                observed.payload,
                label="D1 trainer materialized object",
            )
        return destination

    def await_export(
        self,
        pod_id: str,
        attempt_sha256: str,
        prefix: str,
    ) -> Path:
        session = self._session(pod_id, attempt_sha256)
        if not session.started:
            raise _error("export", "start_missing")
        if session.materialized:
            raise _error("export", "export_exists")
        publication = normalize_d1_relative_artifact_path(
            prefix,
            label="D1 trainer publication prefix",
        )
        marker_key = self._key(pod_id, attempt_sha256, f"{publication}.complete.json")
        marker = self._wait(session=session, stage="export", key=marker_key)
        values = self._listing(session)
        if marker.identity not in values:
            raise _error("export", "marker_unlisted")
        destination = self._materialize(session, values, marker_key=marker_key)
        session.materialized = True
        return destination
