"""Pod-side one-attempt D1 trainer worker over immutable R2 mailbox objects."""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Protocol

from entropic_substrate.composition.entrypoints.d1_trainer_process_worker import (
    D1TrainerProcessWorker,
    D1TrainerStartReportPublisher,
)
from entropic_substrate.kernel.artifacts import (
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile
from entropic_substrate.ladder.register.d1_provenance import require_lowercase_digest
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxCommand,
    D1TrainerMailboxErrorReport,
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
    d1_trainer_mailbox_attempt_prefix,
    d1_trainer_mailbox_error_path,
    parse_d1_trainer_mailbox_command,
    serialize_d1_trainer_mailbox_error,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.transport.d1_trainer_process_contracts import (
    d1_trainer_request_path,
    d1_trainer_runtime_report_path,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    D1ResumeTrainerRequest,
    D1TrainerRequest,
    d1_trainer_request_sha256,
    parse_d1_trainer_request,
)

__all__ = [
    "D1R2CheckpointRecoveryPort",
    "D1R2TrainerMailboxWorker",
    "D1R2TrainerMailboxWorkerError",
]

type D1R2ProcessWorkerFactory = Callable[[D1TrainerStartReportPublisher], D1TrainerProcessWorker]


class D1R2CheckpointRecoveryPort(Protocol):
    def restore(
        self,
        *,
        logical_run_id: str,
        expected_checkpoint_sha256: str,
    ) -> object: ...


class D1R2TrainerMailboxWorkerError(RuntimeError):
    """A bounded R2 mailbox discovery, execution, or publication failure."""

    def __init__(self, *, stage: str, code: str, error_type: str) -> None:
        self.stage = stage
        self.code = code
        self.error_type = error_type
        super().__init__(f"D1 R2 trainer mailbox {stage}: {code}")


def _error(stage: str, code: str, error_type: str = "ValueError") -> D1R2TrainerMailboxWorkerError:
    return D1R2TrainerMailboxWorkerError(stage=stage, code=code, error_type=error_type)


class D1R2TrainerMailboxWorker:
    """Poll R2, run the real trainer locally, and publish terminal marker last."""

    def __init__(
        self,
        *,
        repo_root: Path,
        pod_id: str,
        objects: D1TrainerMailboxObjectPort,
        catalog: Iterable[D1LaunchProfile],
        worker_factory: D1R2ProcessWorkerFactory,
        recovery: D1R2CheckpointRecoveryPort,
        wait_policy: D1TrainerMailboxWaitPolicy,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        root = Path(os.path.abspath(repo_root))
        if root.is_symlink() or not root.is_dir():
            raise ValueError("D1 R2 worker repo_root must be a real directory")
        if not all(callable(getattr(objects, name, None)) for name in ("publish", "read", "list")):
            raise TypeError("D1 R2 worker requires the mailbox object port")
        profiles = tuple(catalog)
        if not profiles or any(type(item) is not D1LaunchProfile for item in profiles):
            raise TypeError("D1 R2 worker requires an exact profile catalog")
        if not callable(worker_factory) or not callable(monotonic) or not callable(sleep):
            raise TypeError("D1 R2 worker requires callable process ports")
        if not callable(getattr(recovery, "restore", None)):
            raise TypeError("D1 R2 worker requires the checkpoint recovery port")
        if type(wait_policy) is not D1TrainerMailboxWaitPolicy:
            raise TypeError("D1 R2 worker requires a nominal wait policy")
        self._root = root
        self._pod_id = require_d1_pod_id(pod_id)
        self._objects = objects
        self._catalog = profiles
        self._worker_factory = worker_factory
        self._recovery = recovery
        self._wait_policy = wait_policy
        self._monotonic = monotonic
        self._sleep = sleep
        self._ran = False

    def _read(self, key: str, *, stage: str) -> D1TrainerMailboxObjectRead | None:
        try:
            observed = self._objects.read(key)
        except Exception as cause:
            raise _error(stage, "object_read", type(cause).__name__) from cause
        if observed is None:
            return None
        if type(observed) is not D1TrainerMailboxObjectRead or observed.identity.key != key:
            raise _error(stage, "object_identity")
        return observed

    def _publish(self, key: str, payload: bytes, *, stage: str) -> D1TrainerMailboxObjectIdentity:
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
            raise _error(stage, "object_publish", type(cause).__name__) from cause
        if type(observed) is not D1TrainerMailboxObjectIdentity or observed != expected:
            raise _error(stage, "object_publish_identity")
        return observed

    def _wait(self, stage: str, probe: Callable[[], object | None]) -> object:
        started = self._monotonic()
        if type(started) not in (int, float) or not math.isfinite(started):
            raise _error(stage, "clock_invalid")
        deadline = float(started) + self._wait_policy.maximum_seconds
        while True:
            value = probe()
            if value is not None:
                return value
            current = self._monotonic()
            if type(current) not in (int, float) or not math.isfinite(current):
                raise _error(stage, "clock_invalid")
            remaining = deadline - float(current)
            if remaining <= 0:
                raise _error(stage, "timed_out", "TimeoutError")
            self._sleep(min(self._wait_policy.poll_interval_seconds, remaining))

    def _attempt(self) -> str | None:
        prefix = f"d1-mailbox/v1/{self._pod_id}/"
        try:
            values = self._objects.list(prefix)
        except Exception as cause:
            raise _error("attempt", "object_list", type(cause).__name__) from cause
        candidates: set[str] = set()
        base = PurePosixPath(prefix)
        for identity in values:
            if type(identity) is not D1TrainerMailboxObjectIdentity:
                raise _error("attempt", "object_list_identity")
            try:
                relative = PurePosixPath(identity.key).relative_to(base)
            except ValueError as cause:
                raise _error("attempt", "foreign_object") from cause
            if len(relative.parts) == 3 and relative.parts[1:] == (
                "commands",
                "bootstrap.json",
            ):
                attempt = relative.parts[0]
                try:
                    require_lowercase_digest(attempt, label="D1 R2 worker attempt")
                except ValueError as cause:
                    raise _error("attempt", "attempt_invalid") from cause
                candidates.add(attempt)
        if not candidates:
            return None
        if len(candidates) != 1:
            raise _error("attempt", "multiple_attempts")
        return candidates.pop()

    def _command(
        self, key: str, expected: D1TrainerMailboxCommand
    ) -> D1TrainerMailboxCommand | None:
        observed = self._read(key, stage=expected.kind.value)
        if observed is None:
            return None
        try:
            command = parse_d1_trainer_mailbox_command(observed.payload)
        except (TypeError, ValueError) as cause:
            raise _error(expected.kind.value, "command_invalid") from cause
        if command != expected:
            raise _error(expected.kind.value, "command_identity")
        return command

    def _request(self, prefix: str, attempt: str) -> tuple[D1TrainerRequest, bytes]:
        relative = d1_trainer_request_path(attempt)
        key = f"{prefix}/{relative}"
        observed = self._wait("request", lambda: self._read(key, stage="request"))
        if type(observed) is not D1TrainerMailboxObjectRead:
            raise _error("request", "object_identity")
        try:
            request = parse_d1_trainer_request(observed.payload, catalog=self._catalog)
        except (TypeError, ValueError) as cause:
            raise _error("request", "request_invalid") from cause
        target = self._root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_d1_atomic_bytes(
            self._root,
            relative,
            observed.payload,
            label="D1 R2 staged trainer request",
        )
        return request, observed.payload

    def _publish_local(self, relative: str, prefix: str, *, stage: str) -> None:
        _, payload, _ = read_d1_artifact_bytes(self._root, relative, label=f"D1 R2 {stage}")
        self._publish(f"{prefix}/{relative}", payload, stage=stage)

    def _publish_terminal(self, request: D1TrainerRequest, prefix: str) -> None:
        run_relative = PurePosixPath(request.locations.checkpoint_path).parent
        run_root = self._root.joinpath(*run_relative.parts)
        if run_root.is_symlink() or not run_root.is_dir():
            raise _error("export", "run_root")
        marker_relative = f"{request.locations.checkpoint_path}.complete.json"
        files = tuple(sorted(path for path in run_root.rglob("*") if path.is_file()))
        relative_files = tuple(path.relative_to(self._root).as_posix() for path in files)
        if marker_relative not in relative_files:
            raise _error("export", "marker_missing")
        ordered = sorted(relative_files, key=lambda item: (item == marker_relative, item))
        for relative in ordered:
            self._publish_local(relative, prefix, stage="export")

    def _publish_error(self, attempt: str, stage: str, cause: Exception) -> None:
        report = D1TrainerMailboxErrorReport(
            self._pod_id,
            attempt,
            stage,
            type(cause).__name__,
        )
        self._publish(
            d1_trainer_mailbox_error_path(self._pod_id, attempt),
            serialize_d1_trainer_mailbox_error(report),
            stage="error",
        )

    def run(self) -> Mapping[str, object]:
        if self._ran:
            raise _error("lifecycle", "worker_already_ran", "RuntimeError")
        self._ran = True
        attempt: str | None = None
        stage = "attempt"
        try:
            observed_attempt = self._wait(stage, self._attempt)
            if type(observed_attempt) is not str:
                raise _error(stage, "attempt_invalid")
            attempt = observed_attempt
            prefix = d1_trainer_mailbox_attempt_prefix(self._pod_id, attempt)
            bootstrap = D1TrainerMailboxCommand.bootstrap(
                pod_id=self._pod_id,
                attempt_sha256=attempt,
            )
            stage = "bootstrap"
            key = f"{prefix}/commands/bootstrap.json"
            self._wait(stage, lambda: self._command(key, bootstrap))

            def start_publisher(relative: str, payload: bytes) -> None:
                self._publish(f"{prefix}/{relative}", payload, stage="start")

            process = self._worker_factory(start_publisher)
            if type(process) is not D1TrainerProcessWorker:
                raise _error(stage, "worker_factory")
            process.bootstrap(self._pod_id, attempt)
            self._publish_local(
                d1_trainer_runtime_report_path(attempt),
                prefix,
                stage="runtime",
            )
            stage = "request"
            request, request_payload = self._request(prefix, attempt)
            request_sha = d1_trainer_request_sha256(request)
            if hashlib.sha256(request_payload).hexdigest() != request_sha:
                raise _error(stage, "request_digest")
            start = D1TrainerMailboxCommand.start(
                pod_id=self._pod_id,
                attempt_sha256=attempt,
                request_sha256=request_sha,
            )
            stage = "start"
            self._wait(
                stage,
                lambda: self._command(f"{prefix}/commands/start-{request_sha}.json", start),
            )
            if type(request) is D1ResumeTrainerRequest:
                stage = "recovery"
                checkpoint_sha256 = request.start.resume_checkpoint_sha256
                if checkpoint_sha256 is None:
                    raise _error(stage, "checkpoint_source")
                self._recovery.restore(
                    logical_run_id=request.plan.logical_run_id,
                    expected_checkpoint_sha256=checkpoint_sha256,
                )
            result = process.start_trainer(self._pod_id, attempt, request_sha)
            stage = "export"
            self._publish_terminal(request, prefix)
            return dict(result)
        except D1R2TrainerMailboxWorkerError:
            raise
        except Exception as cause:
            if attempt is not None:
                self._publish_error(attempt, stage, cause)
            raise _error(stage, "worker_failed", type(cause).__name__) from cause
