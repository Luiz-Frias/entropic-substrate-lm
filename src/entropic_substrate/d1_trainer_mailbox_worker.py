"""Pod-side one-attempt worker over a mounted D1 trainer mailbox."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.d1_operator_values import require_d1_pod_id
from entropic_substrate.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxCommand,
    D1TrainerMailboxCommandKind,
    D1TrainerMailboxErrorReport,
    d1_trainer_mailbox_attempt_prefix,
    d1_trainer_mailbox_command_path,
    parse_d1_trainer_mailbox_command,
    serialize_d1_trainer_mailbox_error,
)
from entropic_substrate.d1_trainer_process_contracts import (
    d1_trainer_runtime_report_path,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.experiments.d1_primitives import require_sha256

__all__ = [
    "D1TrainerMailboxWaitPolicy",
    "D1TrainerMailboxWorker",
    "D1TrainerMailboxWorkerError",
]


class D1TrainerMailboxProcessWorkerPort(Protocol):
    def bootstrap(self, pod_id: str, attempt_sha256: str) -> ArtifactIdentity: ...

    def start_trainer(
        self,
        pod_id: str,
        attempt_sha256: str,
        request_sha256: str,
    ) -> Mapping[str, object]: ...


class D1TrainerMailboxWorkerError(RuntimeError):
    """One bounded, non-secret mailbox failure."""

    def __init__(self, *, stage: str, code: str, error_type: str) -> None:
        self.stage = stage
        self.code = code
        self.error_type = error_type
        super().__init__(f"D1 trainer mailbox {stage}: {code}")


@dataclass(frozen=True, slots=True)
class D1TrainerMailboxWaitPolicy:
    maximum_seconds: float
    poll_interval_seconds: float

    def __post_init__(self) -> None:
        values = (self.maximum_seconds, self.poll_interval_seconds)
        if any(
            type(value) is not float or not math.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError("D1 trainer mailbox waits must be finite positive floats")
        if self.poll_interval_seconds > self.maximum_seconds:
            raise ValueError("D1 trainer mailbox poll interval exceeds its wait budget")


def _error(stage: str, code: str, error_type: str) -> D1TrainerMailboxWorkerError:
    return D1TrainerMailboxWorkerError(stage=stage, code=code, error_type=error_type)


class D1TrainerMailboxWorker:
    """Discover one attempt, attest runtime, then honor one immutable start command."""

    def __init__(
        self,
        *,
        volume_root: Path,
        pod_id: str,
        worker_factory: Callable[[Path], D1TrainerMailboxProcessWorkerPort],
        wait_policy: D1TrainerMailboxWaitPolicy,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._volume = normalize_d1_artifact_root(volume_root)
        self._pod_id = require_d1_pod_id(pod_id)
        if not callable(worker_factory) or not callable(monotonic) or not callable(sleep):
            raise TypeError("D1 trainer mailbox worker requires callable ports")
        if type(wait_policy) is not D1TrainerMailboxWaitPolicy:
            raise TypeError("D1 trainer mailbox worker requires a nominal wait policy")
        self._factory = worker_factory
        self._wait_policy = wait_policy
        self._monotonic = monotonic
        self._sleep = sleep
        self._ran = False

    def _wait(self, stage: str, probe: Callable[[], object | None]) -> object:
        started = self._monotonic()
        if type(started) not in (int, float) or not math.isfinite(started):
            raise _error(stage, "clock is invalid", "ValueError")
        deadline = float(started) + self._wait_policy.maximum_seconds
        while True:
            value = probe()
            if value is not None:
                return value
            current = self._monotonic()
            if type(current) not in (int, float) or not math.isfinite(current):
                raise _error(stage, "clock is invalid", "ValueError")
            remaining = deadline - float(current)
            if remaining <= 0:
                raise _error(stage, "timed out", "TimeoutError")
            self._sleep(min(self._wait_policy.poll_interval_seconds, remaining))

    def _attempt(self) -> Path | None:
        pod_root = self._volume / "d1-mailbox" / "v1" / self._pod_id
        if not pod_root.exists():
            return None
        if pod_root.is_symlink() or not pod_root.is_dir():
            raise _error("attempt", "pod namespace is not a real directory", "ValueError")
        entries = tuple(item for item in pod_root.iterdir() if not item.name.startswith("."))
        if not entries:
            return None
        if len(entries) != 1:
            raise _error("attempt", "multiple attempts are present", "ValueError")
        candidate = entries[0]
        try:
            require_sha256(candidate.name, label="D1 trainer mailbox attempt")
            if candidate.name != candidate.name.lower():
                raise ValueError("attempt digest is not lowercase")
        except ValueError as cause:
            raise _error("attempt", "attempt identity is invalid", type(cause).__name__) from cause
        if candidate.is_symlink() or not candidate.is_dir():
            raise _error("attempt", "attempt namespace is not a real directory", "ValueError")
        expected = self._volume / d1_trainer_mailbox_attempt_prefix(self._pod_id, candidate.name)
        if candidate != expected:
            raise _error("attempt", "attempt namespace differs", "ValueError")
        return candidate

    @staticmethod
    def _relative_command(command: D1TrainerMailboxCommand) -> str:
        prefix = PurePosixPath(
            d1_trainer_mailbox_attempt_prefix(command.pod_id, command.attempt_sha256)
        )
        return (
            PurePosixPath(d1_trainer_mailbox_command_path(command)).relative_to(prefix).as_posix()
        )

    @staticmethod
    def _read_command(root: Path, relative: str) -> D1TrainerMailboxCommand | None:
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.exists():
            return None
        _, payload, _ = read_d1_artifact_bytes(root, relative, label="D1 mailbox command")
        return parse_d1_trainer_mailbox_command(payload)

    def _bootstrap(self, root: Path, attempt: str) -> D1TrainerMailboxCommand | None:
        expected = D1TrainerMailboxCommand.bootstrap(
            pod_id=self._pod_id,
            attempt_sha256=attempt,
        )
        observed = self._read_command(root, self._relative_command(expected))
        if observed is not None and observed != expected:
            raise _error("bootstrap", "command identity differs", "ValueError")
        return observed

    def _start(self, root: Path, attempt: str) -> D1TrainerMailboxCommand | None:
        commands = root / "commands"
        if not commands.exists():
            return None
        if commands.is_symlink() or not commands.is_dir():
            raise _error("start", "command namespace is not a real directory", "ValueError")
        candidates = tuple(
            item
            for item in commands.iterdir()
            if item.name.startswith("start-") and item.name.endswith(".json")
        )
        if not candidates:
            return None
        if len(candidates) != 1:
            raise _error("start", "multiple start commands are present", "ValueError")
        relative = f"commands/{candidates[0].name}"
        observed = self._read_command(root, relative)
        if observed is None:  # pragma: no cover - path was just observed
            return None
        if (
            observed.kind is not D1TrainerMailboxCommandKind.START
            or observed.pod_id != self._pod_id
            or observed.attempt_sha256 != attempt
        ):
            raise _error("start", "command identity differs", "ValueError")
        return observed

    def _publish_error(self, root: Path, error: D1TrainerMailboxWorkerError) -> None:
        payload = serialize_d1_trainer_mailbox_error(
            D1TrainerMailboxErrorReport(
                self._pod_id,
                root.name,
                error.stage,
                error.error_type,
            )
        )
        publish_d1_atomic_bytes(
            root,
            "trainer-error.json",
            payload,
            label="D1 trainer mailbox error",
        )

    def run(self) -> Mapping[str, object]:
        if self._ran:
            raise _error("lifecycle", "worker already ran", "RuntimeError")
        self._ran = True
        root: Path | None = None
        stage = "attempt"
        try:
            observed_root = self._wait(stage, self._attempt)
            if not isinstance(observed_root, Path):  # pragma: no cover - _attempt is closed
                raise TypeError("attempt probe returned a non-path")
            root = observed_root
            attempt = root.name
            stage = "bootstrap"
            self._wait(stage, lambda: self._bootstrap(root, attempt))
            worker = self._factory(root)
            runtime = worker.bootstrap(self._pod_id, attempt)
            expected_runtime = root / d1_trainer_runtime_report_path(attempt)
            if type(runtime) is not ArtifactIdentity or runtime.path != str(expected_runtime):
                raise ValueError("runtime publication differs from mailbox attempt")
            stage = "start"
            start = self._wait(stage, lambda: self._start(root, attempt))
            if type(start) is not D1TrainerMailboxCommand or start.request_sha256 is None:
                raise TypeError("start probe returned a non-command")
            return dict(worker.start_trainer(self._pod_id, attempt, start.request_sha256))
        except D1TrainerMailboxWorkerError as error:
            if root is not None and error.stage != "lifecycle":
                self._publish_error(root, error)
            raise
        except Exception as cause:
            error = _error(stage, "worker action failed", type(cause).__name__)
            if root is not None:
                self._publish_error(root, error)
            raise error from cause
