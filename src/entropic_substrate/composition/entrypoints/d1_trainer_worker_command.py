"""Internal foreground argv command for the pod-side D1 trainer worker."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from entropic_substrate.composition.entrypoints.d1_trainer_process_worker import (
    D1TrainerProcessWorker,
)
from entropic_substrate.composition.wiring.d1_cli import D1CliInvalidEvidence
from entropic_substrate.kernel.artifacts import D1ArtifactSnapshotError
from entropic_substrate.ladder.capture.adapters.d1_observability import (
    AuthoritativeJournalOnlyObserver,
    D1JournalObserver,
)
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile, load_d1_execution_catalog
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1RuntimeObservation
from entropic_substrate.ladder.run.provider.d1_system_runtime_probe import (
    D1RuntimeProbeError,
    SystemD1RuntimeProbe,
)
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerClock
from entropic_substrate.ladder.run.transport.d1_trainer_process_contracts import (
    d1_trainer_runtime_report_path,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import D1TrainerRequestError

__all__ = ["main", "run_d1_trainer_worker_command"]

_SCHEMA = "d1-trainer-worker-command-v1"
_COMMANDS = frozenset({"bootstrap", "start"})
_COMMON = frozenset({"--artifact-root", "--pod-id", "--attempt-sha256"})


class D1TrainerWorkerCommandRefusal(ValueError):
    """The internal argv shape is incomplete, duplicated, or unsupported."""


def _canonical(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _emit(stream: TextIO, value: Mapping[str, object]) -> None:
    stream.write(_canonical(value) + "\n")
    stream.flush()


def _parse_argv(argv: Sequence[str]) -> tuple[str, dict[str, str]]:
    values = tuple(argv)
    if not values or any(type(item) is not str or not item for item in values):
        raise D1TrainerWorkerCommandRefusal("invalid worker argv")
    command, tail = values[0], values[1:]
    if command not in _COMMANDS or len(tail) % 2:
        raise D1TrainerWorkerCommandRefusal("invalid worker argv")
    arguments: dict[str, str] = {}
    for index in range(0, len(tail), 2):
        flag, value = tail[index], tail[index + 1]
        if flag in arguments or not flag.startswith("--") or not value:
            raise D1TrainerWorkerCommandRefusal("invalid worker argv")
        arguments[flag] = value
    expected = _COMMON | ({"--request-sha256"} if command == "start" else frozenset())
    if frozenset(arguments) != expected:
        raise D1TrainerWorkerCommandRefusal("invalid worker argv")
    return command, arguments


def _installed_catalog() -> tuple[D1LaunchProfile, ...]:
    return load_d1_execution_catalog()


def _error(command: str, code: str, error: Exception) -> dict[str, object]:
    return {
        "command": command,
        "error": {"code": code, "type": type(error).__name__},
        "schema_version": _SCHEMA,
        "status": "error",
    }


def run_d1_trainer_worker_command(
    argv: Sequence[str],
    *,
    catalog: Iterable[D1LaunchProfile] | None = None,
    runtime_probe: Callable[[str], D1RuntimeObservation] | None = None,
    process_identity: Callable[[], str] | None = None,
    observer: D1JournalObserver | None = None,
    clock: D1TrainerClock | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run one foreground worker action and emit exactly one canonical result."""
    command = argv[0] if argv and argv[0] in _COMMANDS else "unknown"
    try:
        command, arguments = _parse_argv(argv)
        selected_catalog = _installed_catalog() if catalog is None else tuple(catalog)
        worker = D1TrainerProcessWorker(
            artifact_root=Path(arguments["--artifact-root"]),
            catalog=selected_catalog,
            runtime_probe=runtime_probe if runtime_probe is not None else SystemD1RuntimeProbe(),
            process_identity=(
                process_identity if process_identity is not None else lambda: str(os.getpid())
            ),
            observer=observer if observer is not None else AuthoritativeJournalOnlyObserver(),
            clock=clock,
        )
        pod = arguments["--pod-id"]
        attempt = arguments["--attempt-sha256"]
        if command == "bootstrap":
            identity = worker.bootstrap(pod, attempt)
            payload: Mapping[str, object] = {
                "report_path": d1_trainer_runtime_report_path(attempt),
                "report_sha256": identity.content_hash.digest,
            }
        else:
            payload = worker.start_trainer(pod, attempt, arguments["--request-sha256"])
        _emit(
            stdout,
            {
                "command": command,
                "payload": dict(payload),
                "schema_version": _SCHEMA,
                "status": "success",
            },
        )
        return 0
    except D1RuntimeProbeError as error:
        _emit(stderr, _error(command, error.code, error))
        return 20
    except (
        D1ArtifactSnapshotError,
        D1CliInvalidEvidence,
        D1TrainerRequestError,
        D1TrainerWorkerCommandRefusal,
    ) as error:
        _emit(stderr, _error(command, "worker_refused", error))
        return 21
    except Exception as error:
        _emit(stderr, _error(command, "unexpected_defect", error))
        return 70


def main(argv: Sequence[str] | None = None) -> int:
    return run_d1_trainer_worker_command(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
