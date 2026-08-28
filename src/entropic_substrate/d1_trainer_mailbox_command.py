"""Foreground RunPod command for one mounted D1 trainer mailbox worker."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn, Protocol, TextIO, cast

from entropic_substrate.d1_observability import AuthoritativeJournalOnlyObserver
from entropic_substrate.d1_operator_values import require_d1_pod_id
from entropic_substrate.d1_r2_runtime_composition import (
    D1R2RuntimeCompositionError,
    build_d1_r2_runtime_composition,
)
from entropic_substrate.d1_r2_trainer_mailbox_worker import (
    D1R2TrainerMailboxWorker,
    D1R2TrainerMailboxWorkerError,
)
from entropic_substrate.d1_system_runtime_probe import SystemD1RuntimeProbe
from entropic_substrate.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
    D1TrainerMailboxWorkerError,
)
from entropic_substrate.d1_trainer_process_worker import D1TrainerProcessWorker
from entropic_substrate.experiments.d1_config import load_d1_profile_catalog

__all__ = ["main", "run_d1_trainer_mailbox_command"]

_SCHEMA = "d1-trainer-mailbox-command-result-v1"
_FLAGS = frozenset(
    {
        "--max-mailbox-objects",
        "--maximum-wait-seconds",
        "--poll-interval-seconds",
        "--repo-root",
    }
)


class D1TrainerMailboxCommandRefusal(ValueError):
    """The foreground command lacks an exact safe operator configuration."""


class D1RunnableMailbox(Protocol):
    def run(self) -> Mapping[str, object]: ...


type D1MailboxFactory = Callable[
    [Path, str, D1TrainerMailboxWaitPolicy],
    D1RunnableMailbox,
]


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _emit(stream: TextIO, value: Mapping[str, object]) -> None:
    stream.write(_canonical(value).decode("utf-8") + "\n")
    stream.flush()


def _refuse(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1TrainerMailboxCommandRefusal(message)
    raise D1TrainerMailboxCommandRefusal(message) from error


def _parse_argv(argv: Sequence[str]) -> tuple[Path, D1TrainerMailboxWaitPolicy, int]:
    values = tuple(argv)
    if not values or len(values) % 2 or any(type(item) is not str or not item for item in values):
        _refuse("mailbox command argv is incomplete")
    parsed: dict[str, str] = {}
    for index in range(0, len(values), 2):
        flag, value = values[index], values[index + 1]
        if flag not in _FLAGS or flag in parsed:
            _refuse("mailbox command argv fields differ")
        parsed[flag] = value
    if frozenset(parsed) != _FLAGS:
        _refuse("mailbox command argv fields differ")
    try:
        policy = D1TrainerMailboxWaitPolicy(
            float(parsed["--maximum-wait-seconds"]),
            float(parsed["--poll-interval-seconds"]),
        )
        maximum = int(parsed["--max-mailbox-objects"])
        if maximum < 1:
            raise ValueError("max mailbox objects must be positive")
    except (TypeError, ValueError) as error:
        _refuse("mailbox wait policy is invalid", error)
    return Path(parsed["--repo-root"]), policy, maximum


def _string_environment(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str or type(item) is not str for key, item in value.items()
    ):
        raise TypeError("mailbox command environment must be a string mapping")
    return cast(Mapping[str, str], value)


def _pod_id(environ: Mapping[str, str]) -> str:
    observed = _string_environment(environ)
    try:
        return require_d1_pod_id(observed.get("RUNPOD_POD_ID"))
    except (TypeError, ValueError) as error:
        _refuse("RUNPOD_POD_ID is absent or unsafe", error)


def _default_mailbox_factory(
    root: Path,
    pod_id: str,
    policy: D1TrainerMailboxWaitPolicy,
    *,
    environ: Mapping[str, str],
    max_objects: int,
) -> D1R2TrainerMailboxWorker:
    catalog = (*load_d1_profile_catalog(), *load_d1_profile_catalog(tiny=True))
    runtime = build_d1_r2_runtime_composition(
        repo_root=root,
        environ=environ,
        max_objects=max_objects,
    )

    def worker_factory(start_report_publisher) -> D1TrainerProcessWorker:  # type: ignore[no-untyped-def]
        return D1TrainerProcessWorker(
            artifact_root=root,
            catalog=catalog,
            runtime_probe=SystemD1RuntimeProbe(),
            process_identity=lambda: str(os.getpid()),
            observer=AuthoritativeJournalOnlyObserver(),
            checkpoint_sync_factory=lambda pod, attempt, logical: runtime.bind_checkpoint_sync(
                pod_id=pod,
                attempt_sha256=attempt,
                logical_run_id=logical,
            ),
            start_report_publisher=start_report_publisher,
        )

    return D1R2TrainerMailboxWorker(
        repo_root=root,
        pod_id=pod_id,
        objects=runtime.mailbox,
        catalog=catalog,
        worker_factory=worker_factory,
        recovery=runtime.checkpoint_recovery,
        wait_policy=policy,
    )


def _error(code: str, error_type: str) -> dict[str, object]:
    return {
        "error": {"code": code, "type": error_type},
        "schema_version": _SCHEMA,
        "status": "error",
    }


def run_d1_trainer_mailbox_command(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str],
    mailbox_factory: D1MailboxFactory | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run exactly one mailbox attempt and emit only bounded diagnostic JSON."""
    try:
        root, policy, max_objects = _parse_argv(argv)
        pod = _pod_id(environ)
        if mailbox_factory is None:
            mailbox = _default_mailbox_factory(
                root,
                pod,
                policy,
                environ=environ,
                max_objects=max_objects,
            )
        else:
            mailbox = mailbox_factory(root, pod, policy)
        result = mailbox.run()
        result_sha256 = hashlib.sha256(_canonical(result)).hexdigest()
        _emit(
            stdout,
            {
                "result_sha256": result_sha256,
                "schema_version": _SCHEMA,
                "status": "success",
            },
        )
        return 0
    except D1TrainerMailboxCommandRefusal as error:
        _emit(stderr, _error("command_refused", type(error).__name__))
        return 20
    except D1TrainerMailboxWorkerError as error:
        _emit(stderr, _error(f"mailbox_{error.stage}", error.error_type))
        return 21
    except D1R2TrainerMailboxWorkerError as error:
        _emit(stderr, _error(f"mailbox_{error.stage}", error.error_type))
        return 21
    except D1R2RuntimeCompositionError as error:
        _emit(stderr, _error(f"r2_{error.code}", type(error).__name__))
        return 22
    except Exception as error:
        _emit(stderr, _error("unexpected_defect", type(error).__name__))
        return 70


def main(argv: Sequence[str] | None = None) -> int:
    return run_d1_trainer_mailbox_command(
        sys.argv[1:] if argv is None else argv,
        environ=os.environ,
    )


if __name__ == "__main__":
    raise SystemExit(main())
