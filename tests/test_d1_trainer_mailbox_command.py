"""The RunPod mailbox entrypoint is strict, foreground, and non-secret."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from entropic_substrate.d1_trainer_mailbox_command import (
    run_d1_trainer_mailbox_command,
)
from entropic_substrate.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
    D1TrainerMailboxWorkerError,
)

pytestmark = pytest.mark.unit

_POD = "pod-command-123"


@dataclass
class _Mailbox:
    result: dict[str, object] | Exception
    calls: int = 0

    def run(self) -> dict[str, object]:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _argv(root: Path) -> tuple[str, ...]:
    return (
        "--volume-root",
        str(root),
        "--maximum-wait-seconds",
        "300.0",
        "--poll-interval-seconds",
        "0.25",
    )


def test_command_builds_one_foreground_worker_from_runpod_environment(tmp_path: Path) -> None:
    result = {"handoff_kind": "fresh_completion", "successful_step": 2}
    mailbox = _Mailbox(result)
    observed: list[tuple[Path, str, D1TrainerMailboxWaitPolicy]] = []
    stdout, stderr = io.StringIO(), io.StringIO()

    code = run_d1_trainer_mailbox_command(
        _argv(tmp_path),
        environ={"RUNPOD_POD_ID": _POD},
        mailbox_factory=lambda root, pod, policy: observed.append((root, pod, policy)) or mailbox,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0 and stderr.getvalue() == "" and mailbox.calls == 1
    assert observed == [(tmp_path, _POD, D1TrainerMailboxWaitPolicy(300.0, 0.25))]
    assert json.loads(stdout.getvalue()) == {
        "result_sha256": hashlib.sha256(
            json.dumps(result, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "schema_version": "d1-trainer-mailbox-command-result-v1",
        "status": "success",
    }


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("--volume-root",),
        ("--unknown", "value"),
        ("--volume-root", "/tmp", "--volume-root", "/other"),
        (
            "--volume-root",
            "/tmp",
            "--maximum-wait-seconds",
            "nan",
            "--poll-interval-seconds",
            "0.25",
        ),
    ],
)
def test_command_refuses_invalid_argv_before_factory(argv: tuple[str, ...]) -> None:
    calls: list[object] = []
    stderr = io.StringIO()

    code = run_d1_trainer_mailbox_command(
        argv,
        environ={"RUNPOD_POD_ID": _POD},
        mailbox_factory=lambda *values: calls.append(values),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 20 and calls == []
    assert json.loads(stderr.getvalue()) == {
        "error": {"code": "command_refused", "type": "D1TrainerMailboxCommandRefusal"},
        "schema_version": "d1-trainer-mailbox-command-result-v1",
        "status": "error",
    }


def test_command_requires_safe_injected_pod_id_before_factory(tmp_path: Path) -> None:
    calls: list[object] = []
    for environ in ({}, {"RUNPOD_POD_ID": "../pod"}):
        stderr = io.StringIO()
        code = run_d1_trainer_mailbox_command(
            _argv(tmp_path),
            environ=environ,
            mailbox_factory=lambda *values: calls.append(values),
            stdout=io.StringIO(),
            stderr=stderr,
        )
        assert code == 20
        assert json.loads(stderr.getvalue())["error"]["code"] == "command_refused"
    assert calls == []


def test_command_preserves_typed_mailbox_failure_without_error_text(tmp_path: Path) -> None:
    private_detail = "provider-private-detail-should-not-escape"
    failure = D1TrainerMailboxWorkerError(
        stage="start",
        code=private_detail,
        error_type="ValueError",
    )
    stderr = io.StringIO()

    code = run_d1_trainer_mailbox_command(
        _argv(tmp_path),
        environ={"RUNPOD_POD_ID": _POD},
        mailbox_factory=lambda *_args: _Mailbox(failure),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 21 and private_detail not in stderr.getvalue()
    assert json.loads(stderr.getvalue()) == {
        "error": {"code": "mailbox_start", "type": "ValueError"},
        "schema_version": "d1-trainer-mailbox-command-result-v1",
        "status": "error",
    }


def test_command_sanitizes_unexpected_failure(tmp_path: Path) -> None:
    private_detail = "unexpected-private-detail"
    stderr = io.StringIO()

    code = run_d1_trainer_mailbox_command(
        _argv(tmp_path),
        environ={"RUNPOD_POD_ID": _POD},
        mailbox_factory=lambda *_args: _Mailbox(RuntimeError(private_detail)),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 70 and private_detail not in stderr.getvalue()
    assert json.loads(stderr.getvalue())["error"] == {
        "code": "unexpected_defect",
        "type": "RuntimeError",
    }
