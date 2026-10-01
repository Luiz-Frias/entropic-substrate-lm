"""The RunPod mailbox entrypoint is strict, foreground, and non-secret."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import entropic_substrate.composition.entrypoints.d1_trainer_mailbox_command as command_module
from entropic_substrate.composition.entrypoints.d1_trainer_mailbox_command import (
    run_d1_trainer_mailbox_command,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
    D1TrainerMailboxWorkerError,
)

pytestmark = pytest.mark.unit

_POD = "pod-command-123"
_ACCESS_VALUE = "access-value"
_PRIVATE_VALUE = "private-value"


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
        "--repo-root",
        str(root),
        "--max-mailbox-objects",
        "128",
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


def test_default_command_factory_builds_r2_worker_from_repo_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls: list[dict[str, object]] = []
    worker_calls: list[dict[str, object]] = []
    mailbox = _Mailbox({"handoff_kind": "fresh_completion"})
    runtime = SimpleNamespace(
        mailbox=object(),
        bind_checkpoint_sync=lambda **_kwargs: object(),
        checkpoint_recovery=object(),
    )

    def build(**kwargs: object) -> object:
        build_calls.append(dict(kwargs))
        return runtime

    def r2_worker(**kwargs: object) -> _Mailbox:
        worker_calls.append(dict(kwargs))
        return mailbox

    monkeypatch.setattr(command_module, "build_d1_r2_runtime_composition", build)
    monkeypatch.setattr(command_module, "D1R2TrainerMailboxWorker", r2_worker)
    stdout, stderr = io.StringIO(), io.StringIO()
    environ = {
        "RUNPOD_POD_ID": _POD,
        "AWS_ACCESS_KEY_ID": _ACCESS_VALUE,
        "AWS_SECRET_ACCESS_KEY": _PRIVATE_VALUE,
    }

    code = run_d1_trainer_mailbox_command(
        _argv(tmp_path),
        environ=environ,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0 and stderr.getvalue() == "" and mailbox.calls == 1
    assert build_calls == [{"repo_root": tmp_path, "environ": environ, "max_objects": 128}]
    assert len(worker_calls) == 1
    assert worker_calls[0]["repo_root"] == tmp_path
    assert worker_calls[0]["pod_id"] == _POD
    assert worker_calls[0]["objects"] is runtime.mailbox
    assert worker_calls[0]["recovery"] is runtime.checkpoint_recovery
    assert callable(worker_calls[0]["worker_factory"])


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("--repo-root",),
        ("--unknown", "value"),
        ("--repo-root", "/tmp", "--repo-root", "/other"),
        (
            "--repo-root",
            "/tmp",
            "--max-mailbox-objects",
            "128",
            "--maximum-wait-seconds",
            "nan",
            "--poll-interval-seconds",
            "0.25",
        ),
        (
            "--repo-root",
            "/tmp",
            "--max-mailbox-objects",
            "0",
            "--maximum-wait-seconds",
            "300.0",
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


_CANONICAL_FIXTURES: tuple[dict[str, object], ...] = (
    {},
    {"b": 1, "a": [1, 2.5, None, True], "c": {"z": "x", "y": []}},
    {"unicode": "café ☃ \U0001f600", "nested": [[1], [2, [3]]]},
    {"n": -0.0, "big": 10**30, "small": 1e-9, "quote": 'a"b\\c\n'},
)


def _parent_canonical(value: dict[str, object]) -> bytes:
    """The parent `_canonical` expression, verbatim (oracle for the repoint)."""
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@pytest.mark.parametrize("payload", _CANONICAL_FIXTURES)
def test_emit_and_result_digest_equal_the_parent_canonical_bytes(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    stream = io.StringIO()
    command_module._emit(stream, payload)  # type: ignore[attr-defined]
    assert stream.getvalue().encode("utf-8") == _parent_canonical(payload) + b"\n"

    stdout = io.StringIO()
    code = run_d1_trainer_mailbox_command(
        _argv(tmp_path),
        environ={"RUNPOD_POD_ID": _POD},
        mailbox_factory=lambda root, pod, policy: _Mailbox(payload),
        stdout=stdout,
        stderr=io.StringIO(),
    )
    assert code == 0
    assert (
        json.loads(stdout.getvalue())["result_sha256"]
        == hashlib.sha256(_parent_canonical(payload)).hexdigest()
    )


def test_mailbox_command_uses_the_shared_canonical_json_not_a_local_dump() -> None:
    from entropic_substrate.kernel.identity import d1_primitives

    assert command_module.canonical_json is d1_primitives.canonical_json  # type: ignore[attr-defined]
    assert "json.dumps" not in Path(command_module.__file__).read_text(encoding="utf-8")
