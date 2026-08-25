"""Hermetic contract tests for the hardened D1 external review adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import entropic_substrate.analysis.d1_external_review_adapter as review_adapter_module
from entropic_substrate.analysis.d1_external_review_adapter import (
    D1BoundedSubprocessRunner,
    D1ExternalReviewError,
    D1ExternalReviewSubprocessAdapter,
    D1ReviewSandboxIdentity,
)
from entropic_substrate.analysis.d1_review_execution import (
    D1CloudReviewRequest,
    D1InMemoryReviewReceiptStoreFake,
    execute_d1_blind_review,
)
from entropic_substrate.analysis.d1_review_packet import (
    D1BlindReviewPacket,
    D1FactorialReviewerPrompt,
)

pytestmark = pytest.mark.unit

_IMAGE = "sha256:" + "a" * 64
_PROFILE = "b" * 64
_ATTESTATION = "c" * 64
_RESPONSE = b'{"provider_response":"opaque"}'
_REPORT = b"Review outcome: NO_MAJOR_FINDINGS\n"


def _request(*, round_number: int = 1, lineage: tuple[str, ...] = ()) -> D1CloudReviewRequest:
    packet = D1BlindReviewPacket.from_manifest(
        {"schema_version": "d1-blind-review-packet-v1", "evidence": "secret-sentinel"}
    )
    prompt = D1FactorialReviewerPrompt(f"Packet SHA-256: {packet.packet_digest}\nsecret-sentinel\n")
    return D1CloudReviewRequest(
        packet=packet,
        prompt=prompt,
        provider="provider-secret-sentinel",
        model_id="model-v1",
        round=round_number,
        retry_lineage=lineage,
    )


@dataclass
class _Result:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass
class _Runner:
    returncode: int = 0
    stderr: bytes = b""
    overrides: dict[str, object] = field(default_factory=dict)
    calls: list[tuple[list[str], dict[str, object]]] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: object) -> _Result:
        self.calls.append((argv, kwargs))
        request = json.loads(bytes(kwargs["input"]))
        response = {
            "attempt_id": request["attempt_id"],
            "model_id": request["request"]["model_id"],
            "provider": request["request"]["provider"],
            "provider_response_base64": base64.b64encode(_RESPONSE).decode(),
            "provider_response_sha256": hashlib.sha256(_RESPONSE).hexdigest(),
            "report_base64": base64.b64encode(_REPORT).decode(),
            "report_sha256": hashlib.sha256(_REPORT).hexdigest(),
            "request_digest": request["request_digest"],
            "retry_lineage": request["request"]["retry_lineage"],
            "round": request["request"]["round"],
            "sandbox_attestation_sha256": _ATTESTATION,
            "sandbox_identity_digest": request["sandbox"]["identity_digest"],
            "schema_version": "d1-external-review-result-v1",
        }
        response.update(self.overrides)
        return _Result(
            self.returncode,
            json.dumps(response, separators=(",", ":"), sort_keys=True).encode(),
            self.stderr,
        )


class _NoncanonicalRunner(_Runner):
    def __call__(self, argv: list[str], **kwargs: object) -> _Result:
        result = super().__call__(argv, **kwargs)
        pretty = json.dumps(json.loads(result.stdout), indent=2, sort_keys=True).encode()
        return _Result(result.returncode, pretty, result.stderr)


class _DeepJsonRunner(_Runner):
    def __call__(self, argv: list[str], **kwargs: object) -> _Result:
        result = super().__call__(argv, **kwargs)
        nested = b"[" * 10_000 + b"0" + b"]" * 10_000
        return _Result(result.returncode, nested, result.stderr)


def _adapter(tmp_path: Path, runner: _Runner) -> D1ExternalReviewSubprocessAdapter:
    return D1ExternalReviewSubprocessAdapter(
        work_dir=tmp_path,
        sandbox=D1ReviewSandboxIdentity(
            image_digest=_IMAGE,
            profile_sha256=_PROFILE,
            network_policy="provider-egress-only",
        ),
        attempt_id="attempt-0001",
        runner=runner,
    )


def _process_running(pid: int) -> bool:
    result = subprocess.run(  # nosec B603 -- fixed diagnostic argv in a unit test
        ("ps", "-o", "stat=", "-p", str(pid)),
        capture_output=True,
        check=False,
        shell=False,
    )
    state = result.stdout.decode().strip()
    return result.returncode == 0 and bool(state) and not state.startswith("Z")


def test_adapter_uses_fixed_no_shell_argv_and_stdin_without_inherited_secrets(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    lineage = ("d" * 64,)
    request = _request(round_number=2, lineage=lineage)

    _adapter(tmp_path, runner).review(request)

    assert len(runner.calls) == 1
    argv, kwargs = runner.calls[0]
    assert argv == [
        "/usr/local/bin/d1-review-sandbox",
        "run",
        "--profile-sha256",
        _PROFILE,
        "--image-digest",
        _IMAGE,
        "--stdio-json-v1",
    ]
    assert kwargs["shell"] is False
    assert kwargs["check"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["close_fds"] is True
    assert kwargs["start_new_session"] is True
    assert kwargs["env"] == {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"}
    assert "secret-sentinel" not in " ".join(argv)
    assert "secret-sentinel" not in json.dumps(kwargs["env"])
    wire = json.loads(bytes(kwargs["input"]))
    assert wire["packet"] == request.packet.canonical_json
    assert wire["prompt"] == request.prompt.text
    assert wire["request"]["round"] == 2
    assert wire["request"]["retry_lineage"] == list(lineage)


def test_success_envelope_binds_attempt_exit_sandbox_and_existing_receipt(
    tmp_path: Path,
) -> None:
    runner = _Runner()
    request = _request()
    store = D1InMemoryReviewReceiptStoreFake()

    execution = execute_d1_blind_review(
        request=request,
        reviewer=_adapter(tmp_path, runner),
        receipt_store=store,
    )
    envelope = json.loads(execution.response)

    assert envelope["attempt"]["attempt_id"] == "attempt-0001"
    assert envelope["exit"]["returncode"] == 0
    assert envelope["sandbox"]["image_digest"] == _IMAGE
    assert envelope["sandbox_attestation_sha256"] == _ATTESTATION
    assert base64.b64decode(envelope["provider_response_base64"]) == _RESPONSE
    assert execution.report == _REPORT
    assert execution.receipt.manifest["round"] == 1
    assert (
        execution.receipt.manifest["response_sha256"]
        == hashlib.sha256(execution.response).hexdigest()
    )
    assert execution.receipt.manifest["report_sha256"] == hashlib.sha256(_REPORT).hexdigest()


def test_nonzero_exit_fails_loudly_with_attempt_and_exit_hashes(tmp_path: Path) -> None:
    runner = _Runner(returncode=42, stderr=b"simulated failure")

    with pytest.raises(D1ExternalReviewError) as captured:
        _adapter(tmp_path, runner).review(_request())

    assert captured.value.code == "subprocess_exit"
    assert captured.value.context["attempt_id"] == "attempt-0001"
    assert captured.value.context["returncode"] == 42
    assert (
        captured.value.context["stderr_sha256"] == hashlib.sha256(b"simulated failure").hexdigest()
    )
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    ("runner", "code"),
    [
        (_Runner(overrides={"request_digest": "e" * 64}), "protocol_identity"),
        (_Runner(overrides={"report_sha256": "f" * 64}), "payload_digest"),
        (_Runner(stderr=b"unexpected log"), "stderr_output"),
        (_NoncanonicalRunner(), "protocol_encoding"),
        (
            _Runner(
                overrides={
                    "provider_response_base64": "TR==",
                    "provider_response_sha256": hashlib.sha256(b"M").hexdigest(),
                }
            ),
            "payload_encoding",
        ),
    ],
)
def test_protocol_identity_digest_and_clean_stderr_are_mandatory(
    tmp_path: Path, runner: _Runner, code: str
) -> None:
    with pytest.raises(D1ExternalReviewError) as captured:
        _adapter(tmp_path, runner).review(_request())

    assert captured.value.code == code


@pytest.mark.parametrize(
    ("result", "detail"),
    [
        (_Result(0, "not-bytes", b""), "stdout"),  # type: ignore[arg-type]
        (_Result(0, b"{}", "not-bytes"), "stderr"),  # type: ignore[arg-type]
        (_Result(True, b"{}", b""), "returncode"),  # type: ignore[arg-type]
    ],
)
def test_process_result_types_are_rejected_before_any_hashing(
    tmp_path: Path, result: _Result, detail: str
) -> None:
    class _TypedDriftRunner:
        def __call__(self, _argv: list[str], **_kwargs: object) -> _Result:
            return result

    with pytest.raises(D1ExternalReviewError) as captured:
        _adapter(tmp_path, _TypedDriftRunner()).review(_request())  # type: ignore[arg-type]

    assert captured.value.code == "subprocess_result"
    assert detail in str(captured.value)


def test_sandbox_identity_requires_immutable_image_and_named_policy() -> None:
    with pytest.raises(ValueError, match="immutable"):
        D1ReviewSandboxIdentity("review:latest", _PROFILE, "provider-egress-only")
    with pytest.raises(ValueError, match="network_policy"):
        D1ReviewSandboxIdentity(_IMAGE, _PROFILE, "")


def test_bounded_runner_streams_valid_output_and_enforces_limit(tmp_path: Path) -> None:
    runner = D1BoundedSubprocessRunner(
        attempt_id="attempt-stream",
        timeout_seconds=2.0,
        max_output_bytes=128,
    )
    valid = runner.run(
        (sys.executable, "-c", "import sys; data=sys.stdin.buffer.read(); print(len(data))"),
        cwd=tmp_path,
        input=b"bounded",
        env={"PATH": "/usr/bin:/bin"},
    )
    assert valid.returncode == 0
    assert valid.stdout == b"7\n"
    assert valid.stderr == b""

    with pytest.raises(D1ExternalReviewError) as raised:
        runner.run(
            (sys.executable, "-c", "import os,time; os.write(1,b'x'*4096); time.sleep(60)"),
            cwd=tmp_path,
            input=b"",
            env={"PATH": "/usr/bin:/bin"},
        )
    assert raised.value.code == "output_limit"


def test_output_limit_kills_and_reaps_the_complete_process_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    script = (
        "import os,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "open(sys.argv[1],'w').write(str(child.pid)); os.write(1,b'x'*4096); time.sleep(60)"
    )
    runner = D1BoundedSubprocessRunner(
        attempt_id="attempt-reap",
        timeout_seconds=30.0,
        max_output_bytes=1024,
    )
    with pytest.raises(D1ExternalReviewError) as raised:
        runner.run(
            (sys.executable, "-c", script, str(pid_path)),
            cwd=tmp_path,
            input=b"",
            env={"PATH": "/usr/bin:/bin"},
        )
    assert raised.value.code == "output_limit"
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _process_running(child_pid):
            break
        time.sleep(0.01)
    else:
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass
        pytest.fail("bounded review child remained alive")


def test_timeout_code_covers_overrun_and_closed_pipes(tmp_path: Path) -> None:
    runner = D1BoundedSubprocessRunner(
        attempt_id="attempt-timeout",
        timeout_seconds=0.2,
        max_output_bytes=1024,
    )
    for script in (
        "import time; time.sleep(60)",
        "import os,time; os.close(1); os.close(2); time.sleep(60)",
    ):
        with pytest.raises(D1ExternalReviewError) as raised:
            runner.run(
                (sys.executable, "-c", script),
                cwd=tmp_path,
                input=b"",
                env={"PATH": "/usr/bin:/bin"},
            )
        assert raised.value.code == "timeout"


def test_json_recursion_failure_is_normalized_to_protocol_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def recurse(*_args: object, **_kwargs: object) -> object:
        raise RecursionError("synthetic recursion")

    monkeypatch.setattr(
        review_adapter_module,
        "json",
        SimpleNamespace(
            JSONDecodeError=json.JSONDecodeError,
            dumps=json.dumps,
            loads=recurse,
        ),
    )
    with pytest.raises(D1ExternalReviewError) as raised:
        _adapter(tmp_path, _Runner()).review(_request())
    assert raised.value.code == "protocol_encoding"

    monkeypatch.undo()
    with pytest.raises(D1ExternalReviewError) as real_payload:
        _adapter(tmp_path, _DeepJsonRunner()).review(_request())
    assert real_payload.value.code == "protocol_encoding"


@pytest.mark.parametrize(
    "overrides",
    [
        {"attempt_id": ""},
        {"timeout_seconds": 0.0},
        {"max_output_bytes": 0},
    ],
)
def test_bounded_runner_constructor_guards(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "attempt_id": "attempt-guards",
        "timeout_seconds": 1.0,
        "max_output_bytes": 1024,
        **overrides,
    }
    with pytest.raises(ValueError):
        D1BoundedSubprocessRunner(**cast(Any, values))


@pytest.mark.parametrize(
    ("argv", "cwd", "input_payload", "environment"),
    [
        ((), Path("."), b"", {}),
        ((sys.executable,), "not-path", b"", {}),
        ((sys.executable,), Path("."), "not-bytes", {}),
        ((sys.executable,), Path("."), b"", []),
    ],
)
def test_bounded_runner_type_guards(
    argv: object,
    cwd: object,
    input_payload: object,
    environment: object,
) -> None:
    runner = D1BoundedSubprocessRunner(attempt_id="attempt-types")
    with pytest.raises(TypeError):
        runner.run(
            cast(tuple[str, ...], argv),
            cwd=cwd,
            input=input_payload,
            env=environment,
        )


def test_bounded_runner_rejects_altered_call_controls(tmp_path: Path) -> None:
    runner = D1BoundedSubprocessRunner(attempt_id="attempt-controls", timeout_seconds=1.0)
    common = {
        "cwd": tmp_path,
        "input": b"payload",
        "capture_output": True,
        "check": False,
        "shell": False,
        "timeout": 1.0,
        "env": {"MARKER": "preserved", "PATH": "/usr/bin:/bin"},
        "close_fds": True,
        "start_new_session": True,
    }
    script = (
        "import os,sys; print(os.getcwd()); print(sys.stdin.read()); print(os.environ['MARKER'])"
    )
    accepted = runner([sys.executable, "-c", script], **common)
    assert accepted.returncode == 0
    assert accepted.stdout == f"{tmp_path}\npayload\npreserved\n".encode()
    with pytest.raises(TypeError, match="flags"):
        runner([sys.executable], **{**common, "shell": True})
    with pytest.raises(TypeError, match="timeout"):
        runner([sys.executable], **{**common, "timeout": 2.0})
    with pytest.raises(TypeError, match="keyword"):
        runner([sys.executable], **{**common, "unexpected": True})


def test_clean_leader_exit_sweeps_pipe_closed_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "detached-child.pid"
    script = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "open(sys.argv[1],'w').write(str(child.pid))"
    )
    runner = D1BoundedSubprocessRunner(
        attempt_id="attempt-clean-leader",
        timeout_seconds=2.0,
        max_output_bytes=1024,
    )
    result = runner.run(
        (sys.executable, "-c", script, str(pid_path)),
        cwd=tmp_path,
        input=b"",
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _process_running(child_pid):
            break
        time.sleep(0.01)
    else:
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass
        pytest.fail("clean review leader left a descendant alive")


def test_review_adapter_consumes_attempt_before_success_or_failure(tmp_path: Path) -> None:
    successful_runner = _Runner()
    successful = _adapter(tmp_path, successful_runner)
    successful.review(_request())
    with pytest.raises(D1ExternalReviewError) as consumed:
        successful.review(_request())
    assert consumed.value.code == "attempt_consumed"
    assert len(successful_runner.calls) == 1

    failed_runner = _Runner(returncode=9)
    failed = _adapter(tmp_path, failed_runner)
    with pytest.raises(D1ExternalReviewError):
        failed.review(_request())
    with pytest.raises(D1ExternalReviewError) as consumed_after_failure:
        failed.review(_request())
    assert consumed_after_failure.value.code == "attempt_consumed"
    assert len(failed_runner.calls) == 1


def test_review_adapter_defaults_to_bounded_streaming_runner(tmp_path: Path) -> None:
    adapter = D1ExternalReviewSubprocessAdapter(
        work_dir=tmp_path,
        sandbox=D1ReviewSandboxIdentity(
            image_digest=_IMAGE,
            profile_sha256=_PROFILE,
            network_policy="provider-egress-only",
        ),
        attempt_id="attempt-default-runner",
    )
    assert isinstance(adapter.__dict__["_runner"], D1BoundedSubprocessRunner)
