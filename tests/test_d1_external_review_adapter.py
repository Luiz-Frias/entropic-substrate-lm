"""Hermetic contract tests for the hardened D1 external review adapter."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.analysis.d1_external_review_adapter import (
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
