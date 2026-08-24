"""Hardened one-attempt subprocess adapter for D1 blind review."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess  # nosec B404 -- fixed sandbox argv; shell is explicitly disabled
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast

from entropic_substrate.analysis.d1_review_execution import (
    D1CloudReviewRequest,
    D1CloudReviewResponse,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest

_RUNTIME = "/usr/local/bin/d1-review-sandbox"
_TIMEOUT_SECONDS = 600
_MAX_OUTPUT_BYTES = 16 * 1024 * 1024
_SAFE_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RESULT_FIELDS = frozenset(
    (
        "attempt_id model_id provider provider_response_base64 provider_response_sha256 "
        "report_base64 report_sha256 request_digest retry_lineage round "
        "sandbox_attestation_sha256 sandbox_identity_digest schema_version"
    ).split()
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class D1ReviewSandboxIdentity:
    """Immutable requested image and externally enforced sandbox profile."""

    image_digest: str
    profile_sha256: str
    network_policy: str

    def __post_init__(self) -> None:
        if not self.image_digest.startswith("sha256:"):
            raise ValueError("D1 review image must use an immutable sha256 digest")
        _digest(self.image_digest.removeprefix("sha256:"), "D1 review image")
        _digest(self.profile_sha256, "D1 review sandbox profile")
        if self.network_policy != "provider-egress-only":
            raise ValueError("D1 review sandbox requires provider-egress-only network_policy")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "capabilities": "drop-all",
            "filesystem": "read-only",
            "image_digest": self.image_digest,
            "network_policy": self.network_policy,
            "no_new_privileges": True,
            "profile_sha256": self.profile_sha256,
            "runtime": _RUNTIME,
            "secret_delivery": "sandbox-managed-secret-mount",  # pragma: allowlist secret
        }

    @property
    def identity_digest(self) -> str:
        return canonical_digest(self.manifest)


class D1ExternalReviewError(RuntimeError):
    """Stable fail-loud error with hashes, never raw possibly-secret output."""

    def __init__(self, code: str, detail: str, **context: str | int) -> None:
        self.code = code
        self.context = MappingProxyType(dict(context))
        super().__init__(f"D1 external review {code}: {detail}; context={dict(context)}")


def _exception_hash(value: object, label: str, attempt_id: str) -> str:
    if value is None:
        return _sha(b"")
    if type(value) is not bytes:
        raise D1ExternalReviewError(
            "subprocess_failure",
            f"exception {label} must be bytes",
            attempt_id=attempt_id,
            output_type=type(value).__name__,
        )
    return _sha(value)


class _ProcessResult(Protocol):
    returncode: int
    stdout: bytes
    stderr: bytes


type _Runner = Callable[..., _ProcessResult]


def _work_dir(value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.abspath(value))
    if path.is_symlink() or not path.is_dir():
        raise ValueError("D1 external review work_dir must be an existing real directory")
    return path


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant {value}")


def _parse_result(payload: bytes) -> Mapping[str, object]:
    if not payload or len(payload) > _MAX_OUTPUT_BYTES:
        raise ValueError("worker stdout must be nonempty and within the output limit")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("worker stdout must be strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError("worker result must have an object root")
    result = cast(dict[str, object], value)
    if frozenset(result) != _RESULT_FIELDS:
        raise ValueError("worker result fields must match the protocol exactly")
    if result["schema_version"] != "d1-external-review-result-v1":
        raise ValueError("worker result schema is unsupported")
    if _canonical(result) != payload:
        raise ValueError("worker stdout must use exact canonical JSON")
    return result


def _decode(value: object, label: str) -> bytes:
    if type(value) is not str:
        raise ValueError(f"{label} must be base64 text")
    try:
        payload = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError(f"{label} must be canonical base64") from error
    if base64.b64encode(payload).decode("ascii") != value:
        raise ValueError(f"{label} must round-trip as canonical base64")
    if not payload or len(payload) > _MAX_OUTPUT_BYTES:
        raise ValueError(f"{label} must be nonempty and within the output limit")
    return payload


class D1ExternalReviewSubprocessAdapter:
    """Execute exactly one review attempt through an immutable sandbox wrapper."""

    def __init__(
        self,
        *,
        work_dir: str | os.PathLike[str],
        sandbox: D1ReviewSandboxIdentity,
        attempt_id: str,
        runner: _Runner | None = None,
    ) -> None:
        self._work_dir = _work_dir(work_dir)
        if type(sandbox) is not D1ReviewSandboxIdentity:
            raise TypeError("D1 external review requires D1ReviewSandboxIdentity")
        if type(attempt_id) is not str or _SAFE_ATTEMPT_ID.fullmatch(attempt_id) is None:
            raise ValueError("D1 external review attempt_id must be a safe opaque identifier")
        self._sandbox = sandbox
        self._attempt_id = attempt_id
        self._runner = runner if runner is not None else cast(_Runner, subprocess.run)

    @property
    def argv(self) -> list[str]:
        return [
            _RUNTIME,
            "run",
            "--profile-sha256",
            self._sandbox.profile_sha256,
            "--image-digest",
            self._sandbox.image_digest,
            "--stdio-json-v1",
        ]

    def _wire(self, request: D1CloudReviewRequest) -> bytes:
        return _canonical(
            {
                "attempt_id": self._attempt_id,
                "packet": request.packet.canonical_json,
                "prompt": request.prompt.text,
                "request": request.manifest,
                "request_digest": request.request_digest,
                "sandbox": {
                    **self._sandbox.manifest,
                    "identity_digest": self._sandbox.identity_digest,
                },
                "schema_version": "d1-external-review-request-v1",
            }
        )

    def _invoke(self, request: D1CloudReviewRequest) -> _ProcessResult:
        try:
            return self._runner(  # nosec B603 -- fixed argv and shell=False are contract-tested
                self.argv,
                cwd=self._work_dir,
                input=self._wire(request),
                capture_output=True,
                check=False,
                shell=False,
                timeout=_TIMEOUT_SECONDS,
                env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
                close_fds=True,
                start_new_session=True,
            )
        except Exception as error:
            raise D1ExternalReviewError(
                "subprocess_failure",
                type(error).__name__,
                attempt_id=self._attempt_id,
                stdout_sha256=_exception_hash(
                    getattr(error, "stdout", None), "stdout", self._attempt_id
                ),
                stderr_sha256=_exception_hash(
                    getattr(error, "stderr", None), "stderr", self._attempt_id
                ),
            ) from error

    def review(self, request: D1CloudReviewRequest) -> D1CloudReviewResponse:
        if type(request) is not D1CloudReviewRequest:
            raise TypeError("D1 external review adapter requires D1CloudReviewRequest")
        result = self._invoke(request)
        if type(result.returncode) is not int:
            raise D1ExternalReviewError(
                "subprocess_result", "returncode must be int", attempt_id=self._attempt_id
            )
        if type(result.stdout) is not bytes:
            raise D1ExternalReviewError(
                "subprocess_result", "stdout must be bytes", attempt_id=self._attempt_id
            )
        if type(result.stderr) is not bytes:
            raise D1ExternalReviewError(
                "subprocess_result", "stderr must be bytes", attempt_id=self._attempt_id
            )
        exit_context = {
            "attempt_id": self._attempt_id,
            "returncode": result.returncode,
            "stdout_sha256": _sha(result.stdout),
            "stderr_sha256": _sha(result.stderr),
        }
        if result.returncode != 0:
            raise D1ExternalReviewError("subprocess_exit", "worker exited nonzero", **exit_context)
        if result.stderr:
            raise D1ExternalReviewError(
                "stderr_output", "successful worker emitted stderr", **exit_context
            )
        try:
            wire = _parse_result(result.stdout)
            provider_response = _decode(wire["provider_response_base64"], "provider response")
            report = _decode(wire["report_base64"], "review report")
            self._validate_wire(wire, request, provider_response, report)
        except (TypeError, ValueError) as error:
            detail = str(error)
            if "base64" in detail:
                code = "payload_encoding"
            elif "JSON" in detail or "stdout" in detail:
                code = "protocol_encoding"
            elif "digest" in detail:
                code = "payload_digest"
            else:
                code = "protocol_identity"
            raise D1ExternalReviewError(code, str(error), **exit_context) from error
        envelope = _canonical(
            {
                "attempt": {
                    "argv_sha256": _sha(_canonical(self.argv)),
                    "attempt_id": self._attempt_id,
                    "request_digest": request.request_digest,
                },
                "bindings": request.manifest,
                "exit": exit_context,
                "provider_response_base64": base64.b64encode(provider_response).decode("ascii"),
                "provider_response_sha256": _sha(provider_response),
                "sandbox": {
                    **self._sandbox.manifest,
                    "identity_digest": self._sandbox.identity_digest,
                },
                "sandbox_attestation_sha256": wire["sandbox_attestation_sha256"],
                "schema_version": "d1-external-review-response-evidence-v1",
            }
        )
        return D1CloudReviewResponse(
            request.request_digest,
            request.provider,
            request.model_id,
            envelope,
            report,
        )

    def _validate_wire(
        self,
        wire: Mapping[str, object],
        request: D1CloudReviewRequest,
        provider_response: bytes,
        report: bytes,
    ) -> None:
        expected = {
            "attempt_id": self._attempt_id,
            "model_id": request.model_id,
            "provider": request.provider,
            "request_digest": request.request_digest,
            "retry_lineage": list(request.retry_lineage),
            "round": request.round,
            "sandbox_identity_digest": self._sandbox.identity_digest,
        }
        if any(wire[field] != value for field, value in expected.items()):
            raise ValueError("worker protocol identity differs from request")
        response_digest = cast(str, wire["provider_response_sha256"])
        report_digest = cast(str, wire["report_sha256"])
        attestation = cast(str, wire["sandbox_attestation_sha256"])
        for value, label in (
            (response_digest, "provider response digest"),
            (report_digest, "report digest"),
            (attestation, "sandbox attestation digest"),
        ):
            _digest(value, label)
        if response_digest != _sha(provider_response) or report_digest != _sha(report):
            raise ValueError("worker payload digest differs from decoded bytes")
