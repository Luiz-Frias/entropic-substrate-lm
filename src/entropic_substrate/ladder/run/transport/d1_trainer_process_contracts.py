"""Raw transport and canonical process-report contracts for D1 trainers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, Protocol, cast

from entropic_substrate.kernel.artifacts import normalize_d1_relative_artifact_path
from entropic_substrate.kernel.identity.d1_primitives import require_sha256
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1RuntimeObservation

__all__ = [
    "D1TrainerAdapterPhase",
    "D1TrainerArtifactReadback",
    "D1TrainerProcessTransport",
    "D1TrainerStartReport",
    "d1_trainer_request_path",
    "d1_trainer_runtime_report_path",
    "d1_trainer_start_report_path",
    "parse_d1_runtime_report",
    "parse_d1_start_report",
    "serialize_d1_runtime_report",
    "serialize_d1_start_report",
]

_RUNTIME_SCHEMA = "d1-trainer-runtime-report-v1"
_START_SCHEMA = "d1-trainer-start-report-v1"
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_RUNTIME_KEYS = frozenset(
    "attempt_sha256 compiler_enabled compiler_version cuda_version driver_version pod_id "
    "pytorch_version schema_version trainer_monotonic_origin_ns trainer_started_at".split()
)
_START_KEYS = frozenset(
    "attempt_sha256 command_sha256 pod_id process_id request_sha256 runtime_sha256 "
    "schema_version trainer_started_at".split()
)


@dataclass(frozen=True, slots=True)
class D1TrainerArtifactReadback:
    path: str
    payload: bytes

    def __post_init__(self) -> None:
        if type(self.path) is not str or type(self.payload) is not bytes:
            raise TypeError("D1 trainer readback requires raw path and bytes")


@dataclass(frozen=True, slots=True)
class D1TrainerStartReport:
    """Freely constructible raw DTO; only parser admission may feed a domain receipt."""

    pod_id: str
    attempt_sha256: str
    request_sha256: str
    runtime_sha256: str
    trainer_started_at: datetime
    process_id: str
    command_sha256: str


class D1TrainerAdapterPhase(StrEnum):
    BOOTSTRAP_ATTEMPTED = "bootstrap_attempted"
    RUNTIME_READBACK_ATTEMPTED = "runtime_readback_attempted"
    RUNTIME_ATTESTED = "runtime_attested"
    STAGE_WRITE_ATTEMPTED = "stage_write_attempted"
    STAGE_READBACK_ATTEMPTED = "stage_readback_attempted"
    REQUEST_STAGED = "request_staged"
    START_ATTEMPTED = "start_attempted"
    START_READBACK_ATTEMPTED = "start_readback_attempted"
    TRAINER_STARTED = "trainer_started"
    EXPORT_ATTEMPTED = "export_attempted"
    EXPORT_VERIFIED = "export_verified"


def d1_trainer_request_path(attempt_sha256: str) -> str:
    """Derive the sole remote request target from the bootstrap attempt identity."""
    require_sha256(attempt_sha256, label="D1 trainer request attempt")
    return normalize_d1_relative_artifact_path(
        f"requests/{attempt_sha256}.json",
        label="D1 trainer request",
    )


def d1_trainer_runtime_report_path(attempt_sha256: str) -> str:
    """Derive the immutable runtime-report path for one bootstrap attempt."""
    require_sha256(attempt_sha256, label="D1 trainer runtime-report attempt")
    return normalize_d1_relative_artifact_path(
        f"trainer-runtime-{attempt_sha256}.json",
        label="D1 trainer runtime report",
    )


def d1_trainer_start_report_path(attempt_sha256: str, request_sha256: str) -> str:
    """Derive the immutable start-report path for one staged request attempt."""
    require_sha256(attempt_sha256, label="D1 trainer start-report attempt")
    require_sha256(request_sha256, label="D1 trainer start-report request")
    return normalize_d1_relative_artifact_path(
        f"trainer-start-{attempt_sha256}-{request_sha256}.json",
        label="D1 trainer start report",
    )


class D1TrainerProcessTransport(Protocol):
    def bootstrap(self, pod_id: str, attempt_sha256: str) -> None: ...
    def read_runtime_report(self, pod_id: str, attempt_sha256: str) -> bytes: ...
    def write_request(
        self, pod_id: str, attempt_sha256: str, path: str, payload: bytes
    ) -> None: ...
    def read_request(
        self, pod_id: str, attempt_sha256: str, path: str
    ) -> D1TrainerArtifactReadback: ...
    def start_trainer(self, pod_id: str, attempt_sha256: str, request_sha256: str) -> None: ...
    def read_start_report(self, pod_id: str, attempt_sha256: str, request_sha256: str) -> bytes: ...
    def await_export(self, pod_id: str, attempt_sha256: str, prefix: str) -> Path: ...


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise ValueError(message)
    raise ValueError(message) from error


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail("D1 trainer report repeats a key")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    del value
    _fail("D1 trainer report contains a nonfinite constant")


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _decode(payload: bytes, *, keys: frozenset[str], schema: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload:
        _fail("D1 trainer report must be nonempty bytes")
    value: object | None = None
    decode_failed = False
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        decode_failed = True
    except ValueError:
        raise
    if decode_failed:
        _fail("D1 trainer report must be UTF-8 JSON")
    if type(value) is not dict or frozenset(value) != keys:
        _fail("D1 trainer report keys must be exact")
    root = cast(dict[str, object], value)
    if root["schema_version"] != schema:
        _fail(f"D1 trainer report schema must be {schema}")
    if _canonical(root) != payload:
        _fail("D1 trainer report must be canonical JSON")
    return root


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip() or value.strip() != value:
        _fail(f"{label} must be a nonempty trimmed string")
    return value


def _sha(value: object, label: str) -> str:
    text = _text(value, label)
    try:
        require_sha256(text, label=label)
    except ValueError as error:
        _fail(str(error), error)
    if text.lower() != text:
        _fail(f"{label} must use lowercase hexadecimal")
    return text


def _timestamp(value: object, label: str) -> datetime:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        _fail(f"{label} must be canonical UTC Z time")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        _fail(f"{label} is invalid", error)
    if parsed.isoformat().replace("+00:00", "Z") != value:
        _fail(f"{label} must use one canonical UTC spelling")
    return parsed


def _utc_datetime(value: object, label: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        _fail(f"{label} must be an exact builtin UTC datetime")
    return value


def parse_d1_runtime_report(
    payload: bytes, *, pod_id: str, attempt_sha256: str
) -> D1RuntimeObservation:
    pod, attempt = _text(pod_id, "runtime pod_id"), _sha(attempt_sha256, "runtime attempt")
    root = _decode(payload, keys=_RUNTIME_KEYS, schema=_RUNTIME_SCHEMA)
    if root["pod_id"] != pod or root["attempt_sha256"] != attempt:
        _fail("D1 runtime report differs from bootstrap authority")
    enabled, monotonic = root["compiler_enabled"], root["trainer_monotonic_origin_ns"]
    if type(enabled) is not bool or type(monotonic) is not int or monotonic < 0:
        _fail("D1 runtime report compiler or monotonic type is invalid")
    return D1RuntimeObservation(
        pod,
        _timestamp(root["trainer_started_at"], "runtime trainer_started_at"),
        _text(root["cuda_version"], "runtime cuda_version"),
        _text(root["driver_version"], "runtime driver_version"),
        _text(root["pytorch_version"], "runtime pytorch_version"),
        enabled,
        _text(root["compiler_version"], "runtime compiler_version"),
        monotonic,
    )


def parse_d1_start_report(
    payload: bytes,
    *,
    pod_id: str,
    attempt_sha256: str,
    request_sha256: str,
    runtime_sha256: str,
    trainer_started_at: datetime,
    command_sha256: str,
) -> D1TrainerStartReport:
    pod = _text(pod_id, "expected start pod_id")
    attempt = _sha(attempt_sha256, "expected start attempt")
    request = _sha(request_sha256, "expected start request")
    runtime = _sha(runtime_sha256, "expected start runtime")
    started_at = _utc_datetime(trainer_started_at, "expected start timestamp")
    command = _sha(command_sha256, "expected start command")
    root = _decode(payload, keys=_START_KEYS, schema=_START_SCHEMA)
    observed_pod = _text(root["pod_id"], "observed start pod_id")
    observed_attempt = _sha(root["attempt_sha256"], "observed start attempt")
    observed_request = _sha(root["request_sha256"], "observed start request")
    observed_runtime = _sha(root["runtime_sha256"], "observed start runtime")
    observed_started_at = _timestamp(root["trainer_started_at"], "observed start timestamp")
    observed_command = _sha(root["command_sha256"], "observed start command")
    process_id = _text(root["process_id"], "observed start process_id")
    observed = (
        observed_pod,
        observed_attempt,
        observed_request,
        observed_runtime,
        observed_started_at,
        observed_command,
    )
    expected = (pod, attempt, request, runtime, started_at, command)
    if observed != expected:
        _fail("D1 start report differs from staged bootstrap authority")
    return D1TrainerStartReport(
        observed_pod,
        observed_attempt,
        observed_request,
        observed_runtime,
        observed_started_at,
        process_id,
        observed_command,
    )


def serialize_d1_runtime_report(
    runtime: D1RuntimeObservation,
    *,
    attempt_sha256: str,
) -> bytes:
    """Serialize one measured runtime through the canonical parser contract."""
    if type(runtime) is not D1RuntimeObservation:
        raise TypeError("D1 runtime report serializer requires D1RuntimeObservation")
    attempt = _sha(attempt_sha256, "runtime attempt")
    payload = _canonical(
        {
            "attempt_sha256": attempt,
            "compiler_enabled": runtime.compiler_enabled,
            "compiler_version": runtime.compiler_version,
            "cuda_version": runtime.cuda_version,
            "driver_version": runtime.driver_version,
            "pod_id": runtime.pod_id,
            "pytorch_version": runtime.pytorch_version,
            "schema_version": _RUNTIME_SCHEMA,
            "trainer_monotonic_origin_ns": runtime.trainer_monotonic_origin_ns,
            "trainer_started_at": runtime.trainer_started_at.isoformat().replace("+00:00", "Z"),
        }
    )
    if (
        parse_d1_runtime_report(
            payload,
            pod_id=runtime.pod_id,
            attempt_sha256=attempt,
        )
        != runtime
    ):
        raise ValueError("serialized D1 runtime report does not round-trip")
    return payload


def serialize_d1_start_report(report: D1TrainerStartReport) -> bytes:
    """Serialize one raw start DTO only when parser admission reproduces it."""
    if type(report) is not D1TrainerStartReport:
        raise TypeError("D1 start report serializer requires D1TrainerStartReport")
    pod = _text(report.pod_id, "serialized start pod_id")
    attempt = _sha(report.attempt_sha256, "serialized start attempt")
    request = _sha(report.request_sha256, "serialized start request")
    runtime = _sha(report.runtime_sha256, "serialized start runtime")
    started_at = _utc_datetime(report.trainer_started_at, "serialized start timestamp")
    process_id = _text(report.process_id, "serialized start process_id")
    command = _sha(report.command_sha256, "serialized start command")
    payload = _canonical(
        {
            "attempt_sha256": attempt,
            "command_sha256": command,
            "pod_id": pod,
            "process_id": process_id,
            "request_sha256": request,
            "runtime_sha256": runtime,
            "schema_version": _START_SCHEMA,
            "trainer_started_at": started_at.isoformat().replace("+00:00", "Z"),
        }
    )
    if (
        parse_d1_start_report(
            payload,
            pod_id=pod,
            attempt_sha256=attempt,
            request_sha256=request,
            runtime_sha256=runtime,
            trainer_started_at=started_at,
            command_sha256=command,
        )
        != report
    ):
        raise ValueError("serialized D1 start report does not round-trip")
    return payload
