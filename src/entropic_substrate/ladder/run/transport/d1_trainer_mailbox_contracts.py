"""Canonical command and object-key contracts for the D1 trainer mailbox."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn, cast

from entropic_substrate.kernel.artifacts import normalize_d1_relative_artifact_path
from entropic_substrate.kernel.identity.d1_primitives import require_sha256_hex_any_case
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id

__all__ = [
    "D1TrainerMailboxCommand",
    "D1TrainerMailboxCommandKind",
    "D1TrainerMailboxErrorReport",
    "D1TrainerMailboxObjectIdentity",
    "D1TrainerMailboxObjectRead",
    "d1_trainer_mailbox_attempt_prefix",
    "d1_trainer_mailbox_command_path",
    "d1_trainer_mailbox_error_path",
    "parse_d1_trainer_mailbox_command",
    "parse_d1_trainer_mailbox_error",
    "serialize_d1_trainer_mailbox_command",
    "serialize_d1_trainer_mailbox_error",
]

_SCHEMA = "d1-trainer-mailbox-command-v1"
_FIELDS = frozenset({"attempt_sha256", "kind", "pod_id", "request_sha256", "schema_version"})
_ERROR_SCHEMA = "d1-trainer-mailbox-error-v1"
_ERROR_FIELDS = frozenset({"attempt_sha256", "error_type", "pod_id", "schema_version", "stage"})
_STAGE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ERROR_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


def _digest(value: str, *, label: str) -> str:
    require_sha256_hex_any_case(value, label=label)
    if value != value.lower():
        raise ValueError(f"{label} must use lowercase hexadecimal")
    return value


class D1TrainerMailboxCommandKind(StrEnum):
    BOOTSTRAP = "bootstrap"
    START = "start"


@dataclass(frozen=True, slots=True)
class D1TrainerMailboxObjectIdentity:
    key: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        normalized = normalize_d1_relative_artifact_path(
            self.key,
            label="D1 trainer mailbox object key",
        )
        if normalized != self.key:
            raise ValueError("D1 trainer mailbox object key is noncanonical")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("D1 trainer mailbox object size must be positive")
        _digest(self.sha256, label="D1 trainer mailbox object")


@dataclass(frozen=True, slots=True)
class D1TrainerMailboxObjectRead:
    identity: D1TrainerMailboxObjectIdentity
    payload: bytes

    def __post_init__(self) -> None:
        if type(self.identity) is not D1TrainerMailboxObjectIdentity:
            raise TypeError("D1 trainer mailbox read requires a nominal object identity")
        if type(self.payload) is not bytes:
            raise TypeError("D1 trainer mailbox read payload must be bytes")
        if (
            len(self.payload) != self.identity.size_bytes
            or hashlib.sha256(self.payload).hexdigest() != self.identity.sha256
        ):
            raise ValueError("D1 trainer mailbox object bytes differ from their identity")


@dataclass(frozen=True, slots=True)
class D1TrainerMailboxErrorReport:
    pod_id: str
    attempt_sha256: str
    stage: str
    error_type: str
    schema_version: str = _ERROR_SCHEMA

    def __post_init__(self) -> None:
        require_d1_pod_id(self.pod_id)
        _digest(self.attempt_sha256, label="D1 trainer mailbox error attempt")
        if type(self.stage) is not str or _STAGE.fullmatch(self.stage) is None:
            raise ValueError("D1 trainer mailbox error stage is unsafe")
        if type(self.error_type) is not str or _ERROR_TYPE.fullmatch(self.error_type) is None:
            raise ValueError("D1 trainer mailbox error type is unsafe")
        if type(self.schema_version) is not str:
            raise TypeError("D1 trainer mailbox error schema_version must be text")
        if self.schema_version != _ERROR_SCHEMA:
            raise ValueError("D1 trainer mailbox error schema_version is unsupported")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "attempt_sha256": self.attempt_sha256,
            "error_type": self.error_type,
            "pod_id": self.pod_id,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class D1TrainerMailboxCommand:
    kind: D1TrainerMailboxCommandKind
    pod_id: str
    attempt_sha256: str
    request_sha256: str | None
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        if type(self.kind) is not D1TrainerMailboxCommandKind:
            raise TypeError("D1 trainer mailbox command kind must be nominal")
        require_d1_pod_id(self.pod_id)
        _digest(self.attempt_sha256, label="D1 trainer mailbox attempt")
        if type(self.schema_version) is not str:
            raise TypeError("D1 trainer mailbox command schema_version must be text")
        if self.schema_version != _SCHEMA:
            raise ValueError("D1 trainer mailbox command schema_version is unsupported")
        if self.kind is D1TrainerMailboxCommandKind.BOOTSTRAP:
            if self.request_sha256 is not None:
                raise ValueError("D1 trainer mailbox bootstrap forbids request identity")
        elif self.request_sha256 is None:
            raise ValueError("D1 trainer mailbox start requires request identity")
        else:
            _digest(self.request_sha256, label="D1 trainer mailbox request")

    @classmethod
    def bootstrap(cls, *, pod_id: str, attempt_sha256: str) -> D1TrainerMailboxCommand:
        return cls(D1TrainerMailboxCommandKind.BOOTSTRAP, pod_id, attempt_sha256, None)

    @classmethod
    def start(
        cls,
        *,
        pod_id: str,
        attempt_sha256: str,
        request_sha256: str,
    ) -> D1TrainerMailboxCommand:
        return cls(
            D1TrainerMailboxCommandKind.START,
            pod_id,
            attempt_sha256,
            request_sha256,
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "attempt_sha256": self.attempt_sha256,
            "kind": self.kind.value,
            "pod_id": self.pod_id,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
        }


def d1_trainer_mailbox_attempt_prefix(pod_id: str, attempt_sha256: str) -> str:
    pod = require_d1_pod_id(pod_id)
    _digest(attempt_sha256, label="D1 trainer mailbox attempt")
    return normalize_d1_relative_artifact_path(
        f"d1-mailbox/v1/{pod}/{attempt_sha256}",
        label="D1 trainer mailbox attempt prefix",
    )


def d1_trainer_mailbox_command_path(command: D1TrainerMailboxCommand) -> str:
    if type(command) is not D1TrainerMailboxCommand:
        raise TypeError("D1 trainer mailbox path requires a nominal command")
    prefix = d1_trainer_mailbox_attempt_prefix(command.pod_id, command.attempt_sha256)
    name = (
        "bootstrap.json"
        if command.kind is D1TrainerMailboxCommandKind.BOOTSTRAP
        else f"start-{command.request_sha256}.json"
    )
    return normalize_d1_relative_artifact_path(
        f"{prefix}/commands/{name}",
        label="D1 trainer mailbox command",
    )


def d1_trainer_mailbox_error_path(pod_id: str, attempt_sha256: str) -> str:
    prefix = d1_trainer_mailbox_attempt_prefix(pod_id, attempt_sha256)
    return normalize_d1_relative_artifact_path(
        f"{prefix}/trainer-error.json",
        label="D1 trainer mailbox error",
    )


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def serialize_d1_trainer_mailbox_command(command: D1TrainerMailboxCommand) -> bytes:
    if type(command) is not D1TrainerMailboxCommand:
        raise TypeError("D1 trainer mailbox serializer requires a nominal command")
    return _canonical(command.manifest)


def serialize_d1_trainer_mailbox_error(report: D1TrainerMailboxErrorReport) -> bytes:
    if type(report) is not D1TrainerMailboxErrorReport:
        raise TypeError("D1 trainer mailbox error serializer requires a nominal report")
    return _canonical(report.manifest)


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise ValueError(message)
    raise ValueError(message) from error


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail(f"D1 trainer mailbox payload contains duplicate key {key!r}")
        value[key] = item
    return value


def _decode(payload: bytes, *, fields: frozenset[str], label: str) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError(f"{label} payload must be bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        _fail(f"{label} must be UTF-8 JSON", error)
    try:
        value: object = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(
                f"{label} contains nonfinite constant {constant}"
            ),
        )
    except json.JSONDecodeError as error:
        _fail(f"{label} must be UTF-8 JSON", error)
    except ValueError:
        raise
    if type(value) is not dict:
        _fail(f"{label} root must be an object")
    root = cast(dict[str, object], value)
    if frozenset(root) != fields:
        _fail(f"{label} fields must be exact")
    if _canonical(root) != payload:
        _fail(f"{label} must use canonical JSON bytes")
    return root


def parse_d1_trainer_mailbox_command(payload: bytes) -> D1TrainerMailboxCommand:
    root = _decode(payload, fields=_FIELDS, label="D1 trainer mailbox command")
    try:
        command = D1TrainerMailboxCommand(
            D1TrainerMailboxCommandKind(cast(str, root["kind"])),
            cast(str, root["pod_id"]),
            cast(str, root["attempt_sha256"]),
            cast(str | None, root["request_sha256"]),
            cast(str, root["schema_version"]),
        )
    except (TypeError, ValueError) as error:
        _fail(f"D1 trainer mailbox command value is invalid: {error}", error)
    return command


def parse_d1_trainer_mailbox_error(payload: bytes) -> D1TrainerMailboxErrorReport:
    root = _decode(payload, fields=_ERROR_FIELDS, label="D1 trainer mailbox error")
    try:
        return D1TrainerMailboxErrorReport(
            cast(str, root["pod_id"]),
            cast(str, root["attempt_sha256"]),
            cast(str, root["stage"]),
            cast(str, root["error_type"]),
            cast(str, root["schema_version"]),
        )
    except (TypeError, ValueError) as error:
        _fail(f"D1 trainer mailbox error value is invalid: {error}", error)
