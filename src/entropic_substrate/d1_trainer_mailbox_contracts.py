"""Canonical command and object-key contracts for the D1 trainer mailbox."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn, cast

from entropic_substrate.d1_artifact_snapshot import normalize_d1_relative_artifact_path
from entropic_substrate.d1_operator_values import require_d1_pod_id
from entropic_substrate.experiments.d1_primitives import require_sha256

__all__ = [
    "D1TrainerMailboxCommand",
    "D1TrainerMailboxCommandKind",
    "d1_trainer_mailbox_attempt_prefix",
    "d1_trainer_mailbox_command_path",
    "parse_d1_trainer_mailbox_command",
    "serialize_d1_trainer_mailbox_command",
]

_SCHEMA = "d1-trainer-mailbox-command-v1"
_FIELDS = frozenset({"attempt_sha256", "kind", "pod_id", "request_sha256", "schema_version"})


def _digest(value: str, *, label: str) -> str:
    require_sha256(value, label=label)
    if value != value.lower():
        raise ValueError(f"{label} must use lowercase hexadecimal")
    return value


class D1TrainerMailboxCommandKind(StrEnum):
    BOOTSTRAP = "bootstrap"
    START = "start"


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


def serialize_d1_trainer_mailbox_command(command: D1TrainerMailboxCommand) -> bytes:
    if type(command) is not D1TrainerMailboxCommand:
        raise TypeError("D1 trainer mailbox serializer requires a nominal command")
    return json.dumps(
        command.manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise ValueError(message)
    raise ValueError(message) from error


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail(f"D1 trainer mailbox command contains duplicate key {key!r}")
        value[key] = item
    return value


def parse_d1_trainer_mailbox_command(payload: bytes) -> D1TrainerMailboxCommand:
    if type(payload) is not bytes:
        raise TypeError("D1 trainer mailbox command payload must be bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        _fail("D1 trainer mailbox command must be UTF-8 JSON", error)
    try:
        value: object = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(
                f"D1 trainer mailbox command contains nonfinite constant {constant}"
            ),
        )
    except json.JSONDecodeError as error:
        _fail("D1 trainer mailbox command must be UTF-8 JSON", error)
    except ValueError:
        raise
    if type(value) is not dict:
        _fail("D1 trainer mailbox command root must be an object")
    root = cast(dict[str, object], value)
    if frozenset(root) != _FIELDS:
        _fail("D1 trainer mailbox command fields must be exact")
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
    if serialize_d1_trainer_mailbox_command(command) != payload:
        _fail("D1 trainer mailbox command must use canonical JSON bytes")
    return command
