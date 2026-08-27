"""D1 trainer mailbox commands bind one pod and one immutable attempt."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from entropic_substrate.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxCommand,
    D1TrainerMailboxCommandKind,
    d1_trainer_mailbox_attempt_prefix,
    d1_trainer_mailbox_command_path,
    parse_d1_trainer_mailbox_command,
    serialize_d1_trainer_mailbox_command,
)

pytestmark = pytest.mark.unit

_POD = "pod-mailbox-123"
_ATTEMPT = "a" * 64
_REQUEST = "b" * 64


def test_mailbox_commands_round_trip_under_one_attempt_namespace() -> None:
    bootstrap = D1TrainerMailboxCommand.bootstrap(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
    )
    start = D1TrainerMailboxCommand.start(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
        request_sha256=_REQUEST,
    )

    assert d1_trainer_mailbox_attempt_prefix(_POD, _ATTEMPT) == (f"d1-mailbox/v1/{_POD}/{_ATTEMPT}")
    assert d1_trainer_mailbox_command_path(bootstrap) == (
        f"d1-mailbox/v1/{_POD}/{_ATTEMPT}/commands/bootstrap.json"
    )
    assert d1_trainer_mailbox_command_path(start) == (
        f"d1-mailbox/v1/{_POD}/{_ATTEMPT}/commands/start-{_REQUEST}.json"
    )
    for command in (bootstrap, start):
        payload = serialize_d1_trainer_mailbox_command(command)
        assert payload == json.dumps(
            command.manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        assert parse_d1_trainer_mailbox_command(payload) == command


def test_mailbox_command_kind_controls_request_identity() -> None:
    bootstrap = D1TrainerMailboxCommand.bootstrap(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
    )
    start = D1TrainerMailboxCommand.start(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
        request_sha256=_REQUEST,
    )

    with pytest.raises(ValueError, match=r"bootstrap.*request"):
        replace(bootstrap, request_sha256=_REQUEST)
    with pytest.raises(ValueError, match=r"start.*request"):
        replace(start, request_sha256=None)
    with pytest.raises(TypeError, match="kind"):
        replace(bootstrap, kind="bootstrap")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pod_id", "../pod"),
        ("pod_id", ""),
        ("attempt_sha256", "A" * 64),
        ("attempt_sha256", "short"),
    ],
)
def test_mailbox_command_refuses_unsafe_path_identities(field: str, value: str) -> None:
    values = {"pod_id": _POD, "attempt_sha256": _ATTEMPT}
    values[field] = value
    with pytest.raises(ValueError):
        D1TrainerMailboxCommand.bootstrap(**values)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload + b"\n",
        lambda payload: payload.replace(b'"schema_version"', b'"extra":1,"schema_version"'),
        lambda payload: payload.replace(
            b'"schema_version":"d1-trainer-mailbox-command-v1"',
            b'"schema_version":"foreign"',
        ),
        lambda payload: payload.replace(
            b'"kind":"bootstrap"',
            b'"kind":"unknown"',
        ),
        lambda payload: payload.replace(
            b'"pod_id":"pod-mailbox-123"',
            b'"pod_id":"../pod"',
        ),
    ],
)
def test_mailbox_command_parser_refuses_noncanonical_or_foreign_bytes(mutate) -> None:
    payload = serialize_d1_trainer_mailbox_command(
        D1TrainerMailboxCommand.bootstrap(
            pod_id=_POD,
            attempt_sha256=_ATTEMPT,
        )
    )

    with pytest.raises(ValueError):
        parse_d1_trainer_mailbox_command(mutate(payload))


def test_mailbox_parser_rejects_duplicate_keys_and_non_objects() -> None:
    duplicate = (
        b'{"attempt_sha256":"'
        + _ATTEMPT.encode()
        + b'","kind":"bootstrap","pod_id":"'
        + _POD.encode()
        + b'","pod_id":"foreign","request_sha256":null,'
        b'"schema_version":"d1-trainer-mailbox-command-v1"}'
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_d1_trainer_mailbox_command(duplicate)
    with pytest.raises(ValueError, match="object"):
        parse_d1_trainer_mailbox_command(b"[]")
    with pytest.raises(TypeError, match="bytes"):
        parse_d1_trainer_mailbox_command("{}")  # type: ignore[arg-type]


def test_mailbox_path_helpers_require_nominal_command_and_identity() -> None:
    with pytest.raises(TypeError, match="command"):
        d1_trainer_mailbox_command_path({})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        d1_trainer_mailbox_attempt_prefix("pod", "invalid")
    with pytest.raises(TypeError, match="command"):
        D1TrainerMailboxCommand(
            D1TrainerMailboxCommandKind.BOOTSTRAP,
            _POD,
            _ATTEMPT,
            None,
            schema_version=1,  # type: ignore[arg-type]
        )
