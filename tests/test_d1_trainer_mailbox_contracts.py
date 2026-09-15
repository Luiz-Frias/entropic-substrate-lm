"""D1 trainer mailbox commands bind one pod and one immutable attempt."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from entropic_substrate.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxCommand,
    D1TrainerMailboxCommandKind,
    D1TrainerMailboxErrorReport,
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
    d1_trainer_mailbox_attempt_prefix,
    d1_trainer_mailbox_command_path,
    d1_trainer_mailbox_error_path,
    parse_d1_trainer_mailbox_command,
    parse_d1_trainer_mailbox_error,
    serialize_d1_trainer_mailbox_command,
    serialize_d1_trainer_mailbox_error,
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


def test_mailbox_error_report_round_trips_without_free_text() -> None:
    report = D1TrainerMailboxErrorReport(
        pod_id=_POD,
        attempt_sha256=_ATTEMPT,
        stage="bootstrap",
        error_type="ValueError",
    )

    payload = serialize_d1_trainer_mailbox_error(report)

    assert parse_d1_trainer_mailbox_error(payload) == report
    assert json.loads(payload) == report.manifest
    assert d1_trainer_mailbox_error_path(_POD, _ATTEMPT) == (
        f"d1-mailbox/v1/{_POD}/{_ATTEMPT}/trainer-error.json"
    )
    assert "message" not in report.manifest and "detail" not in report.manifest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pod_id", "../pod"),
        ("attempt_sha256", "A" * 64),
        ("stage", "bad stage"),
        ("stage", ""),
        ("error_type", "Value Error"),
        ("error_type", ""),
    ],
)
def test_mailbox_error_report_refuses_unsafe_values(field: str, value: str) -> None:
    values = {
        "pod_id": _POD,
        "attempt_sha256": _ATTEMPT,
        "stage": "start",
        "error_type": "RuntimeError",
    }
    values[field] = value
    with pytest.raises(ValueError):
        D1TrainerMailboxErrorReport(**values)


def test_mailbox_error_parser_refuses_drift_and_duplicate_keys() -> None:
    report = D1TrainerMailboxErrorReport(_POD, _ATTEMPT, "start", "ValueError")
    payload = serialize_d1_trainer_mailbox_error(report)
    with pytest.raises(ValueError):
        parse_d1_trainer_mailbox_error(payload + b"\n")
    with pytest.raises(ValueError, match="duplicate"):
        parse_d1_trainer_mailbox_error(payload.replace(b'"stage":', b'"stage":"x","stage":'))
    with pytest.raises(TypeError, match="bytes"):
        parse_d1_trainer_mailbox_error("{}")  # type: ignore[arg-type]


def test_mailbox_object_read_binds_key_size_digest_and_bytes() -> None:
    payload = b"canonical mailbox object"
    digest = hashlib.sha256(payload).hexdigest()
    identity = D1TrainerMailboxObjectIdentity(
        key=f"d1-mailbox/v1/{_POD}/{_ATTEMPT}/artifact.json",
        size_bytes=len(payload),
        sha256=digest,
    )

    observed = D1TrainerMailboxObjectRead(identity, payload)

    assert observed.identity is identity and observed.payload == payload
    with pytest.raises(ValueError, match="bytes"):
        D1TrainerMailboxObjectRead(identity, payload + b"drift")
    with pytest.raises(ValueError):
        replace(identity, key="../artifact.json")
    with pytest.raises(ValueError):
        replace(identity, sha256=digest.upper())
    with pytest.raises(ValueError):
        replace(identity, size_bytes=0)
