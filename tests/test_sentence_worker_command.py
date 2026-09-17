"""The H command surface composes the same real worker and object protocol."""

import hashlib
import json
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_d1_trainer_s3_object_adapter import _S3Client
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.sentence_prediction.worker_command import (
    build_worker_objects,
    run_worker_command,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("field", ["operation", "code", "cause_type"])
def test_unknown_storage_diagnostic_fields_never_escape_allowlist(field: str) -> None:
    from entropic_substrate.d1_trainer_s3_object_adapter import D1TrainerS3ObjectError

    error = D1TrainerS3ObjectError(
        operation="private-value" if field == "operation" else "read",
        code="private-value" if field == "code" else "provider_failure",
    )
    if field == "cause_type":
        error.__cause__ = type("private-value", (Exception,), {})("private-message")
    diagnostic = error.safe_diagnostic()
    assert diagnostic[field] == "unrecognized"
    assert "private" not in json.dumps(diagnostic)


@pytest.mark.parametrize(
    ("cause", "provider_code", "http_status", "cause_type"),
    [
        (
            ClientError(
                {
                    "Error": {"Code": "AccessDenied", "Message": "private-value"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "GetObject",
            ),
            "AccessDenied",
            403,
            "ClientError",
        ),
        (
            ClientError(
                {
                    "Error": {"Code": "SignatureDoesNotMatch"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "GetObject",
            ),
            "SignatureDoesNotMatch",
            403,
            "ClientError",
        ),
        (
            ClientError(
                {"Error": {"Code": "private-value"}, "ResponseMetadata": {"HTTPStatusCode": True}},
                "GetObject",
            ),
            "unrecognized",
            None,
            "ClientError",
        ),
        (
            EndpointConnectionError(endpoint_url="https://private-value.invalid"),
            None,
            None,
            "EndpointConnectionError",
        ),
    ],
)
def test_request_failure_retains_only_allowlisted_storage_diagnostics(
    tmp_path: Path,
    cause: Exception,
    provider_code: str | None,
    http_status: int | None,
    cause_type: str,
) -> None:
    from entropic_substrate.d1_trainer_s3_object_adapter import Boto3D1TrainerMailboxObjectAdapter

    client = _S3Client(fail_get=cause)
    objects = Boto3D1TrainerMailboxObjectAdapter(
        client=client, bucket="test-bucket", max_objects=10
    )
    output, error = StringIO(), StringIO()
    workspace = tmp_path / "not-created"
    status = run_worker_command(
        [
            "run",
            "--repo-root",
            str(tmp_path),
            "--workspace-root",
            str(workspace),
            "--max-object-bytes",
            "1000",
            "--max-objects",
            "10",
            "--request-key",
            "request.json",
            "--request-sha256",
            "a" * 64,
            "--request-size",
            "10",
        ],
        environ={},
        objects=objects,
        stdout=output,
        stderr=error,
    )
    assert status == 2 and not output.getvalue() and not workspace.exists()
    record = json.loads(error.getvalue())
    assert record["stage"] == "request_fetch"
    assert record["object_error"] == {
        "operation": "read",
        "code": "provider_failure",
        "cause_type": cause_type,
        "provider_code": provider_code,
        "http_status": http_status,
    }
    assert "private-value" not in error.getvalue()


@pytest.mark.parametrize("name", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
def test_unresolved_secret_reference_is_refused_before_client_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    factory = Mock()
    monkeypatch.setattr(boto3, "client", factory)
    credentials = {"AWS_ACCESS_KEY_ID": "test-id", "AWS_SECRET_ACCESS_KEY": "test-value"}
    credentials[name] = "{{ RUNPOD_SECRET_test_private_name }}"
    output, error = StringIO(), StringIO()
    status = run_worker_command(
        [
            "publish",
            "--repo-root",
            str(tmp_path),
            "--max-object-bytes",
            "1000",
            "--max-objects",
            "10",
            "--artifact-root",
            str(tmp_path),
            "--request-file",
            "missing.json",
        ],
        environ=credentials,
        stdout=output,
        stderr=error,
    )
    assert status == 2
    assert json.loads(error.getvalue())["credential_error"] == "unresolved_reference"
    assert json.loads(error.getvalue())["stage"] == "object_client"
    assert "test_private_name" not in error.getvalue()
    factory.assert_not_called()


def test_public_worker_commands_publish_execute_and_receive(tmp_path: Path) -> None:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    output, error = StringIO(), StringIO()
    common = [
        "--repo-root",
        str(tmp_path),
        "--max-object-bytes",
        "10000000",
        "--max-objects",
        "100",
    ]
    assert (
        run_worker_command(
            ["publish", *common, "--artifact-root", str(inputs), "--request-file", "request.json"],
            environ={},
            objects=objects,
            stdout=output,
            stderr=error,
        )
        == 0
    )
    marker = json.loads(output.getvalue())["object"]
    worker = tmp_path / "command-worker"
    worker.mkdir()
    output = StringIO()
    assert (
        run_worker_command(
            [
                "run",
                *common,
                "--workspace-root",
                str(worker),
                "--request-key",
                marker["key"],
                "--request-sha256",
                marker["sha256"],
                "--request-size",
                str(marker["size_bytes"]),
                "--local-tiny",
            ],
            environ={},
            objects=objects,
            stdout=output,
            stderr=error,
        )
        == 0
    )
    terminal = json.loads(output.getvalue())["object"]
    host = tmp_path / "command-host"
    host.mkdir()
    output = StringIO()
    assert (
        run_worker_command(
            [
                "receive",
                *common,
                "--artifact-root",
                str(inputs),
                "--request-file",
                "request.json",
                "--destination",
                str(host),
                "--terminal-key",
                terminal["key"],
                "--terminal-sha256",
                terminal["sha256"],
                "--terminal-size",
                str(terminal["size_bytes"]),
                "--local-tiny",
            ],
            environ={},
            objects=objects,
            stdout=output,
            stderr=error,
        )
        == 0
    )
    result = json.loads(output.getvalue())
    assert result["status"] == "probe_completion_verified"
    assert result["request_sha256"] == request.digest
    assert error.getvalue() == ""


def test_command_refuses_missing_credentials_without_echoing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topology = Mock(side_effect=AssertionError("credential check must precede topology"))
    monkeypatch.setattr(
        "entropic_substrate.sentence_prediction.worker_command.load_d1_dvc_s3_remote", topology
    )
    output, error = StringIO(), StringIO()
    status = run_worker_command(
        [
            "publish",
            "--repo-root",
            str(tmp_path),
            "--max-object-bytes",
            "1000",
            "--max-objects",
            "10",
            "--artifact-root",
            str(tmp_path),
            "--request-file",
            "missing.json",
        ],
        environ={"AWS_ACCESS_KEY_ID": "sensitive-placeholder"},
        stdout=output,
        stderr=error,
    )
    assert status == 2 and "sensitive-placeholder" not in error.getvalue()
    assert json.loads(error.getvalue())["code"] == "worker_command_refused"
    assert json.loads(error.getvalue())["error_type"] == "ValueError"
    topology.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, True])
def test_factory_refuses_invalid_limits_before_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    client = Mock()
    monkeypatch.setattr(boto3, "client", client)
    for object_bytes, count in ((limit, 10), (1000, limit)):
        with pytest.raises(ValueError, match="positive integer"):
            build_worker_objects(tmp_path, {}, max_object_bytes=object_bytes, max_objects=count)
    client.assert_not_called()


def test_factory_uses_committed_topology_and_explicit_bounded_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(boto3, "client", factory)
    credentials = {"AWS_ACCESS_KEY_ID": "test-id", "AWS_SECRET_ACCESS_KEY": "test-value"}
    adapter = build_worker_objects(
        Path(__file__).resolve().parents[1],
        credentials,
        max_object_bytes=1000,
        max_objects=10,
    )
    assert factory.call_args.args == ("s3",)
    options = factory.call_args.kwargs
    assert options["endpoint_url"].endswith(".r2.cloudflarestorage.com")
    assert options["aws_access_key_id"] == credentials["AWS_ACCESS_KEY_ID"]
    assert options["aws_secret_access_key"] == credentials["AWS_SECRET_ACCESS_KEY"]
    config = options["config"]
    assert config.connect_timeout == config.read_timeout == 15
    assert config.retries == {"max_attempts": 0}
    payload = b"x" * 1001
    with pytest.raises(RuntimeError, match="byte_limit"):
        adapter.publish(
            "bounded.json", payload, expected_sha256=hashlib.sha256(payload).hexdigest()
        )
    client.put_object.assert_not_called()


def test_local_request_cannot_execute_as_remote_by_omitting_flag(tmp_path: Path) -> None:
    from entropic_substrate.sentence_prediction.worker_transport import publish_probe_request

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "refused-worker"
    workspace.mkdir()
    output, error = StringIO(), StringIO()
    result = run_worker_command(
        [
            "run",
            "--repo-root",
            str(tmp_path),
            "--max-object-bytes",
            "10000000",
            "--max-objects",
            "100",
            "--workspace-root",
            str(workspace),
            "--request-key",
            marker.key,
            "--request-sha256",
            marker.sha256,
            "--request-size",
            str(marker.size_bytes),
        ],
        environ={"RUNPOD_POD_ID": "test-owned-pod"},
        objects=objects,
        stdout=output,
        stderr=error,
    )
    assert result == 2
    assert not output.getvalue() and not tuple(workspace.iterdir())
    assert json.loads(error.getvalue())["code"] == "worker_command_refused"
