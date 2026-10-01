"""S3-compatible D1 mailbox objects are immutable and byte-verified."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest

from entropic_substrate.ladder.run.transport.d1_trainer_s3_object_adapter import (
    Boto3D1TrainerMailboxObjectAdapter,
    D1TrainerS3ObjectError,
)

pytestmark = pytest.mark.unit

_BUCKET = "network-volume-123"
_KEY = "d1-mailbox/v1/pod/a/artifact.json"
_PAYLOAD = b'{"artifact":"canonical"}'
_SHA = hashlib.sha256(_PAYLOAD).hexdigest()


class _ClientFailure(RuntimeError):
    def __init__(self, code: str, status: int) -> None:
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }
        super().__init__("provider-private-detail")


@dataclass
class _Body:
    payload: bytes
    truncate: bool = False
    closed: bool = False
    reads: list[int] = field(default_factory=list)

    def read(self, amount: int) -> bytes:
        self.reads.append(amount)
        value = self.payload[:-1] if self.truncate else self.payload
        return value[:amount]

    def close(self) -> None:
        self.closed = True


@dataclass
class _S3Client:
    values: dict[str, bytes] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    puts: list[dict[str, object]] = field(default_factory=list)
    truncate: bool = False
    fail_put: Exception | None = None
    fail_get: Exception | None = None
    fail_list: Exception | None = None
    last_body: _Body | None = None
    # FOR-1567: fails exactly this many of the NEXT get_object calls (any
    # key), regardless of whether the caller is the adapter's own post-PUT
    # readback (inside publish()) or a later explicit objects.read() -- this
    # is what lets a test inject "the first get_object call(s) issued after
    # put_object" without caring which caller made them.
    fail_get_times: int = 0
    get_calls: list[str] = field(default_factory=list)

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.puts.append(dict(kwargs))
        if self.fail_put is not None:
            raise self.fail_put
        key = str(kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and key in self.values:
            raise _ClientFailure("PreconditionFailed", 412)
        payload = kwargs["Body"]
        assert isinstance(payload, bytes)
        self.values[key] = payload
        metadata = kwargs["Metadata"]
        assert isinstance(metadata, dict)
        self.metadata[key] = str(metadata["sha256"])
        return {"ETag": '"opaque"'}

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(str(kwargs["Key"]))
        if self.fail_get is not None:
            raise self.fail_get
        if self.fail_get_times > 0:
            self.fail_get_times -= 1
            raise _ClientFailure("InternalError", 500)
        key = str(kwargs["Key"])
        if key not in self.values:
            raise _ClientFailure("NoSuchKey", 404)
        body = _Body(self.values[key], self.truncate)
        self.last_body = body
        return {
            "Body": body,
            "ContentLength": len(self.values[key]),
            "Metadata": {"sha256": self.metadata[key]},
        }

    def head_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        if key not in self.values:
            raise _ClientFailure("NotFound", 404)
        return {
            "ContentLength": len(self.values[key]),
            "Metadata": {"sha256": self.metadata[key]},
        }

    def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
        if self.fail_list is not None:
            raise self.fail_list
        prefix = str(kwargs["Prefix"])
        return {
            "Contents": [
                {"Key": key, "Size": len(payload)}
                for key, payload in sorted(self.values.items())
                if key.startswith(prefix)
            ],
            "IsTruncated": False,
            "KeyCount": sum(key.startswith(prefix) for key in self.values),
        }


def _adapter(client: _S3Client, *, max_objects: int = 64):
    return Boto3D1TrainerMailboxObjectAdapter(
        client=client,
        bucket=_BUCKET,
        max_objects=max_objects,
    )


def test_explicit_byte_cap_refuses_before_body_read_or_publication() -> None:
    client = _S3Client(values={_KEY: _PAYLOAD}, metadata={_KEY: _SHA})
    adapter = Boto3D1TrainerMailboxObjectAdapter(
        client=client, bucket=_BUCKET, max_objects=64, max_object_bytes=1
    )
    with pytest.raises(D1TrainerS3ObjectError, match="byte_limit"):
        adapter.read(_KEY)
    assert client.last_body is not None and client.last_body.closed
    assert client.last_body.reads == []
    with pytest.raises(D1TrainerS3ObjectError, match="byte_limit"):
        adapter.publish(_KEY + "-new", _PAYLOAD, expected_sha256=_SHA)
    assert client.puts == []
    assert _KEY + "-new" not in client.values


def test_exact_byte_cap_accepts_verified_readback() -> None:
    client = _S3Client()
    adapter = Boto3D1TrainerMailboxObjectAdapter(
        client=client, bucket=_BUCKET, max_objects=64, max_object_bytes=len(_PAYLOAD)
    )
    assert adapter.publish(_KEY, _PAYLOAD, expected_sha256=_SHA).size_bytes == len(_PAYLOAD)
    assert adapter.read(_KEY).payload == _PAYLOAD


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_object_byte_cap_is_not_a_default(limit: object) -> None:
    with pytest.raises(ValueError, match="byte"):
        Boto3D1TrainerMailboxObjectAdapter(
            client=_S3Client(), bucket=_BUCKET, max_objects=64, max_object_bytes=limit
        )


def test_bad_metadata_still_closes_the_received_body() -> None:
    class BadMetadata(_S3Client):
        def get_object(self, **kwargs: object) -> dict[str, object]:
            return super().get_object(**kwargs) | {"Metadata": None}

    client = BadMetadata(values={_KEY: _PAYLOAD}, metadata={_KEY: _SHA})
    with pytest.raises(D1TrainerS3ObjectError, match="metadata_invalid"):
        _adapter(client).read(_KEY)
    assert client.last_body is not None and client.last_body.closed


def test_publish_uses_conditional_put_then_verified_readback() -> None:
    client = _S3Client()

    identity = _adapter(client).publish(_KEY, _PAYLOAD, expected_sha256=_SHA)

    assert identity.key == _KEY and identity.size_bytes == len(_PAYLOAD)
    assert identity.sha256 == _SHA
    assert client.puts == [
        {
            "Body": _PAYLOAD,
            "Bucket": _BUCKET,
            "ContentLength": len(_PAYLOAD),
            "IfNoneMatch": "*",
            "Key": _KEY,
            "Metadata": {"sha256": _SHA},
        }
    ]
    assert client.last_body is not None and client.last_body.closed is True


def test_idempotent_publish_admits_same_bytes_but_refuses_collision() -> None:
    client = _S3Client()
    adapter = _adapter(client)
    first = adapter.publish(_KEY, _PAYLOAD, expected_sha256=_SHA)

    assert adapter.publish(_KEY, _PAYLOAD, expected_sha256=_SHA) == first
    with pytest.raises(D1TrainerS3ObjectError, match="immutable_collision"):
        adapter.publish(
            _KEY,
            b"foreign bytes",
            expected_sha256=hashlib.sha256(b"foreign bytes").hexdigest(),
        )


def test_read_returns_none_for_absent_and_rejects_truncation() -> None:
    client = _S3Client()
    adapter = _adapter(client)
    assert adapter.read(_KEY) is None
    adapter.publish(_KEY, _PAYLOAD, expected_sha256=_SHA)
    client.truncate = True

    with pytest.raises(D1TrainerS3ObjectError, match="object_bytes"):
        adapter.read(_KEY)

    assert client.last_body is not None and client.last_body.closed is True


def test_list_uses_head_metadata_and_returns_sorted_identities() -> None:
    client = _S3Client()
    adapter = _adapter(client)
    other_key = _KEY.replace("artifact", "another")
    adapter.publish(_KEY, _PAYLOAD, expected_sha256=_SHA)
    adapter.publish(other_key, b"other", expected_sha256=hashlib.sha256(b"other").hexdigest())

    values = adapter.list("d1-mailbox/v1/pod/a/")

    assert tuple(item.key for item in values) == tuple(sorted((_KEY, other_key)))
    assert all(item.size_bytes > 0 and len(item.sha256) == 64 for item in values)


def test_listing_bound_and_provider_failures_are_safe() -> None:
    client = _S3Client()
    client.values.update({f"{_KEY}-{index}": b"x" for index in range(2)})
    client.metadata.update({key: hashlib.sha256(b"x").hexdigest() for key in client.values})
    with pytest.raises(D1TrainerS3ObjectError, match="object_limit"):
        _adapter(client, max_objects=1).list("d1-mailbox/")

    for operation, field_name in (
        ("publish", "fail_put"),
        ("read", "fail_get"),
        ("list", "fail_list"),
    ):
        failing = _S3Client()
        setattr(failing, field_name, RuntimeError("provider-private-detail"))
        adapter = _adapter(failing)
        with pytest.raises(D1TrainerS3ObjectError) as raised:
            if operation == "publish":
                adapter.publish(_KEY, _PAYLOAD, expected_sha256=_SHA)
            elif operation == "read":
                adapter.read(_KEY)
            else:
                adapter.list("d1-mailbox/")
        assert "provider-private-detail" not in str(raised.value)


@pytest.mark.parametrize(
    ("bucket", "maximum"),
    [("", 1), ("bad bucket", 1), (_BUCKET, 0), (_BUCKET, True)],
)
def test_adapter_requires_safe_bucket_and_positive_object_bound(
    bucket: str,
    maximum: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Boto3D1TrainerMailboxObjectAdapter(
            client=_S3Client(),
            bucket=bucket,
            max_objects=maximum,  # type: ignore[arg-type]
        )
