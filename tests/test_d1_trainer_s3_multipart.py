"""FOR-1694 (h1-l4c): an object above the multipart threshold is published in verified parts.

The fake client counts every ``put_object``, ``create_multipart_upload``, ``upload_part`` and
``complete_multipart_upload`` the adapter issues, so a test reads what reached the PROVIDER, not
what the adapter reports. R2's handling of conditional completion and part checksums is
UNMEASURED; these tests pin the adapter's own contract against the S3 shape.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field

import pytest
from tests.test_d1_trainer_s3_object_adapter import _BUCKET, _ClientFailure, _S3Client

from entropic_substrate.ladder.run.transport.d1_trainer_s3_object_adapter import (
    MULTIPART_PART_BYTES,
    MULTIPART_THRESHOLD_BYTES,
    Boto3D1TrainerMailboxObjectAdapter,
    D1TrainerS3ObjectError,
)

pytestmark = pytest.mark.unit

KEY = "h-probes/v1/n/outputs/run/donor/checkpoint.pt"
THRESHOLD, PART = 10, 4  # tiny injected sizes: the production constants are pinned separately
PAYLOAD = bytes(range(25))  # 25 bytes: above the threshold, in 7 parts of 4 (the last is 1)


@dataclass
class _MultipartClient(_S3Client):
    uploads: dict[str, dict[int, bytes]] = field(default_factory=dict)
    upload_keys: dict[str, str] = field(default_factory=dict)
    upload_metadata: dict[str, str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    corrupt_part: int | None = None  # the provider echoes a WRONG digest for this part
    omit_echo: bool = False
    fail_part: int | None = None
    fail_complete: Exception | None = None
    aborted: list[str] = field(default_factory=list)
    corrupt_stored: bool = False

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append("put_object")
        return super().put_object(**kwargs)

    def create_multipart_upload(self, **kwargs: object) -> dict[str, object]:
        self.calls.append("create")
        upload_id = f"upload-{len(self.uploads) + 1}"
        self.uploads[upload_id] = {}
        self.upload_keys[upload_id] = str(kwargs["Key"])
        metadata = kwargs["Metadata"]
        assert isinstance(metadata, dict)
        self.upload_metadata[upload_id] = str(metadata["sha256"])
        return {"UploadId": upload_id}

    def upload_part(self, **kwargs: object) -> dict[str, object]:
        number = int(str(kwargs["PartNumber"]))
        self.calls.append(f"part-{number}")
        if self.fail_part == number:
            raise _ClientFailure("InternalError", 500)
        body = kwargs["Body"]
        assert isinstance(body, bytes) and kwargs["ContentLength"] == len(body)
        sent = str(kwargs["ChecksumSHA256"])
        assert sent == base64.b64encode(hashlib.sha256(body).digest()).decode()
        self.uploads[str(kwargs["UploadId"])][number] = body
        echoed = "AAAA" if self.corrupt_part == number else sent
        row: dict[str, object] = {"ETag": f'"etag-{number}"'}
        if not self.omit_echo:
            row["ChecksumSHA256"] = echoed
        return row

    def complete_multipart_upload(self, **kwargs: object) -> dict[str, object]:
        self.calls.append("complete")
        if self.fail_complete is not None:
            raise self.fail_complete
        upload_id = str(kwargs["UploadId"])
        key = self.upload_keys[upload_id]
        if kwargs.get("IfNoneMatch") == "*" and key in self.values:
            raise _ClientFailure("PreconditionFailed", 412)
        multipart = kwargs["MultipartUpload"]
        assert isinstance(multipart, dict)
        numbers = [part["PartNumber"] for part in multipart["Parts"]]
        assert numbers == sorted(numbers) == list(range(1, len(numbers) + 1))
        stored = b"".join(self.uploads[upload_id][n] for n in numbers)
        self.values[key] = stored[:-1] + b"\0" if self.corrupt_stored else stored
        self.metadata[key] = self.upload_metadata[upload_id]
        return {}

    def abort_multipart_upload(self, **kwargs: object) -> dict[str, object]:
        self.calls.append("abort")
        self.aborted.append(str(kwargs["UploadId"]))
        return {}


def adapter(client: _MultipartClient, **options: object) -> Boto3D1TrainerMailboxObjectAdapter:
    return Boto3D1TrainerMailboxObjectAdapter(
        client=client,
        bucket=_BUCKET,
        max_objects=10,
        multipart_threshold_bytes=THRESHOLD,
        multipart_part_bytes=PART,
        **options,  # type: ignore[arg-type]
    )


def publish(client: _MultipartClient, payload: bytes = PAYLOAD, key: str = KEY) -> object:
    return adapter(client).publish(
        key, payload, expected_sha256=hashlib.sha256(payload).hexdigest()
    )


def test_the_production_sizes_are_the_measured_ones_and_the_constructor_rejects_nonsense() -> None:
    assert MULTIPART_THRESHOLD_BYTES == MULTIPART_PART_BYTES == 64 * 1024 * 1024
    # the measured quote-v4 checkpoint (1,149,203,227 bytes) is multipart and fits the part cap
    assert 1_149_203_227 > MULTIPART_THRESHOLD_BYTES
    assert -(-1_149_203_227 // MULTIPART_PART_BYTES) <= 10_000
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="multipart sizes"):
            Boto3D1TrainerMailboxObjectAdapter(
                client=_MultipartClient(),
                bucket=_BUCKET,
                max_objects=1,
                multipart_threshold_bytes=bad,  # type: ignore[arg-type]
            )


def test_an_object_at_or_below_the_threshold_is_one_put_and_above_it_is_parts() -> None:
    client = _MultipartClient()
    small = bytes(THRESHOLD)  # exactly at the threshold: not above it
    publish(client, small, "small")
    assert client.calls == ["put_object"]
    client = _MultipartClient()
    identity = publish(client)
    assert client.calls == [
        "create",
        *[f"part-{n}" for n in range(1, 8)],
        "complete",
    ]  # NO put_object: one object, seven verified parts, one completion
    assert identity.sha256 == hashlib.sha256(PAYLOAD).hexdigest()  # type: ignore[attr-defined]
    assert client.values[KEY] == PAYLOAD
    assert client.metadata[KEY] == hashlib.sha256(PAYLOAD).hexdigest()  # `_read` admits it as is


def test_the_published_object_reads_back_through_the_ordinary_read_and_head() -> None:
    client = _MultipartClient()
    publish(client)
    read = adapter(client).read(KEY)
    assert read is not None and read.payload == PAYLOAD
    assert [i.key for i in adapter(client).list("h-probes/v1/n/")] == [KEY]


def test_a_part_whose_digest_disagrees_is_refused_and_the_upload_is_aborted() -> None:
    client = _MultipartClient(corrupt_part=3)
    with pytest.raises(D1TrainerS3ObjectError) as raised:
        publish(client)
    assert raised.value.code == "part_digest" and raised.value.operation == "publish"
    assert client.calls[-1] == "abort" and "complete" not in client.calls
    assert "part-4" not in client.calls  # the first disagreeing part stops the upload
    assert KEY not in client.values


def test_a_provider_that_echoes_no_part_digest_cannot_be_verified_and_is_refused() -> None:
    client = _MultipartClient(omit_echo=True)
    with pytest.raises(D1TrainerS3ObjectError) as raised:
        publish(client)
    assert raised.value.code == "part_digest" and client.calls[-1] == "abort"


def test_a_failing_part_aborts_without_a_retry() -> None:
    client = _MultipartClient(fail_part=2)
    with pytest.raises(D1TrainerS3ObjectError) as raised:
        publish(client)
    assert raised.value.code == "provider_failure"
    assert client.calls == ["create", "part-1", "part-2", "abort"]  # no retry: house rule 3


def test_stored_bytes_that_differ_from_the_payload_are_refused_by_the_readback() -> None:
    """The final readback is what proves the STORED bytes: a store that assembled other bytes than
    the parts carried fails the readback's own identity check (``object_bytes``: the read refuses
    a payload that does not hash to the object's recorded digest), exactly as a corrupt single PUT
    does, and success is never reported."""
    client = _MultipartClient(corrupt_stored=True)
    with pytest.raises(D1TrainerS3ObjectError) as raised:
        publish(client)
    assert raised.value.code in ("object_bytes", "readback_mismatch")


def test_an_existing_key_is_never_overwritten_identical_is_idempotent_different_collides() -> None:
    client = _MultipartClient()
    publish(client)
    before = list(client.calls)
    publish(client)  # identical bytes: an idempotent success, nothing uploaded
    assert client.calls == before
    different = bytes(reversed(PAYLOAD))
    with pytest.raises(D1TrainerS3ObjectError) as raised:
        publish(client, different)
    assert raised.value.code == "immutable_collision" and client.values[KEY] == PAYLOAD
    assert client.calls == before  # a collision starts no upload either


def race_client(foreign: bytes | None) -> _MultipartClient:
    """A client whose key is taken by ANOTHER writer after our upload began (after the head probe):
    ``foreign`` None means the winner wrote our own bytes."""
    client = _MultipartClient()
    original = client.create_multipart_upload

    def create_then_race(**kwargs: object) -> dict[str, object]:
        row = original(**kwargs)
        winner = PAYLOAD if foreign is None else foreign
        client.values[KEY] = winner
        client.metadata[KEY] = hashlib.sha256(winner).hexdigest()
        return row

    client.create_multipart_upload = create_then_race  # type: ignore[method-assign]
    return client


def test_a_lost_race_at_completion_is_read_and_compared_never_overwritten() -> None:
    same = race_client(None)
    identity = publish(same)
    assert identity.sha256 == hashlib.sha256(PAYLOAD).hexdigest()  # type: ignore[attr-defined]
    assert same.calls[-1] == "abort"  # the loser's upload is cleaned up
    foreign = race_client(bytes(reversed(PAYLOAD)))
    with pytest.raises(D1TrainerS3ObjectError) as raised:
        publish(foreign)
    assert raised.value.code == "immutable_collision"
    assert foreign.values[KEY] == bytes(reversed(PAYLOAD))  # the winner's bytes are untouched


def test_a_client_without_the_multipart_methods_is_refused_by_name_not_downgraded_to_one_put() -> (
    None
):
    client = _S3Client()  # the four-method client every older caller builds
    with pytest.raises(D1TrainerS3ObjectError) as raised:
        publish(client)  # type: ignore[arg-type]
    assert raised.value.code == "multipart_unsupported" and client.puts == []


def test_the_object_byte_limit_still_refuses_before_any_upload() -> None:
    client = _MultipartClient()
    with pytest.raises(D1TrainerS3ObjectError) as raised:
        adapter(client, max_object_bytes=len(PAYLOAD) - 1).publish(
            KEY, PAYLOAD, expected_sha256=hashlib.sha256(PAYLOAD).hexdigest()
        )
    assert raised.value.code == "byte_limit" and client.calls == []


def test_a_failing_abort_is_logged_by_type_and_never_masks_the_cause() -> None:
    from entropic_substrate.kernel.runtime.logging import logger

    client = _MultipartClient(fail_part=2)

    def broken_abort(**_: object) -> dict[str, object]:
        raise _ClientFailure("InternalError", 500)

    client.abort_multipart_upload = broken_abort  # type: ignore[method-assign]
    records: list[str] = []
    handler = logger.add(lambda message: records.append(message.record["message"]), level="WARNING")
    try:
        with pytest.raises(D1TrainerS3ObjectError) as raised:
            publish(client)
    finally:
        logger.remove(handler)
    assert raised.value.code == "provider_failure"  # the part failure, not the abort's
    assert records == ["multipart abort failed: _ClientFailure"]
