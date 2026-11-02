"""Strict S3-compatible object adapter for the D1 trainer mailbox port.

FOR-1694 (h1-l4c): an object above ``MULTIPART_THRESHOLD_BYTES`` is published as a verified
MULTIPART upload instead of one ``put_object``. MEASURED 2026-10-28 (quote v4, pod
qkfduyfla3b6rt): one single PUT of a 1,149,203,227-byte checkpoint raised a
``ReadTimeoutError`` that three retries did not recover, and a complete 1000-step workload ended
``execution_defect`` at ``final_snapshot``. The multipart path keeps the single-PUT contract:

1. an existing key is never overwritten: identical bytes are an idempotent success, different
   bytes raise ``immutable_collision``;
2. the ``sha256`` metadata is set, so ``_read`` and ``_head`` admit the object unchanged;
3. success is reported only once the STORED bytes are proven equal to the payload (a full
   readback, exactly as the single PUT does);
4. every part is sent with its own SHA-256 (``ChecksumSHA256``) and the provider's echo of it is
   compared; a part whose digest disagrees, or whose echo is absent, is refused with
   ``part_digest`` and the upload is aborted;
5. any failure after the upload began aborts it (best effort, never masking the cause).

No retry loop is added and no timeout or retry budget is raised (house rule 3). UNMEASURED:
Cloudflare R2's handling of ``IfNoneMatch`` on ``complete_multipart_upload`` and of
``ChecksumSHA256`` on ``upload_part``; no gate runs multipart against R2 (``compose.yaml`` has no
S3 service), so the first live use is the measurement.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping
from typing import Any, Protocol, cast

from entropic_substrate.kernel.artifacts import normalize_d1_relative_artifact_path
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)

__all__ = ["Boto3D1TrainerMailboxObjectAdapter", "D1TrainerS3ObjectError"]

_BUCKET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Sizes. MEASURED: the dev-width checkpoint is 182 MB (2026-10-27) and the quote-v4 pilot-width one
# is 1,149,203,227 bytes (2026-10-28), so a 64 MiB threshold sends both multipart while every
# small journal, seal and receipt stays one PUT. PROVIDER RULE (S3 and R2 documentation, not
# measured here): a part is 5 MiB to 5 GiB, an upload has at most 10,000 parts, and R2 requires
# every part but the last to be the same size; 64 MiB parts put the 1.15 GB checkpoint in 18 parts
# and keep any object under 640 GB inside the part cap.
MULTIPART_THRESHOLD_BYTES = 64 * 1024 * 1024
MULTIPART_PART_BYTES = 64 * 1024 * 1024
_MAX_PARTS = 10_000


class D1S3BodyPort(Protocol):
    def read(self, amount: int) -> bytes: ...
    def close(self) -> None: ...


class D1S3ClientPort(Protocol):
    def put_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def get_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def head_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def list_objects_v2(self, **kwargs: object) -> Mapping[str, object]: ...


# FOR-1667: a failed object READ that retrying the same request cannot cure. The adapter owns
# this classification (it raises the codes); callers filter on ``is_recoverable``. A deny-list
# of known-permanent facts, so a code nobody has seen yet stays recoverable: a wrongly-killed
# healthy run loses its evidence tail, a permanent read polled to a deadline is bounded.
_INTEGRITY_CODES = frozenset({"response_invalid", "metadata_invalid", "byte_limit", "object_bytes"})
_PERMANENT_PROVIDER_CODES = frozenset(
    {
        "AccessDenied",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "ExpiredToken",
        "InvalidToken",
        "NoSuchBucket",
        "InvalidArgument",
        "AuthorizationHeaderMalformed",
        "RequestTimeTooSkewed",
    }
)
_PERMANENT_HTTP_STATUSES = frozenset({400, 403})


class D1TrainerS3ObjectError(RuntimeError):
    """One bounded S3 protocol, identity, or immutability failure."""

    def __init__(self, *, operation: str, code: str) -> None:
        self.operation = operation
        self.code = code
        super().__init__(f"D1 trainer S3 object {operation}: {code}")

    def safe_diagnostic(self) -> dict[str, str | int | None]:
        """Allowlisted facts only: never messages, URLs, keys, headers or request IDs."""
        cause = self.__cause__
        provider_code, status = (
            _provider_code(cause) if isinstance(cause, Exception) else (None, None)
        )
        codes = {
            "AccessDenied",
            "InvalidAccessKeyId",
            "SignatureDoesNotMatch",
            "ExpiredToken",
            "InvalidToken",
            "RequestTimeTooSkewed",
            "AuthorizationHeaderMalformed",
            "InvalidArgument",
            "NoSuchBucket",
            "NoSuchKey",
            "NotFound",
            "SlowDown",
            "InternalError",
            "ServiceUnavailable",
            "400",
            "403",
            "404",
            "500",
            "503",
        }
        local_codes = {
            "metadata_invalid",
            "response_invalid",
            "provider_failure",
            "byte_limit",
            "body_read",
            "object_bytes",
            "identity_invalid",
            "payload_hash",
            "immutable_collision",
            "readback_mismatch",
            "head_failure",
            "head_identity",
            "row_invalid",
            "object_limit",
            "continuation_invalid",
            "part_digest",
            "multipart_unsupported",
            "multipart_incomplete",
        }
        cause_name = type(cause).__name__ if cause is not None else None
        cause_names = {
            "ClientError",
            "EndpointConnectionError",
            "ConnectTimeoutError",
            "ReadTimeoutError",
            "SSLError",
            "ConnectionClosedError",
            "ProxyConnectionError",
            "ParamValidationError",
        }
        return {
            "operation": self.operation
            if self.operation in {"read", "publish", "list"}
            else "unrecognized",
            "code": self.code if self.code in local_codes else "unrecognized",
            "cause_type": cause_name
            if cause_name in cause_names or cause is None
            else "unrecognized",
            "provider_code": provider_code
            if provider_code in codes or provider_code is None
            else "unrecognized",
            "http_status": status if type(status) is int and 100 <= status <= 599 else None,
        }

    def is_recoverable(self) -> bool:
        """False only for a read failure a retry cannot cure (FOR-1667).

        Reads the allowlisted ``safe_diagnostic()`` facts and nothing else (never the message,
        a key, a URL or a request id). Permanent: an integrity code, or a ``provider_failure``
        whose provider code is an auth or request-shape code, whose HTTP status is 400 or 403,
        or whose cause is a ``ParamValidationError``. Everything else, including every
        ``unrecognized`` fact, a connection or timeout cause, and a 5xx, is recoverable.
        """
        facts = self.safe_diagnostic()
        if facts["code"] in _INTEGRITY_CODES:
            return False
        if facts["code"] != "provider_failure":
            return True
        return not (
            facts["provider_code"] in _PERMANENT_PROVIDER_CODES
            or facts["http_status"] in _PERMANENT_HTTP_STATUSES
            or facts["cause_type"] == "ParamValidationError"
        )


def _error(operation: str, code: str) -> D1TrainerS3ObjectError:
    return D1TrainerS3ObjectError(operation=operation, code=code)


def _provider_code(error: Exception) -> tuple[str | None, int | None]:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None, None
    provider_error = response.get("Error")
    metadata = response.get("ResponseMetadata")
    code = provider_error.get("Code") if isinstance(provider_error, Mapping) else None
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    return (
        code if type(code) is str else None,
        status if type(status) is int else None,
    )


def _not_found(error: Exception) -> bool:
    code, status = _provider_code(error)
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _precondition_failed(error: Exception) -> bool:
    code, status = _provider_code(error)
    return code in {"PreconditionFailed", "412"} or status == 412


def _key(value: str) -> str:
    normalized = normalize_d1_relative_artifact_path(value, label="D1 trainer S3 object key")
    if normalized != value:
        raise ValueError("D1 trainer S3 object key is noncanonical")
    return value


def _prefix(value: str) -> str:
    if type(value) is not str or not value.endswith("/"):
        raise ValueError("D1 trainer S3 object prefix must end with slash")
    base = normalize_d1_relative_artifact_path(
        value.removesuffix("/"),
        label="D1 trainer S3 object prefix",
    )
    return f"{base}/"


def _metadata(value: object, *, operation: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(operation, "metadata_invalid")
    return cast(Mapping[str, object], value)


def _response(value: object, *, operation: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(operation, "response_invalid")
    return cast(Mapping[str, object], value)


class Boto3D1TrainerMailboxObjectAdapter:
    """Implement immutable publish/read/list over one S3-compatible bucket."""

    def __init__(
        self,
        *,
        client: D1S3ClientPort,
        bucket: str,
        max_objects: int,
        max_object_bytes: int | None = None,
        multipart_threshold_bytes: int = MULTIPART_THRESHOLD_BYTES,
        multipart_part_bytes: int = MULTIPART_PART_BYTES,
    ) -> None:
        methods = ("put_object", "get_object", "head_object", "list_objects_v2")
        if not all(callable(getattr(client, method, None)) for method in methods):
            raise TypeError("D1 trainer S3 adapter requires a boto3-compatible client")
        if type(bucket) is not str or _BUCKET.fullmatch(bucket) is None:
            raise ValueError("D1 trainer S3 bucket identity is unsafe")
        if type(max_objects) is not int or max_objects < 1:
            raise ValueError("D1 trainer S3 max_objects must be a positive integer")
        if max_object_bytes is not None and (
            type(max_object_bytes) is not int or max_object_bytes < 1
        ):
            raise ValueError("S3 object byte limit must be a positive integer")
        for value in (multipart_threshold_bytes, multipart_part_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("S3 multipart sizes must be positive integers")
        self._client = client
        self._bucket = bucket
        self._max_objects = max_objects
        self._max_object_bytes = max_object_bytes
        self._multipart_threshold = multipart_threshold_bytes
        self._multipart_part = multipart_part_bytes

    def _read(self, key: str) -> D1TrainerMailboxObjectRead | None:
        try:
            raw_response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception as cause:
            if _not_found(cause):
                return None
            raise _error("read", "provider_failure") from cause
        response = _response(raw_response, operation="read")
        body = response.get("Body")
        if not callable(getattr(body, "read", None)) or not callable(getattr(body, "close", None)):
            raise _error("read", "response_invalid")
        stream = cast(D1S3BodyPort, body)
        try:
            length = response.get("ContentLength")
            metadata = _metadata(response.get("Metadata"), operation="read")
            digest = metadata.get("sha256")
            if type(length) is not int or length < 1 or type(digest) is not str:
                raise _error("read", "response_invalid")
            if self._max_object_bytes is not None and length > self._max_object_bytes:
                raise _error("read", "byte_limit")
            try:
                payload = stream.read(length + 1)
            except Exception as cause:
                raise _error("read", "body_read") from cause
        finally:
            stream.close()
        try:
            identity = D1TrainerMailboxObjectIdentity(key, length, digest)
            return D1TrainerMailboxObjectRead(identity, payload)
        except (TypeError, ValueError) as cause:
            raise _error("read", "object_bytes") from cause

    def publish(
        self,
        key: str,
        payload: bytes,
        *,
        expected_sha256: str,
    ) -> D1TrainerMailboxObjectIdentity:
        normalized = _key(key)
        try:
            expected = D1TrainerMailboxObjectIdentity(
                normalized,
                len(payload),
                expected_sha256,
            )
        except (TypeError, ValueError) as cause:
            raise _error("publish", "identity_invalid") from cause
        if type(payload) is not bytes or hashlib.sha256(payload).hexdigest() != expected.sha256:
            raise _error("publish", "payload_hash")
        if self._max_object_bytes is not None and len(payload) > self._max_object_bytes:
            raise _error("publish", "byte_limit")
        if len(payload) > self._multipart_threshold:
            return self._publish_multipart(normalized, payload, expected)
        try:
            self._client.put_object(
                Body=payload,
                Bucket=self._bucket,
                ContentLength=len(payload),
                IfNoneMatch="*",
                Key=normalized,
                Metadata={"sha256": expected.sha256},
            )
        except Exception as cause:
            if not _precondition_failed(cause):
                raise _error("publish", "provider_failure") from cause
            current = self._read(normalized)
            if current is None or current.identity != expected or current.payload != payload:
                raise _error("publish", "immutable_collision") from cause
            return expected
        observed = self._read(normalized)
        if observed is None or observed.identity != expected or observed.payload != payload:
            raise _error("publish", "readback_mismatch")
        return expected

    def _abort(self, client: Any, key: str, upload_id: str) -> None:
        """Best-effort cleanup of an unfinished upload; it never masks the cause that led here."""
        try:
            client.abort_multipart_upload(Bucket=self._bucket, Key=key, UploadId=upload_id)
        except Exception as cleanup:
            # an unfinished upload is an orphan the provider may bill: say so, by TYPE only (never
            # a message, key or upload id), and let the cause that led here propagate
            logger.warning(
                "multipart abort failed: {error_type}", error_type=type(cleanup).__name__
            )

    def _publish_multipart(
        self, key: str, payload: bytes, expected: D1TrainerMailboxObjectIdentity
    ) -> D1TrainerMailboxObjectIdentity:
        """The verified multipart path; see the module docstring for its five promises."""
        client = cast(Any, self._client)
        methods = (
            "create_multipart_upload",
            "upload_part",
            "complete_multipart_upload",
            "abort_multipart_upload",
        )
        if not all(callable(getattr(client, name, None)) for name in methods):
            raise _error("publish", "multipart_unsupported")
        parts = [
            payload[start : start + self._multipart_part]
            for start in range(0, len(payload), self._multipart_part)
        ]
        if len(parts) > _MAX_PARTS:
            raise _error("publish", "multipart_unsupported")
        # (1) never overwrite: the key is probed BEFORE any upload starts
        try:
            client.head_object(Bucket=self._bucket, Key=key)
        except Exception as cause:
            if not _not_found(cause):
                raise _error("publish", "head_failure") from cause
        else:
            current = self._read(key)
            if current is None or current.identity != expected or current.payload != payload:
                raise _error("publish", "immutable_collision")
            return expected
        try:
            started = client.create_multipart_upload(
                Bucket=self._bucket,
                Key=key,
                Metadata={"sha256": expected.sha256},
                # declared here because every part carries a ChecksumSHA256 (S3 requires the
                # upload's algorithm to match its parts)
                ChecksumAlgorithm="SHA256",
            )
        except Exception as cause:
            raise _error("publish", "provider_failure") from cause
        upload_id = _response(started, operation="publish").get("UploadId")
        if type(upload_id) is not str or not upload_id:
            raise _error("publish", "response_invalid")
        completed: list[dict[str, object]] = []
        try:
            for number, part in enumerate(parts, start=1):
                digest = base64.b64encode(hashlib.sha256(part).digest()).decode()
                try:
                    sent = client.upload_part(
                        Bucket=self._bucket,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=number,
                        Body=part,
                        ContentLength=len(part),
                        ChecksumSHA256=digest,
                    )
                except Exception as cause:
                    raise _error("publish", "provider_failure") from cause
                echoed = _response(sent, operation="publish")
                # (4) the provider's echo of THIS part's digest, compared; absent is a refusal
                if echoed.get("ChecksumSHA256") != digest or type(echoed.get("ETag")) is not str:
                    raise _error("publish", "part_digest")
                completed.append(
                    {"PartNumber": number, "ETag": echoed["ETag"], "ChecksumSHA256": digest}
                )
            try:
                client.complete_multipart_upload(
                    Bucket=self._bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": completed},
                    IfNoneMatch="*",
                )
            except Exception as cause:
                if not _precondition_failed(cause):
                    raise _error("publish", "provider_failure") from cause
                # lost a race: another writer's object holds the key. Clean up OUR upload, then
                # read and compare; never overwrite.
                self._abort(client, key, upload_id)
                current = self._read(key)
                if current is None or current.identity != expected or current.payload != payload:
                    raise _error("publish", "immutable_collision") from cause
                return expected
        except BaseException:
            self._abort(client, key, upload_id)
            raise
        # (3) success only once the stored bytes are proven equal to the payload
        observed = self._read(key)
        if observed is None or observed.identity != expected or observed.payload != payload:
            raise _error("publish", "readback_mismatch")
        return expected

    def read(self, key: str) -> D1TrainerMailboxObjectRead | None:
        return self._read(_key(key))

    def _head(self, key: str, expected_size: int) -> D1TrainerMailboxObjectIdentity:
        try:
            raw_response = self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as cause:
            raise _error("list", "head_failure") from cause
        response = _response(raw_response, operation="list")
        size = response.get("ContentLength")
        digest = _metadata(response.get("Metadata"), operation="list").get("sha256")
        if size != expected_size or type(digest) is not str:
            raise _error("list", "head_identity")
        try:
            return D1TrainerMailboxObjectIdentity(key, expected_size, digest)
        except (TypeError, ValueError) as cause:
            raise _error("list", "head_identity") from cause

    def list(self, prefix: str) -> tuple[D1TrainerMailboxObjectIdentity, ...]:
        normalized = _prefix(prefix)
        identities: dict[str, D1TrainerMailboxObjectIdentity] = {}
        continuation: str | None = None
        while True:
            arguments: dict[str, object] = {
                "Bucket": self._bucket,
                "MaxKeys": min(self._max_objects + 1, 1000),
                "Prefix": normalized,
            }
            if continuation is not None:
                arguments["ContinuationToken"] = continuation
            try:
                raw_response = self._client.list_objects_v2(**arguments)
            except Exception as cause:
                raise _error("list", "provider_failure") from cause
            response = _response(raw_response, operation="list")
            contents = response.get("Contents", [])
            truncated = response.get("IsTruncated")
            if type(contents) is not list or type(truncated) is not bool:
                raise _error("list", "response_invalid")
            for value in contents:
                if not isinstance(value, Mapping):
                    raise _error("list", "row_invalid")
                key, size = value.get("Key"), value.get("Size")
                if (
                    type(key) is not str
                    or not key.startswith(normalized)
                    or type(size) is not int
                    or size < 1
                    or key in identities
                ):
                    raise _error("list", "row_invalid")
                identities[key] = self._head(key, size)
                if len(identities) > self._max_objects:
                    raise _error("list", "object_limit")
            if not truncated:
                return tuple(identities[key] for key in sorted(identities))
            token = response.get("NextContinuationToken")
            if type(token) is not str or not token or token == continuation:
                raise _error("list", "continuation_invalid")
            continuation = token
