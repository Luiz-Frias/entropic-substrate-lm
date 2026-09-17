"""Strict S3-compatible object adapter for the D1 trainer mailbox port."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Protocol, cast

from entropic_substrate.d1_artifact_snapshot import normalize_d1_relative_artifact_path
from entropic_substrate.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)

__all__ = ["Boto3D1TrainerMailboxObjectAdapter", "D1TrainerS3ObjectError"]

_BUCKET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class D1S3BodyPort(Protocol):
    def read(self, amount: int) -> bytes: ...
    def close(self) -> None: ...


class D1S3ClientPort(Protocol):
    def put_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def get_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def head_object(self, **kwargs: object) -> Mapping[str, object]: ...
    def list_objects_v2(self, **kwargs: object) -> Mapping[str, object]: ...


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
        self._client = client
        self._bucket = bucket
        self._max_objects = max_objects
        self._max_object_bytes = max_object_bytes

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
