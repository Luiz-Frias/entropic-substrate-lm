"""Canonical cross-process custody for a D1 COMMIT_REQUIRED capture preparation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, cast

from entropic_substrate.kernel.codec.strict_json import require_lowercase_sha256, strict_loads
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate_codec import (
    D1CaptureAggregateHydrationError,
    hydrate_d1_capture_aggregate,
    parse_d1_capture_aggregate_declaration,
    serialize_d1_capture_aggregate,
)
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import D1DvcBindingState
from entropic_substrate.ladder.capture.custody.d1_capture_ports import D1DvcPointerIdentity
from entropic_substrate.ladder.capture.custody.d1_capture_service import (
    D1CapturePreparationStatus,
    D1HeterogeneousCapturePreparation,
)
from entropic_substrate.ladder.register.d1_provenance import require_lowercase_digest

__all__ = [
    "D1_CAPTURE_PREPARATION_CUSTODY_SCHEMA",
    "D1CampaignCaptureAuthority",
    "D1CapturePreparationCustodyError",
    "D1CapturePreparationCustodyRecord",
    "capture_preparation_custody_from_preparation",
    "hydrate_d1_capture_preparation_custody",
    "parse_d1_capture_preparation_custody",
    "serialize_d1_capture_preparation_custody",
]

D1_CAPTURE_PREPARATION_CUSTODY_SCHEMA: Final = "d1-capture-preparation-custody-v2"
_MAX_BYTES: Final = 8 * 1024 * 1024
_ROOT_KEYS: Final = {
    "aggregate",
    "campaign_authority",
    "pointer",
    "schema_version",
    "status",
}
_AUTHORITY_KEYS: Final = {
    "attempt_ledger_digest",
    "campaign_digest",
    "lifecycle_event_count",
    "lifecycle_prefix",
    "matrix_digest",
    "registered_design_digest",
}


class D1CapturePreparationCustodyError(ValueError):
    """The persisted preparation is incomplete, substituted, or noncanonical."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1CapturePreparationCustodyError(message)
    raise D1CapturePreparationCustodyError(message) from error


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_aggregate_binding(
    aggregate_json: str,
    aggregate_sha256: str,
    pointer: D1DvcPointerIdentity,
) -> None:
    require_lowercase_sha256(
        aggregate_sha256,
        fail=lambda _reason: _fail(
            "D1 capture preparation aggregate is invalid: "
            "capture aggregate sha256 must be a 64-character hexadecimal SHA-256 digest"
        ),
    )
    try:
        declaration = parse_d1_capture_aggregate_declaration(aggregate_json)
    except (TypeError, ValueError, D1CaptureAggregateHydrationError) as error:
        _fail(f"D1 capture preparation aggregate is invalid: {error}", error)
    if declaration.manifest_digest != aggregate_sha256:
        _fail("D1 capture preparation aggregate digest differs from its declaration")
    artifacts = (
        tuple(item for value in declaration.complete_inputs for item in value.artifacts)
        + tuple(item for value in declaration.terminal_defect_inputs for item in value.artifacts)
        + tuple(item for value in declaration.attempt_evidence for item in value.artifacts)
    )
    if not artifacts or any(
        artifact.dvc.state is not D1DvcBindingState.BOUND
        or artifact.dvc.pointer_path != pointer.pointer_path
        or artifact.dvc.pointer_sha256 != pointer.pointer_sha256
        or artifact.dvc.object_id != pointer.object_id
        for artifact in artifacts
    ):
        _fail("D1 capture preparation aggregate and pointer authority differ")


@dataclass(frozen=True, slots=True)
class D1CampaignCaptureAuthority:
    """Campaign and lifecycle identity frozen at the COMMIT_REQUIRED boundary."""

    campaign_digest: str
    registered_design_digest: str
    matrix_digest: str
    attempt_ledger_digest: str
    lifecycle_prefix: str
    lifecycle_event_count: int

    def __post_init__(self) -> None:
        for label, value in (
            ("campaign_digest", self.campaign_digest),
            ("registered_design_digest", self.registered_design_digest),
            ("matrix_digest", self.matrix_digest),
            ("attempt_ledger_digest", self.attempt_ledger_digest),
            ("lifecycle_prefix", self.lifecycle_prefix),
        ):
            require_lowercase_digest(value, label=f"campaign capture {label}")
        if type(self.lifecycle_event_count) is not int or self.lifecycle_event_count < 1:
            raise ValueError("campaign capture lifecycle_event_count must be positive")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "attempt_ledger_digest": self.attempt_ledger_digest,
            "campaign_digest": self.campaign_digest,
            "lifecycle_event_count": self.lifecycle_event_count,
            "lifecycle_prefix": self.lifecycle_prefix,
            "matrix_digest": self.matrix_digest,
            "registered_design_digest": self.registered_design_digest,
        }


@dataclass(frozen=True, slots=True)
class D1CapturePreparationCustodyRecord:
    aggregate_json: str
    aggregate_sha256: str
    pointer: D1DvcPointerIdentity
    pointer_bytes: bytes
    authority: D1CampaignCaptureAuthority
    status: D1CapturePreparationStatus = D1CapturePreparationStatus.COMMIT_REQUIRED
    schema_version: str = D1_CAPTURE_PREPARATION_CUSTODY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.pointer) is not D1DvcPointerIdentity:
            raise TypeError("capture preparation custody requires D1DvcPointerIdentity")
        if type(self.authority) is not D1CampaignCaptureAuthority:
            raise TypeError("capture preparation custody requires campaign authority")
        if type(self.pointer_bytes) is not bytes or not self.pointer_bytes:
            raise ValueError("capture preparation custody pointer bytes must be nonempty")
        if hashlib.sha256(self.pointer_bytes).hexdigest() != self.pointer.pointer_sha256:
            raise ValueError("capture preparation custody pointer bytes differ from pointer SHA")
        if self.status is not D1CapturePreparationStatus.COMMIT_REQUIRED:
            raise ValueError("capture preparation custody status must remain COMMIT_REQUIRED")
        if self.schema_version != D1_CAPTURE_PREPARATION_CUSTODY_SCHEMA:
            raise ValueError("capture preparation custody schema is unsupported")
        _validate_aggregate_binding(self.aggregate_json, self.aggregate_sha256, self.pointer)

    @property
    def manifest(self) -> dict[str, object]:
        encoded = base64.b64encode(self.pointer_bytes).decode("ascii")
        return {
            "aggregate": {"json": self.aggregate_json, "sha256": self.aggregate_sha256},
            "campaign_authority": self.authority.manifest,
            "pointer": {
                "byte_count": len(self.pointer_bytes),
                "bytes_base64": encoded,
                "object_id": self.pointer.object_id,
                "path": self.pointer.pointer_path,
                "sha256": self.pointer.pointer_sha256,
            },
            "schema_version": self.schema_version,
            "status": self.status.value,
        }

    @property
    def record_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.manifest).encode()).hexdigest()


def capture_preparation_custody_from_preparation(
    preparation: D1HeterogeneousCapturePreparation,
    *,
    authority: D1CampaignCaptureAuthority,
) -> D1CapturePreparationCustodyRecord:
    if type(preparation) is not D1HeterogeneousCapturePreparation:
        raise TypeError("capture preparation custody requires heterogeneous preparation")
    aggregate_json = serialize_d1_capture_aggregate(preparation.aggregate)
    return D1CapturePreparationCustodyRecord(
        aggregate_json,
        preparation.aggregate.manifest_digest,
        preparation.pointer,
        preparation.pointer_bytes,
        authority,
    )


def serialize_d1_capture_preparation_custody(
    record: D1CapturePreparationCustodyRecord,
) -> str:
    if type(record) is not D1CapturePreparationCustodyRecord:
        raise TypeError("capture preparation serialization requires its nominal record")
    payload = _canonical(record.manifest)
    if len(payload.encode()) > _MAX_BYTES:
        _fail("D1 capture preparation custody exceeds its byte limit")
    return payload


def _object(value: object, *, label: str, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"D1 capture preparation {label} must be an object")
    item = cast(dict[str, object], value)
    if set(item) != keys:
        _fail(f"D1 capture preparation {label} keys must be exact")
    return item


def parse_d1_capture_preparation_custody(payload: str) -> D1CapturePreparationCustodyRecord:
    if type(payload) is not str or not payload or len(payload.encode()) > _MAX_BYTES:
        _fail("D1 capture preparation custody payload is empty or exceeds its byte limit")
    try:
        value: object = strict_loads(payload, fail=_fail)
    except D1CapturePreparationCustodyError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail("D1 capture preparation custody is invalid JSON", error)
    root = _object(value, label="root", keys=_ROOT_KEYS)
    if root["schema_version"] != D1_CAPTURE_PREPARATION_CUSTODY_SCHEMA:
        _fail("D1 capture preparation custody schema is unsupported")
    if root["status"] != D1CapturePreparationStatus.COMMIT_REQUIRED.value:
        _fail("D1 capture preparation custody status is unsupported")
    aggregate = _object(root["aggregate"], label="aggregate", keys={"json", "sha256"})
    authority = _object(
        root["campaign_authority"],
        label="campaign authority",
        keys=_AUTHORITY_KEYS,
    )
    pointer = _object(
        root["pointer"],
        label="pointer",
        keys={"byte_count", "bytes_base64", "object_id", "path", "sha256"},
    )
    try:
        encoded = pointer["bytes_base64"]
        if type(encoded) is not str:
            _fail("D1 capture preparation pointer bytes_base64 must be text")
        pointer_bytes = base64.b64decode(encoded, validate=True)
        if base64.b64encode(pointer_bytes).decode("ascii") != encoded:
            _fail("D1 capture preparation pointer base64 is noncanonical")
        byte_count = pointer["byte_count"]
        if type(byte_count) is not int or byte_count != len(pointer_bytes):
            _fail("D1 capture preparation pointer byte_count differs")
        record = D1CapturePreparationCustodyRecord(
            cast(str, aggregate["json"]),
            cast(str, aggregate["sha256"]),
            D1DvcPointerIdentity(
                cast(str, pointer["path"]),
                cast(str, pointer["sha256"]),
                cast(str, pointer["object_id"]),
            ),
            pointer_bytes,
            D1CampaignCaptureAuthority(
                campaign_digest=cast(str, authority["campaign_digest"]),
                registered_design_digest=cast(
                    str,
                    authority["registered_design_digest"],
                ),
                matrix_digest=cast(str, authority["matrix_digest"]),
                attempt_ledger_digest=cast(
                    str,
                    authority["attempt_ledger_digest"],
                ),
                lifecycle_prefix=cast(str, authority["lifecycle_prefix"]),
                lifecycle_event_count=cast(int, authority["lifecycle_event_count"]),
            ),
        )
    except D1CapturePreparationCustodyError:
        raise
    except (binascii.Error, TypeError, ValueError) as error:
        _fail(
            f"D1 capture preparation pointer, aggregate, or campaign authority is invalid: {error}",
            error,
        )
    if serialize_d1_capture_preparation_custody(record) != payload:
        _fail("D1 capture preparation custody JSON is noncanonical")
    return record


def hydrate_d1_capture_preparation_custody(
    record: D1CapturePreparationCustodyRecord,
    *,
    capture_root: Path,
) -> D1HeterogeneousCapturePreparation:
    if type(record) is not D1CapturePreparationCustodyRecord:
        raise TypeError("capture preparation hydration requires its nominal record")
    try:
        aggregate = hydrate_d1_capture_aggregate(
            record.aggregate_json,
            expected_manifest_digest=record.aggregate_sha256,
            capture_root=capture_root,
        )
        return D1HeterogeneousCapturePreparation(
            aggregate,
            record.pointer,
            record.pointer_bytes,
        )
    except (D1CaptureAggregateHydrationError, TypeError, ValueError) as error:
        _fail(f"D1 capture preparation hydrate failed: {error}", error)
