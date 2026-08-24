"""Untrusted canonical control values for the external D1 analysis worker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import NoReturn, cast

from entropic_substrate.analysis.d1_machine_analysis import D1ArmAliasMap
from entropic_substrate.evidence.d1_capture_aggregate_codec import (
    D1CaptureAggregateDeclaration,
)
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1DvcBindingState,
)
from entropic_substrate.evidence.d1_capture_ports import D1DvcPointerIdentity
from entropic_substrate.evidence.d1_filing_receipts import (
    parse_alias_commitment,
    parse_alias_reveal,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, canonical_digest
from entropic_substrate.experiments.d1_registered_design import D1PreregistrationArtifact
from entropic_substrate.experiments.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    D1RegisteredDesignDeclaration,
)

__all__ = [
    "D1_COMPLETE_READER_IDENTITY",
    "D1_READER_SOURCE_MODULES",
    "D1_TERMINAL_DEFECT_READER_IDENTITY",
    "D1AliasApplicationControl",
    "D1AliasCommitmentControl",
    "D1AliasRevealControl",
    "D1CaptureReceiptControl",
    "D1ExternalAnalysisControlError",
    "D1PreregistrationControl",
    "D1ReaderSourceEntry",
    "D1ReaderSourceManifestControl",
    "decode_d1_alias_application_control",
    "derive_d1_analysis_code_digest",
    "parse_d1_alias_commitment_control",
    "parse_d1_alias_reveal_control",
    "parse_d1_capture_receipt_control",
    "parse_d1_preregistration_control",
    "parse_d1_reader_source_manifest",
]

_COMMITMENT_SCHEMA = "d1-alias-commitment-v1"
_REVEAL_SCHEMA = "d1-alias-reveal-v1"
_COMMITMENT_FIELDS = frozenset({"reveal_sha256", "schema_version"})
_REVEAL_FIELDS = frozenset({"mapping", "nonce", "schema_version"})
_ALIASES = frozenset({"A", "B", "C", "D"})
_APPLICATION_TOKEN = object()
_CAPTURE_RECEIPT_SCHEMA = "d1-heterogeneous-capture-receipt-v1"
_CAPTURE_RECEIPT_FIELDS = frozenset(
    {"aggregate_digest", "analysis_ready", "pointer_commit", "pointer_identity", "schema_version"}
)
_POINTER_FIELDS = frozenset({"object_id", "pointer_path", "pointer_sha256"})
_PREREGISTRATION_SCHEMA = "d1-preregistration-control-v1"
_PREREGISTRATION_FIELDS = frozenset({"commit", "path", "schema_version", "sha256", "size_bytes"})
_READER_SCHEMA = "d1-reader-source-manifest-v1"
_READER_CODE_SCHEMA = "d1-reader-analysis-code-v1"
_READER_FIELDS = frozenset(
    {
        "analysis_code_digest",
        "complete_reader",
        "modules",
        "schema_version",
        "terminal_defect_reader",
    }
)
_READER_MODULE_FIELDS = frozenset({"identity", "path", "sha256"})

D1_COMPLETE_READER_IDENTITY = "entropic_substrate.telemetry.read_d1_run_evidence"
D1_TERMINAL_DEFECT_READER_IDENTITY = "entropic_substrate.telemetry.read_d1_terminal_defect"
D1_READER_SOURCE_MODULES = (
    "entropic_substrate.telemetry",
    "entropic_substrate.evidence.d1_telemetry",
    "entropic_substrate.evidence.d1_telemetry_query",
    "entropic_substrate.evidence.d1_terminal_defect_query",
    "entropic_substrate.evidence.d1_journal",
    "entropic_substrate.evidence.d1_journal_values",
    "entropic_substrate.evidence.d1_telemetry_measurements",
    "entropic_substrate.evidence.d1_telemetry_payloads",
    "entropic_substrate.evidence.d1_telemetry_values",
    "entropic_substrate.evidence.resource_work",
    "entropic_substrate.evidence.resource_work_codec",
    "entropic_substrate.evidence.resource_provenance",
    "entropic_substrate.evidence.resource_provenance_codec",
    "entropic_substrate.evidence.resource_ledger",
    "entropic_substrate.evidence.resource_ledger_codec",
    "entropic_substrate.d1_numerical_defect",
    "entropic_substrate.data.batching",
    "entropic_substrate.experiments.d1_primitives",
    "entropic_substrate.experiments.d1_plan",
    "entropic_substrate.experiments.d1_execution",
    "entropic_substrate.d1_calibration",
    "entropic_substrate.d1_training_step",
)


class D1ExternalAnalysisControlError(ValueError):
    """External analysis control bytes are malformed, permissive, or inconsistent."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1ExternalAnalysisControlError(message)
    raise D1ExternalAnalysisControlError(message) from error


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate external analysis control key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite external analysis control constant {value!r}")


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode(
    payload: bytes,
    *,
    fields: frozenset[str],
    schema: str,
    label: str,
) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError(f"{label} payload must be bytes")
    try:
        value: object = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except D1ExternalAnalysisControlError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"{label} payload is invalid UTF-8 JSON", error)
    if type(value) is not dict:
        _fail(f"{label} payload must have an object root")
    manifest = cast(dict[str, object], value)
    if frozenset(manifest) != fields:
        _fail(f"{label} fields must be exact")
    if manifest["schema_version"] != schema:
        _fail(f"{label} schema_version is unsupported")
    if _canonical(manifest) != payload:
        _fail(f"{label} payload must use canonical JSON bytes")
    return manifest


def _sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be lowercase SHA-256")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be a nonempty trimmed string")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


def _commit(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if (
        not 7 <= len(text) <= 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        _fail(f"{label} must be lowercase hexadecimal")
    return text


@dataclass(frozen=True, slots=True)
class D1AliasCommitmentControl:
    canonical_bytes: bytes = field(repr=False)
    reveal_sha256: str


@dataclass(frozen=True, slots=True)
class D1AliasRevealControl:
    canonical_bytes: bytes = field(repr=False)
    mapping: Mapping[str, str]
    nonce: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class D1AliasApplicationControl:
    commitment: D1AliasCommitmentControl
    reveal: D1AliasRevealControl = field(repr=False)
    aliases: D1ArmAliasMap
    commitment_sha256: str
    reveal_sha256: str
    _factory_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _APPLICATION_TOKEN:
            raise TypeError(
                "D1AliasApplicationControl is factory-only; use decode_d1_alias_application_control"
            )
        if (
            type(self.commitment) is not D1AliasCommitmentControl
            or type(self.reveal) is not D1AliasRevealControl
            or type(self.aliases) is not D1ArmAliasMap
        ):
            raise TypeError("verified alias application requires typed control values")
        if hashlib.sha256(self.commitment.canonical_bytes).hexdigest() != self.commitment_sha256:
            raise ValueError("verified alias application commitment digest differs")
        if hashlib.sha256(self.reveal.canonical_bytes).hexdigest() != self.reveal_sha256:
            raise ValueError("verified alias application reveal digest differs")
        if self.commitment.reveal_sha256 != self.reveal_sha256:
            raise ValueError("verified alias application commitment does not bind reveal")
        expected = {D1Arm(arm): alias for alias, arm in self.reveal.mapping.items()}
        if dict(self.aliases.aliases) != expected:
            raise ValueError("verified alias application mapping differs from reveal")
        if self.aliases.sealed_commitment != self.commitment_sha256:
            raise ValueError("verified alias application sealed commitment differs")


@dataclass(frozen=True, slots=True)
class D1CaptureReceiptControl:
    canonical_bytes: bytes = field(repr=False)
    aggregate_digest: str
    analysis_ready: bool
    pointer_commit: str
    pointer: D1DvcPointerIdentity


@dataclass(frozen=True, slots=True)
class D1PreregistrationControl:
    canonical_bytes: bytes = field(repr=False)
    path: str
    commit: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class D1ReaderSourceEntry:
    identity: str
    path: str
    sha256: str

    @property
    def manifest(self) -> dict[str, str]:
        return {"identity": self.identity, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class D1ReaderSourceManifestControl:
    canonical_bytes: bytes = field(repr=False)
    modules: tuple[D1ReaderSourceEntry, ...]
    complete_reader: str
    terminal_defect_reader: str
    analysis_code_digest: str


def parse_d1_alias_commitment_control(payload: bytes) -> D1AliasCommitmentControl:
    manifest = _decode(
        payload,
        fields=_COMMITMENT_FIELDS,
        schema=_COMMITMENT_SCHEMA,
        label="alias commitment",
    )
    try:
        parsed = parse_alias_commitment(payload)
    except (TypeError, ValueError) as error:
        _fail(f"alias commitment is invalid: {error}", error)
    reveal_sha256 = _sha256(manifest["reveal_sha256"], label="alias commitment reveal_sha256")
    if parsed.reveal_sha256 != reveal_sha256:
        _fail("alias commitment parser disagrees with canonical bytes")
    return D1AliasCommitmentControl(payload, reveal_sha256)


def parse_d1_alias_reveal_control(payload: bytes) -> D1AliasRevealControl:
    _decode(
        payload,
        fields=_REVEAL_FIELDS,
        schema=_REVEAL_SCHEMA,
        label="alias reveal",
    )
    try:
        parsed = parse_alias_reveal(payload)
    except (TypeError, ValueError) as error:
        _fail(f"alias reveal is invalid: {error}", error)
    nonce = parsed.nonce
    if (
        len(nonce) != 64
        or nonce != nonce.lower()
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        _fail("alias reveal nonce must be exactly 32 lowercase hexadecimal bytes")
    mapping = dict(parsed.mapping)
    if (
        frozenset(mapping) != _ALIASES
        or set(mapping.values()) != {arm.value for arm in D1Arm}
        or len(set(mapping.values())) != len(D1Arm)
    ):
        _fail("alias reveal mapping must be an exact A/B/C/D to D1-arm bijection")
    return D1AliasRevealControl(
        payload,
        MappingProxyType(mapping),
        bytes.fromhex(nonce),
    )


def decode_d1_alias_application_control(
    commitment_payload: bytes,
    reveal_payload: bytes,
) -> D1AliasApplicationControl:
    if type(commitment_payload) is not bytes or type(reveal_payload) is not bytes:
        raise TypeError("alias application control requires commitment and reveal bytes")
    commitment = parse_d1_alias_commitment_control(commitment_payload)
    reveal = parse_d1_alias_reveal_control(reveal_payload)
    reveal_sha256 = hashlib.sha256(reveal_payload).hexdigest()
    if commitment.reveal_sha256 != reveal_sha256:
        _fail("alias commitment reveal_sha256 does not bind exact private reveal bytes")
    commitment_sha256 = hashlib.sha256(commitment_payload).hexdigest()
    aliases = D1ArmAliasMap(
        aliases={D1Arm(arm): alias for alias, arm in reveal.mapping.items()},
        sealed_commitment=commitment_sha256,
    )
    return D1AliasApplicationControl(
        commitment,
        reveal,
        aliases,
        commitment_sha256,
        reveal_sha256,
        _APPLICATION_TOKEN,
    )


def _aggregate_artifacts(
    aggregate: D1CaptureAggregateDeclaration,
) -> tuple[D1ArtifactIdentity, ...]:
    return (
        tuple(artifact for item in aggregate.complete_inputs for artifact in item.artifacts)
        + tuple(
            artifact for item in aggregate.terminal_defect_inputs for artifact in item.artifacts
        )
        + tuple(artifact for item in aggregate.attempt_evidence for artifact in item.artifacts)
    )


def parse_d1_capture_receipt_control(
    payload: bytes,
    *,
    aggregate: D1CaptureAggregateDeclaration,
) -> D1CaptureReceiptControl:
    if type(aggregate) is not D1CaptureAggregateDeclaration:
        raise TypeError("capture receipt control requires D1CaptureAggregateDeclaration")
    manifest = _decode(
        payload,
        fields=_CAPTURE_RECEIPT_FIELDS,
        schema=_CAPTURE_RECEIPT_SCHEMA,
        label="capture receipt",
    )
    if manifest["analysis_ready"] is not True:
        _fail("capture receipt analysis_ready must be exact true")
    aggregate_digest = _sha256(
        manifest["aggregate_digest"], label="capture receipt aggregate_digest"
    )
    if aggregate_digest != aggregate.manifest_digest:
        _fail("capture receipt aggregate_digest differs from aggregate declaration")
    pointer_manifest = manifest["pointer_identity"]
    if type(pointer_manifest) is not dict:
        _fail("capture receipt pointer_identity must be an object")
    pointer_values = cast(dict[str, object], pointer_manifest)
    if frozenset(pointer_values) != _POINTER_FIELDS:
        _fail("capture receipt pointer_identity fields must be exact")
    try:
        pointer = D1DvcPointerIdentity(
            pointer_path=_text(
                pointer_values["pointer_path"], label="capture receipt pointer_path"
            ),
            pointer_sha256=_sha256(
                pointer_values["pointer_sha256"], label="capture receipt pointer_sha256"
            ),
            object_id=_text(pointer_values["object_id"], label="capture receipt object_id"),
        )
    except (TypeError, ValueError) as error:
        _fail(f"capture receipt pointer identity is invalid: {error}", error)
    artifacts = _aggregate_artifacts(aggregate)
    if not artifacts:
        _fail("capture receipt aggregate has no declared artifacts")
    if any(
        artifact.dvc.state is not D1DvcBindingState.BOUND
        or artifact.dvc.pointer_path != pointer.pointer_path
        or artifact.dvc.pointer_sha256 != pointer.pointer_sha256
        or artifact.dvc.object_id != pointer.object_id
        for artifact in artifacts
    ):
        _fail("capture receipt pointer identity does not bind every aggregate artifact")
    return D1CaptureReceiptControl(
        payload,
        aggregate_digest,
        True,
        _commit(manifest["pointer_commit"], label="capture receipt pointer_commit"),
        pointer,
    )


def parse_d1_preregistration_control(
    payload: bytes,
    *,
    design: D1RegisteredDesignDeclaration,
    evidence: D1PreregistrationByteEvidence,
) -> D1PreregistrationControl:
    if type(design) is not D1RegisteredDesignDeclaration:
        raise TypeError("preregistration control requires D1RegisteredDesignDeclaration")
    if type(evidence) is not D1PreregistrationByteEvidence:
        raise TypeError("preregistration control requires D1PreregistrationByteEvidence")
    manifest = _decode(
        payload,
        fields=_PREREGISTRATION_FIELDS,
        schema=_PREREGISTRATION_SCHEMA,
        label="preregistration control",
    )
    path = _text(manifest["path"], label="preregistration control path")
    commit = _commit(manifest["commit"], label="preregistration control commit")
    size_bytes = _positive_integer(
        manifest["size_bytes"], label="preregistration control size_bytes"
    )
    sha256 = _sha256(manifest["sha256"], label="preregistration control sha256")
    try:
        identity = D1PreregistrationArtifact(path=path, blob_sha256=sha256, commit=commit)
    except (TypeError, ValueError) as error:
        _fail(f"preregistration control identity is invalid: {error}", error)
    if identity != design.preregistration:
        _fail("preregistration control identity differs from registered design")
    if size_bytes != evidence.size_bytes or sha256 != evidence.sha256:
        _fail("preregistration control size or SHA differs from measured raw bytes")
    return D1PreregistrationControl(payload, path, commit, size_bytes, sha256)


def derive_d1_analysis_code_digest(modules: object) -> str:
    if type(modules) is not list:
        raise TypeError("analysis code digest requires an ordered JSON module list")
    return canonical_digest(
        {
            "modules": modules,
            "schema_version": _READER_CODE_SCHEMA,
        }
    )


def _reader_path(identity: str, value: object) -> str:
    path = _text(value, label=f"reader module {identity} path")
    parsed = PurePosixPath(path)
    expected = identity.replace(".", "/") + ".py"
    if (
        parsed.is_absolute()
        or "\\" in path
        or ".." in parsed.parts
        or parsed.as_posix() != path
        or path != expected
    ):
        _fail(f"reader module {identity} path must be normalized canonical source path")
    return path


def parse_d1_reader_source_manifest(payload: bytes) -> D1ReaderSourceManifestControl:
    manifest = _decode(
        payload,
        fields=_READER_FIELDS,
        schema=_READER_SCHEMA,
        label="reader source manifest",
    )
    raw_modules = manifest["modules"]
    if type(raw_modules) is not list:
        _fail("reader source manifest modules must be an ordered JSON array")
    entries: list[D1ReaderSourceEntry] = []
    for index, raw in enumerate(cast(list[object], raw_modules)):
        if type(raw) is not dict:
            _fail(f"reader module[{index}] must be an object")
        item = cast(dict[str, object], raw)
        if frozenset(item) != _READER_MODULE_FIELDS:
            _fail(f"reader module[{index}] fields must be exact")
        identity = _text(item["identity"], label=f"reader module[{index}] identity")
        entries.append(
            D1ReaderSourceEntry(
                identity=identity,
                path=_reader_path(identity, item["path"]),
                sha256=_sha256(item["sha256"], label=f"reader module[{index}] sha256"),
            )
        )
    identities = tuple(entry.identity for entry in entries)
    if identities != D1_READER_SOURCE_MODULES:
        _fail("reader source module identities must equal the exact required ordered set")
    paths = tuple(PurePosixPath(entry.path) for entry in entries)
    if len(set(paths)) != len(paths) or any(
        left in right.parents or right in left.parents
        for index, left in enumerate(paths)
        for right in paths[index + 1 :]
    ):
        _fail("reader source paths must be unique and non-overlapping")
    complete_reader = _text(manifest["complete_reader"], label="reader source complete_reader")
    if complete_reader != D1_COMPLETE_READER_IDENTITY:
        _fail("reader source complete_reader identity is unsupported")
    terminal_reader = _text(
        manifest["terminal_defect_reader"],
        label="reader source terminal_defect_reader",
    )
    if terminal_reader != D1_TERMINAL_DEFECT_READER_IDENTITY:
        _fail("reader source terminal_defect_reader identity is unsupported")
    analysis_code_digest = _sha256(
        manifest["analysis_code_digest"], label="reader source analysis_code_digest"
    )
    derived = derive_d1_analysis_code_digest([entry.manifest for entry in entries])
    if analysis_code_digest != derived:
        _fail("reader source analysis_code_digest differs from ordered canonical module list")
    return D1ReaderSourceManifestControl(
        payload,
        tuple(entries),
        complete_reader,
        terminal_reader,
        analysis_code_digest,
    )
