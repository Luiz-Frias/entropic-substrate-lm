"""Untrusted canonical control values for the external D1 analysis worker."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import NoReturn, SupportsIndex, cast

from entropic_substrate.analysis.d1_machine_analysis import D1ArmAliasMap
from entropic_substrate.analysis.d1_reader_source_closure import (
    installed_d1_reader_source_files,
)
from entropic_substrate.evidence.d1_capture_aggregate_codec import (
    D1CaptureAggregateDeclaration,
    parse_d1_capture_aggregate_declaration,
)
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1DvcBindingState,
)
from entropic_substrate.evidence.d1_capture_ports import D1DvcPointerIdentity
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_filing_receipts import (
    parse_alias_commitment,
    parse_alias_reveal,
)
from entropic_substrate.evidence.d1_types import D1_OPAQUE_ARM_ALIAS_IDS
from entropic_substrate.experiments.d1_primitives import D1Arm, canonical_digest
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    decision_contract_digest,
)
from entropic_substrate.experiments.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    D1RegisteredDesignDeclaration,
    parse_decision_contract,
    parse_registered_design_declaration,
    serialize_decision_contract,
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
    "D1ExternalAnalysisInputControl",
    "D1PreregistrationControl",
    "D1ReaderSourceEntry",
    "D1ReaderSourceManifestControl",
    "decode_d1_alias_application_control",
    "decode_d1_external_analysis_input_control",
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
_ALIASES = frozenset(D1_OPAQUE_ARM_ALIAS_IDS)
_INPUT_TOKEN = object()
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
D1_READER_SOURCE_MODULES = tuple(item.identity for item in installed_d1_reader_source_files())
_INPUT_SCHEMA = "d1-external-analysis-input-v1"
_INPUT_BASE64_FIELDS = (
    "alias_commitment_base64",
    "alias_reveal_base64",
    "capture_aggregate_base64",
    "capture_receipt_base64",
    "decision_contract_base64",
    "preregistration_control_base64",
    "reader_source_manifest_base64",
    "registered_design_base64",
)
_INPUT_FIELDS = frozenset(
    {
        *_INPUT_BASE64_FIELDS,
        "expected_capture_digest",
        "expected_design_digest",
        "expected_family_digest",
        "schema_version",
    }
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


@dataclass(frozen=True, slots=True, weakref_slot=True)
class D1AliasApplicationControl:
    commitment: D1AliasCommitmentControl
    reveal: D1AliasRevealControl = field(repr=False)
    aliases: D1ArmAliasMap
    commitment_sha256: str
    reveal_sha256: str
    _attestation: _AliasApplicationAttestation | None = field(
        default=None,
        repr=False,
        compare=False,
        init=False,
    )

    def __post_init__(self) -> None:
        raise TypeError(
            "D1AliasApplicationControl is factory-only; use decode_d1_alias_application_control"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("D1AliasApplicationControl cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("D1AliasApplicationControl cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("D1AliasApplicationControl cannot be copied")

    def require_factory_attested(self) -> None:
        if not _is_minted_alias_application(self):
            raise TypeError("factory-attested alias control was not minted")
        attestation = getattr(self, "_attestation", None)
        if type(attestation) is not _AliasApplicationAttestation:
            raise TypeError("factory-attested alias control marker is absent")
        if attestation.owner_ref() is not self:
            raise TypeError("factory-attested alias control marker belongs to another instance")
        _validate_alias_application_values(
            commitment=self.commitment,
            reveal=self.reveal,
            aliases=self.aliases,
            commitment_sha256=self.commitment_sha256,
            reveal_sha256=self.reveal_sha256,
        )


class _AliasApplicationAttestation:
    __slots__ = ("owner_ref",)

    owner_ref: weakref.ReferenceType[D1AliasApplicationControl]

    def __init__(self, owner: D1AliasApplicationControl) -> None:
        self.owner_ref = weakref.ref(owner)


_ALIAS_APPLICATION_LOCK = threading.Lock()
_MINTED_ALIAS_APPLICATIONS: dict[
    int,
    weakref.ReferenceType[D1AliasApplicationControl],
] = {}


def _record_minted_alias_application(control: D1AliasApplicationControl) -> None:
    identifier = id(control)

    def discard(reference: weakref.ReferenceType[D1AliasApplicationControl]) -> None:
        with _ALIAS_APPLICATION_LOCK:
            if _MINTED_ALIAS_APPLICATIONS.get(identifier) is reference:
                _MINTED_ALIAS_APPLICATIONS.pop(identifier, None)

    reference = weakref.ref(control, discard)
    with _ALIAS_APPLICATION_LOCK:
        _MINTED_ALIAS_APPLICATIONS[identifier] = reference


def _is_minted_alias_application(control: D1AliasApplicationControl) -> bool:
    with _ALIAS_APPLICATION_LOCK:
        reference = _MINTED_ALIAS_APPLICATIONS.get(id(control))
        return reference is not None and reference() is control


def _validate_alias_application_values(
    *,
    commitment: D1AliasCommitmentControl,
    reveal: D1AliasRevealControl,
    aliases: D1ArmAliasMap,
    commitment_sha256: str,
    reveal_sha256: str,
) -> None:
    if (
        type(commitment) is not D1AliasCommitmentControl
        or type(reveal) is not D1AliasRevealControl
        or type(aliases) is not D1ArmAliasMap
    ):
        raise TypeError("verified alias application requires typed control values")
    if hashlib.sha256(commitment.canonical_bytes).hexdigest() != commitment_sha256:
        raise ValueError("verified alias application commitment digest differs")
    if hashlib.sha256(reveal.canonical_bytes).hexdigest() != reveal_sha256:
        raise ValueError("verified alias application reveal digest differs")
    if commitment.reveal_sha256 != reveal_sha256:
        raise ValueError("verified alias application commitment does not bind reveal")
    expected = {D1Arm(arm): alias for alias, arm in reveal.mapping.items()}
    if dict(aliases.aliases) != expected:
        raise ValueError("verified alias application mapping differs from reveal")
    if aliases.sealed_commitment != commitment_sha256:
        raise ValueError("verified alias application sealed commitment differs")


def _mint_alias_application(
    *,
    commitment: D1AliasCommitmentControl,
    reveal: D1AliasRevealControl,
    aliases: D1ArmAliasMap,
    commitment_sha256: str,
    reveal_sha256: str,
) -> D1AliasApplicationControl:
    _validate_alias_application_values(
        commitment=commitment,
        reveal=reveal,
        aliases=aliases,
        commitment_sha256=commitment_sha256,
        reveal_sha256=reveal_sha256,
    )
    control = object.__new__(D1AliasApplicationControl)
    object.__setattr__(control, "commitment", commitment)
    object.__setattr__(control, "reveal", reveal)
    object.__setattr__(control, "aliases", aliases)
    object.__setattr__(control, "commitment_sha256", commitment_sha256)
    object.__setattr__(control, "reveal_sha256", reveal_sha256)
    object.__setattr__(control, "_attestation", _AliasApplicationAttestation(control))
    _record_minted_alias_application(control)
    return control


def _require_alias_application(value: object) -> None:
    if type(value) is not D1AliasApplicationControl:
        raise TypeError("verified external input requires factory-attested alias control")
    value.require_factory_attested()


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


@dataclass(frozen=True, slots=True)
class D1ExternalAnalysisInputControl:
    canonical_bytes: bytes = field(repr=False)
    input_manifest_sha256: str
    expected_family_digest: str
    decision_contract: D1DecisionContract
    design: D1RegisteredDesignDeclaration
    aggregate: D1CaptureAggregateDeclaration
    capture_receipt: D1CaptureReceiptControl
    alias_application: D1AliasApplicationControl = field(repr=False)
    preregistration: D1PreregistrationControl
    reader_source: D1ReaderSourceManifestControl
    _factory_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _INPUT_TOKEN:
            raise TypeError(
                "D1ExternalAnalysisInputControl is factory-only; "
                "use decode_d1_external_analysis_input_control"
            )
        _require_alias_application(self.alias_application)
        if hashlib.sha256(self.canonical_bytes).hexdigest() != self.input_manifest_sha256:
            raise ValueError("verified external input manifest digest differs")
        manifest, nested = _extract_input_manifest(self.canonical_bytes)
        expected_nested = {
            "alias_commitment_base64": self.alias_application.commitment.canonical_bytes,
            "alias_reveal_base64": self.alias_application.reveal.canonical_bytes,
            "capture_aggregate_base64": self.aggregate.canonical_json.encode("utf-8"),
            "capture_receipt_base64": self.capture_receipt.canonical_bytes,
            "decision_contract_base64": serialize_decision_contract(self.decision_contract).encode(
                "utf-8"
            ),
            "preregistration_control_base64": self.preregistration.canonical_bytes,
            "reader_source_manifest_base64": self.reader_source.canonical_bytes,
            "registered_design_base64": _canonical(self.design.manifest),
        }
        if nested != expected_nested:
            raise ValueError("verified nested control differs from canonical input manifest")
        if (
            manifest["expected_family_digest"] != self.expected_family_digest
            or manifest["expected_design_digest"] != self.design.design_digest
            or manifest["expected_capture_digest"] != self.aggregate.manifest_digest
        ):
            raise ValueError("verified identity differs from canonical input manifest")
        _validate_input_relationships(
            expected_family_digest=self.expected_family_digest,
            decision_contract=self.decision_contract,
            design=self.design,
            aggregate=self.aggregate,
            capture_receipt=self.capture_receipt,
            preregistration=self.preregistration,
            reader_source=self.reader_source,
        )


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
    return _mint_alias_application(
        commitment=commitment,
        reveal=reveal,
        aliases=aliases,
        commitment_sha256=commitment_sha256,
        reveal_sha256=reveal_sha256,
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


def _base64_payload(value: object, *, label: str) -> bytes:
    if type(value) is not str or not value:
        _fail(f"{label} must be nonempty canonical base64 text")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        _fail(f"{label} must be canonical base64", error)
    if not payload or base64.b64encode(payload).decode("ascii") != value:
        _fail(f"{label} must be nonempty canonical base64")
    return payload


def _extract_input_manifest(
    payload: bytes,
) -> tuple[dict[str, object], dict[str, bytes]]:
    manifest = _decode(
        payload,
        fields=_INPUT_FIELDS,
        schema=_INPUT_SCHEMA,
        label="external analysis input",
    )
    nested = {
        field: _base64_payload(manifest[field], label=f"external input {field}")
        for field in _INPUT_BASE64_FIELDS
    }
    return manifest, nested


def _utf8(payload: bytes, *, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeError as error:
        _fail(f"{label} must contain UTF-8 canonical bytes", error)


def _validate_input_relationships(
    *,
    expected_family_digest: str,
    decision_contract: D1DecisionContract,
    design: D1RegisteredDesignDeclaration,
    aggregate: D1CaptureAggregateDeclaration,
    capture_receipt: D1CaptureReceiptControl,
    preregistration: D1PreregistrationControl,
    reader_source: D1ReaderSourceManifestControl,
) -> None:
    if (
        type(decision_contract) is not D1DecisionContract
        or type(design) is not D1RegisteredDesignDeclaration
        or type(aggregate) is not D1CaptureAggregateDeclaration
        or type(capture_receipt) is not D1CaptureReceiptControl
        or type(preregistration) is not D1PreregistrationControl
        or type(reader_source) is not D1ReaderSourceManifestControl
    ):
        raise TypeError("verified external input requires typed nested controls")
    if design.family_identity.family_digest != expected_family_digest:
        _fail("external input expected family digest differs from registered design")
    if decision_contract.family_identity != design.family_identity:
        _fail("external input decision contract family differs from registered design")
    if decision_contract_digest(decision_contract) != design.decision_contract_digest:
        _fail("external input decision contract digest differs from registered design")
    if aggregate.roster.family_digest != expected_family_digest:
        _fail("external input aggregate roster family differs from registered design")
    if (
        aggregate.roster.preregistration_path != design.preregistration.path
        or aggregate.roster.preregistration_commit != design.preregistration.commit
    ):
        _fail("external input aggregate preregistration differs from registered design")
    observed_seeds = {item.seed for item in aggregate.roster.expected_confirmatory}
    if observed_seeds != set(design.registered_seeds):
        _fail("external input aggregate seed roster differs from registered design")
    for item in aggregate.roster.expected_confirmatory:
        if (
            item.construction_digest
            != design.family_identity.arm_construction_digests[item.arm.value]
        ):
            _fail("external input aggregate arm construction differs from registered design")
    if capture_receipt.aggregate_digest != aggregate.manifest_digest:
        _fail("external input capture receipt differs from aggregate declaration")
    if (
        preregistration.path != design.preregistration.path
        or preregistration.commit != design.preregistration.commit
        or preregistration.sha256 != design.preregistration.blob_sha256
    ):
        _fail("external input preregistration control differs from registered design")
    registered_seeds = decision_contract.registered_constants.get("seeds")
    if registered_seeds != design.registered_seeds:
        _fail("external input decision constants differ from registered seed roster")


def decode_d1_external_analysis_input_control(
    payload: bytes,
    *,
    preregistration_evidence: D1PreregistrationByteEvidence,
) -> D1ExternalAnalysisInputControl:
    if type(preregistration_evidence) is not D1PreregistrationByteEvidence:
        raise TypeError("external analysis input requires preregistration byte evidence")
    manifest, nested = _extract_input_manifest(payload)
    expected_family_digest = _sha256(
        manifest["expected_family_digest"], label="external input expected_family_digest"
    )
    _sha256(manifest["expected_design_digest"], label="external input expected_design_digest")
    _sha256(manifest["expected_capture_digest"], label="external input expected_capture_digest")
    try:
        decision_contract = parse_decision_contract(
            _utf8(nested["decision_contract_base64"], label="decision contract")
        )
        design = parse_registered_design_declaration(
            _utf8(nested["registered_design_base64"], label="registered design")
        )
        aggregate = parse_d1_capture_aggregate_declaration(
            _utf8(nested["capture_aggregate_base64"], label="capture aggregate")
        )
        alias_application = decode_d1_alias_application_control(
            nested["alias_commitment_base64"], nested["alias_reveal_base64"]
        )
        preregistration = parse_d1_preregistration_control(
            nested["preregistration_control_base64"],
            design=design,
            evidence=preregistration_evidence,
        )
        reader_source = parse_d1_reader_source_manifest(nested["reader_source_manifest_base64"])
        capture_receipt = parse_d1_capture_receipt_control(
            nested["capture_receipt_base64"], aggregate=aggregate
        )
    except D1ExternalAnalysisControlError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"external analysis nested control is invalid: {error}", error)
    try:
        return D1ExternalAnalysisInputControl(
            payload,
            hashlib.sha256(payload).hexdigest(),
            expected_family_digest,
            decision_contract,
            design,
            aggregate,
            capture_receipt,
            alias_application,
            preregistration,
            reader_source,
            _INPUT_TOKEN,
        )
    except D1ExternalAnalysisControlError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"external analysis input validation failed: {error}", error)
