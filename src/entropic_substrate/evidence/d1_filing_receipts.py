"""Typed semantic receipts and read-only ports for D1 filing verification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast


@dataclass(frozen=True, slots=True)
class D1EvidenceGap:
    artifact_role: str
    needed_field: str
    required_measurement: str


class D1EvidenceGapError(ValueError):
    """A required historical measurement is absent, not inferred."""

    def __init__(self, gap: D1EvidenceGap) -> None:
        self.gap = gap
        super().__init__(
            f"D1 evidence gap in {gap.artifact_role}.{gap.needed_field}: "
            f"requires {gap.required_measurement}"
        )


class D1FilingGraphError(ValueError):
    """A present D1 filing edge is malformed or contradicts another edge."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"D1 filing graph {code} at {path}: {detail}")


class D1FilingArtifactPort(Protocol):
    """Read exact current bytes without granting mutation authority."""

    def read_bytes(self, path: str) -> bytes | None: ...


class D1FilingGitPort(Protocol):
    """Read-only Git facts; implementations must never update repository state."""

    def read_file(self, revision: str, path: str) -> bytes | None: ...

    def is_ancestor(self, ancestor: str, descendant: str) -> bool: ...


def _commit(value: str, *, label: str) -> None:
    if (
        type(value) is not str
        or not 7 <= len(value) <= 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a 7-to-64 character lowercase Git commit")


@dataclass(frozen=True, slots=True)
class D1FilingRevisions:
    capture: str
    analysis: str
    alias_commitment: str
    review: str
    alias_reveal: str
    filing: str

    def __post_init__(self) -> None:
        for name, value in zip(
            ("capture", "analysis", "alias_commitment", "review", "alias_reveal", "filing"),
            self.ordered,
            strict=True,
        ):
            _commit(value, label=f"D1 {name} revision")
        if len(set(self.ordered)) != len(self.ordered):
            raise ValueError("D1 filing revisions must be distinct")

    @property
    def ordered(self) -> tuple[str, ...]:
        return (
            self.capture,
            self.analysis,
            self.alias_commitment,
            self.review,
            self.alias_reveal,
            self.filing,
        )


@dataclass(frozen=True, slots=True)
class D1AnalysisReceipt:
    capture_manifest_sha256: str
    preregistration_sha256: str
    reader_sha256: str
    isolation_attestation_sha256: str
    output_sha256: str


@dataclass(frozen=True, slots=True)
class D1AliasCommitment:
    reveal_sha256: str


@dataclass(frozen=True, slots=True)
class D1AliasReveal:
    mapping: Mapping[str, str]
    nonce: str


@dataclass(frozen=True, slots=True)
class D1ReviewReceipt:
    packet_sha256: str
    prompt_sha256: str
    report_sha256: str
    model_id: str
    provider: str
    round: int
    major_finding_ids: tuple[str, ...]


def _gap(role: str, field: str) -> D1EvidenceGapError:
    measurements = {
        "reader_sha256": "the exact analysis reader file digest",
        "isolation_attestation_sha256": "the fresh-session isolation attestation digest",
        "model_id": "the reviewer model identity",
        "provider": "the reviewer provider identity",
        "round": "the reviewer execution round",
        "major_finding_ids": "the report MAJOR finding identifiers",
    }
    return D1EvidenceGapError(
        D1EvidenceGap(role, field, measurements.get(field, f"the exact {field} measurement"))
    )


def _document(payload: bytes, *, role: str, fields: frozenset[str]) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D1FilingGraphError(code="invalid_json", path=role, detail=str(exc)) from exc
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise D1FilingGraphError(code="invalid_document", path=role, detail="must be an object")
    data = cast(Mapping[str, object], value)
    missing = sorted(fields - set(data))
    if missing:
        raise _gap(role, missing[0])
    unknown = sorted(set(data) - fields)
    if unknown:
        raise D1FilingGraphError(
            code="unknown_field", path=role, detail=f"unknown fields: {', '.join(unknown)}"
        )
    return data


def _schema(data: Mapping[str, object], expected: str, *, role: str) -> None:
    if data["schema_version"] != expected:
        raise D1FilingGraphError(
            code="schema_version", path=f"{role}.schema_version", detail=f"expected {expected}"
        )


def _text(value: object, *, path: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise D1FilingGraphError(code="invalid_text", path=path, detail="must be nonempty")
    return value


def _sha256(value: object, *, path: str) -> str:
    text = _text(value, path=path)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise D1FilingGraphError(
            code="invalid_sha256", path=path, detail="must be lowercase SHA-256"
        )
    return text


def parse_analysis_receipt(payload: bytes) -> D1AnalysisReceipt:
    role = "analysis_receipt"
    fields = frozenset(
        {
            "schema_version",
            "capture_manifest_sha256",
            "preregistration_sha256",
            "reader_sha256",
            "isolation_attestation_sha256",
            "output_sha256",
        }
    )
    data = _document(payload, role=role, fields=fields)
    _schema(data, "d1-analysis-receipt-v1", role=role)
    return D1AnalysisReceipt(
        capture_manifest_sha256=_sha256(
            data["capture_manifest_sha256"], path=f"{role}.capture_manifest_sha256"
        ),
        preregistration_sha256=_sha256(
            data["preregistration_sha256"], path=f"{role}.preregistration_sha256"
        ),
        reader_sha256=_sha256(data["reader_sha256"], path=f"{role}.reader_sha256"),
        isolation_attestation_sha256=_sha256(
            data["isolation_attestation_sha256"],
            path=f"{role}.isolation_attestation_sha256",
        ),
        output_sha256=_sha256(data["output_sha256"], path=f"{role}.output_sha256"),
    )


def parse_alias_commitment(payload: bytes) -> D1AliasCommitment:
    role = "alias_commitment"
    data = _document(payload, role=role, fields=frozenset({"schema_version", "reveal_sha256"}))
    _schema(data, "d1-alias-commitment-v1", role=role)
    return D1AliasCommitment(_sha256(data["reveal_sha256"], path=f"{role}.reveal_sha256"))


def parse_alias_reveal(payload: bytes) -> D1AliasReveal:
    role = "alias_reveal"
    data = _document(payload, role=role, fields=frozenset({"schema_version", "mapping", "nonce"}))
    _schema(data, "d1-alias-reveal-v1", role=role)
    mapping = data["mapping"]
    if not isinstance(mapping, Mapping) or any(
        type(key) is not str or type(value) is not str for key, value in mapping.items()
    ):
        raise D1FilingGraphError(
            code="invalid_mapping", path=f"{role}.mapping", detail="must map strings"
        )
    return D1AliasReveal(
        MappingProxyType(dict(cast(Mapping[str, str], mapping))),
        _text(data["nonce"], path=f"{role}.nonce"),
    )


def parse_review_receipt(payload: bytes) -> D1ReviewReceipt:
    role = "reviewer_execution_receipt"
    fields = frozenset(
        {
            "schema_version",
            "packet_sha256",
            "prompt_sha256",
            "report_sha256",
            "model_id",
            "provider",
            "round",
            "major_finding_ids",
        }
    )
    data = _document(payload, role=role, fields=fields)
    _schema(data, "d1-review-receipt-v1", role=role)
    round_number = data["round"]
    if type(round_number) is not int or round_number < 1:
        raise D1FilingGraphError(
            code="invalid_round", path=f"{role}.round", detail="must be positive"
        )
    findings = data["major_finding_ids"]
    if not isinstance(findings, list) or any(
        type(item) is not str or not item for item in findings
    ):
        raise D1FilingGraphError(
            code="invalid_findings", path=f"{role}.major_finding_ids", detail="must be strings"
        )
    if len(findings) != len(set(findings)):
        raise D1FilingGraphError(
            code="duplicate_findings", path=f"{role}.major_finding_ids", detail="must be unique"
        )
    return D1ReviewReceipt(
        packet_sha256=_sha256(data["packet_sha256"], path=f"{role}.packet_sha256"),
        prompt_sha256=_sha256(data["prompt_sha256"], path=f"{role}.prompt_sha256"),
        report_sha256=_sha256(data["report_sha256"], path=f"{role}.report_sha256"),
        model_id=_text(data["model_id"], path=f"{role}.model_id"),
        provider=_text(data["provider"], path=f"{role}.provider"),
        round=round_number,
        major_finding_ids=tuple(findings),
    )
