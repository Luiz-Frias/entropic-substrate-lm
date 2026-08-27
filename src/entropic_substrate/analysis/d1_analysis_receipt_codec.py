"""Dependency-neutral wire contract for D1 analysis execution receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import NoReturn, cast

__all__ = [
    "D1_ANALYSIS_EXECUTION_RECEIPT_SCHEMA",
    "D1AnalysisExecutionReceipt",
    "D1AnalysisFilingBindings",
    "canonical_d1_analysis_receipt_bytes",
    "parse_d1_analysis_execution_receipt",
]

D1_ANALYSIS_EXECUTION_RECEIPT_SCHEMA = "d1-analysis-execution-receipt-v1"
_RECEIPT_FIELDS = frozenset(
    {
        "alias_commitment",
        "analysis_digest",
        "bindings",
        "filing_bindings",
        "filing_eligible",
        "implementation",
        "inputs",
        "isolation",
        "schema_version",
    }
)
_ANALYSIS_BINDING_FIELDS = frozenset(
    {"capture_digest", "decision_contract_digest", "design_digest"}
)
_FILING_BINDING_FIELDS = frozenset(
    {
        "capture_manifest_sha256",
        "isolation_attestation_sha256",
        "output_sha256",
        "preregistration_sha256",
        "reader_sha256",
    }
)


def canonical_d1_analysis_receipt_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class D1AnalysisFilingBindings:
    """Exact filed-byte identities issued with an externally verified analysis."""

    capture_manifest_sha256: str
    preregistration_sha256: str
    reader_sha256: str
    isolation_attestation_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "capture_manifest_sha256",
            "preregistration_sha256",
            "reader_sha256",
            "isolation_attestation_sha256",
            "output_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"D1 filing {name}")

    @property
    def manifest(self) -> dict[str, str]:
        return {
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "isolation_attestation_sha256": self.isolation_attestation_sha256,
            "output_sha256": self.output_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "reader_sha256": self.reader_sha256,
        }


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"D1 analysis execution receipt has duplicate key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise ValueError(f"D1 analysis execution receipt contains non-finite value {value!r}")


def _object(value: object, *, fields: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"D1 analysis execution receipt {label} must be an object")
    result = cast(dict[str, object], value)
    if frozenset(result) != fields:
        raise ValueError(f"D1 analysis execution receipt {label} fields must be exact")
    return result


def _filing_bindings(value: object) -> D1AnalysisFilingBindings:
    root = _object(value, fields=_FILING_BINDING_FIELDS, label="filing bindings")
    try:
        return D1AnalysisFilingBindings(
            capture_manifest_sha256=cast(str, root["capture_manifest_sha256"]),
            preregistration_sha256=cast(str, root["preregistration_sha256"]),
            reader_sha256=cast(str, root["reader_sha256"]),
            isolation_attestation_sha256=cast(str, root["isolation_attestation_sha256"]),
            output_sha256=cast(str, root["output_sha256"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"D1 analysis execution receipt filing bindings are invalid: {error}"
        ) from error


def _decode(payload: str) -> dict[str, object]:
    if type(payload) is not str or not payload:
        raise ValueError("D1 analysis execution receipt must be nonempty text")
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("D1 analysis execution receipt must be valid JSON") from error
    root = _object(value, fields=_RECEIPT_FIELDS, label="root")
    if root["schema_version"] != D1_ANALYSIS_EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("D1 analysis execution receipt schema is unsupported")
    if canonical_d1_analysis_receipt_bytes(root).decode("utf-8") != payload:
        raise ValueError("D1 analysis execution receipt must be canonical JSON")
    _require_sha256(root["alias_commitment"], label="D1 receipt alias commitment")
    analysis_digest = _require_sha256(root["analysis_digest"], label="D1 receipt analysis digest")
    bindings = _object(root["bindings"], fields=_ANALYSIS_BINDING_FIELDS, label="analysis bindings")
    for name, digest in bindings.items():
        _require_sha256(digest, label=f"D1 receipt {name}")
    eligible = root["filing_eligible"]
    if type(eligible) is not bool:
        raise ValueError("D1 analysis execution receipt filing_eligible must be bool")
    filing = root["filing_bindings"]
    if eligible:
        parsed = _filing_bindings(filing)
        if parsed.output_sha256 != analysis_digest:
            raise ValueError(
                "D1 analysis execution receipt filing output differs from analysis digest"
            )
    elif filing is not None:
        raise ValueError(
            "D1 analysis execution receipt non-filing result cannot carry filing bindings"
        )
    if type(root["implementation"]) is not dict:
        raise ValueError("D1 analysis execution receipt implementation must be an object")
    if type(root["inputs"]) is not list or not root["inputs"]:
        raise ValueError("D1 analysis execution receipt inputs must be a nonempty array")
    if type(root["isolation"]) is not dict:
        raise ValueError("D1 analysis execution receipt isolation must be an object")
    return root


@dataclass(frozen=True, slots=True)
class D1AnalysisExecutionReceipt:
    canonical_json: str

    def __post_init__(self) -> None:
        _decode(self.canonical_json)

    @property
    def manifest(self) -> dict[str, object]:
        return _decode(self.canonical_json)

    @property
    def filing_bindings(self) -> D1AnalysisFilingBindings | None:
        value = self.manifest["filing_bindings"]
        return None if value is None else _filing_bindings(value)

    @property
    def receipt_digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def parse_d1_analysis_execution_receipt(payload: bytes) -> D1AnalysisExecutionReceipt:
    """Strictly decode the exact execution receipt used by review and filing."""
    if type(payload) is not bytes:
        raise TypeError("D1 analysis execution receipt payload must be exact bytes")
    try:
        canonical = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("D1 analysis execution receipt must be UTF-8") from error
    return D1AnalysisExecutionReceipt(canonical)
