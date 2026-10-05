"""Dependency-neutral wire contract for D1 analysis execution receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import NoReturn, cast

from entropic_substrate.kernel.codec.strict_json import (
    exact_keys,
    require_lowercase_sha256,
    require_mapping,
    strict_loads,
)

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
    }
)
# Exact wire key set for the receipt "implementation" object. RETRACTED
# (house rule 4, measured 2026-09-20): _decode previously checked only
# `type(...) is dict`, so the retired analysis_code_digest,
# complete_reader_digest, defect_reader_digest and nested "readers" keys
# (see D1AnalysisImplementationIdentity's docstring in
# d1_execution_receipt.py) still parsed on the wire unrejected. Zero
# persisted receipts carry them (brief Scope measurement, conductor grep
# over docs/, data/, results/), so the compat window closes by
# measurement rather than a version bump; d1-analysis-execution-receipt-v1
# stays frozen.
_IMPLEMENTATION_FIELDS = frozenset({"code_commit", "image_digest", "repository_digest"})


def canonical_d1_analysis_receipt_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    def fail(_reason: str) -> NoReturn:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    return require_lowercase_sha256(value, fail=fail)


@dataclass(frozen=True, slots=True)
class D1AnalysisFilingBindings:
    """Exact filed-byte identities issued with an externally verified analysis.

    RETRACTED (house rule 4, measured 2026-09-20): ``reader_sha256`` bound
    the D1 reader-source provenance closure's manifest bytes, retired this
    phase alongside the closure module itself. Zero persisted filings
    (checked via ``git grep`` over tracked source; the field is base64-nested
    and invisible to a plaintext scan of ``results/``) depended on this
    binding for anything other than exercising the now-removed closure.
    """

    capture_manifest_sha256: str
    preregistration_sha256: str
    isolation_attestation_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "capture_manifest_sha256",
            "preregistration_sha256",
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
        }


def _fail(reason: str) -> NoReturn:
    raise ValueError(f"D1 analysis execution receipt: {reason}")


def _object(value: object, *, fields: frozenset[str], label: str) -> dict[str, object]:
    def not_object(_reason: str) -> NoReturn:
        raise ValueError(f"D1 analysis execution receipt {label} must be an object")

    result = require_mapping(value, fail=not_object)
    if any(exact_keys(result, fields)):
        raise ValueError(f"D1 analysis execution receipt {label} fields must be exact")
    return result


def _filing_bindings(value: object) -> D1AnalysisFilingBindings:
    root = _object(value, fields=_FILING_BINDING_FIELDS, label="filing bindings")
    try:
        return D1AnalysisFilingBindings(
            capture_manifest_sha256=cast(str, root["capture_manifest_sha256"]),
            preregistration_sha256=cast(str, root["preregistration_sha256"]),
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
        value: object = strict_loads(payload, fail=_fail)
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
    _object(root["implementation"], fields=_IMPLEMENTATION_FIELDS, label="implementation")
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
