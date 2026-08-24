"""Compatibility facade for the split D1 verdict evidence domain.

New implementation code should import the feature-owning modules directly:
``d1_types`` for value vocabulary, ``d1_validator`` for serialized-artifact
validation, and ``d1_gate`` for the filed decision.  This facade preserves the
stable import surface while the Graphite stack lands those seams independently.
"""

from __future__ import annotations

from entropic_substrate.evidence.d1_contract import D1DecisionContract, D1DecisionRule
from entropic_substrate.evidence.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    D1DetachedFiling,
    D1FilingManifest,
    D1FilingValidationError,
    ValidatedD1Filing,
    validate_d1_filing_manifest,
)
from entropic_substrate.evidence.d1_gate import decide_filed_verdict
from entropic_substrate.evidence.d1_types import (
    D1_ARM_IDS,
    D1_VERDICT_SCHEMA_VERSION,
    D1ArmEvidence,
    D1Clause,
    D1Contrast,
    D1ContrastTerm,
    D1FamilyIdentity,
    D1InputIdentity,
    D1Integrity,
    D1Review,
    D1ReviewAdjudication,
    D1Successor,
    D1VerdictValidationError,
    FiledGateAuthorization,
    FiledGateOutcome,
    FiledGateRetirement,
    FiledGateValidNegative,
    JsonScalar,
    JsonValue,
    PreregistrationIdentity,
)
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict

__all__ = [
    "D1_ARM_IDS",
    "D1_VERDICT_SCHEMA_VERSION",
    "REQUIRED_D1_FILING_ARTIFACTS",
    "D1ArmEvidence",
    "D1Clause",
    "D1Contrast",
    "D1ContrastTerm",
    "D1DecisionContract",
    "D1DecisionRule",
    "D1DetachedFiling",
    "D1FamilyIdentity",
    "D1FilingManifest",
    "D1FilingValidationError",
    "D1InputIdentity",
    "D1Integrity",
    "D1Review",
    "D1ReviewAdjudication",
    "D1Successor",
    "D1VerdictValidationError",
    "FiledGateAuthorization",
    "FiledGateOutcome",
    "FiledGateRetirement",
    "FiledGateValidNegative",
    "JsonScalar",
    "JsonValue",
    "PreregistrationIdentity",
    "ValidatedD1Filing",
    "ValidatedD1Verdict",
    "decide_filed_verdict",
    "validate_d1_filing_manifest",
]
