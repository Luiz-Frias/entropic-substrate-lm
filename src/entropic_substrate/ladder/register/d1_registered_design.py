"""Canonical pre-outcome D1 design registration values.

This module names the machine contract that a later filed preregistration
commits to.  It does not read Git, write a preregistration document, or choose
scientific constants.  Unresolved choices remain typed gaps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final, cast

from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    canonical_digest,
    require_commit,
    require_nonempty_string,
    require_sha256_hex_any_case,
)
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.ladder.register.d1_family import D1ExperimentFamily
from entropic_substrate.ladder.register.d1_provenance import require_lowercase_digest

D1_REGISTERED_DESIGN_SCHEMA: Final = "d1-registered-design-v1"
_DESIGN_ATTESTATION: Final = object()


class D1AdmissionField(StrEnum):
    """Launch-blocking decisions which may still await a scientific ruling."""

    DECISION_CONTRACT = "decision_contract"
    REGISTERED_SEEDS = "registered_seeds"
    PREREGISTRATION = "preregistration"


@dataclass(frozen=True, slots=True)
class D1AdmissionGap:
    """An explicit unresolved value; never a request to infer a default."""

    field: D1AdmissionField
    reason: str

    def __post_init__(self) -> None:
        if type(self.field) is not D1AdmissionField:
            raise TypeError("D1 admission gap field must be a D1AdmissionField")
        require_nonempty_string(self.reason, label="D1 admission gap reason")

    @property
    def manifest(self) -> dict[str, str]:
        return {"field": self.field.value, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class D1AdmissionBlocked:
    """Typed result returned when registration is scientifically incomplete."""

    gaps: tuple[D1AdmissionGap, ...]

    def __post_init__(self) -> None:
        if type(self.gaps) is not tuple or not self.gaps:
            raise ValueError("blocked D1 admission requires at least one typed gap")
        if any(type(gap) is not D1AdmissionGap for gap in self.gaps):
            raise TypeError("blocked D1 admission gaps must be D1AdmissionGap values")
        fields = tuple(gap.field for gap in self.gaps)
        if len(set(fields)) != len(fields):
            raise ValueError("blocked D1 admission refuses duplicate gap fields")


@dataclass(frozen=True, slots=True)
class D1PreregistrationArtifact:
    """Repository-relative path, committed blob content, and containing commit."""

    path: str
    blob_sha256: str
    commit: str

    def __post_init__(self) -> None:
        require_nonempty_string(self.path, label="D1 preregistration path")
        parsed = PurePosixPath(self.path)
        if parsed.is_absolute() or "\\" in self.path:
            raise ValueError("D1 preregistration path must be repository-relative POSIX")
        if ".." in parsed.parts:
            raise ValueError("D1 preregistration path forbids parent traversal")
        require_sha256_hex_any_case(self.blob_sha256, label="D1 preregistration blob_sha256")
        if self.blob_sha256 != self.blob_sha256.lower():
            raise ValueError("D1 preregistration blob_sha256 must be lowercase")
        require_commit(self.commit, label="D1 preregistration commit")
        if self.commit != self.commit.lower():
            raise ValueError("D1 preregistration commit must be lowercase")

    @property
    def manifest(self) -> dict[str, str]:
        return {
            "blob_sha256": self.blob_sha256,
            "commit": self.commit,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class D1RegisteredDesignCandidate:
    """A complete candidate or a carrier for explicit unresolved choices."""

    family: D1ExperimentFamily
    decision_contract: D1DecisionContract | D1AdmissionGap
    registered_seeds: tuple[int, ...] | D1AdmissionGap
    preregistration: D1PreregistrationArtifact | D1AdmissionGap


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def decision_contract_manifest(contract: D1DecisionContract) -> dict[str, object]:
    """Project every registered D1 decision field into canonical JSON data."""
    if type(contract) is not D1DecisionContract:
        raise TypeError("decision contract must be a D1DecisionContract")
    return {
        "arm_ids": list(contract.arm_ids),
        "clause_ids": list(contract.clause_ids),
        "clause_precedence": list(contract.clause_precedence),
        "clause_rules": {
            key: {
                "clause_id": rule.clause_id,
                "decision_vocabulary": rule.decision_vocabulary,
                "margin": rule.margin,
                "registered_rule": rule.registered_rule,
                "threshold": rule.threshold,
            }
            for key, rule in contract.clause_rules.items()
        },
        "contrast_ids": list(contract.contrast_ids),
        "contrast_rules": {
            key: {
                "confidence_level": rule.confidence_level,
                "contrast_id": rule.contrast_id,
                "contrast_kind": rule.contrast_kind,
                "decision_vocabulary": rule.decision_vocabulary,
                "df_rule": rule.df_rule,
                "endpoint": {
                    "estimator": rule.endpoint.estimator,
                    "metric": rule.endpoint.metric,
                    "smoothing_window": rule.endpoint.smoothing_window,
                },
                "margins": dict(rule.margins),
                "multiplicity": rule.multiplicity,
                "n_rule": rule.n_rule,
                "registered_n": rule.registered_n,
                "terms": [
                    {"arm_id": term.arm_id, "coefficient": term.coefficient} for term in rule.terms
                ],
            }
            for key, rule in contract.contrast_rules.items()
        },
        "decision_rules": {
            key: {
                "eligible_strategies": list(rule.eligible_strategies),
                "require_integrity_pass": rule.require_integrity_pass,
                "require_review_clearance": rule.require_review_clearance,
                "required_clause_statuses": dict(rule.required_clause_statuses),
                "successor_authorization": rule.successor_authorization.value,
            }
            for key, rule in contract.decision_rules.items()
        },
        "exclusion_rules": [
            {
                "measurement": rule.measurement,
                "predicate": rule.predicate,
                "reason_id": rule.reason_id,
                "scope": rule.scope,
            }
            for rule in contract.exclusion_rules
        ],
        "family_identity": {
            "arm_construction_digests": dict(contract.family_identity.arm_construction_digests),
            "family_digest": contract.family_identity.family_digest,
        },
        "registered_constants": _plain_json(contract.registered_constants),
        "schema_version": "d1-decision-contract-v1",
        "verdict_vocabulary": contract.verdict_vocabulary,
    }


def decision_contract_digest(contract: D1DecisionContract) -> str:
    """Bind the complete decision surface, including thresholds and precedence."""
    return canonical_digest(decision_contract_manifest(contract))


@dataclass(frozen=True, slots=True)
class D1RegisteredDesign:
    """Factory-attested, outcome-blind identity of one complete D1 design."""

    family_digest: str
    arm_construction_digests: Mapping[str, str]
    decision_contract_digest: str
    registered_seeds: tuple[int, ...]
    preregistration: D1PreregistrationArtifact
    _attestation: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._attestation is not _DESIGN_ATTESTATION:
            raise TypeError("D1RegisteredDesign is factory-only")
        require_lowercase_digest(self.family_digest, label="D1 registered family_digest")
        require_lowercase_digest(
            self.decision_contract_digest,
            label="D1 registered decision_contract_digest",
        )
        expected_arms = tuple(arm.value for arm in D1Arm)
        if set(self.arm_construction_digests) != set(expected_arms):
            raise ValueError("D1 registered design requires every arm construction digest")
        frozen: dict[str, str] = {}
        for arm in expected_arms:
            digest = self.arm_construction_digests[arm]
            require_lowercase_digest(digest, label=f"D1 {arm} construction digest")
            frozen[arm] = digest
        object.__setattr__(self, "arm_construction_digests", MappingProxyType(frozen))
        _validate_seeds(self.registered_seeds)
        if type(self.preregistration) is not D1PreregistrationArtifact:
            raise TypeError("D1 registered design requires preregistration artifact identity")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "arm_construction_digests": dict(self.arm_construction_digests),
            "decision_contract_digest": self.decision_contract_digest,
            "family_digest": self.family_digest,
            "preregistration": self.preregistration.manifest,
            "registered_seeds": list(self.registered_seeds),
            "schema_version": D1_REGISTERED_DESIGN_SCHEMA,
        }

    @property
    def design_digest(self) -> str:
        return canonical_digest(self.manifest)


def _validate_seeds(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise ValueError("D1 registered seeds must be a nonempty tuple")
    seeds = cast(tuple[object, ...], value)
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise TypeError("each D1 registered seed must be a nonnegative integer")
    typed = cast(tuple[int, ...], seeds)
    if len(set(typed)) != len(typed):
        raise ValueError("D1 registered seeds must be unique")
    return typed


def _require_gap(value: D1AdmissionGap, expected: D1AdmissionField) -> D1AdmissionGap:
    if value.field is not expected:
        raise ValueError(f"gap in {expected.value} slot must name {expected.value}")
    return value


def admit_d1_registered_design(
    candidate: D1RegisteredDesignCandidate,
) -> D1RegisteredDesign | D1AdmissionBlocked:
    """Admit a complete design or preserve every unresolved choice as data."""
    if type(candidate) is not D1RegisteredDesignCandidate:
        raise TypeError("D1 design admission requires a D1RegisteredDesignCandidate")
    if type(candidate.family) is not D1ExperimentFamily:
        raise TypeError("D1 design candidate requires a D1ExperimentFamily")
    gaps: list[D1AdmissionGap] = []
    if type(candidate.decision_contract) is D1AdmissionGap:
        gaps.append(_require_gap(candidate.decision_contract, D1AdmissionField.DECISION_CONTRACT))
    if type(candidate.registered_seeds) is D1AdmissionGap:
        gaps.append(_require_gap(candidate.registered_seeds, D1AdmissionField.REGISTERED_SEEDS))
    if type(candidate.preregistration) is D1AdmissionGap:
        gaps.append(_require_gap(candidate.preregistration, D1AdmissionField.PREREGISTRATION))
    if gaps:
        return D1AdmissionBlocked(gaps=tuple(gaps))

    contract = cast(D1DecisionContract, candidate.decision_contract)
    seeds = _validate_seeds(candidate.registered_seeds)
    preregistration = cast(D1PreregistrationArtifact, candidate.preregistration)
    if type(contract) is not D1DecisionContract:
        raise TypeError("D1 decision contract must be typed or an explicit gap")
    if type(preregistration) is not D1PreregistrationArtifact:
        raise TypeError("D1 preregistration must be typed or an explicit gap")

    expected_arm_digests = {
        arm.value: candidate.family.draft_for(arm).construction_digest for arm in D1Arm
    }
    identity = contract.family_identity
    if identity.family_digest != candidate.family.family_digest:
        raise ValueError("D1 decision contract family digest differs from experiment family")
    if dict(identity.arm_construction_digests) != expected_arm_digests:
        raise ValueError("D1 decision contract arm identities differ from experiment family")
    if contract.registered_constants.get("seeds") != seeds:
        raise ValueError("D1 decision contract seed roster differs from registered seeds")

    return D1RegisteredDesign(
        family_digest=candidate.family.family_digest,
        arm_construction_digests=expected_arm_digests,
        decision_contract_digest=decision_contract_digest(contract),
        registered_seeds=seeds,
        preregistration=preregistration,
        _attestation=_DESIGN_ATTESTATION,
    )
