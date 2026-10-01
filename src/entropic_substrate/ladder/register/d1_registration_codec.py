"""Strict canonical JSON codec for the registered D1 decision contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NoReturn, cast

from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    canonical_digest,
    require_sha256_hex_any_case,
)
from entropic_substrate.kernel.lifecycle import SuccessorAuthorization
from entropic_substrate.kernel.verdict.d1_contract import (
    D1DecisionContract,
    D1DecisionRule,
    D1RegisteredClause,
    D1RegisteredContrast,
    D1RegisteredEndpoint,
    D1RegisteredExclusion,
)
from entropic_substrate.kernel.verdict.d1_types import D1ContrastTerm, D1FamilyIdentity
from entropic_substrate.ladder.register.d1_family import D1ExperimentFamily
from entropic_substrate.ladder.register.d1_registered_design import (
    D1AdmissionBlocked,
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
    decision_contract_digest,
    decision_contract_manifest,
)

__all__ = [
    "D1PreregistrationByteEvidence",
    "D1RegisteredDesignDeclaration",
    "D1RegistrationCodecError",
    "decision_contract_from_mapping",
    "decision_contract_to_mapping",
    "hydrate_registered_design",
    "parse_decision_contract",
    "parse_registered_design_declaration",
    "registered_design_declaration_from_mapping",
    "registered_design_to_mapping",
    "serialize_decision_contract",
    "serialize_registered_design",
]

_SCHEMA = "d1-decision-contract-v1"
_ROOT_KEYS = frozenset(
    {
        "arm_ids",
        "clause_ids",
        "clause_precedence",
        "clause_rules",
        "contrast_ids",
        "contrast_rules",
        "decision_rules",
        "exclusion_rules",
        "family_identity",
        "registered_constants",
        "schema_version",
        "verdict_vocabulary",
    }
)
_CONTRAST_KEYS = frozenset(
    {
        "confidence_level",
        "contrast_id",
        "contrast_kind",
        "decision_vocabulary",
        "df_rule",
        "endpoint",
        "margins",
        "multiplicity",
        "n_rule",
        "registered_n",
        "terms",
    }
)
_CLAUSE_KEYS = frozenset(
    {"clause_id", "decision_vocabulary", "margin", "registered_rule", "threshold"}
)
_DECISION_KEYS = frozenset(
    {
        "eligible_strategies",
        "require_integrity_pass",
        "require_review_clearance",
        "required_clause_statuses",
        "successor_authorization",
    }
)


class D1RegistrationCodecError(ValueError):
    """Persisted registration bytes are malformed, permissive, or drifted."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1RegistrationCodecError(message)
    raise D1RegistrationCodecError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _exact(value: Mapping[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"{label} keys must be exact")


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be a nonempty trimmed string")
    return value


def _array(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{label} must be a JSON array")
    return cast(list[object], value)


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(
        _text(item, label=f"{label}[{index}]")
        for index, item in enumerate(_array(value, label=label))
    )


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        _fail(f"{label} must be numeric, not boolean")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _number_or_none(value: object, *, label: str) -> float | None:
    return None if value is None else _number(value, label=label)


def _positive_int_or_none(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        _fail(f"{label} must be a positive integer or null")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be boolean")
    return value


def _string_mapping(value: object, *, label: str) -> dict[str, str]:
    mapping = _mapping(value, label=label)
    return {
        _text(key, label=f"{label} key"): _text(item, label=f"{label}.{key}")
        for key, item in mapping.items()
    }


def _number_mapping(value: object, *, label: str) -> dict[str, float]:
    mapping = _mapping(value, label=label)
    return {
        _text(key, label=f"{label} key"): _number(item, label=f"{label}.{key}")
        for key, item in mapping.items()
    }


def _family(value: object) -> D1FamilyIdentity:
    item = _mapping(value, label="family_identity")
    _exact(
        item,
        frozenset({"arm_construction_digests", "family_digest"}),
        label="family_identity",
    )
    return D1FamilyIdentity(
        family_digest=_text(item["family_digest"], label="family_identity.family_digest"),
        arm_construction_digests=_string_mapping(
            item["arm_construction_digests"],
            label="family_identity.arm_construction_digests",
        ),
    )


def _endpoint(value: object, *, path: str) -> D1RegisteredEndpoint:
    item = _mapping(value, label=path)
    _exact(item, frozenset({"estimator", "metric", "smoothing_window"}), label=path)
    window = item["smoothing_window"]
    if type(window) is not int:
        _fail(f"{path}.smoothing_window must be an integer")
    return D1RegisteredEndpoint(
        metric=_text(item["metric"], label=f"{path}.metric"),
        estimator=_text(item["estimator"], label=f"{path}.estimator"),
        smoothing_window=window,
    )


def _term(value: object, *, path: str) -> D1ContrastTerm:
    item = _mapping(value, label=path)
    _exact(item, frozenset({"arm_id", "coefficient"}), label=path)
    return D1ContrastTerm(
        arm_id=_text(item["arm_id"], label=f"{path}.arm_id"),
        coefficient=_number(item["coefficient"], label=f"{path}.coefficient"),
    )


def _contrast(value: object, *, path: str) -> D1RegisteredContrast:
    item = _mapping(value, label=path)
    _exact(item, _CONTRAST_KEYS, label=path)
    terms = tuple(
        _term(term, path=f"{path}.terms[{index}]")
        for index, term in enumerate(_array(item["terms"], label=f"{path}.terms"))
    )
    return D1RegisteredContrast(
        contrast_id=_text(item["contrast_id"], label=f"{path}.contrast_id"),
        contrast_kind=_text(item["contrast_kind"], label=f"{path}.contrast_kind"),
        terms=terms,
        endpoint=_endpoint(item["endpoint"], path=f"{path}.endpoint"),
        n_rule=_text(item["n_rule"], label=f"{path}.n_rule"),
        registered_n=_positive_int_or_none(item["registered_n"], label=f"{path}.registered_n"),
        df_rule=_text(item["df_rule"], label=f"{path}.df_rule"),
        confidence_level=_number(item["confidence_level"], label=f"{path}.confidence_level"),
        multiplicity=_text(item["multiplicity"], label=f"{path}.multiplicity"),
        margins=_number_mapping(item["margins"], label=f"{path}.margins"),
        decision_vocabulary=_text(item["decision_vocabulary"], label=f"{path}.decision_vocabulary"),
    )


def _clause(value: object, *, path: str) -> D1RegisteredClause:
    item = _mapping(value, label=path)
    _exact(item, _CLAUSE_KEYS, label=path)
    return D1RegisteredClause(
        clause_id=_text(item["clause_id"], label=f"{path}.clause_id"),
        registered_rule=_text(item["registered_rule"], label=f"{path}.registered_rule"),
        decision_vocabulary=_text(item["decision_vocabulary"], label=f"{path}.decision_vocabulary"),
        threshold=_number_or_none(item["threshold"], label=f"{path}.threshold"),
        margin=_number_or_none(item["margin"], label=f"{path}.margin"),
    )


def _exclusion(value: object, *, path: str) -> D1RegisteredExclusion:
    item = _mapping(value, label=path)
    keys = frozenset({"measurement", "predicate", "reason_id", "scope"})
    _exact(item, keys, label=path)
    return D1RegisteredExclusion(
        reason_id=_text(item["reason_id"], label=f"{path}.reason_id"),
        measurement=_text(item["measurement"], label=f"{path}.measurement"),
        predicate=_text(item["predicate"], label=f"{path}.predicate"),
        scope=_text(item["scope"], label=f"{path}.scope"),
    )


def _decision(value: object, *, path: str) -> D1DecisionRule:
    item = _mapping(value, label=path)
    _exact(item, _DECISION_KEYS, label=path)
    try:
        authorization = SuccessorAuthorization(
            _text(item["successor_authorization"], label=f"{path}.successor_authorization")
        )
    except ValueError as error:
        _fail(f"{path}.successor_authorization is unsupported", error)
    return D1DecisionRule(
        successor_authorization=authorization,
        required_clause_statuses=_string_mapping(
            item["required_clause_statuses"], label=f"{path}.required_clause_statuses"
        ),
        require_integrity_pass=_boolean(
            item["require_integrity_pass"], label=f"{path}.require_integrity_pass"
        ),
        require_review_clearance=_boolean(
            item["require_review_clearance"], label=f"{path}.require_review_clearance"
        ),
        eligible_strategies=_strings(
            item["eligible_strategies"], label=f"{path}.eligible_strategies"
        ),
    )


def _canonical(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def decision_contract_to_mapping(contract: D1DecisionContract) -> dict[str, object]:
    if type(contract) is not D1DecisionContract:
        raise TypeError("decision registration codec requires D1DecisionContract")
    return copy.deepcopy(decision_contract_manifest(contract))


def decision_contract_from_mapping(value: object) -> D1DecisionContract:
    root = _mapping(value, label="decision contract")
    _exact(root, _ROOT_KEYS, label="decision contract")
    if root["schema_version"] != _SCHEMA:
        _fail("decision contract schema_version is unsupported")
    contrast_ids = _strings(root["contrast_ids"], label="contrast_ids")
    clause_ids = _strings(root["clause_ids"], label="clause_ids")
    try:
        contract = D1DecisionContract(
            family_identity=_family(root["family_identity"]),
            registered_constants=_mapping(
                root["registered_constants"], label="registered_constants"
            ),
            arm_ids=_strings(root["arm_ids"], label="arm_ids"),
            contrast_ids=contrast_ids,
            clause_ids=clause_ids,
            contrast_rules={
                identifier: _contrast(item, path=f"contrast_rules.{identifier}")
                for identifier, item in _ordered_rules(
                    root["contrast_rules"], contrast_ids, label="contrast_rules"
                )
            },
            clause_rules={
                identifier: _clause(item, path=f"clause_rules.{identifier}")
                for identifier, item in _ordered_rules(
                    root["clause_rules"], clause_ids, label="clause_rules"
                )
            },
            exclusion_rules=tuple(
                _exclusion(item, path=f"exclusion_rules[{index}]")
                for index, item in enumerate(
                    _array(root["exclusion_rules"], label="exclusion_rules")
                )
            ),
            clause_precedence=_strings(root["clause_precedence"], label="clause_precedence"),
            verdict_vocabulary=_text(root["verdict_vocabulary"], label="verdict_vocabulary"),
            decision_rules={
                identifier: _decision(item, path=f"decision_rules.{identifier}")
                for identifier, item in _mapping(
                    root["decision_rules"], label="decision_rules"
                ).items()
            },
        )
    except D1RegistrationCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"decision contract is invalid: {error}", error)
    if _canonical(root) != serialize_decision_contract(contract):
        _fail("decision contract mapping is not canonical")
    return contract


def _ordered_rules(
    value: object, identifiers: tuple[str, ...], *, label: str
) -> tuple[tuple[str, object], ...]:
    mapping = _mapping(value, label=label)
    if frozenset(mapping) != frozenset(identifiers):
        _fail(f"{label} keys must equal the registered identifiers")
    return tuple((identifier, mapping[identifier]) for identifier in identifiers)


def serialize_decision_contract(contract: D1DecisionContract) -> str:
    return _canonical(decision_contract_to_mapping(contract))


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate decision-contract JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite decision-contract JSON constant {value!r}")


def parse_decision_contract(payload: str) -> D1DecisionContract:
    if type(payload) is not str or not payload:
        _fail("decision contract payload must be nonempty text")
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except D1RegistrationCodecError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        _fail("decision contract payload is invalid JSON", error)
    contract = decision_contract_from_mapping(value)
    if serialize_decision_contract(contract) != payload:
        _fail("decision contract JSON is not canonical")
    return contract


_DESIGN_KEYS = frozenset(
    {
        "arm_construction_digests",
        "decision_contract_digest",
        "family_digest",
        "preregistration",
        "registered_seeds",
        "schema_version",
    }
)
_PREREGISTRATION_KEYS = frozenset({"blob_sha256", "commit", "path"})
_DESIGN_SCHEMA = "d1-registered-design-v1"


@dataclass(frozen=True, slots=True)
class D1PreregistrationByteEvidence:
    """Stable raw preregistration bytes with self-verifying size and SHA-256."""

    payload: bytes = field(repr=False)
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes or not self.payload:
            raise ValueError("preregistration payload must be nonempty bytes")
        if type(self.size_bytes) is not int or self.size_bytes != len(self.payload):
            raise ValueError("preregistration size_bytes differs from raw bytes")
        try:
            require_sha256_hex_any_case(self.sha256, label="preregistration sha256")
        except (TypeError, ValueError) as error:
            raise ValueError("preregistration sha256 must be lowercase SHA-256") from error
        if self.sha256 != self.sha256.lower():
            raise ValueError("preregistration sha256 must be lowercase SHA-256")
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("preregistration sha256 differs from raw bytes")

    @classmethod
    def from_bytes(cls, payload: bytes) -> D1PreregistrationByteEvidence:
        if type(payload) is not bytes:
            raise TypeError("preregistration evidence requires bytes")
        return cls(payload, len(payload), hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True)
class D1RegisteredDesignDeclaration:
    """Decoded design claim without factory attestation or gate authority."""

    family_identity: D1FamilyIdentity
    decision_contract_digest: str
    registered_seeds: tuple[int, ...]
    preregistration: D1PreregistrationArtifact

    def __post_init__(self) -> None:
        if type(self.family_identity) is not D1FamilyIdentity:
            raise TypeError("design declaration requires D1FamilyIdentity")
        require_sha256_hex_any_case(
            self.decision_contract_digest,
            label="design declaration decision_contract_digest",
        )
        if (
            type(self.registered_seeds) is not tuple
            or not self.registered_seeds
            or any(type(seed) is not int or seed < 0 for seed in self.registered_seeds)
            or len(set(self.registered_seeds)) != len(self.registered_seeds)
        ):
            raise ValueError("design declaration seeds must be unique nonnegative integers")
        if type(self.preregistration) is not D1PreregistrationArtifact:
            raise TypeError("design declaration requires preregistration identity")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "arm_construction_digests": dict(self.family_identity.arm_construction_digests),
            "decision_contract_digest": self.decision_contract_digest,
            "family_digest": self.family_identity.family_digest,
            "preregistration": self.preregistration.manifest,
            "registered_seeds": list(self.registered_seeds),
            "schema_version": _DESIGN_SCHEMA,
        }

    @property
    def design_digest(self) -> str:
        return canonical_digest(self.manifest)


def _sha256_text(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    try:
        require_sha256_hex_any_case(text, label=label)
    except ValueError as error:
        _fail(f"{label} must be lowercase SHA-256", error)
    if text != text.lower():
        _fail(f"{label} must be lowercase SHA-256")
    return text


def _design_preregistration(value: object) -> D1PreregistrationArtifact:
    item = _mapping(value, label="registered design preregistration")
    _exact(item, _PREREGISTRATION_KEYS, label="registered design preregistration")
    try:
        return D1PreregistrationArtifact(
            path=_text(item["path"], label="preregistration.path"),
            blob_sha256=_sha256_text(item["blob_sha256"], label="preregistration.blob_sha256"),
            commit=_text(item["commit"], label="preregistration.commit"),
        )
    except (TypeError, ValueError) as error:
        _fail(f"registered design preregistration is invalid: {error}", error)


def registered_design_declaration_from_mapping(
    value: object,
) -> D1RegisteredDesignDeclaration:
    root = _mapping(value, label="registered design")
    _exact(root, _DESIGN_KEYS, label="registered design")
    if root["schema_version"] != _DESIGN_SCHEMA:
        _fail("registered design schema_version is unsupported")
    raw_seeds = _array(root["registered_seeds"], label="registered_seeds")
    if any(type(seed) is not int or seed < 0 for seed in raw_seeds):
        _fail("registered_seeds must contain nonnegative integers")
    try:
        declaration = D1RegisteredDesignDeclaration(
            family_identity=D1FamilyIdentity(
                family_digest=_sha256_text(
                    root["family_digest"], label="registered design family_digest"
                ),
                arm_construction_digests=_string_mapping(
                    root["arm_construction_digests"],
                    label="registered design arm_construction_digests",
                ),
            ),
            decision_contract_digest=_sha256_text(
                root["decision_contract_digest"],
                label="registered design decision_contract_digest",
            ),
            registered_seeds=tuple(cast(list[int], raw_seeds)),
            preregistration=_design_preregistration(root["preregistration"]),
        )
    except D1RegistrationCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"registered design is invalid: {error}", error)
    if _canonical(root) != _canonical(declaration.manifest):
        _fail("registered design mapping is not canonical")
    return declaration


def registered_design_to_mapping(design: D1RegisteredDesign) -> dict[str, object]:
    if type(design) is not D1RegisteredDesign:
        raise TypeError("design registration codec requires D1RegisteredDesign")
    return copy.deepcopy(design.manifest)


def serialize_registered_design(design: D1RegisteredDesign) -> str:
    return _canonical(registered_design_to_mapping(design))


def _design_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate registered-design JSON key {key!r}")
        result[key] = value
    return result


def parse_registered_design_declaration(payload: str) -> D1RegisteredDesignDeclaration:
    if type(payload) is not str or not payload:
        _fail("registered design payload must be nonempty text")
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_design_pairs,
            parse_constant=_nonfinite,
        )
    except D1RegistrationCodecError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        _fail("registered design payload is invalid JSON", error)
    declaration = registered_design_declaration_from_mapping(value)
    if _canonical(declaration.manifest) != payload:
        _fail("registered design JSON is not canonical")
    return declaration


def _family_identity(family: D1ExperimentFamily) -> D1FamilyIdentity:
    return D1FamilyIdentity(
        family_digest=family.family_digest,
        arm_construction_digests={
            arm.value: family.draft_for(arm).construction_digest for arm in D1Arm
        },
    )


def hydrate_registered_design(
    payload: str,
    *,
    expected_design_digest: str,
    family: D1ExperimentFamily,
    decision_contract: D1DecisionContract,
    preregistration_evidence: D1PreregistrationByteEvidence,
) -> D1RegisteredDesign:
    """Re-earn design attestation from decoded evidence through admission."""
    if type(family) is not D1ExperimentFamily:
        raise TypeError("design hydration requires D1ExperimentFamily")
    if type(decision_contract) is not D1DecisionContract:
        raise TypeError("design hydration requires D1DecisionContract")
    if type(preregistration_evidence) is not D1PreregistrationByteEvidence:
        raise TypeError("design hydration requires preregistration byte evidence")
    expected = _sha256_text(expected_design_digest, label="expected design digest")
    declaration = parse_registered_design_declaration(payload)
    if declaration.design_digest != expected:
        _fail("registered design digest differs from expected design digest")
    if declaration.preregistration.blob_sha256 != preregistration_evidence.sha256:
        _fail("raw preregistration SHA-256 differs from registered blob")
    if declaration.family_identity != _family_identity(family):
        _fail("registered design family or arm identities differ from supplied family")
    if declaration.decision_contract_digest != decision_contract_digest(decision_contract):
        _fail("registered design differs from decoded decision contract")
    try:
        admitted = admit_d1_registered_design(
            D1RegisteredDesignCandidate(
                family=family,
                decision_contract=decision_contract,
                registered_seeds=declaration.registered_seeds,
                preregistration=declaration.preregistration,
            )
        )
    except (TypeError, ValueError) as error:
        _fail(f"registered design admission failed: {error}", error)
    if type(admitted) is D1AdmissionBlocked:
        _fail("registered design admission remained blocked")
    if type(admitted) is not D1RegisteredDesign:
        _fail("registered design admission returned an unsupported result")
    if admitted.manifest != declaration.manifest:
        _fail("rehydrated design manifest differs from persisted declaration")
    if admitted.design_digest != expected:
        _fail("rehydrated design digest differs from expected design digest")
    return admitted
