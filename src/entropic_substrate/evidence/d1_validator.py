"""Strict construction of the immutable, evidence-bearing D1 verdict."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_decision_validation import (
    parse_review,
    validate_decision_contract,
)
from entropic_substrate.evidence.d1_parsing import (
    error,
    finite_number,
    hash_value,
    number_or_none,
    object_exact,
    object_optional,
    plain_hash,
    positive_int,
    registered_constants,
    sha256_digest,
    status,
    string,
)
from entropic_substrate.evidence.d1_types import (
    CLAUSE_NAMES,
    CONTRAST_DECISIONS,
    CONTRAST_KINDS,
    CONTRAST_VOCABULARY,
    D1_ARM_IDS,
    D1_VERDICT_SCHEMA_VERSION,
    EVIDENCE_CLASSES,
    ROOT_FIELDS,
    VERDICTS,
    D1ArmEvidence,
    D1Clause,
    D1Contrast,
    D1ContrastTerm,
    D1Endpoint,
    D1FamilyIdentity,
    D1InputIdentity,
    D1Integrity,
    D1Review,
    D1Successor,
    JsonValue,
    PreregistrationIdentity,
)
from entropic_substrate.evidence.lifecycle import (
    ContentHash,
    SuccessorAuthorization,
)


@dataclass(frozen=True, init=False)
class ValidatedD1Verdict:
    """Validated D1 machine verdict; the sole accepted filed-gate input."""

    schema_version: str
    rung: str
    preregistration: PreregistrationIdentity
    registered_constants: Mapping[str, JsonValue]
    arms: tuple[D1ArmEvidence, ...]
    contrasts: tuple[D1Contrast, ...]
    clauses: Mapping[str, D1Clause]
    integrity: D1Integrity
    exclusions: tuple[tuple[str, str, ContentHash], ...]
    inputs: tuple[D1InputIdentity, ...]
    review: D1Review
    clause_precedence: tuple[str, ...]
    verdict_vocabulary: str
    verdict: str
    successor: D1Successor

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, object], *, decision_contract: D1DecisionContract
    ) -> ValidatedD1Verdict:
        root = object_exact(payload, path="$", fields=ROOT_FIELDS)
        schema_version = string(root["schema_version"], path="schema_version")
        if schema_version != D1_VERDICT_SCHEMA_VERSION:
            raise error(
                "unsupported_schema_version",
                "schema_version",
                f"expected {D1_VERDICT_SCHEMA_VERSION!r}",
            )
        rung = string(root["rung"], path="rung")
        if rung != "d1":
            raise error("invalid_rung", "rung", "D1 verdict schema only accepts rung 'd1'")
        if type(decision_contract) is not D1DecisionContract:
            raise TypeError("D1 verdict validation requires a D1DecisionContract")
        preregistration = _parse_preregistration(
            root["preregistration"], decision_contract.family_identity
        )
        constants = registered_constants(root["registered_constants"])
        arms = _parse_arms(root["arms"], decision_contract.family_identity)
        contrasts = _parse_contrasts(root["contrasts"])
        clauses = _parse_clauses(root["clauses"])
        integrity = _parse_integrity(root["integrity"])
        exclusions = _parse_exclusions(root["exclusions"])
        inputs = _parse_inputs(root["inputs"])
        review = parse_review(root["review"])
        raw_precedence = root["clause_precedence"]
        if not isinstance(raw_precedence, list) or any(
            not isinstance(name, str) or not name for name in raw_precedence
        ):
            raise error(
                "invalid_clause_precedence",
                "clause_precedence",
                "must be a list of nonempty clause identifiers",
            )
        clause_precedence = tuple(raw_precedence)
        verdict_vocabulary = string(root["verdict_vocabulary"], path="verdict_vocabulary")
        verdict = string(root["verdict"], path="verdict")
        if verdict not in VERDICTS:
            raise error("invalid_verdict", "verdict", f"must be one of {sorted(VERDICTS)}")
        successor = _parse_successor(root["successor"])
        validate_decision_contract(
            contract=decision_contract,
            constants=constants,
            arms=arms,
            contrasts=contrasts,
            clauses=clauses,
            integrity=integrity,
            exclusions=exclusions,
            review=review,
            clause_precedence=clause_precedence,
            verdict_vocabulary=verdict_vocabulary,
            verdict=verdict,
            successor=successor,
        )
        return cls._create(
            schema_version=schema_version,
            rung=rung,
            preregistration=preregistration,
            registered_constants=constants,
            arms=arms,
            contrasts=contrasts,
            clauses=clauses,
            integrity=integrity,
            exclusions=exclusions,
            inputs=inputs,
            review=review,
            clause_precedence=clause_precedence,
            verdict_vocabulary=verdict_vocabulary,
            verdict=verdict,
            successor=successor,
        )

    @classmethod
    def _create(cls, **values: Any) -> ValidatedD1Verdict:
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        return instance


def _parse_preregistration(
    value: object, family_identity: D1FamilyIdentity
) -> PreregistrationIdentity:
    prereg = object_exact(
        value,
        path="preregistration",
        fields=frozenset({"path", "commit", "family_digest"}),
    )
    commit = string(prereg["commit"], path="preregistration.commit")
    if not 7 <= len(commit) <= 64 or any(char not in "0123456789abcdef" for char in commit):
        raise error(
            "invalid_preregistration_commit",
            "preregistration.commit",
            "must be a 7-to-64 character lowercase git hex identifier",
        )
    family_digest = sha256_digest(prereg["family_digest"], path="preregistration.family_digest")
    if family_digest != family_identity.family_digest:
        raise error(
            "invalid_family_digest",
            "preregistration.family_digest",
            "must match the registered D1 experiment family identity",
        )
    return PreregistrationIdentity(
        path=string(prereg["path"], path="preregistration.path"),
        commit=commit,
        family_digest=family_digest,
    )


def _parse_arms(value: object, family_identity: D1FamilyIdentity) -> tuple[D1ArmEvidence, ...]:
    if not isinstance(value, list) or len(value) != len(D1_ARM_IDS):
        raise error("invalid_arms", "arms", "must be the exact four-arm D1 factorial list")
    parsed: list[D1ArmEvidence] = []
    fields = frozenset({"arm_id", "construction_digest"})
    for index, raw_arm in enumerate(value):
        path = f"arms[{index}]"
        arm = object_exact(raw_arm, path=path, fields=fields)
        arm_id = string(arm["arm_id"], path=f"{path}.arm_id")
        parsed.append(
            D1ArmEvidence(
                arm_id=arm_id,
                construction_digest=sha256_digest(
                    arm["construction_digest"], path=f"{path}.construction_digest"
                ),
            )
        )
    if tuple(arm.arm_id for arm in parsed) != D1_ARM_IDS:
        raise error(
            "invalid_arm_set",
            "arms",
            f"must use canonical D1 arm identities and order {D1_ARM_IDS}",
        )
    for arm in parsed:
        if arm.construction_digest != family_identity.arm_construction_digests[arm.arm_id]:
            raise error(
                "invalid_arm_construction_digest",
                f"arms[{arm.arm_id}].construction_digest",
                "must match the canonical construction for its D1 arm",
            )
    return tuple(parsed)


def _parse_contrasts(value: object) -> tuple[D1Contrast, ...]:
    if not isinstance(value, list) or not value:
        raise error("invalid_contrasts", "contrasts", "must be a nonempty list")
    parsed: list[D1Contrast] = []
    identifiers: set[str] = set()
    fields = frozenset(
        {
            "contrast_id",
            "contrast_kind",
            "terms",
            "endpoint",
            "estimate",
            "sample_sd",
            "interval_lower",
            "interval_upper",
            "confidence_level",
            "multiplicity",
            "margins",
            "n",
            "df",
            "evidence_class",
            "decision_vocabulary",
            "decision",
            "evidence_hash",
        }
    )
    for index, raw_contrast in enumerate(value):
        path = f"contrasts[{index}]"
        contrast = object_exact(raw_contrast, path=path, fields=fields)
        contrast_id = string(contrast["contrast_id"], path=f"{path}.contrast_id")
        if contrast_id in identifiers:
            raise error(
                "duplicate_contrast", f"{path}.contrast_id", "contrast identifiers must be unique"
            )
        identifiers.add(contrast_id)
        contrast_kind = string(contrast["contrast_kind"], path=f"{path}.contrast_kind")
        if contrast_kind not in CONTRAST_KINDS:
            raise error("invalid_contrast_kind", f"{path}.contrast_kind", "unknown contrast kind")
        terms = _parse_terms(contrast["terms"], path=f"{path}.terms", contrast_kind=contrast_kind)
        endpoint_data = object_exact(
            contrast["endpoint"],
            path=f"{path}.endpoint",
            fields=frozenset({"metric", "estimator", "smoothing_window"}),
        )
        endpoint = D1Endpoint(
            metric=string(endpoint_data["metric"], path=f"{path}.endpoint.metric"),
            estimator=string(endpoint_data["estimator"], path=f"{path}.endpoint.estimator"),
            smoothing_window=positive_int(
                endpoint_data["smoothing_window"], path=f"{path}.endpoint.smoothing_window"
            ),
        )
        estimate = finite_number(contrast["estimate"], path=f"{path}.estimate")
        sample_sd = number_or_none(contrast["sample_sd"], path=f"{path}.sample_sd")
        if sample_sd is not None and sample_sd < 0:
            raise error("invalid_sample_sd", f"{path}.sample_sd", "must be nonnegative")
        lower = finite_number(contrast["interval_lower"], path=f"{path}.interval_lower")
        upper = finite_number(contrast["interval_upper"], path=f"{path}.interval_upper")
        if lower > upper:
            raise error("reversed_interval", path, "interval_lower must not exceed interval_upper")
        confidence = finite_number(contrast["confidence_level"], path=f"{path}.confidence_level")
        if not 0.0 < confidence < 1.0:
            raise error("invalid_confidence_level", f"{path}.confidence_level", "must be in (0, 1)")
        multiplicity = string(contrast["multiplicity"], path=f"{path}.multiplicity")
        raw_margins = contrast["margins"]
        if not isinstance(raw_margins, Mapping):
            raise error("invalid_margins", f"{path}.margins", "must be an object")
        margins: dict[str, float] = {}
        for name, raw_margin in raw_margins.items():
            if not isinstance(name, str) or not name:
                raise error("invalid_margins", f"{path}.margins", "keys must be nonempty")
            margin = finite_number(raw_margin, path=f"{path}.margins.{name}")
            if margin < 0.0:
                raise error("invalid_margins", f"{path}.margins.{name}", "must be nonnegative")
            margins[name] = margin
        n = positive_int(contrast["n"], path=f"{path}.n")
        raw_df = contrast["df"]
        if n == 1:
            if sample_sd is not None or raw_df is not None:
                raise error(
                    "inapplicable_sample_statistics",
                    path,
                    "n=1 requires sample_sd and df to be explicit null",
                )
            df: int | None = None
        else:
            if sample_sd is None:
                raise error("missing_sample_sd", f"{path}.sample_sd", "n>1 requires sample SD")
            df = positive_int(raw_df, path=f"{path}.df")
        evidence_class = string(contrast["evidence_class"], path=f"{path}.evidence_class")
        if evidence_class not in EVIDENCE_CLASSES:
            raise error(
                "invalid_evidence_class", f"{path}.evidence_class", "unknown evidence class"
            )
        vocabulary = string(contrast["decision_vocabulary"], path=f"{path}.decision_vocabulary")
        if vocabulary != CONTRAST_VOCABULARY:
            raise error(
                "invalid_decision_vocabulary",
                f"{path}.decision_vocabulary",
                f"must be {CONTRAST_VOCABULARY!r}",
            )
        decision = string(contrast["decision"], path=f"{path}.decision")
        if decision not in CONTRAST_DECISIONS:
            raise error("invalid_contrast_decision", f"{path}.decision", "unknown decision")
        parsed.append(
            D1Contrast(
                contrast_id=contrast_id,
                contrast_kind=contrast_kind,
                terms=terms,
                endpoint=endpoint,
                estimate=estimate,
                sample_sd=sample_sd,
                interval_lower=lower,
                interval_upper=upper,
                confidence_level=confidence,
                multiplicity=multiplicity,
                margins=MappingProxyType(margins),
                n=n,
                df=df,
                evidence_class=evidence_class,
                decision_vocabulary=vocabulary,
                decision=decision,
                evidence_hash=hash_value(contrast["evidence_hash"], path=f"{path}.evidence_hash"),
            )
        )
    return tuple(parsed)


def _parse_terms(value: object, *, path: str, contrast_kind: str) -> tuple[D1ContrastTerm, ...]:
    if not isinstance(value, list):
        raise error("invalid_contrast_terms", path, "must be a list")
    expected_count = 2 if contrast_kind == "PAIRWISE" else len(D1_ARM_IDS)
    if len(value) != expected_count:
        raise error(
            "invalid_contrast_terms",
            path,
            f"{contrast_kind} requires exactly {expected_count} terms",
        )
    terms: list[D1ContrastTerm] = []
    seen: set[str] = set()
    for index, raw_term in enumerate(value):
        term_path = f"{path}[{index}]"
        term = object_exact(raw_term, path=term_path, fields=frozenset({"arm_id", "coefficient"}))
        arm_id = string(term["arm_id"], path=f"{term_path}.arm_id")
        if arm_id not in D1_ARM_IDS:
            raise error(
                "unknown_contrast_arm", f"{term_path}.arm_id", "must name a canonical D1 arm"
            )
        if arm_id in seen:
            raise error("duplicate_contrast_arm", f"{term_path}.arm_id", "may appear only once")
        seen.add(arm_id)
        coefficient = finite_number(term["coefficient"], path=f"{term_path}.coefficient")
        if coefficient == 0.0:
            raise error("zero_contrast_coefficient", f"{term_path}.coefficient", "must be nonzero")
        terms.append(D1ContrastTerm(arm_id=arm_id, coefficient=coefficient))
    if contrast_kind == "FACTORIAL_INTERACTION":
        expected = {"none": 1.0, "sg": -1.0, "ac": -1.0, "both": 1.0}
        if {term.arm_id: term.coefficient for term in terms} != expected:
            raise error(
                "invalid_factorial_interaction",
                path,
                "must encode none - sg - ac + both with all four D1 arms",
            )
    return tuple(terms)


def _parse_clauses(value: object) -> Mapping[str, D1Clause]:
    clauses = object_exact(value, path="clauses", fields=CLAUSE_NAMES)
    fields = frozenset(
        {"registered_rule", "decision_vocabulary", "threshold", "margin", "status", "evidence_hash"}
    )
    parsed: dict[str, D1Clause] = {}
    for name in sorted(CLAUSE_NAMES):
        clause = object_exact(clauses[name], path=f"clauses.{name}", fields=fields)
        parsed[name] = D1Clause(
            registered_rule=string(
                clause["registered_rule"], path=f"clauses.{name}.registered_rule"
            ),
            decision_vocabulary=string(
                clause["decision_vocabulary"], path=f"clauses.{name}.decision_vocabulary"
            ),
            threshold=number_or_none(clause["threshold"], path=f"clauses.{name}.threshold"),
            margin=number_or_none(clause["margin"], path=f"clauses.{name}.margin"),
            status=status(clause["status"], path=f"clauses.{name}.status"),
            evidence_hash=hash_value(clause["evidence_hash"], path=f"clauses.{name}.evidence_hash"),
        )
    return MappingProxyType(parsed)


def _parse_integrity(value: object) -> D1Integrity:
    integrity = object_exact(
        value,
        path="integrity",
        fields=frozenset(
            {
                "status",
                "roster_complete",
                "pairing_verified",
                "data_order_verified",
                "completion_verified",
                "evidence_hash",
            }
        ),
    )
    booleans: dict[str, bool] = {}
    for name in (
        "roster_complete",
        "pairing_verified",
        "data_order_verified",
        "completion_verified",
    ):
        raw = integrity[name]
        if not isinstance(raw, bool):
            raise error("invalid_integrity_boolean", f"integrity.{name}", "must be boolean")
        booleans[name] = raw
    parsed = D1Integrity(
        status=status(integrity["status"], path="integrity.status"),
        roster_complete=booleans["roster_complete"],
        pairing_verified=booleans["pairing_verified"],
        data_order_verified=booleans["data_order_verified"],
        completion_verified=booleans["completion_verified"],
        evidence_hash=hash_value(integrity["evidence_hash"], path="integrity.evidence_hash"),
    )
    if parsed.status == "PASS" and not all(
        (
            parsed.roster_complete,
            parsed.pairing_verified,
            parsed.data_order_verified,
            parsed.completion_verified,
        )
    ):
        raise error(
            "inconsistent_integrity",
            "integrity",
            "PASS integrity requires every mandatory integrity boolean true",
        )
    return parsed


def _parse_exclusions(value: object) -> tuple[tuple[str, str, ContentHash], ...]:
    if not isinstance(value, list):
        raise error(
            "invalid_exclusions", "exclusions", "must be a list, including [] when none apply"
        )
    exclusions: list[tuple[str, str, ContentHash]] = []
    for index, raw_exclusion in enumerate(value):
        path = f"exclusions[{index}]"
        exclusion = object_exact(
            raw_exclusion, path=path, fields=frozenset({"subject", "reason", "evidence_hash"})
        )
        exclusions.append(
            (
                string(exclusion["subject"], path=f"{path}.subject"),
                string(exclusion["reason"], path=f"{path}.reason"),
                hash_value(exclusion["evidence_hash"], path=f"{path}.evidence_hash"),
            )
        )
    return tuple(exclusions)


def _parse_inputs(value: object) -> tuple[D1InputIdentity, ...]:
    if not isinstance(value, list) or not value:
        raise error("invalid_inputs", "inputs", "must be a nonempty MD5-bearing list")
    parsed: list[D1InputIdentity] = []
    seen: dict[str, D1InputIdentity] = {}
    for index, raw_input in enumerate(value):
        path = f"inputs[{index}]"
        item = object_optional(
            raw_input,
            path=path,
            required=frozenset({"path", "md5"}),
            optional=frozenset({"sha256"}),
        )
        identity = D1InputIdentity(
            path=string(item["path"], path=f"{path}.path"),
            md5=plain_hash(item["md5"], algorithm="md5", path=f"{path}.md5"),
            sha256=(
                plain_hash(item["sha256"], algorithm="sha256", path=f"{path}.sha256")
                if "sha256" in item
                else None
            ),
        )
        previous = seen.get(identity.path)
        if previous is not None and previous != identity:
            raise error(
                "input_hash_drift", identity.path, "one input path has conflicting content hashes"
            )
        seen[identity.path] = identity
        parsed.append(identity)
    return tuple(parsed)


def _parse_successor(value: object) -> D1Successor:
    successor = object_exact(
        value,
        path="successor",
        fields=frozenset({"authorization", "eligible_strategies"}),
    )
    raw_authorization = string(successor["authorization"], path="successor.authorization")
    try:
        authorization = SuccessorAuthorization(raw_authorization)
    except ValueError as exc:
        raise error(
            "invalid_successor_authorization",
            "successor.authorization",
            f"must be one of {[item.value for item in SuccessorAuthorization]}",
        ) from exc
    strategies = successor["eligible_strategies"]
    if not isinstance(strategies, list) or any(
        not isinstance(strategy, str) or not strategy for strategy in strategies
    ):
        raise error(
            "invalid_eligible_strategies",
            "successor.eligible_strategies",
            "must be a list of nonempty strategy identifiers",
        )
    return D1Successor(authorization=authorization, eligible_strategies=tuple(strategies))
