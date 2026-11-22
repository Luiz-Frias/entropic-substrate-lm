"""Decision-content validation for D1 verdict v2 without filing dependencies."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import replace

import pytest

from d1_verdict_v2_fixtures import decision_contract, machine_verdict
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.kernel.verdict.d1_types import (
    CLAUSE_STATUS,
    CONTRAST_DECISIONS,
    CONTRAST_KINDS,
    EVIDENCE_CLASSES,
    REVIEW_ADJUDICATION_STATUSES,
    REVIEW_FINDING_STATUSES,
    ClauseStatus,
    ContrastDecision,
    ContrastKind,
    D1Clause,
    D1Contrast,
    D1Integrity,
    D1ReviewAdjudication,
    D1ReviewFindingDisposition,
    D1VerdictValidationError,
    ReviewAdjudicationStatus,
    ReviewFindingStatus,
    VerdictEvidenceClass,
)
from entropic_substrate.ladder.file.d1_decision_validation import parse_review
from entropic_substrate.ladder.file.d1_filing import canonical_machine_verdict_bytes
from entropic_substrate.ladder.file.d1_real_arm_verdict import _contrast_decision
from entropic_substrate.ladder.file.d1_validator import ValidatedD1Verdict

pytestmark = pytest.mark.unit


def _validated(payload: dict[str, object]) -> ValidatedD1Verdict:
    return ValidatedD1Verdict.from_mapping(payload, decision_contract=decision_contract())


def _one_seed_payload() -> tuple[D1DecisionContract, dict[str, object]]:
    contract = decision_contract()
    one_seed_contract = replace(
        contract,
        registered_constants={"seeds": [1], "primary": {"alpha": 0.025}},
        contrast_rules={
            name: replace(rule, registered_n=1, df_rule="NOT_APPLICABLE")
            for name, rule in contract.contrast_rules.items()
        },
    )
    payload = machine_verdict()
    payload["registered_constants"] = {
        "seeds": [1],
        "primary": {"alpha": 0.025},
    }
    for contrast in payload["contrasts"]:  # type: ignore[union-attr]
        contrast["n"] = 1
        contrast["df"] = None
        contrast["sample_sd"] = None
        contrast["interval_lower"] = None
        contrast["interval_upper"] = None
    return one_seed_contract, payload


def test_v2_preserves_full_precision_scientific_content_without_filing_hashes() -> None:
    verdict = _validated(machine_verdict())
    assert verdict.schema_version == "d1-verdict/v2"
    assert verdict.contrasts[0].estimate == 0.0123456789012345
    assert verdict.contrasts[0].interval_lower == -0.02
    assert tuple(verdict.clauses_by_arm) == ("sg", "ac", "both")
    assert verdict.feasible_arms == ("sg", "ac", "both")
    assert verdict.successor.eligible_arms == verdict.feasible_arms
    assert verdict.review.adjudication.status == "NOT_REQUIRED"
    assert verdict.review.adjudication.execution_receipt_sha256 == f"{230:064x}"


def test_v2_represents_n1_interval_statistics_as_explicit_nulls() -> None:
    one_seed_contract, payload = _one_seed_payload()

    verdict = ValidatedD1Verdict.from_mapping(
        payload,
        decision_contract=one_seed_contract,
    )

    assert verdict.contrasts[0].interval_lower is None
    assert verdict.contrasts[0].interval_upper is None


@pytest.mark.parametrize(
    ("index", "decision"),
    [(0, "FAVORS_LEFT"), (2, "INTERACTION_DETECTED")],
)
def test_n1_null_intervals_require_inconclusive_decisions(index: int, decision: str) -> None:
    contract, payload = _one_seed_payload()
    payload["contrasts"][index]["decision"] = decision  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="null interval"):
        ValidatedD1Verdict.from_mapping(payload, decision_contract=contract)


def test_interval_presence_must_match_sample_size() -> None:
    contract, payload = _one_seed_payload()
    payload["contrasts"][0]["interval_lower"] = 0.0  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="both be finite"):
        ValidatedD1Verdict.from_mapping(payload, decision_contract=contract)

    contract, payload = _one_seed_payload()
    payload["contrasts"][0]["interval_upper"] = 0.0  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="both be finite"):
        ValidatedD1Verdict.from_mapping(payload, decision_contract=contract)

    contract, payload = _one_seed_payload()
    payload["contrasts"][0]["interval_lower"] = 0.0  # type: ignore[index]
    payload["contrasts"][0]["interval_upper"] = 0.0  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="explicit null"):
        ValidatedD1Verdict.from_mapping(payload, decision_contract=contract)

    payload = machine_verdict()
    payload["contrasts"][0]["interval_lower"] = None  # type: ignore[index]
    payload["contrasts"][0]["interval_upper"] = None  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="finite interval"):
        _validated(payload)


@pytest.mark.parametrize("field", ("filed_artifacts", "machine_sha256", "machine_md5"))
def test_v2_refuses_any_machine_self_hash_or_v1_artifact_field(field: str) -> None:
    payload = machine_verdict()
    payload[field] = "not-admissible"
    with pytest.raises(D1VerdictValidationError, match="unknown"):
        _validated(payload)


def test_v2_rejects_v1_schema_before_decision_interpretation() -> None:
    payload = machine_verdict()
    payload["schema_version"] = "d1-verdict/v1"
    with pytest.raises(D1VerdictValidationError, match="schema_version"):
        _validated(payload)


@pytest.mark.parametrize("feasible_arms", [["sg", "both"], ["none", "sg", "ac", "both"]])
def test_feasible_arms_must_equal_every_all_pass_clause_arm(
    feasible_arms: list[str],
) -> None:
    payload = machine_verdict()
    payload["feasible_arms"] = feasible_arms
    with pytest.raises(D1VerdictValidationError, match="feasible"):
        _validated(payload)


def test_non_authorized_successor_cannot_carry_an_eligible_arm() -> None:
    payload = machine_verdict(verdict="VALID_NEGATIVE")
    payload["successor"]["eligible_arms"] = ["sg"]  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match="eligible_arms"):
        _validated(payload)


def test_feasible_arms_preserve_canonical_stabilization_order() -> None:
    payload = machine_verdict()
    payload["feasible_arms"] = ["both", "sg", "ac"]
    with pytest.raises(D1VerdictValidationError, match="canonical order"):
        _validated(payload)


def test_single_decision_arm_cannot_replace_the_full_frontier() -> None:
    payload = machine_verdict()
    payload["decision_arm"] = "both"
    with pytest.raises(D1VerdictValidationError, match="unknown"):
        _validated(payload)


def test_negative_arm_cannot_downgrade_other_feasible_arms() -> None:
    payload = machine_verdict()
    payload["clauses_by_arm"]["sg"]["noncollapse"]["status"] = "FAIL"  # type: ignore[index]
    payload["clauses_by_arm"]["sg"]["future_grounding"]["status"] = (  # type: ignore[index]
        "INCONCLUSIVE"
    )
    payload["feasible_arms"] = ["ac", "both"]
    payload["verdict"] = "VALID_NEGATIVE"
    payload["successor"] = {
        "authorization": "VALID_NEGATIVE",
        "eligible_arms": [],
        "eligible_strategies": [],
    }
    with pytest.raises(D1VerdictValidationError, match="downgrade"):
        _validated(payload)


def test_contract_binds_sets_constants_integrity_and_decision_table() -> None:
    changed: dict[str, object] = json.loads(json.dumps(machine_verdict()))
    changed["registered_constants"]["seeds"] = [1, 2]
    with pytest.raises(D1VerdictValidationError, match="registered_constants"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    changed["integrity"]["pairing_verified"] = False
    with pytest.raises(D1VerdictValidationError, match="integrity"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    for arm in ("sg", "ac", "both"):
        changed["clauses_by_arm"][arm]["noncollapse"]["status"] = "FAIL"
        changed["clauses_by_arm"][arm]["future_grounding"]["status"] = "INCONCLUSIVE"
    changed["feasible_arms"] = []
    changed["successor"]["eligible_arms"] = []
    with pytest.raises(D1VerdictValidationError, match="decision"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    changed["arms"] = changed["arms"][:-1]
    with pytest.raises(D1VerdictValidationError, match="arm"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    changed["contrasts"] = changed["contrasts"][:-1]
    with pytest.raises(D1VerdictValidationError, match="contrast_set"):
        _validated(changed)

    changed = json.loads(json.dumps(machine_verdict()))
    del changed["clauses_by_arm"]["both"]["language"]
    with pytest.raises(D1VerdictValidationError, match="clause"):
        _validated(changed)


def test_review_major_requires_adjudication_consistent_with_registered_clearance() -> None:
    payload = machine_verdict()
    review = payload["review"]
    review["open_major_findings"] = ["MAJOR-1"]  # type: ignore[index]
    adjudication = review["adjudication"]  # type: ignore[index]
    adjudication["major_finding_ids"] = ["MAJOR-1"]  # type: ignore[index]
    adjudication["open_major_finding_ids"] = ["MAJOR-1"]  # type: ignore[index]
    adjudication["status"] = "OPEN"  # type: ignore[index]
    adjudication["dispositions"] = [  # type: ignore[index]
        {
            "evidence_sha256": None,
            "finding_id": "MAJOR-1",
            "rationale": "Independent recomputation remains open.",
            "status": "OPEN",
        }
    ]
    with pytest.raises(D1VerdictValidationError, match="review"):
        _validated(payload)


# FOR-1573: the closed D1 vocabularies are StrEnum types whose wire values are the
# parent-tip (119081a) literals, so persisted bytes and the read-old window hold.
_PARENT_TIP_CLAUSE_STATUS = frozenset({"PASS", "FAIL", "INCONCLUSIVE"})
_PARENT_TIP_EVIDENCE_CLASSES = frozenset(
    {"MEASURED", "DERIVED_FROM_MEASURED", "MODEL_ESTIMATE", "RATE_ESTIMATE", "PROVIDER_BILLED"}
)
_PARENT_TIP_CONTRAST_KINDS = frozenset({"PAIRWISE", "FACTORIAL_INTERACTION"})
_PARENT_TIP_CONTRAST_DECISIONS = frozenset(
    {
        "FAVORS_LEFT",
        "FAVORS_RIGHT",
        "PRACTICALLY_NULL",
        "INCONCLUSIVE",
        "INTERACTION_DETECTED",
        "INTERACTION_INCONCLUSIVE",
    }
)
_PARENT_TIP_ADJUDICATION_STATUS = frozenset({"OPEN", "RESOLVED", "NOT_REQUIRED"})
_PARENT_TIP_DISPOSITION_STATUS = frozenset({"OPEN", "RESOLVED"})

_ENUMS = (
    (ClauseStatus, _PARENT_TIP_CLAUSE_STATUS),
    (VerdictEvidenceClass, _PARENT_TIP_EVIDENCE_CLASSES),
    (ContrastKind, _PARENT_TIP_CONTRAST_KINDS),
    (ContrastDecision, _PARENT_TIP_CONTRAST_DECISIONS),
    (ReviewAdjudicationStatus, _PARENT_TIP_ADJUDICATION_STATUS),
    (ReviewFindingStatus, _PARENT_TIP_DISPOSITION_STATUS),
)


@pytest.mark.parametrize(
    ("legacy", "literal"),
    [
        (CLAUSE_STATUS, _PARENT_TIP_CLAUSE_STATUS),
        (EVIDENCE_CLASSES, _PARENT_TIP_EVIDENCE_CLASSES),
        (CONTRAST_KINDS, _PARENT_TIP_CONTRAST_KINDS),
        (CONTRAST_DECISIONS, _PARENT_TIP_CONTRAST_DECISIONS),
        (REVIEW_ADJUDICATION_STATUSES, _PARENT_TIP_ADJUDICATION_STATUS),
        (REVIEW_FINDING_STATUSES, _PARENT_TIP_DISPOSITION_STATUS),
    ],
)
def test_legacy_vocabulary_sets_equal_the_parent_tip_literals_as_plain_strings(
    legacy: frozenset[str], literal: frozenset[str]
) -> None:
    # d1_parsing renders sorted(CLAUSE_STATUS) into a caller-visible error, so
    # a member must stay a plain str, never an enum member.
    assert legacy == literal
    assert type(legacy) is frozenset
    assert all(type(member) is str for member in legacy)


@pytest.mark.parametrize(("vocabulary", "literal"), _ENUMS)
def test_enum_values_are_the_parent_tip_wire_literals_and_names_mirror_them(
    vocabulary: type, literal: frozenset[str]
) -> None:
    assert {member.value for member in vocabulary} == literal
    assert all(type(member.value) is str for member in vocabulary)
    assert all(member.name == member.value for member in vocabulary)
    assert all(member == member.value for member in vocabulary)
    assert all(json.dumps(member) == json.dumps(member.value) for member in vocabulary)
    assert all(str(member) == member.value and f"{member}" == member.value for member in vocabulary)


@pytest.mark.parametrize(
    ("kind", "lower", "upper", "margin", "wire"),
    [
        ("PAIRWISE", None, None, 0.1, "INCONCLUSIVE"),
        ("FACTORIAL_INTERACTION", None, None, 0.1, "INTERACTION_INCONCLUSIVE"),
        ("PAIRWISE", 0.2, 0.5, 0.1, "FAVORS_LEFT"),
        ("PAIRWISE", -0.5, -0.2, 0.1, "FAVORS_RIGHT"),
        ("PAIRWISE", -0.05, 0.05, 0.1, "PRACTICALLY_NULL"),
        ("PAIRWISE", -0.2, 0.05, 0.1, "INCONCLUSIVE"),
        ("FACTORIAL_INTERACTION", 0.2, 0.5, 0.1, "INTERACTION_DETECTED"),
        ("FACTORIAL_INTERACTION", -0.5, -0.2, 0.1, "INTERACTION_DETECTED"),
        ("FACTORIAL_INTERACTION", -0.05, 0.05, 0.1, "INTERACTION_INCONCLUSIVE"),
        # Interval limits exactly at +/- margin: strict outside (>, <), inclusive null (>=, <=).
        ("PAIRWISE", 0.1, 0.5, 0.1, "INCONCLUSIVE"),
        ("PAIRWISE", -0.5, -0.1, 0.1, "INCONCLUSIVE"),
        ("PAIRWISE", -0.1, 0.05, 0.1, "PRACTICALLY_NULL"),
        ("PAIRWISE", -0.05, 0.1, 0.1, "PRACTICALLY_NULL"),
        ("FACTORIAL_INTERACTION", 0.1, 0.5, 0.1, "INTERACTION_INCONCLUSIVE"),
        ("FACTORIAL_INTERACTION", -0.5, -0.1, 0.1, "INTERACTION_INCONCLUSIVE"),
    ],
)
def test_contrast_decision_interval_boundary_cases(
    kind: str, lower: float | None, upper: float | None, margin: float, wire: str
) -> None:
    result = _contrast_decision(kind, {"interval_lower": lower, "interval_upper": upper}, margin)
    assert result == wire
    assert type(result) is str


def _integrity_status_contract(*, require_pass: bool) -> D1DecisionContract:
    contract = decision_contract()
    # replace() re-runs __post_init__, which expects the unfrozen constants (as the fixture builds them).
    return replace(
        contract,
        registered_constants={"seeds": [1, 2, 3, 4], "primary": {"alpha": 0.025}},
        decision_rules={
            name: replace(rule, require_integrity_pass=require_pass)
            for name, rule in contract.decision_rules.items()
        },
    )


@pytest.mark.parametrize("raw", ["PASS", "FAIL", "INCONCLUSIVE"])
def test_every_integrity_status_parses_to_its_own_member(raw: str) -> None:
    # The registered decision rules normally require PASS integrity, which hides FAIL and
    # INCONCLUSIVE; relaxing that requirement observes each member through the parser, so a
    # wrong-member mapping (for example FAIL parsed as PASS) goes RED here.
    payload = machine_verdict()
    _set(payload, ("integrity", "status"), raw)
    verdict = ValidatedD1Verdict.from_mapping(
        payload, decision_contract=_integrity_status_contract(require_pass=False)
    )
    assert type(verdict.integrity.status) is ClauseStatus
    assert verdict.integrity.status is ClauseStatus(raw)
    assert verdict.integrity.status.value == raw


@pytest.mark.parametrize("raw", ["FAIL", "INCONCLUSIVE"])
def test_a_pass_requiring_decision_rule_rejects_a_non_pass_integrity_status(raw: str) -> None:
    payload = machine_verdict()
    _set(payload, ("integrity", "status"), raw)
    with pytest.raises(D1VerdictValidationError) as raised:
        ValidatedD1Verdict.from_mapping(
            payload, decision_contract=_integrity_status_contract(require_pass=True)
        )
    assert raised.value.code == "decision_integrity_mismatch"


def test_the_verdict_evidence_class_is_not_the_lowercase_identity_evidence_class() -> None:
    from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass

    assert VerdictEvidenceClass is not EvidenceClass
    assert {member.value for member in VerdictEvidenceClass}.isdisjoint(
        {member.value for member in EvidenceClass}
    )


def _verdict() -> ValidatedD1Verdict:
    return _validated(machine_verdict())


def _contrast_with(**over: object) -> D1Contrast:
    return replace(_verdict().contrasts[0], **over)


def _clause_with(**over: object) -> D1Clause:
    return replace(_verdict().clauses_by_arm["sg"]["noncollapse"], **{"status": over["value"]})


def _integrity_with(**over: object) -> D1Integrity:
    return replace(_verdict().integrity, status=over["value"])


def _adjudication_with(**over: object) -> D1ReviewAdjudication:
    return replace(_verdict().review.adjudication, status=over["value"])


def _disposition_with(**over: object) -> D1ReviewFindingDisposition:
    return D1ReviewFindingDisposition(
        finding_id="MAJOR-1", status=over["value"], rationale="r", evidence_sha256=None
    )


_FIELDS: dict[str, Callable[..., object]] = {
    "contrast_kind": _contrast_with,
    "evidence_class": _contrast_with,
    "decision": _contrast_with,
    "clause_status": _clause_with,
    "integrity_status": _integrity_with,
    "adjudication_status": _adjudication_with,
    "disposition_status": _disposition_with,
}


def _build(field: str, value: object) -> object:
    if field in {"contrast_kind", "evidence_class", "decision"}:
        return _FIELDS[field](**{field: value})
    return _FIELDS[field](value=value)


@pytest.mark.parametrize("field", list(_FIELDS))
@pytest.mark.parametrize("bad", ["MAYBE", "", "pass", " PASS", None, 3, ["PASS"]])
def test_an_out_of_vocabulary_value_raises_at_construction(field: str, bad: object) -> None:
    with pytest.raises(TypeError):
        _build(field, bad)


_VALID_RAW = {
    "contrast_kind": "PAIRWISE",
    "evidence_class": "MEASURED",
    "decision": "FAVORS_LEFT",
    "clause_status": "PASS",
    "integrity_status": "PASS",
    "adjudication_status": "NOT_REQUIRED",
    "disposition_status": "OPEN",
}


@pytest.mark.parametrize("field", list(_FIELDS))
def test_a_raw_string_inside_the_vocabulary_is_refused_nominally(field: str) -> None:
    # The door is nominal: only the parsers map str to member, so a lenient
    # coercion at construction (which would accept this value) goes RED here.
    with pytest.raises(TypeError, match="must be"):
        _build(field, _VALID_RAW[field])


def test_decision_maybe_raises_at_construction() -> None:
    with pytest.raises(TypeError, match=r"D1Contrast\.decision"):
        replace(_verdict().contrasts[0], decision="MAYBE")


def test_the_two_review_vocabularies_do_not_merge() -> None:
    # A disposition cannot accept a member of adjudication_status, even if the value matches.
    with pytest.raises(TypeError):
        D1ReviewFindingDisposition(
            finding_id="MAJOR-1",
            status=ReviewAdjudicationStatus.OPEN,  # Same value "OPEN", wrong enum type
            rationale="r",
            evidence_sha256=None,
        )
    # Both enums have their own members.
    adjudication = _adjudication_with(value=ReviewAdjudicationStatus.NOT_REQUIRED)
    assert adjudication.status is ReviewAdjudicationStatus.NOT_REQUIRED
    for member in (ReviewFindingStatus.OPEN, ReviewFindingStatus.RESOLVED):
        disposition = _disposition_with(value=member)
        assert disposition.status is member


def test_construction_holds_the_member_and_it_equals_the_old_string() -> None:
    verdict = _verdict()
    contrast = verdict.contrasts[0]
    assert type(contrast.contrast_kind) is ContrastKind and contrast.contrast_kind == "PAIRWISE"
    assert type(contrast.evidence_class) is VerdictEvidenceClass
    assert contrast.evidence_class == "DERIVED_FROM_MEASURED"
    assert type(contrast.decision) is ContrastDecision and contrast.decision == "INCONCLUSIVE"
    clause = verdict.clauses_by_arm["sg"]["noncollapse"]
    assert type(clause.status) is ClauseStatus and clause.status == "PASS"
    assert type(verdict.integrity.status) is ClauseStatus and verdict.integrity.status == "PASS"
    adjudication = verdict.review.adjudication
    assert type(adjudication.status) is ReviewAdjudicationStatus
    assert adjudication.status == "NOT_REQUIRED"
    assert (
        replace(contrast, decision=ContrastDecision.FAVORS_LEFT).decision
        is ContrastDecision.FAVORS_LEFT
    )


def _set(payload: dict[str, object], path: tuple[object, ...], value: object) -> None:
    node: object = payload
    for key in path[:-1]:
        node = node[key]  # type: ignore[index]
    node[path[-1]] = value  # type: ignore[index]


@pytest.mark.parametrize(
    ("path", "code"),
    [
        (("contrasts", 0, "contrast_kind"), "invalid_contrast_kind"),
        (("contrasts", 0, "evidence_class"), "invalid_evidence_class"),
        (("contrasts", 0, "decision"), "invalid_contrast_decision"),
        (("clauses_by_arm", "sg", "language", "status"), "invalid_status"),
        (("integrity", "status"), "invalid_status"),
        (("review", "adjudication", "status"), "invalid_adjudication_status"),
    ],
)
def test_the_parser_membership_check_still_runs_before_the_enum_is_built(
    path: tuple[object, ...], code: str
) -> None:
    payload = machine_verdict()
    _set(payload, path, "MAYBE")
    with pytest.raises(D1VerdictValidationError) as raised:
        _validated(payload)
    assert raised.value.code == code
    if code == "invalid_status":
        assert "['FAIL', 'INCONCLUSIVE', 'PASS']" in str(raised.value)


def test_a_disposition_outside_open_or_resolved_keeps_its_parser_code() -> None:
    review = _review("OPEN")
    review["adjudication"]["dispositions"][0]["status"] = "NOT_REQUIRED"  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError) as raised:
        parse_review(review)
    assert raised.value.code == "invalid_adjudication"


def _review(status: str) -> dict[str, object]:
    findings = [] if status == "NOT_REQUIRED" else ["MAJOR-1"]
    dispositions = (
        []
        if status == "NOT_REQUIRED"
        else [
            {
                "evidence_sha256": None if status == "OPEN" else f"{240:064x}",
                "finding_id": "MAJOR-1",
                "rationale": "Independent recomputation.",
                "status": status,
            }
        ]
    )
    return {
        "open_major_findings": findings if status == "OPEN" else [],
        "adjudication": {
            "dispositions": dispositions,
            "execution_receipt_sha256": f"{230:064x}",
            "major_finding_ids": findings,
            "open_major_finding_ids": findings if status == "OPEN" else [],
            "operator_record_sha256": f"{231:064x}",
            "report_sha256": f"{232:064x}",
            "review_packet_sha256": f"{233:064x}",
            "review_request_sha256": f"{234:064x}",
            "schema_version": "d1-review-adjudication-v1",
            "status": status,
        },
    }


def _rebuild_vocabulary(
    payload: dict[str, object], verdict: ValidatedD1Verdict
) -> dict[str, object]:
    """Blank every vocabulary field of the persisted shape, then refill it from the
    parsed value objects through ``.value`` (the serialization contract)."""

    rebuilt = copy.deepcopy(payload)
    for index, contrast in enumerate(verdict.contrasts):
        _set(rebuilt, ("contrasts", index, "contrast_kind"), contrast.contrast_kind.value)
        _set(rebuilt, ("contrasts", index, "evidence_class"), contrast.evidence_class.value)
        _set(rebuilt, ("contrasts", index, "decision"), contrast.decision.value)
    for arm, clauses in verdict.clauses_by_arm.items():
        for name, clause in clauses.items():
            _set(rebuilt, ("clauses_by_arm", arm, name, "status"), clause.status.value)
    _set(rebuilt, ("integrity", "status"), verdict.integrity.status.value)
    adjudication = verdict.review.adjudication
    _set(rebuilt, ("review", "adjudication", "status"), adjudication.status.value)
    return rebuilt


def _blanked(payload: dict[str, object]) -> dict[str, object]:
    blank = copy.deepcopy(payload)
    for index in range(len(blank["contrasts"])):  # type: ignore[arg-type]
        for key in ("contrast_kind", "evidence_class", "decision"):
            _set(blank, ("contrasts", index, key), "UNSET")
    for arm, clauses in blank["clauses_by_arm"].items():  # type: ignore[attr-defined]
        for name in clauses:
            _set(blank, ("clauses_by_arm", arm, name, "status"), "UNSET")
    _set(blank, ("integrity", "status"), "UNSET")
    _set(blank, ("review", "adjudication", "status"), "UNSET")
    return blank


@pytest.mark.parametrize("verdict_name", ["SUPPORTED", "VALID_NEGATIVE"])
def test_read_old_fixture_literals_parse_and_round_trip_byte_identically(
    verdict_name: str,
) -> None:
    # The oracle is the fixture literals (plain strings, never rebuilt from members).
    payload = machine_verdict(verdict=verdict_name)
    verdict = _validated(payload)
    rebuilt = _rebuild_vocabulary(_blanked(payload), verdict)
    for contrast in verdict.contrasts:
        assert type(contrast.decision.value) is str
    assert canonical_machine_verdict_bytes(rebuilt) == canonical_machine_verdict_bytes(payload)


@pytest.mark.parametrize("evidence_class", sorted(_PARENT_TIP_EVIDENCE_CLASSES))
@pytest.mark.parametrize("decision", sorted(_PARENT_TIP_CONTRAST_DECISIONS))
def test_every_registered_evidence_class_and_decision_still_parses_and_round_trips(
    evidence_class: str, decision: str
) -> None:
    payload = machine_verdict()
    for index in range(3):
        _set(payload, ("contrasts", index, "evidence_class"), evidence_class)
        _set(payload, ("contrasts", index, "decision"), decision)
    verdict = _validated(payload)
    assert {c.evidence_class.value for c in verdict.contrasts} == {evidence_class}
    rebuilt = _rebuild_vocabulary(_blanked(payload), verdict)
    assert canonical_machine_verdict_bytes(rebuilt) == canonical_machine_verdict_bytes(payload)


@pytest.mark.parametrize("status", ["NOT_REQUIRED", "OPEN", "RESOLVED"])
def test_every_review_status_still_parses_and_round_trips(status: str) -> None:
    review = _review(status)
    parsed = parse_review(review)
    assert isinstance(parsed.adjudication, D1ReviewAdjudication)
    assert parsed.adjudication.status.value == status
    rebuilt = copy.deepcopy(review)
    rebuilt["adjudication"]["status"] = parsed.adjudication.status.value  # type: ignore[index]
    for item, disposition in zip(
        rebuilt["adjudication"]["dispositions"],  # type: ignore[index]
        parsed.adjudication.dispositions.values(),
        strict=True,
    ):
        item["status"] = disposition.status.value  # type: ignore[index]
        assert type(disposition.status) is ReviewFindingStatus
    assert canonical_machine_verdict_bytes(rebuilt) == canonical_machine_verdict_bytes(review)
