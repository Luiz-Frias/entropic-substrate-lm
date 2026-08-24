"""Guard teeth for the D1 lifecycle and filed-verdict gate.

These tests deliberately exercise the *negative* paths.  A lifecycle that
only accepts happy-path events cannot prove that its scientific-role and
content-addressing boundaries are load-bearing.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from entropic_substrate.evidence.d1_verdict import (
    D1_ARM_IDS,
    D1FamilyIdentity,
    D1VerdictValidationError,
    FiledGateAuthorization,
    FiledGateRetirement,
    FiledGateValidNegative,
    ValidatedD1Verdict,
    decide_filed_verdict,
)
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
    FiledVerdictEvidence,
    LifecycleEvent,
    LifecycleEventKind,
    LifecycleRecord,
    LifecycleState,
    LifecycleTransitionError,
    SuccessorAuthorization,
    transition,
)

pytestmark = pytest.mark.unit


def _digest(index: int) -> str:
    return f"{index:064x}"


def _hash(index: int) -> ContentHash:
    return ContentHash(algorithm="sha256", digest=_digest(index))


def _family_identity() -> D1FamilyIdentity:
    return D1FamilyIdentity(
        family_digest="f" * 64,
        arm_construction_digests={
            arm: _digest(600 + index) for index, arm in enumerate(D1_ARM_IDS)
        },
    )


def _validated(payload: dict[str, object]) -> ValidatedD1Verdict:
    return ValidatedD1Verdict.from_mapping(payload, family_identity=_family_identity())


def _artifact(name: str, index: int) -> ArtifactIdentity:
    return ArtifactIdentity(path=f"evidence/{name}.json", content_hash=_hash(index))


def _actor(role: ActorRole, identifier: str, session: str) -> ActorIdentity:
    return ActorIdentity(actor_id=identifier, session_id=session, role=role)


def _event(
    kind: LifecycleEventKind,
    *,
    actor: ActorIdentity,
    index: int,
    sealed_packet: ArtifactIdentity | None = None,
    filed_verdict: FiledVerdictEvidence | None = None,
) -> LifecycleEvent:
    artifacts = [_artifact(kind.value, index)]
    if sealed_packet is not None:
        artifacts.append(sealed_packet)
    if filed_verdict is not None:
        artifacts.extend([filed_verdict.narrative, filed_verdict.machine])
    return LifecycleEvent(
        event_id=f"event-{index}",
        kind=kind,
        actor=actor,
        artifacts=tuple(artifacts),
        sealed_packet=sealed_packet,
        filed_verdict=filed_verdict,
    )


def _to_confirmatory_running() -> LifecycleRecord:
    record = LifecycleRecord.start("d1")
    record = transition(
        record,
        _event(
            LifecycleEventKind.IMPLEMENT,
            actor=_actor(ActorRole.IMPLEMENTATION, "builder", "implementation-session"),
            index=1,
        ),
    )
    record = transition(
        record,
        _event(
            LifecycleEventKind.VERIFY_CONTRACT,
            actor=_actor(ActorRole.IMPLEMENTATION, "builder", "implementation-session"),
            index=2,
        ),
    )
    record = transition(
        record,
        _event(
            LifecycleEventKind.PREREGISTER,
            actor=_actor(ActorRole.OPERATOR, "operator", "registration-session"),
            index=3,
        ),
    )
    return transition(
        record,
        _event(
            LifecycleEventKind.START_CONFIRMATORY,
            actor=_actor(ActorRole.OPERATOR, "operator", "run-session"),
            index=4,
        ),
    )


def _to_analyzed_record(*, analysis_actor: ActorIdentity | None = None) -> LifecycleRecord:
    record = _to_confirmatory_running()
    record = transition(
        record,
        _event(
            LifecycleEventKind.CAPTURE_DVC,
            actor=_actor(ActorRole.AUTOMATION, "capture", "capture-session"),
            index=5,
        ),
    )
    actor = analysis_actor or _actor(ActorRole.ANALYSIS, "analyst", "analysis-session")
    return transition(record, _event(LifecycleEventKind.ANALYZE, actor=actor, index=6))


def _filed_evidence(
    *,
    open_major_findings: tuple[str, ...] = (),
    authorization: SuccessorAuthorization = SuccessorAuthorization.AUTHORIZED,
    eligible_strategies: tuple[str, ...] = ("d2-decoder-fisher",),
) -> FiledVerdictEvidence:
    return FiledVerdictEvidence(
        narrative=_artifact("d1-verdict", 20),
        machine=_artifact("d1-verdict-machine", 21),
        review_packet=_artifact("d1-review-packet", 22),
        open_major_findings=open_major_findings,
        successor_authorization=authorization,
        eligible_strategies=eligible_strategies,
    )


def _filed_record(*, evidence: FiledVerdictEvidence | None = None) -> LifecycleRecord:
    evidence = evidence or _filed_evidence()
    record = _to_analyzed_record()
    packet = evidence.review_packet
    record = transition(
        record,
        _event(
            LifecycleEventKind.BLIND_REVIEW,
            actor=_actor(ActorRole.REVIEW, "reviewer", "review-session"),
            index=7,
            sealed_packet=packet,
        ),
    )
    return transition(
        record,
        _event(
            LifecycleEventKind.FILE,
            actor=_actor(ActorRole.OPERATOR, "operator", "filing-session"),
            index=8,
            filed_verdict=evidence,
        ),
    )


def test_happy_path_requires_and_reaches_each_scientific_state() -> None:
    record = _filed_record()
    assert record.state is LifecycleState.FILED
    successor = transition(
        record,
        _event(
            LifecycleEventKind.AUTHORIZE_SUCCESSOR,
            actor=_actor(ActorRole.OPERATOR, "operator", "successor-session"),
            index=9,
        ),
    )
    assert successor.state is LifecycleState.SUCCESSOR_AUTHORIZED


def test_analysis_before_dvc_capture_is_an_illegal_transition() -> None:
    record = _to_confirmatory_running()
    with pytest.raises(LifecycleTransitionError, match="DvcCaptured") as excinfo:
        transition(
            record,
            _event(
                LifecycleEventKind.ANALYZE,
                actor=_actor(ActorRole.ANALYSIS, "analyst", "analysis-session"),
                index=10,
            ),
        )
    assert excinfo.value.context["source_state"] == LifecycleState.CONFIRMATORY_RUNNING.value
    assert excinfo.value.context["event_kind"] == LifecycleEventKind.ANALYZE.value


def test_implementation_actor_cannot_claim_fresh_analysis_in_same_lineage() -> None:
    record = _to_confirmatory_running()
    record = transition(
        record,
        _event(
            LifecycleEventKind.CAPTURE_DVC,
            actor=_actor(ActorRole.AUTOMATION, "capture", "capture-session"),
            index=11,
        ),
    )
    with pytest.raises(LifecycleTransitionError, match="fresh analysis"):
        transition(
            record,
            _event(
                LifecycleEventKind.ANALYZE,
                actor=_actor(ActorRole.ANALYSIS, "builder", "new-analysis-session"),
                index=12,
            ),
        )


def test_review_requires_a_sealed_packet_with_a_content_identity() -> None:
    record = _to_analyzed_record()
    with pytest.raises(LifecycleTransitionError, match="sealed review packet"):
        transition(
            record,
            _event(
                LifecycleEventKind.BLIND_REVIEW,
                actor=_actor(ActorRole.REVIEW, "reviewer", "review-session"),
                index=13,
            ),
        )


def test_hash_drift_is_rejected_for_an_existing_artifact_identity() -> None:
    record = LifecycleRecord.start("d1")
    proposal = ArtifactIdentity(path="evidence/proposal.json", content_hash=_hash(30))
    record = transition(
        record,
        LifecycleEvent(
            event_id="proposal",
            kind=LifecycleEventKind.IMPLEMENT,
            actor=_actor(ActorRole.IMPLEMENTATION, "builder", "implementation-session"),
            artifacts=(proposal,),
        ),
    )
    drifted = ArtifactIdentity(path="evidence/proposal.json", content_hash=_hash(31))
    with pytest.raises(LifecycleTransitionError, match="content hash drift"):
        transition(
            record,
            LifecycleEvent(
                event_id="contract",
                kind=LifecycleEventKind.VERIFY_CONTRACT,
                actor=_actor(ActorRole.IMPLEMENTATION, "builder", "implementation-session"),
                artifacts=(drifted,),
            ),
        )


def test_file_requires_both_narrative_and_machine_verdict_identities() -> None:
    record = _to_analyzed_record()
    packet = _artifact("review-packet", 40)
    record = transition(
        record,
        _event(
            LifecycleEventKind.BLIND_REVIEW,
            actor=_actor(ActorRole.REVIEW, "reviewer", "review-session"),
            index=41,
            sealed_packet=packet,
        ),
    )
    with pytest.raises(LifecycleTransitionError, match="filed verdict evidence"):
        transition(
            record,
            _event(
                LifecycleEventKind.FILE,
                actor=_actor(ActorRole.OPERATOR, "operator", "filing-session"),
                index=42,
            ),
        )


def test_open_major_finding_blocks_successor_authorization() -> None:
    record = _filed_record(evidence=_filed_evidence(open_major_findings=("MAJOR-1",)))
    with pytest.raises(LifecycleTransitionError, match="open MAJOR"):
        transition(
            record,
            _event(
                LifecycleEventKind.AUTHORIZE_SUCCESSOR,
                actor=_actor(ActorRole.OPERATOR, "operator", "successor-session"),
                index=50,
            ),
        )


def test_authorized_successor_requires_at_least_one_eligible_strategy() -> None:
    record = _filed_record(evidence=_filed_evidence(eligible_strategies=()))
    with pytest.raises(LifecycleTransitionError, match="eligible successor strategy"):
        transition(
            record,
            _event(
                LifecycleEventKind.AUTHORIZE_SUCCESSOR,
                actor=_actor(ActorRole.OPERATOR, "operator", "successor-session"),
                index=51,
            ),
        )


def _valid_payload() -> dict[str, object]:
    family_identity = _family_identity()
    arms = [
        {
            "arm_id": arm,
            "construction_digest": family_identity.arm_construction_digests[arm],
            "implementation": {"path": f"models/{arm}.pt", "content_hash": _hash(index).__dict__},
            "capture_manifest": {
                "path": f"results/captured/{arm}.manifest.json",
                "content_hash": _hash(index + 10).__dict__,
            },
        }
        for index, arm in enumerate(D1_ARM_IDS, start=60)
    ]
    return {
        "schema_version": "d1-verdict/v1",
        "rung": "d1",
        "preregistration": {
            "path": "docs/d1-preregistration.md",
            "commit": "a" * 40,
            "family_digest": family_identity.family_digest,
        },
        "registered_constants": {"seed_count": 4, "alpha": 0.05},
        "arms": arms,
        "contrasts": [
            {
                "contrast_id": "stop-gradient-at-no-regularizer",
                "contrast_kind": "PAIRWISE",
                "terms": [
                    {"arm_id": "sg", "coefficient": 1.0},
                    {"arm_id": "none", "coefficient": -1.0},
                ],
                "estimate": 0.0123456789012345,
                "sample_sd": 0.0200000000000000,
                "interval_lower": -0.0186543210987655,
                "interval_upper": 0.0433456789012345,
                "confidence_level": 0.975,
                "n": 4,
                "df": 3,
                "evidence_class": "DERIVED_FROM_MEASURED",
                "decision_vocabulary": "d1-contrast/v1",
                "decision": "INCONCLUSIVE",
                "evidence_hash": _hash(80).__dict__,
            }
        ],
        "clauses": {
            name: {
                "registered_rule": f"d1.{name}.v1",
                "decision_vocabulary": "d1-clause/v1",
                "threshold": 0.01,
                "margin": None,
                "status": "PASS",
                "evidence_hash": _hash(index).__dict__,
            }
            for index, name in enumerate(("noncollapse", "future_grounding", "language"), start=81)
        },
        "integrity": {
            "status": "PASS",
            "roster_complete": True,
            "pairing_verified": True,
            "data_order_verified": True,
            "completion_verified": True,
            "evidence_hash": _hash(90).__dict__,
        },
        "exclusions": [],
        "inputs": [
            {
                "path": "results/captured/d1-sg.jsonl",
                "md5": f"{91:032x}",
                "sha256": _hash(91).digest,
            }
        ],
        "review": {
            "packet": {"path": "docs/reviews/d1-packet.jsonl", "content_hash": _hash(92).__dict__},
            "sealed": True,
            "open_major_findings": [],
        },
        "filed_artifacts": {
            "narrative": {
                "path": "docs/verdicts/d1-verdict.md",
                "content_hash": _hash(93).__dict__,
            },
            "machine": {
                "path": "docs/verdicts/d1-verdict.json",
                "content_hash": _hash(94).__dict__,
            },
        },
        "verdict": "SUPPORTED",
        "successor": {
            "authorization": "AUTHORIZED",
            "eligible_strategies": ["d2-decoder-fisher"],
        },
    }


@pytest.mark.parametrize("field", tuple(_valid_payload()))
def test_verdict_rejects_every_required_top_level_field_when_missing(field: str) -> None:
    payload = _valid_payload()
    del payload[field]
    with pytest.raises(D1VerdictValidationError, match=field):
        _validated(payload)


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("preregistration", "path"),
        ("preregistration", "commit"),
        ("preregistration", "family_digest"),
        ("arms", "arm_id"),
        ("arms", "construction_digest"),
        ("arms", "implementation"),
        ("arms", "capture_manifest"),
        ("contrasts", "contrast_id"),
        ("contrasts", "contrast_kind"),
        ("contrasts", "terms"),
        ("contrasts", "estimate"),
        ("contrasts", "sample_sd"),
        ("contrasts", "interval_lower"),
        ("contrasts", "interval_upper"),
        ("contrasts", "confidence_level"),
        ("contrasts", "n"),
        ("contrasts", "df"),
        ("contrasts", "evidence_class"),
        ("contrasts", "decision_vocabulary"),
        ("contrasts", "decision"),
        ("contrasts", "evidence_hash"),
        ("integrity", "status"),
        ("integrity", "roster_complete"),
        ("integrity", "pairing_verified"),
        ("integrity", "data_order_verified"),
        ("integrity", "completion_verified"),
        ("integrity", "evidence_hash"),
        ("inputs", "path"),
        ("inputs", "md5"),
        ("review", "packet"),
        ("review", "sealed"),
        ("review", "open_major_findings"),
        ("filed_artifacts", "narrative"),
        ("filed_artifacts", "machine"),
        ("successor", "authorization"),
        ("successor", "eligible_strategies"),
    ],
)
def test_verdict_rejects_required_nested_fields_when_missing(container: str, field: str) -> None:
    payload = _valid_payload()
    nested = payload[container]
    if isinstance(nested, list):
        del nested[0][field]  # type: ignore[index]
    else:
        del nested[field]  # type: ignore[index]
    with pytest.raises(D1VerdictValidationError, match=field):
        _validated(payload)


@pytest.mark.parametrize("clause", ("noncollapse", "future_grounding", "language"))
def test_verdict_rejects_each_missing_required_clause(clause: str) -> None:
    payload = _valid_payload()
    clauses = payload["clauses"]
    assert isinstance(clauses, dict)
    del clauses[clause]
    with pytest.raises(D1VerdictValidationError, match=clause):
        _validated(payload)


def test_verdict_rejects_unknown_fields_and_hash_drift() -> None:
    payload = _valid_payload()
    payload["unregistered_interpretation"] = True
    with pytest.raises(D1VerdictValidationError, match="unknown"):
        _validated(payload)

    drifted = _valid_payload()
    arms = drifted["arms"]
    assert isinstance(arms, list)
    first_capture = arms[0]["capture_manifest"]
    assert isinstance(first_capture, dict)
    first_capture["path"] = arms[0]["implementation"]["path"]
    with pytest.raises(D1VerdictValidationError, match="hash drift"):
        _validated(drifted)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("arms", 0, "implementation", "content_hash"), "content_hash"),
        (("arms", 0, "implementation", "content_hash", "algorithm"), "algorithm"),
        (("arms", 0, "implementation", "content_hash", "digest"), "digest"),
        (("arms", 0, "implementation", "path"), "path"),
        (("arms", 0, "capture_manifest", "content_hash"), "content_hash"),
        (("contrasts", 0, "evidence_hash", "algorithm"), "algorithm"),
        (("contrasts", 0, "evidence_hash", "digest"), "digest"),
        (("clauses", "noncollapse", "status"), "status"),
        (("clauses", "noncollapse", "evidence_hash"), "evidence_hash"),
        (("review", "packet", "path"), "path"),
        (("review", "packet", "content_hash"), "content_hash"),
        (("filed_artifacts", "narrative", "path"), "path"),
        (("filed_artifacts", "machine", "content_hash"), "content_hash"),
    ],
)
def test_verdict_rejects_every_required_leaf_field_when_missing(
    path: tuple[str | int, ...], field: str
) -> None:
    payload = _valid_payload()
    current: object = payload
    for element in path[:-1]:
        if isinstance(element, int):
            assert isinstance(current, list)
            current = current[element]
        else:
            assert isinstance(current, dict)
            current = current[element]
    assert isinstance(current, dict)
    del current[path[-1]]
    with pytest.raises(D1VerdictValidationError, match=field):
        _validated(payload)


def test_authorized_verdict_rejects_open_major_and_missing_eligible_strategy() -> None:
    open_major = _valid_payload()
    review = open_major["review"]
    assert isinstance(review, dict)
    review["open_major_findings"] = ["MAJOR-1"]
    with pytest.raises(D1VerdictValidationError, match="MAJOR"):
        _validated(open_major)

    no_strategy = _valid_payload()
    successor = no_strategy["successor"]
    assert isinstance(successor, dict)
    successor["eligible_strategies"] = []
    with pytest.raises(D1VerdictValidationError, match="eligible"):
        _validated(no_strategy)


def test_validated_verdict_gate_authorizes_without_a_metric_input() -> None:
    verdict = _validated(_valid_payload())
    outcome = decide_filed_verdict(verdict)
    assert isinstance(outcome, FiledGateAuthorization)
    assert outcome.eligible_strategies == ("d2-decoder-fisher",)
    # The public gate takes exactly one validated artifact.  Raw observations
    # cannot be supplied to re-run a decision under a new calculation.
    with pytest.raises(TypeError):
        decide_filed_verdict(verdict, {"nats_per_byte": 0.1})  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("verdict_label", "authorization", "expected"),
    [
        ("INCONCLUSIVE", "VALID_NEGATIVE", FiledGateValidNegative),
        ("RETIRED", "RETIRED", FiledGateRetirement),
    ],
)
def test_validated_verdict_gate_preserves_human_decision_classes(
    verdict_label: str,
    authorization: str,
    expected: type[FiledGateValidNegative] | type[FiledGateRetirement],
) -> None:
    payload = deepcopy(_valid_payload())
    payload["verdict"] = verdict_label
    successor = payload["successor"]
    assert isinstance(successor, dict)
    successor["authorization"] = authorization
    successor["eligible_strategies"] = []
    outcome = decide_filed_verdict(_validated(payload))
    assert isinstance(outcome, expected)
