"""Second-pass guard teeth for D1 evidence provenance and numeric verdicts."""

from __future__ import annotations

from math import inf
from types import MappingProxyType

import pytest

from entropic_substrate.evidence.d1_family_adapter import family_identity_from_experiment
from entropic_substrate.evidence.d1_verdict import (
    D1_ARM_IDS,
    D1FamilyIdentity,
    D1VerdictValidationError,
    ValidatedD1Verdict,
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
from entropic_substrate.experiments.d1 import (
    CausalEncoderSpec,
    EncoderInitializationIdentity,
    MatchedEncoderPairSpec,
    build_d1_experiment_family,
)
from entropic_substrate.observation import CodecIdentity

pytestmark = pytest.mark.unit


def _digest(index: int, *, width: int = 64) -> str:
    return f"{index:0{width}x}"


def _hash(index: int) -> dict[str, str]:
    return {"algorithm": "sha256", "digest": _digest(index)}


def _artifact(path: str, index: int) -> dict[str, object]:
    return {"path": path, "content_hash": _hash(index)}


def _family_identity() -> D1FamilyIdentity:
    return D1FamilyIdentity(
        family_digest="e" * 64,
        arm_construction_digests={
            arm: _digest(700 + index) for index, arm in enumerate(D1_ARM_IDS)
        },
    )


def _validated(payload: dict[str, object]) -> ValidatedD1Verdict:
    return ValidatedD1Verdict.from_mapping(payload, family_identity=_family_identity())


def _real_family_identity() -> D1FamilyIdentity:
    codec = CodecIdentity(name="identity-byte", construction_digest="a" * 64, cardinality=256)
    encoder = CausalEncoderSpec(
        latent_width=64,
        n_layers=2,
        n_heads=4,
        mlp_ratio=4.0,
        block_size=256,
        dropout=0.0,
        bias=False,
        rms_epsilon=1e-5,
        initialization=EncoderInitializationIdentity(
            name="evidence-integration", construction_digest="b" * 64
        ),
    )
    family = build_d1_experiment_family(
        codec=codec,
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
    )
    return family_identity_from_experiment(family)


def _strict_payload() -> dict[str, object]:
    family_identity = _family_identity()
    arms = [
        {
            "arm_id": arm,
            "construction_digest": family_identity.arm_construction_digests[arm],
            "implementation": _artifact(f"models/d1-{arm}.pt", 100 + index),
            "capture_manifest": _artifact(f"results/captured/d1-{arm}.manifest.json", 110 + index),
        }
        for index, arm in enumerate(D1_ARM_IDS)
    ]
    return {
        "schema_version": "d1-verdict/v1",
        "rung": "d1",
        "preregistration": {
            "path": "docs/d1-preregistration.md",
            "commit": "a" * 40,
            "family_digest": family_identity.family_digest,
        },
        "registered_constants": {
            "seed_count": 4,
            "horizons": [1, 4, 16, 64],
            "target": {"variance_floor": 0.001, "enabled": True},
        },
        "arms": arms,
        "contrasts": [
            {
                "contrast_id": "target-main-effect",
                "contrast_kind": "PAIRWISE",
                "terms": [
                    {"arm_id": "sg", "coefficient": 1.0},
                    {"arm_id": "none", "coefficient": -1.0},
                ],
                "estimate": 0.1234567890123456,
                "sample_sd": 0.0123456789012345,
                "interval_lower": 0.1000000000000001,
                "interval_upper": 0.1469135780246911,
                "confidence_level": 0.975,
                "n": 4,
                "df": 3,
                "evidence_class": "DERIVED_FROM_MEASURED",
                "decision_vocabulary": "d1-contrast/v1",
                "decision": "FAVORS_LEFT",
                "evidence_hash": _hash(120),
            },
            {
                "contrast_id": "stabilizer-interaction",
                "contrast_kind": "FACTORIAL_INTERACTION",
                "terms": [
                    {"arm_id": "none", "coefficient": 1.0},
                    {"arm_id": "sg", "coefficient": -1.0},
                    {"arm_id": "ac", "coefficient": -1.0},
                    {"arm_id": "both", "coefficient": 1.0},
                ],
                "estimate": -0.0100000000000001,
                "sample_sd": 0.0200000000000002,
                "interval_lower": -0.0410000000000002,
                "interval_upper": 0.0209999999999999,
                "confidence_level": 0.975,
                "n": 4,
                "df": 3,
                "evidence_class": "DERIVED_FROM_MEASURED",
                "decision_vocabulary": "d1-contrast/v1",
                "decision": "INTERACTION_INCONCLUSIVE",
                "evidence_hash": _hash(121),
            },
        ],
        "clauses": {
            name: {
                "registered_rule": f"d1.{name}.rule.v1",
                "decision_vocabulary": "d1-clause/v1",
                "threshold": 0.01,
                "margin": None,
                "status": "PASS",
                "evidence_hash": _hash(130 + index),
            }
            for index, name in enumerate(("noncollapse", "future_grounding", "language"))
        },
        "integrity": {
            "status": "PASS",
            "roster_complete": True,
            "pairing_verified": True,
            "data_order_verified": True,
            "completion_verified": True,
            "evidence_hash": _hash(140),
        },
        "exclusions": [],
        "inputs": [
            {
                "path": "results/captured/d1-both-s1.jsonl",
                "md5": _digest(150, width=32),
                "sha256": _digest(151),
            }
        ],
        "review": {
            "packet": _artifact("docs/reviews/d1-packet.jsonl", 160),
            "sealed": True,
            "open_major_findings": [],
        },
        "filed_artifacts": {
            "narrative": _artifact("docs/verdicts/d1-verdict.md", 161),
            "machine": _artifact("docs/verdicts/d1-verdict.json", 162),
        },
        "verdict": "SUPPORTED",
        "successor": {
            "authorization": "AUTHORIZED",
            "eligible_strategies": ["d2-decoder-fisher"],
        },
    }


def test_canonical_four_arm_evidence_schema_preserves_full_precision() -> None:
    verdict = _validated(_strict_payload())
    assert tuple(arm.arm_id for arm in verdict.arms) == D1_ARM_IDS
    assert verdict.preregistration.family_digest == _family_identity().family_digest
    assert verdict.contrasts[0].estimate == 0.1234567890123456
    assert verdict.contrasts[0].interval_lower == 0.1000000000000001
    assert verdict.contrasts[1].contrast_kind == "FACTORIAL_INTERACTION"
    assert tuple(term.arm_id for term in verdict.contrasts[1].terms) == D1_ARM_IDS
    assert verdict.inputs[0].md5.digest == _digest(150, width=32)
    assert verdict.inputs[0].sha256 is not None
    assert verdict.inputs[0].sha256.digest == _digest(151)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["preregistration"].__setitem__("family_digest", _digest(999)),
            "family",
        ),
        (
            lambda payload: payload["arms"].__setitem__(
                0, {**payload["arms"][0], "arm_id": "made-up"}
            ),
            "arm",
        ),
        (
            lambda payload: payload["arms"][0].__setitem__("construction_digest", _digest(998)),
            "construction",
        ),
        (lambda payload: payload["arms"][0].pop("capture_manifest"), "capture_manifest"),
        (lambda payload: payload["contrasts"][0].__setitem__("estimate", True), "estimate"),
        (lambda payload: payload["contrasts"][0].__setitem__("estimate", inf), "estimate"),
        (
            lambda payload: payload["contrasts"][0].__setitem__("confidence_level", 1.0),
            "confidence",
        ),
        (lambda payload: payload["contrasts"][0].__setitem__("interval_lower", 1.0), "interval"),
        (
            lambda payload: payload["contrasts"][0]["terms"][0].__setitem__("arm_id", "unknown"),
            "arm",
        ),
        (lambda payload: payload["inputs"][0].pop("md5"), "md5"),
    ],
)
def test_evidence_schema_rejects_mutated_construction_and_numeric_claims(
    mutate: object, match: str
) -> None:
    payload = _strict_payload()
    assert callable(mutate)
    mutate(payload)
    with pytest.raises(D1VerdictValidationError, match=match):
        _validated(payload)


def test_registered_constants_are_recursive_finite_and_immutable() -> None:
    verdict = _validated(_strict_payload())
    assert isinstance(verdict.registered_constants, MappingProxyType)
    horizons = verdict.registered_constants["horizons"]
    target = verdict.registered_constants["target"]
    assert horizons == (1, 4, 16, 64)
    assert isinstance(target, MappingProxyType)
    with pytest.raises(TypeError):
        verdict.registered_constants["new"] = "forbidden"  # type: ignore[index]
    with pytest.raises(TypeError):
        target["variance_floor"] = 0.1  # type: ignore[index]


def test_real_experiment_family_identity_binds_every_verdict_arm_and_refuses_drift() -> None:
    identity = _real_family_identity()
    payload = _strict_payload()
    preregistration = payload["preregistration"]
    arms = payload["arms"]
    assert isinstance(preregistration, dict)
    assert isinstance(arms, list)
    preregistration["family_digest"] = identity.family_digest
    for arm in arms:
        arm["construction_digest"] = identity.arm_construction_digests[arm["arm_id"]]

    verdict = ValidatedD1Verdict.from_mapping(payload, family_identity=identity)
    assert verdict.preregistration.family_digest == identity.family_digest
    assert tuple(arm.construction_digest for arm in verdict.arms) == tuple(
        identity.arm_construction_digests[arm] for arm in D1_ARM_IDS
    )

    drifted = _strict_payload()
    drifted_preregistration = drifted["preregistration"]
    drifted_arms = drifted["arms"]
    assert isinstance(drifted_preregistration, dict)
    assert isinstance(drifted_arms, list)
    drifted_preregistration["family_digest"] = identity.family_digest
    for arm in drifted_arms:
        arm["construction_digest"] = identity.arm_construction_digests[arm["arm_id"]]
    drifted_arms[0]["construction_digest"] = "f" * 64
    with pytest.raises(D1VerdictValidationError, match="construction"):
        ValidatedD1Verdict.from_mapping(drifted, family_identity=identity)


@pytest.mark.parametrize(
    ("path", "field"),
    [
        (("arms", 0, "capture_manifest", "content_hash"), "content_hash"),
        (("arms", 0, "capture_manifest", "content_hash", "algorithm"), "algorithm"),
        (("arms", 0, "capture_manifest", "content_hash", "digest"), "digest"),
        (("contrasts", 0, "terms", 0, "arm_id"), "arm_id"),
        (("contrasts", 0, "terms", 0, "coefficient"), "coefficient"),
        (("clauses", "noncollapse", "registered_rule"), "registered_rule"),
        (("clauses", "noncollapse", "decision_vocabulary"), "decision_vocabulary"),
        (("clauses", "noncollapse", "threshold"), "threshold"),
        (("clauses", "noncollapse", "margin"), "margin"),
        (("clauses", "noncollapse", "status"), "status"),
        (("clauses", "noncollapse", "evidence_hash"), "evidence_hash"),
        (("inputs", 0, "sha256"), "sha256"),
    ],
)
def test_required_evidence_fields_cannot_be_silently_dropped(
    path: tuple[str | int, ...], field: str
) -> None:
    payload = _strict_payload()
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
    if field == "sha256":
        # SHA-256 is intentionally optional; MD5 stays the required protocol
        # identity and the absent stronger digest is not a hash-drift event.
        assert _validated(payload).inputs[0].sha256 is None
    else:
        with pytest.raises(D1VerdictValidationError, match=field):
            _validated(payload)


def _artifact_identity(name: str, index: int) -> ArtifactIdentity:
    return ArtifactIdentity(
        path=f"evidence/{name}.json", content_hash=ContentHash("sha256", _digest(index))
    )


def _actor(role: ActorRole) -> ActorIdentity:
    return ActorIdentity(
        actor_id=f"{role.value}-actor", session_id=f"{role.value}-session", role=role
    )


def _event(kind: LifecycleEventKind, index: int, role: ActorRole) -> LifecycleEvent:
    return LifecycleEvent(
        event_id=f"event-{index}",
        kind=kind,
        actor=_actor(role),
        artifacts=(_artifact_identity(kind.value, index),),
    )


def _filed_evidence(authorization: SuccessorAuthorization) -> FiledVerdictEvidence:
    strategies = (
        ("d2-decoder-fisher",) if authorization is SuccessorAuthorization.AUTHORIZED else ()
    )
    return FiledVerdictEvidence(
        narrative=_artifact_identity("narrative", 200),
        machine=_artifact_identity("machine", 201),
        review_packet=_artifact_identity("review", 202),
        open_major_findings=(),
        successor_authorization=authorization,
        eligible_strategies=strategies,
    )


def _to_filed(
    authorization: SuccessorAuthorization = SuccessorAuthorization.AUTHORIZED,
) -> LifecycleRecord:
    record = LifecycleRecord.start("d1")
    for index, kind, role in (
        (1, LifecycleEventKind.IMPLEMENT, ActorRole.IMPLEMENTATION),
        (2, LifecycleEventKind.VERIFY_CONTRACT, ActorRole.IMPLEMENTATION),
        (3, LifecycleEventKind.PREREGISTER, ActorRole.OPERATOR),
        (4, LifecycleEventKind.START_CONFIRMATORY, ActorRole.OPERATOR),
        (5, LifecycleEventKind.CAPTURE_DVC, ActorRole.AUTOMATION),
        (6, LifecycleEventKind.ANALYZE, ActorRole.ANALYSIS),
    ):
        record = transition(record, _event(kind, index, role))
    evidence = _filed_evidence(authorization)
    review_event = _event(LifecycleEventKind.BLIND_REVIEW, 7, ActorRole.REVIEW)
    review_event = LifecycleEvent(
        event_id=review_event.event_id,
        kind=review_event.kind,
        actor=review_event.actor,
        artifacts=(*review_event.artifacts, evidence.review_packet),
        sealed_packet=evidence.review_packet,
    )
    record = transition(record, review_event)
    file_event = _event(LifecycleEventKind.FILE, 8, ActorRole.OPERATOR)
    file_event = LifecycleEvent(
        event_id=file_event.event_id,
        kind=file_event.kind,
        actor=file_event.actor,
        artifacts=(*file_event.artifacts, evidence.narrative, evidence.machine),
        filed_verdict=evidence,
    )
    return transition(record, file_event)


def _to_contract_verified() -> LifecycleRecord:
    record = LifecycleRecord.start("d1")
    record = transition(record, _event(LifecycleEventKind.IMPLEMENT, 40, ActorRole.IMPLEMENTATION))
    return transition(
        record, _event(LifecycleEventKind.VERIFY_CONTRACT, 41, ActorRole.IMPLEMENTATION)
    )


def _to_preregistered() -> LifecycleRecord:
    return transition(
        _to_contract_verified(), _event(LifecycleEventKind.PREREGISTER, 42, ActorRole.OPERATOR)
    )


def _to_confirmatory_running() -> LifecycleRecord:
    return transition(
        _to_preregistered(),
        _event(LifecycleEventKind.START_CONFIRMATORY, 43, ActorRole.OPERATOR),
    )


def test_record_is_factory_only_and_rejects_duplicate_event_ids() -> None:
    with pytest.raises(TypeError, match=r"start|transition"):
        LifecycleRecord(rung="d1", state=LifecycleState.FILED, events=())

    record = LifecycleRecord.start("d1")
    record = transition(record, _event(LifecycleEventKind.IMPLEMENT, 1, ActorRole.IMPLEMENTATION))
    duplicate = LifecycleEvent(
        event_id="event-1",
        kind=LifecycleEventKind.VERIFY_CONTRACT,
        actor=_actor(ActorRole.IMPLEMENTATION),
        artifacts=(_artifact_identity("different-contract", 2),),
    )
    with pytest.raises(LifecycleTransitionError, match="duplicate event_id"):
        transition(record, duplicate)


@pytest.mark.parametrize(
    ("kind", "authorization", "expected"),
    [
        (
            LifecycleEventKind.RECORD_VALID_NEGATIVE,
            SuccessorAuthorization.AUTHORIZED,
            "VALID_NEGATIVE",
        ),
        (LifecycleEventKind.RETIRE, SuccessorAuthorization.AUTHORIZED, "RETIRED"),
    ],
)
def test_supported_filing_cannot_be_relabelled_by_terminal_event(
    kind: LifecycleEventKind, authorization: SuccessorAuthorization, expected: str
) -> None:
    record = _to_filed(authorization)
    with pytest.raises(LifecycleTransitionError, match=expected):
        transition(record, _event(kind, 20, ActorRole.OPERATOR))


def test_terminal_events_accept_only_their_matching_filed_authorization() -> None:
    valid_negative = _to_filed(SuccessorAuthorization.VALID_NEGATIVE)
    assert (
        transition(
            valid_negative,
            _event(LifecycleEventKind.RECORD_VALID_NEGATIVE, 21, ActorRole.OPERATOR),
        ).state
        is LifecycleState.VALID_NEGATIVE
    )
    retired = _to_filed(SuccessorAuthorization.RETIRED)
    assert (
        transition(retired, _event(LifecycleEventKind.RETIRE, 22, ActorRole.OPERATOR)).state
        is LifecycleState.RETIRED
    )


def test_non_authorizing_terminal_outcome_rejects_eligible_strategy_leak() -> None:
    evidence = FiledVerdictEvidence(
        narrative=_artifact_identity("negative-narrative", 220),
        machine=_artifact_identity("negative-machine", 221),
        review_packet=_artifact_identity("negative-review", 222),
        open_major_findings=(),
        successor_authorization=SuccessorAuthorization.VALID_NEGATIVE,
        eligible_strategies=("must-not-survive",),
    )
    record = _to_filed()
    reopened = transition(record, _event(LifecycleEventKind.REOPEN_REVIEW, 223, ActorRole.REVIEW))
    review = LifecycleEvent(
        event_id="event-224",
        kind=LifecycleEventKind.BLIND_REVIEW,
        actor=_actor(ActorRole.REVIEW),
        artifacts=(_artifact_identity("negative-review-evidence", 224), evidence.review_packet),
        sealed_packet=evidence.review_packet,
    )
    reviewed = transition(reopened, review)
    file_event = LifecycleEvent(
        event_id="event-225",
        kind=LifecycleEventKind.FILE,
        actor=_actor(ActorRole.OPERATOR),
        artifacts=(
            _artifact_identity("negative-filing-evidence", 225),
            evidence.narrative,
            evidence.machine,
        ),
        filed_verdict=evidence,
    )
    filed = transition(reviewed, file_event)
    with pytest.raises(LifecycleTransitionError, match="must not name"):
        transition(
            filed,
            _event(LifecycleEventKind.RECORD_VALID_NEGATIVE, 226, ActorRole.OPERATOR),
        )


def test_operational_and_problem_event_role_contracts_are_load_bearing() -> None:
    with pytest.raises(LifecycleTransitionError, match="operator"):
        transition(
            _to_contract_verified(),
            _event(LifecycleEventKind.RUN_EXPLORATORY_PILOT, 50, ActorRole.IMPLEMENTATION),
        )
    with pytest.raises(LifecycleTransitionError, match="operator"):
        transition(
            _to_preregistered(),
            _event(LifecycleEventKind.START_CONFIRMATORY, 51, ActorRole.AUTOMATION),
        )
    with pytest.raises(LifecycleTransitionError, match="automation"):
        transition(
            _to_confirmatory_running(),
            _event(LifecycleEventKind.CAPTURE_DVC, 52, ActorRole.OPERATOR),
        )
    draft = LifecycleRecord.start("d1")
    with pytest.raises(LifecycleTransitionError, match="operator"):
        transition(draft, _event(LifecycleEventKind.INVALIDATE, 53, ActorRole.ANALYSIS))
    with pytest.raises(LifecycleTransitionError, match="automation"):
        transition(
            draft, _event(LifecycleEventKind.RECORD_NUMERICAL_DEFECT, 54, ActorRole.OPERATOR)
        )
    with pytest.raises(LifecycleTransitionError, match="analysis"):
        transition(draft, _event(LifecycleEventKind.RECORD_EVIDENCE_GAP, 55, ActorRole.OPERATOR))
    with pytest.raises(LifecycleTransitionError, match="analysis"):
        transition(
            draft,
            _event(LifecycleEventKind.REQUEST_OPERATOR_DECISION, 56, ActorRole.OPERATOR),
        )


def test_review_reopen_path_still_requires_and_accepts_a_new_sealed_packet() -> None:
    record = _to_filed()
    record = transition(record, _event(LifecycleEventKind.REOPEN_REVIEW, 30, ActorRole.REVIEW))
    assert record.state is LifecycleState.REVIEW_REOPENED
    packet = _artifact_identity("reopened-review", 31)
    review = LifecycleEvent(
        event_id="event-31",
        kind=LifecycleEventKind.BLIND_REVIEW,
        actor=_actor(ActorRole.REVIEW),
        artifacts=(_artifact_identity("review-evidence", 32), packet),
        sealed_packet=packet,
    )
    assert transition(record, review).state is LifecycleState.BLIND_REVIEWED
