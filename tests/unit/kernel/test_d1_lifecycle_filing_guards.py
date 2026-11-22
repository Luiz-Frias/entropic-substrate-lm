"""Lifecycle guard teeth retained after the D1 verdict-v2 cutover."""

from __future__ import annotations

import pytest

from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
    FiledVerdictEvidence,
    LifecycleContractError,
    LifecycleEvent,
    LifecycleEventKind,
    LifecycleRecord,
    LifecycleState,
    LifecycleTransitionError,
    SuccessorAuthorization,
    transition,
)

pytestmark = pytest.mark.unit


def _artifact(name: str, index: int) -> ArtifactIdentity:
    return ArtifactIdentity(
        path=f"evidence/{name}.json",
        content_hash=ContentHash(algorithm="sha256", digest=f"{index:064x}"),
    )


def _actor(role: ActorRole, identifier: str | None = None) -> ActorIdentity:
    identity = identifier or role.value
    return ActorIdentity(actor_id=identity, session_id=f"{identity}-session", role=role)


def _event(
    kind: LifecycleEventKind,
    index: int,
    role: ActorRole,
    *,
    sealed_packet: ArtifactIdentity | None = None,
    filed_verdict: FiledVerdictEvidence | None = None,
) -> LifecycleEvent:
    artifacts = [_artifact(kind.value, index)]
    if sealed_packet is not None:
        artifacts.append(sealed_packet)
    if filed_verdict is not None:
        artifacts.extend((filed_verdict.narrative, filed_verdict.machine))
    return LifecycleEvent(
        event_id=f"event-{index}",
        kind=kind,
        actor=_actor(role),
        artifacts=tuple(artifacts),
        sealed_packet=sealed_packet,
        filed_verdict=filed_verdict,
    )


def _to_running() -> LifecycleRecord:
    record = LifecycleRecord.start("d1")
    for index, kind, role in (
        (1, LifecycleEventKind.IMPLEMENT, ActorRole.IMPLEMENTATION),
        (2, LifecycleEventKind.VERIFY_CONTRACT, ActorRole.IMPLEMENTATION),
        (3, LifecycleEventKind.PREREGISTER, ActorRole.OPERATOR),
        (4, LifecycleEventKind.START_CONFIRMATORY, ActorRole.OPERATOR),
    ):
        record = transition(record, _event(kind, index, role))
    return record


def _to_analyzed() -> LifecycleRecord:
    record = transition(
        _to_running(), _event(LifecycleEventKind.CAPTURE_DVC, 5, ActorRole.AUTOMATION)
    )
    return transition(record, _event(LifecycleEventKind.ANALYZE, 6, ActorRole.ANALYSIS))


def _evidence(
    authorization: SuccessorAuthorization = SuccessorAuthorization.AUTHORIZED,
    *,
    open_major_findings: tuple[str, ...] = (),
    eligible_arms: tuple[str, ...] | None = None,
    eligible_strategies: tuple[str, ...] | None = None,
) -> FiledVerdictEvidence:
    arms = eligible_arms
    if arms is None:
        arms = ("sg", "ac", "both") if authorization is SuccessorAuthorization.AUTHORIZED else ()
    strategies = eligible_strategies
    if strategies is None:
        strategies = ("d2-pullback",) if authorization is SuccessorAuthorization.AUTHORIZED else ()
    return FiledVerdictEvidence(
        narrative=_artifact("narrative", 20),
        machine=_artifact("machine", 21),
        review_packet=_artifact("review", 22),
        open_major_findings=open_major_findings,
        successor_authorization=authorization,
        eligible_arms=arms,
        eligible_strategies=strategies,
    )


def _to_filed(
    authorization: SuccessorAuthorization = SuccessorAuthorization.AUTHORIZED,
    *,
    evidence: FiledVerdictEvidence | None = None,
) -> LifecycleRecord:
    evidence = evidence or _evidence(authorization)
    record = transition(
        _to_analyzed(),
        _event(
            LifecycleEventKind.BLIND_REVIEW,
            7,
            ActorRole.REVIEW,
            sealed_packet=evidence.review_packet,
        ),
    )
    return transition(
        record, _event(LifecycleEventKind.FILE, 8, ActorRole.OPERATOR, filed_verdict=evidence)
    )


def test_flywheel_requires_capture_fresh_analysis_sealed_review_and_filed_evidence() -> None:
    with pytest.raises(LifecycleTransitionError, match="DvcCaptured"):
        transition(_to_running(), _event(LifecycleEventKind.ANALYZE, 10, ActorRole.ANALYSIS))
    captured = transition(
        _to_running(), _event(LifecycleEventKind.CAPTURE_DVC, 11, ActorRole.AUTOMATION)
    )
    forged = LifecycleEvent(
        event_id="event-12",
        kind=LifecycleEventKind.ANALYZE,
        actor=_actor(ActorRole.ANALYSIS, "implementation"),
        artifacts=(_artifact("analysis", 12),),
    )
    with pytest.raises(LifecycleTransitionError, match="fresh analysis"):
        transition(captured, forged)
    with pytest.raises(LifecycleTransitionError, match="sealed review packet"):
        transition(_to_analyzed(), _event(LifecycleEventKind.BLIND_REVIEW, 13, ActorRole.REVIEW))

    packet = _artifact("packet", 14)
    reviewed = transition(
        _to_analyzed(),
        _event(LifecycleEventKind.BLIND_REVIEW, 14, ActorRole.REVIEW, sealed_packet=packet),
    )
    with pytest.raises(LifecycleTransitionError, match="filed verdict evidence"):
        transition(reviewed, _event(LifecycleEventKind.FILE, 15, ActorRole.OPERATOR))


def test_lineage_factory_hash_and_event_identity_guards_hold() -> None:
    with pytest.raises(TypeError, match=r"start|transition"):
        LifecycleRecord(rung="d1", state=LifecycleState.FILED, events=())
    record = transition(
        LifecycleRecord.start("d1"),
        _event(LifecycleEventKind.IMPLEMENT, 16, ActorRole.IMPLEMENTATION),
    )
    with pytest.raises(LifecycleTransitionError, match="content hash drift"):
        transition(
            record,
            LifecycleEvent(
                event_id="event-17",
                kind=LifecycleEventKind.VERIFY_CONTRACT,
                actor=_actor(ActorRole.IMPLEMENTATION),
                artifacts=(_artifact("Implement", 99),),
            ),
        )
    with pytest.raises(LifecycleTransitionError, match="duplicate event_id"):
        transition(
            record,
            LifecycleEvent(
                event_id="event-16",
                kind=LifecycleEventKind.VERIFY_CONTRACT,
                actor=_actor(ActorRole.IMPLEMENTATION),
                artifacts=(_artifact("contract", 17),),
            ),
        )


def test_filed_decision_and_role_guards_are_load_bearing() -> None:
    filed = _to_filed()
    assert filed.events[-1].filed_verdict is not None
    assert filed.events[-1].filed_verdict.eligible_arms == ("sg", "ac", "both")
    with pytest.raises(LifecycleTransitionError, match="open MAJOR"):
        transition(
            _to_filed(evidence=_evidence(open_major_findings=("MAJOR-1",))),
            _event(LifecycleEventKind.AUTHORIZE_SUCCESSOR, 18, ActorRole.OPERATOR),
        )
    with pytest.raises(LifecycleTransitionError, match="eligible successor strategy"):
        transition(
            _to_filed(evidence=_evidence(eligible_strategies=())),
            _event(LifecycleEventKind.AUTHORIZE_SUCCESSOR, 19, ActorRole.OPERATOR),
        )
    with pytest.raises(LifecycleTransitionError, match="eligible successor arm"):
        transition(
            _to_filed(evidence=_evidence(eligible_arms=())),
            _event(LifecycleEventKind.AUTHORIZE_SUCCESSOR, 29, ActorRole.OPERATOR),
        )
    with pytest.raises(LifecycleContractError) as invalid_arm:
        _evidence(eligible_arms=(1,))  # type: ignore[arg-type]
    assert invalid_arm.value.code == "invalid_successor_arm"
    with pytest.raises(LifecycleContractError) as invalid_strategy:
        _evidence(eligible_strategies=(1,))  # type: ignore[arg-type]
    assert invalid_strategy.value.code == "invalid_successor_strategy"
    with pytest.raises(LifecycleTransitionError, match="eligible successor arms"):
        transition(
            _to_filed(
                SuccessorAuthorization.VALID_NEGATIVE,
                evidence=_evidence(
                    SuccessorAuthorization.VALID_NEGATIVE,
                    eligible_arms=("sg",),
                ),
            ),
            _event(LifecycleEventKind.RECORD_VALID_NEGATIVE, 28, ActorRole.OPERATOR),
        )
    assert (
        transition(
            _to_filed(SuccessorAuthorization.VALID_NEGATIVE),
            _event(LifecycleEventKind.RECORD_VALID_NEGATIVE, 20, ActorRole.OPERATOR),
        ).state
        is LifecycleState.VALID_NEGATIVE
    )
    before_pilot = transition(
        transition(
            LifecycleRecord.start("d1"),
            _event(LifecycleEventKind.IMPLEMENT, 30, ActorRole.IMPLEMENTATION),
        ),
        _event(LifecycleEventKind.VERIFY_CONTRACT, 31, ActorRole.IMPLEMENTATION),
    )
    with pytest.raises(LifecycleTransitionError, match="operator"):
        transition(
            before_pilot, _event(LifecycleEventKind.RUN_EXPLORATORY_PILOT, 32, ActorRole.ANALYSIS)
        )


def test_reopened_review_returns_to_blind_review_with_new_artifact_identity() -> None:
    reopened = transition(
        _to_filed(), _event(LifecycleEventKind.REOPEN_REVIEW, 34, ActorRole.REVIEW)
    )
    packet = _artifact("reopened-packet", 35)
    rereview = LifecycleEvent(
        event_id="event-35",
        kind=LifecycleEventKind.BLIND_REVIEW,
        actor=_actor(ActorRole.REVIEW),
        artifacts=(_artifact("reopened-review-evidence", 36), packet),
        sealed_packet=packet,
    )
    assert transition(reopened, rereview).state is LifecycleState.BLIND_REVIEWED
