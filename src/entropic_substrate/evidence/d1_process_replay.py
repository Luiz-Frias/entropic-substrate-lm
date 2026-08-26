"""Pure replay policy for persisted D1 flywheel facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from entropic_substrate.evidence.d1_lifecycle_values import (
    D1LifecycleEvent,
    D1ProblemKind,
    D1ProcessEventKind,
    D1ProcessState,
)
from entropic_substrate.evidence.lifecycle import (
    ActorRole,
    ArtifactIdentity,
    FiledVerdictEvidence,
    LifecycleEvent,
    LifecycleEventKind,
    LifecycleRecord,
    LifecycleTransitionError,
    transition,
)


class D1ProcessTransitionError(RuntimeError):
    """An expected, rejected process transition; no event was persisted."""

    def __init__(self, *, code: str, state: str, detail: str) -> None:
        self.code = code
        self.state = state
        self.context = {"code": code, "state": state}
        super().__init__(f"D1 process {code} from {state}: {detail}")


def reject_transition(
    code: str, state: D1ProcessState | None, detail: str
) -> D1ProcessTransitionError:
    return D1ProcessTransitionError(
        code=code,
        state="not_started" if state is None else state.value,
        detail=detail,
    )


_ROLE: Final = {
    D1ProcessEventKind.CONTRACT_VERIFIED: ActorRole.IMPLEMENTATION,
    D1ProcessEventKind.PILOT_FILED: ActorRole.OPERATOR,
    D1ProcessEventKind.PREREGISTRATION_FILED: ActorRole.OPERATOR,
    D1ProcessEventKind.LAUNCH_ADMITTED: ActorRole.OPERATOR,
    D1ProcessEventKind.RUNNING: ActorRole.AUTOMATION,
    D1ProcessEventKind.RESUMED: ActorRole.AUTOMATION,
    D1ProcessEventKind.TRAINER_STOPPED: ActorRole.AUTOMATION,
    D1ProcessEventKind.OPERATOR_FINALIZED: ActorRole.OPERATOR,
    D1ProcessEventKind.CAPTURED: ActorRole.AUTOMATION,
    D1ProcessEventKind.ANALYSIS_FILED: ActorRole.ANALYSIS,
    D1ProcessEventKind.ALIAS_COMMITMENT_FILED: ActorRole.ANALYSIS,
    D1ProcessEventKind.REVIEW_FILED: ActorRole.REVIEW,
    D1ProcessEventKind.ALIAS_REVEALED: ActorRole.OPERATOR,
    D1ProcessEventKind.ADJUDICATED: ActorRole.OPERATOR,
    D1ProcessEventKind.VERDICT_FILED: ActorRole.OPERATOR,
    D1ProcessEventKind.GATE_AUTHORIZED: ActorRole.OPERATOR,
    D1ProcessEventKind.GATE_VALID_NEGATIVE: ActorRole.OPERATOR,
    D1ProcessEventKind.GATE_RETIRED: ActorRole.OPERATOR,
}
_PROBLEM_ROLE: Final = {
    D1ProblemKind.INVALIDATED: ActorRole.OPERATOR,
    D1ProblemKind.NUMERICAL_DEFECT: ActorRole.AUTOMATION,
    D1ProblemKind.EVIDENCE_GAP: ActorRole.ANALYSIS,
    D1ProblemKind.OPERATOR_DECISION_NEEDED: ActorRole.ANALYSIS,
}
_TERMINAL: Final = frozenset(
    {
        D1ProcessState.GATE_AUTHORIZED,
        D1ProcessState.VALID_NEGATIVE,
        D1ProcessState.RETIRED,
        D1ProcessState.PROBLEM,
    }
)
_PURE_KIND: Final = {
    D1ProcessEventKind.PILOT_FILED: LifecycleEventKind.RUN_EXPLORATORY_PILOT,
    D1ProcessEventKind.PREREGISTRATION_FILED: LifecycleEventKind.PREREGISTER,
    D1ProcessEventKind.LAUNCH_ADMITTED: LifecycleEventKind.START_CONFIRMATORY,
    D1ProcessEventKind.CAPTURED: LifecycleEventKind.CAPTURE_DVC,
    D1ProcessEventKind.ANALYSIS_FILED: LifecycleEventKind.ANALYZE,
    D1ProcessEventKind.REVIEW_FILED: LifecycleEventKind.BLIND_REVIEW,
}
_PROBLEM_PURE_KIND: Final = {
    D1ProblemKind.INVALIDATED: LifecycleEventKind.INVALIDATE,
    D1ProblemKind.NUMERICAL_DEFECT: LifecycleEventKind.RECORD_NUMERICAL_DEFECT,
    D1ProblemKind.EVIDENCE_GAP: LifecycleEventKind.RECORD_EVIDENCE_GAP,
    D1ProblemKind.OPERATOR_DECISION_NEEDED: LifecycleEventKind.REQUEST_OPERATOR_DECISION,
}
_ARTIFACT_COUNT: Final = {
    D1ProcessEventKind.CONTRACT_VERIFIED: 2,
    D1ProcessEventKind.REVIEW_FILED: 3,
    D1ProcessEventKind.ALIAS_REVEALED: 2,
    D1ProcessEventKind.VERDICT_FILED: 2,
    D1ProcessEventKind.GATE_AUTHORIZED: 3,
    D1ProcessEventKind.GATE_VALID_NEGATIVE: 3,
    D1ProcessEventKind.GATE_RETIRED: 3,
}


@dataclass(frozen=True, slots=True)
class D1ProcessReplay:
    """Fully derived process state; only canonical events are persisted."""

    state: D1ProcessState
    pure_lifecycle: LifecycleRecord
    events: tuple[D1LifecycleEvent, ...]
    active_subject: str | None = None
    launched_subjects: frozenset[str] = field(default_factory=frozenset)
    finalized_subjects: tuple[str, ...] = ()
    eligible_arms: tuple[str, ...] = ()


def _pure_event(
    event: D1LifecycleEvent,
    kind: LifecycleEventKind,
    *,
    suffix: str,
    artifacts: tuple[ArtifactIdentity, ...] | None = None,
    sealed_packet: ArtifactIdentity | None = None,
    filed_verdict: FiledVerdictEvidence | None = None,
) -> LifecycleEvent:
    return LifecycleEvent(
        event_id=f"{event.event_id}:{suffix}",
        kind=kind,
        actor=event.actor,
        artifacts=event.artifacts if artifacts is None else artifacts,
        sealed_packet=sealed_packet,
        filed_verdict=filed_verdict,
    )


def _check_cardinality(event: D1LifecycleEvent, state: D1ProcessState | None) -> None:
    expected = _ARTIFACT_COUNT.get(event.event_kind, 1)
    if len(event.artifacts) != expected:
        raise reject_transition(
            "artifact_cardinality",
            state,
            f"{event.event_kind.value} requires exactly {expected} artifact identities",
        )


def _seed(event: D1LifecycleEvent) -> D1ProcessReplay:
    if event.event_kind is not D1ProcessEventKind.CONTRACT_VERIFIED:
        raise reject_transition(
            "missing_contract", None, "first event must record contract verification"
        )
    pure = LifecycleRecord.start("d1")
    pure = transition(
        pure,
        _pure_event(
            event,
            LifecycleEventKind.IMPLEMENT,
            suffix="implement",
            artifacts=(event.artifacts[0],),
        ),
    )
    pure = transition(
        pure,
        _pure_event(
            event,
            LifecycleEventKind.VERIFY_CONTRACT,
            suffix="verify",
            artifacts=(event.artifacts[1],),
        ),
    )
    return D1ProcessReplay(D1ProcessState.CONTRACT_VERIFIED, pure, (event,))


def _required_role(event: D1LifecycleEvent) -> ActorRole:
    if event.event_kind is not D1ProcessEventKind.PROBLEM_RECORDED:
        try:
            return _ROLE[event.event_kind]
        except KeyError as exc:
            raise reject_transition("unknown_event", None, "event kind has no role policy") from exc
    problem = event.problem_kind
    if problem is None:
        raise reject_transition("missing_problem", None, "problem event needs a typed kind")
    return _PROBLEM_ROLE[problem]


def _check_actor(replay: D1ProcessReplay | None, event: D1LifecycleEvent) -> None:
    required = _required_role(event)
    if event.actor.role is not required:
        raise reject_transition(
            "wrong_role",
            None if replay is None else replay.state,
            f"requires {required.value} role",
        )
    if replay is None or event.actor.role not in {ActorRole.IMPLEMENTATION, ActorRole.ANALYSIS}:
        return
    opposite = (
        ActorRole.ANALYSIS
        if event.actor.role is ActorRole.IMPLEMENTATION
        else ActorRole.IMPLEMENTATION
    )
    if any(
        prior.actor.role is opposite
        and (
            prior.actor.actor_id == event.actor.actor_id
            or prior.actor.session_id == event.actor.session_id
        )
        for prior in replay.events
    ):
        raise reject_transition(
            "role_separation",
            replay.state,
            "implementation and analysis identities must be fresh",
        )


def _expected_state(replay: D1ProcessReplay, event: D1LifecycleEvent) -> D1ProcessState:
    if replay.state in _TERMINAL:
        raise reject_transition(
            "terminal", replay.state, "terminal lifecycle state forbids later events"
        )
    if event.event_kind is D1ProcessEventKind.PROBLEM_RECORDED:
        return D1ProcessState.PROBLEM
    allowed: dict[D1ProcessEventKind, tuple[set[D1ProcessState], D1ProcessState]] = {
        D1ProcessEventKind.PILOT_FILED: (
            {D1ProcessState.CONTRACT_VERIFIED},
            D1ProcessState.PILOT_FILED,
        ),
        D1ProcessEventKind.PREREGISTRATION_FILED: (
            {D1ProcessState.PILOT_FILED},
            D1ProcessState.PREREGISTERED,
        ),
        D1ProcessEventKind.LAUNCH_ADMITTED: (
            {D1ProcessState.PREREGISTERED, D1ProcessState.OPERATOR_FINALIZED},
            D1ProcessState.LAUNCH_ADMITTED,
        ),
        D1ProcessEventKind.RUNNING: (
            {D1ProcessState.LAUNCH_ADMITTED},
            D1ProcessState.RUNNING,
        ),
        D1ProcessEventKind.RESUMED: ({D1ProcessState.RUNNING}, D1ProcessState.RUNNING),
        D1ProcessEventKind.TRAINER_STOPPED: (
            {D1ProcessState.RUNNING},
            D1ProcessState.TRAINER_STOPPED,
        ),
        D1ProcessEventKind.OPERATOR_FINALIZED: (
            {D1ProcessState.TRAINER_STOPPED},
            D1ProcessState.OPERATOR_FINALIZED,
        ),
        D1ProcessEventKind.CAPTURED: (
            {D1ProcessState.OPERATOR_FINALIZED},
            D1ProcessState.CAPTURED,
        ),
        D1ProcessEventKind.ANALYSIS_FILED: (
            {D1ProcessState.CAPTURED},
            D1ProcessState.ANALYSIS_FILED,
        ),
        D1ProcessEventKind.ALIAS_COMMITMENT_FILED: (
            {D1ProcessState.ANALYSIS_FILED},
            D1ProcessState.ALIAS_COMMITTED,
        ),
        D1ProcessEventKind.REVIEW_FILED: (
            {D1ProcessState.ALIAS_COMMITTED},
            D1ProcessState.REVIEW_FILED,
        ),
        D1ProcessEventKind.ALIAS_REVEALED: (
            {D1ProcessState.REVIEW_FILED},
            D1ProcessState.ALIAS_REVEALED,
        ),
        D1ProcessEventKind.ADJUDICATED: (
            {D1ProcessState.ALIAS_REVEALED},
            D1ProcessState.ADJUDICATED,
        ),
        D1ProcessEventKind.VERDICT_FILED: (
            {D1ProcessState.ADJUDICATED},
            D1ProcessState.VERDICT_FILED,
        ),
        D1ProcessEventKind.GATE_AUTHORIZED: (
            {D1ProcessState.VERDICT_FILED},
            D1ProcessState.GATE_AUTHORIZED,
        ),
        D1ProcessEventKind.GATE_VALID_NEGATIVE: (
            {D1ProcessState.VERDICT_FILED},
            D1ProcessState.VALID_NEGATIVE,
        ),
        D1ProcessEventKind.GATE_RETIRED: (
            {D1ProcessState.VERDICT_FILED},
            D1ProcessState.RETIRED,
        ),
    }
    try:
        source, target = allowed[event.event_kind]
    except KeyError as exc:
        raise reject_transition(
            "unknown_event", replay.state, "event kind has no state policy"
        ) from exc
    if replay.state not in source:
        expected = ", ".join(sorted(value.value for value in source))
        raise reject_transition(
            "wrong_order", replay.state, f"{event.event_kind.value} requires {expected}"
        )
    return target


def _prior(replay: D1ProcessReplay, kind: D1ProcessEventKind) -> D1LifecycleEvent:
    for event in reversed(replay.events):
        if event.event_kind is kind:
            return event
    raise reject_transition("missing_prior", replay.state, f"missing prior {kind.value} event")


def _advance_pure(replay: D1ProcessReplay, event: D1LifecycleEvent) -> LifecycleRecord:
    pure = replay.pure_lifecycle
    if event.event_kind is D1ProcessEventKind.PROBLEM_RECORDED:
        problem = event.problem_kind
        if problem is None:
            raise reject_transition("missing_problem", replay.state, "problem kind is absent")
        return transition(
            pure,
            _pure_event(event, _PROBLEM_PURE_KIND[problem], suffix="problem"),
        )
    kind = _PURE_KIND.get(event.event_kind)
    if kind is not None:
        if kind is LifecycleEventKind.START_CONFIRMATORY and any(
            prior.event_kind is D1ProcessEventKind.LAUNCH_ADMITTED for prior in replay.events
        ):
            return pure
        packet = event.artifacts[0] if kind is LifecycleEventKind.BLIND_REVIEW else None
        return transition(
            pure,
            _pure_event(event, kind, suffix="phase", sealed_packet=packet),
        )
    gate_kinds = {
        D1ProcessEventKind.GATE_AUTHORIZED: LifecycleEventKind.AUTHORIZE_SUCCESSOR,
        D1ProcessEventKind.GATE_VALID_NEGATIVE: LifecycleEventKind.RECORD_VALID_NEGATIVE,
        D1ProcessEventKind.GATE_RETIRED: LifecycleEventKind.RETIRE,
    }
    if event.event_kind not in gate_kinds:
        return pure
    authorization = event.gate_authorization
    if authorization is None:
        raise reject_transition("missing_gate", replay.state, "gate authorization is absent")
    verdict = _prior(replay, D1ProcessEventKind.VERDICT_FILED)
    review = _prior(replay, D1ProcessEventKind.REVIEW_FILED)
    evidence = FiledVerdictEvidence(
        narrative=verdict.artifacts[0],
        machine=verdict.artifacts[1],
        review_packet=review.artifacts[0],
        open_major_findings=(),
        successor_authorization=authorization,
        eligible_arms=event.eligible_arms,
        eligible_strategies=event.eligible_strategies,
    )
    pure = transition(
        pure,
        _pure_event(
            verdict,
            LifecycleEventKind.FILE,
            suffix="file",
            filed_verdict=evidence,
        ),
    )
    return transition(
        pure,
        _pure_event(event, gate_kinds[event.event_kind], suffix="gate"),
    )


def advance_d1_process(replay: D1ProcessReplay | None, event: D1LifecycleEvent) -> D1ProcessReplay:
    """Apply exactly one event through operational and pure lifecycle policy."""

    _check_cardinality(event, None if replay is None else replay.state)
    _check_actor(replay, event)
    if replay is None:
        return _seed(event)
    target = _expected_state(replay, event)
    active = replay.active_subject
    launched = replay.launched_subjects
    finalized = replay.finalized_subjects
    subject = event.subject_id
    if event.event_kind is D1ProcessEventKind.ALIAS_REVEALED:
        review = _prior(replay, D1ProcessEventKind.REVIEW_FILED)
        if event.artifacts[1] != review.artifacts[2]:
            raise reject_transition(
                "review_receipt_mismatch",
                replay.state,
                "alias reveal must name the exact durable review receipt",
            )
    if event.event_kind is D1ProcessEventKind.LAUNCH_ADMITTED:
        if subject in launched:
            raise reject_transition(
                "duplicate_subject", replay.state, f"subject {subject!r} was already launched"
            )
        if subject is None:
            raise reject_transition("missing_subject", replay.state, "launch subject is absent")
        active, launched = subject, launched | {subject}
    elif (
        event.event_kind
        in {
            D1ProcessEventKind.RUNNING,
            D1ProcessEventKind.RESUMED,
            D1ProcessEventKind.TRAINER_STOPPED,
            D1ProcessEventKind.OPERATOR_FINALIZED,
        }
        and subject != active
    ):
        raise reject_transition("subject_mismatch", replay.state, f"active subject is {active!r}")
    if event.event_kind is D1ProcessEventKind.OPERATOR_FINALIZED:
        if subject is None:
            raise reject_transition("missing_subject", replay.state, "finalized subject is absent")
        finalized, active = (*finalized, subject), None
    try:
        pure = _advance_pure(replay, event)
    except LifecycleTransitionError as exc:
        raise reject_transition("pure_lifecycle", replay.state, str(exc)) from exc
    eligible_arms = (
        event.eligible_arms
        if event.event_kind is D1ProcessEventKind.GATE_AUTHORIZED
        else replay.eligible_arms
    )
    return D1ProcessReplay(
        target, pure, (*replay.events, event), active, launched, finalized, eligible_arms
    )


def replay_d1_process(events: tuple[D1LifecycleEvent, ...]) -> D1ProcessReplay:
    """Recompute the full state and reject semantically forked histories."""

    if type(events) is not tuple or not events:
        raise reject_transition("missing_history", None, "history requires at least one event")
    replay: D1ProcessReplay | None = None
    seen: set[str] = set()
    for expected_sequence, event in enumerate(events):
        if event.event_seq != expected_sequence or event.event_id in seen:
            raise reject_transition(
                "event_identity",
                None if replay is None else replay.state,
                "sequence or event ID fork",
            )
        seen.add(event.event_id)
        replay = advance_d1_process(replay, event)
    if replay is None:  # pragma: no cover - guarded by the nonempty check
        raise AssertionError("unreachable empty D1 replay")
    return replay
