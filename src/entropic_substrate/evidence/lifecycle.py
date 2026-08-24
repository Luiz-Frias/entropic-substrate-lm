"""Pure D1 experiment-lifecycle state machine.

This module intentionally is not a workflow runner.  It records only the
scientific states which authorize the next state, content identities, and the
role/session that made each transition.  Adapters may persist or display these
records, but they may not weaken their transition rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

_HASH_LENGTHS: Final[dict[str, int]] = {"md5": 32, "sha256": 64}


class LifecycleContractError(ValueError):
    """A malformed lifecycle value object with machine-readable context."""

    def __init__(self, *, code: str, detail: str, context: dict[str, str]) -> None:
        self.code = code
        self.context = context
        super().__init__(f"{code}: {detail}; context={context}")


class LifecycleTransitionError(RuntimeError):
    """An illegal lifecycle state transition with its causal context."""

    def __init__(self, *, detail: str, context: dict[str, str]) -> None:
        self.context = context
        super().__init__(f"illegal lifecycle transition: {detail}; context={context}")


@dataclass(frozen=True)
class ContentHash:
    """A named, validated content digest.

    The algorithm is carried with the digest so an MD5 imported from a legacy
    DVC manifest cannot be silently compared as though it were a SHA-256.
    """

    algorithm: str
    digest: str

    def __post_init__(self) -> None:
        expected_length = _HASH_LENGTHS.get(self.algorithm)
        if expected_length is None:
            raise LifecycleContractError(
                code="unsupported_hash_algorithm",
                detail=f"unsupported content-hash algorithm {self.algorithm!r}",
                context={"algorithm": self.algorithm},
            )
        if len(self.digest) != expected_length or any(
            c not in "0123456789abcdef" for c in self.digest
        ):
            raise LifecycleContractError(
                code="invalid_content_hash",
                detail=f"{self.algorithm} digest must be {expected_length} lowercase hex characters",
                context={"algorithm": self.algorithm, "digest": self.digest},
            )


@dataclass(frozen=True)
class ArtifactIdentity:
    """A content-addressed artifact identity used throughout one lineage."""

    path: str
    content_hash: ContentHash

    def __post_init__(self) -> None:
        if not self.path or self.path.strip() != self.path:
            raise LifecycleContractError(
                code="invalid_artifact_path",
                detail="artifact path must be a nonempty, trimmed string",
                context={"path": self.path},
            )


class ActorRole(StrEnum):
    IMPLEMENTATION = "implementation"
    ANALYSIS = "analysis"
    REVIEW = "review"
    OPERATOR = "operator"
    AUTOMATION = "automation"


@dataclass(frozen=True)
class ActorIdentity:
    """The person/system and session authorized to make a lifecycle claim."""

    actor_id: str
    session_id: str
    role: ActorRole

    def __post_init__(self) -> None:
        if not self.actor_id or not self.session_id:
            raise LifecycleContractError(
                code="missing_actor_identity",
                detail="actor_id and session_id are both required",
                context={"actor_id": self.actor_id, "session_id": self.session_id},
            )


class LifecycleState(StrEnum):
    DRAFT_DESIGN = "DraftDesign"
    IMPLEMENTED = "Implemented"
    CONTRACT_VERIFIED = "ContractVerified"
    EXPLORATORY_PILOT = "ExploratoryPilot"
    PREREGISTERED = "Preregistered"
    CONFIRMATORY_RUNNING = "ConfirmatoryRunning"
    DVC_CAPTURED = "DvcCaptured"
    ANALYZED = "Analyzed"
    BLIND_REVIEWED = "BlindReviewed"
    FILED = "Filed"
    SUCCESSOR_AUTHORIZED = "SuccessorAuthorized"
    INVALIDATED = "Invalidated"
    NUMERICAL_DEFECT = "NumericalDefect"
    EVIDENCE_GAP = "EvidenceGap"
    REVIEW_REOPENED = "ReviewReopened"
    OPERATOR_DECISION_NEEDED = "OperatorDecisionNeeded"
    VALID_NEGATIVE = "ValidNegative"
    RETIRED = "Retired"


class LifecycleEventKind(StrEnum):
    IMPLEMENT = "Implement"
    VERIFY_CONTRACT = "VerifyContract"
    RUN_EXPLORATORY_PILOT = "RunExploratoryPilot"
    PREREGISTER = "Preregister"
    START_CONFIRMATORY = "StartConfirmatory"
    CAPTURE_DVC = "CaptureDvc"
    ANALYZE = "Analyze"
    BLIND_REVIEW = "BlindReview"
    FILE = "File"
    AUTHORIZE_SUCCESSOR = "AuthorizeSuccessor"
    INVALIDATE = "Invalidate"
    RECORD_NUMERICAL_DEFECT = "RecordNumericalDefect"
    RECORD_EVIDENCE_GAP = "RecordEvidenceGap"
    REOPEN_REVIEW = "ReopenReview"
    REQUEST_OPERATOR_DECISION = "RequestOperatorDecision"
    RECORD_VALID_NEGATIVE = "RecordValidNegative"
    RETIRE = "Retire"


class SuccessorAuthorization(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    VALID_NEGATIVE = "VALID_NEGATIVE"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class FiledVerdictEvidence:
    """The machine-readable filing facts the lifecycle may consume.

    ``ValidatedD1Filing.to_filed_evidence`` is the intended construction
    seam. The FSM never receives losses, estimates, or telemetry records.
    """

    narrative: ArtifactIdentity
    machine: ArtifactIdentity
    review_packet: ArtifactIdentity
    open_major_findings: tuple[str, ...]
    successor_authorization: SuccessorAuthorization
    eligible_strategies: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.narrative == self.machine:
            raise LifecycleContractError(
                code="duplicate_filed_artifact",
                detail="narrative and machine verdict identities must differ",
                context={"path": self.narrative.path},
            )
        if any(not finding for finding in self.open_major_findings):
            raise LifecycleContractError(
                code="invalid_major_finding",
                detail="open major findings must have nonempty identifiers",
                context={},
            )
        if any(not strategy for strategy in self.eligible_strategies):
            raise LifecycleContractError(
                code="invalid_successor_strategy",
                detail="eligible strategies must have nonempty identifiers",
                context={},
            )


@dataclass(frozen=True)
class LifecycleEvent:
    """An immutable, content-addressed proposition to advance a record."""

    event_id: str
    kind: LifecycleEventKind
    actor: ActorIdentity
    artifacts: tuple[ArtifactIdentity, ...]
    sealed_packet: ArtifactIdentity | None = None
    filed_verdict: FiledVerdictEvidence | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise LifecycleContractError(
                code="missing_event_id",
                detail="every lifecycle event needs a stable identity",
                context={},
            )
        if not self.artifacts:
            raise LifecycleContractError(
                code="missing_event_artifacts",
                detail="every lifecycle event needs at least one content-addressed artifact",
                context={"event_id": self.event_id, "event_kind": self.kind.value},
            )


_RECORD_FACTORY_TOKEN: Final = object()


@dataclass(frozen=True, init=False)
class LifecycleRecord:
    """Append-only immutable lineage for one rung's scientific flywheel."""

    rung: str
    state: LifecycleState
    events: tuple[LifecycleEvent, ...]
    _factory_token: object = field(repr=False, compare=False)

    def __init__(self, *_: object, **__: object) -> None:
        msg = "LifecycleRecord is factory-only; use LifecycleRecord.start() or transition()"
        raise TypeError(msg)

    @classmethod
    def start(cls, rung: str) -> LifecycleRecord:
        if not rung:
            raise LifecycleContractError(
                code="missing_rung",
                detail="a lifecycle record must name its rung",
                context={},
            )
        return _attested_record(rung=rung, state=LifecycleState.DRAFT_DESIGN, events=())


def _attested_record(
    *, rung: str, state: LifecycleState, events: tuple[LifecycleEvent, ...]
) -> LifecycleRecord:
    record = object.__new__(LifecycleRecord)
    object.__setattr__(record, "rung", rung)
    object.__setattr__(record, "state", state)
    object.__setattr__(record, "events", events)
    object.__setattr__(record, "_factory_token", _RECORD_FACTORY_TOKEN)
    return record


_ALLOWED: Final[dict[LifecycleState, dict[LifecycleEventKind, LifecycleState]]] = {
    LifecycleState.DRAFT_DESIGN: {LifecycleEventKind.IMPLEMENT: LifecycleState.IMPLEMENTED},
    LifecycleState.IMPLEMENTED: {
        LifecycleEventKind.VERIFY_CONTRACT: LifecycleState.CONTRACT_VERIFIED
    },
    LifecycleState.CONTRACT_VERIFIED: {
        LifecycleEventKind.RUN_EXPLORATORY_PILOT: LifecycleState.EXPLORATORY_PILOT,
        LifecycleEventKind.PREREGISTER: LifecycleState.PREREGISTERED,
    },
    LifecycleState.EXPLORATORY_PILOT: {
        LifecycleEventKind.PREREGISTER: LifecycleState.PREREGISTERED
    },
    LifecycleState.PREREGISTERED: {
        LifecycleEventKind.START_CONFIRMATORY: LifecycleState.CONFIRMATORY_RUNNING,
    },
    LifecycleState.CONFIRMATORY_RUNNING: {
        LifecycleEventKind.CAPTURE_DVC: LifecycleState.DVC_CAPTURED
    },
    LifecycleState.DVC_CAPTURED: {LifecycleEventKind.ANALYZE: LifecycleState.ANALYZED},
    LifecycleState.ANALYZED: {LifecycleEventKind.BLIND_REVIEW: LifecycleState.BLIND_REVIEWED},
    LifecycleState.BLIND_REVIEWED: {
        LifecycleEventKind.FILE: LifecycleState.FILED,
        LifecycleEventKind.REOPEN_REVIEW: LifecycleState.REVIEW_REOPENED,
    },
    LifecycleState.REVIEW_REOPENED: {
        LifecycleEventKind.BLIND_REVIEW: LifecycleState.BLIND_REVIEWED
    },
    LifecycleState.FILED: {
        LifecycleEventKind.AUTHORIZE_SUCCESSOR: LifecycleState.SUCCESSOR_AUTHORIZED,
        LifecycleEventKind.REOPEN_REVIEW: LifecycleState.REVIEW_REOPENED,
        LifecycleEventKind.RECORD_VALID_NEGATIVE: LifecycleState.VALID_NEGATIVE,
        LifecycleEventKind.RETIRE: LifecycleState.RETIRED,
    },
}

_PROBLEM_EVENTS: Final[dict[LifecycleEventKind, LifecycleState]] = {
    LifecycleEventKind.INVALIDATE: LifecycleState.INVALIDATED,
    LifecycleEventKind.RECORD_NUMERICAL_DEFECT: LifecycleState.NUMERICAL_DEFECT,
    LifecycleEventKind.RECORD_EVIDENCE_GAP: LifecycleState.EVIDENCE_GAP,
    LifecycleEventKind.REQUEST_OPERATOR_DECISION: LifecycleState.OPERATOR_DECISION_NEEDED,
}

_REQUIRED_ROLE: Final[dict[LifecycleEventKind, ActorRole]] = {
    LifecycleEventKind.IMPLEMENT: ActorRole.IMPLEMENTATION,
    LifecycleEventKind.VERIFY_CONTRACT: ActorRole.IMPLEMENTATION,
    LifecycleEventKind.RUN_EXPLORATORY_PILOT: ActorRole.OPERATOR,
    LifecycleEventKind.START_CONFIRMATORY: ActorRole.OPERATOR,
    LifecycleEventKind.CAPTURE_DVC: ActorRole.AUTOMATION,
    LifecycleEventKind.ANALYZE: ActorRole.ANALYSIS,
    LifecycleEventKind.BLIND_REVIEW: ActorRole.REVIEW,
    LifecycleEventKind.REOPEN_REVIEW: ActorRole.REVIEW,
    LifecycleEventKind.PREREGISTER: ActorRole.OPERATOR,
    LifecycleEventKind.FILE: ActorRole.OPERATOR,
    LifecycleEventKind.AUTHORIZE_SUCCESSOR: ActorRole.OPERATOR,
    LifecycleEventKind.INVALIDATE: ActorRole.OPERATOR,
    LifecycleEventKind.RECORD_NUMERICAL_DEFECT: ActorRole.AUTOMATION,
    LifecycleEventKind.RECORD_EVIDENCE_GAP: ActorRole.ANALYSIS,
    LifecycleEventKind.REQUEST_OPERATOR_DECISION: ActorRole.ANALYSIS,
    LifecycleEventKind.RECORD_VALID_NEGATIVE: ActorRole.OPERATOR,
    LifecycleEventKind.RETIRE: ActorRole.OPERATOR,
}


def _transition_error(
    record: LifecycleRecord, event: LifecycleEvent, detail: str
) -> LifecycleTransitionError:
    return LifecycleTransitionError(
        detail=detail,
        context={
            "rung": record.rung,
            "source_state": record.state.value,
            "event_kind": event.kind.value,
            "actor_id": event.actor.actor_id,
            "session_id": event.actor.session_id,
            "role": event.actor.role.value,
        },
    )


def _target_state(record: LifecycleRecord, event: LifecycleEvent) -> LifecycleState:
    if event.kind in _PROBLEM_EVENTS:
        return _PROBLEM_EVENTS[event.kind]
    allowed = _ALLOWED.get(record.state, {})
    if target := allowed.get(event.kind):
        return target
    expected = (
        ", ".join(target.value for target in _ALLOWED.get(record.state, {}).values())
        or "no further state"
    )
    raise _transition_error(
        record,
        event,
        f"{event.kind.value} is not permitted from {record.state.value}; expected {expected}",
    )


def _validate_role(record: LifecycleRecord, event: LifecycleEvent) -> None:
    if required := _REQUIRED_ROLE.get(event.kind):
        if event.actor.role is not required:
            raise _transition_error(
                record,
                event,
                f"{event.kind.value} requires role {required.value}, got {event.actor.role.value}",
            )


def _validate_no_hash_drift(record: LifecycleRecord, event: LifecycleEvent) -> None:
    seen: dict[str, ContentHash] = {}
    for prior in record.events:
        for artifact in prior.artifacts:
            seen[artifact.path] = artifact.content_hash
    for artifact in event.artifacts:
        prior_hash = seen.get(artifact.path)
        if prior_hash is not None and prior_hash != artifact.content_hash:
            raise _transition_error(
                record,
                event,
                f"content hash drift for artifact {artifact.path!r}",
            )
        seen[artifact.path] = artifact.content_hash


def _validate_event_identity(record: LifecycleRecord, event: LifecycleEvent) -> None:
    if any(prior.event_id == event.event_id for prior in record.events):
        raise _transition_error(record, event, f"duplicate event_id {event.event_id!r} in lineage")


def _validate_analysis_freshness(record: LifecycleRecord, event: LifecycleEvent) -> None:
    if event.kind is not LifecycleEventKind.ANALYZE:
        return
    for prior in record.events:
        same_actor = prior.actor.actor_id == event.actor.actor_id
        same_session = prior.actor.session_id == event.actor.session_id
        if prior.actor.role is ActorRole.IMPLEMENTATION and (same_actor or same_session):
            raise _transition_error(
                record,
                event,
                "fresh analysis cannot be claimed by an implementation actor or session in this lineage",
            )


def _validate_review_packet(record: LifecycleRecord, event: LifecycleEvent) -> None:
    if event.kind is not LifecycleEventKind.BLIND_REVIEW:
        return
    if event.sealed_packet is None or event.sealed_packet not in event.artifacts:
        raise _transition_error(
            record,
            event,
            "blind review requires a sealed review packet with a recorded content identity",
        )


def _validate_filing(record: LifecycleRecord, event: LifecycleEvent) -> None:
    if event.kind is not LifecycleEventKind.FILE:
        return
    evidence = event.filed_verdict
    if evidence is None:
        raise _transition_error(record, event, "filing requires filed verdict evidence")
    if evidence.narrative not in event.artifacts or evidence.machine not in event.artifacts:
        raise _transition_error(
            record,
            event,
            "filing requires narrative and machine verdict identities in the event artifacts",
        )
    if not any(event.sealed_packet == evidence.review_packet for event in record.events):
        raise _transition_error(
            record,
            event,
            "filed verdict review packet does not match a sealed packet in this lineage",
        )


def _latest_filed_evidence(record: LifecycleRecord, event: LifecycleEvent) -> FiledVerdictEvidence:
    for prior in reversed(record.events):
        if prior.kind is LifecycleEventKind.FILE and prior.filed_verdict is not None:
            return prior.filed_verdict
    raise _transition_error(record, event, "successor authorization requires a filed verdict")


def _validate_filed_outcome(record: LifecycleRecord, event: LifecycleEvent) -> None:
    expected_authorizations = {
        LifecycleEventKind.AUTHORIZE_SUCCESSOR: SuccessorAuthorization.AUTHORIZED,
        LifecycleEventKind.RECORD_VALID_NEGATIVE: SuccessorAuthorization.VALID_NEGATIVE,
        LifecycleEventKind.RETIRE: SuccessorAuthorization.RETIRED,
    }
    expected = expected_authorizations.get(event.kind)
    if expected is None:
        return
    evidence = _latest_filed_evidence(record, event)
    if evidence.open_major_findings:
        raise _transition_error(
            record,
            event,
            "filed outcome is blocked by open MAJOR review findings",
        )
    if evidence.successor_authorization is not expected:
        raise _transition_error(
            record,
            event,
            f"filed verdict authorizes {evidence.successor_authorization.value}, not {expected.value}",
        )
    if expected is SuccessorAuthorization.AUTHORIZED and not evidence.eligible_strategies:
        raise _transition_error(
            record,
            event,
            "successor authorization requires at least one eligible successor strategy",
        )
    if expected is not SuccessorAuthorization.AUTHORIZED and evidence.eligible_strategies:
        raise _transition_error(
            record,
            event,
            f"{expected.value} filed outcome must not name eligible successor strategies",
        )


def transition(record: LifecycleRecord, event: LifecycleEvent) -> LifecycleRecord:
    """Return a new record after one legal, fully-attested transition.

    Integration seams intentionally end here: a capture adapter creates the
    ``CaptureDvc`` event, an analysis session creates ``Analyze``, a review
    adapter supplies the sealed packet, and a validated verdict adapter
    supplies ``FiledVerdictEvidence``.  This function performs no I/O and no
    metric calculation.
    """

    if getattr(record, "_factory_token", None) is not _RECORD_FACTORY_TOKEN:
        msg = (
            "LifecycleRecord is not factory-attested; use LifecycleRecord.start() and transition()"
        )
        raise TypeError(msg)
    target = _target_state(record, event)
    _validate_role(record, event)
    _validate_event_identity(record, event)
    _validate_no_hash_drift(record, event)
    _validate_analysis_freshness(record, event)
    _validate_review_packet(record, event)
    _validate_filing(record, event)
    _validate_filed_outcome(record, event)
    return _attested_record(rung=record.rung, state=target, events=(*record.events, event))
