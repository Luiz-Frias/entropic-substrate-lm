"""Factory-gated verdict filing and successor transitions for D1."""

from __future__ import annotations

from dataclasses import dataclass

from entropic_substrate.evidence.d1_filing_graph import ValidatedD1Filing
from entropic_substrate.evidence.d1_gate import decide_filed_verdict
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessEventKind, D1ProcessState
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.evidence.d1_process_replay import reject_transition, replay_d1_process
from entropic_substrate.evidence.d1_types import (
    FiledGateAuthorization,
    FiledGateRetirement,
    FiledGateValidNegative,
)
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ArtifactIdentity,
    FiledVerdictEvidence,
    SuccessorAuthorization,
)

type _GateOutcomeType = (
    type[FiledGateAuthorization] | type[FiledGateValidNegative] | type[FiledGateRetirement]
)


def _require_state(manager: D1ProcessManager, expected: D1ProcessState, action: str) -> None:
    if manager.state is not expected:
        raise reject_transition("wrong_order", manager.state, f"{action} requires {expected.value}")


def _event_for(manager: D1ProcessManager, kind: D1ProcessEventKind):
    for event in manager.events:
        if event.event_kind is kind:
            return event
    raise reject_transition("missing_prior", manager.state, f"missing {kind.value} identity")


def _check_filing(manager: D1ProcessManager, filing: ValidatedD1Filing) -> FiledVerdictEvidence:
    if type(filing) is not ValidatedD1Filing:
        raise TypeError("D1 process filing requires factory-attested ValidatedD1Filing")
    evidence = filing.to_filed_evidence()
    expected = {
        D1ProcessEventKind.PREREGISTRATION_FILED: filing.manifest.artifacts[
            "preregistration"
        ].artifact_identity(),
        D1ProcessEventKind.CAPTURED: filing.manifest.artifacts[
            "raw_capture_manifest"
        ].artifact_identity(),
        D1ProcessEventKind.ANALYSIS_FILED: filing.manifest.artifacts[
            "blinded_analysis"
        ].artifact_identity(),
        D1ProcessEventKind.ALIAS_COMMITMENT_FILED: filing.manifest.artifacts[
            "alias_commitment"
        ].artifact_identity(),
        D1ProcessEventKind.ALIAS_REVEALED: filing.manifest.artifacts[
            "alias_reveal"
        ].artifact_identity(),
    }
    for kind, artifact in expected.items():
        if _event_for(manager, kind).artifacts[0] != artifact:
            raise reject_transition(
                "filing_identity", manager.state, f"{kind.value} identity differs from filing"
            )
    review = _event_for(manager, D1ProcessEventKind.REVIEW_FILED)
    expected_review = (
        evidence.review_packet,
        filing.manifest.artifacts["reviewer_report"].artifact_identity(),
        filing.manifest.artifacts["reviewer_execution_receipt"].artifact_identity(),
    )
    if review.artifacts != expected_review:
        raise reject_transition(
            "filing_identity", manager.state, "review identities differ from filing"
        )
    return evidence


@dataclass(frozen=True, slots=True, init=False)
class D1FiledProcess(D1ProcessManager):
    """A filed-verdict capability whose methods alone may invoke the gate."""

    filing: ValidatedD1Filing

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1FiledProcess is factory-only; file or resume a validated D1 verdict")

    @classmethod
    def _copy(cls, manager: D1ProcessManager, filing: ValidatedD1Filing) -> D1FiledProcess:
        process = object.__new__(cls)
        object.__setattr__(process, "repository", manager.repository)
        object.__setattr__(process, "history", manager.history)
        object.__setattr__(process, "_replay", replay_d1_process(manager.events))
        object.__setattr__(process, "filing", filing)
        return process

    @property
    def manager(self) -> D1ProcessManager:
        return self

    @classmethod
    def file(
        cls,
        manager: D1ProcessManager,
        *,
        actor: ActorIdentity,
        filing: ValidatedD1Filing,
        event_id: str,
    ) -> D1FiledProcess:
        _require_state(manager, D1ProcessState.ADJUDICATED, "verdict filing")
        evidence = _check_filing(manager, filing)
        process = cls._copy(manager, filing)
        event = process._event(
            D1ProcessEventKind.VERDICT_FILED,
            actor,
            (evidence.narrative, evidence.machine),
            event_id,
        )
        return cls._copy(process._commit(event), filing)

    @classmethod
    def attest_replay(cls, manager: D1ProcessManager, filing: ValidatedD1Filing) -> D1FiledProcess:
        _require_state(manager, D1ProcessState.VERDICT_FILED, "filed process resume")
        evidence = _check_filing(manager, filing)
        verdict = _event_for(manager, D1ProcessEventKind.VERDICT_FILED)
        if verdict.artifacts != (evidence.narrative, evidence.machine):
            raise reject_transition(
                "filing_identity", manager.state, "filed verdict identity differs"
            )
        return cls._copy(manager, filing)

    def _file_gate(
        self,
        *,
        actor: ActorIdentity,
        artifact: ArtifactIdentity,
        event_id: str,
        expected_type: _GateOutcomeType,
        kind: D1ProcessEventKind,
        authorization: SuccessorAuthorization,
    ) -> D1ProcessManager:
        _require_state(self, D1ProcessState.VERDICT_FILED, "successor gate")
        evidence = _check_filing(self, self.filing)
        verdict = _event_for(self, D1ProcessEventKind.VERDICT_FILED)
        if verdict.artifacts != (evidence.narrative, evidence.machine):
            raise reject_transition("filing_identity", self.state, "filed verdict identity differs")
        outcome = decide_filed_verdict(self.filing)
        if not isinstance(outcome, expected_type):
            raise reject_transition(
                "gate_outcome", self.state, f"filed gate did not produce {kind.value}"
            )
        strategies = (
            outcome.eligible_strategies if isinstance(outcome, FiledGateAuthorization) else ()
        )
        event = self._event(
            kind,
            actor,
            (artifact, evidence.narrative, evidence.machine),
            event_id,
            gate=authorization,
            strategies=strategies,
        )
        return self._commit(event)

    def authorize_d2(
        self, *, actor: ActorIdentity, artifact: ArtifactIdentity, event_id: str
    ) -> D1ProcessManager:
        return self._file_gate(
            actor=actor,
            artifact=artifact,
            event_id=event_id,
            expected_type=FiledGateAuthorization,
            kind=D1ProcessEventKind.GATE_AUTHORIZED,
            authorization=SuccessorAuthorization.AUTHORIZED,
        )

    def record_valid_negative(
        self, *, actor: ActorIdentity, artifact: ArtifactIdentity, event_id: str
    ) -> D1ProcessManager:
        return self._file_gate(
            actor=actor,
            artifact=artifact,
            event_id=event_id,
            expected_type=FiledGateValidNegative,
            kind=D1ProcessEventKind.GATE_VALID_NEGATIVE,
            authorization=SuccessorAuthorization.VALID_NEGATIVE,
        )

    def retire(
        self, *, actor: ActorIdentity, artifact: ArtifactIdentity, event_id: str
    ) -> D1ProcessManager:
        return self._file_gate(
            actor=actor,
            artifact=artifact,
            event_id=event_id,
            expected_type=FiledGateRetirement,
            kind=D1ProcessEventKind.GATE_RETIRED,
            authorization=SuccessorAuthorization.RETIRED,
        )


def file_d1_verdict(
    manager: D1ProcessManager,
    *,
    actor: ActorIdentity,
    filing: ValidatedD1Filing,
    event_id: str,
) -> D1FiledProcess:
    return D1FiledProcess.file(manager, actor=actor, filing=filing, event_id=event_id)


def resume_d1_filed_process(manager: D1ProcessManager, filing: ValidatedD1Filing) -> D1FiledProcess:
    return D1FiledProcess.attest_replay(manager, filing)


__all__ = ["D1FiledProcess", "file_d1_verdict", "resume_d1_filed_process"]
