"""Durable facade for the D1 flywheel replay policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from entropic_substrate.evidence.d1_lifecycle_repository import (
    D1LifecycleCorruptionError,
    D1LifecycleHistory,
    D1LifecycleRepository,
    D1LifecycleRepositoryGap,
)
from entropic_substrate.evidence.d1_lifecycle_values import (
    D1LifecycleEvent,
    D1ProblemKind,
    D1ProcessEventKind,
    D1ProcessState,
)
from entropic_substrate.evidence.d1_process_replay import (
    D1ProcessReplay,
    D1ProcessTransitionError,
    advance_d1_process,
    reject_transition,
    replay_d1_process,
)
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ArtifactIdentity,
    LifecycleState,
    SuccessorAuthorization,
)


@dataclass(frozen=True, slots=True, init=False)
class D1ProcessManager:
    """A persisted process snapshot; every method returns its durable successor."""

    repository: D1LifecycleRepository
    history: D1LifecycleHistory
    _replay: D1ProcessReplay = field(repr=False)

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1ProcessManager is factory-only; use start() or load()")

    @classmethod
    def _from_history(
        cls, repository: D1LifecycleRepository, history: D1LifecycleHistory
    ) -> D1ProcessManager:
        try:
            replay = replay_d1_process(history.events)
        except D1ProcessTransitionError as exc:
            raise D1LifecycleCorruptionError(
                code="semantic_history", path=repository.path, detail=str(exc)
            ) from exc
        manager = object.__new__(cls)
        object.__setattr__(manager, "repository", repository)
        object.__setattr__(manager, "history", history)
        object.__setattr__(manager, "_replay", replay)
        return manager

    @classmethod
    def load(cls, repository: D1LifecycleRepository) -> D1ProcessManager | D1LifecycleRepositoryGap:
        loaded = repository.try_load()
        if isinstance(loaded, D1LifecycleRepositoryGap):
            return loaded
        return cls._from_history(repository, loaded)

    @classmethod
    def start(
        cls,
        repository: D1LifecycleRepository,
        *,
        actor: ActorIdentity,
        implementation_artifact: ArtifactIdentity,
        contract_artifact: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        if not isinstance(repository.try_load(), D1LifecycleRepositoryGap):
            raise reject_transition(
                "already_started", None, "repository already contains a lifecycle"
            )
        event = D1LifecycleEvent(
            event_seq=0,
            event_id=event_id,
            event_kind=D1ProcessEventKind.CONTRACT_VERIFIED,
            actor=actor,
            artifacts=(implementation_artifact, contract_artifact),
        )
        advance_d1_process(None, event)
        history = repository.append(event, expected_prefix=None)
        return cls._from_history(repository, history)

    @property
    def state(self) -> D1ProcessState:
        return self._replay.state

    @property
    def events(self) -> tuple[D1LifecycleEvent, ...]:
        return self.history.events

    @property
    def active_subject(self) -> str | None:
        return self._replay.active_subject

    @property
    def finalized_subjects(self) -> tuple[str, ...]:
        return self._replay.finalized_subjects

    @property
    def scientific_state(self) -> LifecycleState:
        return self._replay.pure_lifecycle.state

    def _commit(self, event: D1LifecycleEvent) -> D1ProcessManager:
        advance_d1_process(self._replay, event)
        history = self.repository.append(event, expected_prefix=self.history.prefix)
        return self._from_history(self.repository, history)

    def _event(
        self,
        kind: D1ProcessEventKind,
        actor: ActorIdentity,
        artifacts: tuple[ArtifactIdentity, ...],
        event_id: str,
        *,
        subject_id: str | None = None,
        problem_kind: D1ProblemKind | None = None,
        gate: SuccessorAuthorization | None = None,
        strategies: tuple[str, ...] = (),
    ) -> D1LifecycleEvent:
        return D1LifecycleEvent(
            event_seq=len(self.events),
            event_id=event_id,
            event_kind=kind,
            actor=actor,
            artifacts=artifacts,
            subject_id=subject_id,
            problem_kind=problem_kind,
            gate_authorization=gate,
            eligible_strategies=strategies,
        )

    def file_pilot(
        self, *, actor: ActorIdentity, artifact: ArtifactIdentity, event_id: str
    ) -> D1ProcessManager:
        return self._commit(
            self._event(D1ProcessEventKind.PILOT_FILED, actor, (artifact,), event_id)
        )

    def file_preregistration(
        self, *, actor: ActorIdentity, artifact: ArtifactIdentity, event_id: str
    ) -> D1ProcessManager:
        return self._commit(
            self._event(D1ProcessEventKind.PREREGISTRATION_FILED, actor, (artifact,), event_id)
        )

    def admit_launch(
        self,
        *,
        subject_id: str,
        actor: ActorIdentity,
        artifact: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        return self._subject_event(
            D1ProcessEventKind.LAUNCH_ADMITTED,
            subject_id=subject_id,
            actor=actor,
            artifact=artifact,
            event_id=event_id,
        )

    def start_running(
        self,
        *,
        subject_id: str,
        actor: ActorIdentity,
        artifact: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        return self._subject_event(
            D1ProcessEventKind.RUNNING,
            subject_id=subject_id,
            actor=actor,
            artifact=artifact,
            event_id=event_id,
        )

    def resume(
        self,
        *,
        subject_id: str,
        actor: ActorIdentity,
        artifact: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        return self._subject_event(
            D1ProcessEventKind.RESUMED,
            subject_id=subject_id,
            actor=actor,
            artifact=artifact,
            event_id=event_id,
        )

    def record_trainer_stop(
        self,
        *,
        subject_id: str,
        actor: ActorIdentity,
        artifact: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        return self._subject_event(
            D1ProcessEventKind.TRAINER_STOPPED,
            subject_id=subject_id,
            actor=actor,
            artifact=artifact,
            event_id=event_id,
        )

    def finalize_operator(
        self,
        *,
        subject_id: str,
        actor: ActorIdentity,
        artifact: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        return self._subject_event(
            D1ProcessEventKind.OPERATOR_FINALIZED,
            subject_id=subject_id,
            actor=actor,
            artifact=artifact,
            event_id=event_id,
        )

    def _subject_event(
        self,
        kind: D1ProcessEventKind,
        *,
        subject_id: str,
        actor: ActorIdentity,
        artifact: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        return self._commit(self._event(kind, actor, (artifact,), event_id, subject_id=subject_id))

    def capture(
        self, *, actor: ActorIdentity, artifact: ArtifactIdentity, event_id: str
    ) -> D1ProcessManager:
        return self._commit(self._event(D1ProcessEventKind.CAPTURED, actor, (artifact,), event_id))

    def file_analysis(
        self, *, actor: ActorIdentity, artifact: ArtifactIdentity, event_id: str
    ) -> D1ProcessManager:
        return self._commit(
            self._event(D1ProcessEventKind.ANALYSIS_FILED, actor, (artifact,), event_id)
        )

    def file_review(
        self,
        *,
        actor: ActorIdentity,
        packet: ArtifactIdentity,
        report: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        return self._commit(
            self._event(D1ProcessEventKind.REVIEW_FILED, actor, (packet, report), event_id)
        )

    def adjudicate(
        self, *, actor: ActorIdentity, artifact: ArtifactIdentity, event_id: str
    ) -> D1ProcessManager:
        return self._commit(
            self._event(D1ProcessEventKind.ADJUDICATED, actor, (artifact,), event_id)
        )

    def record_problem(
        self,
        *,
        actor: ActorIdentity,
        kind: D1ProblemKind,
        artifact: ArtifactIdentity,
        event_id: str,
    ) -> D1ProcessManager:
        return self._commit(
            self._event(
                D1ProcessEventKind.PROBLEM_RECORDED,
                actor,
                (artifact,),
                event_id,
                problem_kind=kind,
            )
        )


__all__ = ["D1ProcessManager", "D1ProcessTransitionError"]
