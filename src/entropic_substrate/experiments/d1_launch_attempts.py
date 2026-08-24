"""Immutable, append-only attempt history for D1 launch decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

from entropic_substrate.experiments.d1_primitives import (
    D1Arm,
    RunClass,
    canonical_digest,
    require_nonempty_string,
    require_sha256,
)

D1_LAUNCH_ATTEMPT_SCHEMA: Final = "d1-launch-attempt-v1"
D1_LAUNCH_LEDGER_SCHEMA: Final = "d1-launch-ledger-v1"


class D1LaunchAttemptStatus(StrEnum):
    """Event-sourced states of one launch attempt."""

    RECORDED = "recorded"
    TRAINER_STARTED = "trainer_started"
    FAILED = "failed"
    ABANDONED = "abandoned"
    COMPLETED = "completed"


_TERMINAL: Final = frozenset(
    {
        D1LaunchAttemptStatus.FAILED,
        D1LaunchAttemptStatus.ABANDONED,
        D1LaunchAttemptStatus.COMPLETED,
    }
)
_TRANSITIONS: Final = {
    D1LaunchAttemptStatus.RECORDED: frozenset(
        {
            D1LaunchAttemptStatus.TRAINER_STARTED,
            D1LaunchAttemptStatus.FAILED,
            D1LaunchAttemptStatus.ABANDONED,
        }
    ),
    D1LaunchAttemptStatus.TRAINER_STARTED: frozenset(
        {D1LaunchAttemptStatus.FAILED, D1LaunchAttemptStatus.COMPLETED}
    ),
}


def _utc(value: datetime, *, label: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a UTC timestamp")


@dataclass(frozen=True, slots=True)
class D1LaunchAttemptIdentity:
    """Identity whose digest cannot be changed by merely renaming a run."""

    run_class: RunClass
    family_digest: str
    arm: D1Arm
    seed: int
    logical_run_id: str
    registered_design_digest: str | None

    def __post_init__(self) -> None:
        if self.run_class not in (RunClass.PILOT, RunClass.CONFIRMATORY):
            raise ValueError("D1 attempt ledger records only pilot or confirmatory runs")
        require_sha256(self.family_digest, label="D1 attempt family_digest")
        if type(self.arm) is not D1Arm:
            raise TypeError("D1 attempt arm must be a D1Arm")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("D1 attempt seed must be a nonnegative integer")
        require_nonempty_string(self.logical_run_id, label="D1 attempt logical_run_id")
        if self.run_class is RunClass.CONFIRMATORY:
            if self.registered_design_digest is None:
                raise ValueError("confirmatory attempt requires registered_design_digest")
            require_sha256(
                self.registered_design_digest,
                label="D1 attempt registered_design_digest",
            )
        elif self.registered_design_digest is not None:
            raise ValueError("pilot attempt forbids registered_design_digest")

    @property
    def optional_stopping_manifest(self) -> dict[str, object]:
        """Scientific look identity; display-oriented logical run ID is absent."""
        return {
            "arm": self.arm.value,
            "family_digest": self.family_digest,
            "registered_design_digest": self.registered_design_digest,
            "run_class": self.run_class.value,
            "schema_version": D1_LAUNCH_ATTEMPT_SCHEMA,
            "seed": self.seed,
        }

    @property
    def optional_stopping_digest(self) -> str:
        return canonical_digest(self.optional_stopping_manifest)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            **self.optional_stopping_manifest,
            "logical_run_id": self.logical_run_id,
            "optional_stopping_digest": self.optional_stopping_digest,
        }


@dataclass(frozen=True, slots=True)
class D1LaunchAttemptEvent:
    """One append-only state observation for an attempt identity."""

    event_seq: int
    identity: D1LaunchAttemptIdentity
    status: D1LaunchAttemptStatus
    observed_at: datetime
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.event_seq) is not int or self.event_seq < 0:
            raise ValueError("D1 attempt event_seq must be a nonnegative integer")
        if type(self.identity) is not D1LaunchAttemptIdentity:
            raise TypeError("D1 attempt event requires D1LaunchAttemptIdentity")
        if type(self.status) is not D1LaunchAttemptStatus:
            raise TypeError("D1 attempt event status must be D1LaunchAttemptStatus")
        _utc(self.observed_at, label="D1 attempt observed_at")
        requires_reason = self.status in {
            D1LaunchAttemptStatus.FAILED,
            D1LaunchAttemptStatus.ABANDONED,
        }
        if requires_reason:
            if self.reason is None:
                raise ValueError(f"D1 {self.status.value} attempt requires a reason")
            require_nonempty_string(self.reason, label="D1 attempt reason")
        elif self.reason is not None:
            raise ValueError(f"D1 {self.status.value} attempt forbids a reason")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "event_seq": self.event_seq,
            "identity": self.identity.manifest,
            "observed_at": self.observed_at.isoformat(),
            "reason": self.reason,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class D1LaunchAttemptLedger:
    """Persistent value whose constructor revalidates the complete event prefix."""

    events: tuple[D1LaunchAttemptEvent, ...] = ()

    def __post_init__(self) -> None:
        if type(self.events) is not tuple:
            raise TypeError("D1 attempt ledger events must be an immutable tuple")
        histories: dict[str, list[D1LaunchAttemptEvent]] = {}
        logical_ids: dict[str, str] = {}
        for expected_seq, event in enumerate(self.events):
            if type(event) is not D1LaunchAttemptEvent:
                raise TypeError("D1 attempt ledger contains a non-event value")
            if event.event_seq != expected_seq:
                raise ValueError("D1 attempt ledger event sequences must be consecutive from zero")
            digest = event.identity.optional_stopping_digest
            existing_digest = logical_ids.get(event.identity.logical_run_id)
            if existing_digest is not None and existing_digest != digest:
                raise ValueError("D1 attempt logical_run_id cannot identify two scientific looks")
            logical_ids[event.identity.logical_run_id] = digest
            history = histories.setdefault(digest, [])
            _validate_next(history, event)
            history.append(event)

    def record(
        self,
        identity: D1LaunchAttemptIdentity,
        *,
        observed_at: datetime,
    ) -> D1LaunchAttemptLedger:
        digest = identity.optional_stopping_digest
        if any(event.identity.optional_stopping_digest == digest for event in self.events):
            raise ValueError("duplicate D1 optional-stopping identity is an unregistered rerun")
        if any(event.identity.logical_run_id == identity.logical_run_id for event in self.events):
            raise ValueError("D1 logical_run_id is already present in the attempt ledger")
        return self._append(
            D1LaunchAttemptEvent(
                event_seq=len(self.events),
                identity=identity,
                status=D1LaunchAttemptStatus.RECORDED,
                observed_at=observed_at,
            )
        )

    def transition(
        self,
        identity: D1LaunchAttemptIdentity,
        *,
        status: D1LaunchAttemptStatus,
        observed_at: datetime,
        reason: str | None = None,
    ) -> D1LaunchAttemptLedger:
        if status is D1LaunchAttemptStatus.RECORDED:
            raise ValueError("recorded is an initial D1 attempt status, not a transition")
        history = tuple(
            event
            for event in self.events
            if event.identity.optional_stopping_digest == identity.optional_stopping_digest
        )
        if not history:
            raise ValueError("cannot transition an unregistered D1 launch attempt")
        if any(event.identity != identity for event in history):
            raise ValueError("D1 attempt transition identity differs from its recorded identity")
        return self._append(
            D1LaunchAttemptEvent(
                event_seq=len(self.events),
                identity=identity,
                status=status,
                observed_at=observed_at,
                reason=reason,
            )
        )

    def _append(self, event: D1LaunchAttemptEvent) -> D1LaunchAttemptLedger:
        return D1LaunchAttemptLedger(events=(*self.events, event))

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "events": [event.manifest for event in self.events],
            "schema_version": D1_LAUNCH_LEDGER_SCHEMA,
        }

    @property
    def ledger_digest(self) -> str:
        return canonical_digest(self.manifest)


def _validate_next(
    history: list[D1LaunchAttemptEvent],
    event: D1LaunchAttemptEvent,
) -> None:
    if not history:
        if event.status is not D1LaunchAttemptStatus.RECORDED:
            raise ValueError("D1 attempt history must start with recorded status")
        return
    previous = history[-1]
    if event.identity != previous.identity:
        raise ValueError("D1 optional-stopping identity cannot change within an attempt")
    if event.observed_at < previous.observed_at:
        raise ValueError("D1 attempt event cannot precede its previous event")
    if previous.status in _TERMINAL:
        raise ValueError("D1 attempt terminal status cannot transition")
    if event.status not in _TRANSITIONS[previous.status]:
        raise ValueError(
            f"invalid D1 attempt transition {previous.status.value}->{event.status.value}"
        )
