"""Pure D1 process replay contract tests."""

from __future__ import annotations

import pytest

from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
    LifecycleState,
)
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import (
    D1LifecycleEvent,
    D1ProcessEventKind,
    D1ProcessState,
)
from entropic_substrate.ladder.capture.ledger.d1_process_replay import (
    D1ProcessTransitionError,
    advance_d1_process,
    replay_d1_process,
)

pytestmark = pytest.mark.unit


def _artifact(index: int) -> ArtifactIdentity:
    return ArtifactIdentity(f"replay/{index}.json", ContentHash("sha256", f"{index:064x}"))


def _event(
    sequence: int,
    kind: D1ProcessEventKind,
    role: ActorRole,
    *,
    subject: str | None = None,
    artifacts: tuple[ArtifactIdentity, ...] | None = None,
) -> D1LifecycleEvent:
    return D1LifecycleEvent(
        event_seq=sequence,
        event_id=f"event-{sequence}",
        event_kind=kind,
        actor=ActorIdentity(role.value, f"{role.value}-{sequence}", role),
        artifacts=artifacts or (_artifact(sequence + 1),),
        subject_id=subject,
    )


def _contract() -> D1LifecycleEvent:
    return _event(
        0,
        D1ProcessEventKind.CONTRACT_VERIFIED,
        ActorRole.IMPLEMENTATION,
        artifacts=(_artifact(1), _artifact(2)),
    )


def test_replay_projects_operational_events_through_the_pure_lifecycle() -> None:
    events = (
        _contract(),
        _event(1, D1ProcessEventKind.PILOT_FILED, ActorRole.OPERATOR),
        _event(2, D1ProcessEventKind.PREREGISTRATION_FILED, ActorRole.OPERATOR),
        _event(
            3,
            D1ProcessEventKind.LAUNCH_ADMITTED,
            ActorRole.OPERATOR,
            subject="none-s1",
        ),
        _event(4, D1ProcessEventKind.RUNNING, ActorRole.AUTOMATION, subject="none-s1"),
        _event(5, D1ProcessEventKind.RESUMED, ActorRole.AUTOMATION, subject="none-s1"),
    )

    replay = replay_d1_process(events)

    assert replay.state is D1ProcessState.RUNNING
    assert replay.pure_lifecycle.state is LifecycleState.CONFIRMATORY_RUNNING
    assert replay.active_subject == "none-s1"


def test_replay_refuses_sequence_fork_and_wrong_artifact_cardinality() -> None:
    duplicate_sequence = _event(
        0,
        D1ProcessEventKind.PILOT_FILED,
        ActorRole.OPERATOR,
    )
    with pytest.raises(D1ProcessTransitionError) as error:
        replay_d1_process((_contract(), duplicate_sequence))
    assert error.value.code == "event_identity"

    replay = advance_d1_process(None, _contract())
    malformed_review = _event(
        1,
        D1ProcessEventKind.REVIEW_FILED,
        ActorRole.REVIEW,
    )
    with pytest.raises(D1ProcessTransitionError) as error:
        advance_d1_process(replay, malformed_review)
    assert error.value.code == "artifact_cardinality"


def test_replay_requires_a_nonempty_tuple_history() -> None:
    with pytest.raises(D1ProcessTransitionError) as error:
        replay_d1_process(())

    assert error.value.code == "missing_history"
