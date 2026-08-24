"""Guard teeth for D1 process identity, replay, and problem states."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_lifecycle_repository import D1LifecycleCorruptionError
from entropic_substrate.evidence.d1_lifecycle_values import (
    D1LifecycleEvent,
    D1ProblemKind,
    D1ProcessEventKind,
    D1ProcessState,
)
from entropic_substrate.evidence.d1_process_manager import (
    D1ProcessManager,
    D1ProcessTransitionError,
)
from entropic_substrate.evidence.lifecycle import ActorIdentity, ActorRole

from .test_d1_process_manager import _actor, _artifact, _execute, _start, _through_preregistration

pytestmark = pytest.mark.unit


def _captured(tmp_path: Path) -> D1ProcessManager:
    manager = _execute(_through_preregistration(tmp_path), "none-s1", 10)
    return manager.capture(
        actor=_actor(ActorRole.AUTOMATION), artifact=_artifact(20, "capture"), event_id="capture"
    )


def test_analysis_cannot_reuse_implementation_actor_or_session(tmp_path: Path) -> None:
    manager = _captured(tmp_path)

    for actor in (
        ActorIdentity("implementation", "fresh", ActorRole.ANALYSIS),
        ActorIdentity("fresh", "implementation-session", ActorRole.ANALYSIS),
    ):
        with pytest.raises(D1ProcessTransitionError) as error:
            manager.file_analysis(actor=actor, artifact=_artifact(21, "analysis"), event_id="bad")
        assert error.value.code == "role_separation"

    assert manager.repository.load().prefix == manager.history.prefix


def test_resume_requires_active_subject_and_finalized_subject_cannot_relaunch(
    tmp_path: Path,
) -> None:
    manager = _through_preregistration(tmp_path)
    manager = manager.admit_launch(
        subject_id="none-s1",
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(10, "admission"),
        event_id="admit",
    )
    manager = manager.start_running(
        subject_id="none-s1",
        actor=_actor(ActorRole.AUTOMATION),
        artifact=_artifact(11, "running"),
        event_id="running",
    )

    with pytest.raises(D1ProcessTransitionError) as error:
        manager.resume(
            subject_id="sg-s1",
            actor=_actor(ActorRole.AUTOMATION),
            artifact=_artifact(12, "resume"),
            event_id="wrong-subject",
        )
    assert error.value.code == "subject_mismatch"

    manager = manager.record_trainer_stop(
        subject_id="none-s1",
        actor=_actor(ActorRole.AUTOMATION),
        artifact=_artifact(13, "stop"),
        event_id="stop",
    ).finalize_operator(
        subject_id="none-s1",
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(14, "finalize"),
        event_id="finalize",
    )
    with pytest.raises(D1ProcessTransitionError) as error:
        manager.admit_launch(
            subject_id="none-s1",
            actor=_actor(ActorRole.OPERATOR),
            artifact=_artifact(15, "admission"),
            event_id="duplicate",
        )
    assert error.value.code == "duplicate_subject"


def test_semantically_illegal_canonical_history_fails_hard_on_hydration(tmp_path: Path) -> None:
    manager = _start(tmp_path)
    illegal = D1LifecycleEvent(
        event_seq=1,
        event_id="skip-pilot",
        event_kind=D1ProcessEventKind.PREREGISTRATION_FILED,
        actor=_actor(ActorRole.OPERATOR),
        artifacts=(_artifact(3, "preregistration"),),
    )
    manager.repository.append(illegal, expected_prefix=manager.history.prefix)

    with pytest.raises(D1LifecycleCorruptionError) as error:
        D1ProcessManager.load(manager.repository)

    assert error.value.code == "semantic_history"


@pytest.mark.parametrize(
    ("kind", "role"),
    [
        (D1ProblemKind.INVALIDATED, ActorRole.OPERATOR),
        (D1ProblemKind.NUMERICAL_DEFECT, ActorRole.AUTOMATION),
        (D1ProblemKind.EVIDENCE_GAP, ActorRole.ANALYSIS),
        (D1ProblemKind.OPERATOR_DECISION_NEEDED, ActorRole.ANALYSIS),
    ],
)
def test_bounded_failures_persist_as_typed_terminal_problem_states(
    tmp_path: Path,
    kind: D1ProblemKind,
    role: ActorRole,
) -> None:
    manager = _start(tmp_path / kind.value)

    manager = manager.record_problem(
        actor=_actor(role),
        kind=kind,
        artifact=_artifact(40, kind.value),
        event_id=f"problem-{kind.value}",
    )

    assert manager.state is D1ProcessState.PROBLEM
    assert manager.events[-1].problem_kind is kind
    with pytest.raises(D1ProcessTransitionError) as error:
        manager.record_problem(
            actor=_actor(role),
            kind=kind,
            artifact=_artifact(41, kind.value),
            event_id="second-problem",
        )
    assert error.value.code == "terminal"


def test_capture_analysis_and_review_are_single_campaign_facts(tmp_path: Path) -> None:
    captured = _captured(tmp_path)
    with pytest.raises(D1ProcessTransitionError, match="wrong_order"):
        captured.capture(
            actor=_actor(ActorRole.AUTOMATION),
            artifact=_artifact(21, "capture"),
            event_id="capture-again",
        )
    analyzed = captured.file_analysis(
        actor=_actor(ActorRole.ANALYSIS), artifact=_artifact(22, "analysis"), event_id="analysis"
    )
    with pytest.raises(D1ProcessTransitionError, match="wrong_order"):
        analyzed.file_analysis(
            actor=_actor(ActorRole.ANALYSIS),
            artifact=_artifact(23, "analysis"),
            event_id="analysis-again",
        )
    committed = analyzed.file_alias_commitment(
        actor=_actor(ActorRole.ANALYSIS),
        artifact=_artifact(24, "commitment"),
        event_id="commitment",
    )
    receipt = _artifact(27, "receipt")
    reviewed = committed.file_review(
        actor=_actor(ActorRole.REVIEW),
        packet=_artifact(25, "packet"),
        report=_artifact(26, "report"),
        receipt=receipt,
        event_id="review",
    )
    with pytest.raises(D1ProcessTransitionError, match="wrong_order"):
        reviewed.file_review(
            actor=_actor(ActorRole.REVIEW),
            packet=_artifact(28, "packet"),
            report=_artifact(29, "report"),
            receipt=_artifact(30, "receipt"),
            event_id="review-again",
        )
