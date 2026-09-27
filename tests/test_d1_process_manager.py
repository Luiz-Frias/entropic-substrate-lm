"""Primary persisted-flywheel tests for the D1 process manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_repository import D1LifecycleRepository
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import D1ProcessState
from entropic_substrate.ladder.capture.ledger.d1_process_manager import (
    D1ProcessManager,
    D1ProcessTransitionError,
)

pytestmark = pytest.mark.unit


def _artifact(index: int, name: str) -> ArtifactIdentity:
    return ArtifactIdentity(
        path=f"d1/{name}-{index}.json",
        content_hash=ContentHash("sha256", f"{index:064x}"),
    )


def _actor(role: ActorRole, *, actor: str | None = None, session: str | None = None):
    return ActorIdentity(actor or role.value, session or f"{role.value}-session", role)


def _start(tmp_path: Path) -> D1ProcessManager:
    return D1ProcessManager.start(
        D1LifecycleRepository(tmp_path / "d1-lifecycle.jsonl"),
        actor=_actor(ActorRole.IMPLEMENTATION),
        implementation_artifact=_artifact(1, "implementation"),
        contract_artifact=_artifact(2, "contract"),
        event_id="contract-verified",
    )


def _through_preregistration(tmp_path: Path) -> D1ProcessManager:
    manager = _start(tmp_path)
    manager = manager.file_pilot(
        actor=_actor(ActorRole.OPERATOR), artifact=_artifact(3, "pilot"), event_id="pilot"
    )
    return manager.file_preregistration(
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(4, "preregistration"),
        event_id="preregister",
    )


def _execute(manager: D1ProcessManager, subject: str, base: int) -> D1ProcessManager:
    manager = manager.admit_launch(
        subject_id=subject,
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(base, "admission"),
        event_id=f"{subject}-admit",
    )
    manager = manager.start_running(
        subject_id=subject,
        actor=_actor(ActorRole.AUTOMATION),
        artifact=_artifact(base + 1, "running"),
        event_id=f"{subject}-running",
    )
    manager = manager.resume(
        subject_id=subject,
        actor=_actor(ActorRole.AUTOMATION),
        artifact=_artifact(base + 2, "resume"),
        event_id=f"{subject}-resume",
    )
    manager = manager.record_trainer_stop(
        subject_id=subject,
        actor=_actor(ActorRole.AUTOMATION),
        artifact=_artifact(base + 3, "trainer-stop"),
        event_id=f"{subject}-stop",
    )
    return manager.finalize_operator(
        subject_id=subject,
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(base + 4, "finalization"),
        event_id=f"{subject}-finalize",
    )


def test_full_operational_spine_persists_repeated_segments_and_reload(tmp_path: Path) -> None:
    manager = _execute(_through_preregistration(tmp_path), "none-s1", 10)
    manager = _execute(manager, "sg-s1", 20)
    manager = manager.capture(
        actor=_actor(ActorRole.AUTOMATION), artifact=_artifact(30, "capture"), event_id="capture"
    )
    manager = manager.file_analysis(
        actor=_actor(ActorRole.ANALYSIS), artifact=_artifact(31, "analysis"), event_id="analysis"
    )
    manager = manager.file_alias_commitment(
        actor=_actor(ActorRole.ANALYSIS),
        artifact=_artifact(32, "alias-commitment"),
        event_id="alias-commitment",
    )
    receipt = _artifact(35, "review-receipt")
    manager = manager.file_review(
        actor=_actor(ActorRole.REVIEW),
        packet=_artifact(33, "review-packet"),
        report=_artifact(34, "review-report"),
        receipt=receipt,
        event_id="review",
    )
    manager = manager.reveal_aliases(
        actor=_actor(ActorRole.OPERATOR),
        reveal=_artifact(36, "alias-reveal"),
        review_receipt=receipt,
        event_id="alias-reveal",
    )
    manager = manager.adjudicate(
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(37, "adjudication"),
        event_id="adjudication",
    )

    reloaded = D1ProcessManager.load(manager.repository)

    assert isinstance(reloaded, D1ProcessManager)
    assert reloaded.state is D1ProcessState.ADJUDICATED
    assert reloaded.finalized_subjects == ("none-s1", "sg-s1")
    assert reloaded.active_subject is None
    assert reloaded.events == manager.events


def test_preregistration_cannot_precede_pilot_and_wrong_role_cannot_file_it(
    tmp_path: Path,
) -> None:
    manager = _start(tmp_path)

    with pytest.raises(D1ProcessTransitionError, match="pilot"):
        manager.file_preregistration(
            actor=_actor(ActorRole.OPERATOR), artifact=_artifact(3, "prereg"), event_id="bad"
        )
    with pytest.raises(D1ProcessTransitionError, match="operator"):
        manager.file_pilot(
            actor=_actor(ActorRole.IMPLEMENTATION),
            artifact=_artifact(3, "pilot"),
            event_id="wrong-role",
        )

    assert len(manager.repository.load().events) == 1
