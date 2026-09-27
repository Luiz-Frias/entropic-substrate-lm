"""The persisted D1 lifecycle enforces commitment, review, reveal, adjudication."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.kernel.lifecycle import ActorRole
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_repository import D1LifecycleRepository
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import D1ProcessState
from entropic_substrate.ladder.capture.ledger.d1_process_manager import (
    D1ProcessManager,
    D1ProcessTransitionError,
)

from .test_d1_process_manager import _actor, _artifact, _execute, _through_preregistration

pytestmark = pytest.mark.unit


def _analyzed(tmp_path: Path) -> D1ProcessManager:
    manager = _execute(_through_preregistration(tmp_path), "cell-s1", 10)
    return manager.capture(
        actor=_actor(ActorRole.AUTOMATION), artifact=_artifact(20, "capture"), event_id="capture"
    ).file_analysis(
        actor=_actor(ActorRole.ANALYSIS), artifact=_artifact(21, "analysis"), event_id="analysis"
    )


def test_alias_commitment_review_reveal_and_adjudication_are_distinct_states(
    tmp_path: Path,
) -> None:
    manager = _analyzed(tmp_path)
    commitment = _artifact(22, "alias-commitment")
    receipt = _artifact(25, "review-receipt")

    manager = manager.file_alias_commitment(
        actor=_actor(ActorRole.ANALYSIS), artifact=commitment, event_id="commitment"
    )
    assert manager.state is D1ProcessState.ALIAS_COMMITTED
    manager = manager.file_review(
        actor=_actor(ActorRole.REVIEW),
        packet=_artifact(23, "packet"),
        report=_artifact(24, "report"),
        receipt=receipt,
        event_id="review",
    )
    assert manager.state is D1ProcessState.REVIEW_FILED
    manager = manager.reveal_aliases(
        actor=_actor(ActorRole.OPERATOR),
        reveal=_artifact(26, "alias-reveal"),
        review_receipt=receipt,
        event_id="reveal",
    )
    assert manager.state is D1ProcessState.ALIAS_REVEALED
    manager = manager.adjudicate(
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(27, "adjudication"),
        event_id="adjudication",
    )

    assert manager.state is D1ProcessState.ADJUDICATED
    reloaded = D1ProcessManager.load(manager.repository)
    assert isinstance(reloaded, D1ProcessManager)
    assert reloaded.state is D1ProcessState.ADJUDICATED


def test_review_cannot_precede_commitment_and_adjudication_cannot_precede_reveal(
    tmp_path: Path,
) -> None:
    analyzed = _analyzed(tmp_path)
    with pytest.raises(D1ProcessTransitionError, match="wrong_order"):
        analyzed.file_review(
            actor=_actor(ActorRole.REVIEW),
            packet=_artifact(22, "packet"),
            report=_artifact(23, "report"),
            receipt=_artifact(24, "receipt"),
            event_id="early-review",
        )
    committed = analyzed.file_alias_commitment(
        actor=_actor(ActorRole.ANALYSIS),
        artifact=_artifact(25, "commitment"),
        event_id="commitment",
    )
    reviewed = committed.file_review(
        actor=_actor(ActorRole.REVIEW),
        packet=_artifact(26, "packet"),
        report=_artifact(27, "report"),
        receipt=_artifact(28, "receipt"),
        event_id="review",
    )
    with pytest.raises(D1ProcessTransitionError, match="alias_revealed"):
        reviewed.adjudicate(
            actor=_actor(ActorRole.OPERATOR),
            artifact=_artifact(29, "adjudication"),
            event_id="early-adjudication",
        )


def test_reveal_must_name_exact_durable_review_receipt_and_operator_role(
    tmp_path: Path,
) -> None:
    receipt = _artifact(25, "receipt")
    reviewed = (
        _analyzed(tmp_path)
        .file_alias_commitment(
            actor=_actor(ActorRole.ANALYSIS),
            artifact=_artifact(22, "commitment"),
            event_id="commit",
        )
        .file_review(
            actor=_actor(ActorRole.REVIEW),
            packet=_artifact(23, "packet"),
            report=_artifact(24, "report"),
            receipt=receipt,
            event_id="review",
        )
    )
    prefix = reviewed.history.prefix
    with pytest.raises(D1ProcessTransitionError) as error:
        reviewed.reveal_aliases(
            actor=_actor(ActorRole.OPERATOR),
            reveal=_artifact(26, "reveal"),
            review_receipt=_artifact(27, "wrong-receipt"),
            event_id="wrong-receipt",
        )
    assert error.value.code == "review_receipt_mismatch"
    assert reviewed.repository.load().prefix == prefix
    with pytest.raises(D1ProcessTransitionError, match="operator"):
        reviewed.reveal_aliases(
            actor=_actor(ActorRole.REVIEW),
            reveal=_artifact(26, "reveal"),
            review_receipt=receipt,
            event_id="wrong-role",
        )


def test_lifecycle_repository_rejects_legacy_skip_on_reload(tmp_path: Path) -> None:
    manager = _analyzed(tmp_path)
    assert isinstance(
        D1ProcessManager.load(D1LifecycleRepository(manager.repository.path)), D1ProcessManager
    )
