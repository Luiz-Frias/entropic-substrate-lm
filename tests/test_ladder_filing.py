"""Filing identity and new non-authorizing lifecycle postures."""

from pathlib import Path

import pytest

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.file.ladder_filing import file_ladder_verdict

pytestmark = pytest.mark.unit


def test_terminal_and_inconclusive_are_explicit_non_authorizing_postures() -> None:
    from entropic_substrate.kernel.lifecycle import SuccessorAuthorization

    assert SuccessorAuthorization.COMPLETE.value == "COMPLETE"
    assert SuccessorAuthorization.INCONCLUSIVE.value == "INCONCLUSIVE"
    assert callable(file_ladder_verdict)


@pytest.mark.parametrize("posture", ["COMPLETE", "INCONCLUSIVE"])
def test_new_postures_do_not_expand_historical_d1_verdict_schema(posture: str) -> None:
    from entropic_substrate.ladder.file.d1_validator import _parse_successor

    with pytest.raises(ValueError, match="invalid_successor_authorization"):
        _parse_successor({"authorization": posture, "eligible_arms": [], "eligible_strategies": []})


def test_filing_refuses_capture_from_another_lifecycle(tmp_path: Path) -> None:
    from entropic_substrate.kernel.lifecycle import (
        ActorIdentity,
        ActorRole,
        LifecycleEvent,
        LifecycleEventKind,
        LifecycleRecord,
        transition,
    )
    from entropic_substrate.ladder.analyze.ladder_analysis import (
        LadderAnalysisInputs,
        ScientificRunReader,
    )

    from .test_ladder_campaign import campaign

    design = campaign(tmp_path, rung="d2")
    operator = ActorIdentity("operator", "op-1", ActorRole.OPERATOR)
    record = LifecycleRecord.start("d2")
    for kind, role in (
        ("Implement", "implementation"),
        ("VerifyContract", "implementation"),
        ("Preregister", "operator"),
        ("StartConfirmatory", "operator"),
        ("CaptureDvc", "automation"),
        ("Analyze", "analysis"),
        ("BlindReview", "review"),
    ):
        actor = operator if role == "operator" else ActorIdentity(role, role, ActorRole(role))
        record = transition(
            record,
            LifecycleEvent(
                kind,
                LifecycleEventKind(kind),
                actor,
                (design,),
                sealed_packet=design if kind == "BlindReview" else None,
            ),
        )
    inputs = LadderAnalysisInputs(
        design,
        (ArtifactIdentity("different-capture.json", ContentHash("sha256", "f" * 64)),),
        ScientificRunReader(),
    )
    with pytest.raises(ValueError, match=r"capture.*lifecycle"):
        file_ladder_verdict(
            tmp_path,
            inputs,
            analysis=design,
            packet=design,
            review=design,
            narrative=design,
            record=record,
            operator=operator,
            machine_path="verdict.json",
        )
