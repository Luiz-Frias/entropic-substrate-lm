"""Factory-attested verdict filing and successor-gate process tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from d1_verdict_v2_fixtures import decision_contract, machine_verdict
from entropic_substrate.evidence.d1_contract import D1DecisionRule
from entropic_substrate.evidence.d1_filing_graph import ValidatedD1Filing, verify_d1_filing
from entropic_substrate.evidence.d1_lifecycle_repository import (
    D1LifecyclePrefix,
    D1LifecycleRepository,
)
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.d1_process_filing import (
    file_d1_verdict,
    resume_d1_filed_process,
)
from entropic_substrate.evidence.d1_process_manager import (
    D1ProcessManager,
    D1ProcessTransitionError,
)
from entropic_substrate.evidence.lifecycle import ActorRole, LifecycleState, SuccessorAuthorization

from . import test_d1_filing_graph_gate as filing_cases
from .test_d1_process_manager import _actor, _artifact

pytestmark = pytest.mark.unit


def _retired_contract():
    contract = decision_contract()
    negative = contract.decision_rules["VALID_NEGATIVE"]
    rules = dict(contract.decision_rules)
    rules["RETIRED"] = D1DecisionRule(
        successor_authorization=SuccessorAuthorization.RETIRED,
        required_clause_statuses=negative.required_clause_statuses,
        require_integrity_pass=True,
        require_review_clearance=True,
        eligible_strategies=(),
    )
    return replace(
        contract,
        registered_constants={"seeds": [1, 2, 3, 4], "primary": {"alpha": 0.025}},
        decision_rules=rules,
    )


def _retired_machine() -> dict[str, object]:
    payload = machine_verdict(verdict="VALID_NEGATIVE")
    payload["verdict"] = "RETIRED"
    payload["successor"] = {"authorization": "RETIRED", "eligible_strategies": []}
    return payload


def _filing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> ValidatedD1Filing:
    if outcome == "valid_negative":
        monkeypatch.setattr(
            filing_cases,
            "machine_verdict",
            lambda: machine_verdict(verdict="VALID_NEGATIVE"),
        )
    elif outcome == "retired":
        monkeypatch.setattr(filing_cases, "decision_contract", _retired_contract)
        monkeypatch.setattr(filing_cases, "machine_verdict", _retired_machine)
    return verify_d1_filing(*filing_cases._case(tmp_path))


def _adjudicated(tmp_path: Path, filing: ValidatedD1Filing) -> D1ProcessManager:
    artifacts = filing.manifest.artifacts
    manager = D1ProcessManager.start(
        D1LifecycleRepository(tmp_path / "process" / "d1.jsonl"),
        actor=_actor(ActorRole.IMPLEMENTATION),
        implementation_artifact=_artifact(101, "implementation"),
        contract_artifact=_artifact(102, "contract"),
        event_id="contract",
    )
    manager = manager.file_pilot(
        actor=_actor(ActorRole.OPERATOR), artifact=_artifact(103, "pilot"), event_id="pilot"
    ).file_preregistration(
        actor=_actor(ActorRole.OPERATOR),
        artifact=artifacts["preregistration"].artifact_identity(),
        event_id="preregistration",
    )
    manager = manager.admit_launch(
        subject_id="none-s7",
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(104, "admission"),
        event_id="admission",
    ).start_running(
        subject_id="none-s7",
        actor=_actor(ActorRole.AUTOMATION),
        artifact=_artifact(105, "running"),
        event_id="running",
    )
    manager = manager.record_trainer_stop(
        subject_id="none-s7",
        actor=_actor(ActorRole.AUTOMATION),
        artifact=_artifact(106, "stop"),
        event_id="stop",
    ).finalize_operator(
        subject_id="none-s7",
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(107, "finalized"),
        event_id="finalized",
    )
    manager = manager.capture(
        actor=_actor(ActorRole.AUTOMATION),
        artifact=artifacts["raw_capture_manifest"].artifact_identity(),
        event_id="capture",
    ).file_analysis(
        actor=_actor(ActorRole.ANALYSIS),
        artifact=artifacts["blinded_analysis"].artifact_identity(),
        event_id="analysis",
    )
    manager = manager.file_review(
        actor=_actor(ActorRole.REVIEW),
        packet=artifacts["review_packet"].artifact_identity(),
        report=artifacts["reviewer_report"].artifact_identity(),
        event_id="review",
    )
    return manager.adjudicate(
        actor=_actor(ActorRole.OPERATOR),
        artifact=artifacts["alias_reveal"].artifact_identity(),
        event_id="adjudication",
    )


@pytest.mark.parametrize(
    ("outcome", "method", "expected_state", "expected_pure_state"),
    [
        (
            "authorized",
            "authorize_d2",
            D1ProcessState.GATE_AUTHORIZED,
            LifecycleState.SUCCESSOR_AUTHORIZED,
        ),
        (
            "valid_negative",
            "record_valid_negative",
            D1ProcessState.VALID_NEGATIVE,
            LifecycleState.VALID_NEGATIVE,
        ),
        ("retired", "retire", D1ProcessState.RETIRED, LifecycleState.RETIRED),
    ],
)
def test_only_factory_gate_outcome_reaches_matching_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    method: str,
    expected_state: D1ProcessState,
    expected_pure_state: LifecycleState,
) -> None:
    filing = _filing(tmp_path, monkeypatch, outcome)
    filed = file_d1_verdict(
        _adjudicated(tmp_path, filing),
        actor=_actor(ActorRole.OPERATOR),
        filing=filing,
        event_id="verdict",
    )
    reloaded_filed = D1ProcessManager.load(filed.manager.repository)
    assert isinstance(reloaded_filed, D1ProcessManager)
    filed = resume_d1_filed_process(reloaded_filed, filing)

    terminal = getattr(filed, method)(
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(120, "gate"),
        event_id="gate",
    )

    assert terminal.state is expected_state
    assert terminal.scientific_state is expected_pure_state
    reloaded = D1ProcessManager.load(terminal.repository)
    assert isinstance(reloaded, D1ProcessManager)
    assert reloaded.state is expected_state


def test_detached_filing_and_operator_selected_wrong_outcome_cannot_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filing = _filing(tmp_path, monkeypatch, "authorized")
    adjudicated = _adjudicated(tmp_path, filing)

    with pytest.raises(TypeError, match="factory-attested"):
        file_d1_verdict(
            adjudicated,
            actor=_actor(ActorRole.OPERATOR),
            filing=filing.detached,  # type: ignore[arg-type]
            event_id="detached",
        )
    filed = file_d1_verdict(
        adjudicated,
        actor=_actor(ActorRole.OPERATOR),
        filing=filing,
        event_id="verdict",
    )
    with pytest.raises(D1ProcessTransitionError) as error:
        filed.record_valid_negative(
            actor=_actor(ActorRole.OPERATOR),
            artifact=_artifact(121, "gate"),
            event_id="wrong-outcome",
        )

    assert error.value.code == "gate_outcome"
    assert filed.manager.repository.load().prefix == filed.manager.history.prefix


def test_filing_identity_must_match_every_prior_campaign_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filing = _filing(tmp_path, monkeypatch, "authorized")
    manager = _adjudicated(tmp_path, filing)
    wrong = replace(
        manager.events[2].artifacts[0],
        content_hash=_artifact(122, "wrong").content_hash,
    )
    tampered = D1LifecycleRepository(tmp_path / "tampered.jsonl")
    prefix: D1LifecyclePrefix | None = None
    for index, event in enumerate(manager.events):
        candidate = replace(event, artifacts=(wrong,)) if index == 2 else event
        prefix = tampered.append(candidate, expected_prefix=prefix).prefix
    hydrated = D1ProcessManager.load(tampered)
    assert isinstance(hydrated, D1ProcessManager)

    with pytest.raises(D1ProcessTransitionError) as error:
        file_d1_verdict(
            hydrated,
            actor=_actor(ActorRole.OPERATOR),
            filing=filing,
            event_id="verdict",
        )

    assert error.value.code == "filing_identity"
