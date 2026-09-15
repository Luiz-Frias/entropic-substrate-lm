"""TDD contract for completed-campaign heterogeneous DVC capture."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from entropic_substrate.d1_campaign_capture import (
    D1CampaignCaptureMaterial,
    D1CampaignCaptureService,
    D1CapturedCampaign,
    D1HeterogeneousCaptureAdapter,
    D1PreparedCampaignCapture,
)
from entropic_substrate.evidence.d1_capture_aggregate import D1TerminalDefectInput
from entropic_substrate.evidence.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureInput,
)
from entropic_substrate.evidence.d1_capture_roster import D1LaunchRoster
from entropic_substrate.evidence.d1_capture_service import D1HeterogeneousCaptureReceipt
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.lifecycle import ActorIdentity, ActorRole

from .test_d1_campaign_queue import D1CampaignQueue, FakeGit, _campaign, _Controller, _results
from .test_d1_operator_controller import _manager
from .test_d1_process_manager import _execute

pytestmark = pytest.mark.unit


def _completed(tmp_path: Path):
    design, intents, starts = _campaign()
    campaign = D1CampaignQueue(
        design=design,
        controller=_Controller(_results(intents, starts), []),
        git=FakeGit(),
    ).run(intents)
    manager = _manager(tmp_path)
    for index, intent in enumerate(intents):
        manager = _execute(manager, intent.plan.logical_run_id, 100 + index * 10)
    results = (*campaign.results[:-1], replace(campaign.results[-1], process_manager=manager))
    return replace(campaign, results=results, process_manager=manager), manager


@dataclass
class _Assembler:
    events: list[str]
    wrong_roster: D1LaunchRoster | None = None

    def prepare(self, campaign: object, roster: D1LaunchRoster) -> D1CampaignCaptureMaterial:
        del campaign
        self.events.append("prepare")
        selected = roster if self.wrong_roster is None else self.wrong_roster
        complete = tuple(cast(D1CaptureInput, object()) for _ in selected.expected_confirmatory)
        return D1CampaignCaptureMaterial(selected, complete, (), ())


@dataclass
class _Aggregate:
    roster: D1LaunchRoster


@dataclass
class _Receipt:
    aggregate: _Aggregate

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_ready": True,
            "pointer_commit": "abcdef1",
            "schema_version": "d1-heterogeneous-capture-receipt-v1",
        }


@dataclass
class _Preparation:
    aggregate: _Aggregate


@dataclass
class _Capture:
    events: list[str]
    receipt: _Receipt | None = None
    material: D1CampaignCaptureMaterial | None = None
    preparation: _Preparation | None = None

    def prepare(self, material: D1CampaignCaptureMaterial) -> _Preparation:
        self.events.append("capture_prepare")
        self.material = material
        return self.preparation or _Preparation(_Aggregate(material.roster))

    def finalize(self, preparation: _Preparation) -> D1HeterogeneousCaptureReceipt:
        self.events.append("capture_finalize")
        receipt = self.receipt or _Receipt(preparation.aggregate)
        return cast(D1HeterogeneousCaptureReceipt, receipt)


def _automation() -> ActorIdentity:
    return ActorIdentity("capture", "capture-session", ActorRole.AUTOMATION)


def test_capture_derives_roster_delegates_dvc_then_advances_lifecycle(tmp_path: Path) -> None:
    campaign, before = _completed(tmp_path)
    events: list[str] = []
    capture = _Capture(events)

    service = D1CampaignCaptureService(
        assembler=_Assembler(events),
        capture=capture,
        automation=_automation(),
    )
    prepared = service.prepare(campaign)

    assert type(prepared) is D1PreparedCampaignCapture
    assert events == ["prepare", "capture_prepare"]
    assert before.state is D1ProcessState.OPERATOR_FINALIZED
    assert prepared.process_manager is before
    result = service.finalize(prepared)

    assert type(result) is D1CapturedCampaign
    assert events == ["prepare", "capture_prepare", "capture_finalize"]
    assert capture.material is not None
    roster = capture.material.roster
    assert tuple(item.logical_run_id for item in roster.expected_confirmatory) == tuple(
        launch.plan.logical_run_id for launch in campaign.matrix.launches
    )
    assert (
        roster.preregistration_commit == campaign.matrix.launches[0].plan.run.preregistration.commit
    )
    assert result.process_manager.state is D1ProcessState.CAPTURED
    assert result.capture.aggregate.roster == roster


def test_wrong_assembled_or_returned_roster_never_records_captured_state(
    tmp_path: Path,
) -> None:
    campaign, manager = _completed(tmp_path)
    events: list[str] = []
    expected_service = D1CampaignCaptureService(
        assembler=_Assembler(events), capture=_Capture(events), automation=_automation()
    )
    expected = expected_service.expected_roster(campaign)
    wrong = replace(expected, preregistration_commit="f" * 40)

    with pytest.raises(ValueError, match="assembler roster"):
        D1CampaignCaptureService(
            assembler=_Assembler(events, wrong),
            capture=_Capture(events),
            automation=_automation(),
        ).capture(campaign)
    assert events == ["prepare"]
    assert manager.state is D1ProcessState.OPERATOR_FINALIZED

    events.clear()
    capture = _Capture(events, _Receipt(_Aggregate(wrong)))
    with pytest.raises(ValueError, match="receipt roster"):
        D1CampaignCaptureService(
            assembler=_Assembler(events), capture=capture, automation=_automation()
        ).capture(campaign)
    assert events == ["prepare", "capture_prepare", "capture_finalize"]
    assert manager.state is D1ProcessState.OPERATOR_FINALIZED


@pytest.mark.parametrize("mutation", ["campaign", "lifecycle"])
def test_finalize_rejects_campaign_or_lifecycle_identity_drift_before_capture(
    tmp_path: Path,
    mutation: str,
) -> None:
    campaign, manager = _completed(tmp_path)
    events: list[str] = []
    service = D1CampaignCaptureService(
        assembler=_Assembler(events), capture=_Capture(events), automation=_automation()
    )
    prepared = service.prepare(campaign)
    changed = (
        replace(prepared, campaign=replace(campaign, design_digest="0" * 64))
        if mutation == "campaign"
        else replace(prepared, lifecycle_prefix="0" * 64)
    )

    with pytest.raises(ValueError, match="identity"):
        service.finalize(changed)

    assert events == ["prepare", "capture_prepare"]
    assert manager.state is D1ProcessState.OPERATOR_FINALIZED


def test_capture_adapter_delegates_all_heterogeneous_lanes() -> None:
    calls: dict[str, object] = {}
    receipt = cast(D1HeterogeneousCaptureReceipt, object())
    preparation = object()

    class _Service:
        def prepare_heterogeneous(self, **kwargs: object) -> object:
            calls.update(kwargs)
            return preparation

        def finalize_heterogeneous(self, value: object) -> D1HeterogeneousCaptureReceipt:
            assert value is preparation
            return receipt

    design, _, _ = _campaign()
    roster = D1CampaignCaptureService.expected_roster_from_matrix(
        D1CampaignQueue(
            design=design,
            controller=_Controller(_results(*_campaign()[1:]), []),
            git=FakeGit(),
        ).run(_campaign()[1])
    )
    material = D1CampaignCaptureMaterial(
        roster,
        (cast(D1CaptureInput, object()),),
        (cast(D1TerminalDefectInput, object()),),
        (cast(D1AttemptEvidence, object()),),
    )

    adapter = D1HeterogeneousCaptureAdapter(cast(object, _Service()))
    assert adapter.finalize(adapter.prepare(material)) is receipt
    assert calls == {
        "roster": roster,
        "complete_inputs": material.complete_inputs,
        "terminal_defect_inputs": material.terminal_defect_inputs,
        "attempt_evidence": material.attempt_evidence,
    }
