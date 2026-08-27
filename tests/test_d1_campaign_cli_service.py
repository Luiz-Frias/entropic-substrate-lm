"""Public campaign-run command over fresh and persisted recovery typestates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.d1_campaign_authority_loader import D1LoadedCampaignAuthority
from entropic_substrate.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointRepository,
)
from entropic_substrate.d1_campaign_cli_service import D1CampaignRunCliService
from entropic_substrate.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1FreshCampaignCursor,
    D1PersistedCampaignCursor,
)
from entropic_substrate.d1_cli import (
    D1CliCommand,
    D1CliInvalidEvidence,
    D1CliInvocation,
    D1CliOperatorGap,
)
from entropic_substrate.experiments.d1_registration_codec import (
    parse_registered_design_declaration,
    serialize_registered_design,
)

from .test_d1_campaign_recovery_queue import _ready, _RecoveryController
from .test_d1_campaign_request import _request
from .test_d1_confirmatory_matrix import FakeGit

pytestmark = pytest.mark.unit


@dataclass
class _Loader:
    loaded: D1LoadedCampaignAuthority
    calls: list[str] = field(default_factory=list)

    def load_fresh(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority:
        assert type(request.cursor) is D1FreshCampaignCursor
        assert manifest_path.is_file()
        self.calls.append("fresh")
        return self.loaded

    def load_persisted(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority:
        assert type(request.cursor) is D1PersistedCampaignCursor
        assert manifest_path.is_file()
        self.calls.append("persisted")
        return self.loaded


@dataclass
class _Factory:
    controller: _RecoveryController
    calls: list[D1LoadedCampaignAuthority] = field(default_factory=list)

    def build(self, loaded: D1LoadedCampaignAuthority) -> _RecoveryController:
        self.calls.append(loaded)
        return self.controller


def _case(tmp_path: Path, count: int):  # type: ignore[no-untyped-def]
    design, intents, recovery, suffix = _ready(tmp_path / "recovery", count)
    request = _request(tmp_path / "request")
    checkpoint_path = tmp_path / "campaign" / "checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True)
    receipt = D1CampaignCheckpointRepository(checkpoint_path).save(
        recovery.checkpoint,
        expected_previous_sha256=None,
    )
    request = replace(
        request,
        design=parse_registered_design_declaration(serialize_registered_design(design)),
        intents=intents,
        cursor=D1PersistedCampaignCursor(receipt.sha256),
        locations=replace(request.locations, checkpoint_path=checkpoint_path),
    )
    loaded = D1LoadedCampaignAuthority(
        request,
        design,
        recovery.process_manager,
        recovery.checkpoint,
        receipt,
        recovery,
    )
    controller = _RecoveryController(
        suffix,
        [],
        recovery.process_manager,
        recovery.attempt_ledger,
    )
    manifest = tmp_path / "campaign-run.json"
    manifest.write_text("{}", encoding="utf-8")
    return loaded, controller, manifest


def _invocation(
    manifest: Path,
    request: D1CampaignCommandRequest,
) -> D1CliInvocation:
    return D1CliInvocation(D1CliCommand.CAMPAIGN_RUN, manifest, request.manifest)


def test_persisted_campaign_executes_only_remaining_suffix_and_returns_complete(
    tmp_path: Path,
) -> None:
    loaded, controller, manifest = _case(tmp_path, 2)
    loader = _Loader(loaded)
    factory = _Factory(controller)
    service = D1CampaignRunCliService(
        catalog=fixtures.profiles(),
        authority_loader=loader,
        git=FakeGit(),
        controller_factory=factory,
    )

    result = service.execute(_invocation(manifest, loaded.request))

    assert result["schema_version"] == "d1-campaign-run-result-v1"
    assert result["status"] == "campaign_complete"
    assert result["completed_cell_count"] == 4
    assert result["recovered_prefix_count"] == 2
    assert result["required_next_command"] == "d1-capture-prepare"
    assert loader.calls == ["persisted"]
    assert controller.calls == [intent.plan.logical_run_id for intent in loaded.request.intents[2:]]
    assert controller.closed is True


def test_fresh_campaign_uses_fresh_loader_then_executes_all_cells(tmp_path: Path) -> None:
    loaded, controller, manifest = _case(tmp_path, 0)
    request = replace(loaded.request, cursor=D1FreshCampaignCursor())
    loader = _Loader(loaded)
    service = D1CampaignRunCliService(
        catalog=fixtures.profiles(),
        authority_loader=loader,
        git=FakeGit(),
        controller_factory=_Factory(controller),
    )

    result = service.execute(_invocation(manifest, request))

    assert result["status"] == "campaign_complete"
    assert result["recovered_prefix_count"] == 0
    assert loader.calls == ["fresh"]
    assert len(controller.calls) == 4
    assert controller.closed is True


def test_already_complete_campaign_never_builds_controller(tmp_path: Path) -> None:
    loaded, controller, manifest = _case(tmp_path, 4)
    factory = _Factory(controller)
    service = D1CampaignRunCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        controller_factory=factory,
    )

    result = service.execute(_invocation(manifest, loaded.request))

    assert result["status"] == "campaign_complete"
    assert result["recovered_prefix_count"] == 4
    assert factory.calls == []
    assert controller.calls == []


def test_campaign_run_closes_controller_after_queue_failure(tmp_path: Path) -> None:
    loaded, controller, manifest = _case(tmp_path, 0)
    controller.results = ()
    service = D1CampaignRunCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        controller_factory=_Factory(controller),
    )

    with pytest.raises(D1CliOperatorGap):
        service.execute(_invocation(manifest, loaded.request))

    assert controller.closed is True


def test_persisted_loader_may_promote_only_checkpoint_cursor_after_adoption(
    tmp_path: Path,
) -> None:
    loaded, controller, manifest = _case(tmp_path, 4)
    stale = replace(loaded.request, cursor=D1PersistedCampaignCursor("0" * 64))
    service = D1CampaignRunCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        controller_factory=_Factory(controller),
    )

    result = service.execute(_invocation(manifest, stale))

    assert result["status"] == "campaign_complete"
    assert result["checkpoint_generation"]["sha256"] == loaded.checkpoint_receipt.sha256  # type: ignore[index]
    assert controller.calls == []


def test_campaign_run_rejects_wrong_command_and_bad_payload(tmp_path: Path) -> None:
    loaded, controller, manifest = _case(tmp_path, 0)
    service = D1CampaignRunCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        controller_factory=_Factory(controller),
    )

    with pytest.raises(D1CliInvalidEvidence, match="wrong command"):
        service.execute(D1CliInvocation(D1CliCommand.INSPECT, manifest, {}))
    with pytest.raises(D1CliInvalidEvidence, match="request"):
        service.execute(D1CliInvocation(D1CliCommand.CAMPAIGN_RUN, manifest, {}))
