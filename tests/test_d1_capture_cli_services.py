"""Public D1 capture commands over persisted campaign authority."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.d1_campaign_authority_loader import (
    D1LoadedCampaignAuthority,
)
from entropic_substrate.d1_campaign_request import D1CampaignCommandRequest
from entropic_substrate.d1_capture_cli_services import (
    D1CaptureFinalizeCliService,
    D1CapturePrepareCliService,
)
from entropic_substrate.d1_capture_finalize_coordinator import (
    D1PublishedCaptureFinalization,
)
from entropic_substrate.d1_capture_finalize_request import D1CaptureFinalizeRequest
from entropic_substrate.d1_capture_prepare_coordinator import (
    D1PublishedCapturePreparation,
)
from entropic_substrate.d1_cli import (
    D1CliCommand,
    D1CliInvalidEvidence,
    D1CliInvocation,
    D1CliOperatorGap,
)
from entropic_substrate.evidence.d1_capture_preparation_custody import (
    capture_preparation_custody_from_preparation,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash

from .test_d1_campaign_authority_loader import _loaded_prefix
from .test_d1_capture_finalize_coordinator import _prepared
from .test_d1_capture_preparation_custody import _authority, _case
from .test_d1_confirmatory_matrix import FakeGit

pytestmark = pytest.mark.unit


@dataclass
class _Loader:
    loaded: D1LoadedCampaignAuthority
    calls: list[D1CampaignCommandRequest] = field(default_factory=list)

    def load_persisted(
        self,
        request: D1CampaignCommandRequest,
        *,
        manifest_path: Path,
    ) -> D1LoadedCampaignAuthority:
        assert manifest_path.is_file()
        self.calls.append(request)
        return self.loaded


@dataclass
class _Coordinator:
    published: D1PublishedCapturePreparation
    calls: list[dict[str, object]] = field(default_factory=list)

    def prepare(self, **kwargs: object) -> D1PublishedCapturePreparation:
        self.calls.append(kwargs)
        return self.published


@dataclass
class _Factory:
    coordinator: _Coordinator
    calls: list[D1CampaignCommandRequest] = field(default_factory=list)

    def build(self, request: D1CampaignCommandRequest) -> _Coordinator:
        self.calls.append(request)
        return self.coordinator


@dataclass
class _FinalCoordinator:
    published: D1PublishedCaptureFinalization
    calls: list[dict[str, object]] = field(default_factory=list)

    def finalize(self, **kwargs: object) -> D1PublishedCaptureFinalization:
        self.calls.append(kwargs)
        return self.published


@dataclass
class _FinalFactory:
    coordinator: _FinalCoordinator
    calls: list[D1CampaignCommandRequest] = field(default_factory=list)

    def build(self, request: D1CampaignCommandRequest) -> _FinalCoordinator:
        self.calls.append(request)
        return self.coordinator


def _published(tmp_path: Path) -> D1PublishedCapturePreparation:
    preparation, _capture_root = _case(tmp_path / "capture")
    record = capture_preparation_custody_from_preparation(
        preparation,
        authority=_authority(),
    )
    artifact = ArtifactIdentity(
        str(tmp_path / "preparations" / f"{record.record_sha256}.json"),
        ContentHash("sha256", record.record_sha256),
    )
    return D1PublishedCapturePreparation(record, artifact)


def _invocation(tmp_path: Path, request: D1CampaignCommandRequest) -> D1CliInvocation:
    manifest = tmp_path / "capture-prepare.json"
    manifest.write_text("{}", encoding="utf-8")
    return D1CliInvocation(D1CliCommand.CAPTURE_PREPARE, manifest, request.manifest)


def _finalize_case(tmp_path: Path):  # type: ignore[no-untyped-def]
    _campaign, coordinator, prepared, preparation_root, repo_root, _events = _prepared(
        tmp_path / "real"
    )
    publication = coordinator.finalize(
        campaign=_campaign,
        preparation_artifact=prepared.artifact,
        repo_root=repo_root,
        preparation_custody_root=preparation_root,
    )
    loaded = _loaded_prefix(tmp_path / "authority", 4)
    loaded = replace(
        loaded,
        request=replace(
            loaded.request,
            locations=replace(
                loaded.request.locations,
                repo_root=repo_root,
                capture_preparation_custody_root=preparation_root,
            ),
        ),
    )
    request = D1CaptureFinalizeRequest(loaded.request, prepared.artifact)
    manifest = tmp_path / "capture-finalize.json"
    manifest.write_text("{}", encoding="utf-8")
    invocation = D1CliInvocation(D1CliCommand.CAPTURE_FINALIZE, manifest, request.manifest)
    return loaded, request, invocation, publication


def test_prepare_command_loads_complete_campaign_and_returns_commit_required_result(
    tmp_path: Path,
) -> None:
    loaded = _loaded_prefix(tmp_path / "authority", 4)
    loader = _Loader(loaded)
    coordinator = _Coordinator(_published(tmp_path))
    factory = _Factory(coordinator)
    service = D1CapturePrepareCliService(
        catalog=fixtures.profiles(),
        authority_loader=loader,
        git=FakeGit(),
        coordinator_factory=factory,
    )

    result = service.execute(_invocation(tmp_path, loaded.request))

    assert result["schema_version"] == "d1-capture-prepare-result-v1"
    assert result["status"] == "commit_required"
    assert result["required_next_command"] == "d1-capture-finalize"
    assert result["preparation_artifact"]["sha256"] == coordinator.published.record.record_sha256  # type: ignore[index]
    assert result["pointer"]["sha256"] == coordinator.published.record.pointer.pointer_sha256  # type: ignore[index]
    assert len(loader.calls) == len(factory.calls) == len(coordinator.calls) == 1
    assert set(coordinator.calls[0]["cell_artifacts"]) == {
        cell.logical_run_id for cell in loaded.checkpoint.completed_cells
    }


def test_prepare_command_refuses_partial_campaign_before_coordinator_build(
    tmp_path: Path,
) -> None:
    loaded = _loaded_prefix(tmp_path / "authority", 2)
    factory = _Factory(_Coordinator(_published(tmp_path)))
    service = D1CapturePrepareCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        coordinator_factory=factory,
    )

    with pytest.raises(D1CliOperatorGap, match="not complete"):
        service.execute(_invocation(tmp_path, loaded.request))

    assert factory.calls == []


def test_prepare_command_normalizes_wrong_command_and_bad_payload(
    tmp_path: Path,
) -> None:
    loaded = _loaded_prefix(tmp_path / "authority", 4)
    service = D1CapturePrepareCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        coordinator_factory=_Factory(_Coordinator(_published(tmp_path))),
    )
    invocation = _invocation(tmp_path, loaded.request)

    with pytest.raises(D1CliInvalidEvidence, match="wrong command"):
        service.execute(
            D1CliInvocation(D1CliCommand.CAPTURE_FINALIZE, invocation.manifest_path, {})
        )
    with pytest.raises(D1CliInvalidEvidence, match="request"):
        service.execute(D1CliInvocation(D1CliCommand.CAPTURE_PREPARE, invocation.manifest_path, {}))


def test_prepare_command_refuses_loader_authority_substitution_before_factory(
    tmp_path: Path,
) -> None:
    loaded = _loaded_prefix(tmp_path / "authority", 4)
    substituted = replace(
        loaded,
        request=replace(
            loaded.request,
            locations=replace(
                loaded.request.locations,
                artifact_root=tmp_path / "substituted-artifacts",
            ),
        ),
    )
    factory = _Factory(_Coordinator(_published(tmp_path)))
    service = D1CapturePrepareCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(substituted),
        git=FakeGit(),
        coordinator_factory=factory,
    )

    with pytest.raises(D1CliInvalidEvidence, match="substituted"):
        service.execute(_invocation(tmp_path, loaded.request))

    assert factory.calls == []


def test_finalize_command_reloads_authority_and_returns_captured_result(
    tmp_path: Path,
) -> None:
    loaded, _request, invocation, publication = _finalize_case(tmp_path)
    coordinator = _FinalCoordinator(publication)
    factory = _FinalFactory(coordinator)
    service = D1CaptureFinalizeCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        coordinator_factory=factory,
    )

    result = service.execute(invocation)

    assert result["schema_version"] == "d1-capture-finalize-result-v1"
    assert result["status"] == "captured"
    assert result["pointer_commit"] == "abcdef1"
    assert result["lifecycle"]["state"] == "captured"  # type: ignore[index]
    assert len(factory.calls) == len(coordinator.calls) == 1


def test_finalize_command_refuses_artifact_outside_declared_custody_root(
    tmp_path: Path,
) -> None:
    loaded, request, invocation, publication = _finalize_case(tmp_path)
    outside = replace(
        request,
        preparation_artifact=replace(
            request.preparation_artifact,
            path=str(
                tmp_path / "outside" / f"{request.preparation_artifact.content_hash.digest}.json"
            ),
        ),
    )
    invocation = D1CliInvocation(
        D1CliCommand.CAPTURE_FINALIZE,
        invocation.manifest_path,
        outside.manifest,
    )
    factory = _FinalFactory(_FinalCoordinator(publication))
    service = D1CaptureFinalizeCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(loaded),
        git=FakeGit(),
        coordinator_factory=factory,
    )

    with pytest.raises(D1CliInvalidEvidence, match="custody root"):
        service.execute(invocation)

    assert factory.calls == []


def test_finalize_command_refuses_partial_campaign_before_factory(tmp_path: Path) -> None:
    _loaded, request, invocation, publication = _finalize_case(tmp_path)
    partial = _loaded_prefix(tmp_path / "partial", 2)
    partial = replace(partial, request=request.campaign)
    factory = _FinalFactory(_FinalCoordinator(publication))
    service = D1CaptureFinalizeCliService(
        catalog=fixtures.profiles(),
        authority_loader=_Loader(partial),
        git=FakeGit(),
        coordinator_factory=factory,
    )

    with pytest.raises(D1CliOperatorGap, match="not complete"):
        service.execute(invocation)

    assert factory.calls == []
