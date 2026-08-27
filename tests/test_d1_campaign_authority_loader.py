"""Durable campaign authorities rehydrate before execution or capture."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.d1_campaign_authority_loader import (
    D1CampaignAuthorityLoader,
    D1CampaignAuthorityLoaderError,
)
from entropic_substrate.d1_campaign_bootstrap import bootstrap_d1_ready_checkpoint
from entropic_substrate.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointRepository,
)
from entropic_substrate.d1_campaign_recovery import D1CampaignRecoveryStatus
from entropic_substrate.d1_campaign_request import (
    D1CampaignCommandRequest,
    D1CampaignLocations,
    D1FreshCampaignCursor,
    D1PersistedCampaignCursor,
)
from entropic_substrate.experiments.d1_launch_attempts import D1LaunchAttemptLedger
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)
from entropic_substrate.experiments.d1_registration_codec import (
    parse_registered_design_declaration,
    serialize_registered_design,
)

from .test_d1_campaign_queue import _intent
from .test_d1_operator_controller import _manager

pytestmark = pytest.mark.unit

_PREREGISTRATION = b"# D1 preregistration\n\nCampaign authority loader bytes.\n"


def _case(tmp_path: Path):  # type: ignore[no-untyped-def]
    contract = fixtures.contract(fixtures.family(), seeds=(101,))
    design = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            fixtures.family(),
            contract,
            (101,),
            D1PreregistrationArtifact(
                fixtures.PREREG_PATH,
                hashlib.sha256(_PREREGISTRATION).hexdigest(),
                fixtures.PREREG_COMMIT,
            ),
        )
    )
    assert type(design) is D1RegisteredDesign
    intents = tuple(_intent(design, profile)[0] for profile in fixtures.profiles())
    repo_root = tmp_path / "repo"
    preregistration = repo_root / fixtures.PREREG_PATH
    preregistration.parent.mkdir(parents=True)
    preregistration.write_bytes(_PREREGISTRATION)
    artifact_root = tmp_path / "artifacts"
    cell_root = tmp_path / "custody" / "cells"
    capture_root = tmp_path / "custody" / "capture"
    campaign_root = tmp_path / "campaign"
    for directory in (artifact_root, cell_root, capture_root, campaign_root):
        directory.mkdir(parents=True)
    manager = _manager(campaign_root)
    checkpoint_path = campaign_root / "checkpoint.json"
    checkpoint, receipt = bootstrap_d1_ready_checkpoint(
        design=design,
        process_manager=manager,
        attempt_ledger=D1LaunchAttemptLedger(),
        repository=D1CampaignCheckpointRepository(checkpoint_path),
    )
    declaration = parse_registered_design_declaration(serialize_registered_design(design))
    request = D1CampaignCommandRequest(
        declaration,
        contract,
        intents,
        D1PersistedCampaignCursor(receipt.sha256),
        D1CampaignLocations(
            repo_root,
            artifact_root,
            campaign_root / "lifecycle.jsonl",
            checkpoint_path,
            cell_root,
            capture_root,
        ),
    )
    manifest = tmp_path / "request.json"
    manifest.write_text("{}", encoding="utf-8")
    return request, manifest, design, manager, checkpoint, receipt


def test_loader_rehydrates_design_lifecycle_checkpoint_and_recovery(
    tmp_path: Path,
) -> None:
    request, manifest, design, manager, checkpoint, receipt = _case(tmp_path)

    loaded = D1CampaignAuthorityLoader(catalog=fixtures.profiles()).load_persisted(
        request,
        manifest_path=manifest,
    )

    assert loaded.design == design
    assert loaded.process_manager.history == manager.history
    assert loaded.process_manager.repository.path == manager.repository.path
    assert loaded.checkpoint == checkpoint
    assert loaded.checkpoint_receipt == receipt
    assert loaded.recovery.status is D1CampaignRecoveryStatus.READY
    assert loaded.request.locations.repo_root.is_absolute()


def test_loader_refuses_fresh_cursor_without_reading_checkpoint(tmp_path: Path) -> None:
    request, manifest, *_ = _case(tmp_path)
    fresh = replace(request, cursor=D1FreshCampaignCursor())

    with pytest.raises(D1CampaignAuthorityLoaderError, match="persisted"):
        D1CampaignAuthorityLoader(catalog=fixtures.profiles()).load_persisted(
            fresh,
            manifest_path=manifest,
        )


@pytest.mark.parametrize("mutation", ["preregistration", "checkpoint"])
def test_loader_refuses_substituted_preregistration_or_checkpoint_generation(
    tmp_path: Path,
    mutation: str,
) -> None:
    request, manifest, *_ = _case(tmp_path)
    if mutation == "preregistration":
        (request.locations.repo_root / fixtures.PREREG_PATH).write_bytes(b"substituted\n")
    else:
        request = replace(request, cursor=D1PersistedCampaignCursor("0" * 64))

    with pytest.raises(D1CampaignAuthorityLoaderError):
        D1CampaignAuthorityLoader(catalog=fixtures.profiles()).load_persisted(
            request,
            manifest_path=manifest,
        )


def test_loader_refuses_symlinked_preregistration_path(tmp_path: Path) -> None:
    request, manifest, *_ = _case(tmp_path)
    target = request.locations.repo_root / "real-prereg.md"
    target.write_bytes(_PREREGISTRATION)
    preregistration = request.locations.repo_root / fixtures.PREREG_PATH
    preregistration.unlink()
    preregistration.symlink_to(target)

    with pytest.raises(D1CampaignAuthorityLoaderError, match="preregistration"):
        D1CampaignAuthorityLoader(catalog=fixtures.profiles()).load_persisted(
            request,
            manifest_path=manifest,
        )
