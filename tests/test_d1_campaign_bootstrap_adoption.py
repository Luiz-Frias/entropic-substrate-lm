"""Campaign bootstrap and one-cell crash adoption never relaunch provider work."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, cast

import pytest

from entropic_substrate.d1_campaign_bootstrap import (
    D1CampaignBootstrapError,
    adopt_d1_finalized_cell_after_crash,
    bootstrap_d1_ready_checkpoint,
)
from entropic_substrate.d1_campaign_checkpoint import D1CampaignCheckpoint
from entropic_substrate.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointReceipt,
    D1CampaignCheckpointRepository,
)
from entropic_substrate.d1_operator_controller import D1SequentialOperatorController
from entropic_substrate.evidence.lifecycle import ContentHash
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
)
from entropic_substrate.experiments.d1_primitives import RunClass

from .test_d1_operator_controller import _Case, _case, _controller

pytestmark = pytest.mark.unit


class _BootstrapCase(NamedTuple):
    case: _Case
    controller: D1SequentialOperatorController
    events: list[str]
    repository: D1CampaignCheckpointRepository
    checkpoint: D1CampaignCheckpoint
    receipt: D1CampaignCheckpointReceipt


def _bootstrapped(tmp_path: Path) -> _BootstrapCase:
    case_root = tmp_path / "case"
    operator_root = tmp_path / "operator"
    case_root.mkdir()
    operator_root.mkdir()
    case = _case(case_root)
    events: list[str] = []
    controller, _ = _controller(operator_root, case, events)
    target = tmp_path / "campaign.json"
    repository = D1CampaignCheckpointRepository(target)
    checkpoint, receipt = bootstrap_d1_ready_checkpoint(
        design=case.design,
        process_manager=controller.process_manager,
        attempt_ledger=controller.attempt_ledger,
        repository=repository,
    )
    return _BootstrapCase(case, controller, events, repository, checkpoint, receipt)


def test_bootstrap_saves_empty_ready_checkpoint_from_exact_preregistered_authority(
    tmp_path: Path,
) -> None:
    _case_value, controller, events, repository, checkpoint, receipt = _bootstrapped(tmp_path)

    assert checkpoint.next_index == 0
    assert checkpoint.completed_cells == ()
    assert checkpoint.attempt_ledger == D1LaunchAttemptLedger()
    assert checkpoint.process_state is controller.process_manager.state
    assert repository.load(expected_sha256=receipt.sha256)[0] == checkpoint
    assert events == []


def test_rejected_bootstrap_saves_no_checkpoint_generation(tmp_path: Path) -> None:
    case_root, operator_root = tmp_path / "case", tmp_path / "operator"
    case_root.mkdir()
    operator_root.mkdir()
    case = _case(case_root)
    controller, _ = _controller(operator_root, case, [])
    identity = D1LaunchAttemptIdentity(
        RunClass.CONFIRMATORY,
        case.intent.plan.profile.family.family_digest,
        case.intent.plan.profile.arm,
        case.intent.plan.seed,
        case.intent.plan.logical_run_id,
        case.design.design_digest,
    )
    nonempty = D1LaunchAttemptLedger().record(
        identity,
        observed_at=cast(RunStartProvenance, case.start).trainer_started_at,
    )
    target = tmp_path / "rejected.json"

    with pytest.raises(D1CampaignBootstrapError, match="empty ledger"):
        bootstrap_d1_ready_checkpoint(
            design=case.design,
            process_manager=controller.process_manager,
            attempt_ledger=nonempty,
            repository=D1CampaignCheckpointRepository(target),
        )

    assert not target.exists()


def test_one_finalized_cell_is_adopted_and_saved_without_provider_relaunch(
    tmp_path: Path,
) -> None:
    case, controller, events, repository, checkpoint, receipt = _bootstrapped(tmp_path)
    result = controller.execute_fresh(case.intent)
    assert result.finalized_cell is not None
    before_adoption = tuple(events)

    adopted, generation = adopt_d1_finalized_cell_after_crash(
        design=case.design,
        intent=case.intent,
        checkpoint=checkpoint,
        process_manager=result.process_manager,
        candidates=((result.finalized_cell, result.finalized_cell_artifact),),
        repository=repository,
        generation=receipt,
    )

    assert tuple(events) == before_adoption
    assert adopted.next_index == 1
    assert adopted.attempt_ledger == result.attempt_ledger
    assert adopted.lifecycle_prefix == result.process_manager.history.prefix.prefix_sha256
    assert adopted.completed_cells[0].start == result.finalized_cell.segment_starts[0]
    assert generation.sha256 != receipt.sha256
    assert repository.load(expected_sha256=generation.sha256)[0] == adopted
    with pytest.raises(D1CampaignBootstrapError, match="ahead"):
        adopt_d1_finalized_cell_after_crash(
            design=case.design,
            intent=case.intent,
            checkpoint=adopted,
            process_manager=result.process_manager,
            candidates=((result.finalized_cell, result.finalized_cell_artifact),),
            repository=repository,
            generation=generation,
        )


@pytest.mark.parametrize("candidate_count", [0, 2])
def test_adoption_requires_exactly_one_candidate_before_any_checkpoint_save(
    tmp_path: Path,
    candidate_count: int,
) -> None:
    case, controller, _events, repository, checkpoint, receipt = _bootstrapped(tmp_path)
    result = controller.execute_fresh(case.intent)
    assert result.finalized_cell is not None
    candidate = (result.finalized_cell, result.finalized_cell_artifact)

    with pytest.raises(D1CampaignBootstrapError, match="exactly one"):
        adopt_d1_finalized_cell_after_crash(
            design=case.design,
            intent=case.intent,
            checkpoint=checkpoint,
            process_manager=result.process_manager,
            candidates=tuple(candidate for _ in range(candidate_count)),
            repository=repository,
            generation=receipt,
        )

    assert repository.load(expected_sha256=receipt.sha256)[0] == checkpoint


def test_zero_ahead_lifecycle_and_substituted_artifact_are_refused(tmp_path: Path) -> None:
    case, controller, _events, repository, checkpoint, receipt = _bootstrapped(tmp_path)
    initial_manager = controller.process_manager
    result = controller.execute_fresh(case.intent)
    assert result.finalized_cell is not None
    candidate = (result.finalized_cell, result.finalized_cell_artifact)

    with pytest.raises(D1CampaignBootstrapError, match="ahead"):
        adopt_d1_finalized_cell_after_crash(
            design=case.design,
            intent=case.intent,
            checkpoint=checkpoint,
            process_manager=initial_manager,
            candidates=(candidate,),
            repository=repository,
            generation=receipt,
        )

    foreign_design = replace(case.design, decision_contract_digest="1" * 64)
    with pytest.raises(D1CampaignBootstrapError, match="registration"):
        adopt_d1_finalized_cell_after_crash(
            design=foreign_design,
            intent=case.intent,
            checkpoint=checkpoint,
            process_manager=result.process_manager,
            candidates=(candidate,),
            repository=repository,
            generation=receipt,
        )

    substituted = replace(
        result.finalized_cell_artifact,
        content_hash=ContentHash("sha256", "0" * 64),
    )
    with pytest.raises(D1CampaignBootstrapError, match="artifact"):
        adopt_d1_finalized_cell_after_crash(
            design=case.design,
            intent=case.intent,
            checkpoint=checkpoint,
            process_manager=result.process_manager,
            candidates=((result.finalized_cell, substituted),),
            repository=repository,
            generation=receipt,
        )
    assert repository.load(expected_sha256=receipt.sha256)[0] == checkpoint
