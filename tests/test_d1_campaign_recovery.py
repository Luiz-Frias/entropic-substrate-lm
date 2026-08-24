"""TDD contract for authority-preserving D1 campaign restart hydration."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import cast

import pytest

from entropic_substrate.d1_campaign_checkpoint import (
    D1CampaignCheckpoint,
    D1CampaignCompletedCell,
    D1CampaignDisposition,
)
from entropic_substrate.d1_campaign_recovery import (
    D1CampaignRecoveryError,
    D1CampaignRecoveryHalt,
    D1CampaignRecoveryStatus,
    D1HydratedCampaignRecovery,
    hydrate_d1_campaign_recovery,
)
from entropic_substrate.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.evidence.lifecycle import ActorRole
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.experiments.d1_primitives import RunClass, canonical_digest

from .test_d1_campaign_queue import _campaign
from .test_d1_operator_controller import _manager
from .test_d1_process_manager import _actor, _artifact, _execute

pytestmark = pytest.mark.unit


def _attempt(intent: D1OperatorLaunchIntent) -> D1LaunchAttemptIdentity:
    plan = intent.plan
    return D1LaunchAttemptIdentity(
        RunClass.CONFIRMATORY,
        plan.profile.family.family_digest,
        plan.profile.arm,
        plan.seed,
        plan.logical_run_id,
        intent.registered_design_digest,
    )


def _prefix(tmp_path, count: int):  # type: ignore[no-untyped-def]
    design, intents, raw_starts = _campaign()
    starts = cast(tuple[RunStartProvenance, ...], raw_starts)
    manager = _manager(tmp_path)
    ledger = D1LaunchAttemptLedger()
    cells: list[D1CampaignCompletedCell] = []
    for index, (intent, start) in enumerate(zip(intents[:count], starts[:count], strict=True)):
        identity = _attempt(intent)
        observed = start.trainer_started_at
        ledger = ledger.record(identity, observed_at=observed - timedelta(seconds=1))
        ledger = ledger.transition(
            identity,
            status=D1LaunchAttemptStatus.TRAINER_STARTED,
            observed_at=observed,
        )
        ledger = ledger.transition(
            identity,
            status=D1LaunchAttemptStatus.COMPLETED,
            observed_at=observed + timedelta(seconds=1),
        )
        manager = _execute(manager, intent.plan.logical_run_id, 100 + index * 10)
        cells.append(
            D1CampaignCompletedCell(
                index,
                intent.plan.logical_run_id,
                intent.plan.profile.arm,
                intent.plan.seed,
                intent.plan.plan_digest,
                canonical_digest(intent.manifest),
                start,
                ledger.ledger_digest,
                manager.history.prefix.prefix_sha256,
                len(manager.events),
            )
        )
    checkpoint = D1CampaignCheckpoint(
        design.design_digest,
        design.decision_contract_digest,
        tuple(cells),
        ledger,
        manager.state,
        manager.history.prefix.prefix_sha256,
        len(manager.events),
        manager.active_subject,
        D1CampaignDisposition.READY,
        None,
    )
    return design, intents, starts, manager, ledger, checkpoint


def test_fresh_checkpoint_preserves_normal_first_cell_behavior(tmp_path) -> None:  # type: ignore[no-untyped-def]
    design, intents, _starts, manager, _ledger, _ = _prefix(tmp_path, 0)
    checkpoint = D1CampaignCheckpoint(
        design.design_digest,
        design.decision_contract_digest,
        (),
        D1LaunchAttemptLedger(),
        manager.state,
        manager.history.prefix.prefix_sha256,
        len(manager.events),
        None,
        D1CampaignDisposition.READY,
        None,
    )

    recovered = hydrate_d1_campaign_recovery(
        design=design, intents=intents, checkpoint=checkpoint, process_manager=manager
    )

    assert type(recovered) is D1HydratedCampaignRecovery
    assert recovered.status is D1CampaignRecoveryStatus.READY
    assert recovered.next_index == 0
    assert recovered.remaining_intents == intents
    assert recovered.attempt_ledger.events == ()


def test_completed_prefix_is_skipped_and_ledger_is_carried_without_events(tmp_path) -> None:  # type: ignore[no-untyped-def]
    design, intents, _starts, manager, ledger, checkpoint = _prefix(tmp_path, 2)
    before = ledger.events

    recovered = hydrate_d1_campaign_recovery(
        design=design, intents=intents, checkpoint=checkpoint, process_manager=manager
    )

    assert type(recovered) is D1HydratedCampaignRecovery
    assert recovered.status is D1CampaignRecoveryStatus.READY
    assert recovered.next_index == 2
    assert recovered.remaining_intents == intents[2:]
    assert tuple(item.plan.logical_run_id for item in recovered.completed_launches) == tuple(
        intent.plan.logical_run_id for intent in intents[:2]
    )
    assert recovered.attempt_ledger.events == before
    assert recovered.attempt_ledger.ledger_digest == ledger.ledger_digest


def test_all_cells_complete_returns_complete_without_a_launchable_suffix(tmp_path) -> None:  # type: ignore[no-untyped-def]
    design, intents, _starts, manager, _ledger, checkpoint = _prefix(tmp_path, 4)

    recovered = hydrate_d1_campaign_recovery(
        design=design, intents=intents, checkpoint=checkpoint, process_manager=manager
    )

    assert type(recovered) is D1HydratedCampaignRecovery
    assert recovered.status is D1CampaignRecoveryStatus.COMPLETE
    assert recovered.next_index == len(intents)
    assert recovered.remaining_intents == ()


@pytest.mark.parametrize("drift", ["design", "decision"])
def test_registration_identity_drift_is_a_tamper_error(tmp_path, drift: str) -> None:  # type: ignore[no-untyped-def]
    design, intents, _starts, manager, _ledger, checkpoint = _prefix(tmp_path, 1)
    active = (
        replace(
            design,
            preregistration=replace(design.preregistration, commit="f" * 40),
        )
        if drift == "design"
        else replace(design, decision_contract_digest="0" * 64)
    )

    with pytest.raises(D1CampaignRecoveryError, match="registration"):
        hydrate_d1_campaign_recovery(
            design=active,
            intents=intents,
            checkpoint=checkpoint,
            process_manager=manager,
        )


@pytest.mark.parametrize("mutation", ["plan", "intent", "start"])
def test_completed_cell_must_match_exact_registered_intent_prefix(
    tmp_path,
    mutation: str,  # type: ignore[no-untyped-def]
) -> None:
    design, intents, _starts, manager, _ledger, checkpoint = _prefix(tmp_path, 1)
    cell = checkpoint.completed_cells[0]
    changed = {
        "plan": replace(cell, plan_digest="0" * 64),
        "intent": replace(cell, intent_digest="0" * 64),
        "start": replace(
            cell,
            start=replace(cell.start, provider_sku="different-provider-sku"),
        ),
    }[mutation]

    with pytest.raises(D1CampaignRecoveryError, match=r"completed.*prefix"):
        hydrate_d1_campaign_recovery(
            design=design,
            intents=intents,
            checkpoint=replace(checkpoint, completed_cells=(changed,)),
            process_manager=manager,
        )


def test_lifecycle_prefix_or_finalized_subject_fork_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    design, intents, _starts, _manager_state, _ledger, checkpoint = _prefix(tmp_path, 1)
    unrelated = _manager(tmp_path / "other")
    with pytest.raises(D1CampaignRecoveryError, match="lifecycle"):
        hydrate_d1_campaign_recovery(
            design=design,
            intents=intents,
            checkpoint=checkpoint,
            process_manager=unrelated,
        )


@pytest.mark.parametrize("mutation", ["prefix", "count"])
def test_earlier_completed_cell_requires_exact_operator_finalized_prefix(
    tmp_path,
    mutation: str,  # type: ignore[no-untyped-def]
) -> None:
    design, intents, _starts, manager, _ledger, checkpoint = _prefix(tmp_path, 2)
    first = checkpoint.completed_cells[0]
    forged = (
        replace(first, lifecycle_prefix_after="0" * 64)
        if mutation == "prefix"
        else replace(
            first,
            lifecycle_event_count_after=first.lifecycle_event_count_after + 1,
        )
    )
    tampered = replace(
        checkpoint,
        completed_cells=(forged, checkpoint.completed_cells[1]),
    )

    with pytest.raises(D1CampaignRecoveryError, match="operator-finalized lifecycle prefix"):
        hydrate_d1_campaign_recovery(
            design=design,
            intents=intents,
            checkpoint=tampered,
            process_manager=manager,
        )


def test_hard_halt_with_advanced_inflight_lifecycle_returns_typed_halt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    design, intents, starts, manager, ledger, checkpoint = _prefix(tmp_path, 1)
    next_intent, next_start = intents[1], starts[1]
    pending = _attempt(next_intent)
    ledger = ledger.record(pending, observed_at=next_start.trainer_started_at)
    ledger = ledger.transition(
        pending,
        status=D1LaunchAttemptStatus.TRAINER_STARTED,
        observed_at=next_start.trainer_started_at + timedelta(seconds=1),
    )
    manager = manager.admit_launch(
        subject_id=next_intent.plan.logical_run_id,
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(500, "pending-admission"),
        event_id="pending-admission",
    )
    manager = manager.start_running(
        subject_id=next_intent.plan.logical_run_id,
        actor=_actor(ActorRole.AUTOMATION),
        artifact=_artifact(501, "pending-running"),
        event_id="pending-running",
    )
    halted_checkpoint = replace(
        checkpoint,
        attempt_ledger=ledger,
        process_state=manager.state,
        lifecycle_prefix=manager.history.prefix.prefix_sha256,
        lifecycle_event_count=len(manager.events),
        active_subject=manager.active_subject,
        disposition=D1CampaignDisposition.HARD_HALT,
        halt_reason="provider state is ambiguous after process death",
    )

    halted = hydrate_d1_campaign_recovery(
        design=design,
        intents=intents,
        checkpoint=halted_checkpoint,
        process_manager=manager,
    )

    assert type(halted) is D1CampaignRecoveryHalt
    assert halted.active_subject == next_intent.plan.logical_run_id
    assert "ambiguous" in halted.reason
    assert halted.attempt_ledger.events == ledger.events


def test_ready_checkpoint_observing_inflight_manager_returns_halt_not_guess(tmp_path) -> None:  # type: ignore[no-untyped-def]
    design, intents, _starts, manager, _ledger, checkpoint = _prefix(tmp_path, 1)
    manager = manager.admit_launch(
        subject_id=intents[1].plan.logical_run_id,
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(600, "uncheckpointed-admission"),
        event_id="uncheckpointed-admission",
    )

    halted = hydrate_d1_campaign_recovery(
        design=design,
        intents=intents,
        checkpoint=checkpoint,
        process_manager=manager,
    )

    assert type(halted) is D1CampaignRecoveryHalt
    assert halted.active_subject == intents[1].plan.logical_run_id
    assert "in-flight" in halted.reason


def test_recovery_values_are_factory_attested(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _design, intents, _starts, manager, _ledger, checkpoint = _prefix(tmp_path, 0)
    with pytest.raises(TypeError, match="factory-only"):
        D1HydratedCampaignRecovery(D1CampaignRecoveryStatus.READY, checkpoint, manager, (), intents)
    with pytest.raises(TypeError, match="factory-only"):
        D1CampaignRecoveryHalt("forged", checkpoint, manager, None)
