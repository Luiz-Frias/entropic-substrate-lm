"""TDD contract for canonical durable D1 campaign restart state."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import D1ProcessState
from entropic_substrate.ladder.register.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_checkpoint import (
    D1CampaignCheckpoint,
    D1CampaignCheckpointError,
    D1CampaignCompletedCell,
    D1CampaignDisposition,
    campaign_checkpoint_from_mapping,
    campaign_checkpoint_to_mapping,
    parse_campaign_checkpoint,
    serialize_campaign_checkpoint,
)

pytestmark = pytest.mark.unit


def _identity(logical_run_id: str, seed: int = 101) -> D1LaunchAttemptIdentity:
    return D1LaunchAttemptIdentity(
        RunClass.CONFIRMATORY,
        fixtures.family().family_digest,
        D1Arm.NONE,
        seed,
        logical_run_id,
        "d" * 64,
    )


def _complete(
    ledger: D1LaunchAttemptLedger,
    identity: D1LaunchAttemptIdentity,
    *,
    offset: int,
) -> D1LaunchAttemptLedger:
    observed = fixtures.STARTED_AT + timedelta(seconds=offset)
    ledger = ledger.record(identity, observed_at=observed)
    ledger = ledger.transition(
        identity,
        status=D1LaunchAttemptStatus.TRAINER_STARTED,
        observed_at=observed + timedelta(seconds=1),
    )
    return ledger.transition(
        identity,
        status=D1LaunchAttemptStatus.COMPLETED,
        observed_at=observed + timedelta(seconds=2),
    )


def _ready() -> D1CampaignCheckpoint:
    run = fixtures.plan(fixtures.profiles()[0], 101)
    start = fixtures.start(run)
    ledger = _complete(D1LaunchAttemptLedger(), _identity(run.logical_run_id), offset=0)
    cell = D1CampaignCompletedCell(
        index=0,
        logical_run_id=run.logical_run_id,
        arm=D1Arm.NONE,
        seed=101,
        plan_digest=run.plan_digest,
        intent_digest="a" * 64,
        start=start,
        ledger_digest_after=ledger.ledger_digest,
        lifecycle_prefix_after="b" * 64,
        lifecycle_event_count_after=8,
        terminal_outcome="completed",
        finalized_cell_artifact=ArtifactIdentity(
            "custody/" + "c" * 64 + ".json",
            ContentHash("sha256", "c" * 64),
        ),
    )
    return D1CampaignCheckpoint(
        design_digest="d" * 64,
        decision_contract_digest="e" * 64,
        completed_cells=(cell,),
        attempt_ledger=ledger,
        process_state=D1ProcessState.OPERATOR_FINALIZED,
        lifecycle_prefix="b" * 64,
        lifecycle_event_count=8,
        active_subject=None,
        disposition=D1CampaignDisposition.READY,
        halt_reason=None,
    )


def test_ready_checkpoint_round_trip_preserves_exact_next_action_and_attempt_identity() -> None:
    checkpoint = _ready()
    payload = serialize_campaign_checkpoint(checkpoint)
    restored = parse_campaign_checkpoint(payload)

    assert restored == checkpoint
    assert restored.next_index == 1
    assert restored.next_action == "launch_registered_cell_1"
    assert restored.attempt_ledger.ledger_digest == checkpoint.attempt_ledger.ledger_digest
    assert (
        restored.attempt_ledger.events[0].identity.optional_stopping_digest
        == checkpoint.attempt_ledger.events[0].identity.optional_stopping_digest
    )
    assert (
        campaign_checkpoint_from_mapping(campaign_checkpoint_to_mapping(checkpoint)) == checkpoint
    )
    assert restored.manifest["schema_version"] == "d1-campaign-checkpoint-v2"


def test_checkpoint_round_trip_preserves_registered_terminal_defect_outcome() -> None:
    checkpoint = _ready()
    cell = checkpoint.completed_cells[0]
    failed_event = replace(
        checkpoint.attempt_ledger.events[-1],
        status=D1LaunchAttemptStatus.FAILED,
        reason="terminal numerical defect",
    )
    failed_ledger = D1LaunchAttemptLedger((*checkpoint.attempt_ledger.events[:-1], failed_event))
    failed = replace(
        checkpoint,
        completed_cells=(
            replace(
                cell,
                terminal_outcome="numerical_defect",
                ledger_digest_after=failed_ledger.ledger_digest,
            ),
        ),
        attempt_ledger=failed_ledger,
    )

    restored = parse_campaign_checkpoint(serialize_campaign_checkpoint(failed))

    assert restored.completed_cells[0].terminal_outcome == "numerical_defect"


def test_empty_ready_checkpoint_authorizes_only_first_registered_cell() -> None:
    checkpoint = D1CampaignCheckpoint(
        "d" * 64,
        "e" * 64,
        (),
        D1LaunchAttemptLedger(),
        D1ProcessState.PREREGISTERED,
        "f" * 64,
        3,
        None,
        D1CampaignDisposition.READY,
        None,
    )

    assert checkpoint.next_index == 0
    assert checkpoint.next_action == "launch_registered_cell_0"


def test_nonterminal_provider_state_requires_typed_hard_halt_not_ready_guess() -> None:
    base = _ready()
    pending = _identity("d1-confirmatory-none-s103", seed=103)
    ledger = base.attempt_ledger.record(
        pending, observed_at=fixtures.STARTED_AT + timedelta(minutes=1)
    )
    ledger = ledger.transition(
        pending,
        status=D1LaunchAttemptStatus.TRAINER_STARTED,
        observed_at=fixtures.STARTED_AT + timedelta(minutes=1, seconds=1),
    )

    with pytest.raises(ValueError, match=r"READY.*terminal completed prefix"):
        replace(base, attempt_ledger=ledger)

    halted = replace(
        base,
        attempt_ledger=ledger,
        process_state=D1ProcessState.RUNNING,
        active_subject=pending.logical_run_id,
        disposition=D1CampaignDisposition.HARD_HALT,
        halt_reason="provider pod state was ambiguous after controller crash",
    )
    assert halted.next_action == "hard_halt"
    assert halted.active_subject == pending.logical_run_id


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"next_index": 99}),
        lambda value: value["completed_cells"][0].update({"index": 2}),
        lambda value: value["completed_cells"][0].update({"ledger_digest_after": "0" * 64}),
        lambda value: value["attempt_ledger"]["events"][0]["identity"].update(
            {"optional_stopping_digest": "0" * 64}
        ),
        lambda value: value["lifecycle"].update({"prefix_sha256": "0" * 64}),
    ],
)
def test_mapping_hydration_refuses_forked_or_tampered_checkpoint(mutation: object) -> None:
    value = campaign_checkpoint_to_mapping(_ready())
    mutation(value)  # type: ignore[operator]

    with pytest.raises(D1CampaignCheckpointError):
        campaign_checkpoint_from_mapping(value)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"d1-campaign-checkpoint-v1","schema_version":"fork"}',
        '{"schema_version":NaN}',
        " {}",
    ],
)
def test_text_hydration_refuses_duplicate_nonfinite_or_noncanonical_json(payload: str) -> None:
    with pytest.raises(D1CampaignCheckpointError):
        parse_campaign_checkpoint(payload)


def test_serializer_is_canonical_and_requires_nominal_checkpoint() -> None:
    checkpoint = _ready()
    assert serialize_campaign_checkpoint(checkpoint) == json.dumps(
        campaign_checkpoint_to_mapping(checkpoint),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(TypeError, match="D1CampaignCheckpoint"):
        serialize_campaign_checkpoint(object())  # type: ignore[arg-type]
