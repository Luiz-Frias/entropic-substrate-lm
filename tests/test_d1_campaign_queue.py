"""TDD contract for the sequential registered D1 campaign queue."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import cast

import pytest

import d1_launch_admission_fixtures as fixtures
from entropic_substrate.d1_billing_finalization_values import D1BillingFinalization
from entropic_substrate.d1_campaign_queue import (
    D1CampaignQueue,
    D1CampaignQueueHalt,
    D1CompletedCampaign,
)
from entropic_substrate.d1_operator_controller import D1OperatorRunResult
from entropic_substrate.d1_operator_evidence import D1TrainerExportReceipt
from entropic_substrate.d1_operator_values import (
    D1A100FormFactor,
    D1A100ResourceRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
    D1PodPlacement,
)
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.evidence.resource_billing import (
    ExecutionSegmentTermination,
    ProviderBilling,
)
from entropic_substrate.experiments.d1_confirmatory_matrix import D1ConfirmatoryMatrix
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)

from .test_d1_confirmatory_matrix import FakeGit

pytestmark = pytest.mark.unit


def _design() -> D1RegisteredDesign:
    family = fixtures.family()
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family,
            fixtures.contract(family, seeds=(101,)),
            (101,),
            D1PreregistrationArtifact(
                fixtures.PREREG_PATH,
                fixtures.PREREG_BLOB,
                fixtures.PREREG_COMMIT,
            ),
        )
    )
    assert type(admitted) is D1RegisteredDesign
    return admitted


def _intent(design: D1RegisteredDesign, profile: object) -> tuple[D1OperatorLaunchIntent, object]:
    run = fixtures.plan(profile, 101)
    start = fixtures.start(run)
    intent = D1OperatorLaunchIntent(
        plan=run,
        registered_design_digest=design.design_digest,
        image=D1ImmutableImage(
            "registry.example/entropic-substrate-lm",
            fixtures.IMAGE_DIGEST,
            fixtures.CODE_COMMIT,
        ),
        resource=D1A100ResourceRequest(
            D1A100FormFactor.SXM,
            start.accelerator_model,
            start.accelerator_count,
            start.provider,
            start.provider_sku,
            start.region,
            start.market,
            start.purchase_option,
        ),
        quoted_rate=start.quoted_rate,
        resource_activation=start.resource_activation,
        execution_segment_index=0,
        resume_checkpoint_sha256=None,
        terminate_after_seconds=7200,
    )
    return intent, start


def _campaign() -> tuple[
    D1RegisteredDesign,
    tuple[D1OperatorLaunchIntent, ...],
    tuple[object, ...],
]:
    design = _design()
    pairs = tuple(_intent(design, profile) for profile in fixtures.profiles())
    return design, tuple(item[0] for item in pairs), tuple(item[1] for item in pairs)


def _results(
    intents: tuple[D1OperatorLaunchIntent, ...],
    starts: tuple[object, ...],
    *,
    halt_at: int | None = None,
) -> tuple[D1OperatorRunResult, ...]:
    ledger = D1LaunchAttemptLedger()
    values: list[D1OperatorRunResult] = []
    for index, (intent, start) in enumerate(zip(intents, starts, strict=True)):
        identity = D1LaunchAttemptIdentity(
            RunClass.CONFIRMATORY,
            intent.plan.profile.family.family_digest,
            intent.plan.profile.arm,
            intent.plan.seed,
            intent.plan.logical_run_id,
            intent.registered_design_digest,
        )
        ledger = ledger.record(
            identity, observed_at=start.trainer_started_at - timedelta(seconds=1)
        )
        ledger = ledger.transition(
            identity,
            status=D1LaunchAttemptStatus.TRAINER_STARTED,
            observed_at=start.trainer_started_at,
        )
        ledger = ledger.transition(
            identity,
            status=D1LaunchAttemptStatus.COMPLETED,
            observed_at=start.trainer_started_at + timedelta(seconds=1),
        )
        termination = ExecutionSegmentTermination(
            start.logical_run_id,
            0,
            start.execution_segment_id,
            start.trainer_started_at,
            start.trainer_started_at + timedelta(seconds=1),
        )
        values.append(
            D1OperatorRunResult(
                cast(D1ProcessManager, object()),
                ledger,
                start,
                cast(D1PodPlacement, object()),
                cast(D1TrainerExportReceipt, object()),
                termination,
                cast(ProviderBilling, object()),
                cast(D1BillingFinalization, object()),
                None,
                cast(ArtifactIdentity, object()),
                halt_at != index,
            )
        )
    return tuple(values)


@dataclass
class _Controller:
    results: tuple[D1OperatorRunResult, ...]
    calls: list[str]

    def execute_fresh(self, intent: D1OperatorLaunchIntent) -> D1OperatorRunResult:
        self.calls.append(intent.plan.logical_run_id)
        return self.results[len(self.calls) - 1]


def test_campaign_runs_exact_registered_order_then_attests_completed_matrix() -> None:
    design, intents, starts = _campaign()
    controller = _Controller(_results(intents, starts), [])

    completed = D1CampaignQueue(
        design=design,
        controller=controller,
        git=FakeGit(),
    ).run(intents)

    assert type(completed) is D1CompletedCampaign
    assert type(completed.matrix) is D1ConfirmatoryMatrix
    assert controller.calls == [intent.plan.logical_run_id for intent in intents]
    assert tuple((item.plan.profile.arm, item.plan.seed) for item in completed.matrix.launches) == (
        (D1Arm.NONE, 101),
        (D1Arm.SG, 101),
        (D1Arm.AC, 101),
        (D1Arm.BOTH, 101),
    )
    assert completed.attempt_ledger == completed.results[-1].attempt_ledger


def test_first_inadmissible_result_stops_before_any_later_arm() -> None:
    design, intents, starts = _campaign()
    controller = _Controller(_results(intents, starts, halt_at=1), [])

    with pytest.raises(D1CampaignQueueHalt, match=intents[1].plan.logical_run_id):
        D1CampaignQueue(design=design, controller=controller, git=FakeGit()).run(intents)

    assert controller.calls == [intent.plan.logical_run_id for intent in intents[:2]]


@pytest.mark.parametrize("mutation", ["missing_completion", "wrong_identity"])
def test_malformed_cell_ledger_stops_before_launching_the_next_arm(mutation: str) -> None:
    design, intents, starts = _campaign()
    results = list(_results(intents, starts))
    events = results[0].attempt_ledger.events
    if mutation == "missing_completion":
        ledger = D1LaunchAttemptLedger(events[:-1])
    else:
        drifted = replace(events[0].identity, seed=events[0].identity.seed + 1)
        ledger = D1LaunchAttemptLedger(tuple(replace(item, identity=drifted) for item in events))
    results[0] = replace(results[0], attempt_ledger=ledger)
    controller = _Controller(tuple(results), [])

    with pytest.raises(D1CampaignQueueHalt, match="attempt ledger"):
        D1CampaignQueue(design=design, controller=controller, git=FakeGit()).run(intents)

    assert controller.calls == [intents[0].plan.logical_run_id]


def test_missing_reordered_or_duplicate_roster_is_refused_before_first_launch() -> None:
    design, intents, starts = _campaign()
    controller = _Controller(_results(intents, starts), [])

    for invalid in (intents[:-1], (intents[1], intents[0], *intents[2:]), (*intents, intents[0])):
        with pytest.raises(ValueError, match="registered campaign roster"):
            D1CampaignQueue(design=design, controller=controller, git=FakeGit()).run(invalid)
        assert controller.calls == []


def _drift_provenance(intent: D1OperatorLaunchIntent, field: str) -> D1OperatorLaunchIntent:
    plan = intent.plan
    execution = plan.run.execution
    image = intent.image
    if field == "pairing":
        execution = replace(execution, pairing_id="different-pair")
    elif field == "code":
        execution = replace(
            execution,
            code=replace(execution.code, repository_digest="1" * 64),
        )
    elif field == "image":
        digest = "sha256:" + "2" * 64
        execution = replace(execution, container=replace(execution.container, image_digest=digest))
        image = replace(image, digest=digest)
    else:
        data = replace(plan.run.data, dataset_digest="3" * 64)
        plan = replace(plan, run=replace(plan.run, data=data))
    if field != "data":
        plan = replace(plan, run=replace(plan.run, execution=execution))
    return replace(intent, plan=plan, image=image)


@pytest.mark.parametrize("field", ["pairing", "code", "image", "data"])
def test_provenance_drift_is_refused_before_first_provider_launch(field: str) -> None:
    design, intents, starts = _campaign()
    drifted = (_drift_provenance(intents[0], field), *intents[1:])
    controller = _Controller(_results(intents, starts), [])

    with pytest.raises(ValueError, match="provenance"):
        D1CampaignQueue(design=design, controller=controller, git=FakeGit()).run(drifted)

    assert controller.calls == []
