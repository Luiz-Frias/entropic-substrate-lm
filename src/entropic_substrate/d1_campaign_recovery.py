"""Pure restart hydration for D1 campaign checkpoints and authoritative ledgers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, NoReturn, cast

from entropic_substrate.d1_campaign_checkpoint import (
    D1CampaignCheckpoint,
    D1CampaignCompletedCell,
    D1CampaignDisposition,
)
from entropic_substrate.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.evidence.d1_lifecycle_values import (
    D1ProcessEventKind,
    D1ProcessState,
    canonical_lifecycle_event_line,
)
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
)
from entropic_substrate.experiments.d1_confirmatory_matrix import D1ConfirmatoryLaunch
from entropic_substrate.experiments.d1_launch_attempts import D1LaunchAttemptLedger
from entropic_substrate.experiments.d1_plan import ConfirmatoryD1RunPlan
from entropic_substrate.experiments.d1_primitives import D1Arm, canonical_digest
from entropic_substrate.experiments.d1_registered_design import D1RegisteredDesign

_RECOVERY_TOKEN: Final = object()


class D1CampaignRecoveryError(ValueError):
    """Checkpoint, registration, intent, or lifecycle authorities disagree."""

    def __init__(self, *, code: str, detail: str) -> None:
        self.code = code
        self.context = {"code": code}
        super().__init__(f"D1 campaign recovery {code}: {detail}")


class D1CampaignRecoveryStatus(StrEnum):
    READY = "ready"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class D1HydratedCampaignRecovery:
    status: D1CampaignRecoveryStatus
    checkpoint: D1CampaignCheckpoint
    process_manager: D1ProcessManager
    completed_launches: tuple[D1ConfirmatoryLaunch, ...]
    remaining_intents: tuple[D1OperatorLaunchIntent, ...]
    _token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _RECOVERY_TOKEN:
            raise TypeError("D1HydratedCampaignRecovery is factory-only")

    @property
    def next_index(self) -> int:
        return self.checkpoint.next_index

    @property
    def attempt_ledger(self) -> D1LaunchAttemptLedger:
        return self.checkpoint.attempt_ledger


@dataclass(frozen=True, slots=True)
class D1CampaignRecoveryHalt:
    reason: str
    checkpoint: D1CampaignCheckpoint
    process_manager: D1ProcessManager
    active_subject: str | None
    _token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _RECOVERY_TOKEN:
            raise TypeError("D1CampaignRecoveryHalt is factory-only")

    @property
    def attempt_ledger(self) -> D1LaunchAttemptLedger:
        return self.checkpoint.attempt_ledger


def _fail(code: str, detail: str) -> NoReturn:
    raise D1CampaignRecoveryError(code=code, detail=detail)


def _validate_roster(
    design: D1RegisteredDesign,
    intents: tuple[D1OperatorLaunchIntent, ...],
) -> None:
    if type(intents) is not tuple or any(
        type(intent) is not D1OperatorLaunchIntent for intent in intents
    ):
        _fail("intent_roster", "intents must be a nominal immutable tuple")
    expected = tuple((arm, seed) for seed in design.registered_seeds for arm in D1Arm)
    actual = tuple((intent.plan.profile.arm, intent.plan.seed) for intent in intents)
    if actual != expected or any(
        intent.registered_design_digest != design.design_digest
        or intent.execution_segment_index != 0
        for intent in intents
    ):
        _fail("intent_roster", "intents differ from the exact registered campaign roster")


def validate_d1_campaign_start(intent: D1OperatorLaunchIntent, start: RunStartProvenance) -> None:
    resource = intent.resource
    expected_resource = (
        resource.provider_gpu_type_id,
        resource.accelerator_count,
        resource.provider,
        resource.provider_sku,
        resource.region,
        resource.market,
        resource.purchase_option,
    )
    actual_resource = (
        start.accelerator_model,
        start.accelerator_count,
        start.provider,
        start.provider_sku,
        start.region,
        start.market,
        start.purchase_option,
    )
    if (
        start.logical_run_id != intent.plan.logical_run_id
        or start.execution_segment_index != 0
        or start.segment_start_mode is not SegmentStartMode.FRESH
        or start.resume_checkpoint_sha256 is not None
        or start.image_digest != intent.image.digest
        or start.quoted_rate != intent.quoted_rate
        or start.resource_activation != intent.resource_activation
        or start.compiler_enabled
        is not intent.plan.profile.draft.execution_protocol.optimization.compile
        or actual_resource != expected_resource
    ):
        _fail("completed_prefix", "completed cell run-start identity differs from intent")


def _completed_launches(
    design: D1RegisteredDesign,
    intents: tuple[D1OperatorLaunchIntent, ...],
    cells: tuple[D1CampaignCompletedCell, ...],
) -> tuple[D1ConfirmatoryLaunch, ...]:
    launches: list[D1ConfirmatoryLaunch] = []
    for index, cell in enumerate(cells):
        intent = intents[index]
        if (
            cell.logical_run_id != intent.plan.logical_run_id
            or cell.arm is not intent.plan.profile.arm
            or cell.seed != intent.plan.seed
            or cell.plan_digest != intent.plan.plan_digest
            or cell.intent_digest != canonical_digest(intent.manifest)
        ):
            _fail("completed_prefix", "completed cell differs from registered intent prefix")
        validate_d1_campaign_start(intent, cell.start)
        launches.append(
            D1ConfirmatoryLaunch(
                cast(ConfirmatoryD1RunPlan, intent.plan),
                cell.start,
                design.design_digest,
                design.decision_contract_digest,
            )
        )
    return tuple(launches)


def _validate_finalized_subjects(
    manager: D1ProcessManager,
    cells: tuple[D1CampaignCompletedCell, ...],
) -> None:
    expected = tuple(cell.logical_run_id for cell in cells)
    if manager.finalized_subjects != expected:
        _fail("lifecycle", "finalized lifecycle subjects differ from completed prefix")


def _validate_completed_lifecycle_prefixes(
    manager: D1ProcessManager,
    cells: tuple[D1CampaignCompletedCell, ...],
) -> None:
    events = manager.events
    for cell in cells:
        count = cell.lifecycle_event_count_after
        if count > len(events):
            _fail("lifecycle", "completed cell lifecycle count exceeds hydrated history")
        boundary = events[count - 1]
        prefix = b"".join(canonical_lifecycle_event_line(event) for event in events[:count])
        if (
            hashlib.sha256(prefix).hexdigest() != cell.lifecycle_prefix_after
            or boundary.event_kind is not D1ProcessEventKind.OPERATOR_FINALIZED
            or boundary.subject_id != cell.logical_run_id
        ):
            _fail(
                "lifecycle",
                "completed cell does not bind an exact operator-finalized lifecycle prefix",
            )


def _lifecycle_matches(
    checkpoint: D1CampaignCheckpoint,
    manager: D1ProcessManager,
) -> bool:
    prefix = manager.history.prefix
    return (
        checkpoint.lifecycle_prefix == prefix.prefix_sha256
        and checkpoint.lifecycle_event_count == prefix.event_count
        and checkpoint.process_state is manager.state
        and checkpoint.active_subject == manager.active_subject
    )


def _observed_inflight(manager: D1ProcessManager) -> bool:
    return manager.active_subject is not None or manager.state in {
        D1ProcessState.LAUNCH_ADMITTED,
        D1ProcessState.RUNNING,
        D1ProcessState.TRAINER_STOPPED,
        D1ProcessState.PROBLEM,
    }


def hydrate_d1_campaign_recovery(
    *,
    design: D1RegisteredDesign,
    intents: tuple[D1OperatorLaunchIntent, ...],
    checkpoint: D1CampaignCheckpoint,
    process_manager: D1ProcessManager,
) -> D1HydratedCampaignRecovery | D1CampaignRecoveryHalt:
    """Reconcile all authorities without appending events or invoking providers."""
    if (
        type(design) is not D1RegisteredDesign
        or type(checkpoint) is not D1CampaignCheckpoint
        or type(process_manager) is not D1ProcessManager
    ):
        raise TypeError("D1 campaign hydration requires nominal authorities")
    if (
        checkpoint.design_digest != design.design_digest
        or checkpoint.decision_contract_digest != design.decision_contract_digest
    ):
        _fail("registration", "checkpoint registration identities differ from active design")
    _validate_roster(design, intents)
    if checkpoint.next_index > len(intents):
        _fail("completed_prefix", "checkpoint completed prefix exceeds registered roster")
    launches = _completed_launches(design, intents, checkpoint.completed_cells)
    _validate_completed_lifecycle_prefixes(process_manager, checkpoint.completed_cells)
    _validate_finalized_subjects(process_manager, checkpoint.completed_cells)

    if checkpoint.disposition is D1CampaignDisposition.READY and _observed_inflight(
        process_manager
    ):
        return D1CampaignRecoveryHalt(
            "observed lifecycle is in-flight; provider state must be adjudicated",
            checkpoint,
            process_manager,
            process_manager.active_subject,
            _token=_RECOVERY_TOKEN,
        )
    if not _lifecycle_matches(checkpoint, process_manager):
        _fail("lifecycle", "checkpoint lifecycle head differs from hydrated process manager")
    if checkpoint.disposition is D1CampaignDisposition.HARD_HALT:
        return D1CampaignRecoveryHalt(
            checkpoint.halt_reason or "campaign checkpoint is hard-halted",
            checkpoint,
            process_manager,
            checkpoint.active_subject,
            _token=_RECOVERY_TOKEN,
        )
    status = (
        D1CampaignRecoveryStatus.COMPLETE
        if checkpoint.next_index == len(intents)
        else D1CampaignRecoveryStatus.READY
    )
    return D1HydratedCampaignRecovery(
        status,
        checkpoint,
        process_manager,
        launches,
        intents[checkpoint.next_index :],
        _token=_RECOVERY_TOKEN,
    )
