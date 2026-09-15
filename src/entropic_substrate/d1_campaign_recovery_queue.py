"""Restart-aware execution of the remaining suffix of a D1 campaign."""

from __future__ import annotations

from typing import Protocol, cast

from entropic_substrate.d1_campaign_checkpoint import (
    D1CampaignCheckpoint,
    D1CampaignCompletedCell,
    D1CampaignDisposition,
)
from entropic_substrate.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointReceipt,
)
from entropic_substrate.d1_campaign_queue import (
    D1CampaignQueueHalt,
    D1CompletedCampaign,
    validate_d1_campaign_ledger_extension,
)
from entropic_substrate.d1_campaign_recovery import (
    D1CampaignRecoveryHalt,
    D1HydratedCampaignRecovery,
    validate_d1_campaign_start,
)
from entropic_substrate.d1_operator_controller import D1OperatorRunResult
from entropic_substrate.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.experiments.d1_confirmatory_matrix import (
    D1ConfirmatoryLaunch,
    D1GitProvenancePort,
    attest_d1_confirmatory_matrix,
)
from entropic_substrate.experiments.d1_launch_attempts import D1LaunchAttemptLedger
from entropic_substrate.experiments.d1_plan import ConfirmatoryD1RunPlan
from entropic_substrate.experiments.d1_primitives import canonical_digest
from entropic_substrate.experiments.d1_registered_design import D1RegisteredDesign


class D1CampaignCheckpointRepositoryPort(Protocol):
    def load(
        self, *, expected_sha256: str | None = None
    ) -> tuple[D1CampaignCheckpoint, D1CampaignCheckpointReceipt]: ...

    def save(
        self,
        checkpoint: D1CampaignCheckpoint,
        *,
        expected_previous_sha256: str | None,
    ) -> D1CampaignCheckpointReceipt: ...


class D1RecoveryAttemptControllerPort(Protocol):
    @property
    def process_manager(self) -> D1ProcessManager: ...

    @property
    def attempt_ledger(self) -> D1LaunchAttemptLedger: ...

    def execute_fresh(self, intent: D1OperatorLaunchIntent) -> D1OperatorRunResult: ...


class D1RecoverableCampaignQueue:
    """Execute no cell before replaying checkpoint and lifecycle authority."""

    def __init__(
        self,
        *,
        design: D1RegisteredDesign,
        intents: tuple[D1OperatorLaunchIntent, ...],
        controller: D1RecoveryAttemptControllerPort,
        git: D1GitProvenancePort,
        repository: D1CampaignCheckpointRepositoryPort,
    ) -> None:
        if type(design) is not D1RegisteredDesign or type(intents) is not tuple:
            raise TypeError("recoverable D1 queue requires design and immutable intents")
        self._design, self._intents = design, intents
        self._controller, self._git, self._repository = controller, git, repository

    def run(
        self,
        recovery: D1HydratedCampaignRecovery | D1CampaignRecoveryHalt,
        *,
        generation: D1CampaignCheckpointReceipt,
    ) -> D1CompletedCampaign:
        self._verify_generation(recovery, generation)
        if isinstance(recovery, D1CampaignRecoveryHalt):
            raise D1CampaignQueueHalt(
                recovery.active_subject or "campaign-recovery",
                (),
                recovery.reason,
            )
        checkpoint = recovery.checkpoint
        if (
            checkpoint.design_digest != self._design.design_digest
            or checkpoint.decision_contract_digest != self._design.decision_contract_digest
            or recovery.remaining_intents != self._intents[recovery.next_index :]
        ):
            raise D1CampaignQueueHalt(
                "campaign-recovery", (), "hydrated recovery differs from queue registration"
            )
        launches = list(recovery.completed_launches)
        results: list[D1OperatorRunResult] = []
        ledger = recovery.attempt_ledger
        manager = recovery.process_manager
        receipt = generation
        if not recovery.remaining_intents:
            return self._complete(recovery, results, launches, ledger, manager)
        for intent in recovery.remaining_intents:
            subject = intent.plan.logical_run_id
            if (
                self._controller.process_manager != manager
                or self._controller.attempt_ledger != ledger
            ):
                raise D1CampaignQueueHalt(
                    subject, tuple(results), "controller authority differs from hydrated queue"
                )
            try:
                result = self._controller.execute_fresh(intent)
                self._validate_result(result, intent, ledger, manager)
            except Exception as error:
                raise D1CampaignQueueHalt(subject, tuple(results), str(error)) from error
            results.append(result)
            ledger, manager = result.attempt_ledger, result.process_manager
            launch = D1ConfirmatoryLaunch(
                cast(ConfirmatoryD1RunPlan, intent.plan),
                result.start,
                self._design.design_digest,
                self._design.decision_contract_digest,
            )
            launches.append(launch)
            cell = D1CampaignCompletedCell(
                len(launches) - 1,
                subject,
                intent.plan.profile.arm,
                intent.plan.seed,
                intent.plan.plan_digest,
                canonical_digest(intent.manifest),
                result.start,
                ledger.ledger_digest,
                manager.history.prefix.prefix_sha256,
                len(manager.events),
                (
                    "numerical_defect"
                    if result.terminal_numerical_defect is not None
                    else "completed"
                ),
                result.finalized_cell_artifact,
            )
            try:
                checkpoint = D1CampaignCheckpoint(
                    self._design.design_digest,
                    self._design.decision_contract_digest,
                    (*checkpoint.completed_cells, cell),
                    ledger,
                    manager.state,
                    manager.history.prefix.prefix_sha256,
                    len(manager.events),
                    manager.active_subject,
                    D1CampaignDisposition.READY,
                    None,
                )
                receipt = self._repository.save(
                    checkpoint,
                    expected_previous_sha256=receipt.sha256,
                )
            except Exception as error:
                raise D1CampaignQueueHalt(subject, tuple(results), str(error)) from error
        return self._complete(recovery, results, launches, ledger, manager)

    def _verify_generation(
        self,
        recovery: D1HydratedCampaignRecovery | D1CampaignRecoveryHalt,
        generation: D1CampaignCheckpointReceipt,
    ) -> None:
        try:
            checkpoint, receipt = self._repository.load(expected_sha256=generation.sha256)
        except Exception as error:
            raise D1CampaignQueueHalt("campaign-recovery", (), str(error)) from error
        if receipt != generation or checkpoint != recovery.checkpoint:
            raise D1CampaignQueueHalt(
                "campaign-recovery", (), "repository generation differs from hydrated recovery"
            )

    @staticmethod
    def _validate_result(
        result: D1OperatorRunResult,
        intent: D1OperatorLaunchIntent,
        ledger: D1LaunchAttemptLedger,
        manager: D1ProcessManager,
    ) -> None:
        if type(result) is not D1OperatorRunResult or not result.campaign_continuable:
            raise ValueError("controller result is not a campaign-continuable D1 result")
        if (
            result.start.logical_run_id != intent.plan.logical_run_id
            or result.start.execution_segment_index != 0
        ):
            raise ValueError("controller result differs from recovered intent")
        validate_d1_campaign_start(intent, result.start)
        validate_d1_campaign_ledger_extension(result, intent, ledger)
        observed = result.process_manager
        if (
            type(observed) is not D1ProcessManager
            or observed.state is not D1ProcessState.OPERATOR_FINALIZED
            or observed.active_subject is not None
            or observed.events[: len(manager.events)] != manager.events
            or observed.finalized_subjects
            != (*manager.finalized_subjects, intent.plan.logical_run_id)
        ):
            raise ValueError("controller lifecycle does not extend hydrated authority")

    def _complete(
        self,
        recovery: D1HydratedCampaignRecovery,
        results: list[D1OperatorRunResult],
        launches: list[D1ConfirmatoryLaunch],
        ledger: D1LaunchAttemptLedger,
        manager: D1ProcessManager,
    ) -> D1CompletedCampaign:
        matrix = attest_d1_confirmatory_matrix(
            design=self._design,
            launches=tuple(launches),
            git=self._git,
            ledger=ledger,
            terminal_defect_run_ids=(
                *(
                    cell.logical_run_id
                    for cell in recovery.checkpoint.completed_cells
                    if cell.terminal_outcome == "numerical_defect"
                ),
                *(
                    result.start.logical_run_id
                    for result in results
                    if result.terminal_numerical_defect is not None
                ),
            ),
        )
        return D1CompletedCampaign.from_attested(
            design_digest=self._design.design_digest,
            results=tuple(results),
            matrix=matrix,
            attempt_ledger=ledger,
            process_manager=manager,
            recovered_prefix_count=recovery.next_index,
        )
