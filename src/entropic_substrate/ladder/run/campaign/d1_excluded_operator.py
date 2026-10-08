"""Excluded pilot execution through the existing provider and trainer ports."""

from __future__ import annotations

from dataclasses import dataclass

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    RunStartProvenance,
    validate_execution_segments,
)
from entropic_substrate.kernel.identity.d1_primitives import RunClass
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.resource.resource_billing import (
    ExecutionSegmentTermination,
    TerminalProviderBillingEvidence,
)
from entropic_substrate.ladder.register.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.ladder.register.d1_plan import ExcludedPilotD1RunPlan
from entropic_substrate.ladder.run.campaign.d1_operator_controller import (
    D1OperatorClock,
    D1OperatorCustodyPort,
    D1OperatorQueueHalt,
    D1ResumeOperatorHistory,
)
from entropic_substrate.ladder.run.campaign.d1_operator_evidence import (
    D1TrainerExportReceipt,
    verify_d1_trainer_export,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import (
    D1OperatorLaunchIntent,
    attest_d1_run_start,
)
from entropic_substrate.ladder.run.closure.d1_billing_finalization_values import (
    D1BillingFinalization,
    D1BillingFinalizationRequest,
)
from entropic_substrate.ladder.run.closure.d1_billing_finalizer import finalize_d1_billing
from entropic_substrate.ladder.run.closure.d1_finalized_cell_custody import (
    identify_d1_finalized_journal,
)
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.ladder.run.trainer.d1_trainer_stage import stage_and_start_d1_trainer
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import D1TrainerFailureHandoff
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    d1_trainer_bootstrap_attempt_sha256,
)
from entropic_substrate.ladder.run.values.d1_operator_ports import (
    D1PodDeletionAcknowledgedError,
    D1RunPodPort,
    D1TrainerEvidencePort,
)


@dataclass(frozen=True, slots=True)
class D1ExcludedRunResult:
    """Pilot custody only; this type conveys no confirmatory filing authority."""

    plan: ExcludedPilotD1RunPlan
    attempt_ledger: D1LaunchAttemptLedger
    start: RunStartProvenance
    export: D1TrainerExportReceipt
    termination: ExecutionSegmentTermination
    billing: TerminalProviderBillingEvidence
    finalization: D1BillingFinalization
    journal: ArtifactIdentity
    custody_artifacts: tuple[ArtifactIdentity, ...]


class D1ExcludedOperator:
    """One pilot segment at a time; resumed segments retain the original attempt."""

    def __init__(
        self,
        *,
        attempt_ledger: D1LaunchAttemptLedger,
        runpod: D1RunPodPort,
        trainer: D1TrainerEvidencePort,
        custody: D1OperatorCustodyPort,
        clock: D1OperatorClock,
    ) -> None:
        self.attempt_ledger = attempt_ledger
        self._runpod, self._trainer, self._custody, self._clock = runpod, trainer, custody, clock
        self._active = False
        self._halted = False
        self._pod: str | None = None

    def execute_fresh(self, intent: D1OperatorLaunchIntent) -> D1ExcludedRunResult:
        return self._execute_guarded(intent, history=None)

    def execute_resume(
        self,
        intent: D1OperatorLaunchIntent,
        *,
        history: D1ResumeOperatorHistory,
    ) -> D1ExcludedRunResult:
        if type(history) is not D1ResumeOperatorHistory:
            raise TypeError("excluded pilot resume requires typed segment history")
        return self._execute_guarded(intent, history=history)

    def _execute_guarded(
        self,
        intent: D1OperatorLaunchIntent,
        *,
        history: D1ResumeOperatorHistory | None,
    ) -> D1ExcludedRunResult:
        if self._halted:
            raise D1OperatorQueueHalt("excluded D1 operator is halted")
        if self._active or self._pod is not None:
            raise RuntimeError("excluded D1 operator already has an active attempt")
        plan = intent.plan
        if type(plan) is not ExcludedPilotD1RunPlan or intent.registered_design_digest is not None:
            raise ValueError("excluded operator requires a pilot without registered authority")
        if history is None and (
            intent.execution_segment_index != 0 or intent.resume_checkpoint_sha256 is not None
        ):
            raise ValueError("excluded fresh operator requires segment zero")
        if plan.run.data is None or plan.run.execution.code is None:
            raise ValueError("remote pilot requires data and code identity")
        attempt = D1LaunchAttemptIdentity(
            RunClass.PILOT,
            plan.profile.family.family_digest,
            plan.profile.arm,
            plan.seed,
            plan.logical_run_id,
            None,
        )
        if history is not None:
            if (
                intent.execution_segment_index != len(history.starts)
                or intent.execution_segment_index < 1
            ):
                raise ValueError("pilot resume is not the exact next execution segment")
            if history.starts[0].logical_run_id != plan.logical_run_id:
                raise ValueError("pilot resume history logical identity differs")
            if any(not item.provider_billing_complete for item in history.billings):
                raise ValueError("pilot resume history has incomplete provider billing")
            if any(
                start.image_digest != intent.image.digest
                or start.resource_activation != intent.resource_activation
                for start in history.starts
            ):
                raise ValueError("pilot resume history image or resource activation differs")
            attempt_events = tuple(
                event
                for event in self.attempt_ledger.events
                if event.identity.optional_stopping_digest == attempt.optional_stopping_digest
            )
            if (
                not attempt_events
                or attempt_events[-1].identity != attempt
                or attempt_events[-1].status is not D1LaunchAttemptStatus.TRAINER_STARTED
            ):
                raise ValueError(
                    "pilot resume requires its original nonterminal trainer-started attempt"
                )
        self._active = True
        try:
            return self._execute(intent, plan, attempt, history)
        except Exception as error:
            self._halted = True
            cleanup: list[BaseException] = []
            if self._pod is not None:
                try:
                    self._runpod.cancel(self._pod)
                    self._pod = None
                except Exception as failure:
                    cleanup.append(failure)
            try:
                events = [
                    event for event in self.attempt_ledger.events if event.identity == attempt
                ]
                if events and events[-1].status in (
                    D1LaunchAttemptStatus.RECORDED,
                    D1LaunchAttemptStatus.TRAINER_STARTED,
                ):
                    self.attempt_ledger = self.attempt_ledger.transition(
                        attempt,
                        status=D1LaunchAttemptStatus.FAILED,
                        observed_at=max(self._clock.now_utc(), events[-1].observed_at),
                        reason=str(error),
                    )
                    self._custody.persist_attempt_ledger(self.attempt_ledger)
                self._custody.persist_problem(plan.logical_run_id, str(error))
            except Exception as failure:
                cleanup.append(failure)
            raise D1OperatorQueueHalt(
                str(error), primary=error, cleanup_failures=tuple(cleanup)
            ) from error
        finally:
            self._active = False

    def _execute(
        self,
        intent: D1OperatorLaunchIntent,
        plan: ExcludedPilotD1RunPlan,
        attempt: D1LaunchAttemptIdentity,
        history: D1ResumeOperatorHistory | None,
    ) -> D1ExcludedRunResult:
        auth = self._runpod.verify_authenticated()
        if history is None:
            self.attempt_ledger = self.attempt_ledger.record(
                attempt, observed_at=self._clock.now_utc()
            )
        records = [self._custody.persist_precreate(intent, self.attempt_ledger, auth)]
        placement = self._runpod.create(intent)
        self._pod = placement.pod_id
        attempt_digest = d1_trainer_bootstrap_attempt_sha256(placement.pod_id, intent)
        runtime = self._trainer.bootstrap_runtime(placement.pod_id, attempt_digest)
        start = attest_d1_run_start(intent=intent, placement=placement, runtime=runtime)
        prior_starts = () if history is None else history.starts
        validate_execution_segments((*prior_starts, start))
        if (
            history is not None
            and start.pod_created_at < history.terminations[-1].pod_terminated_at
        ):
            raise ValueError("pilot resume begins before the preceding pod terminated")
        records.append(self._custody.persist_run_start(start))
        records.append(
            stage_and_start_d1_trainer(
                self._trainer,
                placement.pod_id,
                plan,
                start,
                runtime,
                attempt_digest,
            )
        )
        if history is None:
            self.attempt_ledger = self.attempt_ledger.transition(
                attempt,
                status=D1LaunchAttemptStatus.TRAINER_STARTED,
                observed_at=start.trainer_started_at,
            )
        records.append(self._custody.persist_attempt_ledger(self.attempt_ledger))
        candidate = self._trainer.await_exported_handoff(placement.pod_id)
        export = verify_d1_trainer_export(candidate)
        stopped = export.handoff.termination
        if (stopped.logical_run_id, stopped.execution_segment_index) != (
            plan.logical_run_id,
            intent.execution_segment_index,
        ):
            raise ValueError("pilot trainer handoff differs from active segment")
        try:
            termination = self._runpod.terminate(
                placement.pod_id,
                start=start,
                trainer_ended_at=stopped.trainer_ended_at,
            )
        except D1PodDeletionAcknowledgedError as error:
            if error.pod_id == self._pod:
                self._pod = None
            raise
        self._pod = None
        billing = self._runpod.billing(placement.pod_id, start=start)
        finalization = finalize_d1_billing(
            D1BillingFinalizationRequest(
                plan=plan,
                segment_starts=(*prior_starts, start),
                segment_terminations=(*(history.terminations if history else ()), termination),
                provider_billings=(*(history.billings if history else ()), billing),
                handoff=export.handoff,
                journal_path=candidate.journal_path,
                expected_journal_state=stopped.journal_state,
            )
        )
        failed = isinstance(export.handoff, D1TrainerFailureHandoff)
        self.attempt_ledger = self.attempt_ledger.transition(
            attempt,
            status=D1LaunchAttemptStatus.FAILED if failed else D1LaunchAttemptStatus.COMPLETED,
            observed_at=termination.pod_terminated_at,
            reason=(
                export.handoff.error_message
                if isinstance(export.handoff, D1TrainerFailureHandoff)
                else None
            ),
        )
        records.append(self._custody.persist_attempt_ledger(self.attempt_ledger))
        records.append(self._custody.persist_finalization(finalization))
        if (
            isinstance(export.handoff, D1ResumeRunCompletion)
            and export.handoff.tail_quarantine is not None
        ):
            quarantine = export.handoff.tail_quarantine
            records.append(
                ArtifactIdentity(
                    str(quarantine.quarantine_path),
                    ContentHash("sha256", quarantine.quarantined_journal_sha256),
                )
            )
        if failed or not billing.provider_billing_complete:
            self._halted = True
        return D1ExcludedRunResult(
            plan,
            self.attempt_ledger,
            start,
            export,
            termination,
            billing,
            finalization,
            identify_d1_finalized_journal(candidate.journal_path, finalization),
            tuple(records),
        )
