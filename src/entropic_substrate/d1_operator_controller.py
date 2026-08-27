"""Sequential fresh-segment controller for the D1 cloud execution lane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Protocol

from entropic_substrate.d1_billing_finalization_values import D1BillingFinalization
from entropic_substrate.d1_billing_finalizer import finalize_d1_billing
from entropic_substrate.d1_finalized_cell_custody import (
    D1FinalizedCellCustodyRecord,
    attest_d1_finalized_cell_custody,
    identify_d1_finalized_journal,
)
from entropic_substrate.d1_numerical_defect import D1TerminalNumericalDefectRecord
from entropic_substrate.d1_operator_evidence import (
    D1StagedTrainerRequestReceipt,
    D1TrainerExportReceipt,
    D1TrainerStartReceipt,
    d1_runtime_observation_sha256,
    verify_d1_trainer_export,
)
from entropic_substrate.d1_operator_ports import (
    D1PodDeletionAcknowledgedError,
    D1RunPodAuthReceipt,
    D1RunPodPort,
    D1TrainerEvidencePort,
)
from entropic_substrate.d1_operator_values import (
    D1OperatorLaunchIntent,
    D1PodPlacement,
    D1RuntimeObservation,
    attest_d1_run_start,
)
from entropic_substrate.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.d1_trainer_handoff import D1TrainerFailureHandoff
from entropic_substrate.d1_trainer_request_codec import (
    build_d1_trainer_request,
    d1_trainer_bootstrap_attempt_sha256,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
    serialize_d1_trainer_request,
)
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.d1_lifecycle_values import D1ProblemKind
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.evidence.d1_telemetry_query import read_d1_run_evidence
from entropic_substrate.evidence.d1_telemetry_values import D1RunStatus
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.evidence.resource_billing import (
    ExecutionSegmentTermination,
    TerminalProviderBillingEvidence,
)
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    validate_execution_segments,
)
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.experiments.d1_plan import ConfirmatoryD1RunPlan, D1RunPlan
from entropic_substrate.experiments.d1_primitives import RunClass
from entropic_substrate.experiments.d1_registered_design import D1RegisteredDesign


class D1OperatorClock(Protocol):
    def now_utc(self) -> datetime: ...


def _quarantined_siblings(handoff: object) -> tuple[ArtifactIdentity, ...]:
    if type(handoff) is not D1ResumeRunCompletion or handoff.tail_quarantine is None:
        return ()
    receipt = handoff.tail_quarantine
    return (
        ArtifactIdentity(
            str(receipt.quarantine_path),
            ContentHash("sha256", receipt.quarantined_journal_sha256),
        ),
    )


def _terminal_defect(
    intent: D1OperatorLaunchIntent, journal_path: Path
) -> D1TerminalNumericalDefectRecord | None:
    run = read_d1_run_evidence(
        journal_path,
        expected_logical_run_id=intent.plan.logical_run_id,
        expected_family_digest=intent.plan.profile.family.family_digest,
        expected_run_class=RunClass.CONFIRMATORY,
        expected_plan_digest=intent.plan.plan_digest,
        allow_incomplete=True,
    )
    if run.status is not D1RunStatus.TRAINER_ABORTED or run.trainer_abort is None:
        return None
    return run.trainer_abort.numerical_defect


class D1OperatorCustodyPort(Protocol):
    """Durably persist each operator fact before the next external transition."""

    def persist_precreate(
        self,
        intent: D1OperatorLaunchIntent,
        ledger: D1LaunchAttemptLedger,
        auth: D1RunPodAuthReceipt,
    ) -> ArtifactIdentity: ...

    def persist_attempt_ledger(self, ledger: D1LaunchAttemptLedger) -> ArtifactIdentity: ...

    def persist_run_start(self, start: RunStartProvenance) -> ArtifactIdentity: ...

    def persist_finalization(self, finalization: D1BillingFinalization) -> ArtifactIdentity: ...

    def persist_finalized_cell(self, record: D1FinalizedCellCustodyRecord) -> ArtifactIdentity: ...

    def persist_problem(self, subject_id: str, detail: str) -> ArtifactIdentity: ...


class D1OperatorQueueHalt(RuntimeError):
    """Latched queue stop retaining primary and cleanup failures."""

    def __init__(
        self,
        detail: str,
        *,
        primary: BaseException | None = None,
        cleanup_failures: tuple[BaseException, ...] = (),
    ) -> None:
        self.primary = primary
        self.cleanup_failures = cleanup_failures
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class D1ResumeOperatorHistory:
    """Complete physical-segment evidence preceding one explicit resume."""

    starts: tuple[RunStartProvenance, ...]
    terminations: tuple[ExecutionSegmentTermination, ...]
    billings: tuple[TerminalProviderBillingEvidence, ...]

    def __post_init__(self) -> None:
        validate_execution_segments(self.starts)
        if not (len(self.starts) == len(self.terminations) == len(self.billings)):
            raise ValueError("D1 resume history segment evidence counts differ")
        for start, termination, billing in zip(
            self.starts, self.terminations, self.billings, strict=True
        ):
            identity = (start.logical_run_id, start.execution_segment_id)
            if identity != (termination.logical_run_id, termination.execution_segment_id):
                raise ValueError("D1 resume history start and termination identities differ")
            if identity != (billing.logical_run_id, billing.execution_segment_id):
                raise ValueError("D1 resume history start and billing identities differ")


@dataclass(frozen=True, slots=True)
class D1OperatorRunResult:
    process_manager: D1ProcessManager
    attempt_ledger: D1LaunchAttemptLedger
    start: RunStartProvenance
    placement: D1PodPlacement
    export: D1TrainerExportReceipt
    termination: ExecutionSegmentTermination
    billing: TerminalProviderBillingEvidence
    finalization: D1BillingFinalization
    finalized_cell: D1FinalizedCellCustodyRecord | None
    finalized_cell_artifact: ArtifactIdentity
    queue_admissible: bool


def _validate_registered(design: D1RegisteredDesign, intent: D1OperatorLaunchIntent) -> None:
    plan = intent.plan
    if type(plan) is not ConfirmatoryD1RunPlan:
        raise ValueError("D1 operator requires a confirmatory plan")
    if intent.registered_design_digest != design.design_digest:
        raise ValueError("D1 launch intent differs from the registered design")
    if (
        plan.seed not in design.registered_seeds
        or plan.profile.family.family_digest != design.family_digest
        or plan.run.construction_digest != design.arm_construction_digests[plan.profile.arm.value]
    ):
        raise ValueError("D1 launch cell is absent from the registered design")
    preregistration = plan.run.preregistration
    if preregistration is None or (
        preregistration.path,
        preregistration.commit,
        preregistration.family_digest,
    ) != (
        design.preregistration.path,
        design.preregistration.commit,
        design.family_digest,
    ):
        raise ValueError("D1 plan preregistration differs from the registered design")


def _validate_fresh(design: D1RegisteredDesign, intent: D1OperatorLaunchIntent) -> None:
    _validate_registered(design, intent)
    if intent.execution_segment_index != 0:
        raise ValueError("D1 fresh operator requires segment zero")


def _identity(intent: D1OperatorLaunchIntent) -> D1LaunchAttemptIdentity:
    plan = intent.plan
    return D1LaunchAttemptIdentity(
        run_class=RunClass.CONFIRMATORY,
        family_digest=plan.profile.family.family_digest,
        arm=plan.profile.arm,
        seed=plan.seed,
        logical_run_id=plan.logical_run_id,
        registered_design_digest=intent.registered_design_digest,
    )


class D1SequentialOperatorController:
    """Own one active attempt; campaign iteration belongs to the later queue."""

    def __init__(
        self,
        *,
        design: D1RegisteredDesign,
        process_manager: D1ProcessManager,
        attempt_ledger: D1LaunchAttemptLedger,
        runpod: D1RunPodPort,
        trainer: D1TrainerEvidencePort,
        custody: D1OperatorCustodyPort,
        operator: ActorIdentity,
        automation: ActorIdentity,
        clock: D1OperatorClock,
    ) -> None:
        if type(design) is not D1RegisteredDesign:
            raise TypeError("D1 operator requires a registered design")
        if type(process_manager) is not D1ProcessManager:
            raise TypeError("D1 operator requires D1ProcessManager")
        if type(attempt_ledger) is not D1LaunchAttemptLedger:
            raise TypeError("D1 operator requires D1LaunchAttemptLedger")
        if operator.role is not ActorRole.OPERATOR or automation.role is not ActorRole.AUTOMATION:
            raise ValueError("D1 operator actors have incorrect lifecycle roles")
        self.design, self.process_manager, self.attempt_ledger = (
            design,
            process_manager,
            attempt_ledger,
        )
        self._runpod, self._trainer, self._custody = runpod, trainer, custody
        self._operator, self._automation, self._clock = operator, automation, clock
        self._executing = False
        self._halted_detail: str | None = None
        self._active_attempt: D1LaunchAttemptIdentity | None = None
        self._active_pod_id: str | None = None

    def execute_fresh(self, intent: D1OperatorLaunchIntent) -> D1OperatorRunResult:
        if self._halted_detail is not None:
            raise D1OperatorQueueHalt(f"D1 operator queue is halted: {self._halted_detail}")
        if self._executing or self.process_manager.active_subject is not None:
            raise RuntimeError("D1 operator already has one active attempt")
        _validate_fresh(self.design, intent)
        self._executing = True
        try:
            return self._execute_fresh(intent)
        except D1OperatorQueueHalt:
            raise
        except Exception as error:
            self._halt(intent.plan.logical_run_id, error)

    def _execute_fresh(self, intent: D1OperatorLaunchIntent) -> D1OperatorRunResult:
        subject = intent.plan.logical_run_id
        attempt = _identity(intent)
        auth = self._runpod.verify_authenticated()
        ledger = self.attempt_ledger.record(attempt, observed_at=self._clock.now_utc())
        self.attempt_ledger, self._active_attempt = ledger, attempt
        precreate = self._custody.persist_precreate(intent, ledger, auth)
        manager = self.process_manager.admit_launch(
            subject_id=subject,
            actor=self._operator,
            artifact=precreate,
            event_id=f"{subject}:admit",
        )
        self.process_manager = manager
        placement = self._runpod.create(intent)
        self._active_pod_id = placement.pod_id
        attempt_sha256 = d1_trainer_bootstrap_attempt_sha256(placement.pod_id, intent)
        runtime = self._trainer.bootstrap_runtime(placement.pod_id, attempt_sha256)
        start = attest_d1_run_start(intent=intent, placement=placement, runtime=runtime)
        self._custody.persist_run_start(start)
        input_artifact = self._stage_and_start(
            placement.pod_id, intent.plan, start, runtime, attempt_sha256
        )
        manager = manager.start_running(
            subject_id=subject,
            actor=self._automation,
            artifact=input_artifact,
            event_id=f"{subject}:running:0",
        )
        self.process_manager = manager
        ledger = ledger.transition(
            attempt,
            status=D1LaunchAttemptStatus.TRAINER_STARTED,
            observed_at=start.trainer_started_at,
        )
        self._custody.persist_attempt_ledger(ledger)
        self.attempt_ledger = ledger
        candidate = self._trainer.await_exported_handoff(placement.pod_id)
        export = verify_d1_trainer_export(candidate)
        handoff = export.handoff
        stopped = handoff.termination
        if (
            stopped.logical_run_id,
            stopped.execution_segment_index,
        ) != (subject, 0):
            raise ValueError("trainer handoff differs from the active execution segment")
        manager = manager.record_trainer_stop(
            subject_id=subject,
            actor=self._automation,
            artifact=export.journal,
            event_id=f"{subject}:trainer-stop:0",
        )
        self.process_manager = manager
        termination = self._terminate_active_pod(
            placement.pod_id,
            start=start,
            trainer_ended_at=stopped.trainer_ended_at,
        )
        billing = self._runpod.billing(placement.pod_id, start=start)
        finalization = finalize_d1_billing(
            plan=intent.plan,
            segment_starts=(start,),
            segment_terminations=(termination,),
            provider_billings=(billing,),
            handoff=handoff,
            journal_path=candidate.journal_path,
            expected_journal_state=stopped.journal_state,
        )
        failed = isinstance(handoff, D1TrainerFailureHandoff)
        ledger = ledger.transition(
            attempt,
            status=(D1LaunchAttemptStatus.FAILED if failed else D1LaunchAttemptStatus.COMPLETED),
            observed_at=termination.pod_terminated_at,
            reason=(handoff.error_message if failed else None),
        )
        self._custody.persist_attempt_ledger(ledger)
        self.attempt_ledger = ledger
        queue_admissible = not failed and billing.provider_billing_complete
        defect = _terminal_defect(intent, candidate.journal_path) if failed else None
        finalized_cell: D1FinalizedCellCustodyRecord | None = None
        if isinstance(export.checkpoint, ArtifactIdentity) and (not failed or defect is not None):
            journal = identify_d1_finalized_journal(candidate.journal_path, finalization)
            finalized_cell = attest_d1_finalized_cell_custody(
                intent=intent,
                segment_starts=(start,),
                attempt_ledger=ledger,
                finalization=finalization,
                journal=journal,
                checkpoint=export.checkpoint,
                quarantined_siblings=_quarantined_siblings(handoff),
                numerical_defect=defect,
            )
            finalization_artifact = self._custody.persist_finalized_cell(finalized_cell)
        else:
            finalization_artifact = self._custody.persist_finalization(finalization)
        manager = manager.finalize_operator(
            subject_id=subject,
            actor=self._operator,
            artifact=finalization_artifact,
            event_id=f"{subject}:operator-finalized",
        )
        self.process_manager, self._executing = manager, False
        self._active_attempt = None
        if not queue_admissible:
            self._halted_detail = "terminal trainer or provider-billing evidence gap"
        return D1OperatorRunResult(
            manager,
            ledger,
            start,
            placement,
            export,
            termination,
            billing,
            finalization,
            finalized_cell,
            finalization_artifact,
            queue_admissible,
        )

    def execute_resume(
        self,
        intent: D1OperatorLaunchIntent,
        *,
        history: D1ResumeOperatorHistory,
    ) -> D1OperatorRunResult:
        if self._halted_detail is not None:
            raise D1OperatorQueueHalt(f"D1 operator queue is halted: {self._halted_detail}")
        if self._executing:
            raise RuntimeError("D1 operator already has one active attempt")
        _validate_registered(self.design, intent)
        subject = intent.plan.logical_run_id
        if (
            intent.execution_segment_index != len(history.starts)
            or intent.execution_segment_index < 1
        ):
            raise ValueError("D1 resume intent is not the exact next execution segment")
        if history.starts[0].logical_run_id != subject:
            raise ValueError("D1 resume history logical identity differs from the plan")
        if any(not item.provider_billing_complete for item in history.billings):
            raise ValueError("D1 resume history contains a provider billing evidence gap")
        if self.process_manager.active_subject != subject:
            raise ValueError("D1 resume requires the matching active lifecycle subject")
        attempt = _identity(intent)
        attempt_events = tuple(
            event
            for event in self.attempt_ledger.events
            if event.identity.optional_stopping_digest == attempt.optional_stopping_digest
        )
        if (
            not attempt_events
            or attempt_events[-1].status is not D1LaunchAttemptStatus.TRAINER_STARTED
        ):
            raise ValueError("D1 resume requires one nonterminal trainer-started attempt")
        self._executing, self._active_attempt = True, attempt
        try:
            return self._execute_resume(intent, history, attempt)
        except D1OperatorQueueHalt:
            raise
        except Exception as error:
            self._halt(subject, error)

    def _execute_resume(
        self,
        intent: D1OperatorLaunchIntent,
        history: D1ResumeOperatorHistory,
        attempt: D1LaunchAttemptIdentity,
    ) -> D1OperatorRunResult:
        subject, segment = intent.plan.logical_run_id, intent.execution_segment_index
        auth = self._runpod.verify_authenticated()
        precreate = self._custody.persist_precreate(intent, self.attempt_ledger, auth)
        del precreate
        placement = self._runpod.create(intent)
        self._active_pod_id = placement.pod_id
        attempt_sha256 = d1_trainer_bootstrap_attempt_sha256(placement.pod_id, intent)
        runtime = self._trainer.bootstrap_runtime(placement.pod_id, attempt_sha256)
        start = attest_d1_run_start(intent=intent, placement=placement, runtime=runtime)
        validate_execution_segments((*history.starts, start))
        self._custody.persist_run_start(start)
        input_artifact = self._stage_and_start(
            placement.pod_id, intent.plan, start, runtime, attempt_sha256
        )
        manager = self.process_manager.resume(
            subject_id=subject,
            actor=self._automation,
            artifact=input_artifact,
            event_id=f"{subject}:resume:{segment}",
        )
        self.process_manager = manager
        candidate = self._trainer.await_exported_handoff(placement.pod_id)
        export = verify_d1_trainer_export(candidate)
        handoff, stopped = export.handoff, export.handoff.termination
        if (stopped.logical_run_id, stopped.execution_segment_index) != (subject, segment):
            raise ValueError("resumed trainer handoff differs from the active segment")
        manager = manager.record_trainer_stop(
            subject_id=subject,
            actor=self._automation,
            artifact=export.journal,
            event_id=f"{subject}:trainer-stop:{segment}",
        )
        self.process_manager = manager
        termination = self._terminate_active_pod(
            placement.pod_id,
            start=start,
            trainer_ended_at=stopped.trainer_ended_at,
        )
        billing = self._runpod.billing(placement.pod_id, start=start)
        finalization = finalize_d1_billing(
            plan=intent.plan,
            segment_starts=(*history.starts, start),
            segment_terminations=(*history.terminations, termination),
            provider_billings=(*history.billings, billing),
            handoff=handoff,
            journal_path=candidate.journal_path,
            expected_journal_state=stopped.journal_state,
        )
        failed = isinstance(handoff, D1TrainerFailureHandoff)
        ledger = self.attempt_ledger.transition(
            attempt,
            status=(D1LaunchAttemptStatus.FAILED if failed else D1LaunchAttemptStatus.COMPLETED),
            observed_at=termination.pod_terminated_at,
            reason=(handoff.error_message if failed else None),
        )
        self._custody.persist_attempt_ledger(ledger)
        self.attempt_ledger = ledger
        queue_admissible = not failed and billing.provider_billing_complete
        defect = _terminal_defect(intent, candidate.journal_path) if failed else None
        finalized_cell: D1FinalizedCellCustodyRecord | None = None
        if isinstance(export.checkpoint, ArtifactIdentity) and (not failed or defect is not None):
            journal = identify_d1_finalized_journal(candidate.journal_path, finalization)
            finalized_cell = attest_d1_finalized_cell_custody(
                intent=intent,
                segment_starts=(*history.starts, start),
                attempt_ledger=ledger,
                finalization=finalization,
                journal=journal,
                checkpoint=export.checkpoint,
                quarantined_siblings=_quarantined_siblings(handoff),
                numerical_defect=defect,
            )
            artifact = self._custody.persist_finalized_cell(finalized_cell)
        else:
            artifact = self._custody.persist_finalization(finalization)
        manager = manager.finalize_operator(
            subject_id=subject,
            actor=self._operator,
            artifact=artifact,
            event_id=f"{subject}:operator-finalized",
        )
        self.process_manager, self._executing = manager, False
        self._active_attempt = None
        if not queue_admissible:
            self._halted_detail = "terminal trainer or provider-billing evidence gap"
        return D1OperatorRunResult(
            manager,
            ledger,
            start,
            placement,
            export,
            termination,
            billing,
            finalization,
            finalized_cell,
            artifact,
            queue_admissible,
        )

    def _stage_and_start(
        self,
        pod_id: str,
        plan: D1RunPlan,
        start: RunStartProvenance,
        runtime: D1RuntimeObservation,
        attempt_sha256: str,
    ) -> ArtifactIdentity:
        runtime_sha256 = d1_runtime_observation_sha256(runtime)
        origin = D1TrainerWallOrigin(
            runtime.trainer_started_at, runtime.trainer_monotonic_origin_ns
        )
        request = build_d1_trainer_request(plan, start, origin)
        request_bytes = serialize_d1_trainer_request(request)
        request_sha256 = d1_trainer_request_sha256(request)
        live = request
        request_identity = type(live), id(live.start), id(live.origin), id(live.locations)
        staged = self._trainer.stage_inputs(pod_id, request=request)
        if d1_runtime_observation_sha256(runtime) != runtime_sha256:
            raise ValueError("trainer runtime changed while staging its request")
        if type(staged) is not D1StagedTrainerRequestReceipt:
            raise TypeError("staged trainer request must be factory-attested")
        staged.require_factory_attested()
        if (
            (type(live), id(live.start), id(live.origin), id(live.locations)) != request_identity
            or serialize_d1_trainer_request(request) != request_bytes
            or d1_trainer_request_sha256(request) != request_sha256
            or staged.request is not request
            or staged.request_sha256 != request_sha256
            or staged.artifact.content_hash.algorithm != "sha256"
            or staged.artifact.content_hash.digest != request_sha256
        ):
            raise ValueError("staged trainer request differs from controller request")
        start_snapshot = (
            pod_id,
            attempt_sha256,
            staged.request_sha256,
            origin.started_at,
            runtime_sha256,
            d1_trainer_command_sha256(request),
        )
        started = self._trainer.start_trainer(pod_id)
        if type(started) is not D1TrainerStartReceipt:
            raise TypeError("trainer start must return a factory-attested receipt")
        started.require_factory_attested()
        if (
            d1_runtime_observation_sha256(runtime) != runtime_sha256
            or serialize_d1_trainer_request(request) != request_bytes
            or (
                started.pod_id,
                started.attempt_sha256,
                started.request_sha256,
                started.trainer_started_at,
                started.runtime_sha256,
                started.command_sha256,
            )
            != start_snapshot
            or not started.process_id.strip()
            or started.process_id.strip() != started.process_id
        ):
            raise ValueError("trainer start receipt differs from staged request or runtime")
        return staged.artifact

    def _terminate_active_pod(
        self,
        pod_id: str,
        *,
        start: RunStartProvenance,
        trainer_ended_at: datetime,
    ) -> ExecutionSegmentTermination:
        if self._active_pod_id != pod_id:
            raise RuntimeError("termination target differs from active pod custody")
        try:
            termination = self._runpod.terminate(
                pod_id,
                start=start,
                trainer_ended_at=trainer_ended_at,
            )
        except D1PodDeletionAcknowledgedError as error:
            if error.pod_id != pod_id:
                raise ValueError("deletion acknowledgement differs from active pod") from None
            self._active_pod_id = None
            raise
        self._active_pod_id = None
        return termination

    def _halt(self, subject: str, primary: BaseException) -> NoReturn:
        cleanup: list[BaseException] = []
        observed_at = self._clock.now_utc()
        if self._active_pod_id is not None:
            try:
                observed_at = self._runpod.cancel(self._active_pod_id)
            except BaseException as error:
                cleanup.append(error)
            self._active_pod_id = None
        attempt = self._active_attempt
        if attempt is not None:
            try:
                self.attempt_ledger = self.attempt_ledger.transition(
                    attempt,
                    status=D1LaunchAttemptStatus.FAILED,
                    observed_at=observed_at,
                    reason=str(primary) or repr(primary),
                )
                self._custody.persist_attempt_ledger(self.attempt_ledger)
            except BaseException as error:
                cleanup.append(error)
        detail = str(primary) or repr(primary)
        try:
            artifact = self._custody.persist_problem(subject, detail)
            self.process_manager = self.process_manager.record_problem(
                actor=self._operator,
                kind=D1ProblemKind.INVALIDATED,
                artifact=artifact,
                event_id=f"{subject}:operator-problem",
            )
        except BaseException as error:
            cleanup.append(error)
        if cleanup:
            detail += "; cleanup failures: " + "; ".join(str(item) for item in cleanup)
        self._halted_detail, self._executing, self._active_attempt = detail, False, None
        raise D1OperatorQueueHalt(
            detail,
            primary=primary,
            cleanup_failures=tuple(cleanup),
        ) from primary
