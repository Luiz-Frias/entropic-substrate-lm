"""TDD contract for one sequential D1 provider/trainer transaction."""

from __future__ import annotations

import hashlib
import pickle
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

import d1_launch_admission_fixtures as launch_fixtures
from d1_billing_finalizer_fixtures import provider_bill
from d1_trainer_fixtures import micro_plan
from entropic_substrate.d1_finalized_cell_custody import D1FinalizedCellCustodyRecord
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_operator_controller import (
    D1OperatorQueueHalt,
    D1SequentialOperatorController,
)
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1StagedTrainerRequestReceipt,
    D1TrainerStartReceipt,
    attest_d1_staged_trainer_request,
    attest_d1_trainer_start,
    d1_runtime_observation_sha256,
)
from entropic_substrate.d1_operator_ports import D1RunPodAuthReceipt
from entropic_substrate.d1_operator_values import (
    D1A100FormFactor,
    D1A100ResourceRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
    D1PodPlacement,
    D1RuntimeObservation,
)
from entropic_substrate.d1_trainer_process_contracts import D1TrainerStartReport
from entropic_substrate.d1_trainer_request_codec import (
    D1TrainerRequest,
    build_d1_trainer_request,
    d1_trainer_bootstrap_attempt_sha256,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
    serialize_d1_trainer_request,
)
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.data.build import build_corpus
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.d1_lifecycle_repository import D1LifecycleRepository
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.experiments.d1_config_profile import D1LaunchScale
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)

from .test_d1_process_manager import _actor, _artifact

pytestmark = pytest.mark.unit


class _TrainerClock:
    def __init__(self, start: datetime) -> None:
        self._start = start
        self._utc_tick = 0
        self._mono_tick = 0

    def utc_now(self) -> datetime:
        self._utc_tick += 1
        return self._start + timedelta(seconds=self._utc_tick)

    def monotonic_ns(self) -> int:
        self._mono_tick += 1
        return 1_000_000_000 * self._mono_tick


@dataclass(frozen=True, slots=True)
class _Case:
    design: D1RegisteredDesign
    intent: D1OperatorLaunchIntent
    start: object
    handoff: object
    journal_path: Path


def _case(tmp_path: Path) -> _Case:
    smoke = micro_plan(D1Arm.NONE, seed=17)
    profile = replace(smoke.profile, scale=D1LaunchScale.FULL)
    train_tokens = build_corpus(profile.data, "byte").train.tokens.astype("<i8")
    plan = launch_fixtures.plan(
        profile,
        17,
        data={
            "dataset_name": "synthetic-test",
            "dataset_digest": launch_fixtures.DATASET_DIGEST,
            "split_digest": hashlib.sha256(train_tokens.tobytes()).hexdigest(),
            "sampler_construction_digest": launch_fixtures.SAMPLER_DIGEST,
        },
    )
    start = launch_fixtures.start(plan)
    contract = launch_fixtures.contract(profile.family, seeds=(17,))
    design = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=profile.family,
            decision_contract=contract,
            registered_seeds=(17,),
            preregistration=D1PreregistrationArtifact(
                launch_fixtures.PREREG_PATH,
                launch_fixtures.PREREG_BLOB,
                launch_fixtures.PREREG_COMMIT,
            ),
        )
    )
    assert type(design) is D1RegisteredDesign
    journal = tmp_path / "run.jsonl"
    checkpoint = tmp_path / "run.pt"
    completion = run_d1_fresh(
        plan=plan,
        start=start,
        origin=D1TrainerWallOrigin(start.trainer_started_at, 0),
        locations=D1FreshRunLocations(journal, checkpoint),
        clock=_TrainerClock(start.trainer_started_at),
    )
    intent = D1OperatorLaunchIntent(
        plan=plan,
        registered_design_digest=design.design_digest,
        image=D1ImmutableImage(
            repository="registry.example/entropic-substrate-lm",
            digest=launch_fixtures.IMAGE_DIGEST,
            source_commit=launch_fixtures.CODE_COMMIT,
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
    return _Case(design, intent, start, completion, journal)


def _manager(tmp_path: Path) -> D1ProcessManager:
    manager = D1ProcessManager.start(
        D1LifecycleRepository(tmp_path / "lifecycle.jsonl"),
        actor=_actor(ActorRole.IMPLEMENTATION),
        implementation_artifact=_artifact(1, "implementation"),
        contract_artifact=_artifact(2, "contract"),
        event_id="contract",
    )
    manager = manager.file_pilot(
        actor=_actor(ActorRole.OPERATOR), artifact=_artifact(3, "pilot"), event_id="pilot"
    )
    return manager.file_preregistration(
        actor=_actor(ActorRole.OPERATOR),
        artifact=_artifact(4, "prereg"),
        event_id="prereg",
    )


def test_trainer_step_receipts_are_factory_attested_and_noncopyable(tmp_path: Path) -> None:
    case = _case(tmp_path)
    trainer = _Trainer(D1ExportedTrainerHandoff(case.handoff, case.journal_path), [], case.start)
    runtime = trainer.bootstrap_runtime("pod-123", "a" * 64)
    request = build_d1_trainer_request(
        case.intent.plan,
        case.start,
        D1TrainerWallOrigin(runtime.trainer_started_at, runtime.trainer_monotonic_origin_ns),
    )
    with pytest.raises(ValueError, match="request digest"):
        attest_d1_staged_trainer_request(request=request, artifact=_artifact(9, "stage"))
    staged = trainer.stage_inputs("pod-123", request=request)
    started = trainer.start_trainer("pod-123")
    for receipt in (staged, started):
        receipt.require_factory_attested()
        for copier in (copy, deepcopy, pickle.dumps):
            with pytest.raises(TypeError, match="cannot be copied"):
                copier(receipt)
        with pytest.raises(TypeError, match="factory-attested"):
            replace(receipt)
    for receipt_type in (D1StagedTrainerRequestReceipt, D1TrainerStartReceipt):
        with pytest.raises(TypeError, match="factory-attested"):
            receipt_type()


def _identity(events: list[str], label: str) -> ArtifactIdentity:
    events.append(label)
    return ArtifactIdentity(f"operator/{label}.json", ContentHash("sha256", f"{len(events):064x}"))


@dataclass
class _Custody:
    events: list[str]
    fail_finalized_cell: bool = False

    def persist_precreate(self, intent: object, ledger: object, auth: object) -> ArtifactIdentity:
        del intent, ledger, auth
        return _identity(self.events, "persist_precreate")

    def persist_attempt_ledger(self, ledger: object) -> ArtifactIdentity:
        del ledger
        return _identity(self.events, "persist_attempt_ledger")

    def persist_run_start(self, start: object) -> ArtifactIdentity:
        del start
        return _identity(self.events, "persist_run_start")

    def persist_finalization(self, finalization: object) -> ArtifactIdentity:
        del finalization
        return _identity(self.events, "persist_finalization")

    def persist_finalized_cell(self, record: D1FinalizedCellCustodyRecord) -> ArtifactIdentity:
        record.require_factory_attested()
        self.events.append("persist_finalized_cell")
        if self.fail_finalized_cell:
            raise RuntimeError("simulated finalized-cell custody failure")
        return ArtifactIdentity(
            "operator/finalized-cell.json",
            ContentHash("sha256", record.record_digest),
        )

    def persist_problem(self, subject_id: str, detail: str) -> ArtifactIdentity:
        del subject_id, detail
        return _identity(self.events, "persist_problem")


@dataclass
class _Provider:
    start: object
    events: list[str]
    placement_drift: bool = False
    cancel_count: int = 0

    def verify_authenticated(self) -> D1RunPodAuthReceipt:
        self.events.append("authenticate")
        return D1RunPodAuthReceipt("account-123")

    def create(self, intent: D1OperatorLaunchIntent) -> D1PodPlacement:
        del intent
        self.events.append("create")
        start = self.start
        sku = "wrong-sku" if self.placement_drift else start.provider_sku
        return D1PodPlacement(
            "pod-123",
            D1A100FormFactor.SXM,
            start.accelerator_model,
            start.accelerator_count,
            start.provider,
            sku,
            start.region,
            start.market,
            start.purchase_option,
            start.pod_created_at,
        )

    def terminate(self, pod_id: str, *, start: object, trainer_ended_at: datetime):
        from entropic_substrate.evidence.resource_billing import ExecutionSegmentTermination

        del pod_id
        self.events.append("terminate")
        return ExecutionSegmentTermination(
            start.logical_run_id,
            start.execution_segment_index,
            start.execution_segment_id,
            trainer_ended_at,
            trainer_ended_at + timedelta(seconds=1),
        )

    def billing(self, pod_id: str, *, start: object):
        del pod_id
        self.events.append("billing")
        return provider_bill(start)

    def cancel(self, pod_id: str) -> datetime:
        del pod_id
        self.events.append("cancel")
        self.cancel_count += 1
        return self.start.trainer_started_at + timedelta(seconds=1)


@dataclass
class _Trainer:
    candidate: D1ExportedTrainerHandoff
    events: list[str]
    start: object
    stage_drift: str | None = None
    start_drift: str | None = None
    bootstrap_attempt_sha256: str | None = field(default=None, init=False)
    staged_request: D1TrainerRequest | None = field(default=None, init=False)
    runtime: D1RuntimeObservation | None = field(default=None, init=False)

    def bootstrap_runtime(self, pod_id: str, attempt_sha256: str) -> D1RuntimeObservation:
        self.events.append("runtime")
        self.bootstrap_attempt_sha256 = attempt_sha256
        start = self.start
        self.runtime = D1RuntimeObservation(
            pod_id,
            start.trainer_started_at,
            start.cuda_version,
            start.driver_version,
            start.pytorch_version,
            start.compiler_enabled,
            start.compiler_version,
            10_000,
        )
        return self.runtime

    def stage_inputs(
        self, pod_id: str, *, request: D1TrainerRequest
    ) -> D1StagedTrainerRequestReceipt:
        del pod_id
        self.staged_request = request
        payload = serialize_d1_trainer_request(request)
        assert b'"schema_version":"d1-trainer-command-request-v1"' in payload
        mutation = {
            "plan": (request.plan, "logical_run_id", "mutated-run"),
            "start": (request.start, "provider_sku", "mutated-sku"),
            "origin": (request.origin, "monotonic_ns", 11_000),
            "locations": (request.locations, "checkpoint_path", "runs/other.pt"),
            "runtime_version": (self.runtime, "pytorch_version", "stage-mutated"),
        }.get(self.stage_drift)
        if mutation is not None:
            object.__setattr__(*mutation)
        request_sha256 = d1_trainer_request_sha256(request)
        content_hash = (
            ContentHash("md5", "0" * 32)
            if self.stage_drift == "artifact_algorithm"
            else ContentHash(
                "sha256", "0" * 64 if self.stage_drift == "artifact_digest" else request_sha256
            )
        )
        self.events.append("stage_inputs")
        receipt = attest_d1_staged_trainer_request(
            request=request,
            artifact=ArtifactIdentity("operator/stage_inputs.json", content_hash),
        )
        if self.stage_drift == "digest":
            object.__setattr__(receipt, "request_sha256", "0" * 64)
        return receipt

    def start_trainer(self, pod_id: str) -> D1TrainerStartReceipt:
        self.events.append("start_trainer")
        if self.start_drift == "raise":
            raise RuntimeError("trainer start failed")
        if self.start_drift == "none":
            return cast(D1TrainerStartReceipt, None)
        assert self.staged_request is not None and self.runtime is not None
        if self.start_drift == "runtime_time":
            mutated_time = self.runtime.trainer_started_at + timedelta(seconds=1)
            object.__setattr__(self.runtime, "trainer_started_at", mutated_time)
        elif self.start_drift == "runtime_version":
            object.__setattr__(self.runtime, "pytorch_version", "mutated-version")
        if self.start_drift == "raw_report":
            return cast(D1TrainerStartReceipt, object.__new__(D1TrainerStartReport))
        return attest_d1_trainer_start(
            pod_id="other-pod" if self.start_drift == "pod" else pod_id,
            attempt_sha256=(
                "0" * 64
                if self.start_drift == "attempt"
                else self.bootstrap_attempt_sha256 or "0" * 64
            ),
            request_sha256=(
                "0" * 64
                if self.start_drift == "request"
                else d1_trainer_request_sha256(self.staged_request)
            ),
            trainer_started_at=(
                self.runtime.trainer_started_at + timedelta(seconds=1)
                if self.start_drift == "time"
                else self.runtime.trainer_started_at
            ),
            runtime_sha256=(
                "0" * 64
                if self.start_drift == "runtime_digest"
                else d1_runtime_observation_sha256(self.runtime)
            ),
            process_id={"process": " ", "process_padded": " trainer "}.get(
                self.start_drift, "trainer-123"
            ),
            command_sha256=(
                "0" * 64
                if self.start_drift == "command"
                else d1_trainer_command_sha256(self.staged_request)
            ),
        )

    def await_exported_handoff(self, pod_id: str) -> D1ExportedTrainerHandoff:
        del pod_id
        self.events.append("await_handoff")
        return self.candidate


@dataclass
class _Clock:
    value: datetime

    def now_utc(self) -> datetime:
        return self.value


def _controller(
    tmp_path: Path,
    case: _Case,
    events: list[str],
    *,
    drift: bool = False,
    custody_failure: bool = False,
    stage_drift: str | None = None,
    start_drift: str | None = None,
) -> tuple[D1SequentialOperatorController, _Provider]:
    provider = _Provider(
        case.start,
        events,
        placement_drift=drift,
    )
    trainer = _Trainer(
        D1ExportedTrainerHandoff(case.handoff, case.journal_path),
        events,
        case.start,
        stage_drift=stage_drift,
        start_drift=start_drift,
    )
    controller = D1SequentialOperatorController(
        design=case.design,
        process_manager=_manager(tmp_path),
        attempt_ledger=D1LaunchAttemptLedger(),
        runpod=provider,
        trainer=trainer,
        custody=_Custody(events, fail_finalized_cell=custody_failure),
        operator=ActorIdentity("operator", "operator-session", ActorRole.OPERATOR),
        automation=ActorIdentity("automation", "automation-session", ActorRole.AUTOMATION),
        clock=_Clock(launch_fixtures.STARTED_AT - timedelta(hours=1)),
    )
    return controller, provider


def test_fresh_segment_orders_custody_before_create_and_export_before_termination(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    controller, _ = _controller(tmp_path, case, events)
    trainer = cast(_Trainer, controller._trainer)

    result = controller.execute_fresh(case.intent)

    assert events == [
        "authenticate",
        "persist_precreate",
        "create",
        "runtime",
        "persist_run_start",
        "stage_inputs",
        "start_trainer",
        "persist_attempt_ledger",
        "await_handoff",
        "terminate",
        "billing",
        "persist_attempt_ledger",
        "persist_finalized_cell",
    ]
    assert result.process_manager.state is D1ProcessState.OPERATOR_FINALIZED
    assert result.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.COMPLETED
    assert result.finalized_cell is not None
    result.finalized_cell.require_factory_attested()
    assert result.finalized_cell.attempt_ledger == result.attempt_ledger
    assert result.finalized_cell_artifact.content_hash.digest == result.finalized_cell.record_digest
    assert result.process_manager.events[-1].artifacts == (result.finalized_cell_artifact,)
    assert read_journal(case.journal_path)[-1].event_kind is JournalEventKind.RUN_END
    assert result.queue_admissible is True
    assert trainer.bootstrap_attempt_sha256 == d1_trainer_bootstrap_attempt_sha256(
        "pod-123", case.intent
    )


@pytest.mark.parametrize(
    "drift",
    (
        "stage:plan stage:start stage:origin stage:locations stage:digest "
        "stage:artifact_algorithm stage:artifact_digest stage:runtime_version start:pod start:request start:time "
        "start:runtime_digest start:runtime_time start:runtime_version start:attempt start:command "
        "start:process start:process_padded start:raw_report start:none start:raise"
    ).split(),
)
def test_fresh_refuses_stage_or_start_receipt_drift(tmp_path: Path, drift: str) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    stage_drift, start_drift = (
        (drift.split(":", 1)[1], None) if drift.startswith("stage:") else (None, drift[6:])
    )
    controller, _ = _controller(
        tmp_path, case, events, stage_drift=stage_drift, start_drift=start_drift
    )
    with pytest.raises(D1OperatorQueueHalt, match="trainer"):
        controller.execute_fresh(case.intent)
    assert "cancel" in events
    assert controller.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.FAILED
    if stage_drift is not None:
        assert "start_trainer" not in events
    assert all(event.event_kind.value != "running" for event in controller.process_manager.events)
    assert all(
        event.status is not D1LaunchAttemptStatus.TRAINER_STARTED
        for event in controller.attempt_ledger.events
    )


def test_fresh_finalized_cell_custody_failure_never_records_operator_finalized(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    controller, _ = _controller(tmp_path, case, events, custody_failure=True)

    with pytest.raises(D1OperatorQueueHalt, match="custody failure"):
        controller.execute_fresh(case.intent)

    assert controller.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.COMPLETED
    assert controller.process_manager.state is D1ProcessState.PROBLEM
    assert "persist_finalized_cell" in events
    assert all(
        event.event_kind.value != "operator_finalized"
        for event in controller.process_manager.events
    )


def test_resource_drift_cancels_pod_halts_queue_and_never_starts_next_arm(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    controller, provider = _controller(tmp_path, case, events, drift=True)

    with pytest.raises(D1OperatorQueueHalt, match="resource placement"):
        controller.execute_fresh(case.intent)

    assert events == [
        "authenticate",
        "persist_precreate",
        "create",
        "runtime",
        "cancel",
        "persist_attempt_ledger",
        "persist_problem",
    ]
    assert provider.cancel_count == 1
    assert controller.process_manager.state is D1ProcessState.PROBLEM
    assert controller.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.FAILED
    with pytest.raises(D1OperatorQueueHalt, match="halted"):
        controller.execute_fresh(case.intent)
    assert provider.cancel_count == 1
