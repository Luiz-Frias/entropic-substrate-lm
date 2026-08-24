"""TDD contract for one sequential D1 provider/trainer transaction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as launch_fixtures
from d1_billing_finalizer_fixtures import provider_bill
from d1_trainer_fixtures import micro_plan
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_operator_controller import (
    D1OperatorQueueHalt,
    D1SequentialOperatorController,
)
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1TrainerInputEnvelope,
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


def _identity(events: list[str], label: str) -> ArtifactIdentity:
    events.append(label)
    return ArtifactIdentity(f"operator/{label}.json", ContentHash("sha256", f"{len(events):064x}"))


@dataclass
class _Custody:
    events: list[str]

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

    def runtime(self, pod_id: str) -> D1RuntimeObservation:
        self.events.append("runtime")
        start = self.start
        return D1RuntimeObservation(
            pod_id,
            start.trainer_started_at,
            start.cuda_version,
            start.driver_version,
            start.pytorch_version,
            start.compiler_enabled,
            start.compiler_version,
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

    def stage_inputs(self, pod_id: str, envelope: D1TrainerInputEnvelope) -> ArtifactIdentity:
        del pod_id
        assert envelope.canonical_json
        return _identity(self.events, "stage_inputs")

    def await_exported_handoff(self, pod_id: str) -> D1ExportedTrainerHandoff:
        del pod_id
        self.events.append("await_handoff")
        return self.candidate


@dataclass
class _Clock:
    value: datetime

    def now_utc(self) -> datetime:
        return self.value


def _controller(tmp_path: Path, case: _Case, events: list[str], *, drift: bool = False):
    provider = _Provider(
        case.start,
        events,
        placement_drift=drift,
    )
    controller = D1SequentialOperatorController(
        design=case.design,
        process_manager=_manager(tmp_path),
        attempt_ledger=D1LaunchAttemptLedger(),
        runpod=provider,
        trainer=_Trainer(D1ExportedTrainerHandoff(case.handoff, case.journal_path), events),
        custody=_Custody(events),
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

    result = controller.execute_fresh(case.intent)

    assert events == [
        "authenticate",
        "persist_precreate",
        "create",
        "runtime",
        "persist_run_start",
        "stage_inputs",
        "persist_attempt_ledger",
        "await_handoff",
        "terminate",
        "billing",
        "persist_finalization",
        "persist_attempt_ledger",
    ]
    assert result.process_manager.state is D1ProcessState.OPERATOR_FINALIZED
    assert result.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.COMPLETED
    assert read_journal(case.journal_path)[-1].event_kind is JournalEventKind.RUN_END
    assert result.queue_admissible is True


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
