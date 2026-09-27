"""An excluded pilot uses actual trainer evidence without preregistration authority."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as launch
from d1_trainer_fixtures import micro_plan
from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.ladder.capture.custody.build import build_corpus
from entropic_substrate.ladder.capture.records.d1_telemetry_query import read_d1_run_evidence
from entropic_substrate.ladder.register.d1_launch_attempts import (
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.ladder.register.d1_plan import (
    create_d1_excluded_pilot_plan,
    parse_d1_confirmatory_plan,
)
from entropic_substrate.ladder.register.d1_provenance import (
    CodeIdentity,
    ContainerIdentity,
    DataIdentity,
)
from entropic_substrate.ladder.run.campaign.d1_excluded_operator import D1ExcludedOperator
from entropic_substrate.ladder.run.campaign.d1_operator_custody_adapter import (
    D1OperatorCustodyAdapter,
)
from entropic_substrate.ladder.run.campaign.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    attest_d1_staged_trainer_request,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import (
    D1A100FormFactor,
    D1A100ResourceRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
)
from entropic_substrate.ladder.run.trainer.d1_fresh_trainer import run_d1_fresh
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    parse_d1_trainer_request,
    serialize_d1_trainer_request,
)

from .test_d1_operator_controller import _Clock, _Provider, _Trainer, _TrainerClock

pytestmark = pytest.mark.unit


class LivePilotTrainer(_Trainer):
    """Fake transport executes the real trainer only after its verified start."""

    def stage_inputs(self, pod_id, *, request):
        super().stage_inputs(pod_id, request=request)
        artifact = publish_d1_atomic_bytes(
            self.root, "request.json", serialize_d1_trainer_request(request), label="pilot request"
        )
        return attest_d1_staged_trainer_request(request=request, artifact=artifact)

    def await_exported_handoff(self, pod_id: str) -> D1ExportedTrainerHandoff:
        self.events.append("await_handoff")
        assert self.staged_request is not None
        request = parse_d1_trainer_request(
            serialize_d1_trainer_request(self.staged_request),
            catalog=(self.staged_request.plan.profile,),
        )
        (self.root / request.locations.journal_path).parent.mkdir(parents=True)
        locations = request.locations.fresh(self.root)
        completion = run_d1_fresh(
            plan=request.plan,
            start=request.start,
            origin=request.origin,
            locations=locations,
            clock=_TrainerClock(request.start.trainer_started_at),
        )
        return D1ExportedTrainerHandoff(completion, locations.journal_path)


def pilot_case(tmp_path: Path):
    smoke = micro_plan(D1Arm.NONE, seed=17)
    tokens = build_corpus(smoke.profile.data, "byte").train.tokens.astype("<i8")
    plan = create_d1_excluded_pilot_plan(
        smoke.profile,
        logical_run_id="pilot-none-17",
        seed=17,
        pairing_id="pilot-17",
        evidence=smoke.run.evidence,
        code=CodeIdentity(launch.CODE_COMMIT, launch.REPOSITORY_DIGEST),
        container=ContainerIdentity(launch.IMAGE_DIGEST),
        data=DataIdentity(
            "synthetic-test",
            launch.DATASET_DIGEST,
            hashlib.sha256(tokens.tobytes()).hexdigest(),
            launch.SAMPLER_DIGEST,
        ),
    )
    start = launch.start(plan)
    intent = D1OperatorLaunchIntent(
        plan=plan,
        registered_design_digest=None,
        image=D1ImmutableImage(
            "registry.example/entropic", launch.IMAGE_DIGEST, launch.CODE_COMMIT
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
    events: list[str] = []
    trainer = LivePilotTrainer(None, events, start)
    trainer.root = tmp_path
    provider = _Provider(start, events)
    custody_root = tmp_path / "custody"
    custody_root.mkdir()
    operator = D1ExcludedOperator(
        attempt_ledger=D1LaunchAttemptLedger(),
        runpod=provider,
        trainer=trainer,
        custody=D1OperatorCustodyAdapter(root=custody_root, finalized_cell_root=custody_root),
        clock=_Clock(start.trainer_started_at - timedelta(hours=1)),
    )
    return operator, intent, provider, trainer, events


def test_real_pilot_reaches_verified_terminal_evidence_without_confirm_manager(
    tmp_path: Path,
) -> None:
    operator, intent, provider, _trainer, events = pilot_case(tmp_path)
    result = operator.execute_fresh(intent)
    assert result.plan is intent.plan
    assert result.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.COMPLETED
    assert provider.delete_count == 1
    assert events.index("await_handoff") < events.index("terminate") < events.index("billing")
    run = read_d1_run_evidence(Path(result.journal.path), expected_run_class=RunClass.PILOT)
    assert run.plan_digest == intent.plan.plan_digest
    with pytest.raises(ValueError):
        parse_d1_confirmatory_plan(intent.plan.profile, intent.plan.plan_manifest)


def test_pilot_transport_defect_cleans_up_and_latches_operator(tmp_path: Path) -> None:
    operator, intent, provider, trainer, events = pilot_case(tmp_path)
    trainer.start_drift = "pod"
    with pytest.raises(RuntimeError, match="trainer"):
        operator.execute_fresh(intent)
    assert provider.cancel_count == 1
    assert "await_handoff" not in events
    assert operator.attempt_ledger.events[-1].status is D1LaunchAttemptStatus.FAILED
    with pytest.raises(RuntimeError, match="halted"):
        operator.execute_fresh(intent)


def test_missing_pilot_data_is_rejected_before_auth_or_allocation(tmp_path: Path) -> None:
    operator, intent, _provider, _trainer, events = pilot_case(tmp_path)
    changed = replace(intent.plan, run=replace(intent.plan.run, data=None))
    with pytest.raises(ValueError, match="data and code"):
        operator.execute_fresh(replace(intent, plan=changed))
    assert events == []


def test_acknowledged_pilot_deletion_is_not_repeated(tmp_path: Path) -> None:
    operator, intent, provider, _trainer, events = pilot_case(tmp_path)
    provider.acknowledge_then_fail = True
    with pytest.raises(RuntimeError, match="post-delete"):
        operator.execute_fresh(intent)
    assert provider.delete_count == 1
    assert "cancel" not in events
