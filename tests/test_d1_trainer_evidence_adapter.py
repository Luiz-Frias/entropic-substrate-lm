"""Raw process transport cannot mint D1 trainer evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

import entropic_substrate.ladder.run.trainer.d1_segment_engine as segment_engine
from d1_resume_fixtures import (
    RecordingObserver,
    captured_step_one_case,
    origin,
    resume_start,
)
from d1_resume_hard_crash_fixture import hard_crash_tail_case
from d1_trainer_fixtures import ORIGIN_NS, micro_plan, run_start
from entropic_substrate.composition.wiring.d1_cli import D1CliCommand, D1CliInvocation
from entropic_substrate.composition.wiring.d1_trainer_cli_services import (
    D1FreshTrainerCliService,
    D1ResumeTrainerCliService,
)
from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.register.d1_plan import SmokeD1RunPlan
from entropic_substrate.ladder.run.campaign.d1_operator_evidence import (
    attest_d1_trainer_start,
    d1_runtime_observation_sha256,
)
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.ladder.run.transport.d1_trainer_evidence_adapter import (
    D1TrainerEvidenceAdapter,
)
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import D1TrainerFailureHandoff
from entropic_substrate.ladder.run.transport.d1_trainer_process_contracts import (
    D1TrainerAdapterPhase,
    D1TrainerArtifactReadback,
    d1_trainer_request_path,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
    D1TrainerRequest,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
    parse_d1_trainer_request,
    serialize_d1_trainer_request,
)

from . import test_d1_trainer_process_contracts as contract_fixtures

pytestmark = pytest.mark.unit
_POD, _ATTEMPT = contract_fixtures.POD, contract_fixtures.ATTEMPT
_json, _runtime_report = contract_fixtures.canonical_json, contract_fixtures.runtime_report
_start_report, _request = contract_fixtures.start_report, contract_fixtures.trainer_request


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.start, self.tick = start, 0

    def utc_now(self) -> datetime:
        self.tick += 1
        return self.start + timedelta(seconds=self.tick)

    def monotonic_ns(self) -> int:
        return ORIGIN_NS + self.tick * 1_000_000_000


@dataclass
class _LoopbackTransport:
    root: Path
    plan: SmokeD1RunPlan
    start: RunStartProvenance
    fail_on: str | None = None
    request_drift: str | None = None
    foreign_export_start: bool = False
    foreign_export_locations: bool = False
    resume_failure: bool = False
    calls: list[str] = field(default_factory=list)
    payload: bytes | None = None
    request_path: str | None = None
    request: D1TrainerRequest | None = None
    executed_request: D1TrainerRequest | None = None

    def _effect(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_on == name:
            raise RuntimeError(f"simulated {name} failure")

    def bootstrap(self, pod: str, attempt: str) -> None:
        assert (pod, attempt) == (_POD, _ATTEMPT)
        self._effect("bootstrap")

    def read_runtime_report(self, pod: str, attempt: str) -> bytes:
        assert (pod, attempt) == (_POD, _ATTEMPT)
        self._effect("runtime")
        return _json(_runtime_report(self.start))

    def write_request(self, pod: str, attempt: str, path: str, payload: bytes) -> None:
        assert (pod, attempt) == (_POD, _ATTEMPT)
        self._effect("write")
        self.request_path, self.payload = path, payload

    def read_request(self, pod: str, attempt: str, path: str) -> D1TrainerArtifactReadback:
        assert (pod, attempt) == (_POD, _ATTEMPT)
        self._effect("read")
        assert self.payload is not None and path == self.request_path
        payload = self.payload
        if self.request_drift == "path":
            path = " "
        elif self.request_drift == "foreign_path":
            path = f"requests/{'f' * 64}.json"
        elif self.request_drift == "absolute_path":
            path = "/requests/trainer.json"
        elif self.request_drift == "traversal_path":
            path = "../requests/trainer.json"
        elif self.request_drift is not None:
            if self.request_drift == "byte":
                payload += b"\n"
            else:
                value = json.loads(payload)
                target, key = {
                    "schema": (value, "schema_version"),
                    "plan": (value["plan"], "seed"),
                    "start": (value["run_start"], "pytorch_version"),
                    "origin": (value["wall_origin"], "monotonic_ns"),
                    "location": (value["locations"], "checkpoint_path"),
                }[self.request_drift]
                target[key] = "drift"
                payload = _json(value)
        return D1TrainerArtifactReadback(path, payload)

    def start_trainer(self, pod: str, attempt: str, request_sha: str) -> None:
        assert (pod, attempt) == (_POD, _ATTEMPT)
        self._effect("start")
        assert self.payload is not None and request_sha == hashlib.sha256(self.payload).hexdigest()
        self.request = parse_d1_trainer_request(
            self.payload,
            catalog=(self.plan.profile,),
        )
        trainer_start, trainer_payload = self.start, self.payload
        execution_request = self.request
        if self.foreign_export_start:
            trainer_start = replace(
                self.start,
                trainer_started_at=self.start.trainer_started_at + timedelta(seconds=30),
            )
            execution_request = _request(self.plan, trainer_start)
            trainer_payload = serialize_d1_trainer_request(execution_request)
        elif self.foreign_export_locations:
            execution_request = replace(
                self.request,
                locations=D1TrainerLocationSpec(
                    "foreign/journal.jsonl",
                    "foreign/checkpoint.pt",
                ),
            )
            trainer_payload = serialize_d1_trainer_request(execution_request)
        self.executed_request = execution_request
        value = json.loads(trainer_payload)
        path = self.root / "request.json"
        path.write_bytes(trainer_payload)
        if value["command"] == "d1-run-fresh":
            (self.root / execution_request.locations.journal_path).parent.mkdir(parents=True)
            result = D1FreshTrainerCliService(
                catalog=(self.plan.profile,),
                artifact_root=self.root,
                clock=_Clock(trainer_start.trainer_started_at),
            ).execute(D1CliInvocation(D1CliCommand.RUN_FRESH, path, value))
            assert result["handoff_kind"] == "fresh_completion"
        else:
            result = D1ResumeTrainerCliService(
                catalog=(self.plan.profile,),
                artifact_root=self.root,
                observer=RecordingObserver(),
                clock=_Clock(trainer_start.trainer_started_at),
            ).execute(D1CliInvocation(D1CliCommand.RUN_RESUME, path, value))
            assert result["handoff_kind"] == (
                "trainer_failure" if self.resume_failure else "resume_completion"
            )

    def read_start_report(self, pod: str, attempt: str, request_sha: str) -> bytes:
        assert (pod, attempt) == (_POD, _ATTEMPT)
        self._effect("start_report")
        assert self.request is not None and request_sha == d1_trainer_request_sha256(self.request)
        return _json(_start_report(self.request, self.start))

    def await_export(self, pod: str, attempt: str, prefix: str) -> Path:
        assert (pod, attempt) == (_POD, _ATTEMPT)
        self._effect("export")
        assert (self.root / f"{prefix}.complete.json").is_file()
        return self.root


def _adapter(tmp_path: Path, plan: SmokeD1RunPlan, start: RunStartProvenance, **kwargs: object):
    transport = _LoopbackTransport(tmp_path, plan, start, **kwargs)
    return D1TrainerEvidenceAdapter(transport=transport, catalog=(plan.profile,)), transport


def test_raw_adapter_drives_real_fresh_trainer_only_after_start(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    adapter, transport = _adapter(tmp_path, plan, start)
    runtime = adapter.bootstrap_runtime(_POD, _ATTEMPT)
    request = _request(plan, start)
    staged = adapter.stage_inputs(_POD, request=request)
    assert transport.calls == ["bootstrap", "runtime", "write", "read"]
    started = adapter.start_trainer(_POD)
    candidate = adapter.await_exported_handoff(_POD)
    assert staged.request_sha256 == d1_trainer_request_sha256(request)
    assert staged.artifact.path == d1_trainer_request_path(_ATTEMPT)
    assert started.attempt_sha256 == _ATTEMPT and started.process_id == "trainer-123"
    assert runtime.trainer_started_at == started.trainer_started_at
    assert candidate.handoff.successful_step == 2
    assert transport.calls == [
        "bootstrap",
        "runtime",
        "write",
        "read",
        "start",
        "start_report",
        "export",
    ]
    assert adapter.phase(_POD) is D1TrainerAdapterPhase.EXPORT_VERIFIED
    calls = tuple(transport.calls)
    for duplicate in (
        partial(adapter.bootstrap_runtime, _POD, _ATTEMPT),
        partial(adapter.stage_inputs, _POD, request=request),
        partial(adapter.start_trainer, _POD),
        partial(adapter.await_exported_handoff, _POD),
    ):
        with pytest.raises(ValueError):
            duplicate()
    assert tuple(transport.calls) == calls


def test_resume_layout_is_staged_without_execution(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.NONE)
    start = resume_start(plan, "9" * 64)
    adapter, transport = _adapter(tmp_path, plan, start)
    adapter.bootstrap_runtime(_POD, _ATTEMPT)
    request = _request(plan, start)
    adapter.stage_inputs(_POD, request=request)
    assert transport.payload == serialize_d1_trainer_request(request)
    assert request.locations.quarantine_path.endswith("-000001.jsonl")
    assert "start" not in transport.calls


def test_valid_foreign_export_is_rejected_and_latched(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    adapter, transport = _adapter(
        tmp_path,
        plan,
        start,
        foreign_export_start=True,
    )
    adapter.bootstrap_runtime(_POD, _ATTEMPT)
    adapter.stage_inputs(_POD, request=_request(plan, start))
    adapter.start_trainer(_POD)
    with pytest.raises(ValueError, match="request"):
        adapter.await_exported_handoff(_POD)
    calls = tuple(transport.calls)
    assert adapter.phase(_POD) is D1TrainerAdapterPhase.EXPORT_ATTEMPTED
    with pytest.raises(ValueError):
        adapter.await_exported_handoff(_POD)
    assert tuple(transport.calls) == calls


def test_independently_attested_foreign_start_cannot_join_staged_request(
    tmp_path: Path,
) -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    adapter, _ = _adapter(tmp_path, plan, start)
    runtime = adapter.bootstrap_runtime(_POD, _ATTEMPT)
    request = _request(plan, start)
    staged = adapter.stage_inputs(_POD, request=request)
    adapter.start_trainer(_POD)
    foreign = attest_d1_trainer_start(
        pod_id="foreign-pod",
        attempt_sha256="f" * 64,
        request_sha256=staged.request_sha256,
        runtime_sha256="e" * 64,
        trainer_started_at=start.trainer_started_at,
        process_id="foreign-process",
        command_sha256=d1_trainer_command_sha256(request),
    )

    with pytest.raises(ValueError, match="start receipt"):
        session = adapter._sessions[_POD]  # pyright: ignore[reportPrivateUsage]
        session.started = foreign
        adapter.await_exported_handoff(_POD)
    assert d1_runtime_observation_sha256(runtime) != foreign.runtime_sha256


def test_started_process_identity_cannot_be_replaced(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    adapter, _ = _adapter(tmp_path, plan, start)
    adapter.bootstrap_runtime(_POD, _ATTEMPT)
    request = _request(plan, start)
    adapter.stage_inputs(_POD, request=request)
    started = adapter.start_trainer(_POD)
    foreign = attest_d1_trainer_start(
        pod_id=started.pod_id,
        attempt_sha256=started.attempt_sha256,
        request_sha256=started.request_sha256,
        runtime_sha256=started.runtime_sha256,
        trainer_started_at=started.trainer_started_at,
        process_id="foreign-process",
        command_sha256=started.command_sha256,
    )
    session = adapter._sessions[_POD]  # pyright: ignore[reportPrivateUsage]
    session.started = foreign

    with pytest.raises(ValueError, match="start receipt"):
        adapter.await_exported_handoff(_POD)
    assert adapter.phase(_POD) is D1TrainerAdapterPhase.EXPORT_ATTEMPTED


def test_active_request_cannot_drift_from_staged_request(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    adapter, transport = _adapter(
        tmp_path,
        plan,
        start,
        foreign_export_locations=True,
    )
    adapter.bootstrap_runtime(_POD, _ATTEMPT)
    adapter.stage_inputs(_POD, request=_request(plan, start))
    adapter.start_trainer(_POD)
    session = adapter._sessions[_POD]  # pyright: ignore[reportPrivateUsage]
    assert transport.executed_request is not None
    session.request = transport.executed_request

    with pytest.raises(ValueError, match="staged request"):
        adapter.await_exported_handoff(_POD)
    assert adapter.phase(_POD) is D1TrainerAdapterPhase.EXPORT_ATTEMPTED


def test_real_resume_export_is_bound_to_its_exact_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    case = captured_step_one_case(interrupted, monkeypatch)
    start = resume_start(case.plan, case.checkpoint_sha256)
    start = replace(
        start,
        pod_created_at=start.pod_created_at + timedelta(minutes=1),
        trainer_started_at=start.trainer_started_at + timedelta(minutes=1),
    )
    request = D1ResumeTrainerRequest(
        case.plan,
        start,
        replace(origin(), started_at=start.trainer_started_at),
        D1TrainerLocationSpec(
            str(case.journal_path.relative_to(tmp_path)),
            str(case.checkpoint_path.relative_to(tmp_path)),
            "interrupted/quarantined-tail.jsonl",
        ),
    )
    adapter, _ = _adapter(tmp_path, case.plan, start)
    adapter.bootstrap_runtime(_POD, _ATTEMPT)
    adapter.stage_inputs(_POD, request=request)
    adapter.start_trainer(_POD)
    candidate = adapter.await_exported_handoff(_POD)
    assert type(candidate.handoff) is D1ResumeRunCompletion
    assert candidate.handoff.run_start_provenance == start
    assert adapter.phase(_POD) is D1TrainerAdapterPhase.EXPORT_VERIFIED


def test_resume_failure_binds_real_quarantined_tail_through_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    case = hard_crash_tail_case(interrupted, monkeypatch)

    def fail_step(**_kwargs: object) -> object:
        raise RuntimeError("resumed trainer sentinel")

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    start = replace(
        resume_start(case.plan, case.checkpoint_sha256),
        pod_created_at=run_start(case.plan).pod_created_at + timedelta(minutes=1),
        trainer_started_at=run_start(case.plan).trainer_started_at + timedelta(minutes=1),
    )
    quarantine = interrupted / "quarantined-tail.jsonl"
    request = D1ResumeTrainerRequest(
        case.plan,
        start,
        replace(origin(), started_at=start.trainer_started_at),
        D1TrainerLocationSpec(
            str(case.journal_path.relative_to(tmp_path)),
            str(case.checkpoint_path.relative_to(tmp_path)),
            str(quarantine.relative_to(tmp_path)),
        ),
    )
    adapter, _ = _adapter(
        tmp_path,
        case.plan,
        start,
        resume_failure=True,
    )
    adapter.bootstrap_runtime(_POD, _ATTEMPT)
    adapter.stage_inputs(_POD, request=request)
    adapter.start_trainer(_POD)
    candidate = adapter.await_exported_handoff(_POD)
    assert type(candidate.handoff) is D1TrainerFailureHandoff
    assert quarantine.is_file()
    assert adapter.phase(_POD) is D1TrainerAdapterPhase.EXPORT_VERIFIED


@pytest.mark.parametrize(
    "drift",
    "path foreign_path absolute_path traversal_path byte schema plan start origin location".split(),
)
def test_request_readback_drift_latches_stage(tmp_path: Path, drift: str) -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    adapter, transport = _adapter(tmp_path, plan, start, request_drift=drift)
    adapter.bootstrap_runtime(_POD, _ATTEMPT)
    request = _request(plan, start)
    with pytest.raises(ValueError):
        adapter.stage_inputs(_POD, request=request)
    calls = tuple(transport.calls)
    with pytest.raises(ValueError):
        adapter.stage_inputs(_POD, request=request)
    assert tuple(transport.calls) == calls


@pytest.mark.parametrize("effect", "bootstrap runtime write read start start_report export".split())
def test_effect_failure_is_latched_without_same_pod_retry(tmp_path: Path, effect: str) -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    adapter, transport = _adapter(tmp_path, plan, start, fail_on=effect)
    request = _request(plan, start)
    with pytest.raises(RuntimeError):
        adapter.bootstrap_runtime(_POD, _ATTEMPT)
        adapter.stage_inputs(_POD, request=request)
        adapter.start_trainer(_POD)
        adapter.await_exported_handoff(_POD)
    calls = tuple(transport.calls)
    if effect in {"bootstrap", "runtime"}:
        retry = partial(adapter.bootstrap_runtime, _POD, _ATTEMPT)
    elif effect in {"write", "read"}:
        retry = partial(adapter.stage_inputs, _POD, request=request)
    elif effect in {"start", "start_report"}:
        retry = partial(adapter.start_trainer, _POD)
    else:
        retry = partial(adapter.await_exported_handoff, _POD)
    with pytest.raises(ValueError):
        retry()
    assert tuple(transport.calls) == calls
    assert (
        adapter.phase(_POD)
        is {
            "bootstrap": D1TrainerAdapterPhase.BOOTSTRAP_ATTEMPTED,
            "runtime": D1TrainerAdapterPhase.RUNTIME_READBACK_ATTEMPTED,
            "write": D1TrainerAdapterPhase.STAGE_WRITE_ATTEMPTED,
            "read": D1TrainerAdapterPhase.STAGE_READBACK_ATTEMPTED,
            "start": D1TrainerAdapterPhase.START_ATTEMPTED,
            "start_report": D1TrainerAdapterPhase.START_READBACK_ATTEMPTED,
            "export": D1TrainerAdapterPhase.EXPORT_ATTEMPTED,
        }[effect]
    )


class _NoncallableTransport:
    bootstrap = read_runtime_report = write_request = read_request = None
    start_trainer = read_start_report = await_export = None


def test_adapter_rejects_invalid_construction_before_effects(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    transport = _LoopbackTransport(tmp_path, plan, start)
    with pytest.raises(TypeError):
        D1TrainerEvidenceAdapter(transport=transport, catalog=())
    with pytest.raises(ValueError):
        D1TrainerEvidenceAdapter(transport=transport, catalog=(plan.profile, plan.profile))
    with pytest.raises(TypeError):
        D1TrainerEvidenceAdapter(transport=_NoncallableTransport(), catalog=(plan.profile,))
    assert transport.calls == []


def test_invalid_identity_and_order_fail_before_effects(tmp_path: Path) -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    adapter, transport = _adapter(tmp_path, plan, start)
    request = _request(plan, start)
    for invalid in ("", " "):
        with pytest.raises(ValueError):
            adapter.bootstrap_runtime(invalid, _ATTEMPT)
    with pytest.raises(ValueError):
        adapter.bootstrap_runtime(_POD, "invalid")
    for premature in (
        partial(adapter.phase, _POD),
        partial(adapter.stage_inputs, _POD, request=request),
        partial(adapter.start_trainer, _POD),
        partial(adapter.await_exported_handoff, _POD),
    ):
        with pytest.raises(ValueError):
            premature()
    assert transport.calls == []
