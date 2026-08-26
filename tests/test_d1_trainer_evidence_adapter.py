"""Raw process transport cannot mint D1 trainer evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

from d1_resume_fixtures import resume_start
from d1_trainer_fixtures import ORIGIN_NS, micro_plan, run_start
from entropic_substrate.d1_cli import D1CliCommand, D1CliInvocation
from entropic_substrate.d1_trainer_cli_services import D1FreshTrainerCliService
from entropic_substrate.d1_trainer_evidence_adapter import D1TrainerEvidenceAdapter
from entropic_substrate.d1_trainer_process_contracts import (
    D1TrainerAdapterPhase,
    D1TrainerArtifactReadback,
    d1_trainer_request_path,
)
from entropic_substrate.d1_trainer_request_codec import (
    D1TrainerRequest,
    d1_trainer_request_sha256,
    serialize_d1_trainer_request,
)
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.experiments.d1_plan import SmokeD1RunPlan
from entropic_substrate.experiments.d1_primitives import D1Arm

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
    calls: list[str] = field(default_factory=list)
    payload: bytes | None = None
    request_path: str | None = None
    request: D1TrainerRequest | None = None

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
        value = json.loads(self.payload)
        self.request = _request(self.plan, self.start)
        if value["command"] == "d1-run-fresh":
            (self.root / self.request.locations.journal_path).parent.mkdir(parents=True)
            path = self.root / "request.json"
            path.write_bytes(self.payload)
            result = D1FreshTrainerCliService(
                catalog=(self.plan.profile,),
                artifact_root=self.root,
                clock=_Clock(self.start.trainer_started_at),
            ).execute(D1CliInvocation(D1CliCommand.RUN_FRESH, path, value))
            assert result["handoff_kind"] == "fresh_completion"

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
    plan, start = micro_plan(D1Arm.BOTH), run_start(micro_plan(D1Arm.BOTH))
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
    assert adapter.phase(_POD) is D1TrainerAdapterPhase.EXPORT_ADMITTED_UNBOUND
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


@pytest.mark.parametrize(
    "drift",
    "path foreign_path absolute_path traversal_path byte schema plan start origin location".split(),
)
def test_request_readback_drift_latches_stage(tmp_path: Path, drift: str) -> None:
    plan, start = micro_plan(D1Arm.NONE), run_start(micro_plan(D1Arm.NONE))
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
    plan, start = micro_plan(D1Arm.NONE), run_start(micro_plan(D1Arm.NONE))
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
    plan, start = micro_plan(D1Arm.NONE), run_start(micro_plan(D1Arm.NONE))
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
