"""V2 Docker adapter binds worker transport to strict final analysis authority."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import pytest

import entropic_substrate.analysis.d1_execution_receipt as execution_receipt_module
from entropic_substrate.analysis.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.analysis.d1_execution_receipt import (
    D1ExecutorKind,
    D1ExternalIsolationAttestation,
    D1IsolationRequest,
    issue_d1_analysis_receipt,
    issue_d1_verified_analysis_receipt,
)
from entropic_substrate.analysis.d1_external_analysis_adapter import (
    D1ExternalAnalysisCommandResult,
    D1ExternalAnalysisLaunch,
    D1ExternalAnalysisRunnerRequestV2,
    DockerD1ExternalAnalysisAdapterV2,
)
from entropic_substrate.analysis.d1_external_analysis_codec import (
    encode_d1_external_analysis_worker_response_v2,
)
from entropic_substrate.analysis.d1_external_worker import run_d1_external_worker

from .test_d1_analysis_receipt import _implementation
from .test_d1_external_worker import _install_reader_fake, _request_case, _WorkerCase
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION

pytestmark = pytest.mark.unit

_IMAGE = "registry.example/entropic-substrate-lm@sha256:" + "a" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


@dataclass
class _OutputAccessFake:
    active: bool = False
    events: list[tuple[str, Path, int, int]] = field(default_factory=list)

    def grant(self, path: Path, *, uid: int, gid: int) -> None:
        assert path.stat().st_mode & 0o777 == 0o700
        assert not self.active
        self.active = True
        self.events.append(("grant", path, uid, gid))

    def revoke(self, path: Path, *, uid: int, gid: int) -> None:
        assert self.active
        self.active = False
        self.events.append(("revoke", path, uid, gid))


@dataclass
class _RunnerV2:
    launch: D1ExternalAnalysisLaunch
    output_access: _OutputAccessFake
    returncode: int = 0
    argv_mutation: tuple[str, ...] = ()
    response_mutation: dict[str, str] = field(default_factory=dict)
    analysis_mutation: Callable[[dict[str, object]], None] | None = None
    symlink_analysis: bool = False
    extra_output: bool = False
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None,
        stdin: str | None,
    ) -> D1ExternalAnalysisCommandResult:
        assert cwd is None
        assert stdin is None
        assert self.output_access.active
        assert self.launch.output_path.stat().st_mode & 0o777 == 0o700
        self.calls.append(argv)
        artifacts = run_d1_external_worker(
            request_payload=self.launch.request_path.read_bytes(),
            preregistration_payload=self.launch.preregistration_path.read_bytes(),
            capture_root=self.launch.capture_path,
            reader_root=self.launch.reader_path,
            environ={},
        )
        analysis = artifacts.analysis_bytes
        if self.analysis_mutation is not None:
            manifest = json.loads(analysis)
            self.analysis_mutation(manifest)
            analysis = _canonical(manifest)
        response = replace(
            artifacts.response,
            analysis_sha256=hashlib.sha256(analysis).hexdigest(),
        )
        for field_name, value in self.response_mutation.items():
            response = replace(response, **{field_name: value})
        self.launch.analysis_path.write_bytes(analysis)
        self.launch.worker_response_path.write_bytes(
            encode_d1_external_analysis_worker_response_v2(response)
        )
        if self.symlink_analysis:
            target = self.launch.output_path.parent / "elsewhere-analysis.json"
            target.write_bytes(analysis)
            self.launch.analysis_path.unlink()
            self.launch.analysis_path.symlink_to(target)
        if self.extra_output:
            (self.launch.output_path / "unexpected.json").write_bytes(b"{}")
        return D1ExternalAnalysisCommandResult(
            invoked_argv=argv + self.argv_mutation,
            returncode=self.returncode,
            stdout="worker stdout",
            stderr="worker stderr",
        )


type _AdapterCase = tuple[
    _WorkerCase,
    D1ExternalAnalysisLaunch,
    _RunnerV2,
    D1ExternalAnalysisRunnerRequestV2,
]


def _adapter_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **options: object,
) -> _AdapterCase:
    case = _request_case(tmp_path / "case")
    _install_reader_fake(monkeypatch, case.runs)
    preregistration = tmp_path / "preregistration"
    preregistration.write_bytes(_RAW_PREREGISTRATION)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    launch = D1ExternalAnalysisLaunch(
        image_reference=_IMAGE,
        capture_path=case.capture_root,
        preregistration_path=preregistration,
        reader_path=case.reader_root,
        output_path=output,
    )
    output_access = _OutputAccessFake()
    runner = _RunnerV2(launch, output_access, **options)
    request = D1ExternalAnalysisRunnerRequestV2(
        envelope=case.request,
        aggregate=case.aggregate,
        isolation=D1IsolationRequest(),
    )
    return case, launch, runner, request


def _adapter(
    launch: D1ExternalAnalysisLaunch,
    runner: _RunnerV2,
) -> DockerD1ExternalAnalysisAdapterV2:
    return DockerD1ExternalAnalysisAdapterV2(
        launch=launch,
        runner=runner,
        output_access=runner.output_access,
    )


def test_v2_adapter_runs_exact_fixed_command_and_returns_final_typestate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, launch, runner, request = _adapter_case(tmp_path, monkeypatch)

    result = _adapter(launch, runner).run(request)

    assert result.analysis.expected is case.request.input_control
    assert result.analysis.aggregate is case.aggregate
    assert result.worker_response.analysis_sha256 == result.analysis.analysis_digest
    assert result.isolation.executor_kind is D1ExecutorKind.EXTERNAL_ENFORCED
    assert launch.output_path.stat().st_mode & 0o777 == 0o700
    assert runner.output_access.active is False
    assert [event[0] for event in runner.output_access.events] == ["grant", "revoke"]
    assert all(event[2:] == (65532, 65532) for event in runner.output_access.events)
    assert not launch.request_path.exists()
    argv = runner.calls[0]
    assert ("--network", "none") == argv[argv.index("--network") : argv.index("--network") + 2]
    assert ("--user", "65532:65532") == argv[argv.index("--user") : argv.index("--user") + 2]
    for flag, value in (
        ("--platform", "linux/amd64"),
        ("--cpus", "2"),
        ("--memory", "4g"),
        ("--pids-limit", "128"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
    ):
        assert (flag, value) == argv[argv.index(flag) : argv.index(flag) + 2]
    assert "--read-only" in argv
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert len(mounts) == 5
    assert sum("readonly" not in mount for mount in mounts) == 1
    assert "--env" not in argv and "-e" not in argv and "--env-file" not in argv
    assert "execution-receipt.json" not in argv
    assert argv[-1] == "/d1/output/worker-response.json"


def test_enforced_isolation_capability_cannot_be_replayed_or_publicly_minted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, launch, runner, request = _adapter_case(tmp_path, monkeypatch)
    isolation = _adapter(launch, runner).run(request).isolation
    assert isolation.filing_eligible is True

    assert "_verified_external" not in D1ExternalIsolationAttestation.__dict__
    with pytest.raises(TypeError, match="factory-only"):
        replace(isolation, mechanism="forged")
    for copier in (copy, deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="cannot be copied"):
            copier(isolation)

    fields = (
        "request_digest",
        "executor_kind",
        "network_disabled",
        "filesystem_read_only",
        "live_tracking_blocked",
        "mechanism",
        "execution_evidence",
    )
    forged = object.__new__(D1ExternalIsolationAttestation)
    for name in fields:
        object.__setattr__(forged, name, getattr(isolation, name))
    marker = object.__getattribute__(isolation, "_attestation")
    object.__setattr__(forged, "_attestation", marker)
    assert forged.filing_eligible is False
    with pytest.raises(TypeError, match="adapter-minted"):
        forged.require_factory_attested()

    self_owned = object.__new__(D1ExternalIsolationAttestation)
    for name in fields:
        object.__setattr__(self_owned, name, getattr(isolation, name))
    object.__setattr__(
        self_owned,
        "_attestation",
        type(marker)(self_owned, marker.manifest_digest),
    )
    assert self_owned.filing_eligible is False

    with pytest.raises(TypeError, match="factory-only"):
        D1ExternalIsolationAttestation(
            isolation.request_digest,
            D1ExecutorKind.EXTERNAL_ENFORCED,
            True,
            True,
            True,
            "forged",
            isolation.execution_evidence,
        )
    isolation.require_factory_attested()
    assert isolation.filing_eligible is True

    object.__setattr__(isolation, "mechanism", "forged")
    assert isolation.filing_eligible is False
    with pytest.raises(TypeError, match="changed after minting"):
        isolation.require_factory_attested()

    monkeypatch.undo()
    _, fresh_launch, fresh_runner, fresh_request = _adapter_case(
        tmp_path / "fresh",
        monkeypatch,
    )
    fresh = _adapter(fresh_launch, fresh_runner).run(fresh_request).isolation
    fresh.require_factory_attested()
    assert fresh.filing_eligible is True


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("analysis_request_digest", "1" * 64),
        ("isolation_request_digest", "2" * 64),
        ("input_manifest_sha256", "3" * 64),
        ("analysis_sha256", "4" * 64),
    ],
)
def test_v2_adapter_rejects_each_worker_response_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    value: str,
) -> None:
    _, launch, runner, request = _adapter_case(
        tmp_path,
        monkeypatch,
        response_mutation={field_name: value},
    )
    with pytest.raises(ValueError):
        _adapter(launch, runner).run(request)
    assert launch.output_path.stat().st_mode & 0o777 == 0o700


def test_v2_adapter_requires_strict_final_machine_analysis_typestate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(root: dict[str, object]) -> None:
        diagnostics = cast(list[object], root["diagnostics"])
        diagnostic = cast(dict[str, object], diagnostics[0])
        target = cast(dict[str, object], diagnostic["quality_target"])
        target["threshold"] = 0.5

    _, launch, runner, request = _adapter_case(
        tmp_path,
        monkeypatch,
        analysis_mutation=mutate,
    )
    with pytest.raises(ValueError, match="quality target"):
        _adapter(launch, runner).run(request)


def test_v2_adapter_restores_output_permissions_after_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, launch, runner, request = _adapter_case(
        tmp_path,
        monkeypatch,
        returncode=9,
    )
    with pytest.raises(RuntimeError, match="nonzero"):
        _adapter(launch, runner).run(request)
    assert launch.output_path.stat().st_mode & 0o777 == 0o700
    assert runner.output_access.active is False


def test_v2_adapter_rejects_regular_file_reader_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, launch, runner, request = _adapter_case(tmp_path, monkeypatch)
    reader_file = tmp_path / "reader-file"
    reader_file.write_bytes(b"reader")
    file_launch = replace(launch, reader_path=reader_file)

    with pytest.raises(ValueError, match="reader_path"):
        _adapter(file_launch, runner).run(request)


def test_v2_receipt_binds_request_worker_analysis_and_execution_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, launch, runner, request = _adapter_case(tmp_path, monkeypatch)
    result = _adapter(launch, runner).run(request)

    receipt = issue_d1_verified_analysis_receipt(
        analysis=result.analysis,
        request=result.request,
        worker_response=result.worker_response,
        implementation=_implementation(result.request.input_control.reader_source),
        isolation=result.isolation,
    )

    assert receipt.manifest["filing_eligible"] is True
    assert receipt.manifest["analysis_digest"] == result.analysis.analysis_digest
    assert receipt.manifest["filing_bindings"] == {
        "capture_manifest_sha256": hashlib.sha256(
            result.request.input_control.aggregate.canonical_json.encode("utf-8")
        ).hexdigest(),
        "isolation_attestation_sha256": hashlib.sha256(
            json.dumps(
                result.isolation.manifest,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "output_sha256": result.analysis.analysis_digest,
        "preregistration_sha256": result.request.input_control.preregistration.sha256,
        "reader_sha256": hashlib.sha256(
            result.request.input_control.reader_source.canonical_bytes
        ).hexdigest(),
    }


def test_v2_receipt_rejects_implementation_reader_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, launch, runner, request = _adapter_case(tmp_path, monkeypatch)
    result = _adapter(launch, runner).run(request)
    implementation = _implementation(result.request.input_control.reader_source)

    with pytest.raises(ValueError, match="complete reader digest differs"):
        issue_d1_verified_analysis_receipt(
            analysis=result.analysis,
            request=result.request,
            worker_response=result.worker_response,
            implementation=replace(implementation, complete_reader_digest="f" * 64),
            isolation=result.isolation,
        )


def test_v2_receipt_rejects_generic_issuer_and_worker_or_execution_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, launch, runner, request = _adapter_case(tmp_path, monkeypatch)
    result = _adapter(launch, runner).run(request)
    with pytest.raises(ValueError, match="verified_analysis_receipt"):
        issue_d1_analysis_receipt(
            analysis=result.analysis.analysis,
            admitted=D1AdmittedAnalysisCapture(
                design_digest=result.request.input_control.design.design_digest,
                decision_contract_digest=(
                    result.request.input_control.design.decision_contract_digest
                ),
                capture_digest=result.analysis.aggregate.manifest_digest,
                cells=(),
                exclusions=(),
                runs=(),
                input_artifacts=result.analysis.aggregate.dvc_handoff.artifacts,
            ),
            aliases=result.request.input_control.alias_application.aliases,
            implementation=_implementation(),
            isolation=result.isolation,
        )
    for field_name, value, message in (
        ("analysis_request_digest", "1" * 64, "worker response differs from request"),
        ("isolation_request_digest", "2" * 64, "worker response differs from isolation"),
        ("input_manifest_sha256", "3" * 64, "worker response differs from input manifest"),
        ("analysis_sha256", "4" * 64, "worker response differs from analysis"),
    ):
        drifted = replace(result.worker_response, **{field_name: value})
        with pytest.raises(ValueError, match=message):
            issue_d1_verified_analysis_receipt(
                analysis=result.analysis,
                request=result.request,
                worker_response=drifted,
                implementation=_implementation(result.request.input_control.reader_source),
                isolation=result.isolation,
            )

    aggregate = result.analysis.aggregate
    object.__setattr__(
        result.analysis,
        "aggregate",
        replace(aggregate, complete_runs=tuple(reversed(aggregate.complete_runs))),
    )
    with pytest.raises(ValueError, match="aggregate differs from request input"):
        issue_d1_verified_analysis_receipt(
            analysis=result.analysis,
            request=result.request,
            worker_response=result.worker_response,
            implementation=_implementation(result.request.input_control.reader_source),
            isolation=result.isolation,
        )
    object.__setattr__(result.analysis, "aggregate", aggregate)

    monkeypatch.undo()
    _, other_launch, other_runner, other_request = _adapter_case(
        tmp_path / "other",
        monkeypatch,
    )
    other = _adapter(other_launch, other_runner).run(other_request)
    with pytest.raises(ValueError, match="request input"):
        issue_d1_verified_analysis_receipt(
            analysis=result.analysis,
            request=other.request,
            worker_response=result.worker_response,
            implementation=_implementation(other.request.input_control.reader_source),
            isolation=result.isolation,
        )

    object.__setattr__(result.isolation, "request_digest", "0" * 64)
    with pytest.raises(ValueError, match="receipt isolation differs from request"):
        issue_d1_verified_analysis_receipt(
            analysis=result.analysis,
            request=result.request,
            worker_response=result.worker_response,
            implementation=_implementation(result.request.input_control.reader_source),
            isolation=result.isolation,
        )
    object.__setattr__(
        result.isolation,
        "request_digest",
        result.request.isolation_request_digest,
    )
    execution_evidence = result.isolation.execution_evidence
    assert execution_evidence is not None
    object.__setattr__(
        result.isolation,
        "execution_evidence",
        replace(execution_evidence, request_sha256="5" * 64),
    )
    with pytest.raises(TypeError, match="changed after minting"):
        issue_d1_verified_analysis_receipt(
            analysis=result.analysis,
            request=result.request,
            worker_response=result.worker_response,
            implementation=_implementation(result.request.input_control.reader_source),
            isolation=result.isolation,
        )


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("request_sha256", "execution request hash differs"),
        ("response_sha256", "execution response hash differs"),
        ("analysis_sha256", "execution analysis hash differs"),
    ],
)
def test_v2_receipt_rejects_each_inconsistent_execution_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    message: str,
) -> None:
    _, launch, runner, request = _adapter_case(tmp_path, monkeypatch)
    result = _adapter(launch, runner).run(request)
    evidence = result.isolation.execution_evidence
    assert evidence is not None
    issuer = cast(
        Callable[..., D1ExternalIsolationAttestation],
        execution_receipt_module.__dict__["_mint_external_isolation"],
    )
    isolation = issuer(
        request=request.isolation,
        mechanism="execution-hash-test",
        execution_evidence=replace(evidence, **{field_name: "5" * 64}),
    )
    with pytest.raises(ValueError, match=message):
        issue_d1_verified_analysis_receipt(
            analysis=result.analysis,
            request=result.request,
            worker_response=result.worker_response,
            implementation=_implementation(result.request.input_control.reader_source),
            isolation=isolation,
        )


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ("registry.example/model:latest", "immutable"),
        ("registry.example/model:tag@sha256:" + "a" * 64, "tag"),
        ("registry.example/model@sha256:not-a-digest", "digest"),
    ],
)
def test_v2_launch_rejects_mutable_or_malformed_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: str,
    message: str,
) -> None:
    _, launch, _, _ = _adapter_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=message):
        replace(launch, image_reference=image)


@pytest.mark.parametrize(
    "mutation",
    [
        ("--network", "bridge"),
        ("--mount", "type=bind,src=/host,dst=/extra"),
        ("--env", "MLFLOW_TRACKING_URI=https://tracking.invalid"),
        ("--read-write",),
    ],
)
def test_v2_adapter_rejects_observed_argv_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, ...],
) -> None:
    _, launch, runner, request = _adapter_case(
        tmp_path,
        monkeypatch,
        argv_mutation=mutation,
    )
    with pytest.raises(RuntimeError, match="argv"):
        _adapter(launch, runner).run(request)


@pytest.mark.parametrize(
    ("option", "error_type", "message"),
    [
        ("symlink_analysis", ValueError, "symlink"),
        ("extra_output", RuntimeError, "unexpected"),
    ],
)
def test_v2_adapter_rejects_symlinked_or_unexpected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    error_type: type[Exception],
    message: str,
) -> None:
    _, launch, runner, request = _adapter_case(
        tmp_path,
        monkeypatch,
        **{option: True},
    )
    with pytest.raises(error_type, match=message):
        _adapter(launch, runner).run(request)
