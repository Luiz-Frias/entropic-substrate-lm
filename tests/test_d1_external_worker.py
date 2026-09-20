"""Real v2 worker core re-earns D1 analysis authority from mounted evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple, cast

import pytest

import entropic_substrate.analysis.d1_external_worker as worker_module
import entropic_substrate.telemetry as telemetry_api
from d1_launch_admission_fixtures import family
from entropic_substrate.analysis.d1_blind_aliases import (
    D1InMemoryAliasCustodyFake,
    prepare_d1_blind_aliases,
)
from entropic_substrate.analysis.d1_execution_receipt import D1IsolationRequest
from entropic_substrate.analysis.d1_external_analysis_codec import (
    D1ExternalAnalysisRequestEnvelopeV2,
    build_d1_external_analysis_request_v2,
    decode_d1_external_analysis_worker_response_v2,
    encode_d1_external_analysis_request_v2,
)
from entropic_substrate.analysis.d1_external_analysis_control import (
    D1_COMPLETE_READER_IDENTITY,
    D1_TERMINAL_DEFECT_READER_IDENTITY,
    decode_d1_external_analysis_input_control,
    derive_d1_analysis_code_digest,
)
from entropic_substrate.analysis.d1_external_worker import (
    D1ExternalWorkerArtifacts,
    D1ExternalWorkerError,
    run_d1_external_worker,
)
from entropic_substrate.analysis.d1_machine_analysis_binding import (
    bind_d1_machine_analysis_contract,
)
from entropic_substrate.analysis.d1_machine_analysis_verification import (
    verify_d1_machine_analysis,
)
from entropic_substrate.analysis.d1_reader_source_closure import (
    installed_d1_reader_source_files,
)
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.evidence.d1_capture_aggregate_codec import (
    serialize_d1_capture_aggregate,
)
from entropic_substrate.evidence.d1_capture_ports import D1DvcPointerIdentity
from entropic_substrate.evidence.d1_capture_service import D1HeterogeneousCaptureReceipt
from entropic_substrate.evidence.d1_telemetry_values import D1RunEvidence
from entropic_substrate.experiments.d1_config import load_d1_profile_catalog
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)
from entropic_substrate.experiments.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    serialize_decision_contract,
    serialize_registered_design,
)

from .test_d1_analysis_api import _decision
from .test_d1_analysis_capture import _analysis_case
from .test_d1_blind_alias_custody import _DeterministicCrypto
from .test_d1_external_analysis_input import _b64, _canonical
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# RED-first relocated-import regression (design-gate constraint 1, planned
# commit branch 1). ``_load_relocated_copy`` below is a W1 sibling copy of
# the ~18-line helper defined at ``tests/test_worker_source_anchor.py:52``
# (that module is out of this phase's Allowed files and reserved for
# FOR-1519, so it is copied here rather than imported, per the design gate).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLOSURE_SOURCE = (
    _REPO_ROOT / "src" / "entropic_substrate" / "analysis" / "d1_reader_source_closure.py"
)


def _load_relocated_copy(source: Path, relocated_dir: Path, module_name: str) -> ModuleType:
    """Copy ``source`` into ``relocated_dir`` and import it under a distinct
    module name, so its own ``__file__`` differs from the installed
    package's copy while its absolute ``from entropic_substrate...`` imports
    still resolve to the real installed package (unaffected by the copy)."""
    relocated_dir.mkdir(parents=True, exist_ok=True)
    target = relocated_dir / source.name
    shutil.copy2(source, target)
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture
def _relocated_closure(tmp_path: Path) -> Iterator[ModuleType]:
    module_name = "test_relocated_d1_reader_source_closure"
    try:
        yield _load_relocated_copy(_CLOSURE_SOURCE, tmp_path / "elsewhere", module_name)
    finally:
        sys.modules.pop(module_name, None)


def test_closure_resolves_the_real_root_from_a_relocated_copy(
    _relocated_closure: ModuleType,
) -> None:
    """Pin: the closure's root resolution is anchored on the INSTALLED
    ``entropic_substrate`` package's own ``__file__`` (via
    ``entropic_substrate.paths.source_root()``), never on the depth of the
    file that happens to call it. Loading a COPY of
    ``d1_reader_source_closure.py`` from an unrelated tmp directory and
    calling ``installed_d1_reader_source_files()`` on that copy proves this
    directly: a depth-derived ``Path(__file__).resolve().parents[2]``
    anchor reintroduced inside the copied module would resolve against the
    tmp location instead of the real repo and raise
    ``D1ReaderSourceClosureError`` here.

    Guard-teeth MEASURED 2026-09-20 (mutation performed and reverted
    in-branch, evidence in the PR body): replacing this module's
    ``source_root()`` import with a local
    ``Path(__file__).resolve().parents[2]`` definition turns this test RED
    with exactly that error; reverting turns it back GREEN. This test is
    retired together with the closure module later in this phase (nothing
    survives to relocate once ``d1_reader_source_closure.py`` is deleted)."""
    files = _relocated_closure.installed_d1_reader_source_files()
    assert len(files) == 74
    assert any(item.identity == "entropic_substrate.telemetry" for item in files)


class _WorkerCase(NamedTuple):
    request: D1ExternalAnalysisRequestEnvelopeV2
    capture_root: Path
    reader_root: Path
    aggregate: D1HeterogeneousCaptureAggregate
    runs: dict[str, D1RunEvidence]


def _reader_manifest() -> dict[str, object]:
    files = installed_d1_reader_source_files()
    modules = [item.manifest for item in files]
    return {
        "analysis_code_digest": derive_d1_analysis_code_digest(modules),
        "complete_reader": D1_COMPLETE_READER_IDENTITY,
        "modules": modules,
        "schema_version": "d1-reader-source-manifest-v1",
        "terminal_defect_reader": D1_TERMINAL_DEFECT_READER_IDENTITY,
    }


def _mounted_reader_root(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1] / "src"
    mounted = tmp_path / "reader"
    for item in installed_d1_reader_source_files():
        target = mounted / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / item.path, target)
    return mounted


def _request_case(tmp_path: Path) -> _WorkerCase:
    tmp_path.mkdir(parents=True, exist_ok=True)
    capture_root, old_design, base, aggregate, runs = _analysis_case(tmp_path)
    normalized_runs: dict[str, D1RunEvidence] = {}
    for logical_run_id, run in runs.items():
        plan = dict(run.plan_manifest)
        plan["preregistration"] = {
            "commit": aggregate.roster.preregistration_commit,
            "family_digest": aggregate.roster.family_digest,
            "path": aggregate.roster.preregistration_path,
        }
        normalized_runs[logical_run_id] = replace(run, plan_manifest=plan)  # type: ignore[arg-type]
    decision = _decision(base)
    evidence = D1PreregistrationByteEvidence.from_bytes(_RAW_PREREGISTRATION)
    design = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=decision,
            registered_seeds=old_design.registered_seeds,
            preregistration=D1PreregistrationArtifact(
                path=old_design.preregistration.path,
                blob_sha256=evidence.sha256,
                commit=old_design.preregistration.commit,
            ),
        )
    )
    assert type(design) is D1RegisteredDesign
    dvc = aggregate.dvc_handoff.artifacts[0].dvc
    receipt = D1HeterogeneousCaptureReceipt(
        aggregate=aggregate,
        pointer=D1DvcPointerIdentity(
            pointer_path=dvc.pointer_path,
            pointer_sha256=str(dvc.pointer_sha256),
            object_id=str(dvc.object_id),
        ),
        pointer_commit="abcdef1",
    )
    custody = D1InMemoryAliasCustodyFake()
    prepare_d1_blind_aliases(randomness=_DeterministicCrypto(), custody=custody)
    preregistration_control = {
        "commit": design.preregistration.commit,
        "path": design.preregistration.path,
        "schema_version": "d1-preregistration-control-v1",
        "sha256": evidence.sha256,
        "size_bytes": evidence.size_bytes,
    }
    manifest = {
        "alias_commitment_base64": _b64(custody.public_bytes),
        "alias_reveal_base64": _b64(custody.private_bytes),
        "capture_aggregate_base64": _b64(serialize_d1_capture_aggregate(aggregate).encode()),
        "capture_receipt_base64": _b64(_canonical(receipt.manifest)),
        "decision_contract_base64": _b64(serialize_decision_contract(decision).encode("utf-8")),
        "expected_capture_digest": aggregate.manifest_digest,
        "expected_design_digest": design.design_digest,
        "expected_family_digest": design.family_digest,
        "preregistration_control_base64": _b64(_canonical(preregistration_control)),
        "reader_source_manifest_base64": _b64(_canonical(_reader_manifest())),
        "registered_design_base64": _b64(serialize_registered_design(design).encode("utf-8")),
        "schema_version": "d1-external-analysis-input-v1",
    }
    control = decode_d1_external_analysis_input_control(
        _canonical(manifest),
        preregistration_evidence=evidence,
    )
    request = build_d1_external_analysis_request_v2(
        input_control=control,
        isolation_request_digest=D1IsolationRequest().request_digest,
    )
    return _WorkerCase(
        request,
        capture_root,
        _mounted_reader_root(tmp_path),
        aggregate,
        normalized_runs,
    )


def _install_reader_fake(
    monkeypatch: pytest.MonkeyPatch,
    runs: dict[str, D1RunEvidence],
) -> None:
    def read(_path: Path, **expected: object) -> D1RunEvidence:
        return runs[str(expected["expected_logical_run_id"])]

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", read)


def _run_case(
    case: _WorkerCase,
    *,
    request: D1ExternalAnalysisRequestEnvelopeV2 | None = None,
    environ: dict[str, str] | None = None,
) -> D1ExternalWorkerArtifacts:
    return run_d1_external_worker(
        request_payload=encode_d1_external_analysis_request_v2(request or case.request),
        preregistration_payload=_RAW_PREREGISTRATION,
        capture_root=case.capture_root,
        reader_root=case.reader_root,
        environ={} if environ is None else environ,
    )


def test_worker_rehydrates_and_composes_blinded_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path)
    _install_reader_fake(monkeypatch, case.runs)

    artifacts = _run_case(case)

    response = decode_d1_external_analysis_worker_response_v2(artifacts.response_bytes)
    assert response.analysis_request_digest == case.request.analysis_request_digest
    assert response.input_manifest_sha256 == case.request.input_manifest_sha256
    assert response.analysis_sha256 == hashlib.sha256(artifacts.analysis_bytes).hexdigest()
    bound = bind_d1_machine_analysis_contract(
        artifacts.analysis_bytes,
        expected=case.request.input_control,
    )
    verified = verify_d1_machine_analysis(binding=bound, aggregate=case.aggregate)
    assert verified.canonical_bytes == artifacts.analysis_bytes
    assert "nonce" not in repr(artifacts)
    reveal = case.request.input_control.alias_application.reveal.canonical_bytes
    assert reveal not in artifacts.analysis_bytes
    assert reveal not in artifacts.response_bytes


def test_worker_rejects_live_tracking_isolation_and_mounted_reader_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path)
    _install_reader_fake(monkeypatch, case.runs)
    sensitive_tracking_uri = "https://tracking.invalid/private"
    with pytest.raises(D1ExternalWorkerError) as tracking_error:
        _run_case(case, environ={"MLFLOW_TRACKING_URI": sensitive_tracking_uri})
    assert tracking_error.value.code == "live_tracking_config"
    assert sensitive_tracking_uri not in str(tracking_error.value)

    accepted = _run_case(case, environ={"MLFLOW_TRACKING_URI": ""})
    assert accepted.response.analysis_sha256 == hashlib.sha256(accepted.analysis_bytes).hexdigest()

    other_isolation = build_d1_external_analysis_request_v2(
        input_control=case.request.input_control,
        isolation_request_digest="1" * 64,
    )
    with pytest.raises(D1ExternalWorkerError, match="isolation"):
        _run_case(case, request=other_isolation)

    first = installed_d1_reader_source_files()[0]
    (case.reader_root / first.path).write_bytes(b"drift")
    with pytest.raises(D1ExternalWorkerError, match="reader_identity"):
        _run_case(case)


def test_worker_request_refusal_does_not_echo_preregistration_bytes(
    tmp_path: Path,
) -> None:
    case = _request_case(tmp_path)
    sensitive_text = "never-echo-this-value"
    with pytest.raises(D1ExternalWorkerError) as raised:
        run_d1_external_worker(
            request_payload=encode_d1_external_analysis_request_v2(case.request),
            preregistration_payload=sensitive_text.encode(),
            capture_root=case.capture_root,
            reader_root=case.reader_root,
            environ={},
        )
    assert raised.value.code == "request"
    assert sensitive_text not in str(raised.value)


def test_worker_rejects_installed_reader_drift_and_missing_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path)
    _install_reader_fake(monkeypatch, case.runs)
    installed = installed_d1_reader_source_files()
    drifted = (replace(installed[0], sha256="0" * 64), *installed[1:])
    monkeypatch.setattr(worker_module, "installed_d1_reader_source_files", lambda: drifted)
    with pytest.raises(D1ExternalWorkerError, match="reader_identity"):
        _run_case(case)

    monkeypatch.setattr(
        worker_module,
        "installed_d1_reader_source_files",
        installed_d1_reader_source_files,
    )
    monkeypatch.setattr(worker_module, "load_d1_profile_catalog", lambda: ())
    with pytest.raises(D1ExternalWorkerError, match="family"):
        _run_case(case)


def test_worker_rejects_symlinked_capture_and_artifact_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path)
    _install_reader_fake(monkeypatch, case.runs)
    capture_link = tmp_path / "capture-link"
    capture_link.symlink_to(case.capture_root, target_is_directory=True)
    with pytest.raises(D1ExternalWorkerError) as capture_error:
        run_d1_external_worker(
            request_payload=encode_d1_external_analysis_request_v2(case.request),
            preregistration_payload=_RAW_PREREGISTRATION,
            capture_root=capture_link,
            reader_root=case.reader_root,
            environ={},
        )
    assert capture_error.value.code == "capture"

    artifacts = _run_case(case)
    with pytest.raises(ValueError, match="analysis artifact digest"):
        replace(artifacts, analysis_sha256="0" * 64)
    with pytest.raises(ValueError, match="response bytes"):
        replace(artifacts, response_bytes=artifacts.response_bytes + b"\n")


def test_worker_rejects_symlinked_reader_root_and_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path)
    _install_reader_fake(monkeypatch, case.runs)
    reader_link = tmp_path / "reader-link"
    reader_link.symlink_to(case.reader_root, target_is_directory=True)
    with pytest.raises(D1ExternalWorkerError) as root_error:
        run_d1_external_worker(
            request_payload=encode_d1_external_analysis_request_v2(case.request),
            preregistration_payload=_RAW_PREREGISTRATION,
            capture_root=case.capture_root,
            reader_root=reader_link,
            environ={},
        )
    assert root_error.value.code == "reader_identity"

    first = installed_d1_reader_source_files()[0]
    mounted_file = case.reader_root / first.path
    source_bytes = mounted_file.read_bytes()
    mounted_file.unlink()
    outside = tmp_path / "reader-outside.py"
    outside.write_bytes(source_bytes)
    mounted_file.symlink_to(outside)
    with pytest.raises(D1ExternalWorkerError) as file_error:
        _run_case(case)
    assert file_error.value.code == "reader_identity"


def test_reader_join_rejects_lexical_parent_components(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    regular_file = cast(
        Callable[[Path, str], Path],
        worker_module.__dict__["_regular_mounted_file"],
    )
    with pytest.raises(D1ExternalWorkerError) as raised:
        regular_file(tmp_path, "nested/../source.py")
    assert raised.value.code == "reader_identity"


@pytest.mark.parametrize("invalid", ([], {1: "value"}, {"KEY": 1}))
def test_worker_environment_type_errors_remain_type_errors(
    tmp_path: Path,
    invalid: object,
) -> None:
    with pytest.raises(TypeError, match="environment"):
        run_d1_external_worker(
            request_payload=b"request",
            preregistration_payload=b"preregistration",
            capture_root=tmp_path,
            reader_root=tmp_path,
            environ=cast(dict[str, str], invalid),
        )


def test_worker_rejects_ambiguous_family_and_intermediate_reader_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path)
    _install_reader_fake(monkeypatch, case.runs)
    digest = case.request.input_control.expected_family_digest
    profiles = (
        SimpleNamespace(family=SimpleNamespace(family_digest=digest, marker=1)),
        SimpleNamespace(family=SimpleNamespace(family_digest=digest, marker=2)),
    )
    monkeypatch.setattr(worker_module, "load_d1_profile_catalog", lambda: profiles)
    with pytest.raises(D1ExternalWorkerError) as family_error:
        _run_case(case)
    assert family_error.value.code == "family"

    monkeypatch.setattr(worker_module, "load_d1_profile_catalog", load_d1_profile_catalog)
    intermediate = case.reader_root / "entropic_substrate/analysis"
    outside = tmp_path / "outside-analysis"
    shutil.copytree(intermediate, outside)
    shutil.rmtree(intermediate)
    intermediate.symlink_to(outside, target_is_directory=True)
    with pytest.raises(D1ExternalWorkerError) as reader_error:
        _run_case(case)
    assert reader_error.value.code == "reader_identity"
