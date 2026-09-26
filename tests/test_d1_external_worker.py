"""Real v2 worker core re-earns D1 analysis authority from mounted evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple, cast

import pytest

import entropic_substrate.analysis.d1_external_analysis_control as control_module
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
    decode_d1_external_analysis_input_control,
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
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.evidence.d1_capture_aggregate_codec import (
    serialize_d1_capture_aggregate,
)
from entropic_substrate.evidence.d1_capture_ports import D1DvcPointerIdentity
from entropic_substrate.evidence.d1_capture_service import D1HeterogeneousCaptureReceipt
from entropic_substrate.evidence.d1_telemetry_values import D1RunEvidence
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


def test_d1_reader_source_closure_and_its_control_symbol_are_gone() -> None:
    """Guard-teeth (brief table row 5): standing a hardcoded identity tuple
    in for the retired ``D1_READER_SOURCE_MODULES`` symbol, rather than
    leaving it and the closure module gone outright, must turn this test
    RED -- the ``hasattr`` form catches a hardcoded stand-in the way a value
    equality check would not (any tuple, however constructed, would satisfy
    an equality assertion against the old 74-entry set)."""
    assert importlib.util.find_spec("entropic_substrate.analysis.d1_reader_source_closure") is None
    assert not hasattr(control_module, "D1_READER_SOURCE_MODULES")


# ---------------------------------------------------------------------------
# Guard-teeth (brief table row 1): the retired closure's former call path was
# d1_external_analysis_control.py:100, which called it at MODULE IMPORT; the
# two review-ring modules are its only importers. ``_load_relocated_copy``
# below is a W1 sibling copy of the ~18-line helper defined at
# ``tests/test_worker_source_anchor.py:52`` (that module is out of this
# phase's Allowed files and reserved for FOR-1519, so it is copied rather
# than imported, per the design gate).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTROL_SOURCE = (
    _REPO_ROOT
    / "src"
    / "entropic_substrate"
    / "__legacy"
    / "analysis"
    / "d1_external_analysis_control.py"
)
_WORKER_SOURCE = (
    _REPO_ROOT / "src" / "entropic_substrate" / "__legacy" / "analysis" / "d1_external_worker.py"
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
def _relocated_review_ring(tmp_path: Path) -> Iterator[tuple[ModuleType, ModuleType]]:
    control_name = "test_relocated_d1_external_analysis_control"
    worker_name = "test_relocated_d1_external_worker"
    try:
        elsewhere = tmp_path / "elsewhere"
        control = _load_relocated_copy(_CONTROL_SOURCE, elsewhere, control_name)
        worker = _load_relocated_copy(_WORKER_SOURCE, elsewhere, worker_name)
        yield control, worker
    finally:
        sys.modules.pop(control_name, None)
        sys.modules.pop(worker_name, None)


def test_review_ring_imports_from_a_relocated_tree(
    _relocated_review_ring: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    """Pin: the review ring (``d1_external_analysis_control.py`` and
    ``d1_external_worker.py``, the closure's only two importers before its
    retirement) imports cleanly from a relocated copy, and the mounted-root
    reality check that took over the closure's former call path
    (``_real_directory``, see the RETRACTED comment in
    ``d1_external_worker.py``) is anchored on the PATH ARGUMENT it receives,
    never on the depth of the file that happens to define it.

    Guard-teeth MEASURED 2026-09-20 (mutation performed and reverted
    in-branch, evidence in the PR body): adding, at module scope near the
    closure's former call site (``d1_external_analysis_control.py:100``),
    ``_ANCHOR = Path(__file__).resolve().parents[2]`` followed by
    ``if _ANCHOR.name != "src": raise RuntimeError("anchor")`` turns this
    test RED with that ``RuntimeError`` (the real installed module's own
    ``parents[2]`` is the repo's ``src/`` directory, so the check passes for
    it, but the relocated copy's ``parents[2]`` is pytest's own base tmp
    directory, named after the test run rather than ``src``); reverting
    turns it back GREEN."""
    control, worker = _relocated_review_ring
    assert control.__name__ != "entropic_substrate.analysis.d1_external_analysis_control"
    assert worker.__name__ != "entropic_substrate.analysis.d1_external_worker"
    real_dir = tmp_path / "real-reader-root"
    real_dir.mkdir()
    resolved = worker._real_directory(real_dir, code="reader_root")
    assert resolved == real_dir.resolve()


class _WorkerCase(NamedTuple):
    request: D1ExternalAnalysisRequestEnvelopeV2
    capture_root: Path
    reader_root: Path
    aggregate: D1HeterogeneousCaptureAggregate
    runs: dict[str, D1RunEvidence]


# RETRACTED (house rule 4, measured 2026-09-20): this file previously built
# a "reader_source_manifest_base64" envelope field here (`_reader_manifest`)
# and populated the mounted reader tree from the installed D1 reader-source
# closure (`_mounted_reader_root`), both retired this phase. Nothing verifies
# the mounted tree's contents any more (see the retraction comment in
# analysis/d1_external_worker.py), so `_mounted_reader_root` now only needs
# to hand back an existing real directory for `run_d1_external_worker`'s
# `_real_directory` existence check.
def _mounted_reader_root(tmp_path: Path) -> Path:
    mounted = tmp_path / "reader"
    mounted.mkdir(parents=True, exist_ok=True)
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


def test_worker_rejects_live_tracking_and_isolation_drift(
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


def test_worker_rejects_missing_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _request_case(tmp_path)
    _install_reader_fake(monkeypatch, case.runs)
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


def test_worker_rejects_symlinked_reader_root(
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
    assert root_error.value.code == "reader_root"


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


def test_worker_rejects_ambiguous_family(
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
