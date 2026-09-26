"""Real-arm evidence assembles one exact thirteen-role filing preparation."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

import entropic_substrate.ladder.file.d1_filing_application as filing_module
from entropic_substrate.kernel.identity.d1_primitives import canonical_json
from entropic_substrate.kernel.lifecycle import ActorRole
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import D1ProcessState
from entropic_substrate.ladder.file.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    validate_d1_filing_manifest,
)
from entropic_substrate.ladder.file.d1_filing_application import (
    D1FilingArtifactPaths,
    D1LocalFilingArtifactStore,
    finalize_d1_filing,
    prepare_d1_filing_bundle,
)
from entropic_substrate.ladder.file.d1_filing_graph import verify_d1_filing
from entropic_substrate.ladder.file.d1_filing_receipts import D1FilingRevisions

from .test_d1_filing_graph_gate import FakeGit
from .test_d1_process_manager import _actor, _artifact
from .test_d1_process_manager_gate import _adjudicated
from .test_d1_real_arm_verdict_assembly import _lineage
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION

pytestmark = pytest.mark.unit


def _paths() -> D1FilingArtifactPaths:
    return D1FilingArtifactPaths(
        **{role: f"filing/{role}.bin" for role in REQUIRED_D1_FILING_ARTIFACTS}
    )


def test_preparation_derives_exact_thirteen_role_filing_from_attested_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = _lineage(tmp_path, monkeypatch)

    prepared = prepare_d1_filing_bundle(
        verified=lineage.verified,
        analysis_receipt=lineage.analysis_receipt,
        execution=lineage.execution,
        reveal=lineage.reveal,
        adjudication=lineage.adjudication,
        verdict="SUPPORTED",
        narrative_body=b"# D1 stabilized latent verdict\n",
        preregistration_bytes=_RAW_PREREGISTRATION,
        paths=_paths(),
    )

    assert tuple(material.role for material in prepared.materials) == (REQUIRED_D1_FILING_ARTIFACTS)
    assert set(prepared.payloads) == set(REQUIRED_D1_FILING_ARTIFACTS)
    assert prepared.payloads["machine_verdict"] == prepared.verdict.canonical_bytes
    assert prepared.payloads["analysis_receipt"] == (
        lineage.analysis_receipt.canonical_json.encode("utf-8")
    )
    assert prepared.payloads["blinded_analysis"] == lineage.verified.canonical_bytes
    assert prepared.payloads["alias_reveal"] == lineage.reveal.payload
    assert prepared.payloads["reviewer_report"] == lineage.execution.report
    assert b"d1-machine-verdict-sha256:" in prepared.payloads["narrative"]
    assert tuple(prepared.manifest.artifacts) == REQUIRED_D1_FILING_ARTIFACTS


def test_preparation_refuses_preregistration_or_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = _lineage(tmp_path, monkeypatch)
    kwargs = {
        "verified": lineage.verified,
        "analysis_receipt": lineage.analysis_receipt,
        "execution": lineage.execution,
        "reveal": lineage.reveal,
        "adjudication": lineage.adjudication,
        "verdict": "SUPPORTED",
        "narrative_body": b"# D1 stabilized latent verdict\n",
        "preregistration_bytes": _RAW_PREREGISTRATION,
        "paths": _paths(),
    }

    with pytest.raises(ValueError, match="preregistration"):
        prepare_d1_filing_bundle(**{**kwargs, "preregistration_bytes": b"substituted"})
    receipt_manifest = lineage.analysis_receipt.manifest
    implementation = receipt_manifest["implementation"]
    assert isinstance(implementation, dict)
    implementation["repository_digest"] = "0" * 64
    with pytest.raises(ValueError, match="reviewed packet"):
        prepare_d1_filing_bundle(
            **{
                **kwargs,
                "analysis_receipt": replace(
                    lineage.analysis_receipt,
                    canonical_json=canonical_json(receipt_manifest),
                ),
            }
        )
    with pytest.raises(ValueError, match="duplicate"):
        prepare_d1_filing_bundle(
            **{
                **kwargs,
                "paths": replace(
                    _paths(),
                    machine_verdict=_paths().narrative,
                ),
            }
        )


def test_local_store_publishes_payloads_then_manifest_as_restartable_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = _lineage(tmp_path / "lineage", monkeypatch)
    prepared = prepare_d1_filing_bundle(
        verified=lineage.verified,
        analysis_receipt=lineage.analysis_receipt,
        execution=lineage.execution,
        reveal=lineage.reveal,
        adjudication=lineage.adjudication,
        verdict="SUPPORTED",
        narrative_body=b"# D1 stabilized latent verdict\n",
        preregistration_bytes=_RAW_PREREGISTRATION,
        paths=_paths(),
    )
    root = tmp_path / "artifact-root"
    (root / "filing").mkdir(parents=True)
    store = D1LocalFilingArtifactStore(root)

    marker = store.publish(prepared, manifest_path="filing/manifest.json")

    assert marker.path == str(root / "filing/manifest.json")
    assert store.read_bytes("filing/manifest.json") == prepared.manifest_bytes
    assert all(
        store.read_bytes(prepared.paths.mapping[role]) == payload
        for role, payload in prepared.payloads.items()
    )
    assert store.publish(prepared, manifest_path="filing/manifest.json") == marker
    (root / prepared.paths.narrative).write_bytes(b"drift")
    with pytest.raises(ValueError, match="differ"):
        store.publish(prepared, manifest_path="filing/manifest.json")


def test_local_store_missing_artifact_is_explicit_none(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    assert D1LocalFilingArtifactStore(tmp_path).read_bytes("filing/missing.json") is None


def test_local_store_failure_before_manifest_marker_is_restartable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = _lineage(tmp_path / "lineage", monkeypatch)
    prepared = prepare_d1_filing_bundle(
        verified=lineage.verified,
        analysis_receipt=lineage.analysis_receipt,
        execution=lineage.execution,
        reveal=lineage.reveal,
        adjudication=lineage.adjudication,
        verdict="SUPPORTED",
        narrative_body=b"# D1 stabilized latent verdict\n",
        preregistration_bytes=_RAW_PREREGISTRATION,
        paths=_paths(),
    )
    root = tmp_path / "artifact-root"
    (root / "filing").mkdir(parents=True)
    store = D1LocalFilingArtifactStore(root)
    real_publish = filing_module.publish_d1_atomic_bytes

    def fail_marker(*args: object, **kwargs: object):
        if kwargs.get("label") == "D1 filing manifest":
            raise ValueError("injected marker failure")
        return real_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(filing_module, "publish_d1_atomic_bytes", fail_marker)
    with pytest.raises(ValueError, match="marker failure"):
        store.publish(prepared, manifest_path="filing/manifest.json")
    assert store.read_bytes("filing/manifest.json") is None
    assert all(
        store.read_bytes(prepared.paths.mapping[role]) == payload
        for role, payload in prepared.payloads.items()
    )

    monkeypatch.setattr(filing_module, "publish_d1_atomic_bytes", real_publish)
    marker = store.publish(prepared, manifest_path="filing/manifest.json")
    assert marker.path == str(root / "filing/manifest.json")


def _filing_git(prepared, revisions: D1FilingRevisions) -> FakeGit:
    verdict = prepared.verdict.verdict
    chain = (verdict.preregistration.commit, *revisions.ordered)
    git = FakeGit({}, set(pairwise(chain)))
    paths = prepared.paths.mapping
    payloads = prepared.payloads
    git.files[(verdict.preregistration.commit, verdict.preregistration.path)] = payloads[
        "preregistration"
    ]
    stages = {
        revisions.capture: ("raw_capture_manifest",),
        revisions.analysis: (
            "blinded_analysis",
            "analysis_receipt",
            "analysis_isolation_attestation",
        ),
        revisions.alias_commitment: ("alias_commitment",),
        revisions.review: (
            "review_packet",
            "reviewer_prompt",
            "reviewer_report",
            "reviewer_execution_receipt",
        ),
        revisions.alias_reveal: ("alias_reveal",),
        revisions.filing: REQUIRED_D1_FILING_ARTIFACTS,
    }
    for revision, roles in stages.items():
        for role in roles:
            git.files[(revision, paths[role])] = payloads[role]
    return git


def test_finalization_verifies_root_git_files_verdict_and_authorized_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = _lineage(tmp_path / "lineage", monkeypatch)
    prepared = prepare_d1_filing_bundle(
        verified=lineage.verified,
        analysis_receipt=lineage.analysis_receipt,
        execution=lineage.execution,
        reveal=lineage.reveal,
        adjudication=lineage.adjudication,
        verdict="SUPPORTED",
        narrative_body=b"# D1 stabilized latent verdict\n",
        preregistration_bytes=_RAW_PREREGISTRATION,
        paths=_paths(),
    )
    root = tmp_path / "artifact-root"
    (root / "filing").mkdir(parents=True)
    store = D1LocalFilingArtifactStore(root)
    store.publish(prepared, manifest_path="filing/manifest.json")
    revisions = D1FilingRevisions(
        "b" * 40,
        "c" * 40,
        "d" * 40,
        "e" * 40,
        "1" * 40,
        "2" * 40,
    )
    git = _filing_git(prepared, revisions)
    detached = validate_d1_filing_manifest(
        prepared.verdict.manifest,
        json.loads(prepared.manifest_bytes),
        decision_contract=prepared.verdict.verified.expected.decision_contract,
        artifact_root=root,
    )
    filing = verify_d1_filing(
        detached,
        prepared.verdict.verified.aggregate,
        revisions,
        store,
        git,
    )
    manager = _adjudicated(tmp_path, filing)

    result = finalize_d1_filing(
        prepared=prepared,
        artifact_root=root,
        manifest_path="filing/manifest.json",
        revisions=revisions,
        git=git,
        manager=manager,
        actor=_actor(ActorRole.OPERATOR),
        filing_event_id="verdict",
        gate_artifact=_artifact(140, "gate"),
        gate_event_id="gate",
    )

    assert result.filing.graph_digest == filing.graph_digest
    assert result.terminal.state is D1ProcessState.GATE_AUTHORIZED
    assert result.terminal.eligible_arms == ("sg", "ac", "both")


def test_finalization_requires_manifest_marker_before_lifecycle_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = _lineage(tmp_path / "lineage", monkeypatch)
    prepared = prepare_d1_filing_bundle(
        verified=lineage.verified,
        analysis_receipt=lineage.analysis_receipt,
        execution=lineage.execution,
        reveal=lineage.reveal,
        adjudication=lineage.adjudication,
        verdict="SUPPORTED",
        narrative_body=b"# D1 stabilized latent verdict\n",
        preregistration_bytes=_RAW_PREREGISTRATION,
        paths=_paths(),
    )
    root = tmp_path / "artifact-root"
    (root / "filing").mkdir(parents=True)
    revisions = D1FilingRevisions(
        "b" * 40,
        "c" * 40,
        "d" * 40,
        "e" * 40,
        "1" * 40,
        "2" * 40,
    )
    git = _filing_git(prepared, revisions)
    store = D1LocalFilingArtifactStore(root)
    store.publish(prepared, manifest_path="filing/manifest.json")
    detached = validate_d1_filing_manifest(
        prepared.verdict.manifest,
        json.loads(prepared.manifest_bytes),
        decision_contract=prepared.verdict.verified.expected.decision_contract,
        artifact_root=root,
    )
    filing = verify_d1_filing(
        detached,
        prepared.verdict.verified.aggregate,
        revisions,
        store,
        git,
    )
    manager = _adjudicated(tmp_path, filing)
    (root / "filing/manifest.json").unlink()

    with pytest.raises(ValueError, match="manifest marker"):
        finalize_d1_filing(
            prepared=prepared,
            artifact_root=root,
            manifest_path="filing/manifest.json",
            revisions=revisions,
            git=git,
            manager=manager,
            actor=_actor(ActorRole.OPERATOR),
            filing_event_id="verdict",
            gate_artifact=_artifact(141, "gate"),
            gate_event_id="gate",
        )
    assert manager.state is D1ProcessState.ADJUDICATED
