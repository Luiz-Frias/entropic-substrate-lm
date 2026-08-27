"""Real-arm evidence assembles one exact fourteen-role filing preparation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.d1_filing_application as filing_module
from entropic_substrate.d1_filing_application import (
    D1FilingArtifactPaths,
    D1LocalFilingArtifactStore,
    prepare_d1_filing_bundle,
)
from entropic_substrate.evidence.d1_filing import REQUIRED_D1_FILING_ARTIFACTS
from entropic_substrate.experiments.d1_primitives import canonical_json

from .test_d1_real_arm_verdict_assembly import _lineage
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION

pytestmark = pytest.mark.unit


def _paths() -> D1FilingArtifactPaths:
    return D1FilingArtifactPaths(
        **{role: f"filing/{role}.bin" for role in REQUIRED_D1_FILING_ARTIFACTS}
    )


def test_preparation_derives_exact_fourteen_role_filing_from_attested_lineage(
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
