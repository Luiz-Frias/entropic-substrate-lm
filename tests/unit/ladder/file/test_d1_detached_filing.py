"""Detached D1 filing manifests prove bytes, never successor authority."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from d1_verdict_v2_fixtures import decision_contract, machine_verdict
from entropic_substrate.ladder.file.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    D1DetachedFiling,
    D1FilingValidationError,
    canonical_machine_verdict_bytes,
    validate_d1_filing_manifest,
)

pytestmark = pytest.mark.unit


def _record(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        "size": len(payload),
    }


def _filing_manifest(root: Path, machine: dict[str, object]) -> dict[str, object]:
    contents = {
        role: (
            canonical_machine_verdict_bytes(machine) if role == "machine_verdict" else role.encode()
        )
        for role in REQUIRED_D1_FILING_ARTIFACTS
    }
    artifacts: dict[str, dict[str, object]] = {}
    for index, role in enumerate(REQUIRED_D1_FILING_ARTIFACTS):
        relative = f"filing/{index:02d}-{role}.bin"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents[role])
        artifacts[role] = _record(relative, contents[role])
    return {"schema_version": "d1-filing-manifest/v2", "rung": "d1", "artifacts": artifacts}


def test_rehashed_manifest_is_detached_integrity_evidence_only(tmp_path: Path) -> None:
    machine = machine_verdict()
    detached = validate_d1_filing_manifest(
        machine,
        _filing_manifest(tmp_path, machine),
        decision_contract=decision_contract(),
        artifact_root=tmp_path,
    )

    assert isinstance(detached, D1DetachedFiling)
    assert detached.artifact_root_verified
    assert not hasattr(detached, "to_filed_evidence")


@pytest.mark.parametrize("mutator", ("traversal", "duplicate", "missing", "sha", "md5", "size"))
def test_rehash_rejects_unsafe_or_drifted_artifacts(tmp_path: Path, mutator: str) -> None:
    machine = machine_verdict()
    manifest = _filing_manifest(tmp_path, machine)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    if mutator == "traversal":
        artifacts["narrative"]["path"] = "../escape.md"
    elif mutator == "duplicate":
        artifacts["narrative"]["path"] = artifacts["machine_verdict"]["path"]
    elif mutator == "missing":
        (tmp_path / artifacts["narrative"]["path"]).unlink()
    elif mutator == "sha":
        artifacts["narrative"]["sha256"] = "0" * 64
    elif mutator == "md5":
        artifacts["narrative"]["md5"] = "0" * 32
    elif mutator == "size":
        artifacts["narrative"]["size"] = 0
    with pytest.raises(D1FilingValidationError):
        validate_d1_filing_manifest(
            machine,
            manifest,
            decision_contract=decision_contract(),
            artifact_root=tmp_path,
        )


def test_manifest_requires_every_semantic_receipt_role_once(tmp_path: Path) -> None:
    machine = machine_verdict()
    manifest = _filing_manifest(tmp_path, machine)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    del artifacts["analysis_receipt"]

    with pytest.raises(D1FilingValidationError, match="artifact_role_set"):
        validate_d1_filing_manifest(machine, manifest, decision_contract=decision_contract())
