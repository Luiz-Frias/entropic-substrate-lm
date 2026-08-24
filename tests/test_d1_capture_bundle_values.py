"""D1 capture bundle values prevent cross-lane artifact ambiguity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureInput,
    D1CaptureSetProof,
)
from entropic_substrate.evidence.d1_capture_roster import D1AttemptLane, D1DeclaredAttempt
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

pytestmark = pytest.mark.unit


def _artifact(tmp_path: Path, name: str, role: D1ArtifactRole):
    path = tmp_path / name
    path.write_text(name, encoding="utf-8")
    return identify_d1_artifact(
        tmp_path,
        path,
        role=role,
        dvc_pointer_path="results/d1-captured.dvc",
    )


def test_capture_input_requires_exact_roles_billing_and_unique_paths(tmp_path: Path) -> None:
    journal = _artifact(tmp_path, "run.jsonl", D1ArtifactRole.JOURNAL)
    checkpoint = _artifact(tmp_path, "run.pt", D1ArtifactRole.CHECKPOINT)
    billing = _artifact(tmp_path, "run.billing.json", D1ArtifactRole.BILLING_SIDECAR)
    value = D1CaptureInput("run", journal, checkpoint, (billing,), ())

    assert value.artifacts == (journal, checkpoint, billing)
    with pytest.raises(ValueError, match="billing"):
        replace(value, billing_sidecars=())
    with pytest.raises(ValueError, match="role"):
        replace(value, journal=checkpoint)
    duplicate = replace(billing, role=D1ArtifactRole.QUARANTINED_SIBLING)
    with pytest.raises(ValueError, match="unique"):
        replace(value, quarantined_siblings=(duplicate,))


def test_set_proof_makes_all_four_logical_run_sets_equal() -> None:
    expected = frozenset({"a", "b"})
    proof = D1CaptureSetProof(expected, expected, expected, expected)

    assert proof.is_exact is True
    with pytest.raises(ValueError, match="exact set equality"):
        replace(proof, checkpoint_receipt_run_ids=frozenset({"a"}))


def test_abandoned_attempt_requires_explicit_gap_and_other_lanes_require_journal(
    tmp_path: Path,
) -> None:
    abandoned = D1DeclaredAttempt(
        logical_run_id="abandoned",
        arm=D1Arm.NONE,
        seed=7,
        pairing_id="pair-7",
        run_class=RunClass.CONFIRMATORY,
        lane=D1AttemptLane.ABANDONED,
        disposition="pod failed before trainer start",
    )
    value = D1AttemptEvidence(abandoned, (), "no authoritative journal was created")
    assert value.evidence_gap is not None
    with pytest.raises(ValueError, match="gap"):
        replace(value, evidence_gap=None)

    pilot = replace(
        abandoned,
        logical_run_id="pilot",
        run_class=RunClass.PILOT,
        lane=D1AttemptLane.EXCLUDED_PILOT,
    )
    billing = _artifact(tmp_path, "pilot.billing", D1ArtifactRole.BILLING_SIDECAR)
    with pytest.raises(ValueError, match="journal"):
        D1AttemptEvidence(pilot, (billing,), None)
