"""D1 capture admission proves complete, paired, reader-validated evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.telemetry as telemetry_api
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_bundle import D1AttemptEvidence
from entropic_substrate.evidence.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1LaunchRoster,
)
from entropic_substrate.evidence.d1_capture_validation import (
    D1CaptureBundleError,
    build_d1_capture_bundle,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

from .d1_capture_fixtures import DVC_POINTER, capture_family

pytestmark = pytest.mark.unit


def test_bundle_uses_public_reader_and_proves_exact_four_way_set_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster, inputs, root = capture_family(tmp_path)
    real_reader = telemetry_api.read_d1_run_evidence
    calls: list[Path] = []

    def spy(path: str | Path, **expected: object):
        calls.append(Path(path))
        return real_reader(path, **expected)  # type: ignore[arg-type]

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", spy)
    bundle = build_d1_capture_bundle(root, roster=roster, capture_inputs=inputs)

    assert len(calls) == 4
    assert bundle.set_proof.is_exact
    assert bundle.set_proof.expected_logical_run_ids == roster.expected_run_ids
    assert all(item.data_order_segments == ((0, "e" * 64, 2),) for item in bundle.confirmatory)
    assert bundle.dvc_handoff.analysis_ready is False
    assert len(bundle.dvc_handoff.pending_paths) == 12
    assert bundle.manifest["authority"] == {
        "admissible_reader": "entropic_substrate.telemetry.read_d1_run_evidence",
        "journal": "authoritative",
        "mlflow": "optional_mirror_not_input",
    }
    assert len(bundle.manifest_digest) == 64


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_missing_duplicate_or_unexpected_capture_input_fails_before_admission(
    tmp_path: Path, mutation: str
) -> None:
    roster, inputs, root = capture_family(tmp_path)
    if mutation == "missing":
        inputs = inputs[:-1]
    elif mutation == "duplicate":
        inputs = (*inputs, inputs[0])
    else:
        inputs = (*inputs, replace(inputs[0], logical_run_id="not-in-roster"))

    with pytest.raises(D1CaptureBundleError, match="capture input set"):
        build_d1_capture_bundle(root, roster=roster, capture_inputs=inputs)


def test_checkpoint_receipt_substitution_and_post_hash_drift_fail_loudly(tmp_path: Path) -> None:
    roster, inputs, root = capture_family(tmp_path)
    replacement_path = root / "replacement.pt"
    replacement_path.write_bytes(b"different-checkpoint\n")
    replacement = identify_d1_artifact(
        root,
        replacement_path,
        role=D1ArtifactRole.CHECKPOINT,
        dvc_pointer_path=DVC_POINTER,
    )
    substituted = (replace(inputs[0], checkpoint=replacement), *inputs[1:])
    with pytest.raises(D1CaptureBundleError, match="checkpoint_receipt"):
        build_d1_capture_bundle(root, roster=roster, capture_inputs=substituted)

    (root / inputs[0].journal.normalized_path).write_bytes(b"drift\n")
    with pytest.raises(D1CaptureBundleError, match="artifact"):
        build_d1_capture_bundle(root, roster=roster, capture_inputs=inputs)


def test_journal_substitution_is_rejected_by_expected_reader_identity(tmp_path: Path) -> None:
    roster, inputs, root = capture_family(tmp_path)
    substituted = (
        replace(inputs[0], journal=inputs[1].journal),
        replace(inputs[1], journal=inputs[0].journal),
        *inputs[2:],
    )

    with pytest.raises(D1CaptureBundleError, match="authoritative_reader"):
        build_d1_capture_bundle(root, roster=roster, capture_inputs=substituted)


def test_attempt_lane_set_is_total_but_does_not_enter_confirmatory_proof(tmp_path: Path) -> None:
    roster, inputs, root = capture_family(tmp_path)
    attempt = D1DeclaredAttempt(
        logical_run_id="d1-none-s7-attempt-0",
        arm=D1Arm.NONE,
        seed=7,
        pairing_id="pair-7",
        run_class=RunClass.CONFIRMATORY,
        lane=D1AttemptLane.ABANDONED,
        disposition="pod allocation failed before trainer start",
    )
    roster = D1LaunchRoster(
        preregistration_path=roster.preregistration_path,
        preregistration_commit=roster.preregistration_commit,
        family_digest=roster.family_digest,
        expected_confirmatory=roster.expected_confirmatory,
        attempts=(attempt,),
    )
    with pytest.raises(D1CaptureBundleError, match="attempt evidence set"):
        build_d1_capture_bundle(root, roster=roster, capture_inputs=inputs)

    evidence = D1AttemptEvidence(attempt, (), "provider never created a trainer journal")
    bundle = build_d1_capture_bundle(
        root,
        roster=roster,
        capture_inputs=inputs,
        attempt_evidence=(evidence,),
    )
    assert bundle.set_proof.expected_logical_run_ids == roster.expected_run_ids
    assert bundle.attempts == (evidence,)


def test_public_reader_failure_has_no_live_or_skip_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster, inputs, root = capture_family(tmp_path)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise ValueError("reader refused captured journal")

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", refuse)
    with pytest.raises(D1CaptureBundleError, match="authoritative_reader") as captured:
        build_d1_capture_bundle(root, roster=roster, capture_inputs=inputs)
    assert isinstance(captured.value.__cause__, ValueError)
