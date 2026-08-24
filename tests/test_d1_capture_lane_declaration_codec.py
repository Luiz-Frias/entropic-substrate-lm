"""Canonical declaration codecs for complete, defect, and attempt capture lanes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_aggregate import D1TerminalDefectInput
from entropic_substrate.evidence.d1_capture_artifacts import D1ArtifactIdentity, D1ArtifactRole
from entropic_substrate.evidence.d1_capture_bundle import D1AttemptEvidence, D1CaptureInput
from entropic_substrate.evidence.d1_capture_declaration_codec import (
    D1CaptureDeclarationCodecError,
    attempt_evidence_from_mapping,
    attempt_evidence_to_mapping,
    capture_input_from_mapping,
    capture_input_to_mapping,
    parse_attempt_evidence,
    parse_capture_input,
    parse_terminal_defect_input,
    serialize_attempt_evidence,
    serialize_capture_input,
    serialize_terminal_defect_input,
    terminal_defect_input_from_mapping,
    terminal_defect_input_to_mapping,
    validate_capture_lane_path_partition,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1ExpectedRun,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

from .d1_capture_fixtures import capture_family

pytestmark = pytest.mark.unit


def _complete(tmp_path: Path) -> tuple[D1ExpectedRun, D1CaptureInput]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    roster, inputs, _ = capture_family(tmp_path)
    return roster.expected_confirmatory[0], inputs[0]


def _artifact(source: D1ArtifactIdentity, path: str, role: D1ArtifactRole) -> D1ArtifactIdentity:
    return replace(source, normalized_path=path, role=role)


def _terminal(tmp_path: Path) -> tuple[D1ExpectedRun, D1TerminalDefectInput]:
    expected, complete = _complete(tmp_path)
    billing = tuple(
        _artifact(
            item,
            f"defects/{index:02d}-billing.json",
            D1ArtifactRole.BILLING_SIDECAR,
        )
        for index, item in enumerate(complete.billing_sidecars)
    )
    checkpoint = _artifact(
        complete.checkpoint,
        "defects/prior-checkpoint.pt",
        D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT,
    )
    return expected, D1TerminalDefectInput(
        logical_run_id=expected.logical_run_id,
        journal=_artifact(
            complete.journal,
            "defects/d1-none-s7.jsonl",
            D1ArtifactRole.JOURNAL,
        ),
        supporting_artifacts=(*billing, checkpoint),
    )


def _declared_attempt() -> D1DeclaredAttempt:
    return D1DeclaredAttempt(
        logical_run_id="d1-sg-s7-attempt-0",
        arm=D1Arm.SG,
        seed=7,
        pairing_id="pair-7",
        run_class=RunClass.CONFIRMATORY,
        lane=D1AttemptLane.ABORTED,
        disposition="trainer aborted with durable evidence",
    )


def _attempt(tmp_path: Path) -> tuple[D1DeclaredAttempt, D1AttemptEvidence]:
    _, complete = _complete(tmp_path)
    declared = _declared_attempt()
    journal = _artifact(
        complete.journal,
        "attempts/d1-sg-s7-attempt-0.jsonl",
        D1ArtifactRole.JOURNAL,
    )
    return declared, D1AttemptEvidence(declared, (journal,), None)


def test_complete_capture_input_round_trips_with_supplied_expected_run(tmp_path: Path) -> None:
    expected, capture = _complete(tmp_path)
    mapping = capture_input_to_mapping(capture)
    payload = serialize_capture_input(capture)

    assert capture_input_from_mapping(mapping, expected=expected) == capture
    assert parse_capture_input(payload, expected=expected) == capture
    assert mapping == capture.manifest
    assert payload == json.dumps(mapping, separators=(",", ":"), sort_keys=True)


def test_terminal_defect_input_round_trips_without_minting_validated_defect(
    tmp_path: Path,
) -> None:
    expected, capture = _terminal(tmp_path)
    mapping = terminal_defect_input_to_mapping(capture)
    payload = serialize_terminal_defect_input(capture)

    assert terminal_defect_input_from_mapping(mapping, expected=expected) == capture
    assert parse_terminal_defect_input(payload, expected=expected) == capture
    assert mapping == capture.manifest


def test_attempt_evidence_round_trips_with_embedded_and_supplied_declaration(
    tmp_path: Path,
) -> None:
    declared, evidence = _attempt(tmp_path)
    mapping = attempt_evidence_to_mapping(evidence)
    payload = serialize_attempt_evidence(evidence)

    assert attempt_evidence_from_mapping(mapping, declared=declared) == evidence
    assert parse_attempt_evidence(payload, declared=declared) == evidence
    assert mapping == evidence.manifest


def test_complete_lane_rejects_role_path_billing_and_logical_run_drift(tmp_path: Path) -> None:
    expected, capture = _complete(tmp_path)
    mapping = capture_input_to_mapping(capture)
    mapping["journal"]["role"] = D1ArtifactRole.CHECKPOINT.value  # type: ignore[index]
    with pytest.raises(D1CaptureDeclarationCodecError, match="journal"):
        capture_input_from_mapping(mapping, expected=expected)

    mapping = capture_input_to_mapping(capture)
    mapping["billing_sidecars"] = []
    with pytest.raises(D1CaptureDeclarationCodecError, match="billing"):
        capture_input_from_mapping(mapping, expected=expected)

    mapping = capture_input_to_mapping(capture)
    mapping["quarantined_siblings"] = [mapping["journal"]]
    with pytest.raises(D1CaptureDeclarationCodecError):
        capture_input_from_mapping(mapping, expected=expected)

    with pytest.raises(D1CaptureDeclarationCodecError, match="logical_run_id"):
        capture_input_from_mapping(
            capture_input_to_mapping(capture),
            expected=replace(expected, logical_run_id="different-run"),
        )


def test_terminal_lane_rejects_unknown_roles_missing_billing_or_expected_drift(
    tmp_path: Path,
) -> None:
    expected, capture = _terminal(tmp_path)
    mapping = terminal_defect_input_to_mapping(capture)
    mapping["supporting_artifacts"] = []
    with pytest.raises(D1CaptureDeclarationCodecError, match="billing"):
        terminal_defect_input_from_mapping(mapping, expected=expected)

    mapping = terminal_defect_input_to_mapping(capture)
    mapping["supporting_artifacts"][1]["role"] = D1ArtifactRole.CHECKPOINT.value  # type: ignore[index]
    with pytest.raises(D1CaptureDeclarationCodecError, match="role"):
        terminal_defect_input_from_mapping(mapping, expected=expected)

    with pytest.raises(D1CaptureDeclarationCodecError, match="logical_run_id"):
        terminal_defect_input_from_mapping(
            terminal_defect_input_to_mapping(capture),
            expected=replace(expected, logical_run_id="different-run"),
        )


def test_attempt_lane_rejects_declaration_drift_and_journal_gap_contradictions(
    tmp_path: Path,
) -> None:
    declared, evidence = _attempt(tmp_path)
    mapping = attempt_evidence_to_mapping(evidence)
    mapping["attempt"]["seed"] = 99  # type: ignore[index]
    with pytest.raises(D1CaptureDeclarationCodecError, match="declaration"):
        attempt_evidence_from_mapping(mapping, declared=declared)

    mapping = attempt_evidence_to_mapping(evidence)
    mapping["artifacts"] = []
    with pytest.raises(D1CaptureDeclarationCodecError, match="journal"):
        attempt_evidence_from_mapping(mapping, declared=declared)

    mapping = attempt_evidence_to_mapping(evidence)
    mapping["evidence_gap"] = "claimed gap despite journal"
    with pytest.raises(D1CaptureDeclarationCodecError, match="gap"):
        attempt_evidence_from_mapping(mapping, declared=declared)


def test_global_lane_partition_rejects_complete_terminal_or_attempt_path_reuse(
    tmp_path: Path,
) -> None:
    _, complete = _complete(tmp_path / "complete")
    expected, terminal = _terminal(tmp_path / "terminal")
    declared, attempt = _attempt(tmp_path / "attempt")
    validate_capture_lane_path_partition((complete,), (terminal,), (attempt,))

    reused_terminal = replace(terminal, journal=complete.journal)
    with pytest.raises(D1CaptureDeclarationCodecError, match="reused"):
        validate_capture_lane_path_partition((complete,), (reused_terminal,), (attempt,))

    reused_attempt = replace(attempt, artifacts=(complete.journal,))
    with pytest.raises(D1CaptureDeclarationCodecError, match="reused"):
        validate_capture_lane_path_partition((complete,), (terminal,), (reused_attempt,))
    assert expected.logical_run_id == complete.logical_run_id
    assert declared.logical_run_id == attempt.attempt.logical_run_id


@pytest.mark.parametrize(
    ("parse", "payload", "declaration"),
    [
        (parse_capture_input, '{"logical_run_id":"a","logical_run_id":"b"}', "expected"),
        (parse_terminal_defect_input, '{"logical_run_id":NaN}', "expected"),
        (parse_attempt_evidence, " {}", "declared"),
        (parse_attempt_evidence, "{}\n", "declared"),
        (parse_capture_input, "[]", "expected"),
    ],
)
def test_text_lane_decoders_reject_duplicate_nonfinite_noncanonical_or_nonobject_json(
    tmp_path: Path,
    parse: object,
    payload: str,
    declaration: str,
) -> None:
    expected, _ = _complete(tmp_path)
    declared = _declared_attempt()
    kwargs = {declaration: expected if declaration == "expected" else declared}
    with pytest.raises(D1CaptureDeclarationCodecError):
        parse(payload, **kwargs)  # type: ignore[operator]
