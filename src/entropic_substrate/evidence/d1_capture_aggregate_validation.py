"""Admission of an exact complete-run/terminal-defect D1 capture partition."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

import entropic_substrate.telemetry as telemetry_api
from entropic_substrate.d1_numerical_defect import D1CheckpointReferenceKind
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1CapturePartitionProof,
    D1HeterogeneousCaptureAggregate,
    D1TerminalDefectInput,
    D1ValidatedTerminalDefect,
)
from entropic_substrate.evidence.d1_capture_artifacts import D1ArtifactRole
from entropic_substrate.evidence.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureInput,
    D1DvcCaptureHandoff,
    D1ValidatedRunCapture,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1ExpectedRun,
    D1LaunchRoster,
    D1PairExclusion,
    D1PairExclusionReason,
)
from entropic_substrate.evidence.d1_capture_validation import (
    validate_d1_attempt_evidence,
    validate_d1_capture_artifacts,
    validate_d1_complete_pairing,
    validate_d1_complete_run_capture,
)
from entropic_substrate.experiments.d1_primitives import RunClass

__all__ = ["D1CaptureAggregateError", "build_d1_capture_aggregate"]


class D1CaptureAggregateError(ValueError):
    """Stable fail-loud refusal from heterogeneous capture admission."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = MappingProxyType({"code": code, "path": path})
        super().__init__(f"D1 capture aggregate {code} at {path}: {detail}")


def _fail(code: str, path: str, detail: str, cause: Exception | None = None) -> NoReturn:
    error = D1CaptureAggregateError(code=code, path=path, detail=detail)
    if cause is None:
        raise error
    raise error from cause


def _indexed[T](values: tuple[T, ...], identity: Callable[[T], str]) -> dict[str, T]:
    indexed = {identity(item): item for item in values}
    if len(indexed) != len(values):
        _fail("capture_partition", "roster", "a capture lane contains duplicate run identities")
    return indexed


def _partition(
    roster: D1LaunchRoster,
    complete_inputs: tuple[D1CaptureInput, ...],
    defect_inputs: tuple[D1TerminalDefectInput, ...],
) -> tuple[dict[str, D1CaptureInput], dict[str, D1TerminalDefectInput]]:
    complete = _indexed(complete_inputs, lambda item: item.logical_run_id)
    defects = _indexed(defect_inputs, lambda item: item.logical_run_id)
    try:
        D1CapturePartitionProof(roster.expected_run_ids, frozenset(complete), frozenset(defects))
    except (TypeError, ValueError) as error:
        _fail("capture_partition", "roster", str(error), error)
    return complete, defects


def _expected_by_id(roster: D1LaunchRoster) -> dict[str, D1ExpectedRun]:
    return {item.logical_run_id: item for item in roster.expected_confirmatory}


def _validate_complete(
    root: Path,
    roster: D1LaunchRoster,
    complete: dict[str, D1CaptureInput],
) -> tuple[D1ValidatedRunCapture, ...]:
    expected = _expected_by_id(roster)
    try:
        captures = tuple(
            validate_d1_complete_run_capture(
                root,
                capture=complete[item.logical_run_id],
                expected=item,
                roster=roster,
            )
            for item in roster.expected_confirmatory
            if item.logical_run_id in complete
        )
        validate_d1_complete_pairing(captures)
    except Exception as error:
        _fail("complete_run", "complete", str(error), error)
    if any(item.expected != expected[item.expected.logical_run_id] for item in captures):
        _fail("complete_run", "complete", "validated identity differs from roster")
    return captures


def _checkpoint_contract(defect: D1ValidatedTerminalDefect) -> None:
    checkpoints = tuple(
        item
        for item in defect.capture_input.supporting_artifacts
        if item.role is D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT
    )
    reference = defect.record.checkpoint
    if reference.kind is D1CheckpointReferenceKind.RECEIPT:
        if len(checkpoints) != 1 or checkpoints[0].sha256 != reference.sha256:
            _fail(
                "defect_checkpoint",
                defect.capture_input.logical_run_id,
                "checkpoint receipt does not bind exactly one captured checkpoint",
            )
    elif checkpoints:
        _fail(
            "defect_checkpoint",
            defect.capture_input.logical_run_id,
            "checkpoint-gap evidence cannot also capture a checkpoint",
        )


def _validate_defects(
    root: Path,
    roster: D1LaunchRoster,
    defects: dict[str, D1TerminalDefectInput],
) -> tuple[D1ValidatedTerminalDefect, ...]:
    expected = _expected_by_id(roster)
    validated: list[D1ValidatedTerminalDefect] = []
    for item in roster.expected_confirmatory:
        capture = defects.get(item.logical_run_id)
        if capture is None:
            continue
        try:
            validate_d1_capture_artifacts(root, capture.artifacts)
        except Exception as error:
            _fail("defect_artifact", capture.journal.normalized_path, str(error), error)
        try:
            record = telemetry_api.read_d1_terminal_defect(
                root / capture.journal.normalized_path,
                expected_logical_run_id=item.logical_run_id,
                expected_family_digest=item.family_digest,
                expected_run_class=RunClass.CONFIRMATORY,
                expected_plan_digest=item.plan_digest,
            )
            run = telemetry_api.read_d1_run_evidence(
                root / capture.journal.normalized_path,
                expected_logical_run_id=item.logical_run_id,
                expected_family_digest=item.family_digest,
                expected_run_class=RunClass.CONFIRMATORY,
                expected_plan_digest=item.plan_digest,
                allow_incomplete=True,
            )
            value = D1ValidatedTerminalDefect(item, capture, record, run.data_order_segments)
        except Exception as error:
            _fail("defect_reader", capture.journal.normalized_path, str(error), error)
        if value.expected != expected[value.expected.logical_run_id]:
            _fail("defect_reader", capture.journal.normalized_path, "identity differs")
        _checkpoint_contract(value)
        validated.append(value)
    return tuple(validated)


def _pairing(
    complete: tuple[D1ValidatedRunCapture, ...],
    defects: tuple[D1ValidatedTerminalDefect, ...],
) -> None:
    by_pair: dict[str, set[tuple[tuple[int, str, int], ...]]] = {}
    for item in complete:
        by_pair.setdefault(item.expected.pairing_id, set()).add(item.data_order_segments)
    for item in defects:
        by_pair.setdefault(item.expected.pairing_id, set()).add(item.data_order_segments)
    mismatched = sorted(
        pairing_id for pairing_id, identities in by_pair.items() if len(identities) != 1
    )
    if mismatched:
        _fail("data_order_pairing", ",".join(mismatched), "paired cells consumed different order")


def _derive_exclusions(
    roster: D1LaunchRoster,
    defects: tuple[D1ValidatedTerminalDefect, ...],
) -> tuple[D1PairExclusion, ...]:
    defect_pairs = sorted({item.expected.pairing_id for item in defects})
    exclusions = tuple(
        D1PairExclusion(
            pairing_id=pairing_id,
            member_run_ids=tuple(
                item.logical_run_id
                for item in roster.expected_confirmatory
                if item.pairing_id == pairing_id
            ),
            reason=D1PairExclusionReason.NONFINITE_FINAL_LOSS,
            evidence_paths=tuple(
                sorted(
                    item.capture_input.journal.normalized_path
                    for item in defects
                    if item.expected.pairing_id == pairing_id
                )
            ),
        )
        for pairing_id in defect_pairs
    )
    return roster.validate_pair_exclusions(exclusions)


def build_d1_capture_aggregate(
    root: str | Path,
    *,
    roster: D1LaunchRoster,
    complete_inputs: tuple[D1CaptureInput, ...],
    terminal_defect_inputs: tuple[D1TerminalDefectInput, ...],
    attempt_evidence: tuple[D1AttemptEvidence, ...] = (),
) -> D1HeterogeneousCaptureAggregate:
    """Admit every expected cell as complete or canonical terminal defect."""
    if type(roster) is not D1LaunchRoster:
        raise TypeError("aggregate roster must be D1LaunchRoster")
    if type(complete_inputs) is not tuple or type(terminal_defect_inputs) is not tuple:
        raise TypeError("aggregate capture lanes must be tuples")
    capture_root = Path(root)
    complete_by_id, defects_by_id = _partition(roster, complete_inputs, terminal_defect_inputs)
    complete = _validate_complete(capture_root, roster, complete_by_id)
    defects = _validate_defects(capture_root, roster, defects_by_id)
    _pairing(complete, defects)
    exclusions = _derive_exclusions(roster, defects)
    excluded_ids = {run_id for item in exclusions for run_id in item.member_run_ids}
    nonexcluded = roster.expected_run_ids - excluded_ids
    if not nonexcluded <= frozenset(complete_by_id):
        _fail("nonexcluded_incomplete", "roster", "a nonexcluded cell is not complete")
    try:
        attempt_artifacts = validate_d1_attempt_evidence(
            capture_root, roster=roster, evidence=attempt_evidence
        )
    except Exception as error:
        _fail("attempt_evidence", "attempts", str(error), error)
    artifacts = (
        tuple(item for capture in complete_inputs for item in capture.artifacts)
        + tuple(item for capture in terminal_defect_inputs for item in capture.artifacts)
        + attempt_artifacts
    )
    if len({item.normalized_path for item in artifacts}) != len(artifacts):
        _fail("artifact_set", "aggregate", "artifact path is reused across capture lanes")
    proof = D1CapturePartitionProof(
        roster.expected_run_ids,
        frozenset(complete_by_id),
        frozenset(defects_by_id),
    )
    return D1HeterogeneousCaptureAggregate.from_validated_evidence(
        roster=roster,
        partition_proof=proof,
        complete_runs=complete,
        terminal_defects=defects,
        attempts=attempt_evidence,
        pair_exclusions=exclusions,
        dvc_handoff=D1DvcCaptureHandoff(artifacts),
    )
