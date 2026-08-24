"""Dedicated capture admission and pair-exclusion derivation for D1 defects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

import entropic_substrate.telemetry as telemetry_api
from entropic_substrate.d1_numerical_defect import (
    D1CheckpointReferenceKind,
    D1TerminalNumericalDefectRecord,
    terminal_numerical_defect_from_mapping,
)
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    D1CaptureArtifactError,
    verify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1ExpectedRun,
    D1LaunchRoster,
    D1PairExclusion,
    D1PairExclusionReason,
)
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import D1JournalError, JournalEventKind
from entropic_substrate.evidence.d1_telemetry_values import D1RunStatus
from entropic_substrate.experiments.d1_primitives import RunClass, canonical_digest

__all__ = [
    "D1DefectCaptureError",
    "D1TerminalDefectCaptureInput",
    "D1TerminalNumericalDefectEvidence",
    "derive_d1_pair_exclusions",
    "read_d1_terminal_numerical_defect",
    "validate_d1_pair_exclusion_claims",
]

_SOURCE_CODE = "d1.eval.nats_per_byte.nonfinite.v1"


class D1DefectCaptureError(ValueError):
    """Stable refusal from the defect-only capture and exclusion path."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = MappingProxyType({"code": code, "path": path})
        super().__init__(f"D1 numerical defect {code} at {path}: {detail}")


def _fail(code: str, path: str, detail: str, cause: Exception | None = None) -> NoReturn:
    error = D1DefectCaptureError(code=code, path=path, detail=detail)
    if cause is None:
        raise error
    raise error from cause


@dataclass(frozen=True, slots=True)
class D1TerminalDefectCaptureInput:
    """Artifacts in the separate terminal-numerical-defect lane."""

    expected: D1ExpectedRun
    journal: D1ArtifactIdentity
    checkpoint: D1ArtifactIdentity
    billing_sidecars: tuple[D1ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        if type(self.expected) is not D1ExpectedRun:
            raise TypeError("defect capture requires an expected D1 run")
        if self.journal.role is not D1ArtifactRole.JOURNAL:
            raise ValueError("defect journal artifact has the wrong role")
        if self.checkpoint.role is not D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT:
            raise ValueError("defect checkpoint artifact has the wrong role")
        if type(self.billing_sidecars) is not tuple or not self.billing_sidecars:
            raise ValueError("defect capture requires provider billing sidecar evidence")
        if any(item.role is not D1ArtifactRole.BILLING_SIDECAR for item in self.billing_sidecars):
            raise ValueError("defect billing sidecar artifact has the wrong role")
        paths = [item.normalized_path for item in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("defect capture artifact paths must be unique")

    @property
    def artifacts(self) -> tuple[D1ArtifactIdentity, ...]:
        return self.journal, self.checkpoint, *self.billing_sidecars

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "billing_sidecars": [item.manifest for item in self.billing_sidecars],
            "checkpoint": self.checkpoint.manifest,
            "expected": self.expected.manifest,
            "journal": self.journal.manifest,
            "schema_version": "d1-terminal-defect-capture-input-v1",
        }

    @property
    def capture_input_digest(self) -> str:
        return canonical_digest(self.manifest)


@dataclass(frozen=True, slots=True)
class D1TerminalNumericalDefectEvidence:
    """Capture-byte identities joined to one canonical typed defect record."""

    record: D1TerminalNumericalDefectRecord
    expected: D1ExpectedRun
    journal: D1ArtifactIdentity
    checkpoint: D1ArtifactIdentity
    billing_sidecars: tuple[D1ArtifactIdentity, ...]
    capture_input_digest: str

    def __post_init__(self) -> None:
        if type(self.record) is not D1TerminalNumericalDefectRecord:
            raise TypeError("defect evidence requires a typed record")
        if type(self.expected) is not D1ExpectedRun:
            raise TypeError("defect evidence requires an expected run")
        _match_expected(self.record, self.expected)
        if self.record.run_class is not RunClass.CONFIRMATORY:
            raise ValueError("exclusion evidence must come from a confirmatory run")
        if self.record.checkpoint.kind is not D1CheckpointReferenceKind.RECEIPT:
            raise ValueError("exclusion evidence requires a concrete checkpoint receipt")
        if self.record.checkpoint.sha256 != self.checkpoint.sha256:
            raise ValueError("defect record and captured checkpoint differ")
        if self.journal.role is not D1ArtifactRole.JOURNAL:
            raise ValueError("defect evidence journal role differs")
        if self.checkpoint.role is not D1ArtifactRole.NUMERICAL_DEFECT_CHECKPOINT:
            raise ValueError("defect evidence checkpoint role differs")
        if (
            canonical_digest(
                {
                    "billing_sidecars": [item.manifest for item in self.billing_sidecars],
                    "checkpoint": self.checkpoint.manifest,
                    "expected": self.expected.manifest,
                    "journal": self.journal.manifest,
                    "schema_version": "d1-terminal-defect-capture-input-v1",
                }
            )
            != self.capture_input_digest
        ):
            raise ValueError("defect capture-input digest is inconsistent")


def _verify(root: Path, artifacts: tuple[D1ArtifactIdentity, ...]) -> None:
    for artifact in artifacts:
        try:
            verify_d1_artifact(root, artifact)
        except D1CaptureArtifactError as error:
            _fail("artifact", artifact.normalized_path, str(error), error)


def _match_expected(record: D1TerminalNumericalDefectRecord, expected: D1ExpectedRun) -> None:
    for field in (
        "logical_run_id",
        "arm",
        "seed",
        "pairing_id",
        "plan_digest",
        "family_digest",
        "construction_digest",
        "execution_digest",
        "aggregation_digest",
    ):
        if getattr(record, field) != getattr(expected, field):
            _fail("identity", field, "defect record differs from the expected run")


def read_d1_terminal_numerical_defect(
    root: str | Path, capture: D1TerminalDefectCaptureInput
) -> D1TerminalNumericalDefectEvidence:
    """Admit only a typed terminal abort; never consult MLflow or a live server."""
    if type(capture) is not D1TerminalDefectCaptureInput:
        raise TypeError("capture must be D1TerminalDefectCaptureInput")
    capture_root = Path(root)
    _verify(capture_root, capture.artifacts)
    journal_path = capture_root / capture.journal.normalized_path
    try:
        run = telemetry_api.read_d1_run_evidence(
            journal_path,
            expected_logical_run_id=capture.expected.logical_run_id,
            expected_family_digest=capture.expected.family_digest,
            expected_run_class=RunClass.CONFIRMATORY,
            expected_plan_digest=capture.expected.plan_digest,
            allow_incomplete=True,
        )
    except Exception as error:
        _fail("authoritative_reader", capture.journal.normalized_path, str(error), error)
    if run.status is not D1RunStatus.TRAINER_ABORTED:
        _fail("defect_lane", capture.journal.normalized_path, "journal is not trainer_aborted")
    try:
        events = read_journal(journal_path)
    except D1JournalError as error:
        _fail("canonical_journal", capture.journal.normalized_path, str(error), error)
    terminal = events[-1]
    if terminal.event_kind is not JournalEventKind.TRAINER_ABORTED:
        _fail("defect_lane", capture.journal.normalized_path, "abort is not the final event")
    raw = terminal.payload.get("numerical_defect")
    try:
        record = terminal_numerical_defect_from_mapping(raw)
    except (TypeError, ValueError) as error:
        _fail("defect_payload", capture.journal.normalized_path, str(error), error)
    _match_expected(record, capture.expected)
    if record.source_exception_code != _SOURCE_CODE:
        _fail("source_exception", record.source_exception_code, "is not a registered guard")
    if record.successful_step != terminal.successful_step or record.eval_index != len(run.evals):
        _fail("cursor", capture.journal.normalized_path, "defect cursor differs from journal")
    if not run.checkpoint_receipts or record.checkpoint.sha256 != run.checkpoint_receipts[-1][1]:
        _fail("checkpoint_receipt", capture.checkpoint.normalized_path, "journal receipt differs")
    if record.checkpoint.sha256 != capture.checkpoint.sha256:
        _fail("checkpoint_artifact", capture.checkpoint.normalized_path, "captured bytes differ")
    return D1TerminalNumericalDefectEvidence(
        record=record,
        expected=capture.expected,
        journal=capture.journal,
        checkpoint=capture.checkpoint,
        billing_sidecars=capture.billing_sidecars,
        capture_input_digest=capture.capture_input_digest,
    )


def derive_d1_pair_exclusions(
    roster: D1LaunchRoster,
    evidence: tuple[D1TerminalNumericalDefectEvidence, ...],
) -> tuple[D1PairExclusion, ...]:
    """Derive the registered whole-pair action from admitted defect evidence."""
    if type(roster) is not D1LaunchRoster or type(evidence) is not tuple:
        raise TypeError("pair exclusion derivation requires typed roster and evidence tuple")
    if len({item.record.logical_run_id for item in evidence}) != len(evidence):
        _fail("duplicate_evidence", "logical_run_id", "defect evidence is duplicated")
    expected = {item.logical_run_id: item for item in roster.expected_confirmatory}
    by_pair: dict[str, list[D1TerminalNumericalDefectEvidence]] = {}
    for item in evidence:
        run = expected.get(item.record.logical_run_id)
        if run is None:
            _fail("unexpected_evidence", item.record.logical_run_id, "is not in the roster")
        _match_expected(item.record, run)
        by_pair.setdefault(item.record.pairing_id, []).append(item)
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
                sorted(item.journal.normalized_path for item in by_pair[pairing_id])
            ),
        )
        for pairing_id in sorted(by_pair)
    )
    return roster.validate_pair_exclusions(exclusions)


def validate_d1_pair_exclusion_claims(
    roster: D1LaunchRoster,
    claims: tuple[D1PairExclusion, ...],
    evidence: tuple[D1TerminalNumericalDefectEvidence, ...],
) -> tuple[D1PairExclusion, ...]:
    """Reject inherited exclusions unless fresh evidence derives the same claims."""
    admitted = roster.validate_pair_exclusions(claims)
    derived = derive_d1_pair_exclusions(roster, evidence)
    if admitted != derived:
        _fail("unsupported_exclusion", "pair_exclusions", "claims lack admitted defect evidence")
    return admitted
