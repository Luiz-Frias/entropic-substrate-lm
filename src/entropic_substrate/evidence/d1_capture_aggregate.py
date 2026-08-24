"""Factory-attested aggregate for complete and terminal-defect D1 evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from entropic_substrate.d1_numerical_defect import D1TerminalNumericalDefectRecord
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
)
from entropic_substrate.evidence.d1_capture_bundle import (
    D1AttemptEvidence,
    D1DvcCaptureHandoff,
    D1ValidatedRunCapture,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1ExpectedRun,
    D1LaunchRoster,
    D1PairExclusion,
)
from entropic_substrate.experiments.d1_primitives import RunClass, canonical_digest

__all__ = [
    "D1CapturePartitionProof",
    "D1HeterogeneousCaptureAggregate",
    "D1TerminalDefectInput",
    "D1ValidatedTerminalDefect",
]

_AGGREGATE_TOKEN: Final = object()


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty trimmed string")


def _unique(artifacts: tuple[D1ArtifactIdentity, ...]) -> None:
    if len({item.normalized_path for item in artifacts}) != len(artifacts):
        raise ValueError("terminal defect artifact paths must be unique")


@dataclass(frozen=True, slots=True)
class D1TerminalDefectInput:
    """Authoritative journal and support files for one expected failed cell."""

    logical_run_id: str
    journal: D1ArtifactIdentity
    supporting_artifacts: tuple[D1ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        _text(self.logical_run_id, "terminal defect logical_run_id")
        if self.journal.role is not D1ArtifactRole.JOURNAL:
            raise ValueError("terminal defect journal artifact has the wrong role")
        if type(self.supporting_artifacts) is not tuple:
            raise TypeError("terminal defect supporting_artifacts must be a tuple")
        if any(item.role is D1ArtifactRole.JOURNAL for item in self.supporting_artifacts):
            raise ValueError("terminal defect supporting artifact has the wrong role")
        if not any(
            item.role is D1ArtifactRole.BILLING_SIDECAR for item in self.supporting_artifacts
        ):
            raise ValueError("terminal defect capture requires provider billing evidence")
        _unique(self.artifacts)

    @property
    def artifacts(self) -> tuple[D1ArtifactIdentity, ...]:
        return (self.journal, *self.supporting_artifacts)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "journal": self.journal.manifest,
            "logical_run_id": self.logical_run_id,
            "supporting_artifacts": [item.manifest for item in self.supporting_artifacts],
        }


@dataclass(frozen=True, slots=True)
class D1ValidatedTerminalDefect:
    """One expected cell admitted as measured terminal-defect evidence."""

    expected: D1ExpectedRun
    capture_input: D1TerminalDefectInput
    record: D1TerminalNumericalDefectRecord

    def __post_init__(self) -> None:
        if type(self.expected) is not D1ExpectedRun:
            raise TypeError("validated terminal defect requires D1ExpectedRun")
        if type(self.capture_input) is not D1TerminalDefectInput:
            raise TypeError("validated terminal defect requires D1TerminalDefectInput")
        if type(self.record) is not D1TerminalNumericalDefectRecord:
            raise TypeError("validated terminal defect requires canonical defect record")
        expected_identity = (
            self.expected.logical_run_id,
            self.expected.arm,
            self.expected.seed,
            self.expected.pairing_id,
            self.expected.plan_digest,
            self.expected.family_digest,
            self.expected.construction_digest,
            self.expected.execution_digest,
            self.expected.aggregation_digest,
        )
        record_identity = (
            self.record.logical_run_id,
            self.record.arm,
            self.record.seed,
            self.record.pairing_id,
            self.record.plan_digest,
            self.record.family_digest,
            self.record.construction_digest,
            self.record.execution_digest,
            self.record.aggregation_digest,
        )
        if (
            self.capture_input.logical_run_id != self.expected.logical_run_id
            or record_identity != expected_identity
            or self.record.run_class is not RunClass.CONFIRMATORY
        ):
            raise ValueError("terminal defect identity differs from the expected roster cell")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "artifacts": self.capture_input.manifest,
            "defect": self.record.manifest,
            "identity": self.expected.manifest,
        }


@dataclass(frozen=True, slots=True)
class D1CapturePartitionProof:
    """Witness that every expected roster cell appears in exactly one lane."""

    expected_logical_run_ids: frozenset[str]
    complete_run_ids: frozenset[str]
    terminal_defect_run_ids: frozenset[str]

    def __post_init__(self) -> None:
        values = (
            self.expected_logical_run_ids,
            self.complete_run_ids,
            self.terminal_defect_run_ids,
        )
        if any(type(value) is not frozenset for value in values) or not values[0]:
            raise ValueError("capture partition witnesses must be frozensets with expected cells")
        if self.complete_run_ids & self.terminal_defect_run_ids:
            raise ValueError("complete and terminal-defect lanes must be disjoint")
        if self.complete_run_ids | self.terminal_defect_run_ids != self.expected_logical_run_ids:
            raise ValueError("capture lane union must equal the expected launch roster")

    @property
    def is_exact(self) -> bool:
        return True

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "complete_run_ids": sorted(self.complete_run_ids),
            "expected_logical_run_ids": sorted(self.expected_logical_run_ids),
            "terminal_defect_run_ids": sorted(self.terminal_defect_run_ids),
        }


@dataclass(frozen=True, slots=True)
class D1HeterogeneousCaptureAggregate:
    """Single analysis handoff containing every complete and failed cell."""

    roster: D1LaunchRoster
    partition_proof: D1CapturePartitionProof
    complete_runs: tuple[D1ValidatedRunCapture, ...]
    terminal_defects: tuple[D1ValidatedTerminalDefect, ...]
    attempts: tuple[D1AttemptEvidence, ...]
    pair_exclusions: tuple[D1PairExclusion, ...]
    dvc_handoff: D1DvcCaptureHandoff
    _validation_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_token is not _AGGREGATE_TOKEN:
            raise TypeError(
                "D1HeterogeneousCaptureAggregate is factory-only; use build_d1_capture_aggregate"
            )

    @classmethod
    def from_validated_evidence(
        cls,
        *,
        roster: D1LaunchRoster,
        partition_proof: D1CapturePartitionProof,
        complete_runs: tuple[D1ValidatedRunCapture, ...],
        terminal_defects: tuple[D1ValidatedTerminalDefect, ...],
        attempts: tuple[D1AttemptEvidence, ...],
        pair_exclusions: tuple[D1PairExclusion, ...],
        dvc_handoff: D1DvcCaptureHandoff,
    ) -> D1HeterogeneousCaptureAggregate:
        return cls(
            roster,
            partition_proof,
            complete_runs,
            terminal_defects,
            attempts,
            pair_exclusions,
            dvc_handoff,
            _AGGREGATE_TOKEN,
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "attempt_lanes": [item.manifest for item in self.attempts],
            "authority": {
                "complete_reader": "entropic_substrate.telemetry.read_d1_run_evidence",
                "defect_reader": "entropic_substrate.telemetry.read_d1_terminal_defect",
                "journal": "authoritative",
                "mlflow": "optional_mirror_not_input",
            },
            "complete_runs": [item.manifest for item in self.complete_runs],
            "dvc_handoff": self.dvc_handoff.manifest,
            "pair_exclusions": [item.manifest for item in self.pair_exclusions],
            "partition_proof": self.partition_proof.manifest,
            "roster": self.roster.manifest,
            "schema_version": "d1-heterogeneous-capture-aggregate-v1",
            "terminal_defects": [item.manifest for item in self.terminal_defects],
        }

    @property
    def manifest_digest(self) -> str:
        return canonical_digest(self.manifest)
