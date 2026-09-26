"""Immutable values emitted only after complete D1 capture admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    D1DvcBindingState,
)
from entropic_substrate.ladder.capture.custody.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1ExpectedRun,
    D1LaunchRoster,
    D1PairExclusion,
)

__all__ = [
    "D1AttemptEvidence",
    "D1CaptureBundle",
    "D1CaptureInput",
    "D1CaptureSetProof",
    "D1DvcCaptureHandoff",
    "D1ValidatedRunCapture",
]

_BUNDLE_TOKEN: Final = object()


def _nonempty(value: str, label: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty trimmed string")


def _unique_artifacts(artifacts: tuple[D1ArtifactIdentity, ...]) -> None:
    if len({item.normalized_path for item in artifacts}) != len(artifacts):
        raise ValueError("artifact paths must be unique")


@dataclass(frozen=True, slots=True)
class D1CaptureInput:
    """All local artifacts associated with one expected confirmatory run."""

    logical_run_id: str
    journal: D1ArtifactIdentity
    checkpoint: D1ArtifactIdentity
    billing_sidecars: tuple[D1ArtifactIdentity, ...]
    quarantined_siblings: tuple[D1ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        _nonempty(self.logical_run_id, "capture logical_run_id")
        if self.journal.role is not D1ArtifactRole.JOURNAL:
            raise ValueError("journal artifact has the wrong role")
        if self.checkpoint.role is not D1ArtifactRole.CHECKPOINT:
            raise ValueError("checkpoint artifact has the wrong role")
        if type(self.billing_sidecars) is not tuple or not self.billing_sidecars:
            raise ValueError("capture requires at least one provider billing sidecar")
        if type(self.quarantined_siblings) is not tuple:
            raise TypeError("quarantined_siblings must be a tuple")
        if any(item.role is not D1ArtifactRole.BILLING_SIDECAR for item in self.billing_sidecars):
            raise ValueError("billing sidecar artifact has the wrong role")
        if any(
            item.role is not D1ArtifactRole.QUARANTINED_SIBLING
            for item in self.quarantined_siblings
        ):
            raise ValueError("quarantined sibling artifact has the wrong role")
        _unique_artifacts(self.artifacts)

    @property
    def artifacts(self) -> tuple[D1ArtifactIdentity, ...]:
        return (self.journal, self.checkpoint, *self.billing_sidecars, *self.quarantined_siblings)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "billing_sidecars": [item.manifest for item in self.billing_sidecars],
            "checkpoint": self.checkpoint.manifest,
            "journal": self.journal.manifest,
            "logical_run_id": self.logical_run_id,
            "quarantined_siblings": [item.manifest for item in self.quarantined_siblings],
        }


@dataclass(frozen=True, slots=True)
class D1AttemptEvidence:
    """Artifacts or an explicit absence for one non-success launch lane."""

    attempt: D1DeclaredAttempt
    artifacts: tuple[D1ArtifactIdentity, ...]
    evidence_gap: str | None

    def __post_init__(self) -> None:
        if type(self.attempt) is not D1DeclaredAttempt:
            raise TypeError("attempt evidence requires D1DeclaredAttempt")
        if type(self.artifacts) is not tuple:
            raise TypeError("attempt artifacts must be a tuple")
        _unique_artifacts(self.artifacts)
        journals = tuple(item for item in self.artifacts if item.role is D1ArtifactRole.JOURNAL)
        if len(journals) > 1:
            raise ValueError("attempt evidence may contain at most one journal")
        if self.attempt.lane is not D1AttemptLane.ABANDONED and not journals:
            raise ValueError("pilot and aborted attempt evidence requires a journal")
        if journals:
            if self.evidence_gap is not None:
                raise ValueError("attempt with a journal cannot also claim an evidence gap")
        else:
            if self.evidence_gap is None:
                raise ValueError("attempt without a journal requires an explicit evidence gap")
            _nonempty(self.evidence_gap, "attempt evidence gap")

    @property
    def journal(self) -> D1ArtifactIdentity | None:
        return next((item for item in self.artifacts if item.role is D1ArtifactRole.JOURNAL), None)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "artifacts": [item.manifest for item in self.artifacts],
            "attempt": self.attempt.manifest,
            "evidence_gap": self.evidence_gap,
        }


@dataclass(frozen=True, slots=True)
class D1CaptureSetProof:
    """Explicit witness that no expected run is missing or substituted."""

    expected_logical_run_ids: frozenset[str]
    terminal_journal_run_ids: frozenset[str]
    checkpoint_receipt_run_ids: frozenset[str]
    capture_input_run_ids: frozenset[str]

    def __post_init__(self) -> None:
        values = (
            self.expected_logical_run_ids,
            self.terminal_journal_run_ids,
            self.checkpoint_receipt_run_ids,
            self.capture_input_run_ids,
        )
        if any(type(value) is not frozenset for value in values) or not values[0]:
            raise ValueError("capture set witnesses must be nonempty frozensets")
        if any(value != values[0] for value in values[1:]):
            raise ValueError("capture requires exact set equality across all four witnesses")

    @property
    def is_exact(self) -> bool:
        return True

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "capture_input_run_ids": sorted(self.capture_input_run_ids),
            "checkpoint_receipt_run_ids": sorted(self.checkpoint_receipt_run_ids),
            "expected_logical_run_ids": sorted(self.expected_logical_run_ids),
            "terminal_journal_run_ids": sorted(self.terminal_journal_run_ids),
        }


@dataclass(frozen=True, slots=True)
class D1ValidatedRunCapture:
    """One expected run admitted through the public authoritative reader."""

    expected: D1ExpectedRun
    capture_input: D1CaptureInput
    terminal_step: int
    data_order_segments: tuple[tuple[int, str, int], ...]
    checkpoint_receipts: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if self.capture_input.logical_run_id != self.expected.logical_run_id:
            raise ValueError("validated run identity differs from capture input")
        if type(self.terminal_step) is not int or self.terminal_step < 1:
            raise ValueError("validated terminal_step must be positive")
        if not self.data_order_segments or not self.checkpoint_receipts:
            raise ValueError("validated run requires data-order and checkpoint evidence")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "artifacts": self.capture_input.manifest,
            "checkpoint_receipts": [list(item) for item in self.checkpoint_receipts],
            "data_order_segments": [list(item) for item in self.data_order_segments],
            "identity": self.expected.manifest,
            "terminal_step": self.terminal_step,
        }


@dataclass(frozen=True, slots=True)
class D1DvcCaptureHandoff:
    """Pure handoff describing what a later DVC adapter must bind."""

    artifacts: tuple[D1ArtifactIdentity, ...]

    def __post_init__(self) -> None:
        if type(self.artifacts) is not tuple or not self.artifacts:
            raise ValueError("DVC handoff requires a nonempty artifact tuple")
        _unique_artifacts(self.artifacts)

    @property
    def pending_paths(self) -> tuple[str, ...]:
        return tuple(
            item.normalized_path
            for item in self.artifacts
            if item.dvc.state is D1DvcBindingState.PENDING
        )

    @property
    def analysis_ready(self) -> bool:
        return not self.pending_paths

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "analysis_ready": self.analysis_ready,
            "artifact_paths": [item.normalized_path for item in self.artifacts],
            "pending_paths": list(self.pending_paths),
            "pointer_paths": sorted({item.dvc.pointer_path for item in self.artifacts}),
            "schema_version": "d1-dvc-capture-handoff-v1",
        }


@dataclass(frozen=True, slots=True)
class D1CaptureBundle:
    """Factory-attested capture bundle; MLflow is absent by construction."""

    roster: D1LaunchRoster
    set_proof: D1CaptureSetProof
    confirmatory: tuple[D1ValidatedRunCapture, ...]
    attempts: tuple[D1AttemptEvidence, ...]
    pair_exclusions: tuple[D1PairExclusion, ...]
    dvc_handoff: D1DvcCaptureHandoff
    _validation_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._validation_token is not _BUNDLE_TOKEN:
            raise TypeError("D1CaptureBundle is factory-only; use build_d1_capture_bundle")

    @classmethod
    def from_validated_evidence(
        cls,
        *,
        roster: D1LaunchRoster,
        set_proof: D1CaptureSetProof,
        confirmatory: tuple[D1ValidatedRunCapture, ...],
        attempts: tuple[D1AttemptEvidence, ...],
        pair_exclusions: tuple[D1PairExclusion, ...],
        dvc_handoff: D1DvcCaptureHandoff,
    ) -> D1CaptureBundle:
        return cls(
            roster,
            set_proof,
            confirmatory,
            attempts,
            pair_exclusions,
            dvc_handoff,
            _BUNDLE_TOKEN,
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "attempt_lanes": [item.manifest for item in self.attempts],
            "authority": {
                "admissible_reader": "entropic_substrate.telemetry.read_d1_run_evidence",
                "journal": "authoritative",
                "mlflow": "optional_mirror_not_input",
            },
            "confirmatory": [item.manifest for item in self.confirmatory],
            "dvc_handoff": self.dvc_handoff.manifest,
            "pair_exclusions": [item.manifest for item in self.pair_exclusions],
            "roster": self.roster.manifest,
            "schema_version": "d1-authoritative-capture-bundle-v1",
            "set_proof": self.set_proof.manifest,
        }

    @property
    def manifest_digest(self) -> str:
        return canonical_digest(self.manifest)
