"""All-or-nothing launch admission for the registered D1 factorial matrix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Protocol

from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    RunClass,
    canonical_digest,
)
from entropic_substrate.ladder.register.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.ladder.register.d1_plan import ConfirmatoryD1RunPlan
from entropic_substrate.ladder.register.d1_provenance import require_lowercase_digest
from entropic_substrate.ladder.register.d1_registered_design import D1RegisteredDesign

D1_CONFIRMATORY_MATRIX_SCHEMA: Final = "d1-confirmatory-matrix-v2"
_MATRIX_ATTESTATION: Final = object()


class D1GitProvenancePort(Protocol):
    """Read-only facts required from Git; implementations may not mutate it."""

    def commit_time(self, commit: str) -> datetime | None:
        """Return the commit timestamp, or ``None`` when it does not exist."""
        ...

    def blob_sha256(self, commit: str, path: str) -> str | None:
        """Return the repository blob's SHA-256 at a commit and path."""
        ...

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Whether ``descendant`` contains or descends from ``ancestor``."""
        ...


@dataclass(frozen=True, slots=True)
class D1ConfirmatoryLaunch:
    """One planned cell paired with its physical trainer-start envelope."""

    plan: ConfirmatoryD1RunPlan
    start: RunStartProvenance
    registered_design_digest: str
    decision_contract_digest: str

    def __post_init__(self) -> None:
        if type(self.plan) is not ConfirmatoryD1RunPlan:
            raise TypeError("D1 confirmatory launch requires ConfirmatoryD1RunPlan")
        if type(self.start) is not RunStartProvenance:
            raise TypeError("D1 confirmatory launch requires RunStartProvenance")
        require_lowercase_digest(
            self.registered_design_digest,
            label="D1 launch registered_design_digest",
        )
        require_lowercase_digest(
            self.decision_contract_digest,
            label="D1 launch decision_contract_digest",
        )


@dataclass(frozen=True, slots=True)
class D1ResourceEnvelopeIdentity:
    """Placement and runtime facts held identical within one paired seed block."""

    manifest: Mapping[str, object]

    @classmethod
    def from_start(cls, start: RunStartProvenance) -> D1ResourceEnvelopeIdentity:
        return cls(
            manifest={
                "accelerator_count": start.accelerator_count,
                "accelerator_model": start.accelerator_model,
                "compiler_enabled": start.compiler_enabled,
                "compiler_version": start.compiler_version,
                "cuda_version": start.cuda_version,
                "driver_version": start.driver_version,
                "image_digest": start.image_digest,
                "market": start.market,
                "provider": start.provider,
                "provider_sku": start.provider_sku,
                "purchase_option": start.purchase_option.value,
                "pytorch_version": start.pytorch_version,
                "quoted_rate": start.quoted_rate.manifest,
                "region": start.region,
                "resource_activation": start.resource_activation.manifest,
                "schema_version": start.schema_version,
            }
        )

    @property
    def identity_digest(self) -> str:
        return canonical_digest(self.manifest)


@dataclass(frozen=True, slots=True)
class D1ConfirmatoryMatrix:
    """Factory-attested complete matrix in registered seed/arm order."""

    registered_design_digest: str
    decision_contract_digest: str
    registered_seeds: tuple[int, ...]
    launches: tuple[D1ConfirmatoryLaunch, ...]
    terminal_defect_run_ids: tuple[str, ...]
    _attestation: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._attestation is not _MATRIX_ATTESTATION:
            raise TypeError("D1ConfirmatoryMatrix is factory-only")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "cells": [
                {
                    "arm": launch.plan.profile.arm.value,
                    "logical_run_id": launch.plan.logical_run_id,
                    "plan_digest": launch.plan.plan_digest,
                    "resource_envelope_digest": (
                        D1ResourceEnvelopeIdentity.from_start(launch.start).identity_digest
                    ),
                    "seed": launch.plan.seed,
                    "trainer_started_at": launch.start.trainer_started_at.isoformat(),
                }
                for launch in self.launches
            ],
            "decision_contract_digest": self.decision_contract_digest,
            "registered_design_digest": self.registered_design_digest,
            "registered_seeds": list(self.registered_seeds),
            "schema_version": D1_CONFIRMATORY_MATRIX_SCHEMA,
            "terminal_defect_run_ids": list(self.terminal_defect_run_ids),
        }

    @property
    def matrix_digest(self) -> str:
        return canonical_digest(self.manifest)


@dataclass(frozen=True, slots=True)
class D1ConfirmatoryMatrixAdmission:
    matrix: D1ConfirmatoryMatrix
    ledger: D1LaunchAttemptLedger


def _identity_tuple(launch: D1ConfirmatoryLaunch) -> tuple[D1Arm, int]:
    return launch.plan.profile.arm, launch.plan.seed


def _validate_cell(design: D1RegisteredDesign, launch: D1ConfirmatoryLaunch) -> None:
    plan = launch.plan
    arm = plan.profile.arm
    if launch.registered_design_digest != design.design_digest:
        raise ValueError("D1 launch registered-design digest differs from registration")
    if launch.decision_contract_digest != design.decision_contract_digest:
        raise ValueError("D1 launch decision-contract digest differs from registration")
    if plan.profile.family.family_digest != design.family_digest:
        raise ValueError("D1 launch family digest differs from registered design")
    if plan.run.construction_digest != design.arm_construction_digests[arm.value]:
        raise ValueError("D1 launch construction digest differs from registered arm")
    preregistration = plan.run.preregistration
    if preregistration is None or (
        preregistration.path,
        preregistration.commit,
        preregistration.family_digest,
    ) != (
        design.preregistration.path,
        design.preregistration.commit,
        design.family_digest,
    ):
        raise ValueError("D1 launch preregistration identity differs from registered design")
    if launch.start.logical_run_id != plan.logical_run_id:
        raise ValueError("D1 trainer-start logical_run_id differs from its plan")
    if launch.start.execution_segment_index != 0:
        raise ValueError("D1 confirmatory matrix accepts only fresh segment-zero starts")
    container = plan.run.execution.container
    if container is None or launch.start.image_digest != container.image_digest:
        raise ValueError("D1 trainer-start image differs from its immutable plan image")


def _paired_manifest(launch: D1ConfirmatoryLaunch) -> dict[str, object]:
    plan = launch.plan
    code = plan.run.execution.code
    container = plan.run.execution.container
    data = plan.run.data
    if code is None or container is None or data is None:
        raise ValueError("D1 confirmatory launch lacks mandatory paired provenance")
    return {
        "code": code.manifest,
        "container": container.manifest,
        "data": data.manifest,
        "evidence": plan.run.evidence.manifest,
        "pairing_id": plan.run.execution.pairing_id,
        "resource": D1ResourceEnvelopeIdentity.from_start(launch.start).manifest,
    }


def _validate_roster_and_pairing(
    design: D1RegisteredDesign,
    launches: tuple[D1ConfirmatoryLaunch, ...],
) -> tuple[D1ConfirmatoryLaunch, ...]:
    expected = {(arm, seed) for seed in design.registered_seeds for arm in D1Arm}
    seen: dict[tuple[D1Arm, int], D1ConfirmatoryLaunch] = {}
    for launch in launches:
        if type(launch) is not D1ConfirmatoryLaunch:
            raise TypeError("D1 matrix launches must contain D1ConfirmatoryLaunch values")
        if launch.plan.seed not in design.registered_seeds:
            raise ValueError(f"D1 matrix contains unregistered seed {launch.plan.seed}")
        key = _identity_tuple(launch)
        if key in seen:
            raise ValueError(f"D1 matrix contains duplicate cell {key}")
        _validate_cell(design, launch)
        seen[key] = launch
    missing = expected - set(seen)
    if missing:
        raise ValueError(f"D1 matrix is missing registered cells: {sorted(missing)}")
    extras = set(seen) - expected
    if extras:
        raise ValueError(f"D1 matrix contains unregistered cells: {sorted(extras)}")

    for seed in design.registered_seeds:
        block = [seen[(arm, seed)] for arm in D1Arm]
        baseline = _paired_manifest(block[0])
        if any(_paired_manifest(item) != baseline for item in block[1:]):
            raise ValueError(f"D1 paired seed {seed} has provenance or resource drift")
    return tuple(seen[(arm, seed)] for seed in design.registered_seeds for arm in D1Arm)


def _validate_git(
    design: D1RegisteredDesign,
    launches: tuple[D1ConfirmatoryLaunch, ...],
    git: D1GitProvenancePort,
) -> None:
    prereg = design.preregistration
    prereg_time = git.commit_time(prereg.commit)
    if prereg_time is None:
        raise ValueError("D1 preregistration commit does not exist")
    if git.blob_sha256(prereg.commit, prereg.path) != prereg.blob_sha256:
        raise ValueError("D1 preregistration path blob hash differs from registration")
    code_times: dict[str, datetime] = {}
    for launch in launches:
        code = launch.plan.run.execution.code
        if code is None:
            raise ValueError("D1 confirmatory launch lacks code identity")
        code_time = code_times.get(code.commit)
        if code_time is None:
            code_time = git.commit_time(code.commit)
            if code_time is None:
                raise ValueError("D1 code commit does not exist")
            code_times[code.commit] = code_time
            if not git.is_ancestor(prereg.commit, code.commit):
                raise ValueError("D1 preregistration commit is not an ancestor of code commit")
        if prereg_time > launch.start.trainer_started_at:
            raise ValueError("D1 preregistration commit does not precede trainer_start")
        if code_time > launch.start.trainer_started_at:
            raise ValueError("D1 code commit does not precede trainer_start")


def _attempt_identity(
    design: D1RegisteredDesign, launch: D1ConfirmatoryLaunch
) -> D1LaunchAttemptIdentity:
    return D1LaunchAttemptIdentity(
        run_class=RunClass.CONFIRMATORY,
        family_digest=design.family_digest,
        arm=launch.plan.profile.arm,
        seed=launch.plan.seed,
        logical_run_id=launch.plan.logical_run_id,
        registered_design_digest=design.design_digest,
    )


def _matrix(
    design: D1RegisteredDesign,
    ordered: tuple[D1ConfirmatoryLaunch, ...],
    terminal_defect_run_ids: tuple[str, ...],
) -> D1ConfirmatoryMatrix:
    return D1ConfirmatoryMatrix(
        registered_design_digest=design.design_digest,
        decision_contract_digest=design.decision_contract_digest,
        registered_seeds=design.registered_seeds,
        launches=ordered,
        terminal_defect_run_ids=terminal_defect_run_ids,
        _attestation=_MATRIX_ATTESTATION,
    )


def attest_d1_confirmatory_matrix(
    *,
    design: D1RegisteredDesign,
    launches: tuple[D1ConfirmatoryLaunch, ...],
    git: D1GitProvenancePort,
    ledger: D1LaunchAttemptLedger,
    terminal_defect_run_ids: tuple[str, ...] = (),
) -> D1ConfirmatoryMatrix:
    """Attest a post-wave matrix without creating another optional look."""
    if type(design) is not D1RegisteredDesign or type(ledger) is not D1LaunchAttemptLedger:
        raise TypeError("D1 matrix attestation requires registered design and attempt ledger")
    if type(launches) is not tuple:
        raise TypeError("D1 matrix launches must be an immutable tuple")
    ordered = _validate_roster_and_pairing(design, launches)
    ordered_ids = tuple(launch.plan.logical_run_id for launch in ordered)
    if type(terminal_defect_run_ids) is not tuple or terminal_defect_run_ids != tuple(
        run_id for run_id in ordered_ids if run_id in terminal_defect_run_ids
    ):
        raise ValueError("D1 terminal-defect run IDs must be a canonical matrix subset")
    _validate_git(design, ordered, git)
    expected = {_attempt_identity(design, launch).optional_stopping_digest for launch in ordered}
    observed = {event.identity.optional_stopping_digest for event in ledger.events}
    if observed != expected:
        raise ValueError("D1 completed attempt ledger identity set differs from the matrix")
    for launch in ordered:
        identity = _attempt_identity(design, launch)
        history = tuple(
            event
            for event in ledger.events
            if event.identity.optional_stopping_digest == identity.optional_stopping_digest
        )
        terminal_status = (
            D1LaunchAttemptStatus.FAILED
            if launch.plan.logical_run_id in terminal_defect_run_ids
            else D1LaunchAttemptStatus.COMPLETED
        )
        if tuple(event.status for event in history) != (
            D1LaunchAttemptStatus.RECORDED,
            D1LaunchAttemptStatus.TRAINER_STARTED,
            terminal_status,
        ):
            raise ValueError("D1 matrix requires one completed controller attempt per cell")
        if history[1].observed_at != launch.start.trainer_started_at:
            raise ValueError("D1 matrix trainer-start differs from the attempt ledger")
    return _matrix(design, ordered, terminal_defect_run_ids)


def admit_d1_confirmatory_matrix(
    *,
    design: D1RegisteredDesign,
    launches: tuple[D1ConfirmatoryLaunch, ...],
    git: D1GitProvenancePort,
    ledger: D1LaunchAttemptLedger,
) -> D1ConfirmatoryMatrixAdmission:
    """Prove the complete matrix, then atomically return its extended ledger."""
    if type(design) is not D1RegisteredDesign:
        raise TypeError("D1 confirmatory admission requires a completed registered design")
    if type(launches) is not tuple:
        raise TypeError("D1 confirmatory launches must be an immutable tuple")
    if type(ledger) is not D1LaunchAttemptLedger:
        raise TypeError("D1 confirmatory admission requires D1LaunchAttemptLedger")
    ordered = _validate_roster_and_pairing(design, launches)
    _validate_git(design, ordered, git)

    extended = ledger
    for launch in ordered:
        identity = _attempt_identity(design, launch)
        extended = extended.record(identity, observed_at=launch.start.trainer_started_at)
        extended = extended.transition(
            identity,
            status=D1LaunchAttemptStatus.TRAINER_STARTED,
            observed_at=launch.start.trainer_started_at,
        )
    return D1ConfirmatoryMatrixAdmission(matrix=_matrix(design, ordered, ()), ledger=extended)
