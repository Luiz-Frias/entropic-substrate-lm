"""Run-stage lifecycle validation and execution digests for a D1 family cell."""

from __future__ import annotations

from dataclasses import dataclass

from entropic_substrate.experiments.d1_family import D1DraftSpec, D1ExperimentFamily
from entropic_substrate.experiments.d1_primitives import (
    AGGREGATION_SCHEMA,
    EXECUTION_SCHEMA,
    RunClass,
    canonical_digest,
)
from entropic_substrate.experiments.d1_provenance import (
    DataIdentity,
    EvidenceEnvelope,
    ExecutionIdentity,
    PreregistrationIdentity,
)

__all__ = [
    "ConfirmatoryD1RunSpec",
    "D1RunSpec",
    "PilotD1RunSpec",
    "SmokeD1RunSpec",
]


@dataclass(frozen=True, slots=True)
class D1RunSpec:
    """Validated lifecycle stage with per-cell, family, and execution identity."""

    draft: D1DraftSpec
    execution: ExecutionIdentity
    data: DataIdentity | None
    evidence: EvidenceEnvelope
    preregistration: PreregistrationIdentity | None
    family: D1ExperimentFamily | None

    def __post_init__(self) -> None:
        if type(self) is D1RunSpec:
            msg = "D1RunSpec is factory-only; construct a nominal lifecycle subtype"
            raise TypeError(msg)
        if self.run_class is RunClass.CONFIRMATORY:
            self._validate_confirmatory_provenance()
        elif self.preregistration is not None or self.family is not None:
            msg = "smoke and pilot runs must remain excluded from preregistration evidence"
            raise ValueError(msg)

    def _validate_confirmatory_provenance(self) -> None:
        if self.preregistration is None:
            msg = "confirmatory D1 run requires preregistration identity"
            raise ValueError(msg)
        if self.family is None:
            msg = "confirmatory D1 run requires its complete four-arm family"
            raise ValueError(msg)
        if self.family.draft_for(self.draft.arm) != self.draft:
            msg = "confirmatory D1 run draft is not the registered family cell"
            raise ValueError(msg)
        if self.preregistration.family_digest != self.family_digest:
            msg = "preregistration family_digest does not match registered family digest"
            raise ValueError(msg)
        if self.data is None:
            msg = "confirmatory D1 run requires dataset identity"
            raise ValueError(msg)
        if self.execution.code is None:
            msg = "confirmatory D1 run requires code identity"
            raise ValueError(msg)
        if self.execution.container is None:
            msg = "confirmatory D1 run requires container identity"
            raise ValueError(msg)

    @property
    def run_class(self) -> RunClass:
        return self.execution.run_class

    @property
    def construction_manifest(self) -> dict[str, object]:
        return self.draft.construction_manifest

    @property
    def construction_digest(self) -> str:
        return self.draft.construction_digest

    @property
    def family_digest(self) -> str:
        if self.family is None:
            msg = "smoke and pilot runs do not carry a confirmatory family digest"
            raise ValueError(msg)
        return self.family.family_digest

    @property
    def aggregation_manifest(self) -> dict[str, object]:
        """Identity for lawful pooling; seed is deliberately absent."""
        return {
            "code": None if self.execution.code is None else self.execution.code.manifest,
            "construction_digest": self.construction_digest,
            "container": (
                None if self.execution.container is None else self.execution.container.manifest
            ),
            "data": None if self.data is None else self.data.manifest,
            "evidence": self.evidence.manifest,
            "family_digest": None if self.family is None else self.family.family_digest,
            "pairing_id": self.execution.pairing_id,
            "preregistration": (
                None if self.preregistration is None else self.preregistration.manifest
            ),
            "run_class": self.run_class.value,
            "schema_version": AGGREGATION_SCHEMA,
        }

    @property
    def aggregation_digest(self) -> str:
        return canonical_digest(self.aggregation_manifest)

    @property
    def execution_manifest(self) -> dict[str, object]:
        """Per-run identity, including seed, run class, and preregistration."""
        return {
            "aggregation_digest": self.aggregation_digest,
            "execution": self.execution.manifest,
            "family_digest": None if self.family is None else self.family.family_digest,
            "preregistration": (
                None if self.preregistration is None else self.preregistration.manifest
            ),
            "schema_version": EXECUTION_SCHEMA,
        }

    @property
    def execution_digest(self) -> str:
        return canonical_digest(self.execution_manifest)


@dataclass(frozen=True, slots=True)
class SmokeD1RunSpec(D1RunSpec):
    """Nominal smoke-run state that can never become confirmatory evidence."""

    def __post_init__(self) -> None:
        D1RunSpec.__post_init__(self)
        if self.execution.run_class is not RunClass.SMOKE:
            msg = f"smoke run requires run_class=smoke, got {self.execution.run_class.value}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PilotD1RunSpec(D1RunSpec):
    """Nominal excluded-pilot state."""

    def __post_init__(self) -> None:
        D1RunSpec.__post_init__(self)
        if self.execution.run_class is not RunClass.PILOT:
            msg = f"pilot run requires run_class=pilot, got {self.execution.run_class.value}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ConfirmatoryD1RunSpec(D1RunSpec):
    """Nominal confirmatory state requiring the complete registered envelope."""

    def __post_init__(self) -> None:
        D1RunSpec.__post_init__(self)
        if self.execution.run_class is not RunClass.CONFIRMATORY:
            msg = (
                "confirmatory run requires run_class=confirmatory, got "
                f"{self.execution.run_class.value}"
            )
            raise ValueError(msg)
