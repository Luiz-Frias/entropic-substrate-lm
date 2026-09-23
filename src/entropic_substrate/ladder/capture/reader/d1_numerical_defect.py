"""Typed, canonical evidence for a nonfinite D1 terminal metric."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    RunStartProvenance,
    execution_segment_id,
)
from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    EvidenceClass,
    RunClass,
)
from entropic_substrate.ladder.register.d1_plan import D1RunPlan

__all__ = [
    "D1CheckpointReferenceKind",
    "D1DefectCheckpointReference",
    "D1NonfiniteClassification",
    "D1NonfiniteMetricError",
    "D1TerminalNumericalDefectRecord",
    "d1_nonfinite_metric_error",
    "terminal_numerical_defect_from_mapping",
]

D1_TERMINAL_METRIC: Final = "nats_per_byte"
D1_TERMINAL_DEFECT_SCHEMA: Final = "d1-terminal-numerical-defect-v1"
_FIELDS: Final = frozenset(
    {
        "aggregation_digest",
        "arm",
        "checkpoint",
        "classification",
        "construction_digest",
        "eval_index",
        "evidence_class",
        "execution_digest",
        "execution_segment_id",
        "execution_segment_index",
        "family_digest",
        "logical_run_id",
        "max_steps",
        "metric_name",
        "pairing_id",
        "plan_digest",
        "run_class",
        "schema_version",
        "seed",
        "source_exception_code",
        "successful_step",
    }
)


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty trimmed string")


def _sha256(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _count(value: int, label: str, *, positive: bool = False) -> None:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


class D1NonfiniteClassification(StrEnum):
    NAN = "nan"
    POSITIVE_INFINITY = "positive_infinity"
    NEGATIVE_INFINITY = "negative_infinity"


class D1NonfiniteMetricError(ValueError):
    """A guard-site observation with no nonfinite value retained or serialized."""

    def __init__(
        self,
        *,
        metric_name: str,
        classification: D1NonfiniteClassification,
        source_exception_code: str,
    ) -> None:
        if metric_name != D1_TERMINAL_METRIC:
            raise ValueError("D1 numerical defect requires the registered terminal metric")
        if type(classification) is not D1NonfiniteClassification:
            raise TypeError("D1 numerical defect classification must be nominal")
        _text(source_exception_code, "source_exception_code")
        self.metric_name = metric_name
        self.classification = classification
        self.source_exception_code = source_exception_code
        super().__init__(
            f"D1 {metric_name} classified as {classification.value} at {source_exception_code}"
        )


def d1_nonfinite_metric_error(
    *, metric_name: str, value: float, source_exception_code: str
) -> D1NonfiniteMetricError:
    """Classify a measured nonfinite float without retaining the raw value."""
    if metric_name != D1_TERMINAL_METRIC:
        raise ValueError("D1 numerical defect requires the registered terminal metric")
    if type(value) is not float or math.isfinite(value):
        raise ValueError("D1 numerical defect source value must be nonfinite float")
    if math.isnan(value):
        classification = D1NonfiniteClassification.NAN
    elif value > 0.0:
        classification = D1NonfiniteClassification.POSITIVE_INFINITY
    else:
        classification = D1NonfiniteClassification.NEGATIVE_INFINITY
    return D1NonfiniteMetricError(
        metric_name=metric_name,
        classification=classification,
        source_exception_code=source_exception_code,
    )


class D1CheckpointReferenceKind(StrEnum):
    RECEIPT = "receipt"
    GAP = "gap"


@dataclass(frozen=True, slots=True)
class D1DefectCheckpointReference:
    """Latest durable checkpoint receipt or an explicit recording-time gap."""

    kind: D1CheckpointReferenceKind
    sha256: str | None
    gap_code: str | None

    def __post_init__(self) -> None:
        if type(self.kind) is not D1CheckpointReferenceKind:
            raise TypeError("checkpoint reference kind must be nominal")
        if self.kind is D1CheckpointReferenceKind.RECEIPT:
            if self.sha256 is None or self.gap_code is not None:
                raise ValueError("checkpoint receipt requires only sha256")
            _sha256(self.sha256, "checkpoint sha256")
        elif self.sha256 is not None or self.gap_code is None:
            raise ValueError("checkpoint gap requires only gap_code")
        else:
            _text(self.gap_code, "checkpoint gap_code")

    @classmethod
    def receipt(cls, sha256: str) -> D1DefectCheckpointReference:
        return cls(D1CheckpointReferenceKind.RECEIPT, sha256, None)

    @classmethod
    def gap(cls, code: str) -> D1DefectCheckpointReference:
        return cls(D1CheckpointReferenceKind.GAP, None, code)

    @property
    def manifest(self) -> dict[str, object]:
        return {"gap_code": self.gap_code, "kind": self.kind.value, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class D1TerminalNumericalDefectRecord:
    """Run-enriched terminal defect embedded in ``TRAINER_ABORTED``."""

    metric_name: str
    classification: D1NonfiniteClassification
    source_exception_code: str
    successful_step: int
    max_steps: int
    eval_index: int
    arm: D1Arm
    seed: int
    pairing_id: str
    logical_run_id: str
    run_class: RunClass
    execution_segment_index: int
    execution_segment_id: str
    plan_digest: str
    family_digest: str
    construction_digest: str
    execution_digest: str
    aggregation_digest: str
    checkpoint: D1DefectCheckpointReference
    evidence_class: EvidenceClass = EvidenceClass.MEASURED

    def __post_init__(self) -> None:
        if self.metric_name != D1_TERMINAL_METRIC:
            raise ValueError("defect record requires the registered terminal metric")
        if type(self.classification) is not D1NonfiniteClassification:
            raise TypeError("defect classification must be nominal")
        _text(self.source_exception_code, "source_exception_code")
        _count(self.successful_step, "successful_step", positive=True)
        _count(self.max_steps, "max_steps", positive=True)
        _count(self.eval_index, "eval_index")
        if self.successful_step != self.max_steps:
            raise ValueError("terminal numerical defect requires the exact terminal cursor")
        if type(self.arm) is not D1Arm or type(self.run_class) is not RunClass:
            raise TypeError("defect record requires nominal arm and run_class")
        _count(self.seed, "seed")
        _text(self.pairing_id, "pairing_id")
        _text(self.logical_run_id, "logical_run_id")
        _count(self.execution_segment_index, "execution_segment_index")
        if self.execution_segment_id != execution_segment_id(
            self.logical_run_id, self.execution_segment_index
        ):
            raise ValueError("defect execution segment identity is inconsistent")
        for field in (
            "plan_digest",
            "family_digest",
            "construction_digest",
            "execution_digest",
            "aggregation_digest",
        ):
            _sha256(getattr(self, field), field)
        if type(self.checkpoint) is not D1DefectCheckpointReference:
            raise TypeError("defect record requires a checkpoint reference")
        if self.evidence_class is not EvidenceClass.MEASURED:
            raise ValueError("terminal numerical defect evidence_class must be measured")

    @classmethod
    def from_error(
        cls,
        error: D1NonfiniteMetricError,
        *,
        plan: D1RunPlan,
        start: RunStartProvenance,
        successful_step: int,
        eval_index: int,
        checkpoint: D1DefectCheckpointReference,
    ) -> D1TerminalNumericalDefectRecord:
        if type(error) is not D1NonfiniteMetricError:
            raise TypeError("terminal defect requires D1NonfiniteMetricError")
        if not isinstance(cast(object, plan), D1RunPlan) or type(start) is not RunStartProvenance:
            raise TypeError("terminal defect requires typed plan and run start")
        if start.logical_run_id != plan.logical_run_id:
            raise ValueError("terminal defect plan and segment logical_run_id differ")
        return cls(
            metric_name=error.metric_name,
            classification=error.classification,
            source_exception_code=error.source_exception_code,
            successful_step=successful_step,
            max_steps=plan.profile.draft.execution_protocol.optimization.max_steps,
            eval_index=eval_index,
            arm=plan.profile.arm,
            seed=plan.seed,
            pairing_id=plan.run.execution.pairing_id,
            logical_run_id=plan.logical_run_id,
            run_class=plan.run.run_class,
            execution_segment_index=start.execution_segment_index,
            execution_segment_id=start.execution_segment_id,
            plan_digest=plan.plan_digest,
            family_digest=plan.profile.family.family_digest,
            construction_digest=plan.run.construction_digest,
            execution_digest=plan.run.execution_digest,
            aggregation_digest=plan.run.aggregation_digest,
            checkpoint=checkpoint,
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "aggregation_digest": self.aggregation_digest,
            "arm": self.arm.value,
            "checkpoint": self.checkpoint.manifest,
            "classification": self.classification.value,
            "construction_digest": self.construction_digest,
            "eval_index": self.eval_index,
            "evidence_class": self.evidence_class.value,
            "execution_digest": self.execution_digest,
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "family_digest": self.family_digest,
            "logical_run_id": self.logical_run_id,
            "max_steps": self.max_steps,
            "metric_name": self.metric_name,
            "pairing_id": self.pairing_id,
            "plan_digest": self.plan_digest,
            "run_class": self.run_class.value,
            "schema_version": D1_TERMINAL_DEFECT_SCHEMA,
            "seed": self.seed,
            "source_exception_code": self.source_exception_code,
            "successful_step": self.successful_step,
        }


def terminal_numerical_defect_from_mapping(
    value: object,
) -> D1TerminalNumericalDefectRecord:
    """Strictly admit the sole canonical nested defect representation."""
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError("terminal numerical defect must be a JSON object")
    item = dict(cast(Mapping[str, object], value))
    if frozenset(item) != _FIELDS:
        raise ValueError("terminal numerical defect has missing or unknown fields")
    if item["schema_version"] != D1_TERMINAL_DEFECT_SCHEMA:
        raise ValueError("terminal numerical defect schema is unsupported")
    checkpoint_raw = item["checkpoint"]
    if not isinstance(checkpoint_raw, Mapping) or set(checkpoint_raw) != {
        "gap_code",
        "kind",
        "sha256",
    }:
        raise ValueError("terminal numerical defect checkpoint is malformed")
    checkpoint_item = dict(cast(Mapping[str, object], checkpoint_raw))
    try:
        checkpoint = D1DefectCheckpointReference(
            kind=D1CheckpointReferenceKind(cast(str, checkpoint_item["kind"])),
            sha256=cast(str | None, checkpoint_item["sha256"]),
            gap_code=cast(str | None, checkpoint_item["gap_code"]),
        )
        record = D1TerminalNumericalDefectRecord(
            metric_name=cast(str, item["metric_name"]),
            classification=D1NonfiniteClassification(cast(str, item["classification"])),
            source_exception_code=cast(str, item["source_exception_code"]),
            successful_step=cast(int, item["successful_step"]),
            max_steps=cast(int, item["max_steps"]),
            eval_index=cast(int, item["eval_index"]),
            arm=D1Arm(cast(str, item["arm"])),
            seed=cast(int, item["seed"]),
            pairing_id=cast(str, item["pairing_id"]),
            logical_run_id=cast(str, item["logical_run_id"]),
            run_class=RunClass(cast(str, item["run_class"])),
            execution_segment_index=cast(int, item["execution_segment_index"]),
            execution_segment_id=cast(str, item["execution_segment_id"]),
            plan_digest=cast(str, item["plan_digest"]),
            family_digest=cast(str, item["family_digest"]),
            construction_digest=cast(str, item["construction_digest"]),
            execution_digest=cast(str, item["execution_digest"]),
            aggregation_digest=cast(str, item["aggregation_digest"]),
            checkpoint=checkpoint,
            evidence_class=EvidenceClass(cast(str, item["evidence_class"])),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"terminal numerical defect is invalid: {error}") from error
    if record.manifest != item:
        raise ValueError("terminal numerical defect is not canonical")
    return record
