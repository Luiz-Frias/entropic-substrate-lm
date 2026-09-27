"""Bounded, status-discriminated values for operator-visible D1 inspection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    RunClass,
    require_sha256,
)
from entropic_substrate.kernel.verdict.d1_evaluation_values import (
    TARGET_PREDICTED_EXACT_MATCH_SEMANTIC,
)
from entropic_substrate.ladder.capture.reader.d1_numerical_defect import (
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind
from entropic_substrate.ladder.capture.records.d1_telemetry_values import (
    D1_REPORT_ONLY_METRICS,
    D1_SCALAR_METRICS,
    D1RunStatus,
)
from entropic_substrate.ladder.capture.resource.resource_work import ResourceWorkSnapshot
from entropic_substrate.ladder.register.d1_execution import PAIRWISE_COSINE_SEMANTIC

__all__ = [
    "D1_INSPECT_MAX_EVALUATIONS",
    "D1InspectCheckpoint",
    "D1InspectComplete",
    "D1InspectCompletion",
    "D1InspectEvaluation",
    "D1InspectEvaluationWindow",
    "D1InspectEventRef",
    "D1InspectJournal",
    "D1InspectPartial",
    "D1InspectPlan",
    "D1InspectProgress",
    "D1InspectQualityScalars",
    "D1InspectRun",
    "D1InspectRunAborted",
    "D1InspectState",
    "D1InspectTrainerAborted",
    "D1InspectView",
]

D1_INSPECT_MAX_EVALUATIONS: Final = 64
_DECIDING_METRICS: Final = D1_SCALAR_METRICS - D1_REPORT_ONLY_METRICS


def _nonempty(value: str, *, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"D1 inspect {field} must be a nonempty string")


def _count(value: int, *, field: str, positive: bool = False) -> None:
    lower = 1 if positive else 0
    if type(value) is not int or value < lower:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"D1 inspect {field} must be a {qualifier} integer")


@dataclass(frozen=True, slots=True)
class D1InspectJournal:
    path: str
    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.path, field="journal path")
        _nonempty(self.relative_path, field="journal relative_path")
        _count(self.byte_count, field="journal byte_count", positive=True)
        require_sha256(self.sha256, label="D1 inspect journal sha256")


@dataclass(frozen=True, slots=True)
class D1InspectRun:
    logical_run_id: str
    run_class: RunClass
    execution_digest: str
    aggregation_digest: str
    construction_digest: str
    family_digest: str
    plan_digest: str
    authority_digest: str

    def __post_init__(self) -> None:
        _nonempty(self.logical_run_id, field="logical_run_id")
        if type(self.run_class) is not RunClass:
            raise TypeError("D1 inspect run_class must be nominal")
        for field in (
            "execution_digest",
            "aggregation_digest",
            "construction_digest",
            "family_digest",
            "plan_digest",
            "authority_digest",
        ):
            require_sha256(getattr(self, field), label=f"D1 inspect {field}")


@dataclass(frozen=True, slots=True)
class D1InspectPlan:
    profile_name: str
    arm: D1Arm
    seed: int
    pairing_id: str

    def __post_init__(self) -> None:
        _nonempty(self.profile_name, field="plan profile_name")
        if type(self.arm) is not D1Arm:
            raise TypeError("D1 inspect plan arm must be nominal")
        _count(self.seed, field="plan seed")
        _nonempty(self.pairing_id, field="plan pairing_id")


@dataclass(frozen=True, slots=True)
class D1InspectQualityScalars:
    deciding: Mapping[str, float]
    report_only: Mapping[str, float]

    def __post_init__(self) -> None:
        for field, value, expected in (
            ("deciding", self.deciding, _DECIDING_METRICS),
            ("report_only", self.report_only, D1_REPORT_ONLY_METRICS),
        ):
            if not isinstance(cast(object, value), Mapping) or set(value) != set(expected):
                raise ValueError(f"D1 inspect {field} quality metric keys must be exact")
            if any(type(item) is not float or not math.isfinite(item) for item in value.values()):
                raise ValueError(f"D1 inspect {field} quality metrics must be finite floats")
            object.__setattr__(self, field, MappingProxyType(dict(value)))


@dataclass(frozen=True, slots=True)
class D1InspectEvaluation:
    step: int
    eval_index: int
    latent_width: int
    quality: D1InspectQualityScalars
    pairwise_cosine_semantic: str
    target_predicted_exact_match_semantic: str
    resource_work: ResourceWorkSnapshot

    def __post_init__(self) -> None:
        _count(self.step, field="evaluation step")
        _count(self.eval_index, field="evaluation index")
        _count(self.latent_width, field="latent_width", positive=True)
        if type(self.quality) is not D1InspectQualityScalars:
            raise TypeError("D1 inspect evaluation quality must be nominal")
        if self.pairwise_cosine_semantic != PAIRWISE_COSINE_SEMANTIC:
            raise ValueError("D1 inspect pairwise-cosine semantic changed")
        if self.target_predicted_exact_match_semantic != TARGET_PREDICTED_EXACT_MATCH_SEMANTIC:
            raise ValueError("D1 inspect target-predicted exact-match semantic changed")
        if (
            type(self.resource_work) is not ResourceWorkSnapshot
            or self.resource_work.training_step != self.step
            or self.resource_work.eval_index != self.eval_index
        ):
            raise ValueError("D1 inspect evaluation and resource-work cursors differ")


@dataclass(frozen=True, slots=True)
class D1InspectCheckpoint:
    successful_step: int
    sha256: str

    def __post_init__(self) -> None:
        _count(self.successful_step, field="checkpoint successful_step")
        require_sha256(self.sha256, label="D1 inspect checkpoint sha256")


@dataclass(frozen=True, slots=True)
class D1InspectEvaluationWindow:
    total_count: int
    limit: int
    items: tuple[D1InspectEvaluation, ...]

    def __post_init__(self) -> None:
        _count(self.total_count, field="evaluation total_count")
        if type(self.limit) is not int or not 1 <= self.limit <= D1_INSPECT_MAX_EVALUATIONS:
            raise ValueError("D1 inspect evaluation window limit is outside its hard bound")
        if type(self.items) is not tuple or any(
            type(item) is not D1InspectEvaluation for item in self.items
        ):
            raise TypeError("D1 inspect evaluation window must contain nominal items")
        if len(self.items) != min(self.total_count, self.limit):
            raise ValueError("D1 inspect evaluation window returned count differs from its limit")
        expected = tuple(range(self.total_count - len(self.items), self.total_count))
        if tuple(item.eval_index for item in self.items) != expected:
            raise ValueError("D1 inspect evaluation window must be the terminal suffix")

    @property
    def returned_count(self) -> int:
        return len(self.items)

    @property
    def omitted_count(self) -> int:
        return self.total_count - self.returned_count


@dataclass(frozen=True, slots=True)
class D1InspectProgress:
    segment_count: int
    data_order_segment_count: int
    tail_quarantine_count: int
    baseline_present: bool
    total_evaluation_count: int
    total_train_step_count: int
    total_checkpoint_count: int
    latest_segment: RunStartProvenance
    latest_quality: D1InspectEvaluation | None
    latest_checkpoint: D1InspectCheckpoint | None

    def __post_init__(self) -> None:
        _count(self.segment_count, field="segment_count", positive=True)
        for field in (
            "data_order_segment_count",
            "tail_quarantine_count",
            "total_evaluation_count",
            "total_train_step_count",
            "total_checkpoint_count",
        ):
            _count(getattr(self, field), field=field)
        if type(self.baseline_present) is not bool:
            raise TypeError("D1 inspect baseline_present must be a bool")
        if type(self.latest_segment) is not RunStartProvenance:
            raise TypeError("D1 inspect latest_segment must be run-start provenance")
        if (self.latest_quality is None) != (self.total_evaluation_count == 0):
            raise ValueError("D1 inspect latest_quality presence differs from its count")
        if self.latest_quality is not None and (
            type(self.latest_quality) is not D1InspectEvaluation
            or self.latest_quality.eval_index != self.total_evaluation_count - 1
        ):
            raise ValueError("D1 inspect latest_quality cursor differs from its count")
        if (self.latest_checkpoint is None) != (self.total_checkpoint_count == 0):
            raise ValueError("D1 inspect latest_checkpoint presence differs from its count")


@dataclass(frozen=True, slots=True)
class D1InspectCompletion:
    final_quality: D1InspectEvaluation
    final_checkpoint: D1InspectCheckpoint
    run_end: D1InspectEventRef

    def __post_init__(self) -> None:
        if (
            type(self.final_quality) is not D1InspectEvaluation
            or type(self.final_checkpoint) is not D1InspectCheckpoint
            or type(self.run_end) is not D1InspectEventRef
            or self.run_end.event_kind is not JournalEventKind.RUN_END
        ):
            raise ValueError(
                "D1 inspect completion requires final quality, checkpoint, and RUN_END"
            )
        step = self.final_quality.step
        if self.final_checkpoint.successful_step != step or self.run_end.successful_step != step:
            raise ValueError("D1 inspect completion cursors differ")


@dataclass(frozen=True, slots=True)
class D1InspectComplete:
    completion: D1InspectCompletion

    def __post_init__(self) -> None:
        if type(self.completion) is not D1InspectCompletion:
            raise TypeError("complete D1 inspection requires nominal completion evidence")


@dataclass(frozen=True, slots=True)
class D1InspectPartial:
    """An admitted in-flight prefix without terminal evidence."""


@dataclass(frozen=True, slots=True)
class D1InspectEventRef:
    event_kind: JournalEventKind
    event_seq: int
    successful_step: int
    event_sha256: str

    def __post_init__(self) -> None:
        if type(self.event_kind) is not JournalEventKind:
            raise TypeError("D1 inspect event kind must be nominal")
        _count(self.event_seq, field="event sequence")
        _count(self.successful_step, field="event successful_step")
        require_sha256(self.event_sha256, label="D1 inspect event sha256")


@dataclass(frozen=True, slots=True)
class D1InspectTrainerAborted:
    trainer_abort: D1InspectEventRef
    numerical_defect: D1TerminalNumericalDefectRecord | None
    run_finalization: D1InspectEventRef | None

    def __post_init__(self) -> None:
        if (
            type(self.trainer_abort) is not D1InspectEventRef
            or self.trainer_abort.event_kind is not JournalEventKind.TRAINER_ABORTED
        ):
            raise TypeError("trainer-aborted inspection requires a TRAINER_ABORTED event ref")
        if self.numerical_defect is not None and (
            type(self.numerical_defect) is not D1TerminalNumericalDefectRecord
        ):
            raise TypeError("trainer-aborted numerical defect must be validated evidence")
        if self.run_finalization is not None and (
            type(self.run_finalization) is not D1InspectEventRef
            or self.run_finalization.event_kind is not JournalEventKind.RUN_END
        ):
            raise ValueError("trainer-aborted run finalization must be a RUN_END event ref")


@dataclass(frozen=True, slots=True)
class D1InspectRunAborted:
    run_abort: D1InspectEventRef

    def __post_init__(self) -> None:
        if (
            type(self.run_abort) is not D1InspectEventRef
            or self.run_abort.event_kind is not JournalEventKind.RUN_ABORT
        ):
            raise ValueError("run-aborted inspection requires a RUN_ABORT event")


type D1InspectState = (
    D1InspectComplete | D1InspectPartial | D1InspectTrainerAborted | D1InspectRunAborted
)


@dataclass(frozen=True, slots=True)
class D1InspectView:
    journal: D1InspectJournal
    run: D1InspectRun
    plan: D1InspectPlan
    progress: D1InspectProgress
    evaluations: D1InspectEvaluationWindow
    state: D1InspectState

    def __post_init__(self) -> None:
        if self.progress.total_evaluation_count != self.evaluations.total_count:
            raise ValueError("D1 inspect progress and evaluation-window counts differ")
        latest = self.evaluations.items[-1] if self.evaluations.items else None
        if self.progress.latest_quality != latest:
            raise ValueError("D1 inspect progress latest quality differs from its suffix")
        if (
            self.progress.latest_segment.logical_run_id != self.run.logical_run_id
            or self.progress.latest_segment.execution_segment_index
            != self.progress.segment_count - 1
            or any(
                item.resource_work.logical_run_id != self.run.logical_run_id
                for item in self.evaluations.items
            )
        ):
            raise ValueError("D1 inspect nested evidence differs from the outer run identity")
        if type(self.state) is D1InspectComplete and (
            self.state.completion.final_quality != self.progress.latest_quality
            or self.state.completion.final_checkpoint != self.progress.latest_checkpoint
        ):
            raise ValueError("D1 inspect completion differs from latest progress")

    @property
    def status(self) -> D1RunStatus:
        if type(self.state) is D1InspectComplete:
            return D1RunStatus.COMPLETE
        if type(self.state) is D1InspectPartial:
            return D1RunStatus.PARTIAL
        if type(self.state) is D1InspectTrainerAborted:
            return D1RunStatus.TRAINER_ABORTED
        if type(self.state) is D1InspectRunAborted:
            return D1RunStatus.RUN_ABORTED
        raise TypeError("D1 inspect state must be nominal")
