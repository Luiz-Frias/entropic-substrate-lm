"""Typed trainer-abort evidence retained by the authoritative D1 run fold."""

from __future__ import annotations

from dataclasses import dataclass

from entropic_substrate.kernel.identity.d1_primitives import thaw_for_digest
from entropic_substrate.ladder.capture.reader.d1_numerical_defect import (
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
)

__all__ = ["D1TrainerAbortEvidence"]


@dataclass(frozen=True, slots=True)
class D1TrainerAbortEvidence:
    """One exact TRAINER_ABORTED event and its optional validated defect."""

    event: D1JournalEvent
    numerical_defect: D1TerminalNumericalDefectRecord | None

    def __post_init__(self) -> None:
        if (
            type(self.event) is not D1JournalEvent
            or self.event.event_kind is not JournalEventKind.TRAINER_ABORTED
        ):
            raise ValueError("trainer abort evidence requires a TRAINER_ABORTED event")
        raw = self.event.payload.get("numerical_defect")
        if self.numerical_defect is None:
            if raw is not None:
                raise ValueError("trainer abort numerical defect requires validated evidence")
        elif (
            type(self.numerical_defect) is not D1TerminalNumericalDefectRecord
            or thaw_for_digest(raw) != self.numerical_defect.manifest
        ):
            raise ValueError("trainer abort event and numerical defect differ")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "event": self.event.manifest,
            "numerical_defect": (
                None if self.numerical_defect is None else self.numerical_defect.manifest
            ),
        }
