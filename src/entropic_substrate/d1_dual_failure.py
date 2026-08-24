"""Canonical secondary failure evidence without replacing the primary trainer defect."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

__all__ = [
    "D1PrimarySecondaryFailure",
    "D1SecondaryFailureEvidence",
    "D1SecondaryFailureStage",
    "secondary_failure_from_mapping",
]


class D1SecondaryFailureStage(StrEnum):
    TERMINAL_METER_REFRESH = "terminal-meter-refresh"


def _nonempty(value: str, *, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class D1SecondaryFailureEvidence:
    stage: D1SecondaryFailureStage
    error_type: str
    error_message: str
    schema_version: str = "d1-secondary-failure/v1"

    def __post_init__(self) -> None:
        if type(self.stage) is not D1SecondaryFailureStage:
            raise TypeError("secondary failure stage must be D1SecondaryFailureStage")
        _nonempty(self.error_type, label="secondary failure error_type")
        _nonempty(self.error_message, label="secondary failure error_message")
        if self.schema_version != "d1-secondary-failure/v1":
            raise ValueError("unsupported D1 secondary-failure schema")

    @classmethod
    def from_error(
        cls, stage: D1SecondaryFailureStage, error: BaseException
    ) -> D1SecondaryFailureEvidence:
        if type(stage) is not D1SecondaryFailureStage:
            raise TypeError("secondary failure stage must be D1SecondaryFailureStage")
        if not isinstance(cast(object, error), BaseException):
            raise TypeError("secondary failure requires a caught exception")
        return cls(
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error) or repr(error),
        )

    @property
    def manifest(self) -> dict[str, str]:
        return {
            "error_message": self.error_message,
            "error_type": self.error_type,
            "schema_version": self.schema_version,
            "stage": self.stage.value,
        }


def secondary_failure_from_mapping(value: object) -> D1SecondaryFailureEvidence:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError("secondary failure must be a mapping with string keys")
    expected = {"error_message", "error_type", "schema_version", "stage"}
    if set(value) != expected:
        raise ValueError("secondary failure keys must be exact")
    try:
        stage = D1SecondaryFailureStage(value["stage"])
    except (TypeError, ValueError) as error:
        raise ValueError("secondary failure stage is unsupported") from error
    return D1SecondaryFailureEvidence(
        stage=stage,
        error_type=cast(str, value["error_type"]),
        error_message=cast(str, value["error_message"]),
        schema_version=cast(str, value["schema_version"]),
    )


class D1PrimarySecondaryFailure(RuntimeError):
    """Nominal wrapper retaining the causal primary and measurement secondary errors."""

    def __init__(self, primary: BaseException, secondary: BaseException) -> None:
        if not isinstance(cast(object, primary), BaseException) or not isinstance(
            cast(object, secondary), BaseException
        ):
            raise TypeError("dual failure requires primary and secondary exceptions")
        self.primary = primary
        self.secondary = secondary
        self.exceptions = (primary, secondary)
        self.secondary_evidence = D1SecondaryFailureEvidence.from_error(
            D1SecondaryFailureStage.TERMINAL_METER_REFRESH,
            secondary,
        )
        primary_message = str(primary) or repr(primary)
        secondary_message = str(secondary) or repr(secondary)
        super().__init__(
            f"primary {type(primary).__name__}: {primary_message}; "
            f"secondary {type(secondary).__name__}: {secondary_message}"
        )
