"""Strict command-specific request value for operator-visible D1 inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from entropic_substrate.kernel.identity.d1_primitives import RunClass, require_sha256
from entropic_substrate.ladder.analyze.d1_inspect_values import D1_INSPECT_MAX_EVALUATIONS

__all__ = [
    "D1_INSPECT_MAX_JOURNAL_BYTES",
    "D1_INSPECT_REQUEST_SCHEMA",
    "D1InspectRequest",
    "d1_inspect_request_from_mapping",
]

D1_INSPECT_REQUEST_SCHEMA: Final = "d1-inspect-request-v1"
D1_INSPECT_MAX_JOURNAL_BYTES: Final = 268_435_456
_KEYS: Final = {
    "allow_incomplete",
    "evaluation_limit",
    "expected_family_digest",
    "expected_logical_run_id",
    "expected_plan_digest",
    "expected_run_class",
    "journal_path",
    "max_journal_bytes",
    "schema_version",
}


def _optional_text(value: str | None, *, label: str) -> None:
    if value is not None and (type(value) is not str or not value.strip()):
        raise ValueError(f"D1 inspect {label} must be null or a nonempty string")


def _optional_digest(value: str | None, *, label: str) -> None:
    if value is not None:
        require_sha256(value, label=f"D1 inspect {label}")


@dataclass(frozen=True, slots=True)
class D1InspectRequest:
    journal_path: Path
    max_journal_bytes: int
    allow_incomplete: bool
    evaluation_limit: int
    expected_logical_run_id: str | None
    expected_family_digest: str | None
    expected_run_class: RunClass | None
    expected_plan_digest: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.journal_path), Path)
            or not str(self.journal_path)
            or "\x00" in str(self.journal_path)
        ):
            raise ValueError("D1 inspect journal_path must be a nonempty NUL-free Path")
        if (
            type(self.max_journal_bytes) is not int
            or not 1 <= self.max_journal_bytes <= D1_INSPECT_MAX_JOURNAL_BYTES
        ):
            raise ValueError(
                "D1 inspect max_journal_bytes must be a positive integer within its hard bound"
            )
        if type(self.allow_incomplete) is not bool:
            raise TypeError("D1 inspect allow_incomplete must be a bool")
        if (
            type(self.evaluation_limit) is not int
            or not 1 <= self.evaluation_limit <= D1_INSPECT_MAX_EVALUATIONS
        ):
            raise ValueError("D1 inspect evaluation_limit is outside its hard bound")
        _optional_text(self.expected_logical_run_id, label="expected_logical_run_id")
        _optional_digest(self.expected_family_digest, label="expected_family_digest")
        _optional_digest(self.expected_plan_digest, label="expected_plan_digest")
        if self.expected_run_class is not None and type(self.expected_run_class) is not RunClass:
            raise TypeError("D1 inspect expected_run_class must be null or nominal")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "allow_incomplete": self.allow_incomplete,
            "evaluation_limit": self.evaluation_limit,
            "expected_family_digest": self.expected_family_digest,
            "expected_logical_run_id": self.expected_logical_run_id,
            "expected_plan_digest": self.expected_plan_digest,
            "expected_run_class": (
                None if self.expected_run_class is None else self.expected_run_class.value
            ),
            "journal_path": str(self.journal_path),
            "max_journal_bytes": self.max_journal_bytes,
            "schema_version": D1_INSPECT_REQUEST_SCHEMA,
        }


def _optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"D1 inspect {label} must be null or a string")
    return value


def d1_inspect_request_from_mapping(value: object) -> D1InspectRequest:
    """Parse the exact inspect payload nested inside the strict CLI envelope."""
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError("D1 inspect request must be an object with string keys")
    item = cast(dict[str, object], value)
    if set(item) != _KEYS:
        raise ValueError("D1 inspect request keys must be exact")
    if item["schema_version"] != D1_INSPECT_REQUEST_SCHEMA:
        raise ValueError("D1 inspect request schema is unsupported")
    path = item["journal_path"]
    if type(path) is not str or not path or "\x00" in path:
        raise ValueError("D1 inspect journal_path must be a nonempty NUL-free string")
    raw_class = item["expected_run_class"]
    try:
        run_class = None if raw_class is None else RunClass(raw_class)
    except (TypeError, ValueError) as error:
        raise ValueError("D1 inspect expected_run_class is unsupported") from error
    return D1InspectRequest(
        Path(path),
        cast(int, item["max_journal_bytes"]),
        cast(bool, item["allow_incomplete"]),
        cast(int, item["evaluation_limit"]),
        _optional_string(item["expected_logical_run_id"], label="expected_logical_run_id"),
        _optional_string(item["expected_family_digest"], label="expected_family_digest"),
        run_class,
        _optional_string(item["expected_plan_digest"], label="expected_plan_digest"),
    )
