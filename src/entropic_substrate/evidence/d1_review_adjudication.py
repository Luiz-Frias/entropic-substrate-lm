"""Strict D1 review receipt, MAJOR finding, and adjudication values."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import cast

__all__ = [
    "D1_NO_MAJOR_OUTCOME",
    "D1_NO_MAJOR_PROMPT_DIRECTIVE",
    "D1MajorFindingDisposition",
    "D1MajorFindingStatus",
    "D1ParsedReviewReport",
    "D1ReviewAdjudicationSurface",
    "D1ReviewExecutionReceiptV2",
    "adjudicate_d1_review",
    "parse_d1_major_findings",
    "parse_d1_review_execution_receipt",
    "validate_d1_review_outcome_prompt",
]

_RECEIPT_FIELDS = frozenset(
    {
        "model_id",
        "packet_sha256",
        "prompt_sha256",
        "provider",
        "report_sha256",
        "request_digest",
        "response_sha256",
        "retry_lineage",
        "round",
        "schema_version",
    }
)
_MAJOR_HEADING = re.compile(r"(?m)^#{1,6}\s+(MAJOR-[1-9][0-9]*)(?:\s|[-:\u2014])")
_MAJOR_TOKEN = re.compile(r"\bMAJOR-[0-9]+\b")
_MAJOR_ID = re.compile(r"MAJOR-[1-9][0-9]*")
D1_NO_MAJOR_OUTCOME = "Review outcome: NO_MAJOR_FINDINGS"
D1_NO_MAJOR_PROMPT_DIRECTIVE = (
    "If and only if there are no MAJOR findings, emit exactly this standalone line: "
    f"{D1_NO_MAJOR_OUTCOME}"
)


def _sha(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty and trimmed")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"D1 review receipt duplicate field {key!r}")
        result[key] = value
    return result


def _document(payload: bytes) -> Mapping[str, object]:
    if type(payload) is not bytes or not payload:
        raise ValueError("D1 review receipt must be nonempty bytes")
    try:
        value = json.loads(payload, object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("D1 review receipt must be UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError("D1 review receipt must have an object root")
    data = cast(dict[str, object], value)
    if frozenset(data) != _RECEIPT_FIELDS:
        raise ValueError("D1 review receipt fields must match the v2 schema exactly")
    return data


@dataclass(frozen=True, slots=True)
class D1ReviewExecutionReceiptV2:
    packet_sha256: str
    prompt_sha256: str
    provider: str
    model_id: str
    round: int
    request_digest: str
    response_sha256: str
    report_sha256: str
    retry_lineage: tuple[str, ...]


def parse_d1_review_execution_receipt(payload: bytes) -> D1ReviewExecutionReceiptV2:
    """Parse the receipt emitted by the cloud-review execution boundary."""
    data = _document(payload)
    if data["schema_version"] != "d1-review-execution-receipt-v2":
        raise ValueError("D1 review receipt schema is unsupported")
    for field in (
        "packet_sha256",
        "prompt_sha256",
        "request_digest",
        "response_sha256",
        "report_sha256",
    ):
        _sha(cast(str, data[field]), f"D1 review receipt {field}")
    provider = cast(str, data["provider"])
    model_id = cast(str, data["model_id"])
    _text(provider, "D1 review receipt provider")
    _text(model_id, "D1 review receipt model_id")
    round_number = data["round"]
    if type(round_number) is not int or round_number < 1:
        raise ValueError("D1 review receipt round must be positive")
    raw_lineage = data["retry_lineage"]
    if type(raw_lineage) is not list:
        raise ValueError("D1 review receipt retry lineage must be an array")
    lineage = tuple(cast(list[str], raw_lineage))
    if len(lineage) != round_number - 1 or len(set(lineage)) != len(lineage):
        raise ValueError("D1 review receipt retry lineage must equal round minus one uniquely")
    for digest in lineage:
        _sha(digest, "D1 review receipt retry parent")
    return D1ReviewExecutionReceiptV2(
        cast(str, data["packet_sha256"]),
        cast(str, data["prompt_sha256"]),
        provider,
        model_id,
        round_number,
        cast(str, data["request_digest"]),
        cast(str, data["response_sha256"]),
        cast(str, data["report_sha256"]),
        lineage,
    )


@dataclass(frozen=True, slots=True)
class D1ParsedReviewReport:
    report_sha256: str
    major_finding_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha(self.report_sha256, "D1 parsed review report")
        if type(self.major_finding_ids) is not tuple or any(
            type(item) is not str or _MAJOR_ID.fullmatch(item) is None
            for item in self.major_finding_ids
        ):
            raise ValueError("D1 parsed review finding IDs must be canonical MAJOR-N values")
        if len(set(self.major_finding_ids)) != len(self.major_finding_ids):
            raise ValueError("D1 parsed review finding IDs must be unique")


def validate_d1_review_outcome_prompt(text: str) -> None:
    """Require one exact prompt instruction for the empty-finding outcome."""
    if type(text) is not str or text.splitlines().count(D1_NO_MAJOR_PROMPT_DIRECTIVE) != 1:
        raise ValueError("D1 reviewer prompt must register the no-MAJOR outcome exactly once")


def parse_d1_major_findings(
    payload: bytes,
    *,
    reviewer_prompt: bytes,
) -> D1ParsedReviewReport:
    """Extract stable heading IDs and reject silent or contradictory absence."""
    if type(payload) is not bytes or not payload:
        raise ValueError("D1 reviewer report must be nonempty bytes")
    if type(reviewer_prompt) is not bytes or not reviewer_prompt:
        raise ValueError("D1 reviewer prompt must be nonempty bytes")
    try:
        text = payload.decode("utf-8")
        prompt_text = reviewer_prompt.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("D1 reviewer report and prompt must be UTF-8") from error
    validate_d1_review_outcome_prompt(prompt_text)
    headings = tuple(_MAJOR_HEADING.findall(text))
    if len(set(headings)) != len(headings):
        raise ValueError("D1 reviewer report has a duplicate MAJOR heading")
    references = set(_MAJOR_TOKEN.findall(text))
    if references != set(headings):
        raise ValueError("every D1 MAJOR reference requires one canonical heading")
    outcome_count = text.splitlines().count(D1_NO_MAJOR_OUTCOME)
    marker_present = "NO_MAJOR_FINDINGS" in text
    if marker_present and outcome_count == 0:
        raise ValueError("NO_MAJOR_FINDINGS must appear in the standalone outcome declaration")
    if outcome_count > 1:
        raise ValueError("D1 no-MAJOR outcome must appear exactly once")
    if headings and outcome_count:
        raise ValueError("NO_MAJOR_FINDINGS contradicts a MAJOR heading")
    if not headings and outcome_count != 1:
        raise ValueError("D1 report without findings requires explicit NO_MAJOR_FINDINGS")
    return D1ParsedReviewReport(hashlib.sha256(payload).hexdigest(), headings)


class D1MajorFindingStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True, slots=True)
class D1MajorFindingDisposition:
    finding_id: str
    status: D1MajorFindingStatus
    rationale: str
    evidence_sha256: str | None

    def __post_init__(self) -> None:
        if _MAJOR_TOKEN.fullmatch(self.finding_id) is None or self.finding_id == "MAJOR-0":
            raise ValueError("D1 finding disposition requires a canonical MAJOR ID")
        if type(self.status) is not D1MajorFindingStatus:
            raise TypeError("D1 finding disposition status must be nominal")
        _text(self.rationale, "D1 finding disposition rationale")
        if self.status is D1MajorFindingStatus.RESOLVED and self.evidence_sha256 is None:
            raise ValueError("resolved D1 MAJOR requires evidence SHA-256")
        if self.evidence_sha256 is not None:
            _sha(self.evidence_sha256, "D1 finding disposition evidence")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "evidence_sha256": self.evidence_sha256,
            "finding_id": self.finding_id,
            "rationale": self.rationale,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class D1ReviewAdjudicationSurface:
    report_sha256: str
    major_finding_ids: tuple[str, ...]
    dispositions: Mapping[str, D1MajorFindingDisposition]
    open_major_finding_ids: tuple[str, ...]
    status: str
    operator_record_sha256: str

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "dispositions": [self.dispositions[item].manifest for item in self.major_finding_ids],
            "major_finding_ids": list(self.major_finding_ids),
            "open_major_finding_ids": list(self.open_major_finding_ids),
            "operator_record_sha256": self.operator_record_sha256,
            "report_sha256": self.report_sha256,
            "schema_version": "d1-review-adjudication-v1",
            "status": self.status,
        }


def adjudicate_d1_review(
    *,
    report: D1ParsedReviewReport,
    dispositions: tuple[D1MajorFindingDisposition, ...],
    operator_record_sha256: str,
) -> D1ReviewAdjudicationSurface:
    """Account for every parsed finding without conferring successor authority."""
    if type(report) is not D1ParsedReviewReport or type(dispositions) is not tuple:
        raise TypeError("D1 adjudication requires typed report and disposition tuple")
    by_id = {item.finding_id: item for item in dispositions}
    if len(by_id) != len(dispositions) or set(by_id) != set(report.major_finding_ids):
        raise ValueError("D1 adjudication dispositions must exactly cover parsed MAJOR IDs")
    _sha(operator_record_sha256, "D1 adjudication operator record")
    open_ids = tuple(
        finding_id
        for finding_id in report.major_finding_ids
        if by_id[finding_id].status is D1MajorFindingStatus.OPEN
    )
    status = "NOT_REQUIRED" if not report.major_finding_ids else "OPEN" if open_ids else "RESOLVED"
    return D1ReviewAdjudicationSurface(
        report.report_sha256,
        report.major_finding_ids,
        MappingProxyType(by_id),
        open_ids,
        status,
        operator_record_sha256,
    )
