"""D1 review findings are strict, exhaustive, and carry no gate authority."""

from __future__ import annotations

import json

import pytest

from entropic_substrate.evidence.d1_review_adjudication import (
    D1MajorFindingDisposition,
    D1MajorFindingStatus,
    adjudicate_d1_review,
    parse_d1_major_findings,
    parse_d1_review_execution_receipt,
)

pytestmark = pytest.mark.unit


def _receipt(**overrides: object) -> bytes:
    payload: dict[str, object] = {
        "model_id": "review-model-v1",
        "packet_sha256": "1" * 64,
        "prompt_sha256": "2" * 64,
        "provider": "review-provider",
        "report_sha256": "3" * 64,
        "request_digest": "4" * 64,
        "response_sha256": "5" * 64,
        "retry_lineage": [],
        "round": 1,
        "schema_version": "d1-review-execution-receipt-v2",
    }
    payload.update(overrides)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def test_v2_execution_receipt_preserves_raw_response_report_and_retry_lineage() -> None:
    receipt = parse_d1_review_execution_receipt(_receipt(round=2, retry_lineage=["a" * 64]))

    assert receipt.response_sha256 == "5" * 64
    assert receipt.report_sha256 == "3" * 64
    assert receipt.retry_lineage == ("a" * 64,)
    assert receipt.round == 2


def test_major_parser_requires_heading_ids_and_accounts_for_every_reference() -> None:
    report = b"""# D1 review\n\n## MAJOR-1 - Capture mismatch\nDetails.\n\n## MAJOR-3 - Reader drift\nMAJOR-1 remains relevant.\n"""

    parsed = parse_d1_major_findings(report)

    assert parsed.major_finding_ids == ("MAJOR-1", "MAJOR-3")
    assert parsed.report_sha256
    with pytest.raises(ValueError, match="heading"):
        parse_d1_major_findings(b"# Review\nMAJOR-2 appears only in prose.\n")
    with pytest.raises(ValueError, match="duplicate"):
        parse_d1_major_findings(b"## MAJOR-1 A\n## MAJOR-1 B\n")


def test_no_major_marker_is_explicit_and_cannot_contradict_a_finding() -> None:
    parsed = parse_d1_major_findings(b"# Review\n\nReview outcome: NO_MAJOR_FINDINGS\n")
    assert parsed.major_finding_ids == ()

    with pytest.raises(ValueError, match="NO_MAJOR_FINDINGS"):
        parse_d1_major_findings(b"# Review without a finding marker\n")
    with pytest.raises(ValueError, match="contradicts"):
        parse_d1_major_findings(b"Review outcome: NO_MAJOR_FINDINGS\n## MAJOR-1 Contradiction\n")


def test_adjudication_exactly_covers_major_ids_and_has_no_successor_gate() -> None:
    parsed = parse_d1_major_findings(b"## MAJOR-1 Capture mismatch\n## MAJOR-2 Reader drift\n")
    surface = adjudicate_d1_review(
        report=parsed,
        dispositions=(
            D1MajorFindingDisposition(
                "MAJOR-1",
                D1MajorFindingStatus.RESOLVED,
                "Recomputed from captured bytes.",
                "a" * 64,
            ),
            D1MajorFindingDisposition(
                "MAJOR-2",
                D1MajorFindingStatus.OPEN,
                "Independent reader source is still missing.",
                None,
            ),
        ),
        operator_record_sha256="b" * 64,
    )

    assert surface.status == "OPEN"
    assert surface.open_major_finding_ids == ("MAJOR-2",)
    manifest = surface.manifest
    assert "successor_authorization" not in manifest
    assert "eligible_strategies" not in manifest
    with pytest.raises(ValueError, match="exactly cover"):
        adjudicate_d1_review(
            report=parsed,
            dispositions=(
                D1MajorFindingDisposition(
                    "MAJOR-1", D1MajorFindingStatus.OPEN, "Still open.", None
                ),
            ),
            operator_record_sha256="b" * 64,
        )


def test_empty_report_adjudication_is_not_required() -> None:
    report = parse_d1_major_findings(b"Review outcome: NO_MAJOR_FINDINGS\n")
    surface = adjudicate_d1_review(
        report=report,
        dispositions=(),
        operator_record_sha256="c" * 64,
    )

    assert surface.status == "NOT_REQUIRED"
    assert surface.open_major_finding_ids == ()
