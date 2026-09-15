"""Strict typed receipts for every semantic edge in a D1 filing."""

from __future__ import annotations

import json

import pytest

from entropic_substrate.evidence.d1_filing_receipts import (
    D1FilingGraphError,
    D1FilingRevisions,
    parse_alias_commitment,
    parse_alias_reveal,
    parse_analysis_receipt,
    parse_review_receipt,
)

pytestmark = pytest.mark.unit


def _json(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _analysis_receipt() -> dict[str, object]:
    return {
        "alias_commitment": "a" * 64,
        "analysis_digest": "5" * 64,
        "bindings": {
            "capture_digest": "6" * 64,
            "decision_contract_digest": "7" * 64,
            "design_digest": "8" * 64,
        },
        "filing_bindings": {
            "capture_manifest_sha256": "1" * 64,
            "isolation_attestation_sha256": "4" * 64,
            "output_sha256": "5" * 64,
            "preregistration_sha256": "2" * 64,
            "reader_sha256": "3" * 64,
        },
        "filing_eligible": True,
        "implementation": {"analysis_code_digest": "9" * 64},
        "inputs": [{"sha256": "b" * 64}],
        "isolation": {"executor_kind": "external_enforced"},
        "schema_version": "d1-analysis-execution-receipt-v1",
    }


def test_analysis_receipt_binds_every_required_edge() -> None:
    receipt = parse_analysis_receipt(_json(_analysis_receipt()))

    assert receipt.filing_bindings is not None
    assert receipt.filing_bindings.reader_sha256 == "3" * 64
    assert receipt.filing_bindings.isolation_attestation_sha256 == "4" * 64
    assert receipt.manifest["analysis_digest"] == "5" * 64


def test_analysis_receipt_hard_cuts_synthetic_and_incomplete_schemas() -> None:
    incomplete = _analysis_receipt()
    del incomplete["filing_bindings"]

    with pytest.raises(D1FilingGraphError, match="analysis execution receipt"):
        parse_analysis_receipt(_json(incomplete))
    with pytest.raises(D1FilingGraphError, match="analysis execution receipt"):
        parse_analysis_receipt(
            _json(
                {
                    "capture_manifest_sha256": "1" * 64,
                    "isolation_attestation_sha256": "4" * 64,
                    "output_sha256": "5" * 64,
                    "preregistration_sha256": "2" * 64,
                    "reader_sha256": "3" * 64,
                    "schema_version": "d1-analysis-receipt-v1",
                }
            )
        )


def test_alias_and_review_receipts_preserve_exact_binding_values() -> None:
    commitment = parse_alias_commitment(
        _json({"schema_version": "d1-alias-commitment-v1", "reveal_sha256": "6" * 64})
    )
    reveal = parse_alias_reveal(
        _json(
            {
                "schema_version": "d1-alias-reveal-v1",
                "mapping": {"A": "none", "B": "sg", "C": "ac", "D": "both"},
                "nonce": "sealed-before-review",
            }
        )
    )
    review = parse_review_receipt(
        _json(
            {
                "model_id": "review-model",
                "packet_sha256": "7" * 64,
                "prompt_sha256": "8" * 64,
                "provider": "review-provider",
                "report_sha256": "9" * 64,
                "request_digest": "a" * 64,
                "response_sha256": "b" * 64,
                "retry_lineage": ["c" * 64],
                "round": 2,
                "schema_version": "d1-review-execution-receipt-v2",
            }
        )
    )

    assert commitment.reveal_sha256 == "6" * 64
    assert reveal.mapping["D"] == "both"
    assert review.response_sha256 == "b" * 64
    assert review.retry_lineage == ("c" * 64,)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "schema_version": "d1-review-receipt-v1",
                "packet_sha256": "7" * 64,
                "prompt_sha256": "8" * 64,
                "report_sha256": "9" * 64,
                "model_id": "review-model",
                "provider": "review-provider",
                "round": 1,
                "major_finding_ids": [],
            },
            "schema",
        ),
        (
            {
                "schema_version": "d1-review-execution-receipt-v2",
                "packet_sha256": "7" * 64,
                "prompt_sha256": "8" * 64,
                "report_sha256": "9" * 64,
                "model_id": "review-model",
                "provider": "review-provider",
                "round": 1,
                "request_digest": "a" * 64,
                "response_sha256": "b" * 64,
                "retry_lineage": [],
                "major_finding_ids": [],
            },
            "schema_fields",
        ),
    ],
)
def test_review_receipt_hard_cuts_v1_and_mixed_wire_schemas(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(D1FilingGraphError, match=message):
        parse_review_receipt(_json(payload))


def test_filing_revisions_require_six_distinct_commit_identities() -> None:
    values = [f"{index:x}" * 40 for index in range(1, 7)]
    revisions = D1FilingRevisions(*values)
    assert revisions.ordered == tuple(values)

    with pytest.raises(ValueError, match="distinct"):
        D1FilingRevisions(*[*values[:-1], values[-2]])
