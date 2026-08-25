"""Every semantic edge in the D1 filing graph is independently load-bearing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise

import pytest

from d1_verdict_v2_fixtures import decision_contract, machine_verdict
from entropic_substrate.evidence.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    D1FilingManifest,
    canonical_machine_verdict_bytes,
)
from entropic_substrate.evidence.d1_filing_edges import verify_filing_edges
from entropic_substrate.evidence.d1_filing_receipts import (
    D1EvidenceGapError,
    D1FilingGraphError,
    D1FilingRevisions,
)
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict

pytestmark = pytest.mark.unit


def _json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass
class FakeGit:
    files: dict[tuple[str, str], bytes]
    ancestors: set[tuple[str, str]]

    def read_file(self, revision: str, path: str) -> bytes | None:
        return self.files.get((revision, path))

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return (ancestor, descendant) in self.ancestors


def _case(mutation: str | None = None):
    machine = machine_verdict()
    machine_bytes = canonical_machine_verdict_bytes(machine)
    reveal = _json(
        {
            "mapping": {"A": "none", "B": "sg", "C": "ac", "D": "both"},
            "nonce": "sealed-before-review",
            "schema_version": "d1-alias-reveal-v1",
        }
    )
    commitment = _json({"reveal_sha256": _sha(reveal), "schema_version": "d1-alias-commitment-v1"})
    review_fields: dict[str, object] = {
        "model_id": "review-model",
        "packet_sha256": "",
        "prompt_sha256": "",
        "provider": "review-provider",
        "report_sha256": "",
        "request_digest": "a" * 64,
        "response_sha256": "b" * 64,
        "retry_lineage": [],
        "round": 1,
        "schema_version": "d1-review-execution-receipt-v2",
    }
    payloads: dict[str, bytes] = {role: role.encode() for role in REQUIRED_D1_FILING_ARTIFACTS}
    payloads.update(
        {
            "machine_verdict": machine_bytes,
            "narrative": (
                f"# D1 verdict\n<!-- d1-machine-verdict-sha256:{_sha(machine_bytes)} -->\n"
            ).encode(),
            "preregistration": b"registered design bytes\n",
            "raw_capture_manifest": b'{"capture":"bound"}',
            "blinded_analysis": b'{"decision":"aliased"}',
            "analysis_reader": b"def read_capture(): ...\n",
            "analysis_isolation_attestation": b'{"session":"fresh","network":"disabled"}',
            "review_packet": _json(
                {
                    "alias_commitment": _sha(commitment),
                    "schema_version": "d1-blind-review-packet-v1",
                }
            ),
            "reviewer_prompt": b"registered prompt",
            "reviewer_report": b"Review outcome: NO_MAJOR_FINDINGS\n",
            "alias_reveal": reveal,
        }
    )
    payloads["alias_commitment"] = commitment
    if mutation == "packet_alias_lineage":
        payloads["review_packet"] = _json(
            {
                "alias_commitment": "0" * 64,
                "schema_version": "d1-blind-review-packet-v1",
            }
        )
    elif mutation == "packet_invalid_json":
        payloads["review_packet"] = b"\xff"
    elif mutation == "packet_schema":
        payloads["review_packet"] = _json(
            {
                "alias_commitment": _sha(commitment),
                "schema_version": "d1-blind-review-packet-v9",
            }
        )
    elif mutation == "packet_noncanonical":
        payloads["review_packet"] = json.dumps(
            {
                "alias_commitment": _sha(commitment),
                "schema_version": "d1-blind-review-packet-v1",
            },
            indent=2,
        ).encode()
    elif mutation == "packet_nonfinite":
        payloads["review_packet"] = (
            b'{"alias_commitment":"'
            + _sha(commitment).encode()
            + b'","schema_version":"d1-blind-review-packet-v1","x":NaN}'
        )
    review_fields.update(
        {
            "packet_sha256": _sha(payloads["review_packet"]),
            "prompt_sha256": _sha(payloads["reviewer_prompt"]),
            "report_sha256": _sha(payloads["reviewer_report"]),
        }
    )
    if mutation == "major_ids":
        payloads["reviewer_report"] = b"## MAJOR-1 - Unresolved evidence defect\n"
        review_fields["report_sha256"] = _sha(payloads["reviewer_report"])
    elif mutation == "missing_model":
        del review_fields["model_id"]
    elif mutation == "mixed_review_schema":
        review_fields["major_finding_ids"] = []
    payloads["reviewer_execution_receipt"] = _json(review_fields)
    analysis_fields = {
        "capture_manifest_sha256": _sha(payloads["raw_capture_manifest"]),
        "isolation_attestation_sha256": _sha(payloads["analysis_isolation_attestation"]),
        "output_sha256": _sha(payloads["blinded_analysis"]),
        "preregistration_sha256": _sha(payloads["preregistration"]),
        "reader_sha256": _sha(payloads["analysis_reader"]),
        "schema_version": "d1-analysis-receipt-v1",
    }
    if mutation == "analysis_binding":
        analysis_fields["output_sha256"] = "0" * 64
    payloads["analysis_receipt"] = _json(analysis_fields)
    if mutation == "alias_bijection":
        alias = json.loads(payloads["alias_reveal"])
        alias["mapping"]["D"] = "sg"
        payloads["alias_reveal"] = _json(alias)
        payloads["alias_commitment"] = _json(
            {
                "reveal_sha256": _sha(payloads["alias_reveal"]),
                "schema_version": "d1-alias-commitment-v1",
            }
        )
        packet = json.loads(payloads["review_packet"])
        packet["alias_commitment"] = _sha(payloads["alias_commitment"])
        payloads["review_packet"] = _json(packet)
        receipt = json.loads(payloads["reviewer_execution_receipt"])
        receipt["packet_sha256"] = _sha(payloads["review_packet"])
        payloads["reviewer_execution_receipt"] = _json(receipt)
    elif mutation == "narrative":
        payloads["narrative"] = b"# unbound narrative\n"
    artifacts = {
        role: {
            "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            "path": f"filing/{role}.bin",
            "sha256": _sha(payload),
            "size": len(payload),
        }
        for role, payload in payloads.items()
    }
    manifest = D1FilingManifest.from_mapping(
        {"artifacts": artifacts, "rung": "d1", "schema_version": "d1-filing-manifest/v2"}
    )
    verdict = ValidatedD1Verdict.from_mapping(machine, decision_contract=decision_contract())
    revisions = D1FilingRevisions("b" * 40, "c" * 40, "d" * 40, "e" * 40, "1" * 40, "2" * 40)
    chain = (verdict.preregistration.commit, *revisions.ordered)
    git = FakeGit({}, set(pairwise(chain)))
    git.files[(verdict.preregistration.commit, verdict.preregistration.path)] = payloads[
        "preregistration"
    ]
    stages = {
        revisions.capture: ("raw_capture_manifest",),
        revisions.analysis: (
            "blinded_analysis",
            "analysis_receipt",
            "analysis_reader",
            "analysis_isolation_attestation",
        ),
        revisions.alias_commitment: ("alias_commitment",),
        revisions.review: (
            "review_packet",
            "reviewer_prompt",
            "reviewer_report",
            "reviewer_execution_receipt",
        ),
        revisions.alias_reveal: ("alias_reveal",),
        revisions.filing: tuple(REQUIRED_D1_FILING_ARTIFACTS),
    }
    for revision, roles in stages.items():
        for role in roles:
            git.files[(revision, manifest.artifacts[role].path)] = payloads[role]
    return verdict, manifest, revisions, payloads, git


def test_complete_semantic_edge_graph_is_verified() -> None:
    evidence = verify_filing_edges(*_case())
    assert evidence.review_model_id == "review-model"
    assert evidence.alias_mapping["A"] == "none"
    assert len(evidence.digest) == 64


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("analysis_binding", D1FilingGraphError, "analysis"),
        ("alias_bijection", D1FilingGraphError, "bijection"),
        ("missing_model", D1EvidenceGapError, "model_id"),
        ("major_ids", D1FilingGraphError, "MAJOR"),
        ("mixed_review_schema", D1FilingGraphError, "schema_fields"),
        ("packet_alias_lineage", D1FilingGraphError, "packet"),
        ("packet_invalid_json", D1FilingGraphError, "review_packet"),
        ("packet_schema", D1FilingGraphError, "review_packet"),
        ("packet_noncanonical", D1FilingGraphError, "review_packet"),
        ("packet_nonfinite", D1FilingGraphError, "review_packet"),
        ("narrative", D1FilingGraphError, "narrative"),
    ],
)
def test_each_semantic_binding_has_guard_teeth(
    mutation: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        verify_filing_edges(*_case(mutation))


def test_git_precedence_prereg_bytes_and_final_commit_are_required() -> None:
    verdict, manifest, revisions, payloads, git = _case()
    git.ancestors.remove((revisions.alias_commitment, revisions.review))
    with pytest.raises(D1FilingGraphError, match="ancestry"):
        verify_filing_edges(verdict, manifest, revisions, payloads, git)

    verdict, manifest, revisions, payloads, git = _case()
    git.files[(verdict.preregistration.commit, verdict.preregistration.path)] = b"drift"
    with pytest.raises(D1FilingGraphError, match="preregistration"):
        verify_filing_edges(verdict, manifest, revisions, payloads, git)

    verdict, manifest, revisions, payloads, git = _case()
    del git.files[(revisions.filing, manifest.artifacts["narrative"].path)]
    with pytest.raises(D1EvidenceGapError, match="committed"):
        verify_filing_edges(verdict, manifest, revisions, payloads, git)
