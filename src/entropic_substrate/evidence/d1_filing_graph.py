"""Factory-attested semantic authorization for a detached D1 filing."""

from __future__ import annotations

from dataclasses import dataclass

from entropic_substrate.evidence.d1_capture_bundle import D1CaptureBundle, D1ValidatedRunCapture
from entropic_substrate.evidence.d1_filing import D1DetachedFiling, D1FilingManifest
from entropic_substrate.evidence.d1_filing_receipts import D1FilingRevisions
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict
from entropic_substrate.evidence.lifecycle import FiledVerdictEvidence
from entropic_substrate.experiments.d1_primitives import D1Arm, canonical_digest

_VALIDATED_GRAPH_TOKEN = object()


@dataclass(frozen=True, slots=True)
class D1RecomputedIntegrity:
    roster_complete: bool
    pairing_verified: bool
    data_order_verified: bool
    completion_verified: bool
    evidence_digest: str

    @property
    def passed(self) -> bool:
        return all(
            (
                self.roster_complete,
                self.pairing_verified,
                self.data_order_verified,
                self.completion_verified,
            )
        )


def recompute_capture_integrity(bundle: D1CaptureBundle, /) -> D1RecomputedIntegrity:
    """Derive integrity from typed capture evidence, never verdict booleans."""

    if type(bundle) is not D1CaptureBundle:
        raise TypeError("D1 integrity recomputation requires a D1CaptureBundle")
    expected = bundle.roster.expected_run_ids
    observed = frozenset(item.expected.logical_run_id for item in bundle.confirmatory)
    roster_complete = (
        bundle.set_proof.expected_logical_run_ids
        == bundle.set_proof.terminal_journal_run_ids
        == bundle.set_proof.checkpoint_receipt_run_ids
        == bundle.set_proof.capture_input_run_ids
        == expected
        == observed
    )
    by_pair: dict[str, list[D1ValidatedRunCapture]] = {}
    for capture in bundle.confirmatory:
        by_pair.setdefault(capture.expected.pairing_id, []).append(capture)
    pairing_verified = all(
        len(block) == len(D1Arm) and {item.expected.arm for item in block} == set(D1Arm)
        for block in by_pair.values()
    ) and set(by_pair) == set(bundle.roster.pairing_ids)
    data_order_verified = all(
        len({item.data_order_segments for item in block}) == 1 for block in by_pair.values()
    )
    completion_verified = all(
        item.terminal_step > 0 and bool(item.checkpoint_receipts) for item in bundle.confirmatory
    )
    facts = {
        "capture_manifest_digest": bundle.manifest_digest,
        "completion_verified": completion_verified,
        "data_order_verified": data_order_verified,
        "pair_exclusions": [item.manifest for item in bundle.pair_exclusions],
        "pairing_verified": pairing_verified,
        "roster_complete": roster_complete,
        "schema_version": "d1-recomputed-integrity-v1",
    }
    return D1RecomputedIntegrity(
        roster_complete=roster_complete,
        pairing_verified=pairing_verified,
        data_order_verified=data_order_verified,
        completion_verified=completion_verified,
        evidence_digest=canonical_digest(facts),
    )


@dataclass(frozen=True, slots=True, init=False)
class ValidatedD1Filing:
    """The sole D1 filing type permitted to reach the successor gate."""

    detached: D1DetachedFiling
    revisions: D1FilingRevisions
    integrity: D1RecomputedIntegrity
    graph_digest: str

    def __init__(
        self,
        *,
        detached: D1DetachedFiling | None = None,
        revisions: D1FilingRevisions | None = None,
        integrity: D1RecomputedIntegrity | None = None,
        graph_digest: str | None = None,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _VALIDATED_GRAPH_TOKEN:
            raise TypeError("ValidatedD1Filing is factory-only; verify the semantic evidence graph")
        if (
            type(detached) is not D1DetachedFiling
            or type(revisions) is not D1FilingRevisions
            or type(integrity) is not D1RecomputedIntegrity
            or type(graph_digest) is not str
        ):
            raise TypeError("validated D1 filing requires exact graph values")
        object.__setattr__(self, "detached", detached)
        object.__setattr__(self, "revisions", revisions)
        object.__setattr__(self, "integrity", integrity)
        object.__setattr__(self, "graph_digest", graph_digest)

    @property
    def verdict(self) -> ValidatedD1Verdict:
        return self.detached.verdict

    @property
    def manifest(self) -> D1FilingManifest:
        return self.detached.manifest

    def to_filed_evidence(self) -> FiledVerdictEvidence:
        return FiledVerdictEvidence(
            narrative=self.manifest.artifacts["narrative"].artifact_identity(),
            machine=self.manifest.artifacts["machine_verdict"].artifact_identity(),
            review_packet=self.manifest.artifacts["review_packet"].artifact_identity(),
            open_major_findings=self.verdict.review.open_major_findings,
            successor_authorization=self.verdict.successor.authorization,
            eligible_strategies=self.verdict.successor.eligible_strategies,
        )
