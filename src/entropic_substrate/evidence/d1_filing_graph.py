"""Factory-attested semantic authorization for a detached D1 filing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn

from entropic_substrate.evidence.d1_capture_bundle import D1CaptureBundle, D1ValidatedRunCapture
from entropic_substrate.evidence.d1_filing import D1DetachedFiling, D1FilingManifest
from entropic_substrate.evidence.d1_filing_edges import verify_filing_edges
from entropic_substrate.evidence.d1_filing_receipts import (
    D1EvidenceGap,
    D1EvidenceGapError,
    D1FilingArtifactPort,
    D1FilingGitPort,
    D1FilingGraphError,
    D1FilingRevisions,
)
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


def _gap(role: str, field: str, measurement: str) -> NoReturn:
    raise D1EvidenceGapError(D1EvidenceGap(role, field, measurement))


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise D1FilingGraphError(code=code, path=path, detail=detail)


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(payload), allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()


def _read_payloads(
    detached: D1DetachedFiling, artifacts: D1FilingArtifactPort
) -> Mapping[str, bytes]:
    if not detached.artifact_root_verified:
        _gap("detached_filing", "artifact_root_verified", "rehashing against actual filed bytes")
    payloads: dict[str, bytes] = {}
    for role, identity in detached.manifest.artifacts.items():
        payload = artifacts.read_bytes(identity.path)
        if payload is None:
            _gap(role, identity.path, "the actual filed artifact bytes")
        payloads[role] = payload
    return payloads


def _verify_machine_inputs(detached: D1DetachedFiling, bundle: D1CaptureBundle) -> None:
    expected = {
        item.normalized_path: (item.md5, item.sha256) for item in bundle.dvc_handoff.artifacts
    }
    actual: dict[str, tuple[str, str]] = {}
    for index, item in enumerate(detached.verdict.inputs):
        sha256 = item.sha256
        if sha256 is None:
            _gap(
                "machine_verdict",
                f"inputs[{index}].sha256",
                "the SHA-256 of every authoritative capture input",
            )
        actual[item.path] = (item.md5.digest, sha256.digest)
    if len(actual) != len(detached.verdict.inputs) or actual != expected:
        _fail(
            "machine_capture_inputs",
            "machine_verdict.inputs",
            "must equal the actual capture manifest path, MD5, and SHA-256 set",
        )


def _verify_exclusions(detached: D1DetachedFiling, bundle: D1CaptureBundle) -> None:
    expected = {
        (
            item.pairing_id,
            item.reason.value,
            canonical_digest(item.manifest),
        )
        for item in bundle.pair_exclusions
    }
    actual = {
        (subject, reason, evidence_hash.digest)
        for subject, reason, evidence_hash in detached.verdict.exclusions
        if evidence_hash.algorithm == "sha256"
    }
    if actual != expected or len(actual) != len(detached.verdict.exclusions):
        _fail("exclusions", "machine_verdict.exclusions", "must equal recomputed pair exclusions")


def _verify_capture_semantics(
    detached: D1DetachedFiling,
    bundle: D1CaptureBundle,
    payloads: Mapping[str, bytes],
    integrity: D1RecomputedIntegrity,
) -> None:
    if payloads["raw_capture_manifest"] != _canonical_json(bundle.manifest):
        _fail(
            "capture_manifest",
            "raw_capture_manifest",
            "file bytes must equal the factory-attested capture manifest",
        )
    if not bundle.dvc_handoff.analysis_ready:
        _gap("raw_capture_manifest", "dvc_handoff.analysis_ready", "bound DVC pointer objects")
    prereg = detached.verdict.preregistration
    roster = bundle.roster
    if (
        prereg.path != roster.preregistration_path
        or prereg.commit != roster.preregistration_commit
        or prereg.family_digest != roster.family_digest
    ):
        _fail("capture_preregistration", "capture.roster", "differs from machine preregistration")
    _verify_machine_inputs(detached, bundle)
    _verify_exclusions(detached, bundle)
    claimed = detached.verdict.integrity
    facts = (
        claimed.roster_complete,
        claimed.pairing_verified,
        claimed.data_order_verified,
        claimed.completion_verified,
    )
    recomputed = (
        integrity.roster_complete,
        integrity.pairing_verified,
        integrity.data_order_verified,
        integrity.completion_verified,
    )
    if (
        facts != recomputed
        or claimed.status != ("PASS" if integrity.passed else "FAIL")
        or claimed.evidence_hash.algorithm != "sha256"
        or claimed.evidence_hash.digest != integrity.evidence_digest
    ):
        _fail("integrity", "machine_verdict.integrity", "differs from recomputed capture facts")
    observed_n = len(bundle.roster.pairing_ids) - len(bundle.pair_exclusions)
    if any(contrast.n != observed_n for contrast in detached.verdict.contrasts):
        _fail("sample_size", "machine_verdict.contrasts", "n differs from nonexcluded pairs")


def verify_d1_filing(
    detached: D1DetachedFiling,
    capture: D1CaptureBundle,
    revisions: D1FilingRevisions,
    artifacts: D1FilingArtifactPort,
    git: D1FilingGitPort,
    /,
) -> ValidatedD1Filing:
    """Verify the full evidence graph without recomputing the scientific verdict."""

    if type(detached) is not D1DetachedFiling or type(capture) is not D1CaptureBundle:
        raise TypeError("D1 graph verification requires detached filing and capture bundle values")
    if type(revisions) is not D1FilingRevisions:
        raise TypeError("D1 graph verification requires typed filing revisions")
    payloads = _read_payloads(detached, artifacts)
    integrity = recompute_capture_integrity(capture)
    _verify_capture_semantics(detached, capture, payloads, integrity)
    edges = verify_filing_edges(detached.verdict, detached.manifest, revisions, payloads, git)
    graph_digest = canonical_digest(
        {
            "capture_manifest_digest": capture.manifest_digest,
            "edge_digest": edges.digest,
            "integrity_digest": integrity.evidence_digest,
            "schema_version": "d1-verified-filing-graph-v1",
        }
    )
    return ValidatedD1Filing(
        detached=detached,
        revisions=revisions,
        integrity=integrity,
        graph_digest=graph_digest,
        _factory_token=_VALIDATED_GRAPH_TOKEN,
    )
