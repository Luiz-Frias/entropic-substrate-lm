"""Factory-attested semantic authorization for a detached D1 filing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn

from entropic_substrate.__legacy.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
    D1ValidatedTerminalDefect,
)
from entropic_substrate.__legacy.evidence.d1_capture_bundle import D1ValidatedRunCapture
from entropic_substrate.__legacy.evidence.d1_capture_roster import (
    D1PairExclusion,
    D1PairExclusionReason,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, canonical_digest
from entropic_substrate.kernel.lifecycle import FiledVerdictEvidence
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.ladder.file.d1_filing import D1DetachedFiling, D1FilingManifest
from entropic_substrate.ladder.file.d1_filing_edges import verify_filing_edges
from entropic_substrate.ladder.file.d1_filing_receipts import (
    D1EvidenceGap,
    D1EvidenceGapError,
    D1FilingArtifactPort,
    D1FilingGitPort,
    D1FilingGraphError,
    D1FilingRevisions,
)
from entropic_substrate.ladder.file.d1_validator import ValidatedD1Verdict

_VALIDATED_GRAPH_TOKEN = object()


@dataclass(frozen=True, slots=True)
class D1RecomputedIntegrity:
    roster_complete: bool
    pairing_verified: bool
    data_order_verified: bool
    completion_verified: bool
    exclusions_verified: bool
    effective_sample_size: int
    applied_exclusions: tuple[tuple[D1PairExclusion, str, str], ...] | None
    evidence_digest: str

    @property
    def passed(self) -> bool:
        return all(
            (
                self.roster_complete,
                self.pairing_verified,
                self.data_order_verified,
                self.completion_verified,
                self.exclusions_verified,
            )
        )


def _derive_pair_exclusions(
    capture: D1HeterogeneousCaptureAggregate,
) -> tuple[D1PairExclusion, ...]:
    defect_pairs = sorted({item.expected.pairing_id for item in capture.terminal_defects})
    return capture.roster.validate_pair_exclusions(
        tuple(
            D1PairExclusion(
                pairing_id=pairing_id,
                member_run_ids=tuple(
                    item.logical_run_id
                    for item in capture.roster.expected_confirmatory
                    if item.pairing_id == pairing_id
                ),
                reason=D1PairExclusionReason.NONFINITE_FINAL_LOSS,
                evidence_paths=tuple(
                    sorted(
                        item.capture_input.journal.normalized_path
                        for item in capture.terminal_defects
                        if item.expected.pairing_id == pairing_id
                    )
                ),
            )
            for pairing_id in defect_pairs
        )
    )


def _derive_registered_exclusions(
    capture: D1HeterogeneousCaptureAggregate,
    decision_contract: D1DecisionContract,
    claims: tuple[D1PairExclusion, ...],
) -> tuple[tuple[D1PairExclusion, str, str], ...] | None:
    defects_by_pair: dict[str, list[D1ValidatedTerminalDefect]] = {}
    for defect in capture.terminal_defects:
        defects_by_pair.setdefault(defect.expected.pairing_id, []).append(defect)
    applied: list[tuple[D1PairExclusion, str, str]] = []
    for claim in claims:
        defects = defects_by_pair[claim.pairing_id]
        reason_ids: list[str] = []
        for defect in defects:
            matches = tuple(
                rule.reason_id
                for rule in decision_contract.exclusion_rules
                if rule.measurement == defect.record.metric_name
                and rule.predicate == "NONFINITE"
                and rule.scope == "WHOLE_SEED_BLOCK"
            )
            if len(matches) != 1:
                return None
            reason_ids.append(matches[0])
        if len(set(reason_ids)) != 1:
            return None
        applied.append(
            (
                claim,
                reason_ids[0],
                canonical_digest(
                    {
                        "defects": [
                            item.manifest
                            for item in sorted(
                                defects, key=lambda value: value.expected.logical_run_id
                            )
                        ]
                    }
                ),
            )
        )
    return tuple(applied)


def _contract_matches_capture(
    capture: D1HeterogeneousCaptureAggregate,
    decision_contract: D1DecisionContract,
) -> bool:
    raw_seeds = decision_contract.registered_constants.get("seeds")
    roster_seeds = {item.seed for item in capture.roster.expected_confirmatory}
    return (
        decision_contract.family_identity.family_digest == capture.roster.family_digest
        and type(raw_seeds) is tuple
        and all(type(seed) is int for seed in raw_seeds)
        and set(raw_seeds) == roster_seeds
        and len(raw_seeds) == len(roster_seeds)
        and all(
            item.construction_digest
            == decision_contract.family_identity.arm_construction_digests[item.arm.value]
            for item in capture.roster.expected_confirmatory
        )
    )


def _pairing_facts(
    capture: D1HeterogeneousCaptureAggregate,
) -> tuple[dict[str, set[str]], bool, bool]:
    expected_by_pair = {
        pairing_id: {
            item.logical_run_id
            for item in capture.roster.expected_confirmatory
            if item.pairing_id == pairing_id
        }
        for pairing_id in capture.roster.pairing_ids
    }
    observed_by_pair: dict[str, list[D1ValidatedRunCapture | D1ValidatedTerminalDefect]] = {}
    for item in (*capture.complete_runs, *capture.terminal_defects):
        observed_by_pair.setdefault(item.expected.pairing_id, []).append(item)
    pairing_verified = all(
        len(block) == len(D1Arm)
        and {item.expected.arm for item in block} == set(D1Arm)
        and {item.expected.logical_run_id for item in block} == expected_by_pair[pairing_id]
        for pairing_id, block in observed_by_pair.items()
    ) and set(observed_by_pair) == set(expected_by_pair)
    data_order_verified = all(
        len({item.data_order_segments for item in block}) == 1
        for block in observed_by_pair.values()
    )
    return expected_by_pair, pairing_verified, data_order_verified


def recompute_capture_integrity(
    capture: D1HeterogeneousCaptureAggregate,
    decision_contract: D1DecisionContract,
    /,
) -> D1RecomputedIntegrity:
    """Derive integrity from typed capture evidence, never verdict booleans."""

    if (
        type(capture) is not D1HeterogeneousCaptureAggregate
        or type(decision_contract) is not D1DecisionContract
    ):
        raise TypeError(
            "D1 integrity recomputation requires a heterogeneous capture aggregate and contract"
        )
    expected = capture.roster.expected_run_ids
    complete_ids = frozenset(item.expected.logical_run_id for item in capture.complete_runs)
    defect_ids = frozenset(item.expected.logical_run_id for item in capture.terminal_defects)
    proof = capture.partition_proof
    roster_complete = _contract_matches_capture(capture, decision_contract) and (
        proof.expected_logical_run_ids == expected
        and proof.complete_run_ids == complete_ids
        and proof.terminal_defect_run_ids == defect_ids
        and not complete_ids & defect_ids
        and complete_ids | defect_ids == expected
    )
    expected_by_pair, pairing_verified, data_order_verified = _pairing_facts(capture)
    completion_verified = all(
        item.terminal_step > 0 and bool(item.checkpoint_receipts) for item in capture.complete_runs
    )
    derived_exclusions = _derive_pair_exclusions(capture)
    applied_exclusions = _derive_registered_exclusions(
        capture, decision_contract, derived_exclusions
    )
    exclusions_verified = (
        derived_exclusions == capture.pair_exclusions and applied_exclusions is not None
    )
    defect_pairs = {item.expected.pairing_id for item in capture.terminal_defects}
    effective_sample_size = sum(
        pairing_id not in defect_pairs and expected_run_ids <= complete_ids
        for pairing_id, expected_run_ids in expected_by_pair.items()
    )
    facts = {
        "capture_manifest_digest": capture.manifest_digest,
        "completion_verified": completion_verified,
        "data_order_verified": data_order_verified,
        "effective_sample_size": effective_sample_size,
        "exclusions_verified": exclusions_verified,
        "applied_exclusions": [
            {
                "evidence_digest": evidence_digest,
                "pairing_id": claim.pairing_id,
                "reason_id": reason_id,
            }
            for claim, reason_id, evidence_digest in applied_exclusions or ()
        ],
        "pairing_verified": pairing_verified,
        "partition_proof": capture.partition_proof.manifest,
        "roster_complete": roster_complete,
        "schema_version": "d1-recomputed-integrity-v2",
    }
    return D1RecomputedIntegrity(
        roster_complete=roster_complete,
        pairing_verified=pairing_verified,
        data_order_verified=data_order_verified,
        completion_verified=completion_verified,
        exclusions_verified=exclusions_verified,
        effective_sample_size=effective_sample_size,
        applied_exclusions=applied_exclusions,
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
            eligible_arms=self.verdict.successor.eligible_arms,
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


def _verify_machine_inputs(
    detached: D1DetachedFiling, capture: D1HeterogeneousCaptureAggregate
) -> None:
    expected = {
        item.normalized_path: (item.md5, item.sha256) for item in capture.dvc_handoff.artifacts
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


def _verify_exclusions(
    detached: D1DetachedFiling,
    integrity: D1RecomputedIntegrity,
) -> None:
    applied = integrity.applied_exclusions
    if applied is None:
        _fail("exclusions", "capture.terminal_defects", "lack one registered exclusion rule")
    expected = {
        (
            claim.pairing_id,
            reason_id,
            evidence_digest,
        )
        for claim, reason_id, evidence_digest in applied
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
    capture: D1HeterogeneousCaptureAggregate,
    payloads: Mapping[str, bytes],
    integrity: D1RecomputedIntegrity,
) -> None:
    if payloads["raw_capture_manifest"] != _canonical_json(capture.manifest):
        _fail(
            "capture_manifest",
            "raw_capture_manifest",
            "file bytes must equal the factory-attested capture manifest",
        )
    if not capture.dvc_handoff.analysis_ready:
        _gap("raw_capture_manifest", "dvc_handoff.analysis_ready", "bound DVC pointer objects")
    prereg = detached.verdict.preregistration
    roster = capture.roster
    if (
        prereg.path != roster.preregistration_path
        or prereg.commit != roster.preregistration_commit
        or prereg.family_digest != roster.family_digest
    ):
        _fail("capture_preregistration", "capture.roster", "differs from machine preregistration")
    _verify_machine_inputs(detached, capture)
    _verify_exclusions(detached, integrity)
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
    observed_n = integrity.effective_sample_size
    if any(contrast.n != observed_n for contrast in detached.verdict.contrasts):
        _fail("sample_size", "machine_verdict.contrasts", "n differs from nonexcluded pairs")


def verify_d1_filing(
    detached: D1DetachedFiling,
    capture: D1HeterogeneousCaptureAggregate,
    revisions: D1FilingRevisions,
    artifacts: D1FilingArtifactPort,
    git: D1FilingGitPort,
    /,
) -> ValidatedD1Filing:
    """Verify the full evidence graph without recomputing the scientific verdict."""

    if (
        type(detached) is not D1DetachedFiling
        or type(capture) is not D1HeterogeneousCaptureAggregate
    ):
        raise TypeError(
            "D1 graph verification requires detached filing and heterogeneous capture values"
        )
    if type(revisions) is not D1FilingRevisions:
        raise TypeError("D1 graph verification requires typed filing revisions")
    payloads = _read_payloads(detached, artifacts)
    integrity = recompute_capture_integrity(capture, detached.decision_contract)
    _verify_capture_semantics(detached, capture, payloads, integrity)
    edges = verify_filing_edges(detached.verdict, detached.manifest, revisions, payloads, git)
    graph_digest = canonical_digest(
        {
            "capture_manifest_digest": capture.manifest_digest,
            "edge_digest": edges.digest,
            "integrity_digest": integrity.evidence_digest,
            "schema_version": "d1-verified-filing-graph-v2",
        }
    )
    return ValidatedD1Filing(
        detached=detached,
        revisions=revisions,
        integrity=integrity,
        graph_digest=graph_digest,
        _factory_token=_VALIDATED_GRAPH_TOKEN,
    )
