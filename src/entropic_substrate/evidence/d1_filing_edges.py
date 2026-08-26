"""Pure verification of semantic and Git edges in a D1 filing graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import NoReturn, cast

from entropic_substrate.analysis.d1_machine_analysis_codec import (
    D1MachineAnalysisCodecError,
    decode_d1_machine_analysis_declaration,
)
from entropic_substrate.evidence.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    D1FilingManifest,
)
from entropic_substrate.evidence.d1_filing_receipts import (
    D1EvidenceGap,
    D1EvidenceGapError,
    D1FilingGitPort,
    D1FilingGraphError,
    D1FilingRevisions,
    parse_alias_commitment,
    parse_alias_reveal,
    parse_analysis_receipt,
    parse_review_receipt,
)
from entropic_substrate.evidence.d1_review_adjudication import parse_d1_major_findings
from entropic_substrate.evidence.d1_types import (
    D1_ARM_IDS,
    D1_OPAQUE_ARM_ALIAS_IDS,
    D1_STABILIZATION_ARM_IDS,
)
from entropic_substrate.evidence.d1_validator import ValidatedD1Verdict
from entropic_substrate.evidence.lifecycle import SuccessorAuthorization
from entropic_substrate.experiments.d1_primitives import canonical_digest


@dataclass(frozen=True, slots=True)
class D1VerifiedFilingEdges:
    alias_mapping: Mapping[str, str]
    feasible_arms: tuple[str, ...]
    review_model_id: str
    review_provider: str
    review_round: int
    digest: str


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fail(code: str, path: str, detail: str) -> NoReturn:
    raise D1FilingGraphError(code=code, path=path, detail=detail)


def _gap(role: str, field: str, measurement: str) -> None:
    raise D1EvidenceGapError(D1EvidenceGap(role, field, measurement))


def _verify_payloads(manifest: D1FilingManifest, payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != set(REQUIRED_D1_FILING_ARTIFACTS):
        _fail("payload_roles", "payloads", "must contain every detached artifact role exactly")
    for role, artifact in manifest.artifacts.items():
        payload = payloads[role]
        if type(payload) is not bytes:
            _fail("payload_type", role, "must be exact bytes")
        if (
            len(payload) != artifact.size
            or _sha(payload) != artifact.sha256.digest
            or hashlib.md5(payload, usedforsecurity=False).hexdigest() != artifact.md5.digest
        ):
            _fail("payload_drift", role, "bytes differ from the detached manifest")


def _git_edge(
    git: D1FilingGitPort,
    revision: str,
    path: str,
    expected: bytes,
    *,
    role: str,
) -> None:
    actual = git.read_file(revision, path)
    if actual is None:
        _gap("git", f"{revision}:{path}", f"committed {role} bytes at the declared revision")
    if actual != expected:
        _fail("git_blob_mismatch", role, "committed bytes differ from the filed artifact")


def _verify_git(
    verdict: ValidatedD1Verdict,
    manifest: D1FilingManifest,
    revisions: D1FilingRevisions,
    payloads: Mapping[str, bytes],
    git: D1FilingGitPort,
) -> None:
    chain = (verdict.preregistration.commit, *revisions.ordered)
    for ancestor, descendant in pairwise(chain):
        if not git.is_ancestor(ancestor, descendant):
            _fail("git_ancestry", descendant, f"must descend from {ancestor}")
    prereg = verdict.preregistration
    _git_edge(
        git,
        prereg.commit,
        prereg.path,
        payloads["preregistration"],
        role="preregistration",
    )
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
        revisions.filing: REQUIRED_D1_FILING_ARTIFACTS,
    }
    for revision, roles in stages.items():
        for role in roles:
            _git_edge(
                git,
                revision,
                manifest.artifacts[role].path,
                payloads[role],
                role=role,
            )


def _verify_analysis(payloads: Mapping[str, bytes]) -> Mapping[str, object]:
    receipt = parse_analysis_receipt(payloads["analysis_receipt"])
    expected = {
        "capture_manifest_sha256": _sha(payloads["raw_capture_manifest"]),
        "preregistration_sha256": _sha(payloads["preregistration"]),
        "reader_sha256": _sha(payloads["analysis_reader"]),
        "isolation_attestation_sha256": _sha(payloads["analysis_isolation_attestation"]),
        "output_sha256": _sha(payloads["blinded_analysis"]),
    }
    for field, value in expected.items():
        if getattr(receipt, field) != value:
            _fail("analysis_binding", f"analysis_receipt.{field}", "digest differs")
    try:
        return decode_d1_machine_analysis_declaration(payloads["blinded_analysis"]).manifest
    except D1MachineAnalysisCodecError as error:
        _fail("analysis_schema", "blinded_analysis", str(error))


def _verify_aliases(payloads: Mapping[str, bytes]) -> Mapping[str, str]:
    commitment = parse_alias_commitment(payloads["alias_commitment"])
    reveal = parse_alias_reveal(payloads["alias_reveal"])
    try:
        packet = json.loads(payloads["review_packet"])
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        _fail("review_packet", "review_packet", f"must be UTF-8 JSON: {error}")
    if type(packet) is not dict or packet.get("schema_version") != ("d1-blind-review-packet-v1"):
        _fail("review_packet", "review_packet", "schema is unsupported")
    try:
        canonical_packet = json.dumps(
            packet,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError, RecursionError) as error:
        _fail("review_packet", "review_packet", f"must be finite canonical JSON: {error}")
    if canonical_packet != payloads["review_packet"]:
        _fail("review_packet", "review_packet", "bytes must use canonical JSON")
    if packet.get("alias_commitment") != _sha(payloads["alias_commitment"]):
        _fail(
            "review_packet_alias",
            "review_packet.alias_commitment",
            "reviewed packet differs from the filed alias commitment",
        )
    if commitment.reveal_sha256 != _sha(payloads["alias_reveal"]):
        _fail("alias_commitment", "alias_reveal", "does not open the pre-review commitment")
    if (
        frozenset(reveal.mapping) != frozenset(D1_OPAQUE_ARM_ALIAS_IDS)
        or len(set(reveal.mapping)) != len(D1_ARM_IDS)
        or set(reveal.mapping.values()) != set(D1_ARM_IDS)
    ):
        _fail(
            "alias_bijection",
            "alias_reveal.mapping",
            "must map exact opaque A/B/C/D keys bijectively to the four D1 arms",
        )
    return reveal.mapping


def _derived_clause_status(clause: Mapping[str, object]) -> str:
    conditions = cast(list[dict[str, object]], clause["conditions"])
    statuses = {condition["status"] for condition in conditions}
    if "FAIL" in statuses:
        return "FAIL"
    if statuses == {"PASS"}:
        return "PASS"
    return "INCONCLUSIVE"


def _verify_feasible_frontier(
    verdict: ValidatedD1Verdict,
    analysis: Mapping[str, object],
    alias_mapping: Mapping[str, str],
    payloads: Mapping[str, bytes],
) -> tuple[str, ...]:
    if analysis["alias_commitment"] != _sha(payloads["alias_commitment"]):
        _fail("analysis_frontier", "blinded_analysis.alias_commitment", "digest differs")
    precedence = cast(list[str], analysis["clause_precedence"])
    if tuple(precedence) != verdict.clause_precedence:
        _fail(
            "analysis_frontier",
            "blinded_analysis.clause_precedence",
            "differs from the registered machine verdict",
        )
    clauses = cast(dict[str, dict[str, dict[str, object]]], analysis["clauses_by_alias"])
    clause_arms = {alias_mapping[alias] for alias in clauses}
    if clause_arms != set(D1_STABILIZATION_ARM_IDS):
        _fail(
            "analysis_frontier",
            "blinded_analysis.clauses_by_alias",
            "de-aliased clause candidates must equal the canonical stabilization arms",
        )
    derived_by_alias: dict[str, dict[str, str]] = {}
    for alias, by_clause in clauses.items():
        arm_id = alias_mapping[alias]
        derived_by_alias[alias] = {}
        for clause_id in precedence:
            clause = by_clause[clause_id]
            derived = _derived_clause_status(clause)
            if clause["status"] != derived:
                _fail(
                    "analysis_frontier",
                    f"blinded_analysis.clauses_by_alias.{alias}.{clause_id}.status",
                    "must derive from condition statuses",
                )
            if verdict.clauses_by_arm[arm_id][clause_id].status != derived:
                _fail(
                    "analysis_frontier",
                    f"machine_verdict.clauses_by_arm.{arm_id}.{clause_id}.status",
                    "differs from the blinded analysis",
                )
            derived_by_alias[alias][clause_id] = derived
    recomputed = {
        alias
        for alias, by_clause in derived_by_alias.items()
        if all(by_clause[clause_id] == "PASS" for clause_id in precedence)
    }
    declared = cast(list[str], analysis["feasible_aliases"])
    if len(set(declared)) != len(declared) or set(declared) != recomputed:
        _fail("analysis_frontier", "blinded_analysis.feasible_aliases", "differs from clauses")
    real = {alias_mapping[alias] for alias in declared}
    if not real <= set(D1_STABILIZATION_ARM_IDS):
        _fail("analysis_frontier", "alias_reveal.mapping", "frontier is not stabilization-only")
    frontier = tuple(arm for arm in D1_STABILIZATION_ARM_IDS if arm in real)
    if verdict.feasible_arms != frontier:
        _fail("analysis_frontier", "machine_verdict.feasible_arms", "differs from analysis")
    expected_successor = (
        frontier if verdict.successor.authorization is SuccessorAuthorization.AUTHORIZED else ()
    )
    if verdict.successor.eligible_arms != expected_successor:
        _fail("analysis_frontier", "machine_verdict.successor.eligible_arms", "differs")
    return frontier


def _verify_review(
    verdict: ValidatedD1Verdict, payloads: Mapping[str, bytes]
) -> tuple[str, str, int]:
    receipt = parse_review_receipt(payloads["reviewer_execution_receipt"])
    expected = {
        "packet_sha256": _sha(payloads["review_packet"]),
        "prompt_sha256": _sha(payloads["reviewer_prompt"]),
        "report_sha256": _sha(payloads["reviewer_report"]),
    }
    for field, value in expected.items():
        if getattr(receipt, field) != value:
            _fail("review_binding", f"review_receipt.{field}", "digest differs")
    try:
        report = parse_d1_major_findings(
            payloads["reviewer_report"],
            reviewer_prompt=payloads["reviewer_prompt"],
        )
    except ValueError as error:
        _fail("review_report", "reviewer_report", str(error))
    adjudication = verdict.review.adjudication
    identities = {
        "execution_receipt_sha256": _sha(payloads["reviewer_execution_receipt"]),
        "report_sha256": report.report_sha256,
        "review_packet_sha256": receipt.packet_sha256,
        "review_request_sha256": receipt.request_digest,
    }
    if any(getattr(adjudication, field) != value for field, value in identities.items()):
        _fail("review_adjudication", "machine_verdict.review", "execution identity differs")
    if report.major_finding_ids != adjudication.major_findings:
        _fail("review_adjudication", "reviewer_report", "MAJOR ids differ")
    return receipt.model_id, receipt.provider, receipt.round


def _verify_narrative(payloads: Mapping[str, bytes]) -> None:
    marker = f"<!-- d1-machine-verdict-sha256:{_sha(payloads['machine_verdict'])} -->".encode()
    if payloads["narrative"].count(marker) != 1:
        _fail("narrative_binding", "narrative", "must bind the machine verdict digest once")


def verify_filing_edges(
    verdict: ValidatedD1Verdict,
    manifest: D1FilingManifest,
    revisions: D1FilingRevisions,
    payloads: Mapping[str, bytes],
    git: D1FilingGitPort,
    /,
) -> D1VerifiedFilingEdges:
    """Verify semantic bindings and historical ordering without reading metrics."""

    if type(verdict) is not ValidatedD1Verdict or type(manifest) is not D1FilingManifest:
        raise TypeError("D1 filing edges require validated verdict and manifest values")
    if type(revisions) is not D1FilingRevisions:
        raise TypeError("D1 filing edges require typed revisions")
    _verify_payloads(manifest, payloads)
    _verify_git(verdict, manifest, revisions, payloads, git)
    analysis = _verify_analysis(payloads)
    alias_mapping = _verify_aliases(payloads)
    feasible_arms = _verify_feasible_frontier(verdict, analysis, alias_mapping, payloads)
    model_id, provider, round_number = _verify_review(verdict, payloads)
    _verify_narrative(payloads)
    digest = canonical_digest(
        {
            "alias_mapping": dict(alias_mapping),
            "feasible_arms": list(feasible_arms),
            "artifact_sha256": {role: _sha(payload) for role, payload in payloads.items()},
            "revisions": list(revisions.ordered),
            "review": {"model_id": model_id, "provider": provider, "round": round_number},
            "schema_version": "d1-verified-filing-edges-v2",
        }
    )
    return D1VerifiedFilingEdges(
        alias_mapping=MappingProxyType(dict(alias_mapping)),
        feasible_arms=feasible_arms,
        review_model_id=model_id,
        review_provider=provider,
        review_round=round_number,
        digest=digest,
    )
