"""Only a completely verified D1 evidence graph may reach the pure gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import pytest

from d1_verdict_v2_fixtures import decision_contract, machine_verdict
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.evidence.d1_capture_aggregate_validation import (
    build_d1_capture_aggregate,
)
from entropic_substrate.evidence.d1_capture_artifacts import D1ArtifactIdentity
from entropic_substrate.evidence.d1_capture_bundle import D1CaptureInput
from entropic_substrate.evidence.d1_capture_validation import build_d1_capture_bundle
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    D1DetachedFiling,
    canonical_machine_verdict_bytes,
    validate_d1_filing_manifest,
)
from entropic_substrate.evidence.d1_filing_graph import (
    recompute_capture_integrity,
    verify_d1_filing,
)
from entropic_substrate.evidence.d1_filing_receipts import (
    D1EvidenceGapError,
    D1FilingGraphError,
    D1FilingRevisions,
)
from entropic_substrate.evidence.d1_gate import decide_filed_verdict
from entropic_substrate.evidence.d1_review_adjudication import (
    D1_NO_MAJOR_PROMPT_DIRECTIVE,
)
from entropic_substrate.evidence.d1_types import (
    D1FamilyIdentity,
    D1VerdictValidationError,
    FiledGateAuthorization,
)

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .d1_capture_fixtures import capture_family

pytestmark = pytest.mark.unit


def _json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass
class RootPort:
    root: Path

    def read_bytes(self, path: str) -> bytes | None:
        target = self.root / path
        return target.read_bytes() if target.is_file() else None


@dataclass
class FakeGit:
    files: dict[tuple[str, str], bytes]
    ancestors: set[tuple[str, str]]

    def read_file(self, revision: str, path: str) -> bytes | None:
        return self.files.get((revision, path))

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return (ancestor, descendant) in self.ancestors


def _bind_artifact(artifact: D1ArtifactIdentity, *, bound: bool = True) -> D1ArtifactIdentity:
    return (
        replace(
            artifact,
            dvc=artifact.dvc.bind(pointer_sha256="b" * 64, object_id=artifact.md5),
        )
        if bound
        else artifact
    )


def _bind_capture(item: D1CaptureInput, *, bound: bool) -> D1CaptureInput:

    return replace(
        item,
        journal=_bind_artifact(item.journal, bound=bound),
        checkpoint=_bind_artifact(item.checkpoint, bound=bound),
        billing_sidecars=tuple(
            _bind_artifact(value, bound=bound) for value in item.billing_sidecars
        ),
    )


def _capture_contract(aggregate: D1HeterogeneousCaptureAggregate) -> D1DecisionContract:
    identity = D1FamilyIdentity(
        family_digest=aggregate.roster.family_digest,
        arm_construction_digests={
            item.arm.value: item.construction_digest
            for item in aggregate.roster.expected_confirmatory
        },
    )
    base = decision_contract()
    seeds = tuple(sorted({item.seed for item in aggregate.roster.expected_confirmatory}))
    effective_n = len(aggregate.roster.pairing_ids) - len(aggregate.pair_exclusions)
    contrast_rules = (
        {
            name: replace(rule, registered_n=1, df_rule="NOT_APPLICABLE")
            for name, rule in base.contrast_rules.items()
        }
        if effective_n == 1
        else {
            name: replace(
                rule,
                n_rule="NONEXCLUDED_SEED_COUNT",
                registered_n=None,
                df_rule="N_MINUS_ONE",
            )
            for name, rule in base.contrast_rules.items()
        }
    )
    return replace(
        base,
        family_identity=identity,
        registered_constants={"seeds": list(seeds), "primary": {"alpha": 0.025}},
        contrast_rules=contrast_rules,
        exclusion_rules=(replace(base.exclusion_rules[0], measurement="nats_per_byte"),),
    )


def _case(
    tmp_path: Path, mutation: str | None = None, *, mixed: bool = False
) -> tuple[
    D1DetachedFiling,
    D1HeterogeneousCaptureAggregate,
    D1FilingRevisions,
    RootPort,
    FakeGit,
]:
    if mixed:
        roster, complete, defects, attempts, capture_root = heterogeneous_family(
            tmp_path, seeds=(7, 8, 9)
        )
        complete = tuple(_bind_capture(item, bound=True) for item in complete)
        defects = tuple(
            replace(
                item,
                journal=_bind_artifact(item.journal, bound=mutation != "defect_dvc_gap"),
                supporting_artifacts=tuple(
                    _bind_artifact(artifact) for artifact in item.supporting_artifacts
                ),
            )
            for item in defects
        )
        attempts = tuple(
            replace(
                item,
                artifacts=tuple(_bind_artifact(artifact) for artifact in item.artifacts),
            )
            for item in attempts
        )
        aggregate = build_d1_capture_aggregate(
            capture_root,
            roster=roster,
            complete_inputs=complete,
            terminal_defect_inputs=defects,
            attempt_evidence=attempts,
        )
    else:
        roster, inputs, capture_root = capture_family(tmp_path)
        bound = mutation != "dvc_gap"
        inputs = tuple(_bind_capture(item, bound=bound) for item in inputs)
        aggregate = build_d1_capture_aggregate(
            capture_root,
            roster=roster,
            complete_inputs=inputs,
            terminal_defect_inputs=(),
        )
    contract = _capture_contract(aggregate)
    seeds = sorted({item.seed for item in aggregate.roster.expected_confirmatory})
    if mutation == "exclusion_rule":
        contract = replace(
            contract,
            registered_constants={"seeds": seeds, "primary": {"alpha": 0.025}},
            exclusion_rules=(
                replace(contract.exclusion_rules[0], measurement="unregistered_metric"),
            ),
        )
    integrity = recompute_capture_integrity(aggregate, contract)
    machine = machine_verdict()
    machine["registered_constants"] = {"seeds": seeds, "primary": {"alpha": 0.025}}
    machine["preregistration"] = {
        "commit": roster.preregistration_commit,
        "family_digest": roster.family_digest,
        "path": roster.preregistration_path,
    }
    machine["arms"] = [
        {
            "arm_id": arm_id,
            "construction_digest": contract.family_identity.arm_construction_digests[arm_id],
        }
        for arm_id in contract.arm_ids
    ]
    for contrast in machine["contrasts"]:  # type: ignore[union-attr]
        contrast["n"] = integrity.effective_sample_size
        contrast["df"] = (
            None if integrity.effective_sample_size == 1 else integrity.effective_sample_size - 1
        )
        if integrity.effective_sample_size == 1:
            contrast["sample_sd"] = None
            contrast["interval_lower"] = None
            contrast["interval_upper"] = None
        if mutation == "sample_size":
            contrast["n"] = integrity.effective_sample_size + 1
            contrast["df"] = integrity.effective_sample_size
    input_rows = [
        {"md5": item.md5, "path": item.normalized_path, "sha256": item.sha256}
        for item in aggregate.dvc_handoff.artifacts
    ]
    if mutation == "defect_input_gap":
        defect_path = aggregate.terminal_defects[0].capture_input.journal.normalized_path
        input_rows = [item for item in input_rows if item["path"] != defect_path]
    elif mutation == "attempt_input_gap":
        attempt_path = aggregate.attempts[0].artifacts[0].normalized_path
        input_rows = [item for item in input_rows if item["path"] != attempt_path]
    machine["inputs"] = input_rows
    if mutation == "input_gap":
        del machine["inputs"][0]["sha256"]  # type: ignore[index]
    exclusion_rows: list[dict[str, object]] = [
        {
            "subject": claim.pairing_id,
            "reason": reason_id,
            "evidence_hash": {"algorithm": "sha256", "digest": evidence_digest},
        }
        for claim, reason_id, evidence_digest in integrity.applied_exclusions or ()
    ]
    if mutation == "exclusion_missing":
        exclusion_rows.clear()
    elif mutation == "exclusion_subject":
        exclusion_rows[0]["subject"] = "wrong-pair"
    elif mutation == "exclusion_reason":
        exclusion_rows[0]["reason"] = "wrong-reason"
    elif mutation == "exclusion_hash":
        exclusion_rows[0]["evidence_hash"] = {
            "algorithm": "sha256",
            "digest": "0" * 64,
        }
    machine["exclusions"] = exclusion_rows
    machine["integrity"]["evidence_hash"] = {  # type: ignore[index]
        "algorithm": "sha256",
        "digest": "0" * 64 if mutation == "integrity" else integrity.evidence_digest,
    }
    prereg = b"registered design bytes\n"
    capture = _json(aggregate.manifest)
    analysis = b'{"decision":"aliased"}'
    reader = b"def read_capture(): ...\n"
    isolation = b'{"session":"fresh","network":"disabled"}'
    reveal = _json(
        {
            "mapping": {"A": "none", "B": "sg", "C": "ac", "D": "both"},
            "nonce": "sealed",
            "schema_version": "d1-alias-reveal-v1",
        }
    )
    commitment = _json({"reveal_sha256": _sha(reveal), "schema_version": "d1-alias-commitment-v1"})
    payloads = {role: role.encode() for role in REQUIRED_D1_FILING_ARTIFACTS}
    payloads.update(
        {
            "preregistration": prereg,
            "raw_capture_manifest": b'{"wrong":true}' if mutation == "capture" else capture,
            "blinded_analysis": analysis,
            "analysis_reader": reader,
            "analysis_isolation_attestation": isolation,
            "alias_reveal": reveal,
            "alias_commitment": commitment,
            "review_packet": _json(
                {
                    "alias_commitment": _sha(commitment),
                    "schema_version": "d1-blind-review-packet-v1",
                }
            ),
            "reviewer_prompt": f"Prompt\n{D1_NO_MAJOR_PROMPT_DIRECTIVE}\n".encode(),
            "reviewer_report": b"Review outcome: NO_MAJOR_FINDINGS\n",
        }
    )
    payloads["analysis_receipt"] = _json(
        {
            "capture_manifest_sha256": _sha(payloads["raw_capture_manifest"]),
            "isolation_attestation_sha256": _sha(isolation),
            "output_sha256": _sha(analysis),
            "preregistration_sha256": _sha(prereg),
            "reader_sha256": _sha(reader),
            "schema_version": "d1-analysis-receipt-v1",
        }
    )
    payloads["reviewer_execution_receipt"] = _json(
        {
            "model_id": "review-model",
            "packet_sha256": _sha(payloads["review_packet"]),
            "prompt_sha256": _sha(payloads["reviewer_prompt"]),
            "provider": "review-provider",
            "report_sha256": _sha(payloads["reviewer_report"]),
            "request_digest": "a" * 64,
            "response_sha256": "b" * 64,
            "retry_lineage": [],
            "round": 1,
            "schema_version": "d1-review-execution-receipt-v2",
        }
    )
    adjudication = machine["review"]["adjudication"]  # type: ignore[index]
    adjudication.update(  # type: ignore[union-attr]
        {
            "execution_receipt_sha256": _sha(payloads["reviewer_execution_receipt"]),
            "report_sha256": _sha(payloads["reviewer_report"]),
            "review_packet_sha256": _sha(payloads["review_packet"]),
            "review_request_sha256": "a" * 64,
        }
    )
    adjudication_mutations = {
        "adjudication_receipt": "execution_receipt_sha256",
        "adjudication_report": "report_sha256",
        "adjudication_packet": "review_packet_sha256",
        "adjudication_request": "review_request_sha256",
    }
    if mutation in adjudication_mutations:
        adjudication[adjudication_mutations[mutation]] = "0" * 64  # type: ignore[index]
    machine_bytes = canonical_machine_verdict_bytes(machine)
    payloads["machine_verdict"] = machine_bytes
    payloads["narrative"] = (
        f"# D1 verdict\n<!-- d1-machine-verdict-sha256:{_sha(machine_bytes)} -->\n"
    ).encode()
    filing_root = tmp_path / "filing-root"
    artifacts = {}
    for role, payload in payloads.items():
        path = f"filing/{role}.bin"
        target = filing_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        artifacts[role] = {
            "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            "path": path,
            "sha256": _sha(payload),
            "size": len(payload),
        }
    detached = validate_d1_filing_manifest(
        machine,
        {"artifacts": artifacts, "rung": "d1", "schema_version": "d1-filing-manifest/v2"},
        decision_contract=contract,
        artifact_root=filing_root,
    )
    revisions = D1FilingRevisions("b" * 40, "c" * 40, "d" * 40, "e" * 40, "1" * 40, "2" * 40)
    chain = (roster.preregistration_commit, *revisions.ordered)
    git = FakeGit({}, set(pairwise(chain)))
    git.files[(roster.preregistration_commit, roster.preregistration_path)] = prereg
    stage_roles = {
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
    for revision, roles in stage_roles.items():
        for role in roles:
            git.files[(revision, detached.manifest.artifacts[role].path)] = payloads[role]
    return detached, aggregate, revisions, RootPort(filing_root), git


def test_only_semantically_verified_graph_authorizes_pure_gate(tmp_path: Path) -> None:
    detached, aggregate, revisions, artifacts, git = _case(tmp_path)
    filing = verify_d1_filing(detached, aggregate, revisions, artifacts, git)

    outcome = decide_filed_verdict(filing)
    assert isinstance(outcome, FiledGateAuthorization)
    assert outcome.eligible_arms == ("sg", "ac", "both")
    assert filing.to_filed_evidence().eligible_arms == outcome.eligible_arms
    with pytest.raises(TypeError, match="semantic evidence graph"):
        decide_filed_verdict(detached)  # type: ignore[arg-type]


def test_mixed_filing_binds_registered_exclusion_and_effective_sample_size(
    tmp_path: Path,
) -> None:
    detached, aggregate, revisions, artifacts, git = _case(tmp_path, mixed=True)

    filing = verify_d1_filing(detached, aggregate, revisions, artifacts, git)

    assert filing.integrity.effective_sample_size == 2
    assert filing.integrity.applied_exclusions is not None
    claim, reason_id, evidence_digest = filing.integrity.applied_exclusions[0]
    assert detached.verdict.exclusions[0][0] == claim.pairing_id
    assert reason_id == "nonfinite-final-loss"
    assert detached.verdict.exclusions[0][2].digest == evidence_digest


@pytest.mark.parametrize(
    "mutation",
    [
        "adjudication_receipt",
        "adjudication_report",
        "adjudication_packet",
        "adjudication_request",
    ],
)
def test_filing_rejects_adjudication_execution_identity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    with pytest.raises(D1FilingGraphError, match="review_adjudication"):
        verify_d1_filing(*_case(tmp_path, mutation))


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("exclusion_missing", D1FilingGraphError, "exclusions"),
        ("exclusion_subject", D1FilingGraphError, "exclusions"),
        ("exclusion_reason", D1VerdictValidationError, "exclusion reason"),
        ("exclusion_hash", D1FilingGraphError, "exclusions"),
        ("exclusion_rule", D1FilingGraphError, "exclusions"),
    ],
)
def test_mixed_filing_rejects_exclusion_semantic_drift(
    tmp_path: Path,
    mutation: str,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        verify_d1_filing(*_case(tmp_path, mutation, mixed=True))


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("defect_input_gap", D1FilingGraphError, "capture manifest"),
        ("attempt_input_gap", D1FilingGraphError, "capture manifest"),
        ("defect_dvc_gap", D1EvidenceGapError, "DVC"),
        ("sample_size", D1FilingGraphError, "sample_size"),
    ],
)
def test_mixed_filing_rejects_custody_and_sample_size_drift(
    tmp_path: Path,
    mutation: str,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        verify_d1_filing(*_case(tmp_path, mutation, mixed=True))


def test_filing_verification_refuses_legacy_complete_bundle(tmp_path: Path) -> None:
    filing_root = tmp_path / "filing"
    filing_root.mkdir()
    detached, _, revisions, artifacts, git = _case(filing_root)
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    roster, inputs, root = capture_family(legacy_root)
    bundle = build_d1_capture_bundle(root, roster=roster, capture_inputs=inputs)
    with pytest.raises(TypeError, match="heterogeneous capture"):
        verify_d1_filing(  # type: ignore[arg-type]
            detached, bundle, revisions, artifacts, git
        )


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("capture", D1FilingGraphError, "capture manifest"),
        ("dvc_gap", D1EvidenceGapError, "DVC"),
        ("input_gap", D1EvidenceGapError, "sha256"),
        ("integrity", D1FilingGraphError, "integrity"),
    ],
)
def test_capture_inputs_and_recomputed_integrity_are_gate_teeth(
    tmp_path: Path, mutation: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        verify_d1_filing(*_case(tmp_path, mutation))
