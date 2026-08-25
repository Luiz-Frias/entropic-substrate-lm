"""Only a completely verified D1 evidence graph may reach the pure gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import pytest

from d1_verdict_v2_fixtures import decision_contract, machine_verdict
from entropic_substrate.evidence.d1_capture_bundle import D1CaptureInput
from entropic_substrate.evidence.d1_capture_validation import build_d1_capture_bundle
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
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
from entropic_substrate.evidence.d1_types import D1FamilyIdentity, FiledGateAuthorization

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


def _bind_capture(item: D1CaptureInput, *, bound: bool) -> D1CaptureInput:
    if not bound:
        return item

    def bind(artifact):
        return replace(
            artifact,
            dvc=artifact.dvc.bind(pointer_sha256="b" * 64, object_id=artifact.md5),
        )

    return replace(
        item,
        journal=bind(item.journal),
        checkpoint=bind(item.checkpoint),
        billing_sidecars=tuple(bind(value) for value in item.billing_sidecars),
    )


def _one_seed_contract(bundle) -> D1DecisionContract:
    identity = D1FamilyIdentity(
        family_digest=bundle.roster.family_digest,
        arm_construction_digests={
            item.expected.arm.value: item.expected.construction_digest
            for item in bundle.confirmatory
        },
    )
    base = decision_contract()
    contrast_rules = {
        name: replace(rule, registered_n=1, df_rule="NOT_APPLICABLE")
        for name, rule in base.contrast_rules.items()
    }
    return replace(
        base,
        family_identity=identity,
        registered_constants={"seeds": [7], "primary": {"alpha": 0.025}},
        contrast_rules=contrast_rules,
    )


def _case(tmp_path: Path, mutation: str | None = None):
    roster, inputs, capture_root = capture_family(tmp_path)
    bound = mutation != "dvc_gap"
    inputs = tuple(_bind_capture(item, bound=bound) for item in inputs)
    bundle = build_d1_capture_bundle(capture_root, roster=roster, capture_inputs=inputs)
    contract = _one_seed_contract(bundle)
    machine = machine_verdict()
    machine["registered_constants"] = {"seeds": [7], "primary": {"alpha": 0.025}}
    machine["preregistration"] = {
        "commit": roster.preregistration_commit,
        "family_digest": roster.family_digest,
        "path": roster.preregistration_path,
    }
    machine["arms"] = [
        {
            "arm_id": item.expected.arm.value,
            "construction_digest": item.expected.construction_digest,
        }
        for item in bundle.confirmatory
    ]
    for contrast in machine["contrasts"]:  # type: ignore[union-attr]
        contrast["n"] = 1
        contrast["df"] = None
        contrast["sample_sd"] = None
    machine["inputs"] = [
        {"md5": item.md5, "path": item.normalized_path, "sha256": item.sha256}
        for item in bundle.dvc_handoff.artifacts
    ]
    if mutation == "input_gap":
        del machine["inputs"][0]["sha256"]  # type: ignore[index]
    integrity = recompute_capture_integrity(bundle)
    machine["integrity"]["evidence_hash"] = {  # type: ignore[index]
        "algorithm": "sha256",
        "digest": "0" * 64 if mutation == "integrity" else integrity.evidence_digest,
    }
    prereg = b"registered design bytes\n"
    capture = _json(bundle.manifest)
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
    return detached, bundle, revisions, RootPort(filing_root), git


def test_only_semantically_verified_graph_authorizes_pure_gate(tmp_path: Path) -> None:
    detached, bundle, revisions, artifacts, git = _case(tmp_path)
    filing = verify_d1_filing(detached, bundle, revisions, artifacts, git)

    assert isinstance(decide_filed_verdict(filing), FiledGateAuthorization)
    with pytest.raises(TypeError, match="semantic evidence graph"):
        decide_filed_verdict(detached)  # type: ignore[arg-type]


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
