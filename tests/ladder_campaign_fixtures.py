"""Local test-only authority from D1's real filing verifier and artifact services."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.d1_artifact_snapshot import publish_d1_atomic_bytes
from entropic_substrate.d1_filing_application import (
    D1LocalFilingArtifactStore,
    prepare_d1_filing_bundle,
)
from entropic_substrate.evidence.d1_filing import validate_d1_filing_manifest
from entropic_substrate.evidence.d1_filing_graph import verify_d1_filing
from entropic_substrate.evidence.d1_filing_receipts import D1FilingRevisions
from entropic_substrate.evidence.ladder_campaign import (
    FiledLadderGate,
    RunAuthorization,
    artifact_mapping,
    authorize_ladder_run,
    gate_from_d1_filing,
)
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    LifecycleEvent,
    LifecycleEventKind,
    LifecycleRecord,
    transition,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest
from entropic_substrate.experiments.ladder import RungSpec, Treatment

from .test_d1_filing_application import _filing_git, _paths
from .test_d1_real_arm_verdict_assembly import _lineage
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION


def d1_gate(root: Path, monkeypatch: pytest.MonkeyPatch) -> FiledLadderGate:
    """Reuse D1's validator; never construct a successor from a summary boolean."""
    root.mkdir(parents=True, exist_ok=True)
    lineage = _lineage(root / "finite-lineage", monkeypatch)
    prepared = prepare_d1_filing_bundle(
        verified=lineage.verified,
        analysis_receipt=lineage.analysis_receipt,
        execution=lineage.execution,
        reveal=lineage.reveal,
        adjudication=lineage.adjudication,
        verdict="SUPPORTED",
        narrative_body=b"# Local fixture D1 verdict\n",
        preregistration_bytes=_RAW_PREREGISTRATION,
        paths=_paths(),
    )
    (root / "filing").mkdir()
    store = D1LocalFilingArtifactStore(root)
    store.publish(prepared, manifest_path="filing/manifest.json")
    revisions = D1FilingRevisions("b" * 40, "c" * 40, "d" * 40, "e" * 40, "1" * 40, "2" * 40)
    detached = validate_d1_filing_manifest(
        prepared.verdict.manifest,
        json.loads(prepared.manifest_bytes),
        decision_contract=prepared.verdict.verified.expected.decision_contract,
        artifact_root=root,
    )
    filing = verify_d1_filing(
        detached,
        prepared.verdict.verified.aggregate,
        revisions,
        store,
        _filing_git(prepared, revisions),
    )
    evidence = filing.to_filed_evidence()
    record = LifecycleRecord.start("d1")
    phases = (
        (LifecycleEventKind.IMPLEMENT, ActorRole.IMPLEMENTATION),
        (LifecycleEventKind.VERIFY_CONTRACT, ActorRole.IMPLEMENTATION),
        (LifecycleEventKind.RUN_EXPLORATORY_PILOT, ActorRole.OPERATOR),
        (LifecycleEventKind.PREREGISTER, ActorRole.OPERATOR),
        (LifecycleEventKind.START_CONFIRMATORY, ActorRole.OPERATOR),
        (LifecycleEventKind.CAPTURE_DVC, ActorRole.AUTOMATION),
        (LifecycleEventKind.ANALYZE, ActorRole.ANALYSIS),
        (LifecycleEventKind.BLIND_REVIEW, ActorRole.REVIEW),
        (LifecycleEventKind.FILE, ActorRole.OPERATOR),
        (LifecycleEventKind.AUTHORIZE_SUCCESSOR, ActorRole.OPERATOR),
    )
    for index, (kind, role) in enumerate(phases):
        record = transition(
            record,
            LifecycleEvent(
                f"fixture-{index}",
                kind,
                ActorIdentity(role.value, role.value, role),
                (evidence.narrative, evidence.machine, evidence.review_packet),
                sealed_packet=evidence.review_packet
                if kind is LifecycleEventKind.BLIND_REVIEW
                else None,
                filed_verdict=evidence if kind is LifecycleEventKind.FILE else None,
            ),
        )
    return gate_from_d1_filing(root, filing, record)


def authorization(
    root: Path,
    spec: RungSpec,
    source_sha256: str,
    split: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_class: str = "pilot",
) -> RunAuthorization:
    gate = d1_gate(root, monkeypatch)
    operator = ActorIdentity("operator", "op-fixture", ActorRole.OPERATOR)
    arms = ("language", "supervised", "geometry")
    arm = spec.treatment.value
    seeds = (spec.seed, spec.seed + 1)
    prereg = None
    if run_class == "confirmatory":
        prereg = publish_d1_atomic_bytes(
            root, "preregistration.md", b"# Test-only registration\n", label="test"
        )
    statistical = {
        "schema": "d-ladder-statistical-design-v1",
        "run_class": run_class,
        "registered_design_digest": None if prereg is None else prereg.content_hash.digest,
        "family_digest": canonical_digest(spec.base.construction_manifest),
        "construction_digests": {
            name: replace(spec, treatment=Treatment(name)).scientific_digest for name in arms
        },
        "seeds": seeds,
        "contrasts": [
            {
                "contrast_id": "test-only-three-arm",
                "terms": {arms[0]: -0.5, arms[1]: -0.5, arms[2]: 1.0},
                "metric": "language_nats",
                "units": "nats/byte",
                "eval_steps": [spec.base.execution_protocol.optimization.max_steps],
                "source_bytes": spec.base.execution_protocol.optimization.max_steps
                * spec.base.execution_protocol.batching.batch_size
                * spec.base.execution_protocol.batching.sequence_length,
                "kind": "superiority",
                "expected_effect": 0.01,
                "lower_bound": 0.0,
                "upper_bound": None,
            }
        ],
    }
    stats = publish_d1_atomic_bytes(
        root, "statistics.json", json.dumps(statistical).encode(), label="test"
    )
    design = {
        "schema": "d-ladder-campaign-design-v1",
        "rung": spec.rung.value,
        "d4_phase": None,
        "statistics": artifact_mapping(stats),
        "preregistration": None if prereg is None else artifact_mapping(prereg),
        "registration_git": None,
        "implementation": {
            "actor_id": "builder",
            "session_id": "builder",
            "role": "implementation",
        },
        "operator": {
            "actor_id": operator.actor_id,
            "session_id": operator.session_id,
            "role": operator.role.value,
        },
        "execution_protocol_digest": spec.base.execution_protocol.protocol_digest,
        "source_sha256": source_sha256,
        "source_split": split,
        "reader_identity": "d-ladder-scientific-reader-v1",
        "device": "cpu",
        "cells": [
            {
                "logical_run_id": f"{name}-{seed}",
                "arm": name,
                "seed": seed,
                "pairing_id": f"pair-{seed}",
                "spec_digest": replace(spec, treatment=Treatment(name), seed=seed).digest,
                "predecessor_strategy": gate.evidence.eligible_strategies[0],
            }
            for name in arms
            for seed in seeds
        ],
    }
    source = publish_d1_atomic_bytes(
        root, "campaign.json", json.dumps(design).encode(), label="test"
    )
    return authorize_ladder_run(
        root, source, f"{arm}-{spec.seed}", operator=operator, predecessor=gate
    )
