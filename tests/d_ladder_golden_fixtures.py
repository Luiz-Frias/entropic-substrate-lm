"""Test-only services around real tiny scientific trainer/capture bytes.

All scientific choices are fixed before execution. The deliberately broad
toy margin exercises the positive pipeline branch; it is no research claim.
Git, DVC/cloud and review are explicitly test doubles except the real-Git case.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from entropic_substrate.composition.wiring.d_ladder_cli import tiny_rung_spec
from entropic_substrate.historical.d_ladder.d_ladder_execution import run_ladder
from entropic_substrate.historical.d_ladder.d_ladder_request import ScientificRequest
from entropic_substrate.historical.d_ladder.d_ladder_resources import LadderResources
from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes
from entropic_substrate.kernel.evidence_class.resource_provenance import execution_segment_id
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
    LifecycleEvent,
    LifecycleEventKind,
    LifecycleRecord,
    transition,
)
from entropic_substrate.ladder.analyze.ladder_analysis import (
    LadderAnalysisInputs,
    ScientificRunReader,
    analyze_ladder_sources,
)
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.ladder.capture.custody.d1_capture_bundle import D1DvcCaptureHandoff
from entropic_substrate.ladder.file.ladder_filing import (
    file_ladder_verdict,
    prepare_ladder_blind_packet,
)
from entropic_substrate.ladder.gate.ladder_campaign import artifact_mapping, authorize_ladder_run
from entropic_substrate.ladder.run.provider.d1_git_provenance_adapter import (
    SubprocessD1GitProvenance,
)
from entropic_substrate.ladder.run.resource.d1_resource_activation import (
    UnavailableResourceActivationFactory,
)

from .d1_telemetry_fixtures import start_provenance
from .ladder_campaign_fixtures import d1_gate

BUILDER = ActorIdentity("builder", "builder", ActorRole.IMPLEMENTATION)
OPERATOR = ActorIdentity("operator", "operator", ActorRole.OPERATOR)
ANALYST = ActorIdentity("analyst", "fresh-analysis", ActorRole.ANALYSIS)
REVIEWER = ActorIdentity("test-reviewer", "external-review-fixture", ActorRole.REVIEW)
AUTOMATION = ActorIdentity("capture-service", "capture-session", ActorRole.AUTOMATION)
SEEDS = (7, 8)
TOY_MARGIN = 1_000_000.0


def predecessor_gate(root, monkeypatch):
    return d1_gate(root, monkeypatch)


def publish(root: Path, name: str, value: object) -> ArtifactIdentity:
    return publish_d1_atomic_bytes(
        root,
        name,
        json.dumps(value, sort_keys=True, allow_nan=False).encode(),
        label="golden fixture",
    )


def actor(value: ActorIdentity) -> dict[str, str]:
    return {"actor_id": value.actor_id, "session_id": value.session_id, "role": value.role.value}


def tiny_spec(experiment: str, seed: int):
    original = tiny_rung_spec(experiment, seed=seed)
    protocol = original.base.execution_protocol
    base = replace(
        original.base,
        execution_protocol=replace(
            protocol,
            optimization=replace(protocol.optimization, max_steps=2, warmup_steps=0),
            evaluation=replace(
                protocol.evaluation, num_batches=1, eval_interval=1, log_interval=1, save_interval=1
            ),
        ),
    )

    def adapt(spec):
        return replace(
            spec,
            base=base,
            predecessor=None if spec.predecessor is None else adapt(spec.predecessor),
        )

    return adapt(original)


@dataclass
class FixtureGit:
    blobs: dict[tuple[str, str], str]

    def commit_time(self, commit):
        return datetime(2026, 9, 4, tzinfo=UTC) if commit in {"a" * 40, "b" * 40} else None

    def is_ancestor(self, ancestor, descendant):
        return (ancestor, descendant) == ("a" * 40, "b" * 40)

    def blob_sha256(self, commit, path):
        return self.blobs.get((commit, path))


@dataclass
class PreparedCampaign:
    root: Path
    rung: str
    specs: dict
    predecessor: object
    design: ArtifactIdentity
    policy: ArtifactIdentity | None
    git: object


def build_campaign(
    root, rung, experiments, predecessor, *, run_class="confirmatory", real_git=False
):
    specs = {(arm, seed): tiny_spec(arm, seed) for arm in experiments for seed in SEEDS}
    anchor = next(iter(specs.values()))
    source = bytes(range(256)) * 8
    source_path = root / "golden-source.bin"
    if not source_path.exists():
        source_path.write_bytes(source)
    policy = None
    if run_class == "confirmatory":
        policy = publish(
            root,
            f"{rung}-preregistration.json",
            {
                "schema": "d-ladder-successor-policy-v1",
                "successor_rung": "complete" if rung == "d5" else f"d{int(rung[1:]) + 1}",
                "successor_phase": "fixed" if rung == "d3" else None,
                "strategies": {
                    "toy-eligible": {"arms": list(experiments), "requires": ["toy-control"]}
                },
            },
        )
    terms = {arm: -1 / (len(experiments) - 1) for arm in experiments[:-1]} | {experiments[-1]: 1.0}
    statistics = publish(
        root,
        f"{rung}-statistics.json",
        {
            "schema": "d-ladder-statistical-design-v1",
            "run_class": run_class,
            "registered_design_digest": None if policy is None else policy.content_hash.digest,
            "family_digest": canonical_digest(anchor.base.construction_manifest),
            "construction_digests": {
                arm: specs[arm, SEEDS[0]].scientific_digest for arm in experiments
            },
            "seeds": SEEDS,
            "contrasts": [
                {
                    "contrast_id": "toy-control",
                    "terms": terms,
                    "metric": "language_nats",
                    "units": "nats/byte",
                    "eval_steps": [2],
                    "source_bytes": 32,
                    "kind": "noninferiority",
                    "expected_effect": 0.0,
                    "lower_bound": -TOY_MARGIN,
                    "upper_bound": None,
                }
            ],
        },
    )
    proof = None
    git = FixtureGit({})
    if policy is not None:
        paths = (f"{rung}-preregistration.json", f"{rung}-statistics.json")
        before, after = "a" * 40, "b" * 40
        git = FixtureGit(
            {
                (before, paths[0]): policy.content_hash.digest,
                (before, paths[1]): statistics.content_hash.digest,
            }
        )
        if real_git:

            def command(*args):
                return (
                    subprocess.check_output(
                        [
                            "git",
                            "-c",
                            "user.name=Golden fixture",
                            "-c",
                            "user.email=fixture@example.invalid",
                            *args,
                        ],
                        cwd=root,
                        stderr=subprocess.PIPE,
                    )
                    .decode()
                    .strip()
                )

            command("init", "-q")
            command("add", *paths)
            command("commit", "-qm", "test: preregister golden fixture")
            before = command("rev-parse", "HEAD")
            command("commit", "--allow-empty", "-qm", "test: implementation anchor")
            after = command("rev-parse", "HEAD")
            git = SubprocessD1GitProvenance(repo_root=root)
        proof = {
            "preregistration_commit": before,
            "implementation_commit": after,
            "preregistration_path": paths[0],
            "statistics_path": paths[1],
        }
    design = publish(
        root,
        f"{rung}-campaign.json",
        {
            "schema": "d-ladder-campaign-design-v1",
            "rung": rung,
            "d4_phase": "fixed" if rung == "d4" else None,
            "statistics": artifact_mapping(statistics),
            "preregistration": None if policy is None else artifact_mapping(policy),
            "registration_git": proof,
            "implementation": actor(BUILDER),
            "operator": actor(OPERATOR),
            "execution_protocol_digest": anchor.base.execution_protocol.protocol_digest,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "source_split": {
                "train_start": 0,
                "train_end": 1024,
                "eval_start": 1024,
                "eval_end": 2048,
                "source_total_bytes": 2048,
            },
            "reader_identity": "d-ladder-scientific-reader-v1",
            "device": "cpu",
            "cells": [
                {
                    "logical_run_id": f"{arm}-{seed}",
                    "arm": arm,
                    "seed": seed,
                    "pairing_id": f"{rung}-pair-{seed}",
                    "spec_digest": spec.digest,
                    "predecessor_strategy": predecessor.evidence.eligible_strategies[0],
                }
                for (arm, seed), spec in specs.items()
            ],
        },
    )
    return PreparedCampaign(root, rung, specs, predecessor, design, policy, git)


@dataclass
class CapturedCampaign:
    prepared: PreparedCampaign
    inputs: LadderAnalysisInputs
    dvc_events: list[str]


def execute_campaign(prepared):
    root, rung = prepared.root, prepared.rung
    sources = []
    for (arm, seed), spec in prepared.specs.items():
        auth = authorize_ladder_run(
            root,
            prepared.design,
            f"{arm}-{seed}",
            operator=OPERATOR,
            predecessor=prepared.predecessor,
            registration_git=prepared.git,
        )
        request = ScientificRequest(spec, auth, root, "golden-source.bin")
        start = replace(
            start_provenance(),
            logical_run_id=auth.logical_run_id,
            execution_segment_id=execution_segment_id(auth.logical_run_id, 0),
            accelerator_model="CPU-test-fixture",
            provider="local-test-fixture",
        )
        result = run_ladder(
            request,
            root / f"{rung}-runs" / auth.logical_run_id,
            resources=LadderResources(start, UnavailableResourceActivationFactory()),
        )
        path = Path(result["capture"])
        sources.append(
            ArtifactIdentity(
                str(path.relative_to(root)),
                ContentHash("sha256", hashlib.sha256(path.read_bytes()).hexdigest()),
            )
        )
    tracked = root / f"{rung}-runs"
    paths = sorted(path for path in tracked.rglob("*") if path.is_file())
    # Explicit test-double DVC add -> push -> existence proof over actual bytes.
    events = ["add"]
    directory = json.dumps(
        [
            {
                "relpath": str(path.relative_to(tracked)),
                "md5": hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
            }
            for path in paths
        ],
        sort_keys=True,
    ).encode()
    object_id = hashlib.md5(directory, usedforsecurity=False).hexdigest() + ".dir"
    pointer = root / f"{rung}-runs.dvc"
    pointer.write_text(f"outs:\n- md5: {object_id}\n  path: {tracked.name}\n")
    cloud_objects = {object_id: directory}
    events.append("push")
    assert cloud_objects[object_id] == directory
    events.append("verify")
    pointer_sha = hashlib.sha256(pointer.read_bytes()).hexdigest()
    artifacts = []
    for path in paths:
        item = identify_d1_artifact(
            root,
            path,
            role=D1ArtifactRole.CHECKPOINT if path.suffix == ".pt" else D1ArtifactRole.JOURNAL,
            dvc_pointer_path=pointer.name,
        )
        artifacts.append(
            replace(item, dvc=item.dvc.bind(pointer_sha256=pointer_sha, object_id=object_id))
        )
    inputs = LadderAnalysisInputs(
        prepared.design,
        tuple(sources),
        ScientificRunReader(),
        D1DvcCaptureHandoff(tuple(artifacts)),
    )
    return CapturedCampaign(prepared, inputs, events)


def file_campaign(captured, *, open_major_findings=()):
    root, rung = captured.prepared.root, captured.prepared.rung
    report = analyze_ladder_sources(root, captured.inputs, analyst=ANALYST)
    analysis = publish(root, f"{rung}-analysis.json", report)
    packet_value = prepare_ladder_blind_packet(root, captured.inputs, analysis)
    packet = publish(root, f"{rung}-blind-packet.json", packet_value)
    review = publish(
        root,
        f"{rung}-review.json",
        {
            "schema": "d-ladder-review-v1",
            "actor": actor(REVIEWER),
            "packet": artifact_mapping(packet),
            "analysis_sha256": analysis.content_hash.digest,
            "agrees": True,
            "open_major_findings": list(open_major_findings),
        },
    )
    narrative = publish_d1_atomic_bytes(
        root,
        f"{rung}-verdict.md",
        f"# Test-only verdict\nAnalysis-SHA256: {analysis.content_hash.digest}\n".encode(),
        label="golden narrative",
    )
    record = LifecycleRecord.start(rung)
    steps = (
        (LifecycleEventKind.IMPLEMENT, BUILDER, (captured.prepared.design,)),
        (LifecycleEventKind.VERIFY_CONTRACT, BUILDER, (captured.prepared.design,)),
        (LifecycleEventKind.RUN_EXPLORATORY_PILOT, OPERATOR, (captured.prepared.design,)),
        (LifecycleEventKind.PREREGISTER, OPERATOR, (captured.prepared.design,)),
        (LifecycleEventKind.START_CONFIRMATORY, OPERATOR, (captured.prepared.design,)),
        (LifecycleEventKind.CAPTURE_DVC, AUTOMATION, captured.inputs.sources),
        (LifecycleEventKind.ANALYZE, ANALYST, (analysis,)),
        (LifecycleEventKind.BLIND_REVIEW, REVIEWER, (packet, review)),
    )
    for index, (kind, who, artifacts) in enumerate(steps):
        record = transition(
            record,
            LifecycleEvent(
                f"{rung}-{index}",
                kind,
                who,
                artifacts,
                sealed_packet=packet if kind is LifecycleEventKind.BLIND_REVIEW else None,
            ),
        )
    filed = file_ladder_verdict(
        root,
        captured.inputs,
        analysis=analysis,
        packet=packet,
        review=review,
        narrative=narrative,
        record=record,
        operator=OPERATOR,
        machine_path=f"{rung}-verdict.json",
    )
    if isinstance(captured.prepared.git, SubprocessD1GitProvenance):
        from entropic_substrate.ladder.gate.ladder_predecessor import load_ladder_predecessor

        bundle_path = root / f"{rung}-registration.bundle"
        subprocess.run(
            ["git", "bundle", "create", str(bundle_path), "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        bundle = ArtifactIdentity(
            bundle_path.name,
            ContentHash("sha256", hashlib.sha256(bundle_path.read_bytes()).hexdigest()),
        )
        descriptor = publish(
            root,
            f"{rung}-predecessor.json",
            {
                "schema": "d-ladder-predecessor-v1",
                "design": artifact_mapping(captured.inputs.design),
                "sources": [artifact_mapping(item) for item in captured.inputs.sources],
                "analysis": artifact_mapping(analysis),
                "packet": artifact_mapping(packet),
                "review": artifact_mapping(review),
                "narrative": artifact_mapping(narrative),
                "machine": artifact_mapping(filed.machine),
                "registration_git_bundle": artifact_mapping(bundle),
                "dvc_artifacts": [item.manifest for item in captured.inputs.dvc_handoff.artifacts],
                "lifecycle_prefix": [
                    {
                        "event_id": event.event_id,
                        "kind": event.kind.value,
                        "actor": actor(event.actor),
                        "artifacts": [artifact_mapping(item) for item in event.artifacts],
                        "sealed_packet": None
                        if event.sealed_packet is None
                        else artifact_mapping(event.sealed_packet),
                    }
                    for event in record.events
                ],
            },
        )
        rehydrated = load_ladder_predecessor(root, descriptor)
        assert rehydrated.evidence == filed.gate.evidence
        filed = replace(filed, gate=rehydrated)
        # Descriptor tampering is a separate artifact; the original lineage stays intact.
        corrupt = publish(root, f"{rung}-bad-predecessor.json", {"schema": "tampered"})
        try:
            load_ladder_predecessor(root, replace(descriptor, path=corrupt.path))
        except ValueError as error:
            assert "digest" in str(error)
        else:
            raise AssertionError("changed predecessor bytes unexpectedly retained authority")
    return filed, report, packet_value
