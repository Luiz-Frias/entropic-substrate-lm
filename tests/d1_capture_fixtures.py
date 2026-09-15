"""Content-addressed four-arm D1 capture fixtures."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import cast

from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_bundle import D1CaptureInput
from entropic_substrate.evidence.d1_capture_roster import D1ExpectedRun, D1LaunchRoster
from entropic_substrate.evidence.d1_journal_values import D1JournalEvent, JournalEventKind
from entropic_substrate.evidence.resource_provenance import execution_segment_id
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass, canonical_digest
from entropic_substrate.experiments.d1_provenance import D1_TELEMETRY_SCHEMA_VERSION

from .d1_telemetry_fixtures import start_provenance, valid_events, write_events

FAMILY_DIGEST = "a" * 64
PAIRING_ID = "pair-7"
DVC_POINTER = "results/d1-captured.dvc"


def _replace_payload_identity(payload: dict[str, object], logical_run_id: str) -> None:
    for key in ("resource_ledger", "resource_work"):
        value = payload.get(key)
        if type(value) is dict:
            value["logical_run_id"] = logical_run_id
            value["execution_segment_id"] = execution_segment_id(logical_run_id, 0)


def _events_for(
    arm: D1Arm,
    *,
    logical_run_id: str,
    construction_digest: str,
    execution_digest: str,
    aggregation_digest: str,
    checkpoint_sha256: str,
    seed: int = 7,
    pairing_id: str = PAIRING_ID,
    family_digest: str = FAMILY_DIGEST,
    code_commit: str = "abcdef0",
    repository_digest: str = "d" * 64,
    image_digest: str = "sha256:" + "5" * 64,
    dataset_digest: str = "f" * 64,
    split_digest: str = "0" * 64,
    sampler_construction_digest: str = "9" * 64,
    preregistration_path: str = "docs/d1-preregistration.md",
    preregistration_commit: str = "abcdef0",
    required_evidence_classes: tuple[str, ...] = ("measured",),
) -> tuple[list[D1JournalEvent], str]:
    events = valid_events(arm=arm)
    start_payload = cast(dict[str, object], events[0].manifest["payload"])
    plan = cast(dict[str, object], start_payload["plan"])
    plan.update(
        {
            "arm": arm.value,
            "code": {"commit": code_commit, "repository_digest": repository_digest},
            "construction_digest": construction_digest,
            "container": {"image_digest": image_digest},
            "data": {
                "dataset_digest": dataset_digest,
                "dataset_name": "enwik8",
                "sampler_construction_digest": sampler_construction_digest,
                "split_digest": split_digest,
            },
            "evidence": {
                "checkpoint_schema_version": 1,
                "dvc_capture_required": True,
                "required_classes": list(required_evidence_classes),
                "schema_version": "d1-evidence-v1",
                "telemetry_schema_version": D1_TELEMETRY_SCHEMA_VERSION,
            },
            "family_digest": family_digest,
            "logical_run_id": logical_run_id,
            "pairing_id": pairing_id,
            "preregistration": {
                "commit": preregistration_commit,
                "family_digest": family_digest,
                "path": preregistration_path,
            },
            "profile_name": f"d1-{arm.value}",
            "run_class": RunClass.CONFIRMATORY.value,
            "seed": seed,
        }
    )
    start_payload["plan_digest"] = canonical_digest(plan)
    start_payload["run_start_provenance"] = run_start_to_mapping(
        replace(
            start_provenance(),
            logical_run_id=logical_run_id,
            execution_segment_id=execution_segment_id(logical_run_id, 0),
            image_digest=image_digest,
        )
    )
    rewritten: list[D1JournalEvent] = []
    for event in events:
        payload = (
            start_payload
            if event.event_kind is JournalEventKind.RUN_START
            else cast(dict[str, object], event.manifest["payload"])
        )
        _replace_payload_identity(payload, logical_run_id)
        if (
            event.event_kind is JournalEventKind.CHECKPOINT
            and event.event_seq == events[-2].event_seq
        ):
            payload["checkpoint_sha256"] = checkpoint_sha256
        rewritten.append(
            replace(
                event,
                logical_run_id=logical_run_id,
                execution_digest=execution_digest,
                aggregation_digest=aggregation_digest,
                construction_digest=construction_digest,
                family_digest=family_digest,
                payload=payload,
            )
        )
    return rewritten, canonical_digest(plan)


def capture_family(
    tmp_path: Path, *, seeds: tuple[int, ...] = (7,)
) -> tuple[D1LaunchRoster, tuple[D1CaptureInput, ...], Path]:
    root = tmp_path / "capture"
    root.mkdir()
    expected: list[D1ExpectedRun] = []
    inputs: list[D1CaptureInput] = []
    arms = tuple(D1Arm)
    for seed, arm in product(seeds, arms):
        index = arms.index(arm) + 1
        logical_run_id = f"d1-{arm.value}-s{seed}"
        pairing_id = f"pair-{seed}"
        construction = str(index) * 64
        execution = str(index + 4) * 64
        aggregation = str(index + 5) * 64
        checkpoint_path = root / f"{logical_run_id}.pt"
        checkpoint_path.write_bytes(f"checkpoint:{arm.value}:{seed}\n".encode())
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        events, plan_digest = _events_for(
            arm,
            logical_run_id=logical_run_id,
            construction_digest=construction,
            execution_digest=execution,
            aggregation_digest=aggregation,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            pairing_id=pairing_id,
        )
        journal_path = root / f"{logical_run_id}.jsonl"
        write_events(journal_path, events)
        billing_path = root / f"{logical_run_id}.billing.json"
        billing_path.write_text('{"evidence":"provider_billed"}\n', encoding="utf-8")
        expected.append(
            D1ExpectedRun(
                logical_run_id=logical_run_id,
                arm=arm,
                seed=seed,
                pairing_id=pairing_id,
                family_digest=FAMILY_DIGEST,
                construction_digest=construction,
                execution_digest=execution,
                aggregation_digest=aggregation,
                plan_digest=plan_digest,
                code_commit="abcdef0",
                repository_digest="d" * 64,
                image_digest="sha256:" + "5" * 64,
                dataset_digest="f" * 64,
                split_digest="0" * 64,
                sampler_construction_digest="9" * 64,
            )
        )
        inputs.append(
            D1CaptureInput(
                logical_run_id=logical_run_id,
                journal=identify_d1_artifact(
                    root, journal_path, role=D1ArtifactRole.JOURNAL, dvc_pointer_path=DVC_POINTER
                ),
                checkpoint=identify_d1_artifact(
                    root,
                    checkpoint_path,
                    role=D1ArtifactRole.CHECKPOINT,
                    dvc_pointer_path=DVC_POINTER,
                ),
                billing_sidecars=(
                    identify_d1_artifact(
                        root,
                        billing_path,
                        role=D1ArtifactRole.BILLING_SIDECAR,
                        dvc_pointer_path=DVC_POINTER,
                    ),
                ),
                quarantined_siblings=(),
            )
        )
    roster = D1LaunchRoster(
        preregistration_path="docs/d1-preregistration.md",
        preregistration_commit="abcdef0",
        family_digest=FAMILY_DIGEST,
        expected_confirmatory=tuple(expected),
        attempts=(),
    )
    return roster, tuple(inputs), root
