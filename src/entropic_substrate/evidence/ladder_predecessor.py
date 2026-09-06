"""Re-earn D2-D4 successor authority from the same filed scientific lineage."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from entropic_substrate.analysis.ladder_analysis import LadderAnalysisInputs, ScientificRunReader
from entropic_substrate.analysis.ladder_filing import file_ladder_verdict
from entropic_substrate.d1_predecessor_loader import verified_git_bundle
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1ArtifactRole,
    D1DvcBinding,
    D1DvcBindingState,
)
from entropic_substrate.evidence.d1_capture_bundle import D1DvcCaptureHandoff
from entropic_substrate.evidence.ladder_campaign import (
    FiledLadderGate,
    actor_from_mapping,
    artifact_from_mapping,
    object_fields,
    read_bound_artifact,
    read_campaign_design,
    read_json_artifact,
    verify_ladder_registration,
)
from entropic_substrate.evidence.lifecycle import (
    ArtifactIdentity,
    LifecycleEvent,
    LifecycleEventKind,
    LifecycleRecord,
    transition,
)


def _dvc_artifact(value: object) -> D1ArtifactIdentity:
    row = object_fields(value, "path role sha256 md5 size_bytes dvc")
    dvc = object_fields(row["dvc"], "state pointer_path pointer_sha256 object_id")
    if type(row["size_bytes"]) is not int or any(
        type(row[key]) is not str for key in ("path", "role", "sha256", "md5")
    ):
        raise ValueError("DVC artifact metadata must use exact scalar types")
    if any(type(value) is not str for value in dvc.values()):
        raise ValueError("predecessor DVC binding must be complete")
    return D1ArtifactIdentity(
        cast(str, row["path"]),
        D1ArtifactRole(row["role"]),
        row["size_bytes"],
        cast(str, row["md5"]),
        cast(str, row["sha256"]),
        D1DvcBinding(
            D1DvcBindingState(dvc["state"]),
            cast(str, dvc["pointer_path"]),
            cast(str, dvc["pointer_sha256"]),
            cast(str, dvc["object_id"]),
        ),
    )


def load_ladder_predecessor(root: Path, declaration: ArtifactIdentity) -> FiledLadderGate:
    """Replay legal roles/states and recompute filing; perform no publication.

    The declaration carries references and the pre-filing lifecycle events.
    No recorded label, authorization boolean, or cached report grants a gate.
    """
    row = object_fields(
        read_json_artifact(root, declaration),
        "schema design sources analysis packet review narrative machine dvc_artifacts lifecycle_prefix registration_git_bundle",
    )
    if row["schema"] != "d-ladder-predecessor-v1":
        raise ValueError("unknown ladder predecessor descriptor schema")
    design = artifact_from_mapping(row["design"])
    campaign, _ = read_campaign_design(root, design)
    if campaign["rung"] not in {"d2", "d3", "d4"}:
        raise ValueError("generic predecessor must be a filed D2-D4 rung")
    if any(
        type(row[key]) is not list or not row[key]
        for key in ("sources", "dvc_artifacts", "lifecycle_prefix")
    ):
        raise ValueError("predecessor requires sources, DVC artifacts and lifecycle prefix")
    inputs = LadderAnalysisInputs(
        design,
        tuple(artifact_from_mapping(value) for value in cast(list[object], row["sources"])),
        ScientificRunReader(),
        D1DvcCaptureHandoff(
            tuple(_dvc_artifact(value) for value in cast(list[object], row["dvc_artifacts"]))
        ),
    )
    record = LifecycleRecord.start(cast(str, campaign["rung"]))
    for value in cast(list[object], row["lifecycle_prefix"]):
        event = object_fields(value, "event_id kind actor artifacts sealed_packet")
        if (
            type(event["event_id"]) is not str
            or type(event["kind"]) is not str
            or type(event["artifacts"]) is not list
        ):
            raise ValueError("malformed predecessor lifecycle event")
        kind = LifecycleEventKind(event["kind"])
        if kind in {LifecycleEventKind.FILE, LifecycleEventKind.AUTHORIZE_SUCCESSOR}:
            raise ValueError("predecessor prefix cannot assert its own filing or gate")
        artifacts = tuple(artifact_from_mapping(item) for item in event["artifacts"])
        for artifact in artifacts:
            read_bound_artifact(root, artifact)
        packet = (
            None
            if event["sealed_packet"] is None
            else artifact_from_mapping(event["sealed_packet"])
        )
        record = transition(
            record,
            LifecycleEvent(
                event["event_id"],
                kind,
                actor_from_mapping(event["actor"]),
                artifacts,
                sealed_packet=packet,
            ),
        )
    with verified_git_bundle(root, artifact_from_mapping(row["registration_git_bundle"])) as git:
        verify_ladder_registration(root, design, git)
    machine = artifact_from_mapping(row["machine"])
    result = file_ladder_verdict(
        root,
        inputs,
        analysis=artifact_from_mapping(row["analysis"]),
        packet=artifact_from_mapping(row["packet"]),
        review=artifact_from_mapping(row["review"]),
        narrative=artifact_from_mapping(row["narrative"]),
        record=record,
        operator=actor_from_mapping(campaign["operator"]),
        machine_path=machine.path,
        existing_machine=machine,
    )
    if result.gate is None:
        raise ValueError("filed predecessor has no authorized successor")
    result.gate.verify_files(root)
    return result.gate
