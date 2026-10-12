"""Rung-neutral capture arithmetic over authoritative, hash-bound run records.

External execution, DVC transport and human review remain adapters. This module
does not manufacture those observations from a successful local smoke test.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import torch

from entropic_substrate.core.corpus.batching import DATA_ORDER_DIGEST_BATCHES
from entropic_substrate.core.corpus.ladder_source_order import SourceOrderAudit
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    artifact_mapping,
    object_fields,
    read_bound_artifact,
    read_json_artifact,
    relative_d1_artifact_path,
)
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.ladder.analyze.ladder_registration import (
    StatisticalEvaluation,
    StatisticalRun,
    paired_contrasts,
)
from entropic_substrate.ladder.analyze.seed_sizing import dual_confidence_intervals
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import verify_d1_artifact
from entropic_substrate.ladder.capture.custody.d1_capture_bundle import D1DvcCaptureHandoff
from entropic_substrate.ladder.capture.records.d1_telemetry_values import (
    D1_SCALAR_METRICS,
    D1RunStatus,
)
from entropic_substrate.ladder.capture.resource.resource_work_codec import (
    resource_snapshot_from_mapping,
)
from entropic_substrate.ladder.gate.ladder_campaign import (
    actor_from_mapping,
    read_campaign_design,
)
from entropic_substrate.ladder.register.d1_provenance import require_lowercase_digest


def parse_scientific_data_order(value: object, step: int) -> tuple[str, int] | None:
    """Require the complete bounded training-prefix attestation at each eval."""
    if type(step) is not int or step < 0:
        raise ValueError("data-order step must be a nonnegative integer")
    if step == 0:
        if value is not None:
            raise ValueError("step-zero data order must be null before training")
        return None
    if type(value) is not list or len(value) != 2:
        raise ValueError("positive scientific eval requires data-order digest and count")
    digest, count = value
    if (
        type(digest) is not str
        or type(count) is not int
        or count != min(step, DATA_ORDER_DIGEST_BATCHES)
    ):
        raise ValueError("data-order attestation must cover the complete bounded prefix")
    require_lowercase_digest(digest, label="scientific data order")
    return digest, count


class RungRunReader(Protocol):
    """Normalize actual evidence without conferring authority on arbitrary values."""

    def read_run(self, root: Path, artifact: ArtifactIdentity) -> StatisticalRun: ...

    @property
    def rungs(self) -> frozenset[str]: ...

    def source_artifacts(
        self, root: Path, artifact: ArtifactIdentity
    ) -> tuple[ArtifactIdentity, ...]: ...


class D1RunReader:
    """Explicit D1 adapter retains the complete historical public reader."""

    rungs = frozenset({"d1"})

    def source_artifacts(
        self, root: Path, artifact: ArtifactIdentity
    ) -> tuple[ArtifactIdentity, ...]:
        read_bound_artifact(root, artifact)
        return (artifact,)

    def read_run(self, root: Path, artifact: ArtifactIdentity) -> StatisticalRun:
        from entropic_substrate.ladder.capture.reader.telemetry import read_d1_run_evidence_bytes

        run = read_d1_run_evidence_bytes(
            read_bound_artifact(root, artifact), journal_path=Path(artifact.path)
        )
        if run.status is not D1RunStatus.COMPLETE:
            raise ValueError("scientific arithmetic requires complete admitted evidence")
        return StatisticalRun(
            cast(str, run.plan_manifest["arm"]),
            cast(int, run.plan_manifest["seed"]),
            run.identity.run_class.value,
            run.identity.family_digest,
            run.identity.construction_digest,
            tuple(
                StatisticalEvaluation(
                    item.step,
                    item.resource_work.committed_quality_source_bytes,
                    tuple(
                        (name, item.quality.scalar_metric(name))
                        for name in sorted(D1_SCALAR_METRICS)
                    ),
                )
                for item in run.evals
            ),
        )


def read_ladder_run(
    root: Path, artifact: ArtifactIdentity, reader: RungRunReader, *, expected_class: str
) -> StatisticalRun:
    read_bound_artifact(root, artifact)
    run = reader.read_run(root, artifact)
    if expected_class not in {"pilot", "confirmatory"} or run.run_class != expected_class:
        raise ValueError("persisted scientific run class differs from registered class")
    return run


class ScientificRunReader:
    """Read the shared executor's scientific capture, checkpoint and eval lineage."""

    rungs = frozenset({"d2", "d3", "d4", "d5"})

    def source_artifacts(
        self, root: Path, artifact: ArtifactIdentity
    ) -> tuple[ArtifactIdentity, ...]:
        index = object_fields(
            read_json_artifact(root, artifact),
            "schema authorization_digest manifest records checkpoints completed_step",
        )
        if index["schema"] != "d-ladder-scientific-capture-v1":
            raise ValueError("unknown scientific capture schema")
        if type(index["records"]) is not list or type(index["checkpoints"]) is not list:
            raise ValueError("capture dependencies must be explicit lists")
        prefix = PurePosixPath(artifact.path).parent
        result = [artifact]
        for value in (index["manifest"], *index["records"], *index["checkpoints"]):
            item = artifact_from_mapping(value)
            if PurePosixPath(item.path).is_absolute():
                raise ValueError("capture dependency must be relative")
            child = ArtifactIdentity(str(prefix / item.path), item.content_hash)
            read_bound_artifact(root, child)
            result.append(child)
        if len({item.path for item in result}) != len(result):
            raise ValueError("duplicate scientific capture dependency")
        return tuple(result)

    def read_run(self, root: Path, artifact: ArtifactIdentity) -> StatisticalRun:
        index = object_fields(
            read_json_artifact(root, artifact),
            "schema authorization_digest manifest records checkpoints completed_step",
        )
        if index["schema"] != "d-ladder-scientific-capture-v1":
            raise ValueError("scientific reader refuses smoke or unknown capture schema")
        prefix = PurePosixPath(artifact.path).parent

        def child(value: object) -> ArtifactIdentity:
            item = artifact_from_mapping(value)
            if PurePosixPath(item.path).is_absolute():
                raise ValueError("capture dependency must be relative to its index")
            return ArtifactIdentity(str(prefix / item.path), item.content_hash)

        manifest = read_json_artifact(root, child(index["manifest"]))
        if manifest.get("schema") != "d-ladder-scientific-run-v1" or manifest.get(
            "run_class"
        ) not in {"pilot", "confirmatory"}:
            raise ValueError("scientific reader requires persisted pilot/confirmatory provenance")
        auth = cast(dict[str, object], manifest["authorization"])
        if (
            auth.get("schema") != "d-ladder-run-authorization-v1"
            or canonical_digest(auth) != index["authorization_digest"]
            or auth["run_class"] != manifest["run_class"]
        ):
            raise ValueError("scientific manifest authorization differs from capture")
        campaign, design = read_campaign_design(
            root, artifact_from_mapping(auth["campaign_artifact"])
        )
        if auth.get("d4_phase") != campaign["d4_phase"] or auth.get("rung") != campaign["rung"]:
            raise ValueError("scientific rung/phase differs from campaign")
        expected = [
            cell
            for cell in cast(list[dict[str, object]], campaign["cells"])
            if cell["logical_run_id"] == auth["logical_run_id"]
        ]
        if len(expected) != 1 or any(
            auth[key] != expected[0][key] for key in ("arm", "seed", "pairing_id", "spec_digest")
        ):
            raise ValueError("scientific run is outside the prepared paired roster")
        if (
            auth["run_class"] != design.run_class
            or auth["family_digest"] != design.family_digest
            or auth["construction_digest"]
            != dict(design.construction_digests)[cast(str, auth["arm"])]
            or auth["statistical_design_digest"] != design.source_sha256
        ):
            raise ValueError("scientific run class/family/construction/statistics identity differs")
        execution = cast(dict[str, object], manifest["execution"])
        if (
            canonical_digest(execution) != auth["execution_protocol_digest"]
            or auth["execution_protocol_digest"] != campaign["execution_protocol_digest"]
        ):
            raise ValueError("execution protocol identity differs from registration")
        for key in ("source_sha256", "source_split", "device"):
            if manifest[key] != auth[key] or auth[key] != campaign[key]:
                raise ValueError("source or device identity differs from registration")
        if (
            canonical_digest(cast(dict[str, object], manifest["construction"]))
            != auth["spec_digest"]
            or manifest["reader"] != auth["reader_identity"]
        ):
            raise ValueError("construction or reader identity differs")
        optimization = cast(dict[str, object], execution["optimization"])
        batching = cast(dict[str, int], execution["batching"])
        evaluation = cast(dict[str, int], execution["evaluation"])
        final = optimization["max_steps"]
        if type(final) is not int or final != index["completed_step"]:
            raise ValueError("scientific capture has not reached the registered endpoint")
        expected_steps = sorted(
            {0, final, *range(evaluation["eval_interval"], final + 1, evaluation["eval_interval"])}
        )
        if type(index["records"]) is not list or type(index["checkpoints"]) is not list:
            raise ValueError("scientific capture requires explicit records and checkpoints")
        checkpoints = {
            entry.content_hash.digest: entry
            for value in index["checkpoints"]
            for entry in (child(value),)
        }
        if len(checkpoints) != len(index["checkpoints"]):
            raise ValueError("scientific capture duplicates a checkpoint")
        evals: list[StatisticalEvaluation] = []
        data_order: list[tuple[int, str, int]] = []
        identity = canonical_digest(manifest)
        used_checkpoints: set[str] = set()
        for value in index["records"]:
            event = read_json_artifact(root, child(value))
            if (
                event.get("schema") != "d-ladder-scientific-step-v1"
                or event.get("identity") != identity
                or event.get("authorization_digest") != index["authorization_digest"]
                or event.get("run_class") != design.run_class
            ):
                raise ValueError("scientific eval identity/class differs from its manifest")
            step = event["step"]
            if type(step) is not int:
                raise ValueError("scientific eval step must be integer")
            if "data_order" not in event:
                raise ValueError("scientific eval lacks data-order attestation")
            order = parse_scientific_data_order(event["data_order"], step)
            if order is not None:
                data_order.append((step, *order))
            exposure = step * batching["batch_size"] * batching["sequence_length"]
            if event["source_bytes"] != exposure or event["train_online_positions"] != exposure:
                raise ValueError("source-byte exposure differs from registered batching")
            resource = resource_snapshot_from_mapping(event["resource_snapshot"])
            if (
                resource.training_step != step
                or resource.committed_quality_source_bytes != exposure
                or resource.logical_run_id != auth["logical_run_id"]
            ):
                raise ValueError("resource cursor/exposure differs from eval")
            digest = event["checkpoint_sha256"]
            if digest is not None:
                if type(digest) is not str or digest not in checkpoints:
                    raise ValueError("scientific eval lacks its captured checkpoint")
                checkpoint = torch.load(
                    io.BytesIO(read_bound_artifact(root, checkpoints[digest])),
                    map_location="cpu",
                    weights_only=True,
                )
                measurements = {
                    key: item for key, item in event.items() if key != "checkpoint_sha256"
                }
                if (
                    checkpoint.get("schema") != "d-ladder-scientific-checkpoint-v1"
                    or checkpoint.get("identity") != identity
                    or checkpoint.get("step") != step
                    or checkpoint.get("measurement_digest") != canonical_digest(measurements)
                ):
                    raise ValueError("checkpoint does not bind the measured eval bytes")
                audit = SourceOrderAudit.restore(
                    checkpoint.get("source_order"), step=step, batch_size=batching["batch_size"]
                )
                if audit.attestation != order:
                    raise ValueError(
                        "checkpoint observed source offsets differ from eval attestation"
                    )
                used_checkpoints.add(digest)
            elif step in (0, final) or step % evaluation["save_interval"] == 0:
                raise ValueError("scientific eval lacks a scheduled checkpoint")
            metrics = cast(dict[str, object], event["metrics"])
            canonical_digest(metrics)  # Reject nonfinite nested mechanism evidence too.
            scalar = tuple(
                (name, float(cast(int | float, item)))
                for name, item in metrics.items()
                if type(item) in (int, float)
            )
            evals.append(StatisticalEvaluation(step, exposure, scalar))
        if [item.step for item in evals] != expected_steps or used_checkpoints != set(checkpoints):
            raise ValueError("scientific capture eval/checkpoint population is incomplete or extra")
        return StatisticalRun(
            cast(str, auth["arm"]),
            cast(int, auth["seed"]),
            design.run_class,
            design.family_digest,
            cast(str, auth["construction_digest"]),
            tuple(evals),
            data_order=tuple(data_order),
        )


@dataclass(frozen=True, slots=True)
class LadderAnalysisInputs:
    """Re-readable sources; a normalized run list alone cannot authorize analysis."""

    design: ArtifactIdentity
    sources: tuple[ArtifactIdentity, ...]
    reader: RungRunReader
    dvc_handoff: D1DvcCaptureHandoff | None = None


def analyze_ladder_sources(
    root: Path, inputs: LadderAnalysisInputs, *, analyst: ActorIdentity
) -> dict[str, object]:
    """Compute the registered paired contrasts afresh from the captured file bytes."""
    row, design = read_campaign_design(root, inputs.design)
    if row["rung"] not in inputs.reader.rungs:
        raise ValueError("reader adapter does not implement the registered rung")
    handoff = inputs.dvc_handoff
    if type(handoff) is not D1DvcCaptureHandoff or not handoff.analysis_ready:
        raise ValueError("scientific analysis requires the bound DVC capture handoff")
    captured = {(item.normalized_path, item.sha256) for item in handoff.artifacts}
    required = {
        (
            relative_d1_artifact_path(root, root / item.path, label="scientific source"),
            item.content_hash.digest,
        )
        for source in inputs.sources
        for item in inputs.reader.source_artifacts(root, source)
    }
    if not required <= captured:
        raise ValueError("DVC handoff omits scientific run sources or checkpoint/eval dependencies")
    for artifact in handoff.artifacts:
        verify_d1_artifact(root, artifact)
        pointer = ArtifactIdentity(
            artifact.dvc.pointer_path, ContentHash("sha256", cast(str, artifact.dvc.pointer_sha256))
        )
        read_bound_artifact(root, pointer)
    builder = actor_from_mapping(row["implementation"])
    if (
        analyst.role is not ActorRole.ANALYSIS
        or analyst.actor_id == builder.actor_id
        or analyst.session_id == builder.session_id
    ):
        raise ValueError("analysis requires a fresh non-implementation actor and session")
    if design.run_class != "confirmatory":
        raise ValueError("confirmatory analysis cannot admit pilot or smoke inputs")
    if len({item.path for item in inputs.sources}) != len(inputs.sources):
        raise ValueError("duplicate captured source")
    runs = tuple(
        read_ladder_run(root, source, inputs.reader, expected_class="confirmatory")
        for source in inputs.sources
    )
    contrasts = paired_contrasts(runs, design)
    seal = canonical_digest(
        {
            "design": artifact_mapping(inputs.design),
            "sources": [artifact_mapping(source) for source in inputs.sources],
        }
    )
    reports = {
        item.contrast_id: json.loads(
            json.dumps(
                asdict(
                    dual_confidence_intervals(
                        item.pairs, family_size=len(contrasts), sealed_data_id=seal
                    )
                ),
                allow_nan=False,
            )
        )
        for item in contrasts
    }
    return {
        "schema": "d-ladder-analysis-v1",
        "capture_digest": seal,
        "rung": row["rung"],
        "design": artifact_mapping(inputs.design),
        "sources": [artifact_mapping(source) for source in inputs.sources],
        "statistics_sha256": design.source_sha256,
        "analyst": {
            "actor_id": analyst.actor_id,
            "session_id": analyst.session_id,
            "role": analyst.role.value,
        },
        "population": [
            {"arm": run.arm, "seed": run.seed, "run_class": run.run_class} for run in runs
        ],
        "contrasts": reports,
    }
