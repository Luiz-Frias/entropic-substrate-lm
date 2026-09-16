"""Portable, closed artifact graph for a model trace; not a scientific verdict."""

from pathlib import Path

from entropic_substrate.d1_artifact_snapshot import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
    snapshot_d1_artifact,
)
from entropic_substrate.evidence.ladder_campaign import artifact_from_mapping
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.sentence_prediction.generation import read_generation
from entropic_substrate.sentence_prediction.journal import artifact_reference, read_json, write_json
from entropic_substrate.sentence_prediction.profile import HProfile
from entropic_substrate.sentence_prediction.recovery_state import recovery_record
from entropic_substrate.sentence_prediction.runner import read_training_progress

_SCHEMAS = frozenset(
    (
        "h-run-profile-v1",
        "h-probe-training-progress-v1",
        "h-checkpoint-seal-v1",
        "h-journal-shard-v1",
        "h-predictive-evaluation-v1",
        "h-generation-receipt-v1",
        "h-generation-receipt-v2",
        "h-probe-training-failure-v1",
        "h-generation-failure-v1",
    )
)


def _references(value: object) -> list[ArtifactIdentity]:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            return [artifact_from_mapping(value)]
        return [item for child in value.values() for item in _references(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _references(child)]
    return []


def _limits(*values: int) -> None:
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("capture budgets must be positive integers")


def _collect(
    root: Path,
    roots: tuple[ArtifactIdentity, ...],
    profile: HProfile,
    *,
    max_bytes: int,
    max_artifacts: int,
) -> tuple[dict[str, ArtifactIdentity], int]:
    pending = list(roots)
    collected: dict[str, ArtifactIdentity] = {}
    total = 0
    while pending:
        artifact = pending.pop()
        ref = artifact_reference(root, artifact)
        name = ref["path"]
        if len(Path(name).parts) != 1 or Path(name).suffix not in (".json", ".pt"):
            raise ValueError("capture requires flat producer artifact paths")
        normalized = artifact_from_mapping(ref)
        if name in collected:
            if normalized != collected[name]:
                raise ValueError("capture contains conflicting artifact identities")
            continue
        if len(collected) >= max_artifacts:
            raise ValueError("capture artifact budget exceeded")
        _, size, digest = snapshot_d1_artifact(root, name, label="model trace artifact")
        if digest != ref["sha256"] or size > max_bytes:
            raise ValueError("capture artifact hash or byte budget mismatch")
        total += size
        collected[name] = normalized
        if name.endswith(".json"):
            payload = read_json(root, normalized, max_bytes=max_bytes)
            if payload.get("schema") not in _SCHEMAS:
                raise ValueError("capture encountered an unknown producer artifact schema")
            if payload.get("schema") == "h-run-profile-v1":
                if HProfile.from_manifest(payload) != profile:
                    raise ValueError("captured profile differs from requested model trace")
            elif payload.get("run_identity") != profile.identity:
                raise ValueError("capture artifact belongs to another run")
            pending.extend(_references(payload))
    return collected, total


def capture_model_trace(
    root: Path,
    training: ArtifactIdentity,
    generation: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    destination: Path,
    *,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
    forensic_roots: tuple[ArtifactIdentity, ...] = (),
) -> ArtifactIdentity:
    """Copy verified bytes unchanged; publish the capture manifest last.

    Extra forensic roots must be supplied explicitly; no arbitrary directory
    sweep or deletion is performed. Partial destinations have no final capture.
    """
    _limits(max_bytes, max_source_bytes, max_total_bytes, max_artifacts)
    target = normalize_d1_artifact_root(destination)
    if any(target.iterdir()):
        raise ValueError("capture destination must be empty")
    train = read_training_progress(
        root,
        training,
        profile,
        source_root,
        max_source_bytes=max_source_bytes,
        max_artifact_bytes=max_bytes,
    )
    if train["status"] != "training_complete":
        raise ValueError("model trace capture requires complete training")
    generated = read_generation(
        root,
        generation,
        training,
        profile,
        source_root,
        max_source_bytes=max_source_bytes,
        max_artifact_bytes=max_bytes,
    )
    if generated["status"] != "generation_complete":
        raise ValueError("model trace capture requires a complete generation outcome")
    collected, total = _collect(
        root,
        (training, generation, *forensic_roots),
        profile,
        max_bytes=max_bytes,
        max_artifacts=max_artifacts,
    )
    _, source_bytes, source_digest = read_d1_artifact_bytes(
        source_root, "prepared-corpus.json", label="captured source", max_bytes=max_source_bytes
    )
    if source_digest != profile.source_digest:
        raise ValueError("capture source differs from the profile")
    total += len(source_bytes)
    if total > max_total_bytes or len(collected) + 1 > max_artifacts:
        raise ValueError("capture total byte or artifact budget exceeded")
    references = []
    for name, artifact in sorted(collected.items()):
        _, data, digest = read_d1_artifact_bytes(
            root, name, label="capture copy", max_bytes=max_bytes
        )
        if digest != artifact.content_hash.digest:
            raise ValueError("artifact changed during capture")
        copied = publish_d1_atomic_bytes(target, name, data, label="captured model artifact")
        references.append(artifact_reference(target, copied))
    (target / "source").mkdir()
    source = publish_d1_atomic_bytes(
        target, "source/prepared-corpus.json", source_bytes, label="captured source"
    )
    references.append(artifact_reference(target, source))
    return write_json(
        target,
        "capture.json",
        {
            "schema": "h-model-trace-capture-v1",
            "kind": "model_trace",
            "scientific_authority": "none",
            "run_identity": profile.identity,
            "run_class": profile.run_class,
            "profile": train["profile"],
            "training": artifact_reference(root, training),
            "generation": artifact_reference(root, generation),
            "source": artifact_reference(target, source),
            "forensic_roots": [artifact_reference(root, item) for item in forensic_roots],
            "artifacts": references,
            "file_count": len(references),
            "total_bytes": total,
        },
        max_bytes=max_bytes,
    )


def read_model_trace(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
) -> dict[str, object]:
    """Verify the full copied closure and replay the numerical readers."""
    _limits(max_bytes, max_source_bytes, max_total_bytes, max_artifacts)
    manifest = recovery_record(
        read_json(root, artifact, max_bytes=max_bytes),
        "schema kind scientific_authority run_identity run_class profile training generation source forensic_roots artifacts file_count total_bytes",
    )
    if (
        manifest["schema"] != "h-model-trace-capture-v1"
        or manifest["kind"] != "model_trace"
        or manifest["scientific_authority"] != "none"
    ):
        raise ValueError("unknown capture scope or scientific authority")
    profile = HProfile.from_manifest(
        read_json(root, artifact_from_mapping(manifest["profile"]), max_bytes=max_bytes)
    )
    if manifest["run_identity"] != profile.identity or manifest["run_class"] != profile.run_class:
        raise ValueError("capture profile identity mismatch")
    source = artifact_from_mapping(manifest["source"])
    if source.path != "source/prepared-corpus.json" or source.content_hash != ContentHash(
        "sha256", profile.source_digest
    ):
        raise ValueError("capture source binding is invalid")
    training, generation = (
        artifact_from_mapping(manifest["training"]),
        artifact_from_mapping(manifest["generation"]),
    )
    extra = manifest["forensic_roots"]
    if type(extra) is not list:
        raise ValueError("capture forensic roots must be explicit")
    collected, total = _collect(
        root,
        (training, generation, *(artifact_from_mapping(item) for item in extra)),
        profile,
        max_bytes=max_bytes,
        max_artifacts=max_artifacts,
    )
    _, size, digest = snapshot_d1_artifact(root, source.path, label="captured source")
    if digest != profile.source_digest or size > max_source_bytes:
        raise ValueError("captured source hash or size mismatch")
    collected[source.path] = source
    total += size
    artifacts = manifest["artifacts"]
    if type(artifacts) is not list or len(artifacts) > max_artifacts or total > max_total_bytes:
        raise ValueError("capture budget or artifact inventory is invalid")
    listed = [artifact_from_mapping(item) for item in artifacts]
    if (
        len({item.path for item in listed}) != len(listed)
        or {item.path: item for item in listed} != collected
    ):
        raise ValueError("capture inventory differs from its referenced closure")
    if (
        type(manifest["file_count"]) is not int
        or manifest["file_count"] != len(listed)
        or type(manifest["total_bytes"]) is not int
        or manifest["total_bytes"] != total
    ):
        raise ValueError("capture measured totals disagree with artifacts")
    expected_paths = set(collected) | {artifact_reference(root, artifact)["path"]}
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("capture contains an unlisted or unsafe symlink")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise ValueError("capture contains unlisted files or missing artifacts")
    train = read_training_progress(
        root,
        training,
        profile,
        root / "source",
        max_source_bytes=max_source_bytes,
        max_artifact_bytes=max_bytes,
    )
    gen = read_generation(
        root,
        generation,
        training,
        profile,
        root / "source",
        max_source_bytes=max_source_bytes,
        max_artifact_bytes=max_bytes,
    )
    if train["status"] != "training_complete" or gen["status"] != "generation_complete":
        raise ValueError("capture does not contain a completed model trace")
    for item in collected.values():
        if item.path.endswith(".json") and item != generation:
            payload = read_json(root, item, max_bytes=max_bytes)
            if payload.get("schema") == "h-generation-receipt-v2":
                read_generation(
                    root,
                    item,
                    training,
                    profile,
                    root / "source",
                    max_source_bytes=max_source_bytes,
                    max_artifact_bytes=max_bytes,
                )
    return manifest
