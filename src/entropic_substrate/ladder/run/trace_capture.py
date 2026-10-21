"""Portable, closed artifact graph for a model trace; not a scientific verdict."""

from dataclasses import dataclass
from pathlib import Path
from typing import overload

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
    snapshot_d1_artifact,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.generation import read_generation
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.runner import read_training_progress
from entropic_substrate.ladder.run.seals import SealMemo
from entropic_substrate.ladder.run.span_diagnostics import read_span_diagnostics

_SCHEMAS = frozenset(
    (
        "h-run-profile-v2",
        "h-probe-training-progress-v1",
        "h-checkpoint-seal-v1",
        "h-journal-shard-v1",
        "h-journal-shard-v2",
        "h-predictive-evaluation-v1",
        "h-generation-receipt-v1",
        "h-generation-receipt-v2",
        "h-probe-training-failure-v1",
        "h-probe-training-failure-v2",
        "h-generation-failure-v1",
        "h-span-diagnostics-v1",
        "h-span-diagnostics-failure-v1",
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
            if payload.get("schema") == "h-run-profile-v2":
                if HProfile.from_manifest(payload) != profile:
                    raise ValueError("captured profile differs from requested model trace")
            elif payload.get("run_identity") != profile.identity:
                raise ValueError("capture artifact belongs to another run")
            pending.extend(_references(payload))
    return collected, total


# The two ``@overload`` stubs below (and on the three readers after it) exist for
# the TYPE CHECKER only, with no runtime effect: ``composition/capture_cli.py``
# calls these with ``**limits`` where ``limits: dict[str, int]``, and a checker
# matches such an unpacked dict against EVERY keyword parameter it does not
# otherwise bind, so a plain ``memo: SealMemo | None = None`` would read as
# "an int passed as ``memo``" at call sites that are frozen (never-edit). The
# first stub is the memo-less signature those callers see; the second is the
# memo-bearing one the worker uses.
@overload
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
    diagnostic_roots: tuple[ArtifactIdentity, ...] = (),
) -> ArtifactIdentity: ...


@overload
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
    diagnostic_roots: tuple[ArtifactIdentity, ...] = (),
    memo: SealMemo | None,
) -> ArtifactIdentity: ...


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
    diagnostic_roots: tuple[ArtifactIdentity, ...] = (),
    memo: SealMemo | None = None,
) -> ArtifactIdentity:
    """Copy verified bytes unchanged; publish the capture manifest last.

    Extra forensic roots must be supplied explicitly; no arbitrary directory
    sweep or deletion is performed. Partial destinations have no final capture.

    ``memo`` is the admission memo of the ``root`` (the worker's ``/run``
    directory) and only of it; the CAPTURE root that :func:`read_model_trace`
    reads receives a memo of its own, never this one.
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
        memo=memo,
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
        memo=memo,
    )
    if generated["status"] != "generation_complete":
        raise ValueError("model trace capture requires a complete generation outcome")
    if type(diagnostic_roots) is not tuple or len(set(diagnostic_roots)) != len(diagnostic_roots):
        raise ValueError("diagnostic roots must be an explicit unique tuple")
    for diagnostic in diagnostic_roots:
        read_span_diagnostics(
            root,
            diagnostic,
            training,
            profile,
            source_root,
            max_source_bytes=max_source_bytes,
            max_artifact_bytes=max_bytes,
            memo=memo,
        )
    collected, total = _collect(
        root,
        (training, generation, *forensic_roots, *diagnostic_roots),
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
            "schema": "h-model-trace-capture-v2",
            "kind": "model_trace",
            "scientific_authority": "none",
            "run_identity": profile.identity,
            "run_class": profile.run_class,
            "profile": train["profile"],
            "training": artifact_reference(root, training),
            "generation": artifact_reference(root, generation),
            "source": artifact_reference(target, source),
            "forensic_roots": [artifact_reference(root, item) for item in forensic_roots],
            "diagnostic_roots": [artifact_reference(root, item) for item in diagnostic_roots],
            "artifacts": references,
            "file_count": len(references),
            "total_bytes": total,
        },
        max_bytes=max_bytes,
    )


@dataclass(frozen=True)
class _TraceInventory:
    manifest: dict[str, object]
    profile: HProfile
    training: ArtifactIdentity
    generation: ArtifactIdentity
    diagnostics: tuple[ArtifactIdentity, ...]
    artifacts: dict[str, ArtifactIdentity]


def _trace_inventory(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
) -> _TraceInventory:
    """Verify only the immutable byte graph; do not deserialize model state."""
    _limits(max_bytes, max_source_bytes, max_total_bytes, max_artifacts)
    raw = read_json(root, artifact, max_bytes=max_bytes)
    version = raw.get("schema")
    if version not in ("h-model-trace-capture-v1", "h-model-trace-capture-v2"):
        raise ValueError("unknown model capture reader version")
    manifest = recovery_record(
        raw,
        "schema kind scientific_authority run_identity run_class profile training generation source forensic_roots artifacts file_count total_bytes"
        + (" diagnostic_roots" if version == "h-model-trace-capture-v2" else ""),
    )
    if manifest["kind"] != "model_trace" or manifest["scientific_authority"] != "none":
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
    diagnostics = manifest["diagnostic_roots"] if version == "h-model-trace-capture-v2" else []
    if type(diagnostics) is not list:
        raise ValueError("capture diagnostic roots must be explicit")
    diagnostic_refs = tuple(artifact_from_mapping(item) for item in diagnostics)
    if len(set(diagnostic_refs)) != len(diagnostic_refs):
        raise ValueError("capture diagnostic roots must be unique")
    collected, total = _collect(
        root,
        (training, generation, *(artifact_from_mapping(item) for item in extra), *diagnostic_refs),
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
    return _TraceInventory(manifest, profile, training, generation, diagnostic_refs, collected)


def inspect_model_custody(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
) -> dict[str, object]:
    """Portable byte custody only; native admission remains a separate required check."""
    inventory = _trace_inventory(
        root,
        artifact,
        max_bytes=max_bytes,
        max_source_bytes=max_source_bytes,
        max_total_bytes=max_total_bytes,
        max_artifacts=max_artifacts,
    )
    return {
        "schema": "h-model-custody-inspection-v1",
        "status": "byte_custody_verified",
        "native_validation": "not_performed",
        "scientific_authority": "none",
        "capture": artifact_reference(root, artifact),
        "run_identity": inventory.profile.identity,
        "declared_training_device": inventory.profile.device,
        "counts_scope": "referenced_artifacts_excluding_capture_manifest",
        "file_count": inventory.manifest["file_count"],
        "total_bytes": inventory.manifest["total_bytes"],
    }


@overload
def read_model_trace(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
) -> dict[str, object]: ...


@overload
def read_model_trace(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
    memo: SealMemo | None,
) -> dict[str, object]: ...


def read_model_trace(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    max_bytes: int,
    max_source_bytes: int,
    max_total_bytes: int,
    max_artifacts: int,
    memo: SealMemo | None = None,
) -> dict[str, object]:
    """Verify byte custody and run native model/source/RNG/numerical admission.

    ``memo`` belongs to THIS capture root alone (it holds byte-identical copies
    of the ``/run`` seals, and the memo is keyed by digest): pass a memo created
    for the capture root, never the ``/run`` memo, or the seal readers would
    never run on the captured bytes and the native receipt would attest less.
    """
    inventory = _trace_inventory(
        root,
        artifact,
        max_bytes=max_bytes,
        max_source_bytes=max_source_bytes,
        max_total_bytes=max_total_bytes,
        max_artifacts=max_artifacts,
    )
    profile, training, generation = inventory.profile, inventory.training, inventory.generation
    train = read_training_progress(
        root,
        training,
        profile,
        root / "source",
        max_source_bytes=max_source_bytes,
        max_artifact_bytes=max_bytes,
        memo=memo,
    )
    gen = read_generation(
        root,
        generation,
        training,
        profile,
        root / "source",
        max_source_bytes=max_source_bytes,
        max_artifact_bytes=max_bytes,
        memo=memo,
    )
    if train["status"] != "training_complete" or gen["status"] != "generation_complete":
        raise ValueError("capture does not contain a completed model trace")
    for diagnostic in inventory.diagnostics:
        read_span_diagnostics(
            root,
            diagnostic,
            training,
            profile,
            root / "source",
            max_source_bytes=max_source_bytes,
            max_artifact_bytes=max_bytes,
            memo=memo,
        )
    for item in inventory.artifacts.values():
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
                    memo=memo,
                )
    return inventory.manifest
