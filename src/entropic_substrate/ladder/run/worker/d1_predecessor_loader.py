"""Re-earn D1 predecessor authority from captured bytes and real Git history."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from entropic_substrate.__legacy.evidence.ladder_campaign import (
    FiledLadderGate,
    actor_from_mapping,
    gate_from_d1_filing,
)
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    object_fields,
    publish_d1_atomic_bytes,
    read_bound_artifact,
    read_json_artifact,
)
from entropic_substrate.kernel.lifecycle import (
    ArtifactIdentity,
    FiledVerdictEvidence,
    LifecycleEvent,
    LifecycleEventKind,
    LifecycleRecord,
    transition,
)
from entropic_substrate.kernel.runtime.git_environment import isolated_git_environment
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate_codec import (
    hydrate_d1_capture_aggregate,
)
from entropic_substrate.ladder.file.d1_filing import D1FilingManifest, validate_d1_filing_manifest
from entropic_substrate.ladder.file.d1_filing_application import D1LocalFilingArtifactStore
from entropic_substrate.ladder.file.d1_filing_graph import verify_d1_filing
from entropic_substrate.ladder.file.d1_filing_receipts import D1FilingGitPort, D1FilingRevisions
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile, load_d1_profile_catalog
from entropic_substrate.ladder.register.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    hydrate_registered_design,
    parse_decision_contract,
    parse_registered_design_declaration,
)
from entropic_substrate.ladder.run.provider.d1_command_runner import D1BoundedProcessRunner
from entropic_substrate.ladder.run.provider.d1_git_provenance_adapter import (
    SubprocessD1GitProvenance,
)


@contextmanager
def verified_git_bundle(
    root: Path, bundle: ArtifactIdentity
) -> Generator[SubprocessD1GitProvenance]:
    """Verify a hash-bound bundle in an isolated bare repository, without checkout."""
    payload = read_bound_artifact(root, bundle)
    with TemporaryDirectory(prefix="d1-git-proof-") as temporary:
        temporary_root = Path(temporary).resolve()
        publish_d1_atomic_bytes(temporary_root, "source.bundle", payload, label="Git bundle")
        proof = temporary_root / "proof.git"
        runner = D1BoundedProcessRunner(timeout_seconds=60.0, max_output_bytes=1024 * 1024)
        result = runner.run(
            (
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "clone",
                "--bare",
                "--no-local",
                "--",
                str(temporary_root / "source.bundle"),
                str(proof),
            ),
            cwd=temporary_root,
            input=b"",
            env=isolated_git_environment(os.environ),
        )
        if result.returncode != 0:
            raise ValueError("Git bundle cannot produce a complete bare proof repository")
        yield SubprocessD1GitProvenance(repo_root=proof)


def _committed(git: D1FilingGitPort, revision: str, artifact: ArtifactIdentity, root: Path) -> None:
    if git.read_file(revision, artifact.path) != read_bound_artifact(root, artifact):
        raise ValueError(f"D1 predecessor committed bytes differ: {artifact.path}")


def _lifecycle(
    root: Path,
    artifact: ArtifactIdentity,
    evidence: FiledVerdictEvidence,
) -> LifecycleRecord:
    row = object_fields(read_json_artifact(root, artifact), "schema events")
    if row["schema"] != "d1-predecessor-lifecycle-v1" or type(row["events"]) is not list:
        raise ValueError("D1 predecessor lifecycle schema or event list differs")
    record = LifecycleRecord.start("d1")
    for value in cast(list[object], row["events"]):
        event = object_fields(value, "event_id kind actor artifacts sealed_packet")
        if type(event["event_id"]) is not str or type(event["kind"]) is not str:
            raise ValueError("D1 predecessor lifecycle event identities must be strings")
        if type(event["artifacts"]) is not list:
            raise ValueError("D1 predecessor lifecycle artifacts must be a list")
        kind = LifecycleEventKind(event["kind"])
        record = transition(
            record,
            LifecycleEvent(
                event["event_id"],
                kind,
                actor_from_mapping(event["actor"]),
                tuple(
                    artifact_from_mapping(item) for item in cast(list[object], event["artifacts"])
                ),
                sealed_packet=None
                if event["sealed_packet"] is None
                else artifact_from_mapping(event["sealed_packet"]),
                filed_verdict=evidence if kind is LifecycleEventKind.FILE else None,
            ),
        )
    return record


def load_d1_predecessor(
    root: Path,
    declaration: ArtifactIdentity,
    *,
    catalog: tuple[D1LaunchProfile, ...] | None = None,
    git: D1FilingGitPort | None = None,
) -> FiledLadderGate:
    """Hydrate capture, registration, thirteen filing roles, and persisted lifecycle.

    Declaration schema ``d1-predecessor-bundle-v1`` uses artifact references for
    filing_manifest, decision_contract, registered_design and lifecycle. It also
    names capture_root, lifecycle_revision, all six filing revisions and an
    optional git_bundle reference. A worker without .git must supply that bundle.
    """
    root = normalize_d1_artifact_root(root)
    row = object_fields(
        read_json_artifact(root, declaration),
        "schema filing_manifest decision_contract registered_design capture_root lifecycle lifecycle_revision revisions git_bundle",
    )
    if row["schema"] != "d1-predecessor-bundle-v1":
        raise ValueError("D1 predecessor declaration schema differs")
    capture_name = normalize_d1_relative_artifact_path(row["capture_root"], label="capture root")
    capture_root = normalize_d1_artifact_root(root / capture_name)
    revisions_row = object_fields(
        row["revisions"], "capture analysis alias_commitment review alias_reveal filing"
    )
    if (
        any(type(value) is not str for value in revisions_row.values())
        or type(row["lifecycle_revision"]) is not str
    ):
        raise ValueError("D1 predecessor revisions must be commit strings")
    revisions = D1FilingRevisions(**cast(dict[str, str], revisions_row))
    lifecycle_revision = row["lifecycle_revision"]
    filing_identity = artifact_from_mapping(row["filing_manifest"])
    manifest_value = read_json_artifact(root, filing_identity)
    manifest = D1FilingManifest.from_mapping(manifest_value)

    def filed_artifact(role: str) -> ArtifactIdentity:
        artifact = manifest.artifacts[role]
        return ArtifactIdentity(artifact.path, artifact.sha256)

    decision_identity = artifact_from_mapping(row["decision_contract"])
    decision = parse_decision_contract(read_bound_artifact(root, decision_identity).decode())
    design_identity = artifact_from_mapping(row["registered_design"])
    design_payload = read_bound_artifact(root, design_identity).decode()
    design_claim = parse_registered_design_declaration(design_payload)
    families = {
        profile.family.family_digest: profile.family
        for profile in (load_d1_profile_catalog() if catalog is None else catalog)
    }
    family = families.get(design_claim.family_identity.family_digest)
    if family is None:
        raise ValueError("D1 predecessor family is not installed")
    preregistration = filed_artifact("preregistration")
    design = hydrate_registered_design(
        design_payload,
        expected_design_digest=design_identity.content_hash.digest,
        family=family,
        decision_contract=decision,
        preregistration_evidence=D1PreregistrationByteEvidence.from_bytes(
            read_bound_artifact(root, preregistration)
        ),
    )
    capture_identity = filed_artifact("raw_capture_manifest")
    capture = hydrate_d1_capture_aggregate(
        read_bound_artifact(root, capture_identity).decode(),
        expected_manifest_digest=capture_identity.content_hash.digest,
        capture_root=capture_root,
    )
    detached = validate_d1_filing_manifest(
        read_json_artifact(root, filed_artifact("machine_verdict")),
        manifest_value,
        decision_contract=decision,
        artifact_root=root,
    )
    if (detached.verdict.preregistration.path, detached.verdict.preregistration.commit) != (
        design.preregistration.path,
        design.preregistration.commit,
    ):
        raise ValueError("D1 predecessor filing and registration identities differ")
    lifecycle_identity = artifact_from_mapping(row["lifecycle"])
    if git is not None and row["git_bundle"] is not None:
        raise ValueError("D1 predecessor must select exactly one Git proof source")
    context = (
        verified_git_bundle(root, artifact_from_mapping(row["git_bundle"]))
        if row["git_bundle"] is not None
        else nullcontext(git if git is not None else SubprocessD1GitProvenance(repo_root=root))
    )
    with context as proof:
        _committed(proof, design.preregistration.commit, decision_identity, root)
        _committed(proof, revisions.capture, design_identity, root)
        _committed(proof, lifecycle_revision, lifecycle_identity, root)
        if not proof.is_ancestor(revisions.filing, lifecycle_revision):
            raise ValueError("D1 predecessor lifecycle predates the verified filing")
        filing = verify_d1_filing(
            detached, capture, revisions, D1LocalFilingArtifactStore(root), proof
        )
        record = _lifecycle(root, lifecycle_identity, filing.to_filed_evidence())
        return gate_from_d1_filing(root, filing, record)
