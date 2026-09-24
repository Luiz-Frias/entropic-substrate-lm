"""Prepare the exact D1 verdict and thirteen-role filing bundle without filing authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import cast

from entropic_substrate.kernel.artifacts import (
    D1ArtifactSnapshotError,
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.kernel.verdict.d1_types import (
    FiledGateAuthorization,
    FiledGateRetirement,
    FiledGateValidNegative,
)
from entropic_substrate.ladder.analyze.d1_analysis_receipt_codec import (
    D1AnalysisExecutionReceipt,
    canonical_d1_analysis_receipt_bytes,
)
from entropic_substrate.ladder.capture.ledger.d1_process_manager import D1ProcessManager
from entropic_substrate.ladder.file.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    D1FilingManifest,
    validate_d1_filing_manifest,
)
from entropic_substrate.ladder.file.d1_filing_assembly import (
    D1FilingArtifactMaterial,
    assemble_d1_filing_manifest,
    assemble_d1_narrative,
)
from entropic_substrate.ladder.file.d1_filing_graph import (
    ValidatedD1Filing,
    verify_d1_filing,
)
from entropic_substrate.ladder.file.d1_filing_receipts import (
    D1FilingGitPort,
    D1FilingRevisions,
)
from entropic_substrate.ladder.file.d1_gate import decide_filed_verdict
from entropic_substrate.ladder.file.d1_process_filing import file_d1_verdict
from entropic_substrate.ladder.file.d1_real_arm_verdict import (
    D1AssembledRealArmVerdict,
    D1VerdictDecision,
    assemble_d1_real_arm_verdict,
)
from entropic_substrate.ladder.review.d1_machine_analysis_verification import (
    D1VerifiedMachineAnalysis,
)
from entropic_substrate.ladder.review.d1_review_adjudication import (
    D1ReviewAdjudicationSurface,
)
from entropic_substrate.ladder.review.d1_review_execution import (
    D1ExecutedBlindReview,
    D1ReleasedAliasReveal,
)

__all__ = [
    "D1FilingArtifactPaths",
    "D1FinalizedFiling",
    "D1LocalFilingArtifactStore",
    "D1PreparedFilingBundle",
    "finalize_d1_filing",
    "prepare_d1_filing_bundle",
]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class D1FilingArtifactPaths:
    narrative: str
    machine_verdict: str
    preregistration: str
    raw_capture_manifest: str
    blinded_analysis: str
    analysis_receipt: str
    analysis_isolation_attestation: str
    review_packet: str
    reviewer_prompt: str
    reviewer_report: str
    reviewer_execution_receipt: str
    alias_commitment: str
    alias_reveal: str

    def __post_init__(self) -> None:
        names = tuple(item.name for item in fields(self))
        if names != REQUIRED_D1_FILING_ARTIFACTS:
            raise AssertionError("D1 filing path fields differ from registered role order")
        values = tuple(getattr(self, name) for name in names)
        if any(type(value) is not str or not value or value.strip() != value for value in values):
            raise ValueError("D1 filing paths must be nonempty trimmed strings")
        if len(set(values)) != len(values):
            raise ValueError("D1 filing paths contain a duplicate")

    @property
    def mapping(self) -> MappingProxyType[str, str]:
        return MappingProxyType(
            {role: cast(str, getattr(self, role)) for role in REQUIRED_D1_FILING_ARTIFACTS}
        )


@dataclass(frozen=True, slots=True)
class D1PreparedFilingBundle:
    """Byte-owned preparation only; semantic graph verification grants authority later."""

    verdict: D1AssembledRealArmVerdict
    paths: D1FilingArtifactPaths
    materials: tuple[D1FilingArtifactMaterial, ...]
    manifest: D1FilingManifest

    def __post_init__(self) -> None:
        if type(self.verdict) is not D1AssembledRealArmVerdict:
            raise TypeError("D1 filing preparation requires an assembled real-arm verdict")
        self.verdict.require_factory_attested()
        if type(self.paths) is not D1FilingArtifactPaths:
            raise TypeError("D1 filing preparation requires typed artifact paths")
        if (
            type(self.materials) is not tuple
            or tuple(item.role for item in self.materials) != REQUIRED_D1_FILING_ARTIFACTS
        ):
            raise ValueError("D1 filing preparation material roles differ")
        if type(self.manifest) is not D1FilingManifest:
            raise TypeError("D1 filing preparation requires a typed manifest")
        if any(
            self.manifest.artifacts[item.role].path != self.paths.mapping[item.role]
            for item in self.materials
        ):
            raise ValueError("D1 filing preparation paths differ from its manifest")

    @property
    def payloads(self) -> MappingProxyType[str, bytes]:
        return MappingProxyType({item.role: item.payload for item in self.materials})

    @property
    def manifest_bytes(self) -> bytes:
        return _canonical(
            {
                "artifacts": {
                    role: {
                        "md5": artifact.md5.digest,
                        "path": artifact.path,
                        "sha256": artifact.sha256.digest,
                        "size": artifact.size,
                    }
                    for role, artifact in self.manifest.artifacts.items()
                },
                "rung": "d1",
                "schema_version": "d1-filing-manifest/v2",
            }
        )


class D1LocalFilingArtifactStore:
    """Restartable local publisher and exact-byte read adapter for one filing root."""

    def __init__(self, root: Path) -> None:
        self._root = normalize_d1_artifact_root(root)

    def read_bytes(self, path: str) -> bytes | None:
        relative = normalize_d1_relative_artifact_path(path, label="D1 filing")
        try:
            _, payload, _ = read_d1_artifact_bytes(
                self._root,
                relative,
                label="D1 filing",
            )
        except D1ArtifactSnapshotError as error:
            if isinstance(error.__cause__, FileNotFoundError):
                return None
            raise
        return payload

    def _ensure(self, path: str, payload: bytes, *, label: str) -> ArtifactIdentity:
        existing = self.read_bytes(path)
        if existing is None:
            return publish_d1_atomic_bytes(self._root, path, payload, label=label)
        if existing != payload:
            raise ValueError(f"{label} existing bytes differ from prepared filing")
        digest = hashlib.sha256(payload).hexdigest()
        return ArtifactIdentity(
            str(self._root.joinpath(*Path(path).parts)),
            ContentHash("sha256", digest),
        )

    def publish(
        self,
        prepared: D1PreparedFilingBundle,
        *,
        manifest_path: str,
    ) -> ArtifactIdentity:
        if type(prepared) is not D1PreparedFilingBundle:
            raise TypeError("D1 filing store requires a prepared filing bundle")
        marker = normalize_d1_relative_artifact_path(
            manifest_path,
            label="D1 filing manifest",
        )
        if marker in set(prepared.paths.mapping.values()):
            raise ValueError("D1 filing manifest path duplicates an artifact path")
        for material in prepared.materials:
            self._ensure(
                material.path,
                material.payload,
                label=f"D1 filing {material.role}",
            )
        return self._ensure(
            marker,
            prepared.manifest_bytes,
            label="D1 filing manifest",
        )


def _require_process_manager(value: object) -> None:
    if not isinstance(value, D1ProcessManager):
        raise TypeError("D1 finalized filing requires a terminal process manager")


@dataclass(frozen=True, slots=True)
class D1FinalizedFiling:
    filing: ValidatedD1Filing
    terminal: D1ProcessManager

    def __post_init__(self) -> None:
        if type(self.filing) is not ValidatedD1Filing:
            raise TypeError("D1 finalized filing requires verified semantic evidence")
        _require_process_manager(self.terminal)


def finalize_d1_filing(
    *,
    prepared: D1PreparedFilingBundle,
    artifact_root: Path,
    manifest_path: str,
    revisions: D1FilingRevisions,
    git: D1FilingGitPort,
    manager: D1ProcessManager,
    actor: ActorIdentity,
    filing_event_id: str,
    gate_artifact: ArtifactIdentity,
    gate_event_id: str,
) -> D1FinalizedFiling:
    """Verify committed bytes, file the verdict, and apply only its derived gate."""
    if (
        type(prepared) is not D1PreparedFilingBundle
        or type(revisions) is not D1FilingRevisions
        or type(manager) is not D1ProcessManager
        or type(actor) is not ActorIdentity
        or type(gate_artifact) is not ArtifactIdentity
    ):
        raise TypeError("D1 filing finalization requires exact application values")
    if not all(hasattr(git, method) for method in ("read_file", "is_ancestor")):
        raise TypeError("D1 filing finalization requires a read-only Git port")
    store = D1LocalFilingArtifactStore(artifact_root)
    marker = store.read_bytes(manifest_path)
    if marker is None or marker != prepared.manifest_bytes:
        raise ValueError("D1 filing manifest marker is absent or differs")
    try:
        manifest_value: object = json.loads(marker)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:  # pragma: no cover - equality pins
        raise ValueError("D1 filing manifest marker is invalid JSON") from error
    if type(manifest_value) is not dict:
        raise ValueError("D1 filing manifest marker must be an object")
    detached = validate_d1_filing_manifest(
        prepared.verdict.manifest,
        cast(dict[str, object], manifest_value),
        decision_contract=prepared.verdict.verified.expected.decision_contract,
        artifact_root=artifact_root,
    )
    filing = verify_d1_filing(
        detached,
        prepared.verdict.verified.aggregate,
        revisions,
        store,
        git,
    )
    filed = file_d1_verdict(
        manager,
        actor=actor,
        filing=filing,
        event_id=filing_event_id,
    )
    outcome = decide_filed_verdict(filing)
    if type(outcome) is FiledGateAuthorization:
        terminal = filed.authorize_d2(
            actor=actor,
            artifact=gate_artifact,
            event_id=gate_event_id,
        )
    elif type(outcome) is FiledGateValidNegative:
        terminal = filed.record_valid_negative(
            actor=actor,
            artifact=gate_artifact,
            event_id=gate_event_id,
        )
    elif type(outcome) is FiledGateRetirement:
        terminal = filed.retire(
            actor=actor,
            artifact=gate_artifact,
            event_id=gate_event_id,
        )
    else:  # pragma: no cover - gate returns a closed union
        raise TypeError("D1 filing gate returned an unsupported outcome")
    return D1FinalizedFiling(filing=filing, terminal=terminal)


def _require_bytes(value: object, *, label: str) -> bytes:
    if type(value) is not bytes or not value:
        raise ValueError(f"D1 filing {label} must be nonempty exact bytes")
    return value


# RETRACTED (house rule 4, measured 2026-09-20): this function previously
# also materialized an "analysis_reader" filing artifact from
# ``verified.expected.reader_source.canonical_bytes`` and bound it via a
# ``reader_sha256`` filing digest. The D1 reader-source closure that
# produced ``reader_source`` is retired this phase; `git grep` over tracked
# source for `reader_sha256`/`analysis_reader` returns no persisted-artifact
# consumer of either field once this call site is removed.
def _validate_analysis_materials(
    *,
    verified: D1VerifiedMachineAnalysis,
    analysis_receipt: D1AnalysisExecutionReceipt,
    execution: D1ExecutedBlindReview,
    preregistration_bytes: bytes,
) -> dict[str, bytes]:
    filing = analysis_receipt.filing_bindings
    if filing is None or analysis_receipt.manifest["filing_eligible"] is not True:
        raise ValueError("D1 filing requires a filing-eligible analysis execution receipt")
    raw_capture = _canonical(verified.aggregate.manifest)
    isolation = canonical_d1_analysis_receipt_bytes(analysis_receipt.manifest["isolation"])
    expected = {
        "capture_manifest_sha256": hashlib.sha256(raw_capture).hexdigest(),
        "isolation_attestation_sha256": hashlib.sha256(isolation).hexdigest(),
        "output_sha256": hashlib.sha256(verified.canonical_bytes).hexdigest(),
        "preregistration_sha256": hashlib.sha256(preregistration_bytes).hexdigest(),
    }
    for name, digest in expected.items():
        if getattr(filing, name) != digest:
            raise ValueError(f"D1 filing {name.removesuffix('_sha256')} differs")
    packet = execution.request.packet
    if packet.manifest.get("analysis_receipt_digest") != analysis_receipt.receipt_digest:
        raise ValueError("D1 filing analysis receipt differs from reviewed packet")
    return {
        "raw_capture_manifest": raw_capture,
        "blinded_analysis": verified.canonical_bytes,
        "analysis_receipt": analysis_receipt.canonical_json.encode("utf-8"),
        "analysis_isolation_attestation": isolation,
    }


def prepare_d1_filing_bundle(
    *,
    verified: D1VerifiedMachineAnalysis,
    analysis_receipt: D1AnalysisExecutionReceipt,
    execution: D1ExecutedBlindReview,
    reveal: D1ReleasedAliasReveal,
    adjudication: D1ReviewAdjudicationSurface,
    verdict: str,
    narrative_body: bytes,
    preregistration_bytes: bytes,
    paths: D1FilingArtifactPaths,
) -> D1PreparedFilingBundle:
    """Assemble exact bytes; publication, Git proof, filing, and gating remain separate."""
    if (
        type(verified) is not D1VerifiedMachineAnalysis
        or type(analysis_receipt) is not D1AnalysisExecutionReceipt
        or type(execution) is not D1ExecutedBlindReview
        or type(reveal) is not D1ReleasedAliasReveal
        or type(adjudication) is not D1ReviewAdjudicationSurface
        or type(paths) is not D1FilingArtifactPaths
    ):
        raise TypeError("D1 filing preparation requires exact attested lineage types")
    narrative_body = _require_bytes(narrative_body, label="narrative body")
    preregistration_bytes = _require_bytes(preregistration_bytes, label="preregistration")
    assembled = assemble_d1_real_arm_verdict(
        verified=verified,
        execution=execution,
        reveal=reveal,
        adjudication=adjudication,
        decision=D1VerdictDecision(verdict),
    )
    analysis_materials = _validate_analysis_materials(
        verified=verified,
        analysis_receipt=analysis_receipt,
        execution=execution,
        preregistration_bytes=preregistration_bytes,
    )
    machine = assembled.canonical_bytes
    payloads = {
        "narrative": assemble_d1_narrative(body=narrative_body, machine_verdict=machine),
        "machine_verdict": machine,
        "preregistration": preregistration_bytes,
        **analysis_materials,
        "review_packet": _canonical(execution.request.packet.manifest),
        "reviewer_prompt": execution.request.prompt.text.encode("utf-8"),
        "reviewer_report": execution.report,
        "reviewer_execution_receipt": execution.receipt.canonical_json.encode("utf-8"),
        "alias_commitment": verified.expected.alias_application.commitment.canonical_bytes,
        "alias_reveal": reveal.payload,
    }
    materials = tuple(
        D1FilingArtifactMaterial(role, paths.mapping[role], payloads[role])
        for role in REQUIRED_D1_FILING_ARTIFACTS
    )
    return D1PreparedFilingBundle(
        verdict=assembled,
        paths=paths,
        materials=materials,
        manifest=assemble_d1_filing_manifest(materials),
    )
