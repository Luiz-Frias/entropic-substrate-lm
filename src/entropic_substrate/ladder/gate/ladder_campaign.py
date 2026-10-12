"""Content-bound scientific authorization shared by explicit rung adapters.

This is a boundary over the existing lifecycle and artifact services, not a
workflow engine. Local smoke records have no promotion path through it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    artifact_mapping,
    object_fields,
    read_bound_artifact,
    read_json_artifact,
)
from entropic_substrate.kernel.identity.d1_primitives import (
    canonical_digest,
)
from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    FiledVerdictEvidence,
    LifecycleRecord,
    LifecycleState,
    SuccessorAuthorization,
)
from entropic_substrate.ladder.register.d1_provenance import require_lowercase_digest

if TYPE_CHECKING:
    from entropic_substrate.ladder.analyze.ladder_registration import StatisticalDesign
    from entropic_substrate.ladder.file.d1_filing_graph import ValidatedD1Filing
    from entropic_substrate.ladder.register.d1_confirmatory_matrix import D1GitProvenancePort


def actor_from_mapping(value: object) -> ActorIdentity:
    row = object_fields(value, "actor_id session_id role")
    if any(type(item) is not str for item in row.values()):
        raise ValueError("actor identity fields must be strings")
    return ActorIdentity(
        cast(str, row["actor_id"]), cast(str, row["session_id"]), ActorRole(row["role"])
    )


def _text(value: object) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("ladder identifier must be nonempty trimmed text")
    return value


def _sha(value: object) -> str:
    result = _text(value)
    require_lowercase_digest(result, label="ladder identity")
    return result


_TOKEN = object()


def validate_ladder_phase_transition(
    rung: str, phase: str | None, target: str, target_phase: str | None
) -> None:
    """The learned D4 campaign is gated by a filed fixed-control campaign."""
    allowed = {
        ("d1", None, "d2", None),
        ("d2", None, "d3", None),
        ("d3", None, "d4", "fixed"),
        ("d3", None, "d5", None),
        ("d4", "fixed", "d4", "learned"),
        ("d4", "fixed", "d5", None),
        ("d4", "learned", "d5", None),
    }
    if (rung, phase, target, target_phase) not in allowed:
        raise ValueError("invalid registered ladder phase transition")


@dataclass(frozen=True, slots=True, init=False)
class FiledLadderGate:
    """Eligibility derived from linked filed artifacts and lifecycle authority."""

    rung: str
    successor_rung: str
    d4_phase: str | None
    successor_phase: str | None
    record: LifecycleRecord
    evidence: FiledVerdictEvidence

    def __init__(
        self,
        rung: str,
        record: LifecycleRecord,
        evidence: FiledVerdictEvidence,
        *,
        successor_rung: str | None = None,
        d4_phase: str | None = None,
        successor_phase: str | None = None,
        _token: object = None,
    ) -> None:
        if _token is not _TOKEN:
            raise TypeError("FiledLadderGate requires a verified filing adapter")
        target = successor_rung or f"d{int(rung[1:]) + 1}"
        validate_ladder_phase_transition(rung, d4_phase, target, successor_phase)
        if record.rung.lower() != rung or record.state is not LifecycleState.SUCCESSOR_AUTHORIZED:
            raise ValueError("filed gate requires the authorized rung lifecycle")
        if (
            evidence.successor_authorization is not SuccessorAuthorization.AUTHORIZED
            or evidence.open_major_findings
        ):
            raise ValueError("filed gate has no scientific successor authority")
        if not any(event.filed_verdict == evidence for event in record.events):
            raise ValueError("filed evidence does not belong to this lifecycle")
        object.__setattr__(self, "rung", rung)
        object.__setattr__(self, "successor_rung", target)
        object.__setattr__(self, "d4_phase", d4_phase)
        object.__setattr__(self, "successor_phase", successor_phase)
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "evidence", evidence)

    def verify_files(self, root: Path) -> None:
        for event in self.record.events:
            for artifact in event.artifacts:
                read_bound_artifact(root, artifact)


def gate_from_d1_filing(
    root: Path, filing: ValidatedD1Filing, record: LifecycleRecord
) -> FiledLadderGate:
    """Retain D1's complete existing graph validator as its explicit adapter."""
    from entropic_substrate.ladder.file.d1_filing_graph import ValidatedD1Filing

    if type(filing) is not ValidatedD1Filing:
        raise TypeError("D1 predecessor requires ValidatedD1Filing")
    gate = FiledLadderGate("d1", record, filing.to_filed_evidence(), _token=_TOKEN)
    gate.verify_files(root)
    return gate


@dataclass(frozen=True, slots=True, init=False)
class RunAuthorization:
    """An operator-admitted exact cell; execution settings live in the bound protocol."""

    rung: str
    d4_phase: str | None
    run_class: str
    logical_run_id: str
    arm: str
    seed: int
    pairing_id: str
    family_digest: str
    construction_digest: str
    spec_digest: str
    registration_digest: str
    statistical_design_digest: str
    execution_protocol_digest: str
    source_sha256: str
    split_digest: str
    source_split: Mapping[str, int]
    reader_identity: str
    device: str
    campaign_artifact: ArtifactIdentity

    def __init__(self, values: Mapping[str, object], *, _token: object = None) -> None:
        if _token is not _TOKEN:
            raise TypeError("RunAuthorization requires authorize_ladder_run")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema": "d-ladder-run-authorization-v1",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"source_split", "campaign_artifact"}
            },
            "source_split": dict(self.source_split),
            "campaign_artifact": artifact_mapping(self.campaign_artifact),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.manifest)


def read_campaign_design(
    root: Path, artifact: ArtifactIdentity
) -> tuple[dict[str, object], StatisticalDesign]:
    from entropic_substrate.ladder.analyze.ladder_registration import read_statistical_design

    row = object_fields(
        read_json_artifact(root, artifact),
        "schema rung d4_phase statistics preregistration registration_git implementation operator execution_protocol_digest source_sha256 source_split reader_identity device cells",
    )
    if row["schema"] != "d-ladder-campaign-design-v1" or row["rung"] not in {
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
    }:
        raise ValueError("unknown ladder campaign schema or rung")
    if (row["rung"] == "d4" and row["d4_phase"] not in {"fixed", "learned"}) or (
        row["rung"] != "d4" and row["d4_phase"] is not None
    ):
        raise ValueError(
            "D4 campaign requires an explicit fixed/learned phase; other rungs require null"
        )
    stats = artifact_from_mapping(row["statistics"])
    design = read_statistical_design(
        read_bound_artifact(root, stats), expected_sha256=stats.content_hash.digest
    )
    if design.run_class == "confirmatory":
        prereg = artifact_from_mapping(row["preregistration"])
        read_bound_artifact(root, prereg)
        if design.registered_design_digest != prereg.content_hash.digest:
            raise ValueError("statistical design and preregistration digest differ")
    elif row["preregistration"] is not None:
        raise ValueError("pilot cannot claim confirmatory preregistration")
    if design.run_class == "pilot" and row["registration_git"] is not None:
        raise ValueError("pilot must not claim confirmatory Git registration")
    if actor_from_mapping(row["implementation"]).role is not ActorRole.IMPLEMENTATION:
        raise ValueError("campaign requires implementation actor")
    if actor_from_mapping(row["operator"]).role is not ActorRole.OPERATOR:
        raise ValueError("campaign requires operator actor")
    if row["rung"] == "d1" and set(dict(design.construction_digests)) != {
        "none",
        "sg",
        "ac",
        "both",
    }:
        raise ValueError("D1 adapter requires its exact four-arm roster")
    if type(row["cells"]) is not list:
        raise ValueError("campaign cells must be an explicit roster")
    cells = [
        object_fields(item, "logical_run_id arm seed pairing_id spec_digest predecessor_strategy")
        for item in row["cells"]
    ]
    expected = {(arm, seed) for arm, _ in design.construction_digests for seed in design.seeds}
    observed = {(item["arm"], item["seed"]) for item in cells}
    if (
        len(cells) != len(expected)
        or observed != expected
        or len({_text(item["logical_run_id"]) for item in cells}) != len(cells)
    ):
        raise ValueError("campaign roster must equal the exact paired arm/seed matrix")
    pairs: dict[int, str] = {}
    for item in cells:
        seed = item["seed"]
        if type(seed) is not int:
            raise ValueError("campaign seed must be an integer")
        pair = _text(item["pairing_id"])
        if seed in pairs and pairs[seed] != pair:
            raise ValueError("paired seed has inconsistent pairing identity")
        pairs[seed] = pair
        _sha(item["spec_digest"])
        _text(item["predecessor_strategy"])
    if len(set(pairs.values())) != len(pairs):
        raise ValueError("pairing identity aliases distinct seeds")
    return row, design


def verify_ladder_registration(
    root: Path,
    design_artifact: ArtifactIdentity,
    registration_git: D1GitProvenancePort | None = None,
) -> None:
    """Check committed scientific rules through the existing read-only Git port."""
    row, design = read_campaign_design(root, design_artifact)
    if design.run_class == "confirmatory":
        proof = object_fields(
            row["registration_git"],
            "preregistration_commit implementation_commit preregistration_path statistics_path",
        )
        if registration_git is None:
            raise ValueError("confirmatory authorization requires committed Git registration")
        before, implementation = (
            _text(proof["preregistration_commit"]),
            _text(proof["implementation_commit"]),
        )
        prereg = artifact_from_mapping(row["preregistration"])
        if (
            registration_git.commit_time(before) is None
            or registration_git.commit_time(implementation) is None
            or not registration_git.is_ancestor(before, implementation)
            or registration_git.blob_sha256(before, _text(proof["preregistration_path"]))
            != prereg.content_hash.digest
            or registration_git.blob_sha256(before, _text(proof["statistics_path"]))
            != design.source_sha256
        ):
            raise ValueError(
                "committed preregistration/statistics or implementation ancestry mismatch"
            )


def authorize_ladder_run(
    root: Path,
    design_artifact: ArtifactIdentity,
    logical_run_id: str,
    *,
    operator: ActorIdentity,
    predecessor: FiledLadderGate | None = None,
    registration_git: D1GitProvenancePort | None = None,
) -> RunAuthorization:
    """Bind a prepared design cell without allocating resources or inferring a gate."""
    row, design = read_campaign_design(root, design_artifact)
    if operator != actor_from_mapping(row["operator"]) or operator.role is not ActorRole.OPERATOR:
        raise ValueError("run authorization requires the prepared operator identity")
    verify_ladder_registration(root, design_artifact, registration_git)
    cells = cast(list[dict[str, object]], row["cells"])
    selected = [item for item in cells if item["logical_run_id"] == logical_run_id]
    if len(selected) != 1:
        raise ValueError("logical run ID is absent from prepared roster")
    cell = selected[0]
    rung = cast(str, row["rung"])
    if rung != "d1":
        if (
            type(predecessor) is not FiledLadderGate
            or predecessor.successor_rung != rung
            or predecessor.successor_phase != row["d4_phase"]
        ):
            raise ValueError("registered filed predecessor gate is required")
        predecessor.verify_files(root)
        if cell["predecessor_strategy"] not in predecessor.evidence.eligible_strategies:
            raise ValueError("predecessor did not authorize this strategy")
    split = object_fields(
        row["source_split"], "train_start train_end eval_start eval_end source_total_bytes"
    )
    if any(type(value) is not int for value in split.values()):
        raise ValueError("source split positions must be integers")
    bounds = cast(dict[str, int], split)
    if not (
        0
        <= bounds["train_start"]
        < bounds["train_end"]
        <= bounds["eval_start"]
        < bounds["eval_end"]
        <= bounds["source_total_bytes"]
    ):
        raise ValueError("source split must contain disjoint ordered train/eval intervals")
    device = _text(row["device"])
    if (
        device != "cpu"
        and device != "cuda"
        and not (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("scientific device must be CPU or an explicit CUDA device")
    values: dict[str, object] = {
        **{
            key: cell[key] for key in ("logical_run_id", "arm", "seed", "pairing_id", "spec_digest")
        },
        "rung": rung,
        "d4_phase": row["d4_phase"],
        "run_class": design.run_class,
        "family_digest": design.family_digest,
        "construction_digest": dict(design.construction_digests)[cast(str, cell["arm"])],
        "registration_digest": design.registered_design_digest
        or design_artifact.content_hash.digest,
        "statistical_design_digest": design.source_sha256,
        "execution_protocol_digest": _sha(row["execution_protocol_digest"]),
        "source_sha256": _sha(row["source_sha256"]),
        "split_digest": canonical_digest(split),
        "source_split": MappingProxyType(bounds),
        "reader_identity": _text(row["reader_identity"]),
        "device": device,
        "campaign_artifact": design_artifact,
    }
    return RunAuthorization(values, _token=_TOKEN)
