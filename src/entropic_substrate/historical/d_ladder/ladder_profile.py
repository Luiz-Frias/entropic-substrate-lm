"""Strict complete D2-D5 profiles over hash-bound D1 and predecessor inputs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast

import yaml

from entropic_substrate.core.latent.dynamics import SolverConfig
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.gate.ladder_campaign import (
    artifact_from_mapping,
    artifact_mapping,
    object_fields,
    read_bound_artifact,
    read_json_artifact,
)
from entropic_substrate.ladder.register.d1_config_profile import (
    D1LaunchScale,
    parse_d1_profile_payload,
)
from entropic_substrate.ladder.register.ladder import (
    CoarseSpec,
    MatchingLane,
    ParameterMatchReference,
    Rung,
    RungSpec,
    Treatment,
)

_FIELDS = (
    "schema base_profile base_scale rung treatment seed horizons coarse permutation "
    "solver predecessor matching_lane parameter_reference spec_digest"
)


def _text(value: object) -> str:
    if type(value) is not str:
        raise ValueError("profile string field has the wrong type")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("profile integer field has the wrong type")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("profile numerical field has the wrong type")
    return float(cast(float, value))


def _integers(value: object) -> tuple[int, ...]:
    if type(value) is not list:
        raise ValueError("profile integer sequence must be an explicit array")
    return tuple(_integer(item) for item in value)


def rung_profile_mapping(
    spec: RungSpec,
    *,
    base_profile: ArtifactIdentity,
    base_scale: str,
    predecessor: ArtifactIdentity | None,
) -> dict[str, object]:
    """Serialize every scientific setting; referenced bytes are verified on load."""
    if type(spec) is not RungSpec:
        raise TypeError("profile requires an exact RungSpec")
    D1LaunchScale(base_scale)
    if (spec.predecessor is None) != (predecessor is None):
        raise ValueError("profile predecessor reference must match the construction")
    return {
        "schema": "d-ladder-profile-v1",
        "base_profile": artifact_mapping(base_profile),
        "base_scale": base_scale,
        "rung": spec.rung.value,
        "treatment": spec.treatment.value,
        "seed": spec.seed,
        "horizons": list(spec.horizons),
        "coarse": None if spec.coarse is None else asdict(spec.coarse),
        "permutation": None if spec.permutation is None else list(spec.permutation),
        "solver": asdict(spec.solver),
        "predecessor": None if predecessor is None else artifact_mapping(predecessor),
        "matching_lane": spec.matching_lane.value,
        "parameter_reference": None
        if spec.parameter_reference is None
        else asdict(spec.parameter_reference),
        "spec_digest": spec.digest,
    }


def load_rung_profile(root: Path, artifact: ArtifactIdentity) -> RungSpec:
    """Reconstruct one bounded predecessor chain from verified artifact snapshots."""
    return _load(root, artifact, frozenset())


def _load(root: Path, artifact: ArtifactIdentity, seen: frozenset[str]) -> RungSpec:
    digest = artifact.content_hash.digest
    if digest in seen or len(seen) >= 4:
        raise ValueError("profile predecessor chain must be acyclic and at most four rungs")
    row = object_fields(read_json_artifact(root, artifact), _FIELDS)
    if row["schema"] != "d-ladder-profile-v1":
        raise ValueError("unsupported ladder profile schema")
    base_bytes = read_bound_artifact(root, artifact_from_mapping(row["base_profile"]))
    payload = yaml.safe_load(base_bytes)
    if type(payload) is not dict:
        raise ValueError("base profile must contain a complete YAML mapping")
    base = parse_d1_profile_payload(payload, scale=D1LaunchScale(_text(row["base_scale"]))).draft
    previous = (
        None
        if row["predecessor"] is None
        else _load(root, artifact_from_mapping(row["predecessor"]), seen | {digest})
    )
    solver = object_fields(
        row["solver"], "damping max_iterations relative_tolerance absolute_tolerance"
    )
    coarse = None
    if row["coarse"] is not None:
        fields = object_fields(row["coarse"], "mode window smoothing random_seed rate_weight")
        coarse = CoarseSpec(
            _text(fields["mode"]),
            _integer(fields["window"]),
            _number(fields["smoothing"]),
            _integer(fields["random_seed"]),
            _number(fields["rate_weight"]),
        )
    reference = None
    if row["parameter_reference"] is not None:
        fields = object_fields(
            row["parameter_reference"], "construction_identity total_parameters max_relative_gap"
        )
        reference = ParameterMatchReference(
            _text(fields["construction_identity"]),
            _integer(fields["total_parameters"]),
            _number(fields["max_relative_gap"]),
        )
    spec = RungSpec(
        base,
        Rung(_text(row["rung"])),
        Treatment(_text(row["treatment"])),
        _integer(row["seed"]),
        horizons=_integers(row["horizons"]),
        coarse=coarse,
        permutation=None if row["permutation"] is None else _integers(row["permutation"]),
        solver=SolverConfig(
            _number(solver["damping"]),
            _integer(solver["max_iterations"]),
            _number(solver["relative_tolerance"]),
            _number(solver["absolute_tolerance"]),
        ),
        predecessor=previous,
        matching_lane=MatchingLane(_text(row["matching_lane"])),
        parameter_reference=reference,
    )
    if spec.digest != row["spec_digest"]:
        raise ValueError("loaded profile differs from its registered spec_digest")
    return spec
