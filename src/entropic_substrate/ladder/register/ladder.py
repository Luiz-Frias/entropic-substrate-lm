"""Immutable scientific construction for D2-D5, separate from run authority."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

from entropic_substrate.core.latent.dynamics import SolverConfig
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest, require_sha256
from entropic_substrate.ladder.register.d1_family import D1DraftSpec


class Rung(StrEnum):
    D2 = "d2"
    D3 = "d3"
    D4 = "d4"
    D5 = "d5"


class Treatment(StrEnum):
    L = "language"
    S = "supervised"
    G = "geometry"


class MatchingLane(StrEnum):
    MATCHED_BACKBONE = "matched_backbone"
    MATCHED_TOTAL = "matched_total"


@dataclass(frozen=True, slots=True)
class ParameterMatchReference:
    """Measured reference inventory to be rechecked by candidate selection."""

    construction_identity: str
    total_parameters: int
    max_relative_gap: float = 0.01

    def __post_init__(self) -> None:
        require_sha256(self.construction_identity, label="parameter reference")
        if type(self.total_parameters) is not int or self.total_parameters < 1:
            raise ValueError("reference total_parameters must be a positive integer")
        if (
            type(self.max_relative_gap) not in (int, float)
            or not math.isfinite(self.max_relative_gap)
            or not 0 <= self.max_relative_gap < 1
        ):
            raise ValueError("max_relative_gap must be finite in [0, 1)")


@dataclass(frozen=True, slots=True)
class CoarseSpec:
    """Registered router settings, including the explicit objective weight."""

    mode: str
    window: int = 8
    smoothing: float = 0.5
    random_seed: int = 42
    rate_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "identity",
            "fixed",
            "learned",
            "always",
            "never",
            "random",
            "fixed_duration",
            "whitespace",
            "punctuation",
        }:
            raise ValueError("unsupported coarse-graining mode")
        if type(self.window) is not int or self.window < 1:
            raise ValueError("window must be a positive integer")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**63:
            raise ValueError("random_seed must be a nonnegative signed-64-bit integer")
        if not math.isfinite(self.smoothing) or not 0 <= self.smoothing < 1:
            raise ValueError("smoothing must be finite in [0, 1)")
        if not math.isfinite(self.rate_weight) or self.rate_weight <= 0:
            raise ValueError("rate_weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class RungSpec:
    """A fresh seed's complete mechanism selection, never a launch permission.

    Later rungs bind a predecessor construction and permit only their own axis
    to vary. This identity contains no learned predecessor weights.
    """

    base: D1DraftSpec
    rung: Rung
    treatment: Treatment
    seed: int
    horizons: tuple[int, ...] = (1,)
    coarse: CoarseSpec | None = None
    permutation: tuple[int, ...] | None = None
    solver: SolverConfig = field(default_factory=SolverConfig)
    predecessor: RungSpec | None = None
    matching_lane: MatchingLane = MatchingLane.MATCHED_BACKBONE
    parameter_reference: ParameterMatchReference | None = None

    def __post_init__(self) -> None:
        if type(self.base) is not D1DraftSpec or type(self.rung) is not Rung:
            raise TypeError("rung construction requires typed D1 base and Rung")
        if type(self.treatment) is not Treatment or type(self.solver) is not SolverConfig:
            raise TypeError("rung construction requires typed treatment and solver")
        if type(self.matching_lane) is not MatchingLane:
            raise TypeError("matching_lane must be a MatchingLane")
        if self.matching_lane is MatchingLane.MATCHED_TOTAL:
            if type(self.parameter_reference) is not ParameterMatchReference:
                raise ValueError(
                    "matched-total construction requires a measured parameter reference"
                )
        elif self.parameter_reference is not None:
            raise ValueError("matched-backbone construction forbids a parameter reference")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative signed-64-bit integer")
        if (
            type(self.horizons) is not tuple
            or any(type(horizon) is not int for horizon in self.horizons)
            or self.horizons not in ((1,), (4,), (16,), (64,), (1, 4, 16, 64))
        ):
            raise ValueError("horizons must select one registered horizon or the full composite")
        if self.coarse is not None and type(self.coarse) is not CoarseSpec:
            raise TypeError("coarse must be a CoarseSpec")
        if self.permutation is not None and (
            type(self.permutation) is not tuple
            or len(self.permutation) != 256
            or any(type(value) is not int for value in self.permutation)
            or set(self.permutation) != set(range(256))
        ):
            raise ValueError("permutation must be a bijection of 256 byte IDs")
        if self.permutation is not None and self.base.codec.cardinality != 256:
            raise ValueError("byte permutation requires a byte observation boundary")
        if self.rung is Rung.D2:
            if self.horizons != (1,) or self.coarse is not None or self.permutation is not None:
                raise ValueError("D2 varies only the L/S/G treatment")
            if self.predecessor is not None:
                raise ValueError("D2 takes its D1 construction in base")
            if self.matching_lane is not MatchingLane.MATCHED_BACKBONE:
                raise ValueError("D2 holds its predecessor backbone fixed")
            return
        previous = self.predecessor
        if self.treatment is Treatment.L:
            raise ValueError("later rungs require a predictive predecessor treatment")
        if self.rung is Rung.D5 and self.base.codec.cardinality != 256:
            raise ValueError("D5 requires a byte observation boundary")
        expected = {Rung.D3: (Rung.D2,), Rung.D4: (Rung.D3,), Rung.D5: (Rung.D3, Rung.D4)}[
            self.rung
        ]
        # A filed D3 result can retire chunking yet carry the short-horizon
        # state to D5. Construction permits that shape; launch still requires
        # the explicit D3-to-D5 successor authorization.
        if type(previous) is not RungSpec or previous.rung not in expected:
            raise ValueError("later rungs require an admissible predecessor construction")
        if self.treatment is not previous.treatment:
            raise ValueError("later rungs must preserve predecessor treatment")
        if self.rung is Rung.D3 and self.matching_lane is MatchingLane.MATCHED_TOTAL:
            self._validate_matching_shape(previous)
        elif self.base != previous.base:
            raise ValueError("later rungs must preserve predecessor base")
        if self.solver != previous.solver:
            raise ValueError("later rungs must preserve predecessor solver")
        if self.rung is Rung.D3:
            if self.coarse is not None or self.permutation is not None:
                raise ValueError("D3 varies horizons only")
        elif self.horizons != previous.horizons:
            raise ValueError("D4/D5 must preserve predecessor horizons")
        if self.rung is Rung.D4 and (self.coarse is None or self.permutation is not None):
            raise ValueError("D4 requires an explicit coarse-graining control")
        if self.rung is Rung.D5 and self.coarse != previous.coarse:
            raise ValueError("D5 must preserve predecessor coarse-graining construction")
        if self.rung in (Rung.D4, Rung.D5) and (
            self.matching_lane != previous.matching_lane
            or self.parameter_reference != previous.parameter_reference
        ):
            raise ValueError("D4/D5 inherit the selected predecessor matching lane")

    def _validate_matching_shape(self, previous: RungSpec) -> None:
        if replace(self.base, encoder_pair=previous.base.encoder_pair) != previous.base:
            raise ValueError("matched-total may change encoder shape only")
        for selected, anchor in (
            (self.base.encoder_pair.online, previous.base.encoder_pair.online),
            (self.base.encoder_pair.target, previous.base.encoder_pair.target),
        ):
            if (
                replace(selected, latent_width=anchor.latent_width, n_layers=anchor.n_layers)
                != anchor
            ):
                raise ValueError("matched-total may change encoder width/depth shape only")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema": "d-ladder-construction-v1",
            "sampling_horizon_reservation": self.sampling_horizon_reservation,
            "base": self.base.construction_manifest,
            "rung": self.rung.value,
            "treatment": self.treatment.value,
            "seed": self.seed,
            "horizons": self.horizons,
            "coarse": None if self.coarse is None else asdict(self.coarse),
            "permutation": self.permutation,
            "solver": asdict(self.solver),
            "predecessor": None if self.predecessor is None else self.predecessor.digest,
            "matching_lane": self.matching_lane.value,
            "parameter_reference": None
            if self.parameter_reference is None
            else asdict(self.parameter_reference),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.manifest)

    @property
    def sampling_horizon_reservation(self) -> int:
        """D3's horizon comparison shares exactly one source-offset population."""
        return 64 if self.rung is Rung.D3 else max(self.horizons)

    @property
    def scientific_digest(self) -> str:
        """Bind the treatment construction independently of the paired run seed."""
        manifest = self.manifest
        del manifest["seed"]
        manifest["predecessor"] = (
            None if self.predecessor is None else self.predecessor.scientific_digest
        )
        return canonical_digest(manifest)
