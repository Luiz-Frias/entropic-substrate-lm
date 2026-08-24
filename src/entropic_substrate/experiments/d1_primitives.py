"""Scalar, encoder, and boundary contracts shared by the D1 family.

This module intentionally has no trainer or factory dependency.  It supplies
the immutable values from which the later stabilization, provenance, and
family modules compose a D1 experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import string
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

_SHA256_LENGTH: Final[int] = 64
CONSTRUCTION_SCHEMA: Final[str] = "d1-construction-v2"
FAMILY_SCHEMA: Final[str] = "d1-family-v1"
AGGREGATION_SCHEMA: Final[str] = "d1-aggregation-v2"
EXECUTION_SCHEMA: Final[str] = "d1-execution-v2"


def require_nonempty_string(value: str, *, label: str) -> None:
    """Refuse blank construction identities at their boundary."""
    if type(value) is not str or not value.strip():
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)


def require_sha256(value: str, *, label: str) -> None:
    """Refuse any identity that is not a complete SHA-256 digest."""
    if type(value) is not str or len(value) != _SHA256_LENGTH:
        msg = f"{label} must be a 64-character hexadecimal SHA-256 digest"
        raise ValueError(msg)
    if any(character not in string.hexdigits for character in value):
        msg = f"{label} must be a 64-character hexadecimal SHA-256 digest"
        raise ValueError(msg)


def require_commit(value: str, *, label: str) -> None:
    """Accept a short or full hexadecimal VCS commit identity."""
    if type(value) is not str or not 7 <= len(value) <= _SHA256_LENGTH:
        msg = f"{label} must be a 7-to-64-character hexadecimal commit identity"
        raise ValueError(msg)
    if any(character not in string.hexdigits for character in value):
        msg = f"{label} must be a 7-to-64-character hexadecimal commit identity"
        raise ValueError(msg)


def require_positive_finite(value: float, *, label: str) -> None:
    """Require a finite positive numerical construction constant."""
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0.0:
        msg = f"{label} must be a positive finite scalar"
        raise ValueError(msg)


def canonical_digest(manifest: Mapping[str, object]) -> str:
    """Return the SHA-256 of canonical recursively key-sorted JSON."""
    payload = json.dumps(
        dict(manifest),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


D1_ENCODER_INITIALIZATION_NAME: Final[str] = "d1-seeded-normal-v1"
D1_ENCODER_INITIALIZATION_MANIFEST: Final[Mapping[str, object]] = MappingProxyType(
    {
        "algorithm": D1_ENCODER_INITIALIZATION_NAME,
        "embedding": {"distribution": "normal", "mean": 0.0, "std": 0.02},
        "layer_norm": {"bias": 0.0, "weight": 1.0},
        "linear": {"bias": 0.0, "distribution": "normal", "mean": 0.0, "std": 0.02},
        "schema_version": "d1-encoder-initialization-v1",
    }
)
D1_ENCODER_INITIALIZATION_DIGEST: Final[str] = canonical_digest(D1_ENCODER_INITIALIZATION_MANIFEST)


class RunClass(StrEnum):
    """Lifecycle class; a pilot is excluded by type, not its display name."""

    SMOKE = "smoke"
    PILOT = "pilot"
    CONFIRMATORY = "confirmatory"


class D1Arm(StrEnum):
    """The complete orthogonal D1 stabilization matrix."""

    NONE = "none"
    SG = "sg"
    AC = "ac"
    BOTH = "both"


class TargetUpdateStrategy(StrEnum):
    """How target-encoder parameters receive updates."""

    ONLINE = "online"
    STOP_GRADIENT_EMA = "stop_gradient_ema"


class CollapseStrategy(StrEnum):
    """Whether D1 applies an explicit state anti-collapse penalty."""

    NONE = "none"
    MOMENT = "moment"


class PredictiveMetric(StrEnum):
    """Metric family named explicitly while D1 keeps geometry off."""

    EUCLIDEAN = "euclidean"
    DECODER_FISHER_PULLBACK = "decoder_fisher_pullback"


class GeometryConsumer(StrEnum):
    """Whether a metric changes latent dynamics."""

    OFF = "off"
    PULLBACK_RESIDUAL_RETRACTION = "pullback_residual_retraction"


class FutureLawFamily(StrEnum):
    """The deliberately minimal D1 target law."""

    NONLINEAR_FIXED_VARIANCE_GAUSSIAN = "nonlinear_fixed_variance_gaussian"


class FutureActivation(StrEnum):
    """Nonlinearity is an explicit construction constant."""

    SILU = "silu"
    GELU = "gelu"


class EMAKind(StrEnum):
    """Target-teacher schedule identity for stop-gradient arms."""

    FIXED = "fixed"
    COSINE_SCHEDULE = "cosine_schedule"


class CovarianceNormalization(StrEnum):
    """Exact normalization of the covariance penalty."""

    OFF_DIAGONAL_SUM_PER_DIMENSION = "off_diagonal_sum_per_dimension"


class StateLiftPosture(StrEnum):
    """D1 does not introduce an additional hidden-state lift."""

    IDENTITY = "identity"


class LatentDynamicsPosture(StrEnum):
    """D1 holds latent dynamics at the Euclidean residual control."""

    EUCLIDEAN_RESIDUAL = "euclidean_residual"


class EvidenceClass(StrEnum):
    """Evidence categories visible through every lifecycle transition."""

    MEASURED = "measured"
    DERIVED_FROM_MEASURED = "derived_from_measured"
    MODEL_ESTIMATE = "model_estimate"
    RATE_ESTIMATE = "rate_estimate"
    PROVIDER_BILLED = "provider_billed"


@dataclass(frozen=True, slots=True)
class EncoderShape:
    """Legacy width-only contract retained for narrow adapter compatibility."""

    online_width: int
    target_width: int

    def __post_init__(self) -> None:
        if type(self.online_width) is not int or self.online_width < 1:
            msg = "online_width must be a positive integer"
            raise ValueError(msg)
        if type(self.target_width) is not int or self.target_width < 1:
            msg = "target_width must be a positive integer"
            raise ValueError(msg)
        if self.online_width != self.target_width:
            msg = "D1 requires equal widths for online and target encoders"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {"online_width": self.online_width, "target_width": self.target_width}


@dataclass(frozen=True, slots=True)
class EncoderInitializationIdentity:
    """Immutable initialization construction used by the causal encoder factory."""

    name: str
    construction_digest: str

    def __post_init__(self) -> None:
        require_nonempty_string(self.name, label="encoder initialization name")
        require_sha256(self.construction_digest, label="encoder initialization digest")

    @property
    def manifest(self) -> dict[str, object]:
        return {"construction_digest": self.construction_digest, "name": self.name}


def default_d1_encoder_initialization_identity() -> EncoderInitializationIdentity:
    """Identity of the one initialization algorithm executable by D1 v1."""
    return EncoderInitializationIdentity(
        name=D1_ENCODER_INITIALIZATION_NAME,
        construction_digest=D1_ENCODER_INITIALIZATION_DIGEST,
    )


@dataclass(frozen=True, slots=True)
class CausalEncoderSpec:
    """Full factory-relevant construction of one causal encoder."""

    latent_width: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    block_size: int
    dropout: float
    bias: bool
    rms_epsilon: float
    initialization: EncoderInitializationIdentity

    def __post_init__(self) -> None:
        if type(self.latent_width) is not int or self.latent_width < 1:
            msg = "latent_width must be a positive integer"
            raise ValueError(msg)
        if type(self.n_layers) is not int or self.n_layers < 1:
            msg = "n_layers must be a positive integer"
            raise ValueError(msg)
        if type(self.n_heads) is not int or self.n_heads < 1:
            msg = "n_heads must be a positive integer"
            raise ValueError(msg)
        if self.latent_width % self.n_heads:
            msg = "latent_width must be divisible by n_heads"
            raise ValueError(msg)
        if (self.latent_width // self.n_heads) % 2:
            msg = "D1 requires an even attention head dimension"
            raise ValueError(msg)
        require_positive_finite(self.mlp_ratio, label="mlp_ratio")
        if type(self.block_size) is not int or self.block_size < 1:
            msg = "block_size must be a positive integer"
            raise ValueError(msg)
        if (
            type(self.dropout) not in (int, float)
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            msg = "dropout must be finite and in [0, 1)"
            raise ValueError(msg)
        if type(self.bias) is not bool:
            msg = "bias must be a bool"
            raise ValueError(msg)
        require_positive_finite(self.rms_epsilon, label="rms_epsilon")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "bias": self.bias,
            "block_size": self.block_size,
            "dropout": float(self.dropout),
            "initialization": self.initialization.manifest,
            "latent_width": self.latent_width,
            "mlp_ratio": float(self.mlp_ratio),
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "rms_epsilon": float(self.rms_epsilon),
        }


@dataclass(frozen=True, slots=True)
class MatchedEncoderPairSpec:
    """One exact online/target encoder construction, represented twice on purpose."""

    online: CausalEncoderSpec
    target: CausalEncoderSpec

    def __post_init__(self) -> None:
        if self.online != self.target:
            msg = "D1 online and target encoder constructions must exactly match"
            raise ValueError(msg)

    @property
    def latent_width(self) -> int:
        return self.online.latent_width

    @property
    def manifest(self) -> dict[str, object]:
        return {"online": self.online.manifest, "target": self.target.manifest}


@dataclass(frozen=True, slots=True)
class StateConstructionSpec:
    """State identity that explicitly refuses the retired E1.5 field axis."""

    causal_latent_state: bool = True
    legacy_e15_field_axis: None = None

    def __post_init__(self) -> None:
        if self.causal_latent_state is not True:
            msg = "D1 state construction requires a causal latent state"
            raise ValueError(msg)
        if self.legacy_e15_field_axis is not None:
            msg = "D1 state construction forbids E1.5 field axis leakage"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {"causal_latent_state": self.causal_latent_state}


@dataclass(frozen=True, slots=True)
class D1BoundaryStateSpec:
    """D1 boundary/state choices that must not silently vary by arm.

    Alphabet cardinality is intentionally absent: it enters solely through the
    observation-layer ``CodecIdentity``.
    """

    online_target_observation_adapters_tied: bool = False
    language_readout_tied_to_observation_adapter: bool = False
    state_lift: StateLiftPosture = StateLiftPosture.IDENTITY
    dynamics: LatentDynamicsPosture = LatentDynamicsPosture.EUCLIDEAN_RESIDUAL

    def __post_init__(self) -> None:
        if self.online_target_observation_adapters_tied is not False:
            msg = "D1 requires untied online and target observation adapters"
            raise ValueError(msg)
        if self.language_readout_tied_to_observation_adapter is not False:
            msg = "D1 requires an untied language readout"
            raise ValueError(msg)
        if self.state_lift is not StateLiftPosture.IDENTITY:
            msg = "D1 requires identity state-lift posture"
            raise ValueError(msg)
        if self.dynamics is not LatentDynamicsPosture.EUCLIDEAN_RESIDUAL:
            msg = "D1 requires Euclidean residual dynamics posture"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "dynamics": self.dynamics.value,
            "language_readout_tied_to_observation_adapter": (
                self.language_readout_tied_to_observation_adapter
            ),
            "online_target_observation_adapters_tied": (
                self.online_target_observation_adapters_tied
            ),
            "state_lift": self.state_lift.value,
        }
