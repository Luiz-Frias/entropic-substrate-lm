"""Strict, feature-owned launch configuration for the D1 four-arm family."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import yaml

from entropic_substrate.config import DataConfig, TrainConfig, config_from_dict
from entropic_substrate.experiments.d1_execution import (
    BaselineSampling,
    CalibrationSplit,
    D1AmpPolicy,
    D1BaselineCalibrationSpec,
    D1BatchingSpec,
    D1EvaluationSpec,
    D1ExecutionProtocol,
    D1OptimizationSpec,
    D1ResourceProtocol,
)
from entropic_substrate.experiments.d1_family import (
    D1DraftSpec,
    D1ExperimentFamily,
    build_d1_experiment_family,
)
from entropic_substrate.experiments.d1_primitives import (
    CausalEncoderSpec,
    CovarianceNormalization,
    D1Arm,
    D1BoundaryStateSpec,
    EMAKind,
    EncoderInitializationIdentity,
    FutureActivation,
    FutureLawFamily,
    GeometryConsumer,
    LatentDynamicsPosture,
    MatchedEncoderPairSpec,
    PredictiveMetric,
    StateConstructionSpec,
    StateLiftPosture,
    canonical_digest,
    default_d1_encoder_initialization_identity,
)
from entropic_substrate.experiments.d1_stabilization import (
    CollapseParameters,
    D1GeometrySpec,
    D1TargetSpec,
    EMAParameters,
    FutureLawSpec,
)
from entropic_substrate.observation import ByteObservationCodec, CodecIdentity

_PROFILE_ORDER: Final = ("d1-none", "d1-sg", "d1-ac", "d1-both")
_EXPECTED_ARMS: Final = dict(zip(_PROFILE_ORDER, D1Arm, strict=True))
_TOP_LEVEL_KEYS: Final = frozenset(
    "kind name arm codec encoder boundary state target future_law geometry ema collapse data train".split()
)
_ENCODER_KEYS: Final = "latent_width n_layers n_heads mlp_ratio block_size dropout bias rms_epsilon initialization".split()
_BOUNDARY_KEYS: Final = "online_target_observation_adapters_tied language_readout_tied_to_observation_adapter state_lift dynamics".split()
_FUTURE_KEYS: Final = "predictor_hidden_width activation family variance future_loss_weight reduce_per_latent_dimension mean_baseline_calibration_batches mean_baseline_sampling".split()
_EMA_KEYS: Final = (
    "kind decay final_decay schedule_steps updates_after_successful_optimizer_step".split()
)
_COLLAPSE_KEYS: Final = (
    "std_floor variance_weight covariance_weight variance_epsilon covariance_normalization".split()
)


class D1LaunchScale(StrEnum):
    """Full construction or an explicitly selected CPU/GPU tiny execution."""

    FULL = "full"
    TINY = "tiny"
    GPU_TINY = "gpu_tiny"


@dataclass(frozen=True, slots=True)
class D1LaunchProfile:
    """One selected arm plus the complete immutable four-arm construction."""

    name: str
    arm: D1Arm
    family: D1ExperimentFamily
    scale: D1LaunchScale
    _data_snapshot: str
    _train_snapshot: str

    @property
    def draft(self) -> D1DraftSpec:
        """The selected cell; the family remains attached for identity checks."""
        return self.family.draft_for(self.arm)

    @property
    def data(self) -> DataConfig:
        """Return a fresh legacy data adapter; the stored profile stays immutable."""
        payload = cast(dict[str, object], json.loads(self._data_snapshot))
        return config_from_dict({"data": payload}).data

    @property
    def train(self) -> TrainConfig:
        """Return a fresh legacy train adapter; the stored profile stays immutable."""
        payload = cast(dict[str, object], json.loads(self._train_snapshot))
        return config_from_dict({"train": payload}).train

    @property
    def data_manifest(self) -> str:
        return canonical_digest(cast(Mapping[str, object], json.loads(self._data_snapshot)))

    @property
    def train_manifest(self) -> str:
        return canonical_digest(cast(Mapping[str, object], json.loads(self._train_snapshot)))


def read_d1_profile_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        raw: object = yaml.safe_load(handle)
    if not isinstance(raw, dict) or any(type(key) is not str for key in raw):
        msg = f"{path} must contain a string-keyed YAML mapping"
        raise TypeError(msg)
    return cast(dict[str, object], raw)


def _section(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload[name]
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        msg = f"D1 {name} must be a string-keyed mapping"
        raise TypeError(msg)
    return cast(dict[str, object], value)


def require_d1_exact_keys(
    payload: Mapping[str, object],
    expected: Iterable[str],
    *,
    label: str,
) -> None:
    expected_keys = frozenset(expected)
    actual = frozenset(payload)
    unknown = sorted(actual - expected_keys)
    if unknown:
        msg = f"unknown keys for {label}: {unknown}"
        raise ValueError(msg)
    missing = sorted(expected_keys - actual)
    if missing:
        msg = f"missing keys for {label}: {missing}"
        raise ValueError(msg)


def _require_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)
    return value


def _require_int(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        msg = f"{label} must be an integer >= {minimum}"
        raise ValueError(msg)
    return value


def _require_float(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value) or value < minimum:
        msg = f"{label} must be a finite float >= {minimum}"
        raise ValueError(msg)
    if maximum is not None and value > maximum:
        msg = f"{label} must be a finite float <= {maximum}"
        raise ValueError(msg)
    return value


def _snapshot(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _enum_value[T: StrEnum](enum_type: type[T], value: object, *, label: str) -> T:
    if type(value) is not str:
        msg = f"{label} must be a string"
        raise TypeError(msg)
    try:
        return enum_type(value)
    except ValueError as error:
        msg = f"invalid {label}: {value!r}"
        raise ValueError(msg) from error


def _build_codec(payload: Mapping[str, object]) -> CodecIdentity:
    require_d1_exact_keys(
        payload,
        ("kind", "construction_label", "name", "construction_digest", "cardinality"),
        label="D1 codec",
    )
    if payload["kind"] != "byte":
        msg = "D1 requires the identity byte codec"
        raise ValueError(msg)
    if type(payload["construction_label"]) is not str:
        msg = "D1 codec construction_label must be a string"
        raise TypeError(msg)
    codec = ByteObservationCodec(construction_label=payload["construction_label"])
    identity = codec.codec_identity
    manifest = {
        "name": identity.name,
        "construction_digest": identity.construction_digest,
        "cardinality": identity.cardinality,
    }
    stated = {key: payload[key] for key in manifest}
    if codec != ByteObservationCodec() or stated != manifest:
        msg = "D1 requires the registered identity byte codec construction"
        raise ValueError(msg)
    return identity


def _build_encoder(payload: Mapping[str, object]) -> CausalEncoderSpec:
    require_d1_exact_keys(payload, _ENCODER_KEYS, label="D1 encoder")
    initialization = _section(payload, "initialization")
    require_d1_exact_keys(
        initialization,
        ("name", "construction_digest"),
        label="D1 encoder initialization",
    )
    identity = EncoderInitializationIdentity(
        name=cast(str, initialization["name"]),
        construction_digest=cast(str, initialization["construction_digest"]),
    )
    if identity != default_d1_encoder_initialization_identity():
        msg = "D1 encoder initialization identity is not the registered constructor"
        raise ValueError(msg)
    return CausalEncoderSpec(
        latent_width=cast(int, payload["latent_width"]),
        n_layers=cast(int, payload["n_layers"]),
        n_heads=cast(int, payload["n_heads"]),
        mlp_ratio=cast(float, payload["mlp_ratio"]),
        block_size=cast(int, payload["block_size"]),
        dropout=cast(float, payload["dropout"]),
        bias=cast(bool, payload["bias"]),
        rms_epsilon=cast(float, payload["rms_epsilon"]),
        initialization=identity,
    )


def _build_family(
    payload: Mapping[str, object],
    *,
    execution_protocol: D1ExecutionProtocol,
) -> D1ExperimentFamily:
    encoder = _build_encoder(_section(payload, "encoder"))
    boundary = _section(payload, "boundary")
    state = _section(payload, "state")
    target = _section(payload, "target")
    future = _section(payload, "future_law")
    geometry = _section(payload, "geometry")
    ema = _section(payload, "ema")
    collapse = _section(payload, "collapse")
    require_d1_exact_keys(boundary, _BOUNDARY_KEYS, label="D1 boundary")
    require_d1_exact_keys(state, ("causal_latent_state",), label="D1 state")
    require_d1_exact_keys(
        target,
        ("horizon", "latent_displacement", "rms_normalized"),
        label="D1 target",
    )
    require_d1_exact_keys(future, _FUTURE_KEYS, label="D1 future_law")
    require_d1_exact_keys(geometry, ("metric", "consumer"), label="D1 geometry")
    require_d1_exact_keys(ema, _EMA_KEYS, label="D1 EMA")
    require_d1_exact_keys(collapse, _COLLAPSE_KEYS, label="D1 collapse")
    law = FutureLawSpec(
        predictor_hidden_width=cast(int, future["predictor_hidden_width"]),
        activation=_enum_value(FutureActivation, future["activation"], label="activation"),
        family=_enum_value(FutureLawFamily, future["family"], label="future-law family"),
        variance=cast(float, future["variance"]),
        future_loss_weight=cast(float, future["future_loss_weight"]),
        reduce_per_latent_dimension=cast(bool, future["reduce_per_latent_dimension"]),
        mean_baseline_calibration_batches=cast(int, future["mean_baseline_calibration_batches"]),
        mean_baseline_sampling=cast(str, future["mean_baseline_sampling"]),
    )
    return build_d1_experiment_family(
        codec=_build_codec(_section(payload, "codec")),
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
        boundary_state=D1BoundaryStateSpec(
            online_target_observation_adapters_tied=cast(
                bool, boundary["online_target_observation_adapters_tied"]
            ),
            language_readout_tied_to_observation_adapter=cast(
                bool, boundary["language_readout_tied_to_observation_adapter"]
            ),
            state_lift=_enum_value(StateLiftPosture, boundary["state_lift"], label="state_lift"),
            dynamics=_enum_value(LatentDynamicsPosture, boundary["dynamics"], label="dynamics"),
        ),
        state_construction=StateConstructionSpec(
            causal_latent_state=cast(bool, state["causal_latent_state"])
        ),
        target=D1TargetSpec(
            horizon=cast(int, target["horizon"]),
            latent_displacement=cast(bool, target["latent_displacement"]),
            rms_normalized=cast(bool, target["rms_normalized"]),
        ),
        future_law=law,
        geometry=D1GeometrySpec(
            metric=_enum_value(PredictiveMetric, geometry["metric"], label="metric"),
            consumer=_enum_value(GeometryConsumer, geometry["consumer"], label="consumer"),
        ),
        execution_protocol=execution_protocol,
        ema=EMAParameters(
            kind=_enum_value(EMAKind, ema["kind"], label="EMA kind"),
            decay=cast(float, ema["decay"]),
            final_decay=cast(float | None, ema["final_decay"]),
            schedule_steps=cast(int | None, ema["schedule_steps"]),
            updates_after_successful_optimizer_step=cast(
                bool, ema["updates_after_successful_optimizer_step"]
            ),
        ),
        collapse=CollapseParameters(
            std_floor=cast(float, collapse["std_floor"]),
            variance_weight=cast(float, collapse["variance_weight"]),
            covariance_weight=cast(float, collapse["covariance_weight"]),
            variance_epsilon=cast(float, collapse["variance_epsilon"]),
            covariance_normalization=_enum_value(
                CovarianceNormalization,
                collapse["covariance_normalization"],
                label="covariance normalization",
            ),
        ),
    )


def _runtime_contracts(
    payload: Mapping[str, object],
) -> tuple[str, str, D1ExecutionProtocol]:
    data_payload = _section(payload, "data")
    train_payload = _section(payload, "train")
    require_d1_exact_keys(
        data_payload,
        (field.name for field in dataclasses.fields(DataConfig)),
        label="D1 DataConfig",
    )
    require_d1_exact_keys(
        train_payload,
        (field.name for field in dataclasses.fields(TrainConfig)),
        label="D1 TrainConfig",
    )
    source = _require_string(data_payload["source"], label="data.source")
    if source not in {"enwik8", "fineweb", "synthetic"}:
        msg = "data.source must be enwik8, fineweb, or synthetic"
        raise ValueError(msg)
    for key in ("path", "fineweb_name"):
        _require_string(data_payload[key], label=f"data.{key}")
    for key in ("fineweb_chars", "batch_size", "seq_len", "num_eval_batches", "synthetic_bytes"):
        _require_int(data_payload[key], label=f"data.{key}", minimum=1)
    val_fraction = _require_float(
        data_payload["val_fraction"], label="data.val_fraction", minimum=0.0, maximum=1.0
    )
    if val_fraction in {0.0, 1.0}:
        msg = "data.val_fraction must be strictly between zero and one"
        raise ValueError(msg)

    positive_ints = ("max_steps", "eval_interval", "log_interval")
    nonnegative_ints = ("warmup_steps", "seed", "save_interval")
    for key in positive_ints:
        _require_int(train_payload[key], label=f"train.{key}", minimum=1)
    for key in nonnegative_ints:
        _require_int(train_payload[key], label=f"train.{key}", minimum=0)
    if cast(int, train_payload["warmup_steps"]) > cast(int, train_payload["max_steps"]):
        msg = "train.warmup_steps cannot exceed train.max_steps"
        raise ValueError(msg)
    for key in ("lr", "min_lr_ratio", "weight_decay", "grad_clip"):
        _require_float(train_payload[key], label=f"train.{key}", minimum=0.0)
    if cast(float, train_payload["lr"]) == 0.0 or cast(float, train_payload["grad_clip"]) == 0.0:
        msg = "train.lr and train.grad_clip must be positive"
        raise ValueError(msg)
    if not 0.0 < cast(float, train_payload["min_lr_ratio"]) <= 1.0:
        msg = "train.min_lr_ratio must be in (0, 1]"
        raise ValueError(msg)
    betas = train_payload["betas"]
    if (
        type(betas) is not list
        or len(betas) != 2
        or any(type(beta) is not float or not 0.0 <= beta < 1.0 for beta in betas)
    ):
        msg = "train.betas must be a YAML list of two floats in [0, 1)"
        raise ValueError(msg)
    amp = _enum_value(D1AmpPolicy, train_payload["amp_dtype"], label="train.amp_dtype")
    if type(train_payload["compile"]) is not bool:
        msg = "train.compile must be a bool"
        raise ValueError(msg)
    for key in ("device", "results_dir", "models_dir"):
        _require_string(train_payload[key], label=f"train.{key}")
    resume = train_payload["resume"]
    if resume is not None:
        _require_string(resume, label="train.resume")

    target = _section(payload, "target")
    future = _section(payload, "future_law")
    protocol = D1ExecutionProtocol(
        optimization=D1OptimizationSpec(
            learning_rate=cast(float, train_payload["lr"]),
            min_learning_rate_ratio=cast(float, train_payload["min_lr_ratio"]),
            warmup_steps=cast(int, train_payload["warmup_steps"]),
            max_steps=cast(int, train_payload["max_steps"]),
            weight_decay=cast(float, train_payload["weight_decay"]),
            betas=(cast(float, betas[0]), cast(float, betas[1])),
            epsilon=1e-8,
            grad_clip_norm=cast(float, train_payload["grad_clip"]),
            amp=amp,
            compile=train_payload["compile"],
        ),
        batching=D1BatchingSpec(
            batch_size=cast(int, data_payload["batch_size"]),
            sequence_length=cast(int, data_payload["seq_len"]),
            horizon=cast(int, target["horizon"]),
        ),
        baseline=D1BaselineCalibrationSpec(
            split=CalibrationSplit.TRAIN,
            num_batches=cast(int, future["mean_baseline_calibration_batches"]),
            sampling=_enum_value(
                BaselineSampling,
                future["mean_baseline_sampling"],
                label="baseline sampling",
            ),
        ),
        evaluation=D1EvaluationSpec(
            num_batches=cast(int, data_payload["num_eval_batches"]),
            eval_interval=cast(int, train_payload["eval_interval"]),
            log_interval=cast(int, train_payload["log_interval"]),
            save_interval=cast(int, train_payload["save_interval"]),
        ),
        resources=D1ResourceProtocol(),
    )
    return _snapshot(data_payload), _snapshot(train_payload), protocol


def parse_d1_profile_payload(
    payload: dict[str, object], *, scale: D1LaunchScale
) -> D1LaunchProfile:
    require_d1_exact_keys(payload, _TOP_LEVEL_KEYS, label="D1 launch profile")
    if payload["kind"] != "d1":
        msg = "D1 launch profile requires kind='d1'"
        raise ValueError(msg)
    name = payload["name"]
    if type(name) is not str or name not in _EXPECTED_ARMS:
        msg = f"D1 profile name must be one of {_PROFILE_ORDER}"
        raise ValueError(msg)
    arm = _enum_value(D1Arm, payload["arm"], label="D1 arm")
    if _EXPECTED_ARMS[name] is not arm:
        msg = f"D1 profile name {name!r} requires arm={_EXPECTED_ARMS[name].value!r}"
        raise ValueError(msg)
    data_snapshot, train_snapshot, execution_protocol = _runtime_contracts(payload)
    family = _build_family(payload, execution_protocol=execution_protocol)
    expected_source = "enwik8" if scale is D1LaunchScale.FULL else "synthetic"
    if _section(payload, "data")["source"] != expected_source:
        msg = f"D1 {scale.value} profile requires data.source={expected_source!r}"
        raise ValueError(msg)
    if scale is D1LaunchScale.TINY and _section(payload, "train")["device"] != "cpu":
        raise ValueError("D1 CPU tiny profile requires explicit CPU execution")
    if scale is D1LaunchScale.GPU_TINY:
        train = _section(payload, "train")
        if train["device"] != "cuda" or train["amp_dtype"] != "bf16":
            raise ValueError("D1 GPU tiny profile requires CUDA with bf16")
    return D1LaunchProfile(
        name=name,
        arm=arm,
        family=family,
        scale=scale,
        _data_snapshot=data_snapshot,
        _train_snapshot=train_snapshot,
    )


def load_d1_profile(path: str | Path) -> D1LaunchProfile:
    """Parse one strict full-scale D1 YAML profile."""
    return parse_d1_profile_payload(read_d1_profile_mapping(Path(path)), scale=D1LaunchScale.FULL)
