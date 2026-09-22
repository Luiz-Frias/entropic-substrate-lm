"""Complete numerical H profiles, separate from spending and scientific authority."""

import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from inspect import signature
from typing import Literal, Self, cast

from entropic_substrate.__legacy.evidence.lifecycle import ContentHash
from entropic_substrate.__legacy.predictive_state.dynamics import SolverConfig
from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.codec import CodecSpec
from entropic_substrate.core.sentence.consumer import ConsumerMode
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.records import content_identity
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.spine import SpineSpec


def _numbers(values: tuple[float, ...]) -> None:
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("profile numbers must be finite and not booleans")


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    epsilon: float
    clip_norm: float
    future_weight: float

    def __post_init__(self) -> None:
        _numbers(
            (
                self.learning_rate,
                self.weight_decay,
                self.epsilon,
                self.clip_norm,
                self.future_weight,
            )
        )
        if (
            min(self.learning_rate, self.epsilon, self.clip_norm) <= 0
            or min(self.weight_decay, self.future_weight) < 0
        ):
            raise ValueError("invalid optimizer rates or objective weight")
        if type(self.betas) is not tuple or len(self.betas) != 2:
            raise ValueError("optimizer betas require an explicit pair")
        _numbers(self.betas)
        if any(not 0 <= beta < 1 for beta in self.betas):
            raise ValueError("optimizer betas must lie in [0, 1)")


@dataclass(frozen=True, slots=True)
class RunSchedule:
    warmup_steps: int
    steps: int
    batch_size: int
    checkpoint_interval: int
    evaluation_interval: int
    evaluation_batches: int
    importance_samples: tuple[int, ...]
    importance_chunk: int
    generation_sentences: int

    def __post_init__(self) -> None:
        integers = (
            self.warmup_steps,
            self.steps,
            self.batch_size,
            self.checkpoint_interval,
            self.evaluation_interval,
            self.evaluation_batches,
            self.importance_chunk,
            self.generation_sentences,
        )
        if any(type(n) is not int or n < 1 for n in integers):
            raise ValueError("run schedule counts must be positive integers")
        counts = self.importance_samples
        if (
            type(counts) is not tuple
            or not counts
            or any(type(n) is not int or n < 2 for n in counts)
            or tuple(sorted(set(counts))) != counts
        ):
            raise ValueError("importance sample counts must be increasing integers >= 2")
        if self.importance_chunk > counts[-1]:
            raise ValueError("importance chunk exceeds the sample envelope")


def _construct[T](factory: Callable[..., T], value: object) -> T:
    """Every constructor field is required, even where development defaults exist."""
    fields = " ".join(signature(factory).parameters)
    return factory(**recovery_record(value, fields))


def _array(value: object) -> tuple[object, ...]:
    if type(value) is not list:
        raise ValueError("profile sequences must be explicit JSON arrays")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class HProfile:
    name: str
    stage: Literal["H0", "H1", "H2", "H3", "H4", "H5"]
    run_class: Literal["local_tiny", "remote_tiny", "pilot", "confirmatory"]
    source_digest: str
    seed: int
    source_seed: int
    device: Literal["cpu", "cuda"]
    precision: Literal["fp32", "bf16"]
    codec: CodecSpec
    spine: SpineSpec
    law: LawSpec
    consumer: ConsumerMode
    solver: SolverConfig
    optimization: OptimizationSpec
    schedule: RunSchedule
    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.stage not in ("H0", "H1", "H2", "H3", "H4", "H5"):
            raise ValueError("unknown H stage")
        if (
            type(self.name) is not str
            or re.fullmatch(r"h[0-5]-[a-z0-9]+(?:-[a-z0-9]+)*", self.name) is None
            or not self.name.startswith(self.stage.lower() + "-")
        ):
            raise ValueError("profile name must be a safe stage-prefixed name")
        if self.run_class not in ("local_tiny", "remote_tiny", "pilot", "confirmatory"):
            raise ValueError("unknown run class")
        if self.name.endswith("-tiny") != (self.run_class in ("local_tiny", "remote_tiny")):
            raise ValueError("tiny naming and authority class must agree")
        if self.device not in ("cpu", "cuda") or self.precision not in ("fp32", "bf16"):
            raise ValueError("unknown device or precision")
        if self.run_class != "local_tiny" and (self.device, self.precision) != ("cuda", "bf16"):
            raise ValueError("remote and scientific execution requires CUDA/bf16")
        if any(type(n) is not int or n < 0 for n in (self.seed, self.source_seed)):
            raise ValueError("profile seeds must be nonnegative integers")
        if type(self.source_digest) is not str:
            raise ValueError("source digest must be a string")
        ContentHash("sha256", self.source_digest)
        for value, kind in (
            (self.codec, CodecSpec),
            (self.spine, SpineSpec),
            (self.law, LawSpec),
            (self.solver, SolverConfig),
            (self.optimization, OptimizationSpec),
            (self.schedule, RunSchedule),
        ):
            if type(value) is not kind:
                raise TypeError("profile components require exact validated specification types")
        if (
            self.codec.width != self.law.width
            or self.codec.max_sentence_bytes > 4096
            or self.spine.context_sentences > 64
        ):
            raise ValueError("profile violates the declared sentence construction envelope")
        if self.consumer not in ("isotropic", "fisher"):
            raise ValueError("unknown sentence consumer")
        _numbers(
            (self.solver.damping, self.solver.relative_tolerance, self.solver.absolute_tolerance)
        )
        if self.stage == "H1" and (
            self.consumer != "isotropic" or self.optimization.future_weight != 0
        ):
            raise ValueError("H1 preservation cannot bundle Fisher or future-law supervision")
        if self.stage == "H2" and self.consumer != "isotropic":
            raise ValueError("H2 prediction contrasts require isotropic dynamics")
        SentenceByteCodec(self.permutation)

    @property
    def manifest(self) -> dict[str, object]:
        value = {"schema": "h-run-profile-v1", **asdict(self)}
        return cast(dict[str, object], json.loads(json.dumps(value, allow_nan=False)))

    @property
    def identity(self) -> str:
        return content_identity((json.dumps(self.manifest, sort_keys=True, separators=(",", ":")),))

    @classmethod
    def from_manifest(cls, value: object) -> Self:
        row = recovery_record(value, "schema " + " ".join(signature(cls).parameters)).copy()
        if row.pop("schema") != "h-run-profile-v1":
            raise ValueError("unknown H profile version")
        row["codec"] = _construct(CodecSpec, row["codec"])
        row["spine"] = _construct(SpineSpec, row["spine"])
        row["law"] = _construct(LawSpec, row["law"])
        row["solver"] = _construct(SolverConfig, row["solver"])
        optimization = recovery_record(
            row["optimization"], " ".join(signature(OptimizationSpec).parameters)
        ).copy()
        optimization["betas"] = _array(optimization["betas"])
        row["optimization"] = _construct(OptimizationSpec, optimization)
        schedule = recovery_record(
            row["schedule"], " ".join(signature(RunSchedule).parameters)
        ).copy()
        schedule["importance_samples"] = _array(schedule["importance_samples"])
        row["schedule"] = _construct(RunSchedule, schedule)
        row["permutation"] = _array(row["permutation"])
        return _construct(cls, row)
