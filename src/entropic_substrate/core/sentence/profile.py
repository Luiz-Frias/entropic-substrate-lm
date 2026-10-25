"""Complete numerical H profiles, separate from spending and scientific authority."""

import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from inspect import signature
from typing import Literal, Self, cast

from entropic_substrate.core.latent.dynamics import SolverConfig
from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.codec import CodecSpec
from entropic_substrate.core.sentence.consumer import ConsumerMode
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.records import content_identity
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.spine import SpineSpec
from entropic_substrate.kernel.lifecycle import ContentHash


def _numbers(values: tuple[float, ...]) -> None:
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("profile numbers must be finite and not booleans")


# FOR-1664 (operator ruling 2026-10-20). The registered development importance-sample
# setting: a curve ending in the 64-sample headline (``eval/*_k64``) plus ONE fixed
# 256-sample diagnostic carried apart in ``RunSchedule.diagnostic_samples``. Every run
# class outside ``SMOKE_RUN_CLASSES`` must carry exactly this; the smoke classes may carry
# any valid schedule, e.g. ``(2, 4)``, which is a plumbing setting whose effective sample
# size is bounded by k and must not be read as convergence.
DEVELOPMENT_IMPORTANCE_SAMPLES: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
DEVELOPMENT_DIAGNOSTIC_SAMPLES: int = 256
# FOR-1674 (operator ruling D1 2026-10-23, provisional, re-ruled at H1 preregistration):
# how many continuations of one prompt a generation run draws. Every run class outside
# ``SMOKE_RUN_CLASSES`` must carry exactly this; the smoke classes may carry any count in
# ``[MIN_GENERATION_CONTINUATIONS, MAX_GENERATION_CONTINUATIONS]`` (the probe kit uses 4).
# The cap is 256 because the per-continuation seed packs the index into the low 8 bits of
# the step key (ladder/run/generation.py ``continuation_seed``): index 256 would alias
# index 0 of the next step. The floor is 2 because one continuation proves no stochasticity.
DEVELOPMENT_GENERATION_CONTINUATIONS: int = 5
MIN_GENERATION_CONTINUATIONS: int = 2
MAX_GENERATION_CONTINUATIONS: int = 256
# The same seed packs the step count above those 8 index bits into a 32-bit key, so a
# schedule may run at most 2**24 - 1 steps; RunSchedule refuses more at construction
# (PR #385 review), rather than generation refusing it after training has been paid for.
MAX_RUN_STEPS: int = 2**24 - 1
SMOKE_RUN_CLASSES: tuple[str, ...] = ("local_tiny", "remote_tiny")


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
    # The importance-sample curve. ``(2, 4)`` is the smoke setting (see
    # ``SMOKE_RUN_CLASSES``): its ESS is bounded by k, never convergence evidence.
    importance_samples: tuple[int, ...]
    importance_chunk: int
    generation_sentences: int
    # Required, no default, right after ``generation_sentences`` (FOR-1674 ruling D1): how
    # many continuations of the one validation prompt a generation run draws, each from its
    # own seed. Placed here, not last, so the FOR-1664 "diagnostic_samples is last" pin holds.
    generation_continuations: int
    # Required, no default, last (FOR-1664 door A (ii)): the fixed diagnostic population,
    # strictly beyond the curve. It tops the single nested population that is drawn once.
    diagnostic_samples: int

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
        if self.steps > MAX_RUN_STEPS:
            raise ValueError(
                f"run schedule steps must be at most {MAX_RUN_STEPS}: the generation seed "
                "packs the step count into 24 bits"
            )
        if (
            type(self.generation_continuations) is not int
            or not MIN_GENERATION_CONTINUATIONS
            <= self.generation_continuations
            <= MAX_GENERATION_CONTINUATIONS
        ):
            raise ValueError(
                "generation continuation count must be an integer from "
                f"{MIN_GENERATION_CONTINUATIONS} to {MAX_GENERATION_CONTINUATIONS}"
            )
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
        if type(self.diagnostic_samples) is not int or self.diagnostic_samples <= counts[-1]:
            raise ValueError(
                "diagnostic sample count must be an integer beyond the importance-sample curve"
            )

    @property
    def evaluated_samples(self) -> tuple[int, ...]:
        """Every k an evaluation reports: the curve, then the diagnostic count.

        The one definition read by the engine (what is scored), the record reader (the
        weight-matrix envelope and the ``by_sample_count`` key set). Derived, never a
        field, so the manifest does not carry it.
        """
        return (*self.importance_samples, self.diagnostic_samples)


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
        if self.name.endswith("-tiny") != (self.run_class in SMOKE_RUN_CLASSES):
            raise ValueError("tiny naming and authority class must agree")
        if self.device not in ("cpu", "cuda") or self.precision not in ("fp32", "bf16"):
            raise ValueError("unknown device or precision")
        if self.run_class != "local_tiny" and (self.device, self.precision) != ("cuda", "bf16"):
            raise ValueError("remote and scientific execution requires CUDA/bf16")
        if any(type(n) is not int or n < 0 for n in (self.seed, self.source_seed)):
            raise ValueError("profile seeds must be nonnegative integers")
        # Two seed families mix ``seed`` with an odd multiplier: the step-keyed EVALUATION
        # seed (seed * 0x9E3779B1 + step) % 2**32 (ladder/run/execution.py) and the
        # GENERATION seed (seed * 0x9E3779B1 + ((steps << 8) | index)) % 2**32
        # (ladder/run/generation.py ``continuation_seed``, FOR-1674). Each is a bijection
        # in seed only for seed < 2**32; at or above it seed and seed + 2**32 would alias.
        # source_seed is not mixed.
        if self.seed >= 2**32:
            raise ValueError("profile seed must be below 2**32 for the step-keyed evaluation seed")
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
        if self.run_class not in SMOKE_RUN_CLASSES and (
            self.schedule.importance_samples != DEVELOPMENT_IMPORTANCE_SAMPLES
            or self.schedule.diagnostic_samples != DEVELOPMENT_DIAGNOSTIC_SAMPLES
            or self.schedule.generation_continuations != DEVELOPMENT_GENERATION_CONTINUATIONS
        ):
            raise ValueError(
                "development runs require the registered importance-sample curve "
                f"{DEVELOPMENT_IMPORTANCE_SAMPLES}, diagnostic count "
                f"{DEVELOPMENT_DIAGNOSTIC_SAMPLES} and generation continuation count "
                f"{DEVELOPMENT_GENERATION_CONTINUATIONS}"
            )
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
        value = {"schema": "h-run-profile-v3", **asdict(self)}
        return cast(dict[str, object], json.loads(json.dumps(value, allow_nan=False)))

    @property
    def identity(self) -> str:
        return content_identity((json.dumps(self.manifest, sort_keys=True, separators=(",", ":")),))

    @classmethod
    def from_manifest(cls, value: object) -> Self:
        row = recovery_record(value, "schema " + " ".join(signature(cls).parameters)).copy()
        if row.pop("schema") != "h-run-profile-v3":
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
