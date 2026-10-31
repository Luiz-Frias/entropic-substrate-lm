"""H1 pilot block generator: eight seed blocks of the preservation 2x2, declared only.

Emits complete ``HProfile`` blocks and nothing else. It names no corpus, registers nothing,
authorizes no run and spends nothing: the source digest, the prediction step count and the
worker transfer limits are caller inputs. The limits are checked against the measured
checkpoint floor, which is necessary and not sufficient: ``pilot_transfer_limits`` sizes
limits that fit a cell, from the caller's own measurements of everything the floor omits.

Ruled regime values (FOR-1690) are constants here. The schedule fields the ruling leaves
open (checkpoint interval, importance chunk, generation sentences) are required parameters
with no default, so no unruled number enters a pilot silently. Further profile values are
carried unchanged from the H0 profile fixtures and are unruled for H1: the optimizer epsilon
1e-8, the default ``SolverConfig``, the identity byte permutation, the spine ``alpha_init``
0.05 (the ``SpineSpec`` default) and the device and precision (cuda, bf16).

A profile name is the arm label ``h1-<sentence>-<spine>-pilot``: it recurs across the eight
seed blocks (four arms by eight seeds), and the seed lives in ``HProfile.seed`` only.

This is the first ``ladder.register`` module that imports ``ladder.run`` (``TransferLimits``
from ``worker_request``). The edge is accepted: ``ladder.run`` imports ``ladder.register``,
but this module is a leaf nothing in either package imports, so there is no runtime cycle
(both import orders measured clean 2026-10-28), and the type is the one the worker consumes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import lcm

from entropic_substrate.core.latent.dynamics import SolverConfig
from entropic_substrate.core.sentence.codec import CodecSpec
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.numerics import InterfaceMode
from entropic_substrate.core.sentence.profile import (
    DEVELOPMENT_DIAGNOSTIC_SAMPLES,
    DEVELOPMENT_GENERATION_CONTINUATIONS,
    DEVELOPMENT_IMPORTANCE_SAMPLES,
    HProfile,
    OptimizationSpec,
    RunSchedule,
)
from entropic_substrate.core.sentence.spine import SpineSpec
from entropic_substrate.ladder.register.h_preservation import H1PreservationBlock
from entropic_substrate.ladder.run.worker_request import TransferLimits

# (law.mode, spine.carrier) in the order H1PreservationBlock keeps: MM, SM, MS, SS.
CELL_AXES: tuple[tuple[InterfaceMode, InterfaceMode], ...] = (
    ("magnitude", "magnitude"),
    ("sphere", "magnitude"),
    ("magnitude", "sphere"),
    ("sphere", "sphere"),
)
# One seed per block, shared by its four cells. Confirmation seeds are 1..n with n at most
# the ceiling, so the pilot range starts above it and never overlaps.
CONFIRMATORY_SEED_CEILING = 1000
PILOT_SEEDS: tuple[int, ...] = tuple(range(1001, 1009))

STEP_DIVISOR = 20  # evaluation interval = steps / 20
WARMUP_STEPS = 250
BATCH_SIZE = 4
EVALUATION_BATCHES = 20

# MEASURED on quote pods s3tkc88i0oyp96 and qkfduyfla3b6rt (FOR-1690, 2026-10-28), dev
# widths, optimizer moments included. The earlier "~182 MB" figure was the codec's
# trainable count, not a checkpoint, and is retracted.
CODEC_WARMUP_CHECKPOINT_BYTES = 524_752_197
PREDICTION_CHECKPOINT_BYTES = 1_149_203_227
# Objects the runner writes per seal, measured on the real object worker: the journal, the
# checkpoint, the seal receipt (seal_step) and the progress receipt (runner progress write).
OBJECTS_PER_SEAL = 4
# Objects per evaluation, prediction phase only: the evaluation record (runner evaluation write).
OBJECTS_PER_EVALUATION = 1
# Objects per generation continuation: one generation-N snapshot file each (generation.py).
OBJECTS_PER_CONTINUATION = 1
# Objects that do not scale with the schedule, counted with their capture copies: the
# generation receipt, profile, span diagnostics and span probes (run and capture each, 8), the
# capture manifest and the prepared corpus (2), the native verification (1), the worker start
# and result (2) and the snapshot marker (1). Measured at 14 on the real object worker.
FIXED_WORKER_OBJECTS = 14
# The final snapshot covers run, capture, verification and worker, so every run object is
# published a second time by the capture copy. The same factor doubles checkpoint bytes until
# the capture path dedupes them (FOR-1690 note); a ruled allowance, not a measurement.
CAPTURE_COPIES = 2


@dataclass(frozen=True, slots=True)
class H1PilotBlock:
    """One seed block: the validated 2x2 plus the order its cells must be launched in."""

    seed: int
    block: H1PreservationBlock
    launch_order: tuple[HProfile, ...]


def _positive(value: object, label: str, *, zero_ok: bool = False) -> int:
    if type(value) is not int or value < (0 if zero_ok else 1):
        raise ValueError(f"{label} must be a {'nonnegative' if zero_ok else 'positive'} integer")
    return value


def _seal_counts(schedule: RunSchedule) -> tuple[int, int]:
    """Seals per phase, as the runner takes them: every checkpoint or evaluation step, the last."""
    interval, evaluation = schedule.checkpoint_interval, schedule.evaluation_interval
    warmup = schedule.warmup_steps
    codec = warmup // interval + (1 if warmup % interval else 0)
    steps = schedule.steps
    prediction = steps // interval + steps // evaluation - steps // lcm(interval, evaluation)
    if steps % interval and steps % evaluation:
        prediction += 1  # the final step seals even off both grids
    return codec, prediction


def required_checkpoint_floor(
    schedule: RunSchedule, *, capture_copies: int = CAPTURE_COPIES
) -> TransferLimits:
    """Lower bound on one cell's transfer limits from its retained checkpoints alone.

    Every seal retains one checkpoint (no overwrite), so the byte total is the measured warm-up
    and prediction checkpoint sizes times their seal counts, times the capture copies. The
    object total is the per-seal, per-evaluation and per-continuation file kinds the worker
    writes, times the capture copies, plus the fixed worker objects. The source, request and
    other caller inputs are not in this floor, and neither are the non-checkpoint files the
    final snapshot publishes. The floor is therefore necessary and not sufficient; see
    ``pilot_transfer_limits``.
    """
    if type(schedule) is not RunSchedule:
        raise TypeError("transfer floor requires a validated run schedule")
    _positive(capture_copies, "capture copies")
    codec, prediction = _seal_counts(schedule)
    # The runner also evaluates at the final step, so an off-grid last step adds one.
    evaluations = schedule.steps // schedule.evaluation_interval + (
        1 if schedule.steps % schedule.evaluation_interval else 0
    )
    return TransferLimits(
        max_object_bytes=max(CODEC_WARMUP_CHECKPOINT_BYTES, PREDICTION_CHECKPOINT_BYTES),
        max_total_bytes=capture_copies
        * (codec * CODEC_WARMUP_CHECKPOINT_BYTES + prediction * PREDICTION_CHECKPOINT_BYTES),
        max_objects=capture_copies
        * (
            OBJECTS_PER_SEAL * (codec + prediction)
            + OBJECTS_PER_EVALUATION * evaluations
            + OBJECTS_PER_CONTINUATION * schedule.generation_continuations
        )
        + FIXED_WORKER_OBJECTS,
    )


def pilot_transfer_limits(
    schedule: RunSchedule, *, source_bytes: int, fixed_objects: int, non_checkpoint_bytes: int
) -> TransferLimits:
    """Limits for one cell, from the checkpoint floor and three stated caller inputs.

    ``max_total_bytes`` is exactly the sum of three terms and nothing else: every retained
    checkpoint once per capture copy (``required_checkpoint_floor``), the prepared corpus once
    (``source_bytes``) and ``non_checkpoint_bytes``. The worker's final snapshot also publishes
    the journals, seal and progress receipts, evaluation records, generation states and
    receipt, profile, span diagnostics, capture manifest, verification receipt, worker start
    and result and the snapshot marker, and publishes each run file a second time as its
    capture copy; none of that is a checkpoint, so ``non_checkpoint_bytes`` is the caller's
    measured total of those files, capture copies included. It has no default and no value is
    supplied here: a dev-width figure is measured by the caller before registration, never
    estimated in this module. Zero is accepted as a stated value. On the tiny test worker the
    checkpoint-and-source grant of 983,219 bytes fell short of the final snapshot by 198,313 and
    198,319 bytes on two runs (20.2 percent of the grant, measured 2026-10-30); that ratio does
    not transfer to dev widths.

    ``fixed_objects`` raises only the object count, and the allowance moves neither the
    per-object nor the object bound.
    """
    _positive(source_bytes, "source bytes")
    _positive(fixed_objects, "fixed objects", zero_ok=True)
    _positive(non_checkpoint_bytes, "non-checkpoint bytes", zero_ok=True)
    floor = required_checkpoint_floor(schedule)
    return TransferLimits(
        max_object_bytes=max(floor.max_object_bytes, source_bytes),
        max_total_bytes=floor.max_total_bytes + source_bytes + non_checkpoint_bytes,
        max_objects=floor.max_objects + fixed_objects,
    )


def _check_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    chosen = tuple(seeds)
    if len(chosen) != len(PILOT_SEEDS):
        raise ValueError(f"pilot requires exactly {len(PILOT_SEEDS)} seed blocks")
    if any(type(seed) is not int for seed in chosen) or len(set(chosen)) != len(chosen):
        raise ValueError("pilot seeds must be distinct integers")
    if any(seed <= CONFIRMATORY_SEED_CEILING for seed in chosen):
        raise ValueError(
            f"pilot seeds must lie above the confirmatory range 1..{CONFIRMATORY_SEED_CEILING}"
        )
    return chosen


def _profile(
    sentence: InterfaceMode,
    spine: InterfaceMode,
    seed: int,
    source_digest: str,
    schedule: RunSchedule,
) -> HProfile:
    return HProfile(
        name=f"h1-{sentence}-{spine}-pilot",  # the arm label: recurs across blocks, never the seed
        stage="H1",
        run_class="pilot",
        source_digest=source_digest,
        seed=seed,
        source_seed=seed,
        device="cuda",
        precision="bf16",
        codec=CodecSpec(
            width=384,
            encoder_layers=4,
            decoder_width=384,
            decoder_layers=4,
            head_width=64,
            max_sentence_bytes=4096,
        ),
        spine=SpineSpec(
            width=768,
            layers=12,
            head_width=64,
            context_sentences=64,
            carrier=spine,
            alpha_init=0.05,
        ),
        law=LawSpec(width=384, mode=sentence),
        consumer="isotropic",
        solver=SolverConfig(),
        optimization=OptimizationSpec(1e-3, 0.01, (0.9, 0.95), 1e-8, 1.0, 0.0),
        schedule=schedule,
        permutation=tuple(range(256)),
    )


def h1_pilot_blocks(
    *,
    source_digest: str,
    steps: int,
    limits: TransferLimits,
    checkpoint_interval: int,
    importance_chunk: int,
    generation_sentences: int,
    seeds: Sequence[int] = PILOT_SEEDS,
) -> tuple[H1PilotBlock, ...]:
    """Eight validated paired blocks, one per seed; raises on any inconsistent input.

    ``limits`` is checked against ``required_checkpoint_floor``, which is necessary and not
    sufficient: this function sees neither the corpus nor the non-checkpoint allowance, so a
    pass is not a claim that the limits fit a cell. Size them with ``pilot_transfer_limits``.
    """
    if type(limits) is not TransferLimits:
        raise TypeError("pilot requires explicit worker transfer limits")
    if type(steps) is not int or steps < 1:
        raise ValueError("pilot steps must be a positive integer")
    if steps % STEP_DIVISOR:
        raise ValueError(f"pilot steps must be a multiple of {STEP_DIVISOR}")
    chosen = _check_seeds(seeds)
    schedule = RunSchedule(
        warmup_steps=WARMUP_STEPS,
        steps=steps,
        batch_size=BATCH_SIZE,
        checkpoint_interval=checkpoint_interval,
        evaluation_interval=steps // STEP_DIVISOR,
        evaluation_batches=EVALUATION_BATCHES,
        importance_samples=DEVELOPMENT_IMPORTANCE_SAMPLES,
        importance_chunk=importance_chunk,
        generation_sentences=generation_sentences,
        generation_continuations=DEVELOPMENT_GENERATION_CONTINUATIONS,
        diagnostic_samples=DEVELOPMENT_DIAGNOSTIC_SAMPLES,
    )
    floor = required_checkpoint_floor(schedule)
    for field in ("max_object_bytes", "max_total_bytes", "max_objects"):
        if getattr(limits, field) < getattr(floor, field):
            raise ValueError(
                f"transfer limits {field} {getattr(limits, field)} is below the checkpoint "
                f"floor {getattr(floor, field)}; the floor is necessary, not sufficient (it "
                "omits the source, the non-checkpoint bytes and the fixed objects), so size "
                "limits with pilot_transfer_limits"
            )
    result: list[H1PilotBlock] = []
    for seed in chosen:
        block = H1PreservationBlock(
            tuple(_profile(*axes, seed, source_digest, schedule) for axes in CELL_AXES)
        )
        by_name = {profile.name: profile for profile in block.profiles}
        order = tuple(by_name[name] for group in block.codec_sharing_groups for name in group)
        result.append(H1PilotBlock(seed, block, order))
    return tuple(result)
