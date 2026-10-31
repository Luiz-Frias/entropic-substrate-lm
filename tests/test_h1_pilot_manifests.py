"""The H1 pilot block generator: eight seed blocks of the 2x2, validated, never launched.

The generator emits declared configurations only. It names no corpus, authorizes no run and
spends nothing; digests, steps and the transfer limits are parameters.
"""

import json
import re
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path

import pytest
import tests.test_span_state_collection as span_fixture
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_worker_command_observability import MAX_REQUEST, _worker
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.core.sentence.codec import CodecSpec
from entropic_substrate.core.sentence.profile import (
    DEVELOPMENT_DIAGNOSTIC_SAMPLES,
    DEVELOPMENT_GENERATION_CONTINUATIONS,
    DEVELOPMENT_IMPORTANCE_SAMPLES,
    MAX_RUN_STEPS,
    HProfile,
    RunSchedule,
)
from entropic_substrate.core.sentence.records import content_identity
from entropic_substrate.ladder.register.h_preservation import (
    H1PreservationBlock,
    h1_construction_digest,
)
from entropic_substrate.ladder.register.rungs import h1_pilot
from entropic_substrate.ladder.register.rungs.h1_pilot import (
    BATCH_SIZE,
    CELL_AXES,
    CODEC_WARMUP_CHECKPOINT_BYTES,
    CONFIRMATORY_SEED_CEILING,
    EVALUATION_BATCHES,
    PILOT_SEEDS,
    PREDICTION_CHECKPOINT_BYTES,
    H1PilotBlock,
    _seal_counts,
    h1_pilot_blocks,
    pilot_transfer_limits,
    required_checkpoint_floor,
)
from entropic_substrate.ladder.run.object_worker import run_object_worker
from entropic_substrate.ladder.run.worker.h_probe_log_events import checkpoint_steps
from entropic_substrate.ladder.run.worker_request import TransferLimits
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit

SOURCE = content_identity(("h1-pilot-fixture-source",))
STEPS = 1000
KNOBS = {"checkpoint_interval": 250, "importance_chunk": 8, "generation_sentences": 4}
SOURCE_BYTES = 100_000_000
FIXED_OBJECTS = 20
# A TEST value for the sizing arithmetic only. It is not the dev-width non-checkpoint
# measurement, which the caller takes before registration; see the real-worker tests below.
NON_CHECKPOINT_BYTES = 50_000_000
MM, SM, MS, SS = CELL_AXES
UNBOUNDED = TransferLimits(2**62, 2**62, 2**31)


def probe_schedule(steps: int = STEPS) -> RunSchedule:
    """A pilot schedule built without sizing limits, so a sizing test fails on sizing alone."""
    return (
        h1_pilot_blocks(source_digest=SOURCE, steps=steps, limits=UNBOUNDED, **KNOBS)[0]
        .block.profiles[0]
        .schedule
    )


def limits_for(steps: int = STEPS) -> TransferLimits:
    probe = probe_schedule(steps)
    return pilot_transfer_limits(
        probe,
        source_bytes=SOURCE_BYTES,
        fixed_objects=FIXED_OBJECTS,
        non_checkpoint_bytes=NON_CHECKPOINT_BYTES,
    )


def blocks(steps: int = STEPS, **changes: object) -> tuple[H1PilotBlock, ...]:
    arguments: dict[str, object] = {"source_digest": SOURCE, "steps": steps, **KNOBS}
    arguments.update(changes)
    if "limits" not in arguments:
        arguments["limits"] = limits_for(steps)
    return h1_pilot_blocks(**arguments)  # type: ignore[arg-type]


def test_eight_blocks_with_the_ruled_seeds_shared_by_each_blocks_four_cells() -> None:
    assert PILOT_SEEDS == tuple(range(1001, 1009))
    result = blocks()
    assert [entry.seed for entry in result] == list(PILOT_SEEDS)
    for entry in result:
        assert len(entry.block.profiles) == 4
        assert {p.seed for p in entry.block.profiles} == {entry.seed}
        assert {p.source_seed for p in entry.block.profiles} == {entry.seed}


def test_pilot_seeds_never_overlap_the_confirmatory_range() -> None:
    assert CONFIRMATORY_SEED_CEILING == 1000
    assert set(PILOT_SEEDS).isdisjoint(range(1, CONFIRMATORY_SEED_CEILING + 1))
    assert len(set(PILOT_SEEDS)) == len(PILOT_SEEDS) == 8


@pytest.mark.parametrize(
    "seeds",
    [
        (1000, *range(1002, 1009)),  # one seed inside the confirmatory range
        tuple(range(1, 9)),  # the whole list inside it
        (*range(1001, 1008), 1001),  # a duplicate
        tuple(range(1001, 1008)),  # seven blocks
        (*range(1001, 1009), 1009),  # nine blocks
    ],
)
def test_overlapping_duplicate_or_miscounted_seed_lists_are_refused(
    seeds: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="seed"):
        blocks(seeds=seeds)


def test_cells_are_donor_first_from_the_codec_sharing_groups() -> None:
    assert CELL_AXES == (
        ("magnitude", "magnitude"),
        ("sphere", "magnitude"),
        ("magnitude", "sphere"),
        ("sphere", "sphere"),
    )
    for entry in blocks():
        axes = tuple((p.law.mode, p.spine.carrier) for p in entry.block.profiles)
        assert axes == CELL_AXES
        groups = entry.block.codec_sharing_groups
        assert tuple(p.name for p in entry.launch_order) == tuple(chain.from_iterable(groups))
        launch_axes = tuple((p.law.mode, p.spine.carrier) for p in entry.launch_order)
        # donor MM before its recipient MS, donor SM before its recipient SS
        assert launch_axes == (MM, MS, SM, SS)
        positions = {axis: i for i, axis in enumerate(launch_axes)}
        assert positions[MM] < positions[MS] and positions[SM] < positions[SS]


def test_names_encode_the_axes_and_not_the_seed() -> None:
    """A profile name is the arm label downstream, so it must recur across the seed blocks."""
    entry = blocks()[0]
    for profile in entry.block.profiles:
        assert profile.name == f"h1-{profile.law.mode}-{profile.spine.carrier}-pilot"
    assert len({p.name for p in entry.block.profiles}) == 4


def test_the_roster_is_four_arms_by_eight_seeds_not_thirty_two_arms() -> None:
    result = blocks()
    names_per_block = [tuple(p.name for p in entry.block.profiles) for entry in result]
    assert all(names == names_per_block[0] for names in names_per_block)
    every = [p for entry in result for p in entry.block.profiles]
    assert len(every) == 32 and len({p.name for p in every}) == 4
    for arm in names_per_block[0]:
        assert sorted(p.seed for p in every if p.name == arm) == list(PILOT_SEEDS)


def test_a_valid_custom_seed_list_is_honored_in_order() -> None:
    custom = tuple(range(2001, 2009))
    result = blocks(seeds=custom)
    assert [entry.seed for entry in result] == list(custom)
    for entry in result:
        assert {p.seed for p in entry.block.profiles} == {entry.seed}
        assert {p.source_seed for p in entry.block.profiles} == {entry.seed}


def test_the_emitted_codec_ties_to_the_codec_default_sentence_limit() -> None:
    """The corpus policy check compares against CodecSpec(); a moved default must fail here."""
    for entry in blocks():
        for profile in entry.block.profiles:
            assert profile.codec.max_sentence_bytes == CodecSpec().max_sentence_bytes == 4096


@pytest.mark.parametrize("steps", [20, 1000, 6000])
def test_evaluation_interval_is_a_twentieth_of_the_steps(steps: int) -> None:
    for entry in blocks(steps):
        for profile in entry.block.profiles:
            assert profile.schedule.steps == steps
            assert profile.schedule.evaluation_interval == steps // 20


@pytest.mark.parametrize("steps", [1010, 1001, 19, 21, 1])
def test_steps_that_are_not_a_multiple_of_twenty_are_refused(steps: int) -> None:
    with pytest.raises(ValueError, match="multiple of 20"):
        blocks(steps, limits=UNBOUNDED)


@pytest.mark.parametrize("steps", [0, -20, True, 20.0, "1000"])
def test_zero_negative_boolean_and_non_integer_steps_are_refused(steps: object) -> None:
    with pytest.raises(ValueError, match="steps"):
        blocks(steps, limits=UNBOUNDED)  # type: ignore[arg-type]


def test_the_run_step_cap_is_honored() -> None:
    at_cap = MAX_RUN_STEPS - MAX_RUN_STEPS % 20
    assert at_cap % 20 == 0 and at_cap <= MAX_RUN_STEPS < at_cap + 20
    accepted = blocks(at_cap, checkpoint_interval=at_cap, limits=UNBOUNDED)
    assert accepted[0].block.profiles[0].schedule.steps == at_cap
    with pytest.raises(ValueError, match="at most"):
        blocks(at_cap + 20, checkpoint_interval=at_cap, limits=UNBOUNDED)


def test_the_ruled_regime_values_and_development_widths_are_encoded() -> None:
    for entry in blocks():
        for profile in entry.block.profiles:
            assert (profile.stage, profile.run_class) == ("H1", "pilot")
            assert (profile.device, profile.precision) == ("cuda", "bf16")
            assert profile.consumer == "isotropic"
            assert profile.optimization.future_weight == 0
            assert profile.optimization.learning_rate == 1e-3
            assert profile.optimization.weight_decay == 0.01
            assert profile.optimization.betas == (0.9, 0.95)
            assert profile.optimization.clip_norm == 1.0
            schedule = profile.schedule
            assert schedule.batch_size == 4 and schedule.warmup_steps == 250
            assert schedule.evaluation_batches == 20
            assert schedule.importance_samples == (2, 4, 8, 16, 32, 64)
            assert schedule.importance_samples == DEVELOPMENT_IMPORTANCE_SAMPLES
            assert schedule.diagnostic_samples == 256 == DEVELOPMENT_DIAGNOSTIC_SAMPLES
            assert schedule.generation_continuations == 5
            assert (schedule.checkpoint_interval, schedule.importance_chunk) == (250, 8)
            assert schedule.generation_sentences == 4
            # The widths are pinned here, not read back from the dataclass defaults.
            assert profile.codec == CodecSpec(384, 4, 384, 4, 64, 4096)
            assert profile.law.width == 384
            assert (profile.spine.width, profile.spine.layers) == (768, 12)
            assert (profile.spine.head_width, profile.spine.context_sentences) == (64, 64)
            assert profile.permutation == tuple(range(256))


def test_source_digest_is_the_caller_input_and_must_be_a_hash() -> None:
    other = content_identity(("another-source",))
    assert {p.source_digest for e in blocks(source_digest=other) for p in e.block.profiles} == {
        other
    }
    with pytest.raises(ValueError):
        blocks(source_digest="not-a-digest")


@pytest.mark.parametrize("knob", list(KNOBS))
def test_unruled_schedule_knobs_are_required_and_positive(knob: str) -> None:
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            blocks(**{knob: bad})


def test_every_profile_round_trips_through_its_manifest() -> None:
    for entry in blocks():
        for profile in entry.block.profiles:
            assert HProfile.from_manifest(profile.manifest) == profile
        restored = H1PreservationBlock.from_manifest(entry.block.manifest)
        assert restored.identity == entry.block.identity
        assert restored.manifest["scientific_authority"] == "none"


def test_blocks_differ_only_by_seed_and_name() -> None:
    first, second = blocks()[:2]
    for left, right in zip(first.block.profiles, second.block.profiles, strict=True):
        assert left.identity != right.identity
        assert h1_construction_digest(left) == h1_construction_digest(right)
        assert (
            replace(left, name=right.name, seed=right.seed, source_seed=right.source_seed) == right
        )


# Characterization pin, not a derivation: the four construction digests of the fixture
# regime (SOURCE, STEPS=1000, KNOBS). A change to any ruled value moves all four.
CONSTRUCTION_DIGESTS = {
    "MM": (
        "29fde91b45fbcce86e350e18a6a1c188e5e79ff131b8fc10b68883cd7f64d40f"  # pragma: allowlist secret
    ),
    "SM": (
        "9261cac498c8c1047eeca9e4cb0634b9c2f1fe36e0baecf6e9885e9ae437a755"  # pragma: allowlist secret
    ),
    "MS": (
        "7f280be54ff8f3916a16cdd5c92dd9736a5a54cb656ad937117ee1aa13bc5587"  # pragma: allowlist secret
    ),
    "SS": (
        "42f25b581b31ab8af70da429d4883d7da658fd157d401e7afb3185cfe7b90f51"  # pragma: allowlist secret
    ),
}


def test_construction_digests_are_pinned_and_invariant_across_the_eight_blocks() -> None:
    per_cell = [[h1_construction_digest(p) for p in entry.block.profiles] for entry in blocks()]
    assert all(row == per_cell[0] for row in per_cell)
    assert len(set(per_cell[0])) == 4
    assert dict(zip(("MM", "SM", "MS", "SS"), per_cell[0], strict=True)) == CONSTRUCTION_DIGESTS


def test_the_construction_digest_binds_steps_and_source() -> None:
    base = {h1_construction_digest(p) for p in blocks()[0].block.profiles}
    assert base.isdisjoint(h1_construction_digest(p) for p in blocks(2000)[0].block.profiles)
    other = content_identity(("another-source",))
    assert base.isdisjoint(
        h1_construction_digest(p) for p in blocks(source_digest=other)[0].block.profiles
    )


def test_the_checkpoint_byte_counts_are_the_measured_ones() -> None:
    # Measured on quote pods s3tkc88i0oyp96 and qkfduyfla3b6rt (FOR-1690, 2026-10-28):
    # the codec warm-up checkpoint and each prediction-phase checkpoint, optimizer included.
    assert CODEC_WARMUP_CHECKPOINT_BYTES == 524_752_197
    assert PREDICTION_CHECKPOINT_BYTES == 1_149_203_227


def test_transfer_limits_follow_the_checkpoint_count_per_cell() -> None:
    schedule = blocks()[0].block.profiles[0].schedule  # steps 1000, interval 250, eval 50
    floor = required_checkpoint_floor(schedule, capture_copies=1)
    # warm-up: one seal at step 250; prediction: multiples of 50 up to 1000 are 20 seals
    assert floor.max_total_bytes == 1 * 524_752_197 + 20 * 1_149_203_227
    assert floor.max_object_bytes == 1_149_203_227
    # 4 objects per seal (journal, checkpoint, seal and progress receipts), 1 per evaluation
    # (20 of them), 1 per generation continuation, plus the 14 fixed worker objects
    assert DEVELOPMENT_GENERATION_CONTINUATIONS == 5
    assert floor.max_objects == 4 * 21 + 20 + 5 + 14
    doubled = required_checkpoint_floor(schedule, capture_copies=2)
    assert doubled.max_total_bytes == 2 * floor.max_total_bytes
    assert doubled.max_objects == 2 * (4 * 21 + 20 + 5) + 14
    sized = pilot_transfer_limits(
        schedule,
        source_bytes=SOURCE_BYTES,
        fixed_objects=FIXED_OBJECTS,
        non_checkpoint_bytes=NON_CHECKPOINT_BYTES,
    )
    assert sized.max_total_bytes == doubled.max_total_bytes + SOURCE_BYTES + NON_CHECKPOINT_BYTES
    assert sized.max_object_bytes == 1_149_203_227
    assert sized.max_objects == doubled.max_objects + FIXED_OBJECTS


def test_the_floor_refuses_a_non_schedule_and_non_positive_capture_copies() -> None:
    schedule = blocks()[0].block.profiles[0].schedule
    with pytest.raises(TypeError):
        required_checkpoint_floor({"steps": 1000})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capture copies"):
        required_checkpoint_floor(schedule, capture_copies=0)


def test_a_source_larger_than_every_checkpoint_sets_the_object_bound() -> None:
    schedule = blocks()[0].block.profiles[0].schedule
    big = 2 * PREDICTION_CHECKPOINT_BYTES
    sized = pilot_transfer_limits(
        schedule, source_bytes=big, fixed_objects=FIXED_OBJECTS, non_checkpoint_bytes=0
    )
    assert sized.max_object_bytes == big


def _schedule(warmup: int, steps: int, interval: int, evaluation: int) -> RunSchedule:
    return RunSchedule(
        warmup_steps=warmup,
        steps=steps,
        batch_size=BATCH_SIZE,
        checkpoint_interval=interval,
        evaluation_interval=evaluation,
        evaluation_batches=EVALUATION_BATCHES,
        importance_samples=DEVELOPMENT_IMPORTANCE_SAMPLES,
        importance_chunk=8,
        generation_sentences=4,
        generation_continuations=DEVELOPMENT_GENERATION_CONTINUATIONS,
        diagnostic_samples=DEVELOPMENT_DIAGNOSTIC_SAMPLES,
    )


@pytest.mark.parametrize("warmup", [1, 2, 7, 250])
@pytest.mark.parametrize("interval", [1, 2, 3, 7, 75, 100, 250])
@pytest.mark.parametrize("evaluation", [1, 2, 3, 5, 50])
def test_seal_counts_equal_the_runner_checkpoint_predicate(
    warmup: int, interval: int, evaluation: int
) -> None:
    """The closed form matches the measured seal predicate, including the off-grid final seal."""
    for steps in (1, 2, 3, 7, 20, 50, 101, 250, 1000):
        seals = checkpoint_steps(
            warmup_steps=warmup,
            steps=steps,
            checkpoint_interval=interval,
            evaluation_interval=evaluation,
        )
        codec = sum(1 for step in seals if step <= warmup)
        prediction = len(seals) - codec
        assert _seal_counts(_schedule(warmup, steps, interval, evaluation)) == (codec, prediction)


def _runner_evaluation_count(steps: int, evaluation: int) -> int:
    """The runner's declared evaluation grid (runner.py): every interval multiple and the last."""
    return sum(1 for step in range(1, steps + 1) if step % evaluation == 0 or step == steps)


@pytest.mark.parametrize("evaluation", [1, 3, 50])
@pytest.mark.parametrize("steps", [1, 7, 101, 1000, 1001])
def test_object_floor_counts_the_runner_evaluations_including_an_off_grid_final(
    steps: int, evaluation: int
) -> None:
    schedule = _schedule(250, steps, 250, evaluation)
    codec, prediction = _seal_counts(schedule)
    expected = (
        2
        * (
            4 * (codec + prediction)
            + _runner_evaluation_count(steps, evaluation)
            + DEVELOPMENT_GENERATION_CONTINUATIONS
        )
        + 14
    )
    assert required_checkpoint_floor(schedule).max_objects == expected


def _with_continuations(profile: HProfile, continuations: int) -> HProfile:
    return replace(
        profile, schedule=replace(profile.schedule, generation_continuations=continuations)
    )


@pytest.mark.parametrize("continuations", [2, 3])
def test_object_floor_matches_a_real_tiny_object_worker_run(
    continuations: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the object model against the objects the real worker publishes for its final snapshot.

    The published set is run, capture, verification and worker outputs, distinct by key, plus the
    snapshot marker. The continuation count is varied so the per-continuation term is measured
    rather than inferred from the fixture's equal sentence and continuation counts.
    """
    base = span_fixture.profile
    monkeypatch.setattr(span_fixture, "profile", lambda: _with_continuations(base(), continuations))
    request, objects, marker, root = _worker(tmp_path)
    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST)
    assert request.profile.schedule.generation_continuations == continuations
    written = {key for key in objects.publishes if "/outputs/" in key}
    assert required_checkpoint_floor(request.profile.schedule).max_objects == len(written) + 1


def test_transfer_limits_are_never_the_probe_kit_constants() -> None:
    kit = TransferLimits(2 * 2**30, 16 * 2**30, 5000)
    # 20 retained 1.07 GiB prediction checkpoints already exceed the kit's 16 GiB total
    assert limits_for().max_total_bytes > kit.max_total_bytes
    with pytest.raises(ValueError, match="transfer"):
        blocks(limits=kit)


@pytest.mark.parametrize("field", ["max_object_bytes", "max_total_bytes", "max_objects"])
def test_limits_one_below_the_floor_are_refused_and_the_floor_is_accepted(field: str) -> None:
    """The floor is a NECESSARY bound: the generator cannot see the corpus or the allowance.

    Acceptance here is not a claim that the limits fit a cell; the real-worker tests below
    measure how far a checkpoint-and-source grant falls short of a final snapshot.
    """
    schedule = blocks()[0].block.profiles[0].schedule
    floor = required_checkpoint_floor(schedule, capture_copies=2)
    assert blocks(limits=floor)  # control: exactly the floor passes
    short = replace(floor, **{field: getattr(floor, field) - 1})
    with pytest.raises(ValueError, match=field):
        blocks(limits=short)


def test_a_refused_limit_names_the_floor_as_necessary_and_not_sufficient() -> None:
    """The refusal says what the floor is, so a pass is never read as a fit."""
    schedule = probe_schedule()
    floor = required_checkpoint_floor(schedule, capture_copies=2)
    short = replace(floor, max_total_bytes=floor.max_total_bytes - 1)
    with pytest.raises(ValueError, match="necessary") as raised:
        blocks(limits=short)
    assert "not sufficient" in str(raised.value)


def _sized(schedule: RunSchedule, *, non_checkpoint_bytes: int) -> TransferLimits:
    return pilot_transfer_limits(
        schedule,
        source_bytes=SOURCE_BYTES,
        fixed_objects=FIXED_OBJECTS,
        non_checkpoint_bytes=non_checkpoint_bytes,
    )


def test_sizing_requires_an_explicit_non_checkpoint_allowance() -> None:
    """No default: the final snapshot also publishes journals, receipts and records, and the
    generator holds no measurement of them, so an omitted allowance cannot be filled in here.
    """
    schedule = probe_schedule()
    with pytest.raises(TypeError, match="non_checkpoint_bytes"):
        pilot_transfer_limits(  # type: ignore[call-arg]
            schedule, source_bytes=SOURCE_BYTES, fixed_objects=FIXED_OBJECTS
        )


@pytest.mark.parametrize("bad", [None, 1.5, "7", True, False, -1])
def test_sizing_refuses_a_non_int_bool_or_negative_non_checkpoint_allowance(bad: object) -> None:
    schedule = probe_schedule()
    with pytest.raises(ValueError, match="non-checkpoint bytes"):
        _sized(schedule, non_checkpoint_bytes=bad)  # type: ignore[arg-type]


def test_the_non_checkpoint_allowance_is_added_to_the_total_exactly_once() -> None:
    schedule = probe_schedule()
    none = _sized(schedule, non_checkpoint_bytes=0)  # zero is a stated value, accepted
    some = _sized(schedule, non_checkpoint_bytes=7)
    many = _sized(schedule, non_checkpoint_bytes=10**9)
    assert some.max_total_bytes - none.max_total_bytes == 7
    assert many.max_total_bytes - none.max_total_bytes == 10**9
    # The allowance is a total only: it moves neither the per-object nor the object count.
    assert (some.max_object_bytes, some.max_objects) == (none.max_object_bytes, none.max_objects)
    assert (many.max_object_bytes, many.max_objects) == (none.max_object_bytes, none.max_objects)


_CHECKPOINT_FILE = re.compile(r"-(codec_warmup|prediction)-\d{8}\.pt$")


@dataclass(frozen=True, slots=True)
class MeasuredFinalSnapshot:
    """What one real tiny object worker published, measured from its own accounting."""

    schedule: RunSchedule
    source_bytes: int
    codec_checkpoint_bytes: int
    prediction_checkpoint_bytes: int
    checkpoint_bytes: int  # every retained checkpoint, run copy and capture copy together
    need_bytes: int  # what publish_worker_snapshot compares with max_total_bytes

    @property
    def non_checkpoint_bytes(self) -> int:
        return self.need_bytes - self.checkpoint_bytes - self.source_bytes


@pytest.fixture(scope="module")
def measured_final_snapshot(tmp_path_factory: pytest.TempPathFactory) -> MeasuredFinalSnapshot:
    """Run the real tiny object worker once and read its final snapshot back.

    The totals are measured at run time and never pinned: result.json and the journals carry
    timing fields, so the need varies by a few dozen bytes between runs (measured 2026-10-30,
    1,181,538 against 1,181,501 on two runs).
    """
    request, objects, marker, root = _worker(tmp_path_factory.mktemp("real-final-snapshot"))
    run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST)
    outputs = f"{request.prefix}/outputs/"
    sizes = {
        key.removeprefix(outputs): len(payload)
        for key, payload in objects.values.items()
        if key.startswith(outputs)
    }
    final = objects.values[f"{request.prefix}/snapshots/complete.json"]
    manifest = json.loads(final)
    assert manifest["artifact_bytes"] == sum(sizes.values())  # the worker's own byte accounting
    assert manifest["artifact_count"] == len(sizes)
    checkpoints = {name: size for name, size in sizes.items() if _CHECKPOINT_FILE.search(name)}
    retained = {name: size for name, size in checkpoints.items() if name.startswith("run/")}
    # CAPTURE_COPIES = 2 checked, not assumed: every run checkpoint has an equal capture twin.
    assert retained and len(checkpoints) == 2 * len(retained)
    for name, size in retained.items():
        assert sizes["capture/" + name.removeprefix("run/")] == size
    per_phase = {
        phase: {size for name, size in checkpoints.items() if f"-{phase}-" in name}
        for phase in ("codec_warmup", "prediction")
    }
    assert all(len(found) == 1 for found in per_phase.values())  # one size per phase
    source = len(objects.values[f"{request.prefix}/inputs/source/prepared-corpus.json"])
    assert sizes["capture/source/prepared-corpus.json"] == source  # the corpus is published once
    return MeasuredFinalSnapshot(
        schedule=request.profile.schedule,
        source_bytes=source,
        codec_checkpoint_bytes=next(iter(per_phase["codec_warmup"])),
        prediction_checkpoint_bytes=next(iter(per_phase["prediction"])),
        checkpoint_bytes=sum(checkpoints.values()),
        need_bytes=manifest["artifact_bytes"] + len(final),
    )


def _grant(
    snapshot: MeasuredFinalSnapshot, patch: pytest.MonkeyPatch, *, non_checkpoint_bytes: int
) -> TransferLimits:
    """The helper applied to the tiny fixture's own regime.

    The helper's checkpoint sizes are the dev-width constants; against a tiny worker they
    would grant gigabytes and prove nothing. The formula is what is under test, so the
    constants are swapped for the sizes this fixture measured, and the corpus is the fixture's.
    """
    patch.setattr(h1_pilot, "CODEC_WARMUP_CHECKPOINT_BYTES", snapshot.codec_checkpoint_bytes)
    patch.setattr(h1_pilot, "PREDICTION_CHECKPOINT_BYTES", snapshot.prediction_checkpoint_bytes)
    return pilot_transfer_limits(
        snapshot.schedule,
        source_bytes=snapshot.source_bytes,
        fixed_objects=0,
        non_checkpoint_bytes=non_checkpoint_bytes,
    )


def test_a_real_workers_final_snapshot_exceeds_the_checkpoint_and_source_grant(
    measured_final_snapshot: MeasuredFinalSnapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-level companion to the object-floor test: the snapshot also carries journals,
    receipts, evaluation and generation records, the profile, span diagnostics, the capture
    manifest, the verification receipt, the worker start and result, and the marker."""
    snapshot = measured_final_snapshot
    grant = _grant(snapshot, monkeypatch, non_checkpoint_bytes=0)
    # The checkpoint-and-source model is exact for what it counts ...
    assert grant.max_total_bytes == snapshot.checkpoint_bytes + snapshot.source_bytes
    # ... and strictly short of the snapshot, by the bytes it does not count.
    excess = snapshot.need_bytes - grant.max_total_bytes
    assert excess == snapshot.non_checkpoint_bytes > 0
    # Passed as the allowance, the measured excess covers the snapshot exactly, and one byte
    # less does not: the allowance enters the total once, at full value.
    covered = _grant(snapshot, monkeypatch, non_checkpoint_bytes=excess)
    assert covered.max_total_bytes == snapshot.need_bytes
    assert _grant(snapshot, monkeypatch, non_checkpoint_bytes=excess - 1).max_total_bytes < (
        snapshot.need_bytes
    )


def test_a_real_worker_refuses_a_cell_sized_without_the_allowance_at_its_final_snapshot(
    measured_final_snapshot: MeasuredFinalSnapshot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The defect end to end, with its control: the measuring run completed under 50 MB, and the
    same request with only its byte total replaced by the helper's grant fails after training.

    Only ``max_total_bytes`` is replaced. With the helper's exact object count the failure
    snapshot trips the object budget first (measured 2026-10-30: 'worker snapshot object budget
    exceeded' as the second error), which is a different bound from the one under test.
    """
    grant = _grant(measured_final_snapshot, monkeypatch, non_checkpoint_bytes=0)
    request, inputs = prepared(tmp_path)
    assert request.limits.max_total_bytes > grant.max_total_bytes  # control: the budget is lower
    sized = replace(request, limits=replace(request.limits, max_total_bytes=grant.max_total_bytes))
    objects = _ObjectStore()
    marker = publish_probe_request(objects, sized, inputs)
    root = tmp_path / "object-worker"
    root.mkdir()
    with pytest.raises(ExceptionGroup) as raised:
        run_object_worker(objects, marker, root, max_request_bytes=MAX_REQUEST)
    # The worker failure and the failure snapshot's delivery fail the same byte check.
    assert [str(error) for error in raised.value.exceptions] == [
        "worker snapshot byte budget exceeded"
    ] * 2
    assert any("/snapshots/checkpoint-" in key for key in objects.values)  # training had run
    assert f"{sized.prefix}/snapshots/complete.json" not in objects.values


def test_limits_must_be_a_transfer_limits() -> None:
    with pytest.raises(TypeError):
        blocks(limits={"max_object_bytes": 2**62})  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, True])
def test_sizing_refuses_non_positive_source_bytes_and_negative_fixed_objects(bad: object) -> None:
    schedule = blocks()[0].block.profiles[0].schedule
    with pytest.raises(ValueError):
        pilot_transfer_limits(
            schedule,
            source_bytes=bad,  # type: ignore[arg-type]
            fixed_objects=1,
            non_checkpoint_bytes=0,
        )
    with pytest.raises(ValueError):
        pilot_transfer_limits(schedule, source_bytes=1, fixed_objects=-1, non_checkpoint_bytes=0)


def test_zero_fixed_objects_is_accepted_and_adds_nothing() -> None:
    schedule = blocks()[0].block.profiles[0].schedule
    floor = required_checkpoint_floor(schedule)
    sized = pilot_transfer_limits(schedule, source_bytes=1, fixed_objects=0, non_checkpoint_bytes=0)
    assert sized.max_objects == floor.max_objects


def test_the_module_adds_no_registry_or_launch_surface() -> None:
    assert not hasattr(h1_pilot, "launch") and not hasattr(h1_pilot, "RUNGS")
