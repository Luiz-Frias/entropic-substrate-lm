"""FOR-1692 (h1-l4a): hand the warmed codec from the donor cell to its pair recipient.

The tracer runs two real ``local_tiny`` H1 profiles that share a codec by construction (same
law mode, different spine carrier) through the real training driver on CPU: the donor trains
its warm-up and prediction; the recipient is MATERIALIZED from the donor's warm-up-final seal
(no warm-up rows of its own) and then continues through ``run_probe_training(resume=...)``;
an independent recipient warms up for itself as the control. Every equality is bitwise
(``torch.equal`` or byte equality); the one named exclusion is the journal timing columns.
"""

import ast
import copy
import json
import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError, asdict, dataclass, fields, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch
from tests.latent_usage_support import POSTERIOR_MEAN
from tests.test_h1_observations import LABELS, build_recipient_block, standard_specs
from tests.test_h_aggregate import IDENTITY_B, REQUEST_B, _drop, write_h_run
from tests.test_sentence_profile import profile
from tests.test_sentence_sampling import corpus

import entropic_substrate.ladder.run.checkpoint as checkpoint_module
import entropic_substrate.ladder.run.codec_handoff as handoff_module
import entropic_substrate.ladder.run.trace_capture as trace_capture_module
from entropic_substrate.core.sentence.corpus import SplitPolicy, prepare_documents
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import capture_rng
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.analyze import h_aggregate
from entropic_substrate.ladder.analyze.h1_observations import observe_h1_blocks
from entropic_substrate.ladder.analyze.h_aggregate import CodecHandoffEvidence, load_h_run
from entropic_substrate.ladder.capture.capture import write_prepared_corpus
from entropic_substrate.ladder.register.h_preservation import H1ExclusionReason
from entropic_substrate.ladder.run.checkpoint import save_checkpoint
from entropic_substrate.ladder.run.codec_handoff import (
    CodecHandoff,
    CodecHandoffDefect,
    VerifiedHandoff,
    admit_codec_pair,
    materialize_codec_handoff,
)
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.runner import (
    TrainingProgress,
    run_probe_training,
    write_progress_receipt,
)
from entropic_substrate.ladder.run.seals import (
    HANDOFF_SEAL_SCHEMA,
    SealedStep,
    SealMemo,
    read_seal,
    resume_seal,
)

pytestmark = pytest.mark.unit
LIMIT = 10_000_000
SOURCE_LIMIT = 1_000_000
# The journal's timing columns are measurements of the host, not of the run (execution.py
# writes them from perf_counter and the CUDA adapters); every other key is compared.
TIMING = frozenset({"wall_seconds", "cuda_seconds", "cuda_peak_bytes"})
CODES = {
    "independent_warmup",
    "foreign_donor",
    "pair_predicate",
    "chained_handoff",
    "construction_divergence",
    "payload_mismatch",
}


# ----------------------------------------------------------------------------- fixtures
@dataclass(frozen=True)
class Pair:
    """Two profiles that share a codec (mm donor, ms recipient) over one prepared corpus."""

    base: Path
    source: Path
    donor: HProfile
    recipient: HProfile

    def root(self, name: str) -> Path:
        path = self.base / name
        path.mkdir()
        return path


def make_pair(base: Path) -> Pair:
    source = base / "source"
    source.mkdir()
    original = corpus()
    prepared = prepare_documents(
        original.candidates,
        replace(original.spec, splits=SplitPolicy(validation_buckets=2000, test_buckets=1000)),
    )
    artifact = write_prepared_corpus(source, prepared, max_bytes=SOURCE_LIMIT)
    template = profile()
    donor = replace(
        template,
        name="h1-pair-mm-tiny",
        stage="H1",
        consumer="isotropic",
        optimization=replace(template.optimization, future_weight=0),
        source_digest=artifact.content_hash.digest,
    )
    return Pair(
        base, source, donor, replace(donor, name="h1-pair-ms-tiny", spine=carrier(donor, "sphere"))
    )


def carrier(value: HProfile, mode: str) -> Any:
    return replace(value.spine, carrier=mode)


def train(pair: Pair, subject: HProfile, root: Path, **options: Any) -> TrainingProgress:
    return run_probe_training(
        subject,
        pair.source,
        root,
        max_source_bytes=SOURCE_LIMIT,
        max_artifact_bytes=LIMIT,
        **options,
        latent_usage=POSTERIOR_MEAN,
    )


def identity_of(path: Path) -> ArtifactIdentity:
    return artifact_from_mapping(
        {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}
    )


def warmup_final(root: Path) -> ArtifactIdentity:
    found = sorted(root.glob("*-codec_warmup-00000002-seal.json"))
    assert len(found) == 1, found
    return identity_of(found[0])


def materialize(pair: Pair, donor_root: Path, recipient_root: Path) -> CodecHandoff:
    return materialize_codec_handoff(
        donor_root,
        warmup_final(donor_root),
        pair.donor,
        recipient_root,
        pair.recipient,
        pair.source,
        max_source_bytes=SOURCE_LIMIT,
        max_artifact_bytes=LIMIT,
    )


def admit(
    pair: Pair, donor_root: Path, recipient_root: Path, recipient_seal: ArtifactIdentity
) -> VerifiedHandoff:
    return admit_codec_pair(
        donor_root,
        warmup_final(donor_root),
        pair.donor,
        recipient_root,
        recipient_seal,
        pair.recipient,
        pair.source,
        max_source_bytes=SOURCE_LIMIT,
        max_artifact_bytes=LIMIT,
    )


def restored(
    pair: Pair, subject: HProfile, root: Path, seal: ArtifactIdentity
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Codec tensors, optimizer state and RNG state exactly as a checkpoint restores them."""
    engine = SentenceExecution(subject, pair.source, max_source_bytes=SOURCE_LIMIT)
    with torch.random.fork_rng(devices=[]):
        resume_seal(root, seal, engine, max_bytes=LIMIT)
        rng = capture_rng()
    codec = {k: v for k, v in engine.model.state_dict().items() if k.startswith("codec.")}
    return codec, engine.optimizer.state_dict(), rng


def same_bits(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_bits(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same_bits(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def prediction_rows(root: Path) -> list[dict[str, Any]]:
    shards = [read_json(root, identity_of(p), max_bytes=LIMIT) for p in root.glob("*-journal.json")]
    ordered = sorted(
        (s for s in shards if s["phase"] == "prediction"), key=lambda s: s["start_step"]
    )
    return [row for shard in ordered for row in shard["rows"]]


def comparable(rows: list[dict[str, Any]]) -> str:
    """Canonical JSON of every key except the named timing exclusions."""
    return json.dumps(
        [{k: v for k, v in row.items() if k not in TIMING} for row in rows], sort_keys=True
    )


def seal_row(root: Path, seal: ArtifactIdentity) -> dict[str, Any]:
    return dict(read_json(root, seal, max_bytes=LIMIT))


def finish(
    pair: Pair, recipient_root: Path, handoff: CodecHandoff, **options: Any
) -> TrainingProgress:
    return train(pair, pair.recipient, recipient_root, resume=handoff.receipt, **options)


# --------------------------------------------------------------------------------- RED
def test_recipient_trains_from_the_donor_warmup_and_matches_an_independent_recipient(
    tmp_path: Path,
) -> None:
    pair = make_pair(tmp_path)
    donor_root, independent_root, recipient_root = (
        pair.root("donor"),
        pair.root("independent"),
        pair.root("recipient"),
    )
    train(pair, pair.donor, donor_root)
    independent = train(pair, pair.recipient, independent_root)
    handoff = materialize(pair, donor_root, recipient_root)
    done = finish(pair, recipient_root, handoff)
    assert done.status == "training_complete" and independent.status == "training_complete"

    # the recipient's first seal is a hand-off seal citing the donor's step-2 seal
    donor_seal = warmup_final(donor_root)
    final = read_seal(
        recipient_root,
        done.last_seal,
        pair.recipient,
        pair.source,
        max_bytes=LIMIT,
        max_source_bytes=SOURCE_LIMIT,
    )
    assert final.phase == "prediction" and final.step == 4 and final.chain_length == 3
    assert final.root_kind == "handoff"
    assert final.seal_hashes[-1] == handoff.seal.content_hash.digest
    row = seal_row(recipient_root, handoff.seal)
    assert row["schema"] == "h-codec-handoff-seal-v1"
    assert (row["phase"], row["step"], row["run_identity"]) == (
        "codec_warmup",
        2,
        pair.recipient.identity,
    )
    record = read_json(recipient_root, artifact_from_mapping(row["journal"]), max_bytes=LIMIT)
    assert record["schema"] == "h-codec-handoff-v1" and record["parent"] is None
    assert record["donor_seal_sha256"] == donor_seal.content_hash.digest
    assert record["donor_identity"] == pair.donor.identity
    assert record["run_identity"] == pair.recipient.identity and record["warmup_steps"] == 2

    # the recipient has no warm-up rows of its own, and the independent control has them
    phases = [
        read_json(recipient_root, identity_of(p), max_bytes=LIMIT)["phase"]
        for p in recipient_root.glob("*-journal.json")
    ]
    assert phases and set(phases) == {"prediction"}
    control = [
        read_json(independent_root, identity_of(p), max_bytes=LIMIT)["phase"]
        for p in independent_root.glob("*-journal.json")
    ]
    assert "codec_warmup" in control

    # hand-off tensors, optimizer and RNG equal the donor's and an independent recipient's
    handed = restored(pair, pair.recipient, recipient_root, handoff.seal)
    donated = restored(pair, pair.donor, donor_root, donor_seal)
    warm = restored(pair, pair.recipient, independent_root, warmup_final(independent_root))
    for name, left, right in zip(("codec", "optimizer", "rng"), handed, donated, strict=True):
        assert same_bits(left, right), f"hand-off {name} differs from the donor's"
    for name, left, right in zip(("codec", "optimizer", "rng"), handed, warm, strict=True):
        assert same_bits(left, right), f"hand-off {name} differs from the independent recipient's"

    # prediction journal rows equal bit for bit
    assert comparable(prediction_rows(recipient_root)) == comparable(
        prediction_rows(independent_root)
    )
    assert len(prediction_rows(recipient_root)) == 4

    # the pair check passes for the hand-off recipient and names an independent one
    verified = admit(pair, donor_root, recipient_root, done.last_seal)
    assert verified.recipient_identity == pair.recipient.identity
    assert verified.donor_identity == pair.donor.identity
    assert verified.warmup_steps == 2
    assert verified.donor_seal_digest == donor_seal.content_hash.digest
    assert verified.record_digest == row["journal"]["sha256"]
    with pytest.raises(CodecHandoffDefect) as raised:
        admit(pair, donor_root, independent_root, independent.last_seal)
    assert raised.value.code == "independent_warmup"
    assert raised.value.code == H1ExclusionReason.INDEPENDENT_WARMUP.value


# ----------------------------------------------------------------------- shared donor runs
@dataclass(frozen=True)
class World:
    """Read-only evidence the defect tests share: two donor runs and an independent control."""

    pair: Pair
    donor: Path
    other_donor: Path
    independent: Path
    independent_seal: ArtifactIdentity
    donor_files: list[str]  # the donor root's listing, taken before any test materializes from it


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    pair = make_pair(tmp_path_factory.mktemp("pair"))
    donor, other, independent = pair.root("donor"), pair.root("other"), pair.root("independent")
    train(pair, pair.donor, donor)
    train(pair, pair.donor, other)
    control = train(pair, pair.recipient, independent)
    return World(pair, donor, other, independent, control.last_seal, listing(donor))


def prediction_seal(root: Path) -> ArtifactIdentity:
    return identity_of(sorted(root.glob("*-prediction-00000004-seal.json"))[0])


def listing(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir())


# ------------------------------------------------------------------------ the defect set
def test_the_defect_set_is_closed_and_independent_warmup_is_the_exclusion_vocabulary() -> None:
    assert CodecHandoffDefect.CODES == CODES
    assert issubclass(CodecHandoffDefect, ValueError)
    for code in CODES:
        assert CodecHandoffDefect(code, "x").code == code
        assert str(CodecHandoffDefect(code, "x")) == f"{code}: x"
    with pytest.raises(ValueError, match="unknown codec hand-off defect code"):
        CodecHandoffDefect("pair_predicates", "x")
    assert CodecHandoffDefect("independent_warmup", "x").code == (
        H1ExclusionReason.INDEPENDENT_WARMUP.value
    )


def test_an_independent_recipient_is_named_by_its_chain_root(world: World) -> None:
    pair = world.pair
    with pytest.raises(CodecHandoffDefect) as raised:
        admit(pair, world.donor, world.independent, world.independent_seal)
    assert raised.value.code == "independent_warmup"
    # its tensors equal the donor's, so tensor comparison could not have said so
    left, right = (
        restored(pair, pair.recipient, world.independent, warmup_final(world.independent)),
        restored(pair, pair.donor, world.donor, warmup_final(world.donor)),
    )
    assert all(same_bits(a, b) for a, b in zip(left, right, strict=True))


def test_a_donor_that_is_not_the_cited_one_is_foreign(world: World) -> None:
    pair = world.pair
    root = pair.root("foreign-a")
    handoff = materialize(pair, world.donor, root)
    with pytest.raises(CodecHandoffDefect) as raised:
        admit(pair, world.other_donor, root, handoff.seal)
    assert raised.value.code == "foreign_donor"


def test_only_the_donors_warmup_final_seal_can_be_materialized(world: World) -> None:
    pair = world.pair
    root = pair.root("foreign-b")
    with pytest.raises(CodecHandoffDefect) as raised:
        materialize_codec_handoff(
            world.donor,
            prediction_seal(world.donor),
            pair.donor,
            root,
            pair.recipient,
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    assert raised.value.code == "foreign_donor" and listing(root) == []


def test_a_later_donor_seal_resolves_to_its_warmup_final_for_the_pair_check(world: World) -> None:
    pair = world.pair
    root = pair.root("later-seal")
    handoff = materialize(pair, world.donor, root)
    verified = admit_codec_pair(
        world.donor,
        prediction_seal(world.donor),
        pair.donor,
        root,
        handoff.seal,
        pair.recipient,
        pair.source,
        max_source_bytes=SOURCE_LIMIT,
        max_artifact_bytes=LIMIT,
    )
    assert verified == handoff.verified


PAIR_BREAKS = {
    "same-carrier": lambda pair: replace(pair.donor, name="h1-pair-twin-tiny"),
    "law-mode": lambda pair: replace(
        pair.recipient, law=replace(pair.recipient.law, mode="sphere")
    ),
    "seed": lambda pair: replace(pair.recipient, seed=43),
    "source-seed": lambda pair: replace(pair.recipient, source_seed=43),
    "schedule": lambda pair: replace(
        pair.recipient, schedule=replace(pair.recipient.schedule, steps=6)
    ),
}


@pytest.mark.parametrize("kind", sorted(PAIR_BREAKS))
def test_a_pair_must_differ_in_the_carrier_alone(world: World, kind: str) -> None:
    pair = world.pair
    root = pair.root(f"pair-{kind}")
    with pytest.raises(CodecHandoffDefect) as raised:
        materialize_codec_handoff(
            world.donor,
            warmup_final(world.donor),
            pair.donor,
            root,
            PAIR_BREAKS[kind](pair),
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    assert raised.value.code == "pair_predicate" and listing(root) == []


def test_a_donor_that_was_itself_handed_off_cannot_donate(world: World) -> None:
    pair = world.pair
    first, second = pair.root("chain-first"), pair.root("chain-second")
    handoff = materialize(pair, world.donor, first)
    with pytest.raises(CodecHandoffDefect) as raised:
        materialize_codec_handoff(
            first,
            handoff.seal,
            pair.recipient,
            second,
            pair.donor,
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    assert raised.value.code == "chained_handoff" and listing(second) == []


# One ulp-scale nudge (1e-7 is below every tensor's allclose tolerance, and above float32
# resolution for values below one), so only a bitwise comparison can see it. ``qk_scale`` sits at
# 2.0, where 1e-7 rounds away, so it is not a probe key.
NUDGE = 1e-7
NON_CODEC_KEYS = (
    "spine.bos",
    "spine.projection.weight",
    "spine.blocks.1.down.weight",
    "head.direction.bias",
    "head.radius.weight",
)


def changed_recipient(change: Callable[[SentenceExecution], None]) -> type[SentenceExecution]:
    """A future construction change that makes a sphere-carrier engine differ from the donor's."""

    class _Changed(SentenceExecution):
        def __init__(self, profile: HProfile, source_root: Path, *, max_source_bytes: int) -> None:
            super().__init__(profile, source_root, max_source_bytes=max_source_bytes)
            if profile.spine.carrier == "sphere":
                with torch.no_grad():
                    change(self)

    return _Changed


def perturbed_recipient(key: str) -> type[SentenceExecution]:
    """One non-codec tensor differs by ``NUDGE``."""

    def nudge(engine: SentenceExecution) -> None:
        engine.model.get_parameter(key).add_(NUDGE)

    return changed_recipient(nudge)


def widen_head(engine: SentenceExecution) -> None:
    """The same values in another dtype: ``torch.equal`` alone cannot see it."""
    head = next(iter(engine.model.head.parameters()))
    head.data = head.data.double()


def add_spine_buffer(engine: SentenceExecution) -> None:
    """An extra non-codec buffer: every shared tensor still matches, the inventories do not."""
    engine.model.spine.register_buffer("extra_probe", torch.zeros(1))


def retune_optimizer(engine: SentenceExecution) -> None:
    """A consistent optimizer policy change: only the construction binding moves."""
    engine.optimizer.defaults["lr"] = 0.123
    for group in engine.optimizer.param_groups:
        group["lr"] = 0.123


CONSTRUCTION_CHANGES = {
    "dtype": widen_head,
    "inventory": add_spine_buffer,
    "binding": retune_optimizer,
}


@pytest.mark.parametrize("kind", sorted(CONSTRUCTION_CHANGES))
def test_each_arm_of_the_construction_guard_refuses_its_own_kind_of_divergence(
    world: World, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    pair = world.pair
    root = pair.root(f"changed-{kind}")
    monkeypatch.setattr(
        handoff_module, "SentenceExecution", changed_recipient(CONSTRUCTION_CHANGES[kind])
    )
    with pytest.raises(CodecHandoffDefect) as raised:
        materialize(pair, world.donor, root)
    assert raised.value.code == "construction_divergence", kind
    assert listing(root) == []


@pytest.mark.parametrize("key", NON_CODEC_KEYS)
def test_a_non_codec_tensor_that_differs_from_the_donors_is_a_construction_divergence(
    world: World, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    pair = world.pair
    root = pair.root(f"divergent-{key}")
    monkeypatch.setattr(handoff_module, "SentenceExecution", perturbed_recipient(key))
    with pytest.raises(CodecHandoffDefect) as raised:
        materialize(pair, world.donor, root)
    assert raised.value.code == "construction_divergence", key
    assert listing(root) == []


@pytest.mark.parametrize("copy_step", ["_adopt_codec", "_adopt_optimizer"])
def test_a_copy_that_does_not_reproduce_the_donor_is_a_payload_mismatch(
    world: World, monkeypatch: pytest.MonkeyPatch, copy_step: str
) -> None:
    pair = world.pair
    root = pair.root(f"mismatch-{copy_step}")
    monkeypatch.setattr(handoff_module, copy_step, lambda recipient, donor: None)
    with pytest.raises(CodecHandoffDefect) as raised:
        materialize(pair, world.donor, root)
    assert raised.value.code == "payload_mismatch" and listing(root) == []


def test_a_failed_proof_leaves_no_progress_receipt_to_resume_from(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = world.pair
    root = pair.root("unproved")

    def refuse(*args: object, **kwargs: object) -> VerifiedHandoff:
        raise CodecHandoffDefect("payload_mismatch", "forced after the seal was written")

    monkeypatch.setattr(handoff_module, "admit_codec_pair", refuse)
    with pytest.raises(CodecHandoffDefect):
        materialize(pair, world.donor, root)
    names = listing(root)
    assert "handoff-record.json" in names and "handoff-codec_warmup-00000002-seal.json" in names
    assert not [n for n in names if n.endswith("-progress.json")]


def test_a_recipient_root_must_start_empty_and_that_is_not_a_defect(world: World) -> None:
    pair = world.pair
    root = pair.root("occupied")
    (root / "stray.json").write_text("{}")
    with pytest.raises(ValueError, match="must be empty") as raised:
        materialize(pair, world.donor, root)
    assert not isinstance(raised.value, CodecHandoffDefect)


def test_materializing_and_admitting_leave_the_callers_rng_alone_and_admit_writes_nothing(
    world: World,
) -> None:
    pair = world.pair
    root = pair.root("neutral")
    before = torch.get_rng_state()
    handoff = materialize(pair, world.donor, root)
    assert torch.equal(torch.get_rng_state(), before)
    written = listing(root)
    admit(pair, world.donor, root, handoff.seal)
    assert torch.equal(torch.get_rng_state(), before) and listing(root) == written
    # neither call writes into the donor's root (the snapshot predates every materialization)
    assert listing(world.donor) == world.donor_files


def test_the_record_is_a_pure_function_of_the_donor_seal_and_the_recipient_profile(
    world: World,
) -> None:
    pair = world.pair
    first, second = pair.root("pure-a"), pair.root("pure-b")
    a, b = materialize(pair, world.donor, first), materialize(pair, world.donor, second)
    assert a.record.content_hash == b.record.content_hash
    assert a.checkpoint.content_hash == b.checkpoint.content_hash
    assert a.seal.content_hash == b.seal.content_hash
    record = read_json(first, a.record, max_bytes=LIMIT)
    assert record["parent"] is None
    donor_refs = {k: v for k, v in record.items() if k.startswith("donor_")}
    # bare digests, never a {"path", "sha256"} mapping the capture closure would follow
    assert all(
        isinstance(v, str) for k, v in donor_refs.items() if k != "donor_manifest"
    ) and "path" not in json.dumps(donor_refs)


def test_the_recipient_carries_the_donors_bytes_not_a_re_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Honest warm-up is carrier independent, so a recipient that re-warmed itself would equal the
    donor and the headline could not tell. A donor whose warm-up was disturbed AFTER each step
    differs from every honest warm-up; the recipient must carry the disturbed bytes."""
    pair = make_pair(tmp_path)
    honest, disturbed, recipient = (
        pair.root("honest"),
        pair.root("disturbed"),
        pair.root("recipient"),
    )
    train(pair, pair.donor, honest)
    original = SentenceExecution.advance

    def disturb(self: SentenceExecution) -> dict[str, object]:
        row = original(self)
        if self.phase == "codec_warmup":
            with torch.no_grad():
                next(self.model.codec.parameters()).add_(1e-3)
        return row

    with monkeypatch.context() as patch:
        patch.setattr(SentenceExecution, "advance", disturb)
        train(pair, pair.donor, disturbed)
    handoff = materialize(pair, disturbed, recipient)
    carried = restored(pair, pair.recipient, recipient, handoff.seal)
    donated = restored(pair, pair.donor, disturbed, warmup_final(disturbed))
    rewarmed = restored(pair, pair.donor, honest, warmup_final(honest))
    assert all(same_bits(a, b) for a, b in zip(carried, donated, strict=True))
    assert not same_bits(carried[0], rewarmed[0])


# ------------------------------------------------------------------- fresh-process resume
def test_a_paused_handoff_recipient_resumes_in_a_fresh_engine_and_equals_an_uninterrupted_run(
    world: World,
) -> None:
    """Pause at prediction step 2, resume, and equal an uninterrupted run, bit for bit.

    "Fresh process" is read as this repo reads it (tests/test_sentence_runner.py): an in-process
    call that builds a NEW ``SentenceExecution`` and walks the seals in full with no memo, and
    carries no object over from the first leg. It is not an operating-system process boundary.
    """
    pair = world.pair
    whole, split = pair.root("whole"), pair.root("split")
    done = finish(pair, whole, materialize(pair, world.donor, whole))
    paused = finish(
        pair,
        split,
        materialize(pair, world.donor, split),
        stop_after_seal=lambda phase, step: phase == "prediction" and step == 2,
    )
    assert paused.status == "paused" and len(prediction_rows(split)) == 2
    # a new engine and the memo-less full walk: no Python object survives from the first leg
    resumed = train(pair, pair.recipient, split, resume=paused.receipt)
    assert resumed.status == "training_complete" and done.status == "training_complete"
    assert comparable(prediction_rows(split)) == comparable(prediction_rows(whole))
    final = restored(pair, pair.recipient, split, resumed.last_seal)
    assert all(
        same_bits(a, b)
        for a, b in zip(final, restored(pair, pair.recipient, whole, done.last_seal), strict=True)
    )
    assert (
        read_seal(
            split,
            resumed.last_seal,
            pair.recipient,
            pair.source,
            max_bytes=LIMIT,
            max_source_bytes=SOURCE_LIMIT,
        ).root_kind
        == "handoff"
    )


def test_the_row_comparison_excludes_only_the_named_timing_keys() -> None:
    assert TIMING == {"wall_seconds", "cuda_seconds", "cuda_peak_bytes"}
    row = {
        "step": 1,
        "decoder_ce": 1.5,
        "wall_seconds": 1.0,
        "cuda_seconds": None,
        "cuda_peak_bytes": None,
    }
    assert comparable([row]) == comparable([{**row, "wall_seconds": 9.0, "cuda_seconds": 2.0}])
    for key in ("step", "decoder_ce"):
        assert comparable([row]) != comparable([{**row, key: 2}])
    assert comparable([row]) != comparable([{**row, "extra": 1}])


# --------------------------------------------------------------------------- the seal walk
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"step": 1}, "complete codec warm-up"),
        ({"phase": "prediction"}, "complete codec warm-up"),
        ({"run_identity": "0" * 64}, "seal construction, phase, or step is invalid"),
        ({"schema": "h-codec-handoff-seal-v2"}, "seal construction, phase, or step is invalid"),
    ],
)
def test_a_forged_handoff_seal_is_refused(
    world: World, change: dict[str, object], message: str
) -> None:
    pair = world.pair
    root = pair.root(f"forged-{len(list(pair.base.iterdir()))}")
    handoff = materialize(pair, world.donor, root)
    forged_row = {**seal_row(root, handoff.seal), **change}
    (root / "forged.json").write_text(json.dumps(forged_row))
    with pytest.raises(ValueError, match=message):
        read_seal(
            root,
            identity_of(root / "forged.json"),
            pair.recipient,
            pair.source,
            max_bytes=LIMIT,
            max_source_bytes=SOURCE_LIMIT,
        )


def test_a_handoff_chain_is_walked_with_and_without_a_memo(world: World) -> None:
    pair = world.pair
    root = pair.root("memo")
    handoff = materialize(pair, world.donor, root)
    done = finish(pair, root, handoff)
    kwargs = {"max_bytes": LIMIT, "max_source_bytes": SOURCE_LIMIT}
    memo = SealMemo()
    walked = read_seal(root, done.last_seal, pair.recipient, pair.source, memo=memo, **kwargs)
    assert (memo.seals_walked, memo.journals_read) == (3, 3) and walked.root_kind == "handoff"
    again = read_seal(root, done.last_seal, pair.recipient, pair.source, memo=memo, **kwargs)
    assert again == walked and (memo.seals_walked, memo.journals_read) == (3, 3)
    assert read_seal(root, done.last_seal, pair.recipient, pair.source, **kwargs) == walked
    # a hand-off root alone is an ordinary-length-one chain that a fresh memo also serves
    only = read_seal(root, handoff.seal, pair.recipient, pair.source, memo=SealMemo(), **kwargs)
    assert (only.phase, only.step, only.chain_length, only.root_kind) == (
        "codec_warmup",
        2,
        1,
        "handoff",
    )


# --------------------------------------------------- forged roots: hashes valid, re-sealed
def forged_root(
    world: World,
    name: str,
    *,
    record: dict[str, object] | None = None,
    nudge: Any = None,
    claim_own: bool = False,
    checkpoint_from: Literal["recipient", "donor", "fresh"] = "recipient",
    measurement: ContentHash | None = None,
) -> tuple[Path, ArtifactIdentity]:
    """A recipient root that is self-consistent byte for byte (every hash recomputed, the seal
    re-written over the forged record and checkpoint) and differs from an honest hand-off only
    in the one thing under test: ``record`` keys overwrite the record's, ``nudge(engine)``
    disturbs the restored engine before its checkpoint is saved again, and ``claim_own`` makes the
    record claim the disturbed engine's digests instead of the donor's. ``checkpoint_from`` names
    the engine the checkpoint is saved from (the restored recipient, the donor restored at its
    warm-up-final, or a fresh recipient that never trained, at its cursor 0), and
    ``measurement`` replaces the forged record's digest as the checkpoint's measurement."""
    pair = world.pair
    root = pair.root(f"forged-{name}")
    honest = materialize(pair, world.donor, root)
    row = seal_row(root, honest.seal)
    forged = dict(read_json(root, artifact_from_mapping(row["journal"]), max_bytes=LIMIT))
    forged.update(record or {})
    subject = pair.donor if checkpoint_from == "donor" else pair.recipient
    engine = SentenceExecution(subject, pair.source, max_source_bytes=SOURCE_LIMIT)
    with torch.random.fork_rng(devices=[]):
        if checkpoint_from == "recipient":
            resume_seal(root, honest.seal, engine, max_bytes=LIMIT)
        elif checkpoint_from == "donor":
            resume_seal(world.donor, warmup_final(world.donor), engine, max_bytes=LIMIT)
        if nudge is not None:
            nudge(engine)
        if claim_own:
            forged["codec_sha256"] = handoff_module.codec_digest(engine)
            forged["optimizer_sha256"] = handoff_module.optimizer_digest(engine)
            forged["rng_sha256"] = handoff_module.rng_digest(capture_rng())
        journal = write_json(root, f"forged-{name}-record.json", forged, max_bytes=LIMIT)
        checkpoint = save_checkpoint(
            root,
            f"forged-{name}.pt",
            engine.model,
            engine.optimizer,
            engine.sampler,
            phase="codec_warmup",
            step=engine.sampler.cursor,
            run_identity=pair.recipient.identity,
            measurement=journal.content_hash if measurement is None else measurement,
            max_bytes=LIMIT,
        )
    seal = write_json(
        root,
        f"forged-{name}-seal.json",
        {
            "schema": HANDOFF_SEAL_SCHEMA,
            "run_identity": pair.recipient.identity,
            "phase": "codec_warmup",
            "step": 2,
            "checkpoint": artifact_reference(root, checkpoint),
            "journal": artifact_reference(root, journal),
        },
        max_bytes=LIMIT,
    )
    return root, artifact_from_mapping(artifact_reference(root, seal))


def nudge_codec(engine: SentenceExecution) -> None:
    with torch.no_grad():
        next(engine.model.codec.parameters()).add_(NUDGE)


def nudge_decoder(engine: SentenceExecution) -> None:
    """A decoder tensor, the half of ``codec.*`` that ``nudge_codec`` (an encoder tensor) misses."""
    with torch.no_grad():
        next(engine.model.codec.decoder.parameters()).add_(NUDGE)


def nudge_moment(engine: SentenceExecution, moment: str) -> None:
    for state in engine.optimizer.state.values():
        for key, value in state.items():
            if key == moment and isinstance(value, torch.Tensor):
                value.add_(NUDGE)
                return
    raise AssertionError(f"the restored optimizer carries no {moment} state to disturb")


def nudge_optimizer(engine: SentenceExecution) -> None:
    nudge_moment(engine, "exp_avg")


def nudge_second_moment(engine: SentenceExecution) -> None:
    """``_optimizer_payload`` checks ``exp_avg_sq`` only for shape, dtype and sign, so the digest
    is the sole guard on its values."""
    nudge_moment(engine, "exp_avg_sq")


def nudge_rng(engine: SentenceExecution) -> None:
    torch.rand(1)


def nudge_spine(engine: SentenceExecution) -> None:
    with torch.no_grad():
        engine.model.spine.bos.add_(1.0)


ADMIT_HAZARDS: dict[str, tuple[dict[str, object] | str | None, Any, str]] = {
    "codec": (None, nudge_codec, "payload_mismatch"),
    "decoder": (None, nudge_decoder, "payload_mismatch"),
    "optimizer": (None, nudge_optimizer, "payload_mismatch"),
    "second-moment": (None, nudge_second_moment, "payload_mismatch"),
    "rng": (None, nudge_rng, "payload_mismatch"),
    "spine": (None, nudge_spine, "construction_divergence"),
    "donor-identity": ({"donor_identity": "1" * 64}, None, "foreign_donor"),
    "donor-manifest": ("manifest", None, "foreign_donor"),
    "donor-seal": ({"donor_seal_sha256": "2" * 64}, None, "foreign_donor"),
    "donor-checkpoint": ({"donor_checkpoint_sha256": "3" * 64}, None, "foreign_donor"),
}


@pytest.mark.parametrize("kind", sorted(ADMIT_HAZARDS))
def test_a_forged_recipient_root_is_refused_by_the_pair_check(world: World, kind: str) -> None:
    """Each refusal in ``admit_codec_pair`` is load-bearing: the forged root is internally valid
    (so ``read_seal`` admits it) and only the donor-side proof can name the lie."""
    pair = world.pair
    overrides, nudge, code = ADMIT_HAZARDS[kind]
    if overrides == "manifest":
        overrides = {"donor_manifest": {**pair.donor.manifest, "name": "someone-else"}}
    assert overrides is None or isinstance(overrides, dict)
    root, seal = forged_root(world, kind, record=overrides, nudge=nudge)
    read_seal(
        root, seal, pair.recipient, pair.source, max_bytes=LIMIT, max_source_bytes=SOURCE_LIMIT
    )
    with pytest.raises(CodecHandoffDefect) as raised:
        admit(pair, world.donor, root, seal)
    assert raised.value.code == code, kind


def test_the_progress_receipt_carries_the_profiles_own_run_class(
    world: World, tmp_path: Path
) -> None:
    """Every fixture profile is ``local_tiny``, so a receipt that hard-coded that class would pass
    every other test; the one receipt definition must read it off the profile it is given."""
    remote = replace(world.pair.recipient, run_class="remote_tiny", device="cuda", precision="bf16")
    assert remote.run_class != world.pair.recipient.run_class
    stub = write_json(tmp_path, "stub.json", {"stub": 1}, max_bytes=LIMIT)
    receipt = write_progress_receipt(
        tmp_path,
        "receipt.json",
        remote,
        stub,
        status="checkpoint",
        phase="codec_warmup",
        step=2,
        last_seal=stub,
        evaluations=[],
        parent_receipt=None,
        inventory={},
        max_bytes=LIMIT,
    )
    assert read_json(tmp_path, receipt, max_bytes=LIMIT)["run_class"] == "remote_tiny"


def test_a_recipient_that_warmed_differently_and_claims_its_own_digests_is_refused(
    world: World,
) -> None:
    """The central forgery: a different warm-up that cites the donor's seal and signs its own
    payload. The recipient equals its claim, so only the donor side of the proof can say no."""
    pair = world.pair
    root, seal = forged_root(world, "own-claim", nudge=nudge_codec, claim_own=True)
    read_seal(
        root, seal, pair.recipient, pair.source, max_bytes=LIMIT, max_source_bytes=SOURCE_LIMIT
    )
    with pytest.raises(CodecHandoffDefect) as raised:
        admit(pair, world.donor, root, seal)
    assert raised.value.code == "payload_mismatch" and "donor" in raised.value.detail


RECORD_FORGERIES: dict[str, dict[str, object]] = {
    "schema": {"schema": "h-codec-handoff-v2"},
    "run-identity": {"run_identity": "0" * 64},
    "parent": {"parent": {"path": "x.json", "sha256": "0" * 64}},
    "warmup-length": {"warmup_steps": 3},
    "warmup-type": {"warmup_steps": 2.0},  # equal to the profile's 2, not an int
    "manifest-missing": {"donor_manifest": None},
    # not None and not a mapping: only the type arm of the manifest check can refuse it
    "manifest-not-a-mapping": {"donor_manifest": ["h1-pair-mm-tiny"]},
}
# Every key the seal walk validates as a digest is pinned on its own: dropping one from the
# walk's tuple must not stay green because the pair check's later equality would still refuse
# a wrong value (that is a different, later guard).
for _key in (
    "donor_identity",
    "donor_seal_sha256",
    "donor_checkpoint_sha256",
    "codec_sha256",
    "optimizer_sha256",
    "rng_sha256",
):
    RECORD_FORGERIES[f"digest-not-hex-{_key}"] = {_key: "zz"}
    RECORD_FORGERIES[f"digest-not-string-{_key}"] = {_key: 5}


@pytest.mark.parametrize("kind", sorted(RECORD_FORGERIES))
def test_a_forged_handoff_record_is_refused_by_the_seal_walk(world: World, kind: str) -> None:
    pair = world.pair
    root, seal = forged_root(world, f"record-{kind}", record=RECORD_FORGERIES[kind])
    with pytest.raises(ValueError) as raised:
        read_seal(
            root, seal, pair.recipient, pair.source, max_bytes=LIMIT, max_source_bytes=SOURCE_LIMIT
        )
    assert not isinstance(raised.value, CodecHandoffDefect), kind


def read_forged(world: World, root: Path, seal: ArtifactIdentity) -> SealedStep:
    pair = world.pair
    return read_seal(
        root, seal, pair.recipient, pair.source, max_bytes=LIMIT, max_source_bytes=SOURCE_LIMIT
    )


def test_a_handoff_checkpoint_bound_to_a_foreign_measurement_is_refused_by_the_seal_walk(
    world: World,
) -> None:
    """The hand-off node runs the walk's per-node checkpoint checks, and each must be its own
    tooth there: a hand-off seal whose checkpoint names another measurement than its record (here
    the donor's own warm-up journal) is refused by the walk, never first by a later restore."""
    foreign = ContentHash(
        "sha256", seal_row(world.donor, warmup_final(world.donor))["journal"]["sha256"]
    )
    root, seal = forged_root(world, "foreign-measurement", measurement=foreign)
    assert artifact_from_mapping(seal_row(root, seal)["journal"]).content_hash != foreign
    with pytest.raises(ValueError, match="checkpoint measurement does not belong to this journal"):
        read_forged(world, root, seal)


def test_a_handoff_checkpoint_saved_from_the_donors_engine_is_refused_by_the_seal_walk(
    world: World,
) -> None:
    """The donor's warmed engine equals the recipient's construction in every tensor and differs
    in its binding by the spine carrier alone, so only the walk's construction check sees it."""
    root, seal = forged_root(world, "donor-construction", checkpoint_from="donor")
    with pytest.raises(ValueError, match="checkpoint construction differs from sealed profile"):
        read_forged(world, root, seal)


def test_a_handoff_checkpoint_at_another_step_is_refused_by_the_seal_walk(world: World) -> None:
    """A fresh recipient checkpoint at step 0 under a step-2 hand-off seal. The public writer
    keeps a checkpoint's step equal to its source cursor, so the walk's step check and its
    source-cursor check refuse it redundantly: this pins the PAIR (either message), since
    exempting either one alone for a hand-off node leaves the other to refuse."""
    root, seal = forged_root(world, "fresh-step", checkpoint_from="fresh")
    with pytest.raises(
        ValueError,
        match=r"checkpoint and seal steps disagree|checkpoint source cursor differs from journal",
    ):
        read_forged(world, root, seal)


def test_an_rng_restore_that_does_nothing_is_refused_before_the_first_write(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The donor's RNG state is restored LAST and then checked. A restore that silently did
    nothing is refused there, with the recipient root still empty; the later pair check would
    refuse it too, but only after profile, record, checkpoint and seal were written."""
    pair = world.pair
    root = pair.root("rng-no-op")
    monkeypatch.setattr(handoff_module, "restore_rng", lambda payload: None)
    with pytest.raises(CodecHandoffDefect) as raised:
        materialize(pair, world.donor, root)
    assert raised.value.code == "payload_mismatch" and listing(root) == []


def test_the_rng_digest_follows_the_generator_state() -> None:
    first = handoff_module.rng_digest(capture_rng())
    assert handoff_module.rng_digest(capture_rng()) == first
    torch.rand(1)
    assert handoff_module.rng_digest(capture_rng()) != first


def test_nothing_between_the_construction_guard_and_the_final_restore_draws_a_number(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final RNG restore overwrites a draw made before it, so the end state cannot show one:
    the generator's state at the guard's exit must equal its state at the restore's entry."""
    seen: dict[str, torch.Tensor] = {}
    guard, restore = handoff_module.guard_construction, handoff_module.restore_rng

    def after_guard(donor: SentenceExecution, recipient: SentenceExecution) -> None:
        guard(donor, recipient)
        seen.setdefault("guard", torch.get_rng_state())  # the first call is the materializer's

    def at_restore(payload: Any) -> None:
        seen.setdefault("restore", torch.get_rng_state())
        restore(payload)

    monkeypatch.setattr(handoff_module, "guard_construction", after_guard)
    monkeypatch.setattr(handoff_module, "restore_rng", at_restore)
    materialize(world.pair, world.donor, world.pair.root("draw-free"))
    assert set(seen) == {"guard", "restore"}
    assert torch.equal(seen["guard"], seen["restore"]), "the copy window drew a random number"


# --------------------------------------------------------------------- the capture closure
def test_the_capture_closure_admits_the_handoff_names_and_follows_no_donor_pointer(
    world: World,
) -> None:
    pair = world.pair
    root = pair.root("closure")
    done = finish(pair, root, materialize(pair, world.donor, root))
    schemas = trace_capture_module._SCHEMAS  # pyright: ignore[reportPrivateUsage]
    assert {"h-codec-handoff-v1", "h-codec-handoff-seal-v1", "h-checkpoint-seal-v1"} <= schemas
    collected, _ = trace_capture_module._collect(  # pyright: ignore[reportPrivateUsage]
        root, (done.receipt,), pair.recipient, max_bytes=LIMIT, max_artifacts=100
    )
    assert {"handoff-record.json", "handoff-codec_warmup-00000002-seal.json"} <= set(collected)
    assert {p.name for p in root.iterdir()} >= set(collected)


# The vocabulary the capture closure admitted before FOR-1692, listed here so that "no schema id
# moves except the two new names" is an equality, not a subset.
PUBLISHED_SCHEMAS = frozenset(
    {
        "h-run-profile-v3",
        "h-probe-training-progress-v1",
        "h-checkpoint-seal-v1",
        "h-journal-shard-v1",
        "h-journal-shard-v2",
        "h-predictive-evaluation-v1",
        "h-generation-receipt-v1",
        "h-generation-receipt-v2",
        "h-generation-receipt-v3",
        "h-probe-training-failure-v1",
        "h-probe-training-failure-v2",
        "h-generation-failure-v1",
        "h-span-diagnostics-v1",
        "h-span-diagnostics-failure-v1",
    }
)


def test_the_capture_vocabulary_gains_exactly_the_two_handoff_names() -> None:
    """FOR-1693 amendment (house rule 4): progress v2 joins the two hand-off names. The v1
    progress name stays in the vocabulary only so ``_collect`` can refuse it BY NAME."""
    schemas = trace_capture_module._SCHEMAS  # pyright: ignore[reportPrivateUsage]
    # FOR-1704: the evaluation record gains its v2 name; v1 stays so the H0 captures replay
    assert schemas == PUBLISHED_SCHEMAS | {
        "h-codec-handoff-v1",
        "h-codec-handoff-seal-v1",
        "h-probe-training-progress-v2",
        "h-predictive-evaluation-v2",
    }


# ------------------------------------------------------------------- the analysis seam
def test_the_verified_result_is_the_evidence_the_analysis_reader_takes(
    world: World, tmp_path: Path
) -> None:
    assert {f.name for f in fields(VerifiedHandoff)} == {
        f.name for f in fields(CodecHandoffEvidence)
    }
    pair = world.pair
    root = pair.root("evidence")
    verified = materialize(pair, world.donor, root).verified
    evidence = CodecHandoffEvidence(**asdict(verified))
    paths = write_h_run(
        tmp_path,
        verified.recipient_identity,
        REQUEST_B,
        mutate_metrics=_drop(lambda r: r.get("phase") == "codec_warmup"),
    )
    run = load_h_run(*paths, handoff=evidence)
    assert run.handoff == evidence and run.warmup_steps == verified.warmup_steps
    assert IDENTITY_B != verified.recipient_identity


def test_the_defect_code_is_the_adapters_exclusion_reason(world: World, tmp_path: Path) -> None:
    pair = world.pair
    with pytest.raises(CodecHandoffDefect) as raised:
        admit(pair, world.donor, world.independent, world.independent_seal)
    reason = H1ExclusionReason(raised.value.code)
    capture = build_recipient_block(tmp_path, 101, standard_specs(0))
    flagged = replace(capture, independent_warmup=(LABELS[2],))
    (observation,) = observe_h1_blocks([flagged])
    assert [(f.reason, f.cell) for f in observation.findings] == [(reason, LABELS[2])]
    (clean,) = observe_h1_blocks([capture])
    assert clean.findings == ()


# ------------------------------------------------------------------------ layering (AST)
SRC = Path(handoff_module.__file__).parent


def _tree(name: str) -> ast.Module:
    return ast.parse((SRC / name).read_text(encoding="utf-8"))


def _strings(tree: ast.Module) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def _imported(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
            names.extend(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    return names


def test_codec_handoff_never_reaches_registration_or_analysis_by_import_or_string() -> None:
    tree = _tree("codec_handoff.py")
    forbidden = ("entropic_substrate.ladder.register", "entropic_substrate.ladder.analyze")
    assert not [n for n in _imported(tree) if n.startswith(forbidden)]
    # a string that is a bare dotted name is what an importlib shim would pass; prose has spaces
    names = [s for s in _strings(tree) if re.fullmatch(r"[A-Za-z0-9_.]+", s)]
    assert not [s for s in names if s.startswith(forbidden) or "h_preservation" in s]
    assert not [
        n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "h_preservation"
    ]
    for node in ast.walk(tree):  # no deferred import and no importlib shim breaks the layering
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert node in tree.body, "imports are module level only"
    called = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert not called & {"import_module", "__import__"}
    # a relative import resolves to the same module without a dotted name the checks above see
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level > 0]
    # an alias (``import_module as _im``) hides the call, so the dynamic-import modules are out
    assert not [n for n in _imported(tree) if n.split(".")[0] in {"importlib", "runpy"}]


def test_the_edge_between_seals_and_the_handoff_member_is_one_way() -> None:
    for name in ("seals.py", "runner.py", "checkpoint.py", "trace_capture.py", "execution.py"):
        assert not [n for n in _imported(_tree(name)) if "codec_handoff" in n], name
    analysis = ast.parse(Path(h_aggregate.__file__).read_text(encoding="utf-8"))
    assert not [n for n in _imported(analysis) if "codec_handoff" in n]


# --------------------------------------------------- guards of admit_codec_pair itself
def test_admit_refuses_a_recipient_that_is_not_the_donors_pair(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``admit_codec_pair`` runs the pair predicate on its own. The twin shares the donor's
    carrier, so no honest materialize can produce it: the root is built with the materializer's
    predicate switched off, then admitted with the real one."""
    pair = world.pair
    twin = replace(pair.donor, name="h1-pair-twin-tiny")
    root = pair.root("twin")
    with monkeypatch.context() as patch:
        patch.setattr(handoff_module, "require_pair", lambda donor, recipient: None)
        built = materialize_codec_handoff(
            world.donor,
            warmup_final(world.donor),
            pair.donor,
            root,
            twin,
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    with pytest.raises(CodecHandoffDefect) as raised:
        admit_codec_pair(
            world.donor,
            warmup_final(world.donor),
            pair.donor,
            root,
            built.seal,
            twin,
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    assert raised.value.code == "pair_predicate"


def test_admit_asserts_that_the_walked_root_is_the_handoff_seal(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw walk (``codec_handoff._walk``) and the verified walk (``seals``) are two
    traversals of one chain; the reader asserts they agree on its root (filed debt R4)."""
    pair = world.pair
    root = pair.root("root-shape")
    done = finish(pair, root, materialize(pair, world.donor, root))
    walk = handoff_module._walk  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(handoff_module, "_walk", lambda *a, **k: list(reversed(walk(*a, **k))))
    with pytest.raises(ValueError, match="recipient chain root is not its hand-off seal"):
        admit(pair, world.donor, root, done.last_seal)


def test_the_sealed_step_default_root_kind_is_initial_for_every_older_constructor() -> None:
    """``root_kind`` trails with a default so constructors written before the hand-off
    (tests/test_final_leg_admission.py, tests/test_h_stage_metrics_bridge.py) are unchanged."""
    identity = artifact_from_mapping({"path": "x.json", "sha256": "a" * 64})
    assert SealedStep("prediction", 1, identity, identity, 1).root_kind == "initial"
    assert SealedStep("prediction", 1, identity, identity, 1, ("a" * 64,)).root_kind == "initial"


# ------------------------------------------- a donor whose warm-up spans two checkpoints
@dataclass(frozen=True)
class LongWorld:
    """Warm-up 4 at checkpoint interval 2: the donor's chain holds a PARTIAL warm-up seal at step
    2 below its warm-up-final at step 4 (the tracer pair, warm-up 2, has none), and a prediction
    seal at step 4, the warm-up length in the wrong phase."""

    pair: Pair
    donor: Path
    recipient: Path
    recipient_seal: ArtifactIdentity


def seal_at(root: Path, phase: str, step: int) -> ArtifactIdentity:
    found = sorted(root.glob(f"*-{phase}-{step:08d}-seal.json"))
    assert len(found) == 1, found
    return identity_of(found[0])


@pytest.fixture(scope="module")
def long_world(tmp_path_factory: pytest.TempPathFactory) -> LongWorld:
    short = make_pair(tmp_path_factory.mktemp("long"))
    donor = replace(short.donor, schedule=replace(short.donor.schedule, warmup_steps=4))
    pair = replace(
        short,
        donor=donor,
        recipient=replace(donor, name=short.recipient.name, spine=carrier(donor, "sphere")),
    )
    donor_root, recipient_root = pair.root("donor"), pair.root("recipient")
    train(pair, pair.donor, donor_root)
    handoff = materialize_codec_handoff(
        donor_root,
        seal_at(donor_root, "codec_warmup", 4),
        pair.donor,
        recipient_root,
        pair.recipient,
        pair.source,
        max_source_bytes=SOURCE_LIMIT,
        max_artifact_bytes=LIMIT,
    )
    assert handoff.verified.warmup_steps == 4
    return LongWorld(pair, donor_root, recipient_root, handoff.seal)


@pytest.mark.parametrize(("phase", "step"), [("codec_warmup", 2), ("prediction", 4)])
def test_materialize_refuses_a_donor_seal_that_is_half_a_warmup_final(
    long_world: LongWorld, phase: str, step: int
) -> None:
    """A partial warm-up seal (right phase, short step) and a prediction seal at the warm-up length
    (right step, wrong phase) each satisfy ONE half of the warm-up-final predicate; both are
    foreign before the first write."""
    pair = long_world.pair
    root = pair.root(f"half-{phase}-{step}")
    with pytest.raises(CodecHandoffDefect) as raised:
        materialize_codec_handoff(
            long_world.donor,
            seal_at(long_world.donor, phase, step),
            pair.donor,
            root,
            pair.recipient,
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    assert raised.value.code == "foreign_donor" and listing(root) == []


def test_admit_names_a_donor_seal_below_its_warmup_final_foreign(long_world: LongWorld) -> None:
    """No ancestor of a partial warm-up seal is the warm-up-final, so the pair check refuses it
    as foreign; it never falls through to an empty, untyped walk."""
    pair = long_world.pair
    with pytest.raises(CodecHandoffDefect) as raised:
        admit_codec_pair(
            long_world.donor,
            seal_at(long_world.donor, "codec_warmup", 2),
            pair.donor,
            long_world.recipient,
            long_world.recipient_seal,
            pair.recipient,
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    assert raised.value.code == "foreign_donor"


# ------------------------------------------------------------- small values and members
def test_the_evidence_admits_a_one_step_inherited_warmup_and_nothing_shorter() -> None:
    def evidence(warmup: int) -> CodecHandoffEvidence:
        return CodecHandoffEvidence("a" * 64, "b" * 64, warmup, "c" * 64, "d" * 64)

    assert evidence(1).warmup_steps == 1
    with pytest.raises(ValueError, match="positive inherited warm-up length"):
        evidence(0)


def test_the_handoff_member_is_bound_into_the_checkpoint_reader_identity() -> None:
    """Door D6: the member that reads and writes hand-off checkpoints is one of the modules whose
    bytes the checkpoint reader identity binds."""
    modules = checkpoint_module._READER_MODULES  # pyright: ignore[reportPrivateUsage]
    assert "ladder.run.codec_handoff" in modules


def test_the_handoff_values_are_frozen_and_the_evidence_has_slots() -> None:
    ref = artifact_from_mapping({"path": "x.json", "sha256": "a" * 64})
    verified = VerifiedHandoff("a" * 64, "b" * 64, 2, "c" * 64, "d" * 64)
    evidence = CodecHandoffEvidence(**asdict(verified))
    sealed = SealedStep("prediction", 1, ref, ref, 1)
    for value in (verified, CodecHandoff(ref, ref, ref, ref, verified), evidence, sealed):
        with pytest.raises(FrozenInstanceError):
            setattr(value, fields(value)[0].name, None)
    assert not hasattr(evidence, "__dict__")


def test_the_digest_stream_frames_every_field_by_length_and_types_every_tensor() -> None:
    stream = handoff_module._Stream()  # pyright: ignore[reportPrivateUsage]
    stream.feed("t", b"data")
    framed = (1).to_bytes(4, "little") + b"t" + (4).to_bytes(8, "little") + b"data"
    assert stream.hexdigest() == sha256(framed).hexdigest()
    zeros, digests = torch.zeros(4), set[str]()
    for value in (zeros, zeros.view(torch.int32), zeros.reshape(2, 2)):  # same bytes, three heads
        stream = handoff_module._Stream()  # pyright: ignore[reportPrivateUsage]
        stream.value("x", value)
        digests.add(stream.hexdigest())
    assert len(digests) == 3


# ======================================================================================
# FOR-1713 (h1 hardening, tests only): the residual guards the L4a adversarial sweep left
# without a tooth. Every residual ends in ONE disposition, recorded in ``FOR_1713`` at the
# foot of this file and checked by ``test_every_for_1713_residual_has_exactly_one_disposition``:
# a pin (a test here goes RED under the named flip, the source restored byte-identical), or a
# one-line reason (redundant, unreachable, measurement-gated, superseded by the FOR-1712
# fold). No member source changed; the flips were measured in the lane's own worktree.
# ======================================================================================
def defect_of(call: Callable[[], object]) -> CodecHandoffDefect:
    with pytest.raises(CodecHandoffDefect) as raised:
        call()
    return raised.value


def materialize_refused(
    world: World,
    name: str,
    *,
    recipient: HProfile | None = None,
    seal: ArtifactIdentity | None = None,
) -> CodecHandoffDefect:
    """The defect ``materialize_codec_handoff`` raises, with the recipient root still empty."""
    pair = world.pair
    root = pair.root(name)
    defect = defect_of(
        lambda: materialize_codec_handoff(
            world.donor,
            warmup_final(world.donor) if seal is None else seal,
            pair.donor,
            root,
            pair.recipient if recipient is None else recipient,
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    )
    assert listing(root) == []
    return defect


# ---------------------------------------------------- the detail text of every typed refusal
SAME_CARRIER = "a pair needs two different spine carriers"
OTHER_MANIFEST = "donor and recipient manifests differ beyond their name and spine carrier"


@pytest.mark.parametrize("kind", sorted(PAIR_BREAKS))
def test_each_refusal_names_why_in_its_detail_for_the_pair_predicate(
    world: World, kind: str
) -> None:
    pair = world.pair
    defect = materialize_refused(world, f"detail-pair-{kind}", recipient=PAIR_BREAKS[kind](pair))
    expected = SAME_CARRIER if kind == "same-carrier" else OTHER_MANIFEST
    assert (defect.code, defect.detail) == ("pair_predicate", expected), kind


def test_each_refusal_names_why_in_its_detail_for_a_chained_donor(world: World) -> None:
    pair = world.pair
    first, second = pair.root("detail-chain-first"), pair.root("detail-chain-second")
    handoff = materialize(pair, world.donor, first)
    defect = defect_of(
        lambda: materialize_codec_handoff(
            first,
            handoff.seal,
            pair.recipient,
            second,
            pair.donor,
            pair.source,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
        )
    )
    assert (defect.code, defect.detail) == (
        "chained_handoff",
        "the donor's own codec arrived by hand-off; a donor warms up for itself",
    )


def test_each_refusal_names_why_in_its_detail_for_a_donor_seal_that_is_not_the_warmup_final(
    world: World,
) -> None:
    defect = materialize_refused(world, "detail-foreign", seal=prediction_seal(world.donor))
    assert (defect.code, defect.detail) == (
        "foreign_donor",
        "a hand-off cites the donor's warm-up-final seal and no other",
    )


def head_parameter_name(pair: Pair) -> str:
    engine = SentenceExecution(pair.recipient, pair.source, max_source_bytes=SOURCE_LIMIT)
    return f"head.{next(iter(engine.model.head.named_parameters()))[0]}"


@pytest.mark.parametrize("kind", sorted(CONSTRUCTION_CHANGES))
def test_each_refusal_names_why_in_its_detail_for_each_arm_of_the_construction_guard(
    world: World, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    pair = world.pair
    monkeypatch.setattr(
        handoff_module, "SentenceExecution", changed_recipient(CONSTRUCTION_CHANGES[kind])
    )
    expected = {
        "binding": "donor and recipient bindings differ beyond the run identity and spine carrier",
        "inventory": "model inventories differ",
        "dtype": f"non-codec state {head_parameter_name(pair)} differs from the donor's",
    }[kind]
    defect = materialize_refused(world, f"detail-changed-{kind}")
    assert (defect.code, defect.detail) == ("construction_divergence", expected), kind


@pytest.mark.parametrize("key", NON_CODEC_KEYS)
def test_each_refusal_names_the_non_codec_tensor_that_differs(
    world: World, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setattr(handoff_module, "SentenceExecution", perturbed_recipient(key))
    defect = materialize_refused(world, f"detail-divergent-{key}")
    assert (defect.code, defect.detail) == (
        "construction_divergence",
        f"non-codec state {key} differs from the donor's",
    )


@pytest.mark.parametrize("copy_step", ["_adopt_codec", "_adopt_optimizer"])
def test_each_refusal_names_why_in_its_detail_for_a_copy_that_did_not_reproduce(
    world: World, monkeypatch: pytest.MonkeyPatch, copy_step: str
) -> None:
    monkeypatch.setattr(handoff_module, copy_step, lambda recipient, donor: None)
    defect = materialize_refused(world, f"detail-mismatch-{copy_step}")
    assert (defect.code, defect.detail) == (
        "payload_mismatch",
        "the recipient's copied warm-up differs from the donor's",
    )


def test_each_refusal_names_why_in_its_detail_for_an_rng_restore_that_did_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handoff_module, "restore_rng", lambda payload: None)
    defect = materialize_refused(world, "detail-rng")
    assert (defect.code, defect.detail) == (
        "payload_mismatch",
        "the donor's RNG state was not restored",
    )


def test_each_refusal_names_why_in_its_detail_for_an_independent_recipient(world: World) -> None:
    defect = defect_of(
        lambda: admit(world.pair, world.donor, world.independent, world.independent_seal)
    )
    assert (defect.code, defect.detail) == (
        "independent_warmup",
        "the recipient's chain starts with an ordinary warm-up step-one seal",
    )


def test_each_refusal_names_why_in_its_detail_for_a_record_that_cites_another_donor(
    world: World,
) -> None:
    pair = world.pair
    root = pair.root("detail-other-donor")
    handoff = materialize(pair, world.donor, root)
    defect = defect_of(lambda: admit(pair, world.other_donor, root, handoff.seal))
    assert (defect.code, defect.detail) == (
        "foreign_donor",
        "the record cites evidence that is not this donor's warm-up-final",
    )


@pytest.mark.parametrize(
    ("kind", "nudge", "claim_own", "who"),
    [
        ("recipient", nudge_codec, False, "recipient"),
        ("donor", nudge_codec, True, "donor"),
    ],
)
def test_each_refusal_names_the_side_whose_payload_differs_from_the_record(
    world: World, kind: str, nudge: Any, claim_own: bool, who: str
) -> None:
    pair = world.pair
    root, seal = forged_root(world, f"detail-payload-{kind}", nudge=nudge, claim_own=claim_own)
    defect = defect_of(lambda: admit(pair, world.donor, root, seal))
    assert (defect.code, defect.detail) == (
        "payload_mismatch",
        f"the {who}'s warm-up payload differs from the record's",
    )


# ------------------------------------------------ the digests, against an independent oracle
def framed(tag: str, data: bytes) -> bytes:
    """The digest stream's frame, restated here so a change of framing is a visible event."""
    label = tag.encode()
    return len(label).to_bytes(4, "little") + label + len(data).to_bytes(8, "little") + data


def tensor_frames(name: str, value: torch.Tensor) -> bytes:
    head = f"{value.dtype}:{list(value.shape)}".encode()
    body = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return framed(f"{name}:tensor", head) + framed(f"{name}:bytes", body)


def json_frame(name: str, value: object) -> bytes:
    text = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return framed(f"{name}:json", text.encode())


def test_the_optimizer_digest_is_the_framed_stream_of_groups_hyperparameters_and_named_state() -> (
    None
):
    """CH004 (the ``groups`` feed) and CH005 (the ``hyperparameters`` feed), with the state
    labelled by PARAMETER NAME in group order and keys sorted. The expected bytes are built here
    from the optimizer's own ``state_dict`` and the stream framing, never from the member's
    helpers. Group order is b then a, so name order, index order and group order all differ."""
    model = torch.nn.ModuleDict(
        {"a": torch.nn.Linear(1, 1, bias=False), "b": torch.nn.Linear(1, 1, bias=False)}
    )
    a, b = model["a"].weight, model["b"].weight
    optimizer = torch.optim.SGD([{"params": [b]}, {"params": [a], "lr": 0.2}], lr=0.1, momentum=0.9)
    for parameter in (a, b):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[b]["extra"] = 5  # a non-tensor value takes the json path, and sorts first
    engine: Any = SimpleNamespace(model=model, optimizer=optimizer)
    saved = optimizer.state_dict()
    hyper = [{k: v for k, v in g.items() if k != "params"} for g in saved["param_groups"]]
    expected = sha256(
        framed("groups", json.dumps(["b.weight", "a.weight"], separators=(",", ":")).encode())
        + framed(
            "hyperparameters",
            json.dumps(hyper, sort_keys=True, allow_nan=False, separators=(",", ":")).encode(),
        )
        + json_frame("b.weight/extra", 5)
        + tensor_frames("b.weight/momentum_buffer", saved["state"][0]["momentum_buffer"])
        + tensor_frames("a.weight/momentum_buffer", saved["state"][1]["momentum_buffer"])
    )
    assert handoff_module.optimizer_digest(engine) == expected.hexdigest()


def test_the_optimizer_digest_follows_the_group_order_and_the_hyperparameters_alone() -> None:
    """With no state yet, only the two feeds can tell these engines apart: the same parameters in
    another group order, and the same order under another learning rate."""

    def digest(order: tuple[str, ...], lr: float) -> str:
        model = torch.nn.ModuleDict(
            {"a": torch.nn.Linear(1, 1, bias=False), "b": torch.nn.Linear(1, 1, bias=False)}
        )
        optimizer = torch.optim.SGD([model[name].weight for name in order], lr=lr)
        engine: Any = SimpleNamespace(model=model, optimizer=optimizer)
        return handoff_module.optimizer_digest(engine)

    base = digest(("a", "b"), 0.1)
    assert digest(("a", "b"), 0.1) == base
    assert digest(("b", "a"), 0.1) != base  # the groups feed
    assert digest(("a", "b"), 0.2) != base  # the hyperparameters feed


def test_the_rng_digest_is_the_framed_cpu_state_then_every_cuda_state_in_order() -> None:
    """CH006 (the loop over CUDA states), CH032 (its ``cuda:<index>`` labels) and CH033 (the
    tuple requirement), pinned on CPU with synthetic states: the CPU fixtures carry an empty
    CUDA tuple, so no other test reaches these lines. This is the pure function only; the
    states of a REAL device are the measurement-gated test below."""
    cpu = torch.arange(6, dtype=torch.uint8)
    first, second = torch.arange(3, dtype=torch.uint8), torch.arange(4, dtype=torch.uint8) + 7
    payload = {"schema": "h-torch-rng-v1", "cpu": cpu, "cuda": (first, second)}
    expected = sha256(
        tensor_frames("cpu", cpu) + tensor_frames("cuda:0", first) + tensor_frames("cuda:1", second)
    )
    assert handoff_module.rng_digest(payload) == expected.hexdigest()
    swapped = {**payload, "cuda": (second, first)}
    assert handoff_module.rng_digest(swapped) != expected.hexdigest()
    with pytest.raises(ValueError, match="RNG payload must carry a tuple of CUDA states"):
        handoff_module.rng_digest({**payload, "cuda": [first, second]})


def test_the_digest_stream_frames_a_non_tensor_value_as_sorted_compact_finite_json() -> None:
    stream = handoff_module._Stream()  # pyright: ignore[reportPrivateUsage]
    stream.value("k", {"b": [1, 2.5], "a": None})
    assert stream.hexdigest() == sha256(json_frame("k", {"a": None, "b": [1, 2.5]})).hexdigest()
    assert json_frame("k", {"a": None, "b": [1, 2.5]}).endswith(b'{"a":null,"b":[1,2.5]}')
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            handoff_module._Stream().value("k", value)  # pyright: ignore[reportPrivateUsage]


def test_the_tensor_digest_refuses_a_big_endian_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``pragma: no cover`` branch of ``_Stream.value``: no supported host reaches it, so the
    host byte order is the one thing this test stands in for."""
    monkeypatch.setattr(handoff_module.sys, "byteorder", "big")
    with pytest.raises(RuntimeError, match="little-endian"):
        handoff_module._Stream().value("x", torch.zeros(1))  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------- ``_same_value``, arm by arm
def test_same_value_compares_a_tensor_by_dtype_and_bits_and_anything_else_canonically() -> None:
    """CH034, CH073 and CH074: the tensor arm's dtype and bit comparison, the tensor-versus-other
    guard in each direction, and the canonical arm. ``guard_construction`` reaches only the
    tensor arm (a ``state_dict`` holds tensors), so the rest is exercised directly."""
    same = handoff_module._same_value  # pyright: ignore[reportPrivateUsage]
    base = torch.tensor([1.0, 2.0])
    assert same(base, base.clone()) is True
    assert same(base, base.double()) is False  # equal values, another dtype: dtype alone differs
    assert same(base, torch.tensor([1.0, 2.0000002])) is False  # one ulp: only the bits differ
    # the shape comparison is redundant with ``torch.equal`` (False for unequal sizes), which is why
    # removing it alone (flip S02) stays GREEN; the outcome is what is pinned, with its reason
    assert torch.equal(torch.zeros(2, 1), torch.zeros(2)) is False
    assert same(torch.zeros(2, 1), torch.zeros(2)) is False
    assert same(base, [1.0, 2.0]) is False  # a tensor is never a list, whatever its values
    assert same([1.0, 2.0], base) is False  # and a list is never a tensor
    assert same({"a": 1, "b": 2}, {"b": 2, "a": 1}) is True  # canonical: key order is not a value
    assert same({"a": 1}, {"a": 2}) is False
    assert same(1, 1.0) is False  # canonical text, so an int is not its float


# --------------------------------------------------------- the pair predicate's JSON flags
class _Shaped:
    """What ``require_pair`` reads of a profile: ``spine.carrier`` and a fresh ``manifest``."""

    def __init__(self, carrier: str, manifest: dict[str, object]) -> None:
        self.spine = SimpleNamespace(carrier=carrier)
        self._manifest = manifest

    @property
    def manifest(self) -> dict[str, object]:
        return copy.deepcopy(self._manifest)


def shaped_pair(donor: dict[str, object], recipient: dict[str, object]) -> tuple[Any, Any]:
    return _Shaped("magnitude", donor), _Shaped("sphere", recipient)


def test_the_pair_predicate_ignores_key_order_and_refuses_what_it_cannot_serialize() -> None:
    """The flags of ``_pair_shape``'s ``json.dumps``. No real ``HProfile`` reaches them
    (``HProfile.manifest`` is a ``json.loads(json.dumps(..., allow_nan=False))`` round trip of
    one field order), so the profiles here are stand-ins for the two attributes read."""
    donor, recipient = shaped_pair(
        {"name": "d", "law": {"mode": "m", "k": 1}, "spine": {"carrier": "a", "width": 4}},
        {"spine": {"width": 4, "carrier": "b"}, "name": "r", "law": {"k": 1, "mode": "m"}},
    )
    handoff_module.require_pair(donor, recipient)  # the same content in another key order
    other = shaped_pair(
        {"name": "d", "law": {"k": 1}, "spine": {"carrier": "a"}},
        {"name": "r", "law": {"k": 2}, "spine": {"carrier": "b"}},
    )
    assert defect_of(lambda: handoff_module.require_pair(*other)).detail == OTHER_MANIFEST
    nan = shaped_pair(
        {"name": "d", "x": float("nan"), "spine": {"carrier": "a"}},
        {"name": "r", "x": float("nan"), "spine": {"carrier": "b"}},
    )
    with pytest.raises(ValueError, match="not JSON compliant") as raised:
        handoff_module.require_pair(*nan)
    assert not isinstance(raised.value, CodecHandoffDefect)


def test_the_pair_predicate_refuses_a_manifest_whose_spine_is_not_a_section() -> None:
    """The ``pragma: no cover`` branch of ``_pair_shape``: a real manifest always nests it."""
    donor, recipient = shaped_pair({"name": "d", "spine": ["a"]}, {"name": "r", "spine": ["b"]})
    with pytest.raises(ValueError, match="profile manifest lacks its spine section") as raised:
        handoff_module.require_pair(donor, recipient)
    assert not isinstance(raised.value, CodecHandoffDefect)


# --------------------------------------------------------- the devices a fork guards
def test_the_devices_a_hand_off_forks_are_the_one_cuda_device_for_a_cuda_profile_and_none(
    world: World,
) -> None:
    """CH077 and CH110, the PURE mapping only: whether the fork then protects a real generator is
    the measurement-gated test below."""
    recipient = world.pair.recipient
    cuda = replace(recipient, run_class="remote_tiny", device="cuda", precision="bf16")
    assert handoff_module._devices(recipient) == []  # pyright: ignore[reportPrivateUsage]
    assert handoff_module._devices(cuda) == [0]  # pyright: ignore[reportPrivateUsage]


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "measurement-gated (FOR-1713 CH006, CH032, CH033, CH077, CH110): needs a cuda host; "
        "this lane ran on a host without one, so the body below has never been executed"
    ),
)


@needs_cuda
def test_the_rng_digest_follows_a_real_cuda_generator() -> None:
    """CH006, CH032, CH033 on a device: ``capture_rng`` reads one state per visible device once
    CUDA is initialized, and the digest moves when the device generator does."""
    torch.zeros(1, device="cuda")  # initializes the CUDA context, as a pod's engine has
    payload = capture_rng()
    cuda = payload["cuda"]
    assert isinstance(cuda, tuple) and len(cuda) == torch.cuda.device_count() >= 1  # pyright: ignore[reportUnknownArgumentType]
    before = handoff_module.rng_digest(payload)
    torch.rand(1, device="cuda")
    assert handoff_module.rng_digest(capture_rng()) != before


@needs_cuda
def test_the_devices_a_cuda_hand_off_forks_leave_the_callers_cuda_generator_untouched(
    world: World,
) -> None:
    """CH077 and CH110 on a device: ``fork_rng(devices=_devices(cuda profile))`` restores the
    device generator, which ``devices=[]`` (the CPU arm) would leave advanced."""
    recipient = world.pair.recipient
    cuda = replace(recipient, run_class="remote_tiny", device="cuda", precision="bf16")
    devices = handoff_module._devices(cuda)  # pyright: ignore[reportPrivateUsage]
    torch.zeros(1, device="cuda")
    before = torch.cuda.get_rng_state(0)
    with torch.random.fork_rng(devices=devices):
        torch.rand(1, device="cuda")
    assert torch.equal(torch.cuda.get_rng_state(0), before)


# --------------------------------------------- the older journal checks of the seal walk
def forged_donor_seal(
    world: World, name: str, edit: Callable[[Path, dict[str, Any]], dict[str, Any]]
) -> tuple[Path, ArtifactIdentity]:
    """A copy of the donor root with one extra seal: the warm-up-final row after ``edit``.

    The copy keeps every hash honest, so the walk reaches the check under test and no earlier
    one; ``world.donor`` itself is read-only evidence (a test lists it to prove nothing writes
    there)."""
    root = world.pair.base / name
    shutil.copytree(world.donor, root)
    row = edit(root, seal_row(root, warmup_final(root)))
    write_json(root, f"{name}-seal.json", row, max_bytes=LIMIT)
    return root, identity_of(root / f"{name}-seal.json")


def read_donor_seal(world: World, root: Path, seal: ArtifactIdentity) -> SealedStep:
    pair = world.pair
    return read_seal(
        root, seal, pair.donor, pair.source, max_bytes=LIMIT, max_source_bytes=SOURCE_LIMIT
    )


def as_legacy_journal(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """The same shard under the v1 schema names (rows without their ``mechanism`` block)."""
    journal = dict(read_json(root, artifact_from_mapping(row["journal"]), max_bytes=LIMIT))
    legacy = {
        **journal,
        "schema": "h-journal-shard-v1",
        "rows": [
            {**{k: v for k, v in r.items() if k != "mechanism"}, "schema": "h-training-step-v1"}
            for r in journal["rows"]
        ],
    }
    reference = write_json(root, "legacy-journal.json", legacy, max_bytes=LIMIT)
    return {**row, "journal": artifact_reference(root, reference)}


def test_a_seal_over_a_v1_journal_is_refused_by_the_walk(world: World) -> None:
    """U1: the walk admits only v2 shards for a current seal (the v1 reader is a different one).
    The v1 shard is otherwise valid, so the version check is the first thing to refuse it."""
    root, seal = forged_donor_seal(world, "u1-v1-journal", as_legacy_journal)
    with pytest.raises(ValueError, match="current restart seals require journal v2"):
        read_donor_seal(world, root, seal)


@pytest.mark.parametrize(
    "change", [{"phase": "prediction"}, {"step": 1}], ids=["phase-half", "end-step-half"]
)
def test_a_seal_whose_endpoints_disagree_with_its_journal_is_refused_by_the_walk(
    world: World, change: dict[str, object]
) -> None:
    """U2: the seal's phase and step must equal the journal's phase and END step; each half is a
    tooth of its own (a wrong phase or a wrong step alone would otherwise be refused later, by a
    checkpoint check with another message)."""
    root, seal = forged_donor_seal(
        world, f"u2-{'-'.join(change)}", lambda _root, row: {**row, **change}
    )
    with pytest.raises(ValueError, match="seal and journal endpoints disagree"):
        read_donor_seal(world, root, seal)


# --------------------------------------------------------- FOR-1713: one disposition per residual
# The flips named here are the lane's measurement ids (evidence file of the lane handoff). The
# ticket's own CH numbers come from the sweep's ``mutations.txt``, which this lane did not have,
# so each residual is keyed by the ticket's GROUP and the flips are this lane's. HRun frozen
# flag (HA020) predates the hand-off and is not in this lane's scope.
FOR_1713: dict[str, str] = {
    "defect-detail-texts": (
        "pinned by test_each_refusal_names_why_in_its_detail_for_* (flips D01 to D12 RED); "
        "the two texts raised only from _resolve_warmup_final and the redundant post-resolve "
        "check are not pinned: superseded by the FOR-1712 fold (FOR-1717) and redundant"
    ),
    "optimizer-digest-feeds (CH004, CH005)": (
        "pinned by test_the_optimizer_digest_is_the_framed_stream_of_groups_hyperparameters_and_"
        "named_state and test_the_optimizer_digest_follows_the_group_order_and_the_hyperparameters"
        "_alone (flips O01 to O05 RED)"
    ),
    "cuda-rng-arms (CH006, CH032, CH033)": (
        "pinned on CPU by test_the_rng_digest_is_the_framed_cpu_state_then_every_cuda_state_in_"
        "order (flips R01 to R03 RED); the states of a real device are measurement-gated: "
        "test_the_rng_digest_follows_a_real_cuda_generator needs a cuda host"
    ),
    "same-value-internals (CH034, CH073, CH074)": (
        "pinned by test_same_value_compares_a_tensor_by_dtype_and_bits_and_anything_else_"
        "canonically (flips S01, S03 to S06 RED); redundant: the shape comparison, because "
        "torch.equal is already False for unequal sizes (flip S02 GREEN)"
    ),
    "devices-cuda-arm (CH077, CH110)": (
        "pinned for the mapping by test_the_devices_a_hand_off_forks_are_the_one_cuda_device_"
        "for_a_cuda_profile_and_none (flips V01, V02 RED); the fork's effect on a real generator "
        "is measurement-gated: test_the_devices_a_cuda_hand_off_forks_leave_the_callers_cuda_"
        "generator_untouched needs a cuda host"
    ),
    "post-resolve-check (CH049, CH089)": (
        "redundant: _resolve_warmup_final returns a node whose raw row already has the "
        "warm-up-final phase and step, and read_seal derives the SealedStep from that same row "
        "(flip X01, deleting the check, stays GREEN across all five hand-off test files); "
        "FOR-1717's fold keeps or drops it with the traversal"
    ),
    "json-flags": (
        "pinned by test_the_digest_stream_frames_a_non_tensor_value_as_sorted_compact_finite_json "
        "and test_the_pair_predicate_ignores_key_order_and_refuses_what_it_cannot_serialize "
        "(flips J01, J02, J04 to J06 RED); unobservable: the separators of _pair_shape, whose "
        "string is only compared with its twin built by the same function (flip J03 GREEN)"
    ),
    "pragma-branches": (
        "pinned by test_the_pair_predicate_refuses_a_manifest_whose_spine_is_not_a_section and "
        "test_the_tensor_digest_refuses_a_big_endian_host (flips P01, P02 RED)"
    ),
    "journal-checks (U1, U2)": (
        "pinned by test_a_seal_over_a_v1_journal_is_refused_by_the_walk and "
        "test_a_seal_whose_endpoints_disagree_with_its_journal_is_refused_by_the_walk "
        "(flips U01, U02a, U02b RED)"
    ),
    "handoff-node-step-and-cursor": (
        "pinned as a pair by test_a_handoff_checkpoint_at_another_step_is_refused_by_the_seal_walk "
        "(flips HS1, HS2 GREEN alone, HS3 RED together); each alone is redundant with the other "
        "because save_checkpoint keeps a checkpoint's step equal to its source cursor"
    ),
    "binding-type (H6)": (
        "redundant: a non-string binding is refused one line later by the binding comparison "
        "with 'checkpoint construction differs from sealed profile' (flip H06 stays GREEN "
        "across all five hand-off test files); no public writer produces one"
    ),
}


def test_every_for_1713_residual_has_exactly_one_disposition() -> None:
    kinds = (
        "pinned",
        "redundant",
        "unobservable",
        "unreachable",
        "measurement-gated",
        "superseded",
    )
    assert len(FOR_1713) == 11
    names = {k for k, v in globals().items() if k.startswith("test_") and callable(v)}
    for residual, text in FOR_1713.items():
        assert "\n" not in text and text.startswith(kinds), residual
        cited = re.findall(r"test_[a-z0-9_]+", text)
        assert bool(cited) == (text.startswith("pinned") or "measurement-gated" in text), residual
        for name in cited:  # a trailing underscore names the family, not one test
            assert name.endswith("_") or name in names, (residual, name)
    assert any(n.startswith("test_each_refusal_names_why_in_its_detail_for_") for n in names)
