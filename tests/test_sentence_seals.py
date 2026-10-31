"""A valid journal and checkpoint must also belong to the same measured lineage."""

import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from tests.test_sentence_execution import engine

import entropic_substrate.ladder.run.seals as seals_module
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.seals import SealMemo, read_seal, resume_seal, seal_step

pytestmark = pytest.mark.unit
LIMIT = 10_000_000


def _advance(run: SentenceExecution) -> dict[str, object]:
    """Advance past the tiny warmup/prediction phase boundary transparently."""
    if run.phase == "codec_warmup" and run.step == run.profile.schedule.warmup_steps:
        run.begin_prediction()
    return run.advance()


def _counted_ancestor_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch the module's OWN from-import bindings (FOR-1567/C13): a counter
    patched on ``ladder.run.journal.read_journal`` or
    ``ladder.run.checkpoint.checkpoint_metadata`` would never fire, because
    ``seals.py`` imported these names directly into its own namespace."""
    calls: list[int] = []
    original_journal = seals_module.read_journal
    original_metadata = seals_module.checkpoint_metadata

    def counted_journal(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original_journal(*args, **kwargs)

    def counted_metadata(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original_metadata(*args, **kwargs)

    monkeypatch.setattr(seals_module, "read_journal", counted_journal)
    monkeypatch.setattr(seals_module, "checkpoint_metadata", counted_metadata)
    return calls


def test_constant_ancestor_reads_per_checkpoint_with_a_shared_memo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _counted_ancestor_reads(monkeypatch)
    run = engine(tmp_path)
    memo = SealMemo()
    parent: ArtifactIdentity | None = None
    per_checkpoint: list[int] = []
    for _ in range(5):
        seal = seal_step(
            tmp_path, "chain", run, (_advance(run),), parent=parent, max_bytes=LIMIT, memo=memo
        )
        before = len(calls)
        read_seal(
            tmp_path,
            seal,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
            memo=memo,
        )
        per_checkpoint.append(len(calls) - before)
        parent = seal
    # Each checkpoint reads exactly its OWN new node's journal+metadata (2
    # calls: one read_journal, one checkpoint_metadata), independent of k --
    # never the whole chain behind it.
    assert per_checkpoint == [2, 2, 2, 2, 2]


def test_seal_step_itself_uses_the_memoised_parent_without_rewalking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolates seal_step's OWN internal ``read_seal(parent, memo=...)`` call
    (L1) from the follow-up verification call a caller like
    ``read_training_progress`` makes -- dropping ``memo=memo`` at seal_step's
    call site must be visible here even though the chain is still admitted
    correctly (the guard is about READ COUNT, not correctness)."""
    calls = _counted_ancestor_reads(monkeypatch)
    run = engine(tmp_path)
    memo = SealMemo()
    seal1 = seal_step(
        tmp_path, "chain", run, (_advance(run),), parent=None, max_bytes=LIMIT, memo=memo
    )
    read_seal(
        tmp_path,
        seal1,
        run.profile,
        run.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
        memo=memo,
    )
    before = len(calls)
    seal_step(tmp_path, "chain", run, (_advance(run),), parent=seal1, max_bytes=LIMIT, memo=memo)
    # seal_step's OWN call to resolve `previous` must be a memo hit: zero
    # ancestor reads, independent of how deep the chain behind seal1 is.
    assert len(calls) - before == 0


def test_a_fresh_memo_each_call_grows_with_chain_length_control_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control arm (playbook include-control-in-probe): a FRESH memo every
    call must NOT be fast -- proving the constant count above comes from
    memo reuse within one invocation, not from some other shortcut."""
    calls = _counted_ancestor_reads(monkeypatch)
    run = engine(tmp_path)
    parent: ArtifactIdentity | None = None
    per_call: list[int] = []
    for _ in range(4):
        seal = seal_step(tmp_path, "chain", run, (_advance(run),), parent=parent, max_bytes=LIMIT)
        before = len(calls)
        read_seal(
            tmp_path,
            seal,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
            memo=SealMemo(),  # a BRAND NEW, empty memo every time
        )
        per_call.append(len(calls) - before)
        parent = seal
    assert per_call == [2, 4, 6, 8]


def test_a_tampered_ancestor_is_refused_at_a_fresh_process_resume(tmp_path: Path) -> None:
    """Guard teeth: a chain built and (correctly) memoised within ONE process
    must still be refused, in full, by a FRESH process that never saw that
    memo, which has none (``memo=None``). RETRACTION (FOR-1662, house rule 4):
    this docstring used to say "resume/pause/completion always pass
    ``memo=None``". A fresh-process resume still does; the runner's own
    completion/pause walk now passes an EMPTY memo (the same full walk) and
    hands the populated memo to the later stages of that same process."""
    run = engine(tmp_path)
    memo = SealMemo()
    parent: ArtifactIdentity | None = None
    seals: list[ArtifactIdentity] = []
    for _ in range(3):
        seal = seal_step(
            tmp_path, "chain", run, (_advance(run),), parent=parent, max_bytes=LIMIT, memo=memo
        )
        read_seal(
            tmp_path,
            seal,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
            memo=memo,
        )
        seals.append(seal)
        parent = seal
    # Tamper with the FIRST (root) checkpoint's bytes, now that later seals
    # are memoised as trusting it.
    payload = read_json(tmp_path, seals[0], max_bytes=LIMIT)
    checkpoint_path = tmp_path / payload["checkpoint"]["path"]
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        read_seal(
            tmp_path,
            seals[-1],
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
            memo=None,  # a fresh process never shares the old memo
        )


def test_sealed_chain_replays_warmup_transition_and_resume(tmp_path: Path) -> None:
    run = engine(tmp_path)
    warmup = seal_step(
        tmp_path, "first", run, (run.advance(), run.advance()), parent=None, max_bytes=LIMIT
    )
    run.begin_prediction()
    prediction = seal_step(
        tmp_path, "first", run, (run.advance(), run.advance()), parent=warmup, max_bytes=LIMIT
    )
    seal = read_seal(
        tmp_path,
        prediction,
        run.profile,
        run.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
    )
    assert seal.phase == "prediction" and seal.step == 2
    fresh = SentenceExecution(run.profile, run.source_root, max_source_bytes=1_000_000)
    resume_seal(tmp_path, prediction, fresh, max_bytes=LIMIT)
    tail = seal_step(
        tmp_path,
        "resumed",
        fresh,
        (fresh.advance(), fresh.advance()),
        parent=prediction,
        max_bytes=LIMIT,
    )
    final = read_seal(
        tmp_path, tail, run.profile, run.source_root, max_bytes=LIMIT, max_source_bytes=1_000_000
    )
    assert final.step == 4 and final.chain_length == 3


def test_initial_or_phase_transition_gaps_are_refused(tmp_path: Path) -> None:
    run = engine(tmp_path)
    run.advance()
    with pytest.raises(ValueError, match="initial"):
        seal_step(tmp_path, "gap", run, (run.advance(),), parent=None, max_bytes=LIMIT)
    assert not list(tmp_path.glob("gap*"))
    run.begin_prediction()
    with pytest.raises(ValueError, match="initial"):
        seal_step(tmp_path, "skip", run, (run.advance(),), parent=None, max_bytes=LIMIT)


def test_the_ordinary_first_seal_refusals_stay_exactly_as_published(tmp_path: Path) -> None:
    """Companion pin (FOR-1692, house rule 4): the two refusals the hand-off variant must
    NOT loosen, quoted by their exact text. A hand-off chain root is a different schema with
    its own predicate; an ordinary seal that does not start at warm-up step one, and a seal
    row whose schema is not an admitted one, keep failing with these words."""
    run = engine(tmp_path)
    run.advance()
    with pytest.raises(ValueError, match=r"^initial seal must start codec warm-up at step one$"):
        seal_step(tmp_path, "gap", run, (run.advance(),), parent=None, max_bytes=LIMIT)
    assert not list(tmp_path.glob("gap*"))
    fresh = SentenceExecution(run.profile, run.source_root, max_source_bytes=1_000_000)
    seal = seal_step(tmp_path, "ok", fresh, (fresh.advance(),), parent=None, max_bytes=LIMIT)
    row = read_json(tmp_path, seal, max_bytes=LIMIT)
    row["schema"] = "h-checkpoint-seal-v2"
    raw = json.dumps(row).encode()
    (tmp_path / "unknown-schema.json").write_bytes(raw)
    forged = ArtifactIdentity("unknown-schema.json", ContentHash("sha256", sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match=r"^seal construction, phase, or step is invalid$"):
        read_seal(
            tmp_path,
            forged,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
        )


def test_swapped_individually_valid_checkpoint_fails_measurement_binding(tmp_path: Path) -> None:
    run = engine(tmp_path)
    first = seal_step(
        tmp_path, "a", run, (run.advance(), run.advance()), parent=None, max_bytes=LIMIT
    )
    other = SentenceExecution(run.profile, run.source_root, max_source_bytes=1_000_000)
    torch.randn(11)
    second = seal_step(
        tmp_path, "b", other, (other.advance(), other.advance()), parent=None, max_bytes=LIMIT
    )
    payload = read_json(tmp_path, first, max_bytes=LIMIT)
    payload["checkpoint"] = read_json(tmp_path, second, max_bytes=LIMIT)["checkpoint"]
    raw = json.dumps(payload).encode()
    (tmp_path / "swapped.json").write_bytes(raw)
    forged = ArtifactIdentity("swapped.json", ContentHash("sha256", sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match="measurement"):
        read_seal(
            tmp_path,
            forged,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
        )


def test_seal_prefix_and_artifact_tampering_are_not_silently_repaired(tmp_path: Path) -> None:
    run = engine(tmp_path)
    rows = (run.advance(), run.advance())
    with pytest.raises(ValueError, match="prefix"):
        seal_step(tmp_path, "../escape", run, rows, parent=None, max_bytes=LIMIT)
    seal = seal_step(tmp_path, "one", run, rows, parent=None, max_bytes=LIMIT)
    payload = read_json(tmp_path, seal, max_bytes=LIMIT)
    checkpoint = tmp_path / payload["checkpoint"]["path"]
    checkpoint.write_bytes(b"not the captured checkpoint")
    with pytest.raises(ValueError, match="hash"):
        read_seal(
            tmp_path,
            seal,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
        )


def test_optimizer_drift_cannot_keep_the_old_profile_label(tmp_path: Path) -> None:
    run = engine(tmp_path)
    run.optimizer.defaults["lr"] *= 2
    for group in run.optimizer.param_groups:
        group["lr"] *= 2
    rows = (run.advance(), run.advance())
    with pytest.raises(ValueError, match="construction"):
        artifact = seal_step(tmp_path, "drift", run, rows, parent=None, max_bytes=LIMIT)
        read_seal(
            tmp_path,
            artifact,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
        )


def test_prediction_cannot_skip_unsealed_warmup_work(tmp_path: Path) -> None:
    run = engine(tmp_path)
    parent = seal_step(tmp_path, "partial", run, (run.advance(),), parent=None, max_bytes=LIMIT)
    run.advance()
    run.begin_prediction()
    with pytest.raises(ValueError, match="complete warm-up"):
        seal_step(tmp_path, "skip", run, (run.advance(),), parent=parent, max_bytes=LIMIT)


def test_seal_inspection_preserves_training_rng(tmp_path: Path) -> None:
    run = engine(tmp_path)
    artifact = seal_step(
        tmp_path, "one", run, (run.advance(), run.advance()), parent=None, max_bytes=LIMIT
    )
    before = torch.get_rng_state()
    read_seal(
        tmp_path,
        artifact,
        run.profile,
        run.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
    )
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)


def _chain(tmp_path: Path, length: int) -> tuple[SentenceExecution, list[ArtifactIdentity]]:
    run = engine(tmp_path)
    seals: list[ArtifactIdentity] = []
    parent: ArtifactIdentity | None = None
    for _ in range(length):
        parent = seal_step(tmp_path, "chain", run, (_advance(run),), parent=parent, max_bytes=LIMIT)
        seals.append(parent)
    return run, seals


def test_a_memo_populated_by_a_memoless_walk_answers_get_for_every_node(tmp_path: Path) -> None:
    """(d) A fresh memo handed to ONE walk of a chain built WITHOUT a memo ends up
    holding every node of that walk, with the chain length each node had."""
    run, seals = _chain(tmp_path, 3)
    memo = SealMemo()
    assert memo.populated is False
    tip = read_seal(
        tmp_path,
        seals[-1],
        run.profile,
        run.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
        memo=memo,
    )
    assert memo.populated is True and tip.chain_length == 3
    for length, seal in enumerate(seals, start=1):
        remembered = memo.get(seal)
        assert remembered is not None and remembered.chain_length == length
    # Measured work, not the chain length: three nodes verified from disk, three journals.
    assert (memo.seals_walked, memo.journals_read) == (3, 3)
    assert memo.admission_seconds > 0.0


def test_a_second_walk_with_the_populated_memo_reads_nothing_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, seals = _chain(tmp_path, 3)
    memo = SealMemo()
    kwargs = {"max_bytes": LIMIT, "max_source_bytes": 1_000_000}
    read_seal(tmp_path, seals[-1], run.profile, run.source_root, memo=memo, **kwargs)
    calls = _counted_ancestor_reads(monkeypatch)
    again = read_seal(tmp_path, seals[-1], run.profile, run.source_root, memo=memo, **kwargs)
    assert again.chain_length == 3 and calls == []
    assert (memo.seals_walked, memo.journals_read) == (3, 3)  # unchanged: no new work


def test_resume_seal_forwards_its_memo_and_defaults_to_the_full_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, seals = _chain(tmp_path, 2)
    memo = SealMemo()
    read_seal(
        tmp_path,
        seals[-1],
        run.profile,
        run.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
        memo=memo,
    )
    calls = _counted_ancestor_reads(monkeypatch)
    fresh = SentenceExecution(run.profile, run.source_root, max_source_bytes=1_000_000)
    resume_seal(tmp_path, seals[-1], fresh, max_bytes=LIMIT, memo=memo)
    assert calls == []  # the memo reached read_seal
    again = SentenceExecution(run.profile, run.source_root, max_source_bytes=1_000_000)
    resume_seal(tmp_path, seals[-1], again, max_bytes=LIMIT)
    assert len(calls) == 4  # memo-less: 2 nodes x (journal + checkpoint metadata)


def test_a_memo_never_skips_the_source_digest_check(tmp_path: Path) -> None:
    """Guard teeth for the unconditional corpus verification: a memo HIT must still
    refuse a source whose digest differs from ``profile.source_digest``."""
    run, seals = _chain(tmp_path, 2)
    memo = SealMemo()
    kwargs = {"max_bytes": LIMIT, "max_source_bytes": 1_000_000}
    read_seal(tmp_path, seals[-1], run.profile, run.source_root, memo=memo, **kwargs)
    assert memo.get(seals[-1]) is not None  # the tip itself is a memo hit
    forged = tmp_path / "forged-source"
    shutil.copytree(run.source_root, forged)
    corpus = forged / "prepared-corpus.json"
    corpus.write_bytes(corpus.read_bytes() + b" ")
    with pytest.raises(ValueError):
        read_seal(tmp_path, seals[-1], run.profile, forged, memo=memo, **kwargs)
