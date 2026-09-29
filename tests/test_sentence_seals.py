"""A valid journal and checkpoint must also belong to the same measured lineage."""

import json
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
    memo -- resume/pause/completion always pass ``memo=None``."""
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
