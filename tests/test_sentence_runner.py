"""The training driver traverses one real profile and resumes the same lineage."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_sentence_execution import engine

import entropic_substrate.ladder.run.seals as seals_module
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.runner import (
    TrainingProgress,
    read_training_progress,
    run_probe_training,
)
from entropic_substrate.ladder.run.seals import SealMemo, read_seal

pytestmark = pytest.mark.unit
LIMIT = 10_000_000


def test_training_driver_completes_declared_budgets_and_evaluation_grid(tmp_path: Path) -> None:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    result = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    assert result.status == "training_complete"
    seal = read_seal(
        output,
        result.last_seal,
        template.profile,
        template.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
    )
    assert seal.phase == "prediction" and seal.step == 4
    receipt = read_json(output, result.receipt, max_bytes=LIMIT)
    assert receipt["successful_optimizer_steps"] == 6
    assert receipt["run_class"] == "local_tiny" and receipt["scientific_authority"] == "none"
    assert [item["step"] for item in receipt["evaluations"]] == [2, 4]


def test_pause_and_resume_finish_without_restarting_the_codec(tmp_path: Path) -> None:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    paused = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        stop_after_seal=lambda phase, step: phase == "prediction" and step == 2,
    )
    assert paused.status == "paused"
    completed = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        resume=paused.receipt,
    )
    assert completed.status == "training_complete"
    last = read_seal(
        output,
        completed.last_seal,
        template.profile,
        template.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
    )
    assert last.chain_length == 3
    report = read_json(output, completed.receipt, max_bytes=LIMIT)
    assert [item["step"] for item in report["evaluations"]] == [2, 4]
    assert report["parent_receipt"]["sha256"] == paused.receipt.content_hash.digest


def test_fresh_run_never_overwrites_an_existing_output(tmp_path: Path) -> None:
    template = engine(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        run_probe_training(
            template.profile,
            template.source_root,
            tmp_path,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )


def test_numerical_failure_retains_evidence_and_does_not_emit_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run.execution import SentenceExecution

    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    original = SentenceExecution.advance

    def fail_after_first(self: SentenceExecution) -> dict[str, object]:
        if self.step == 1:
            raise RuntimeError("injected numerical failure")
        return original(self)

    monkeypatch.setattr(SentenceExecution, "advance", fail_after_first)
    with pytest.raises(RuntimeError, match="injected"):
        run_probe_training(
            template.profile,
            template.source_root,
            output,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )
    failures = list(output.glob("*-failure.json"))
    assert len(failures) == 1
    assert not list(output.glob("*-progress.json"))


def test_probe_driver_refuses_scientific_authority_and_remote_successors(tmp_path: Path) -> None:
    from dataclasses import replace

    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    for profile in (
        replace(
            template.profile,
            name="h0-fisher",
            run_class="confirmatory",
            device="cuda",
            precision="bf16",
        ),
        replace(
            template.profile,
            name="h3-fisher-tiny",
            stage="H3",
            run_class="remote_tiny",
            device="cuda",
            precision="bf16",
        ),
    ):
        with pytest.raises(ValueError, match="authority"):
            run_probe_training(
                profile,
                template.source_root,
                output,
                max_source_bytes=1_000_000,
                max_artifact_bytes=LIMIT,
            )
    assert not list(output.iterdir())


def test_empty_evaluation_cannot_satisfy_the_completion_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run.execution import SentenceExecution

    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    original = SentenceExecution.evaluate

    def empty(self: SentenceExecution) -> dict[str, object]:
        result = original(self)
        result["examples"] = []
        return result

    monkeypatch.setattr(SentenceExecution, "evaluate", empty)
    with pytest.raises(ValueError, match="evaluation"):
        run_probe_training(
            template.profile,
            template.source_root,
            output,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )


def test_completed_receipt_is_idempotent_without_reinitializing_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    first = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    files = sorted(p.name for p in output.iterdir())

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("completed execution must not restart")

    monkeypatch.setattr(SentenceExecution, "__init__", forbidden)
    second = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        resume=first.receipt,
    )
    assert second == first and sorted(p.name for p in output.iterdir()) == files


def _paused(tmp_path: Path) -> tuple[TrainingProgress, Path, SentenceExecution]:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    paused = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        stop_after_seal=lambda phase, step: phase == "prediction" and step == 2,
    )
    return paused, output, template


def test_the_completion_walk_hands_its_populated_memo_to_the_caller(tmp_path: Path) -> None:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    result = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    memo = result.admission_memo
    assert memo is not None and memo.populated
    # The completion walk is a FULL walk: all three nodes from disk, both evaluations.
    assert (memo.seals_walked, memo.journals_read, memo.evaluations_read) == (3, 3, 2)
    assert memo.get(result.last_seal) is not None


def test_training_progress_keeps_its_value_semantics_with_the_memo_riding_along(
    tmp_path: Path,
) -> None:
    """The memo is a process-local hand-off: invisible to ==, hash and repr."""
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    result = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    bare = replace(result, admission_memo=None)
    assert result == bare and hash(result) == hash(bare)
    assert "memo" not in repr(result) and repr(result) == repr(bare)
    again = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        resume=result.receipt,
    )
    # An already-complete resume ran no completion walk here: nothing to hand on.
    assert again == result and again.admission_memo is None


def test_a_fresh_process_resume_still_walks_every_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) A resume owns no memo from an earlier process: every node is read from disk."""
    paused, output, template = _paused(tmp_path)
    assert paused.admission_memo is not None  # the pausing process's own memo
    profile = template.profile
    reads: list[int] = []
    original = seals_module.read_journal

    def counted(*args: object, **kwargs: object) -> object:
        reads.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(seals_module, "read_journal", counted)
    kwargs = {
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": LIMIT,
    }
    source = template.source_root
    fresh = SealMemo()  # a new process's memo: empty, so it must visit all of the chain
    read_training_progress(output, paused.receipt, profile, source, memo=fresh, **kwargs)
    assert (fresh.seals_walked, fresh.journals_read) == (2, 2)  # chain length 2, none skipped
    # With no memo at all every read re-walks: 2 for the last seal + 2 + 2 for its evaluation.
    reads.clear()
    read_training_progress(output, paused.receipt, profile, source, memo=None, **kwargs)
    assert len(reads) == 6


def test_a_tampered_journal_is_refused_by_a_fresh_process_resume(tmp_path: Path) -> None:
    paused, output, template = _paused(tmp_path)
    journal = sorted(output.glob("*-codec_warmup-*-journal.json"))[0]
    journal.write_bytes(journal.read_bytes() + b" ")
    with pytest.raises(ValueError):
        run_probe_training(
            template.profile,
            template.source_root,
            output,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
            resume=paused.receipt,
        )
