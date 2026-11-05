"""FOR-1670: an admitted evaluation record is remembered, so a shared admission stops re-reading it.

Instrument (MEASURED, not inferred): ``evaluation_records.read_json`` as bound in that
module is wrapped and counts calls per ``artifact.content_hash.digest``. It is the
ONLY place an evaluation artifact is read from disk (the receipt and seal reads go
through ``journal``/``seals``' own bindings), so a count of zero for a roster digest
means the record was served from the memo, not re-read.

Trust boundary pinned here (design-gate ruling): the map is keyed by CONTENT digest and a
hit does no filesystem access. A file rewritten after admission but presented with the
SAME identity is therefore a legitimate hit; detection of that belongs to a fresh memo or
a fresh process (DG-B). Tamper detection is pinned on the disk-read path: an identity
whose digest was never remembered is read, hashed, and refused on a mismatch.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from tests.latent_usage_support import POSTERIOR_MEAN
from tests.test_h_worker_log_lines import bind_logs, of, read_log
from tests.test_sentence_execution import engine

import entropic_substrate.ladder.run.evaluation_records as records_module
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.gate.ladder_campaign import artifact_from_mapping
from entropic_substrate.ladder.run.evaluation_records import read_evaluation
from entropic_substrate.ladder.run.journal import read_json, write_json
from entropic_substrate.ladder.run.runner import read_training_progress, run_probe_training
from entropic_substrate.ladder.run.seals import SealMemo, narrated_admission

pytestmark = pytest.mark.unit
LIMIT = 10_000_000
SOURCE_LIMIT = 1_000_000


class _Run:
    """One real tiny training run: its profile, roots, receipt and evaluation roster."""

    def __init__(self, tmp_path: Path) -> None:
        template = engine(tmp_path)
        self.profile = template.profile
        self.source_root = template.source_root
        self.output = tmp_path / "run"
        self.output.mkdir()
        result = run_probe_training(
            self.profile,
            self.source_root,
            self.output,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
            latent_usage=POSTERIOR_MEAN,
        )
        self.receipt = result.receipt
        roster = read_json(self.output, self.receipt, max_bytes=LIMIT)["evaluations"]
        assert isinstance(roster, list)
        self.artifacts = [artifact_from_mapping(item["artifact"]) for item in roster]
        assert len(self.artifacts) == 2
        self.digests = {a.content_hash.digest for a in self.artifacts}

    def progress(self, memo: SealMemo | None) -> dict[str, object]:
        return read_training_progress(
            self.output,
            self.receipt,
            self.profile,
            self.source_root,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
            memo=memo,
        )

    def evaluation(self, artifact: ArtifactIdentity, memo: SealMemo | None) -> dict[str, object]:
        return read_evaluation(
            self.output,
            artifact,
            self.profile,
            self.source_root,
            max_bytes=LIMIT,
            max_source_bytes=SOURCE_LIMIT,
            memo=memo,
        )


def _count_reads(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    reads: Counter[str] = Counter()
    original = records_module.read_json

    def counted(root: Path, artifact: ArtifactIdentity, **kwargs: Any) -> dict[str, object]:
        reads[artifact.content_hash.digest] += 1
        return original(root, artifact, **kwargs)

    monkeypatch.setattr(records_module, "read_json", counted)
    return reads


def _evaluation_reads(reads: Counter[str], run: _Run) -> int:
    return sum(count for digest, count in reads.items() if digest in run.digests)


def test_a_shared_admission_reads_no_evaluation_record_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) The second admission with the same memo performs ZERO evaluation reads."""
    run = _Run(tmp_path)
    reads = _count_reads(monkeypatch)
    memo = SealMemo()
    first = run.progress(memo)
    assert _evaluation_reads(reads, run) == 2
    reads.clear()
    second = run.progress(memo)
    assert _evaluation_reads(reads, run) == 0, reads
    assert second == first


def test_a_hit_equals_an_independent_disk_read_and_is_the_remembered_object(
    tmp_path: Path,
) -> None:
    """The hit is compared with a memo-less disk read, never with the memo itself.

    Pinned choice: a hit returns the remembered object BY REFERENCE (callers treat it
    as read-only), so two hits are the same object.
    """
    run = _Run(tmp_path)
    memo = SealMemo()
    first = run.evaluation(run.artifacts[0], memo)
    hit = run.evaluation(run.artifacts[0], memo)
    assert hit is first and memo.evaluations_memo_hits == 1
    assert hit == run.evaluation(run.artifacts[0], None)


def test_without_a_memo_every_call_reads_every_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) DG-B: a fresh process (``memo=None``) walks and reads everything, every time."""
    run = _Run(tmp_path)
    reads = _count_reads(monkeypatch)
    run.evaluation(run.artifacts[0], None)
    run.evaluation(run.artifacts[0], None)
    assert reads[run.artifacts[0].content_hash.digest] == 2
    reads.clear()
    run.progress(None)
    run.progress(None)
    assert {digest: reads[digest] for digest in run.digests} == dict.fromkeys(run.digests, 2)


def test_the_counters_separate_disk_reads_from_memo_hits(tmp_path: Path) -> None:
    """(c) ``evaluations_read`` counts disk reads only; the hits ride beside it."""
    run = _Run(tmp_path)
    memo = SealMemo()
    assert (memo.evaluations_read, memo.evaluations_memo_hits) == (0, 0)
    run.progress(memo)
    assert (memo.evaluations_read, memo.evaluations_memo_hits) == (2, 0)
    run.progress(memo)
    assert (memo.evaluations_read, memo.evaluations_memo_hits) == (2, 2)


def test_remembering_an_evaluation_does_not_make_the_memo_populated() -> None:
    """``populated`` still means "holds a verified seal": it feeds fresh/shared provenance."""
    memo = SealMemo()
    identity = ArtifactIdentity("e.json", ContentHash("sha256", "e" * 64))
    memo.remember_evaluation(identity, {"step": 1})
    assert memo.get_evaluation(identity) == {"step": 1} and not memo.populated
    assert "step" not in repr(memo)


def test_an_identity_never_remembered_is_read_and_a_wrong_digest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) The key is the content digest: a remembered PATH with an unremembered digest reads."""
    run = _Run(tmp_path)
    memo = SealMemo()
    run.progress(memo)
    reads = _count_reads(monkeypatch)
    known = run.artifacts[0]
    stale = ArtifactIdentity(known.path, ContentHash("sha256", "0" * 64))
    with pytest.raises(ValueError, match="hash mismatch"):
        run.evaluation(stale, memo)
    assert reads["0" * 64] == 1, "the unremembered digest must be a disk read, not a hit"
    assert (memo.evaluations_read, memo.evaluations_memo_hits) == (2, 0)
    # A DIFFERENT file whose digest was never remembered is a disk read as well.
    payload = copy.deepcopy(read_json(run.output, known, max_bytes=LIMIT))
    wall = payload["wall_seconds"]
    assert isinstance(wall, float)
    payload["wall_seconds"] = wall + 1.0
    other = write_json(run.output, "other-evaluation.json", payload, max_bytes=LIMIT)
    row = run.evaluation(other, memo)
    assert reads[other.content_hash.digest] == 1 and row["wall_seconds"] == wall + 1.0
    assert (memo.evaluations_read, memo.evaluations_memo_hits) == (3, 0)


def test_a_hit_trusts_the_remembered_content_and_does_not_touch_the_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trust boundary: a rewrite after admission under the SAME identity is still a hit.

    Rewrite detection belongs to a fresh memo (below) or a fresh process, never to the
    hit path (no stat, no mtime, no re-hash).
    """
    run = _Run(tmp_path)
    memo = SealMemo()
    remembered = run.evaluation(run.artifacts[0], memo)
    (run.output / run.artifacts[0].path).write_bytes(b'{"tampered":true}')
    reads = _count_reads(monkeypatch)
    assert run.evaluation(run.artifacts[0], memo) is remembered
    assert _evaluation_reads(reads, run) == 0
    with pytest.raises(ValueError, match="hash mismatch"):
        run.evaluation(run.artifacts[0], SealMemo())


def test_a_refused_record_is_neither_counted_nor_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counter and memo advance only after the WHOLE verification returned."""
    run = _Run(tmp_path)
    payload = copy.deepcopy(read_json(run.output, run.artifacts[0], max_bytes=LIMIT))
    payload["wall_seconds"] = -1
    bad = write_json(run.output, "bad-evaluation.json", payload, max_bytes=LIMIT)
    memo = SealMemo()
    reads = _count_reads(monkeypatch)
    for attempt in (1, 2):
        with pytest.raises(ValueError, match="nonnegative"):
            run.evaluation(bad, memo)
        assert reads[bad.content_hash.digest] == attempt
    assert memo.get_evaluation(bad) is None
    assert (memo.evaluations_read, memo.evaluations_memo_hits) == (0, 0)


def test_admission_end_reports_disk_reads_and_memo_hits_separately(tmp_path: Path) -> None:
    """(e) The log-only narration: ``evaluations`` is disk reads, hits are their own field."""
    run = _Run(tmp_path)
    log = tmp_path / "obs" / "worker-log.jsonl"
    bind_logs(log)
    memo = SealMemo()
    with narrated_admission("completion", "run", memo):
        run.progress(memo)
    with narrated_admission("generation", "run", memo):
        run.progress(memo)
    rows = read_log(log)
    first, second = of(rows, "admission.end")
    assert (first["evaluations"], first["evaluations_memo_hits"]) == (2, 0)
    assert (second["evaluations"], second["evaluations_memo_hits"]) == (0, 2)
    # The start line carries only what is known before the walk.
    assert all("evaluations_memo_hits" not in row for row in of(rows, "admission.start"))


def test_the_remembered_record_is_the_one_read_back_from_disk(tmp_path: Path) -> None:
    """A writer's in-memory report is never written into the memo: only disk-verified rows."""
    run = _Run(tmp_path)
    memo = SealMemo()
    run.evaluation(run.artifacts[0], memo)
    remembered = memo.get_evaluation(run.artifacts[0])
    assert remembered == read_json(run.output, run.artifacts[0], max_bytes=LIMIT)
