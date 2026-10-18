"""FOR-1662: the final leg admits each artifact root with ONE fresh walk, not one per stage.

Instrument (design-gate ruling D2, MEASURED): every consumer module's OWN
from-import binding of ``read_training_progress`` is wrapped, and an entry
counts when its ``memo`` is ``None`` or EMPTY at entry. Counting ``read_seal``
calls would read 0 at a correct tip as well as at the parent, because a
threaded memo makes the per-call count identical either way; the memo state AT
ENTRY is what distinguishes "this call walks the whole chain" from "this call
trusts an admitted ancestry". ``resume_seal`` and the bare span ``read_seal``
are counted separately, and only AFTER training returns (the training loop's
own invocation memo is legitimately empty at its first ``seal_step``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_worker_log_lines import bind_logs, of, read_log
from tests.test_sentence_worker_execution import prepared

import entropic_substrate.ladder.run.generation as generation_module
import entropic_substrate.ladder.run.object_worker as object_worker_module
import entropic_substrate.ladder.run.runner as runner_module
import entropic_substrate.ladder.run.seals as seals_module
import entropic_substrate.ladder.run.span_diagnostics as span_module
import entropic_substrate.ladder.run.trace_capture as trace_module
import entropic_substrate.ladder.run.worker_execution as execution_module
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.gate.ladder_campaign import artifact_from_mapping
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.object_worker import run_object_worker
from entropic_substrate.ladder.run.seals import SealedStep, SealMemo, narrated_admission
from entropic_substrate.ladder.run.worker_execution import execute_probe_request
from entropic_substrate.ladder.run.worker_log_events import ADMISSION_CALLERS, ADMISSION_ROOTS
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit
LIMIT = 10_000_000


def _blank(value: object) -> bool:
    """True when a call would walk the full chain: no memo, or one holding nothing yet."""
    return value is None or not cast(SealMemo, value).populated


@dataclass
class _Walks:
    """Memo-less-or-empty walk ENTRIES, by the module whose binding was called."""

    armed: bool = False
    progress: list[str] = field(default_factory=list)
    resume: list[str] = field(default_factory=list)
    bare: list[str] = field(default_factory=list)


def _instrument(monkeypatch: pytest.MonkeyPatch) -> _Walks:
    walks = _Walks()

    def wrap(module: Any, name: str, label: str, sink: list[str], *, after_training: bool) -> None:
        original = getattr(module, name)

        def counted(*args: object, **kwargs: object) -> object:
            if (walks.armed or not after_training) and _blank(kwargs.get("memo")):
                sink.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    for module in (
        runner_module,
        execution_module,
        generation_module,
        span_module,
        trace_module,
        object_worker_module,
    ):
        wrap(
            module,
            "read_training_progress",
            module.__name__.rsplit(".", 1)[1],
            walks.progress,
            after_training=False,
        )
    for module in (generation_module, span_module):
        wrap(
            module,
            "resume_seal",
            module.__name__.rsplit(".", 1)[1],
            walks.resume,
            after_training=True,
        )
    wrap(span_module, "read_seal", "span_diagnostics", walks.bare, after_training=True)
    wrap(seals_module, "read_seal", "seals", walks.bare, after_training=True)

    trained = execution_module.run_probe_training

    def arming(*args: object, **kwargs: object) -> object:
        try:
            return trained(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            walks.armed = True

    monkeypatch.setattr(execution_module, "run_probe_training", arming)
    return walks


def test_final_leg_admits_each_artifact_root_with_one_fresh_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) The run root is walked ONCE (the completion walk), the capture root once."""
    request, root = prepared(tmp_path)
    walks = _instrument(monkeypatch)
    result = execute_probe_request(request, root)
    # Artifacts and status are unchanged by the reuse.
    assert result["status"] == "probe_workload_complete"
    assert result["scientific_authority"] == "none"
    assert result["successful_optimizer_steps"] == 6
    assert set(result) == {
        "schema",
        "request_sha256",
        "run_identity",
        "scientific_authority",
        "status",
        "runtime",
        "successful_optimizer_steps",
        "training",
        "generation",
        "diagnostics",
        "capture",
        "verification",
        "wall_seconds",
    }
    json.dumps(result)  # the memo never rides in the serialized result
    native = read_json(root, artifact_from_mapping(result["verification"]), max_bytes=LIMIT)
    assert native["status"] == "native_reader_passed"
    assert native["request_sha256"] == request.digest
    # HEADLINE: the /run completion walk plus the capture root's own fresh walk.
    assert walks.progress == ["runner", "trace_capture"], walks.progress
    # No stage after the completion walk re-walks through resume or a bare seal read.
    assert walks.resume == [] and walks.bare == [], (walks.resume, walks.bare)


def test_the_object_worker_result_admission_reuses_the_same_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) through ``run_object_worker``, where ``result_admission`` is reached.

    The worker always passes ``on_checkpoint``, so the training loop's FIRST
    per-checkpoint walk runs on its still-empty invocation memo: that one entry
    is the mid-training call (it admits a one-node chain, not the final leg) and
    is expected beside the two per-root fresh walks. ``result_admission`` itself
    must not appear, which is what a threaded memo means.
    """
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "object-worker"
    workspace.mkdir()
    walks = _instrument(monkeypatch)
    run_object_worker(objects, marker, workspace, max_request_bytes=1_000_000)
    assert walks.progress == ["runner", "runner", "trace_capture"], walks.progress
    assert "object_worker" not in walks.progress
    assert walks.resume == [] and walks.bare == [], (walks.resume, walks.bare)


def test_a_paused_leg_still_hands_its_walk_to_the_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, root = prepared(tmp_path, pause=("prediction", 2))
    walks = _instrument(monkeypatch)
    result = execute_probe_request(request, root)
    assert result["status"] == "paused"
    assert walks.progress == ["runner"], walks.progress


def _admission_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("event", "")).startswith("admission.")]


def test_one_admission_pair_per_caller_with_the_ruled_memo_provenance(tmp_path: Path) -> None:
    """(b) Exactly one start/end pair per caller; fresh only for completion and native."""
    log = tmp_path / "obs" / "worker-log.jsonl"
    bind_logs(log)
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    workspace = tmp_path / "object-worker"
    workspace.mkdir()
    before = len(read_log(log))
    run_object_worker(objects, marker, workspace, max_request_bytes=1_000_000)
    rows = read_log(log)[before:]
    starts, ends = of(rows, "admission.start"), of(rows, "admission.end")
    callers = [
        "completion",
        "training_readback",
        "generation",
        "span_diagnostics",
        "trace_capture",
        "native_verification",
        "result_admission",
    ]
    assert [r["caller"] for r in starts] == callers
    assert [r["caller"] for r in ends] == callers
    assert set(callers) == ADMISSION_CALLERS
    fresh = {r["caller"] for r in ends if r["memo"] == "fresh"}
    assert fresh == {"completion", "native_verification"}
    assert {r["caller"] for r in ends if r["memo"] == "shared"} == set(callers) - fresh
    for start, end in zip(starts, ends, strict=True):
        assert start["caller"] == end["caller"] and start["root"] == end["root"]
        assert start["memo"] == end["memo"]
        assert end["root"] == ("capture" if end["caller"] == "native_verification" else "run")
        assert end["root"] in ADMISSION_ROOTS
        assert isinstance(end["seconds"], float) and end["seconds"] >= 0.0
        # The start line carries only what is known before the walk.
        assert set(start) & {"seals_walked", "journals_read", "evaluations", "seconds"} == set()
    # Fresh walks read the whole chain (3 seals, 3 journals) and its 2 evaluations;
    # every shared stage reads nothing from disk for the seal chain.
    for end in ends:
        walked = (end["seals_walked"], end["journals_read"])
        assert walked == ((3, 3) if end["memo"] == "fresh" else (0, 0)), end
    assert next(r for r in ends if r["caller"] == "completion")["evaluations"] == 2
    # The pair brackets the stage it belongs to, never the whole worker.
    for caller in callers[1:]:
        assert (
            read_log_index(rows, "stage.start", caller)
            < read_log_index(rows, "admission.start", caller)
            < read_log_index(rows, "admission.end", caller)
            < read_log_index(rows, "stage.end", caller)
        )
    # The completion pair sits between the last evaluation and the final snapshot.
    assert read_log_index(rows, "evaluation.end", None, last=True) < read_log_index(
        rows, "admission.start", "completion"
    )


def read_log_index(
    rows: list[dict[str, Any]], event: str, caller: str | None, *, last: bool = False
) -> int:
    hits = [
        position
        for position, row in enumerate(rows)
        if row.get("event") == event
        and (caller is None or row.get("caller") == caller or row.get("stage") == caller)
    ]
    assert hits, f"no {event} row for {caller}"
    return hits[-1] if last else hits[0]


def test_a_paused_run_emits_the_completion_pair_and_the_readback_pair_only(
    tmp_path: Path,
) -> None:
    log = tmp_path / "obs" / "worker-log.jsonl"
    bind_logs(log)
    request, root = prepared(tmp_path, pause=("prediction", 2))
    execute_probe_request(request, root)
    rows = _admission_rows(read_log(log))
    assert [(r["event"], r["caller"], r["memo"]) for r in rows] == [
        ("admission.start", "completion", "fresh"),
        ("admission.end", "completion", "fresh"),
        ("admission.start", "training_readback", "shared"),
        ("admission.end", "training_readback", "shared"),
    ]


def test_mid_training_checkpoint_walks_emit_no_admission_event(tmp_path: Path) -> None:
    """A per-checkpoint call is not an admission site: nothing per checkpoint (C7)."""
    log = tmp_path / "obs" / "worker-log.jsonl"
    bind_logs(log)
    request, root = prepared(tmp_path, pause=("prediction", 2))
    execute_probe_request(request, root, on_checkpoint=lambda _artifact: None)
    callers = [r["caller"] for r in of(read_log(log), "admission.start")]
    assert callers == ["completion", "training_readback"]


def test_the_bracket_refuses_an_unknown_caller_or_a_path_as_root() -> None:
    with pytest.raises(ValueError, match="caller"):
        with narrated_admission("bogus", "run", SealMemo()):
            pass
    with pytest.raises(ValueError, match="root"):
        with narrated_admission("generation", "/home/operator/run", SealMemo()):
            pass


def test_the_bracket_logs_no_end_for_a_refused_walk_and_nothing_without_a_memo(
    tmp_path: Path,
) -> None:
    log = tmp_path / "obs" / "worker-log.jsonl"
    bind_logs(log)
    with narrated_admission("generation", "run", None):
        pass
    assert _admission_rows(read_log(log)) == []
    with pytest.raises(RuntimeError, match="refused"):
        with narrated_admission("generation", "run", SealMemo()):
            raise RuntimeError("refused")
    rows = _admission_rows(read_log(log))
    assert [r["event"] for r in rows] == ["admission.start"]


def test_memo_provenance_is_read_from_the_memo_state_at_entry(tmp_path: Path) -> None:
    log = tmp_path / "obs" / "worker-log.jsonl"
    bind_logs(log)
    memo = SealMemo()
    identity = ArtifactIdentity("a.json", ContentHash("sha256", "a" * 64))
    sealed = SealedStep("prediction", 1, identity, identity, 1, ("a" * 64,))
    with narrated_admission("generation", "run", memo):
        memo.remember(identity, sealed)  # populated DURING the bracket: still "fresh"
    with narrated_admission("generation", "run", memo):
        pass
    rows = read_log(log)
    assert [r["memo"] for r in of(rows, "admission.start")] == ["fresh", "shared"]
    assert [r["memo"] for r in of(rows, "admission.end")] == ["fresh", "shared"]


def test_the_vocabulary_declares_exactly_the_seven_callers_and_two_roots() -> None:
    assert ADMISSION_ROOTS == {"run", "capture"}
    assert len(ADMISSION_CALLERS) == 7
