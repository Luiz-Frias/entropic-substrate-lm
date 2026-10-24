"""H0 final-probe check suite (K1-K7): machine verdicts over a probe's evidence.

Unit tests only. Two evidence classes are kept apart: synthetic captured files built
here (low-entropy identities, no fixture), and receipts the REAL local training and
worker pipeline writes into ``tmp_path`` (CPU, tiny profile). The HEpP7g capture is
read by ``tests/test_h_final_probe_checks_hepp7g.py`` (integration), never here.

Verdicts are PASS, FAIL or UNPROVEN; only PASS is a pass. Each tamper case asserts the
check's OWN reason string, so a mutation that lets a reader refusal stand in for the
check's comparison still goes RED.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_aggregate import (
    IDENTITY_A,
    IDENTITY_B,
    REQUEST_A,
    REQUEST_B,
    _annotate,
    _eval_row,
    _log,
    _step_row,
    write_h_run,
)
from tests.test_h_probe_operator_stop import _prepared, _remote_probe_request, _seed_marker
from tests.test_sentence_execution import engine

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.analyze.h_aggregate import load_h_run
from entropic_substrate.ladder.analyze.h_final_probe_checks import (
    CheckResult,
    check_analysis_surface,
    check_capture,
    check_completion,
    check_generation,
    check_importance_curve,
    check_read_floor,
    check_resume_chain,
    check_resume_request,
    check_resume_seam,
    evaluated_samples_from_manifest,
)
from entropic_substrate.ladder.analyze.h_resumed_run import _stitch, load_h_resumed_run
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap
from entropic_substrate.ladder.run.h_probe_collection import collect_probe_terminal
from entropic_substrate.ladder.run.h_probe_resume import build_resume_request
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.object_worker import run_object_worker
from entropic_substrate.ladder.run.runner import run_probe_training
from entropic_substrate.ladder.run.worker_execution import execute_probe_request
from entropic_substrate.ladder.run.worker_request import ProbeRequest
from entropic_substrate.ladder.run.worker_transport import (
    materialize_probe_request,
    publish_probe_request,
)

pytestmark = pytest.mark.unit

LIMIT = 10_000_000
REQUEST_C = "1" * 63 + "c"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h0_final_probe_checks.py"


def _passes(result: CheckResult) -> None:
    assert result.verdict == "PASS", result.to_json()


def _fails_with(result: CheckResult, fragment: str) -> None:
    assert result.verdict == "FAIL", result.to_json()
    assert any(fragment in reason for reason in result.reasons), result.reasons


# ------------------------------------------------------------------ empty evidence (PR #374)
class TestEmptyEvidenceFails:
    """A requested check with no evidence must not pass (K7 already says "no arm was given")."""

    def test_k2_no_requests_fails(self) -> None:
        _fails_with(check_completion([], []), "no arm requests were given")

    def test_k3_no_metrics_files_fails(self) -> None:
        _fails_with(check_capture([], [REQUEST_A]), "no metrics files were given")

    def test_k3_no_request_digests_fails(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        _fails_with(check_capture([a[0]], []), "no request digests were given")

    def test_k4_no_worker_logs_fails(self) -> None:
        _fails_with(check_read_floor([]), "no worker logs were given")

    def test_k6_no_metrics_files_fails(self) -> None:
        _fails_with(check_importance_curve([], (2, 4)), "no metrics files were given")

    def test_k6_no_evaluated_samples_fails(self, tmp_path: Path) -> None:
        metrics, _ = write_h_run(tmp_path)
        _fails_with(check_importance_curve([metrics], ()), "no evaluated samples were given")

    @pytest.mark.parametrize(
        "plan",
        [
            {"k2": {"records": [], "requests": []}},
            {"k3": {"metrics": [], "request_digests": []}},
            {"k4": {"worker_logs": []}},
            {"k6": {"metrics": [], "evaluated_samples": []}},
        ],
    )
    def test_the_script_exits_non_zero_on_an_empty_evidence_plan(
        self, tmp_path: Path, plan: dict[str, object], capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = _script().main(["--plan", str(_plan(tmp_path, plan))])
        assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"
        assert code != 0


# ------------------------------------------------------------------ K1 chain (real receipts)
class _Chain:
    """A real stop-then-resume lineage with a checkpoint receipt between the declared
    stopped receipt and the completion receipt (every step seals)."""

    def __init__(self, root: Path) -> None:
        template = engine(root)
        schedule = replace(template.profile.schedule, checkpoint_interval=1)
        self.profile = replace(template.profile, schedule=schedule)
        self.source_root = template.source_root
        self.run_root = root / "run"
        self.run_root.mkdir()
        paused = run_probe_training(
            self.profile,
            self.source_root,
            self.run_root,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
            stop_after_seal=lambda phase, step: phase == "prediction" and step == 2,
        )
        assert paused.status == "paused"
        completed = run_probe_training(
            self.profile,
            self.source_root,
            self.run_root,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
            resume=paused.receipt,
        )
        assert completed.status == "training_complete"
        self.stopped = paused.receipt
        self.final = completed.receipt

    def check(
        self, final: ArtifactIdentity | None = None, stopped: ArtifactIdentity | None = None
    ) -> CheckResult:
        return check_resume_chain(
            self.run_root,
            self.final if final is None else final,
            self.stopped if stopped is None else stopped,
            self.profile,
            self.source_root,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )

    def rewrite(self, base: ArtifactIdentity, name: str, **changes: object) -> ArtifactIdentity:
        """A new receipt file: ``base``'s fields with ``changes`` applied."""
        row = read_json(self.run_root, base, max_bytes=LIMIT)
        row.update(changes)
        return write_json(self.run_root, name, row, max_bytes=LIMIT)


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> _Chain:
    return _Chain(tmp_path_factory.mktemp("k1-chain"))


class TestK1Chain:
    def test_walks_past_a_checkpoint_receipt_back_to_the_stopped_receipt(
        self, chain: _Chain
    ) -> None:
        result = chain.check()
        _passes(result)
        steps = result.facts["chain_optimizer_steps"]
        # completion 6 -> checkpoint 5 -> stopped 4 -> ... -> 1: the stopped receipt is
        # two hops back, so a one-hop walk cannot reach it
        assert steps == [6, 5, 4, 3, 2, 1], steps
        assert result.facts["stopped_hop"] == 2

    def test_a_declared_stopped_receipt_with_another_sha256_is_not_in_the_chain(
        self, chain: _Chain
    ) -> None:
        forged = ArtifactIdentity(chain.stopped.path, ContentHash("sha256", "f" * 64))
        _fails_with(chain.check(stopped=forged), "never reaches the declared stopped receipt")

    def test_a_chain_that_never_reaches_the_stopped_receipt_fails(self, chain: _Chain) -> None:
        absent = ArtifactIdentity("absent-progress.json", ContentHash("sha256", "e" * 64))
        _fails_with(chain.check(stopped=absent), "never reaches the declared stopped receipt")

    def test_a_tampered_link_fails_at_the_reader_naming_the_hash(self, chain: _Chain) -> None:
        reference = artifact_reference(chain.run_root, chain.stopped)
        tampered = chain.rewrite(
            chain.final,
            "tampered-link-progress.json",
            parent_receipt={"path": reference["path"], "sha256": "0" * 64},
        )
        _fails_with(chain.check(final=tampered), "hash mismatch")

    def test_a_foreign_run_identity_fails(self, chain: _Chain) -> None:
        foreign = chain.rewrite(chain.final, "foreign-progress.json", run_identity="a" * 64)
        _fails_with(chain.check(final=foreign), "construction or authority mismatch")

    def test_a_link_that_does_not_decrease_the_step_fails(self, chain: _Chain) -> None:
        # the reader admits a parent sealed at the SAME checkpoint; the check must not
        looped = chain.rewrite(
            chain.stopped,
            "looped-progress.json",
            parent_receipt=artifact_reference(chain.run_root, chain.stopped),
        )
        _fails_with(chain.check(final=looped), "not strictly decreasing")

    def test_a_final_receipt_that_is_not_training_complete_fails(self, chain: _Chain) -> None:
        _fails_with(chain.check(final=chain.stopped), "training_complete")

    def test_a_malformed_parent_receipt_is_a_hop_failure_not_a_traceback(
        self, chain: _Chain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the real reader refuses a malformed parent itself, so the check's own conversion
        # is only reachable through a reader that admitted the row: stub that seam
        row = {"successful_optimizer_steps": 6, "status": "training_complete"}
        monkeypatch.setattr(
            "entropic_substrate.ladder.analyze.h_final_probe_checks.read_training_progress",
            lambda *a, **k: {**row, "parent_receipt": {"path": 5}},
        )
        _fails_with(chain.check(), "hop 0")


# ------------------------------------------------------------------ K1 request (D4 builder)
def _collection(
    previous: ProbeRequest, resume: dict[str, str], **changes: object
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "h-probe-terminal-collection-v1",
        "scientific_authority": "none",
        "request_sha256": previous.digest,
        "capture_status": "stop_bytes_verified",
        "reported_outcome": "stopped_by_operator",
        "resume_artifact": resume,
    }
    row.update(changes)
    return row


class TestK1Request:
    RESUME: dict[str, str] = {  # noqa: RUF012 -- read-only test constant
        "path": "run/x-prediction-00000002-progress.json",
        "sha256": "5" * 64,
    }

    def _pair(self) -> tuple[dict[str, object], ProbeRequest, ProbeRequest]:
        previous = _remote_probe_request()
        resume = ArtifactIdentity(self.RESUME["path"], ContentHash("sha256", self.RESUME["sha256"]))
        resumed = replace(
            previous, inputs=(*previous.inputs, resume), nonce="4" * 64, resume=resume
        )
        return _collection(previous, dict(self.RESUME)), previous, resumed

    def test_a_request_resuming_the_collected_receipt_with_a_fresh_nonce_passes(self) -> None:
        _passes(check_resume_request(*self._pair()))

    def test_no_resume_fails(self) -> None:
        collection, previous, _ = self._pair()
        fresh = replace(previous, nonce="4" * 64)
        _fails_with(check_resume_request(collection, previous, fresh), "carries no resume")

    def test_a_different_resume_digest_fails(self) -> None:
        collection, previous, _ = self._pair()
        other = ArtifactIdentity(self.RESUME["path"], ContentHash("sha256", "6" * 64))
        resumed = replace(previous, inputs=(*previous.inputs, other), nonce="4" * 64, resume=other)
        _fails_with(check_resume_request(collection, previous, resumed), "resume differs")

    def test_the_same_nonce_fails(self) -> None:
        collection, previous, resumed = self._pair()
        _fails_with(
            check_resume_request(collection, previous, replace(resumed, nonce=previous.nonce)),
            "nonce",
        )

    def test_a_collection_of_another_request_fails(self) -> None:
        collection, previous, resumed = self._pair()
        collection["request_sha256"] = "7" * 64
        _fails_with(check_resume_request(collection, previous, resumed), "another request")


def _staged(collected_run: Path, worker_root: Path, staging: Path) -> Path:
    staging.mkdir()
    shutil.copytree(worker_root / "source", staging / "source")
    shutil.copytree(collected_run, staging / "run")
    return staging


class TestResumeRequestBuilder:
    def test_operator_stop_then_built_resume_request_completes_the_same_lineage(
        self, tmp_path: Path
    ) -> None:
        # pod A: the REAL worker, stopped by the operator marker at prediction step 2
        request, root = _prepared(tmp_path)
        objects = _seed_marker(request, appear_after=1)
        marker = publish_probe_request(objects, request, root)
        worker = tmp_path / "worker-a"
        worker.mkdir()
        run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
        collected = tmp_path / "collected-a"
        collected.mkdir()
        report = collect_probe_terminal(
            objects, request, collected, expected_pod_id=None, max_marker_bytes=1_000_000
        )
        assert report is not None and report["capture_status"] == "stop_bytes_verified"

        # the builder: existing `resume` field, serializer untouched, fresh nonce
        resumed = build_resume_request(request, report, collected / "run", nonce="2" * 64)
        assert resumed.resume is not None
        assert {
            "path": resumed.resume.path,
            "sha256": resumed.resume.content_hash.digest,
        } == report["resume_artifact"]
        assert resumed.profile == request.profile and resumed.pause is None
        _passes(check_resume_request(report, request, resumed))

        # pod B: the REAL worker admits the built request and completes the lineage
        staging = _staged(collected / "run", root, tmp_path / "staging")
        store = _ObjectStore()
        published = publish_probe_request(store, resumed, staging)
        destination = tmp_path / "worker-b"
        destination.mkdir()
        materialize_probe_request(store, published, destination, max_request_bytes=1_000_000)
        completed = execute_probe_request(resumed, destination)
        assert completed["status"] == "probe_workload_complete"
        training = completed["training"]
        assert isinstance(training, dict)
        final = ArtifactIdentity(
            Path(training["path"]).name, ContentHash("sha256", training["sha256"])
        )
        stopped_ref = report["resume_artifact"]
        assert isinstance(stopped_ref, dict)
        stopped = ArtifactIdentity(
            Path(stopped_ref["path"]).name, ContentHash("sha256", stopped_ref["sha256"])
        )
        _passes(
            check_resume_chain(
                destination / "run",
                final,
                stopped,
                request.profile,
                destination / "source",
                max_source_bytes=1_000_000,
                max_artifact_bytes=LIMIT,
            )
        )

    def test_refuses_a_collection_that_is_not_a_stop_or_pause(self, tmp_path: Path) -> None:
        previous = _remote_probe_request()
        report = _collection(
            previous,
            {"path": "run/x.json", "sha256": "5" * 64},
            capture_status="native_completion_receipt_verified",
        )
        with pytest.raises(ValueError, match="stop or pause"):
            build_resume_request(previous, report, tmp_path, nonce="2" * 64)

    def test_refuses_a_reused_nonce(self, tmp_path: Path) -> None:
        previous = _remote_probe_request()
        report = _collection(previous, {"path": "run/x.json", "sha256": "5" * 64})
        with pytest.raises(ValueError, match="fresh nonce"):
            build_resume_request(previous, report, tmp_path, nonce=previous.nonce)

    def test_refuses_a_resume_receipt_whose_bytes_differ(self, tmp_path: Path) -> None:
        previous = _remote_probe_request()
        receipt = write_json(tmp_path, "x.json", {"schema": "other"}, max_bytes=LIMIT)
        assert receipt.content_hash.digest != "5" * 64
        report = _collection(previous, {"path": "run/x.json", "sha256": "5" * 64})
        with pytest.raises(ValueError, match="differs from the collected bytes"):
            build_resume_request(previous, report, tmp_path, nonce="2" * 64)


# ------------------------------------------------------------------ K1 seam + D3 stitch reader
def _segment(
    directory: Path,
    tag: str,
    *,
    identity: str,
    request: str,
    warmup: Sequence[int],
    prediction: Sequence[int],
    result_status: str,
) -> tuple[Path, Path]:
    """One pod's captured files: the warm-up and prediction steps it committed."""
    metrics: list[dict[str, Any]] = [
        {
            "code_commit": "c" * 40,
            "event": "run_start",
            "image": "example.invalid/image@sha256:" + "4" * 64,
            "profile_identity": identity,
            "request_sha256": request,
            "run_class": "remote_tiny",
            "runtime": {"accelerator": {"count": 1, "model": "synthetic"}, "pod_id": tag},
            "schema": "h-run-lifecycle-v1",
        }
    ]
    log: list[dict[str, Any]] = [_log("worker.start", request_sha256=request)]
    for step in warmup:
        metrics.append(_step_row(identity, "codec_warmup", step, 1.0))
        if step == 2:
            metrics.append(_annotate(2, "1.5"))
            log.append(_log("checkpoint.seal", phase="codec_warmup", step=2, global_step=2))
    for step in prediction:
        metrics.append(_step_row(identity, "prediction", step, 1.0))
        if step % 2 == 0:
            metrics.append(_eval_row(identity, step, 1.0))
            metrics.append(_annotate(2 + step, "2.5"))
            log.append(_log("checkpoint.seal", phase="prediction", step=step, global_step=2 + step))
            log.append(_log("evaluation.start", global_step=2 + step))
    last = (2 + prediction[-1]) if prediction else warmup[-1]
    metrics.append(
        {
            "event": "run_end",
            "result_status": result_status,
            "schema": "h-run-lifecycle-v1",
            "status": "ok",
            "successful_optimizer_steps": last,
            "wall_seconds": 3.0,
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    pair = (directory / f"{tag}-metrics.jsonl", directory / f"{tag}-worker-log.jsonl")
    for path, rows in zip(pair, (metrics, log), strict=True):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return pair


def _resumed_pair(
    directory: Path,
    *,
    b_prediction: Sequence[int] = (3, 4),
    b_identity: str = IDENTITY_A,
    b_request: str = REQUEST_B,
    a_status: str = "stopped_by_operator",
) -> list[tuple[Path, Path]]:
    a = _segment(
        directory,
        "a",
        identity=IDENTITY_A,
        request=REQUEST_A,
        warmup=(1, 2),
        prediction=(1, 2),
        result_status=a_status,
    )
    b = _segment(
        directory,
        "b",
        identity=b_identity,
        request=b_request,
        warmup=(),
        prediction=b_prediction,
        result_status="probe_workload_complete",
    )
    return [a, b]


class TestStitchReader:
    def test_pod_b_alone_is_refused_by_the_single_run_reader(self, tmp_path: Path) -> None:
        # the reason the stitch reader exists (D3): no warm-up rows on the resumed pod
        _, b = _resumed_pair(tmp_path)
        with pytest.raises(TelemetryGap, match="committed codec_warmup rows number 0"):
            load_h_run(*b)

    def test_a_stopped_run_and_its_continuation_share_one_continuous_global_axis(
        self, tmp_path: Path
    ) -> None:
        stitched = load_h_resumed_run(_resumed_pair(tmp_path))
        run = stitched.run
        assert [step for step, _, _ in run.training] == [1, 2, 3, 4, 5, 6]
        assert [step for step, _ in run.evaluations] == [4, 6]
        assert run.warmup_steps == 2
        assert run.run_end["result_status"] == "probe_workload_complete"
        assert run.run_start["request_sha256"] == REQUEST_A
        assert run.request_sha256 == REQUEST_B
        assert [s.request_sha256 for s in stitched.segments] == [REQUEST_A, REQUEST_B]

    def test_a_segment_that_completed_cannot_be_resumed(self, tmp_path: Path) -> None:
        pair = _resumed_pair(tmp_path, a_status="probe_workload_complete")
        with pytest.raises(TelemetryGap, match="not a stop or pause"):
            load_h_resumed_run(pair)

    def test_the_same_request_twice_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TelemetryGap, match="request"):
            load_h_resumed_run(_resumed_pair(tmp_path, b_request=REQUEST_A))


class TestStitchEvaluationAxis:
    def test_a_repeated_evaluation_step_across_pods_is_refused(self, tmp_path: Path) -> None:
        a, b = load_h_resumed_run(_resumed_pair(tmp_path)).segments
        # pod B re-reports pod A's evaluation step; dict(arm.evaluations) would collapse it
        repeated = replace(b, evaluations=(a.evaluations[0], *b.evaluations))
        with pytest.raises(TelemetryGap, match=r"evaluation.*step 4.*not strictly increasing"):
            _stitch((a, repeated), a.warmup_steps)


class TestK1Seam:
    def test_contiguous_continuation_passes(self, tmp_path: Path) -> None:
        result = check_resume_seam(_resumed_pair(tmp_path))
        _passes(result)
        assert result.facts["segments"] == [[1, 4], [5, 6]]

    def test_a_one_step_gap_fails(self, tmp_path: Path) -> None:
        _fails_with(check_resume_seam(_resumed_pair(tmp_path, b_prediction=(4,))), "continuous")

    def test_an_overlap_fails(self, tmp_path: Path) -> None:
        _fails_with(
            check_resume_seam(_resumed_pair(tmp_path, b_prediction=(2, 3, 4))), "continuous"
        )

    def test_a_changed_profile_identity_fails(self, tmp_path: Path) -> None:
        _fails_with(
            check_resume_seam(_resumed_pair(tmp_path, b_identity=IDENTITY_B)), "profile identity"
        )


# ------------------------------------------------------------------ K2 completion on both arms
def _record(request: ProbeRequest, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": "h-probe-controller-result-v1",
        "scientific_authority": "none",
        "request_sha256": request.digest,
        "status": "terminal_collected",
        "workload_verified": True,
        "termination_verified": True,
        "collection": {"capture_status": "native_completion_receipt_verified"},
        "error_types": [],
    }
    row.update(changes)
    return row


def _arms() -> tuple[ProbeRequest, ProbeRequest]:
    magnitude = _remote_probe_request(nonce="3" * 64)
    profile = magnitude.profile
    sphere_profile = replace(profile, spine=replace(profile.spine, carrier="sphere"))
    return magnitude, replace(magnitude, profile=sphere_profile, nonce="4" * 64)


class TestK2:
    def test_both_arms_terminal_collected_and_verified_pass(self) -> None:
        a, b = _arms()
        result = check_completion([_record(a), _record(b)], [a, b])
        _passes(result)
        assert sorted(result.facts["carriers"]) == ["magnitude", "sphere"]

    def test_one_arm_missing_fails(self) -> None:
        a, b = _arms()
        _fails_with(check_completion([_record(a)], [a, b]), "no controller record")

    def test_controller_failed_fails(self) -> None:
        a, b = _arms()
        records = [_record(a), _record(b, status="controller_failed")]
        _fails_with(check_completion(records, [a, b]), "controller_failed")

    def test_non_empty_error_types_fails(self) -> None:
        a, b = _arms()
        records = [_record(a), _record(b, error_types=["RuntimeError"])]
        _fails_with(check_completion(records, [a, b]), "error_types")

    def test_a_non_string_request_digest_is_an_undeclared_request_not_a_traceback(self) -> None:
        a, b = _arms()
        records = [_record(a), _record(b), _record(a, request_sha256=["x"])]
        _fails_with(check_completion(records, [a, b]), "undeclared request")

    def test_two_arms_with_one_carrier_fail(self) -> None:
        a, _ = _arms()
        same = replace(a, nonce="5" * 64)
        _fails_with(check_completion([_record(a), _record(same)], [a, same]), "carrier")


# ------------------------------------------------------------------ K3 capture admissibility
class TestK3:
    def test_admissible_files_bound_to_the_declared_requests_pass(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        b = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b")
        _passes(check_capture([a[0], b[0]], [REQUEST_A, REQUEST_B]))

    def test_a_file_of_another_request_fails(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        _fails_with(check_capture([a[0]], [REQUEST_C]), "not a declared request")

    def test_an_inadmissible_file_fails_naming_it(self, tmp_path: Path) -> None:
        a = write_h_run(
            tmp_path,
            IDENTITY_A,
            REQUEST_A,
            tag="a",
            mutate_metrics=lambda rows: rows.pop(),  # drop run_end
        )
        _fails_with(check_capture([a[0]], [REQUEST_A]), str(a[0]))


# ------------------------------------------------------------------ K4 read floor
_FINAL_LEG = (
    "training_readback",
    "generation",
    "span_diagnostics",
    "trace_capture",
    "native_verification",
    "result_admission",
)


def _admission(caller: str, **changes: object) -> dict[str, Any]:
    root = "capture" if caller == "native_verification" else "run"
    memo = "fresh" if caller in ("completion", "native_verification") else "shared"
    walked = 4 if memo == "fresh" else 0
    row = _log(
        "admission.end",
        caller=caller,
        root=root,
        memo=memo,
        seals_walked=walked,
        journals_read=walked,
        evaluations=3 if memo == "fresh" else 0,
        evaluations_memo_hits=0 if memo == "fresh" else 3,
        seconds=1.5,
    )
    row.update(changes)
    return row


def _worker_log(
    directory: Path, mutate: Callable[[list[dict[str, Any]]], None] | None = None
) -> Path:
    rows: list[dict[str, Any]] = [_log("worker.start", request_sha256=REQUEST_A)]
    rows.append(_admission("completion"))
    for stage in _FINAL_LEG:
        rows.append(_log("stage.end", stage=stage, seconds=2.0))
        rows.append(_admission(stage))
    if mutate is not None:
        mutate(rows)
    path = directory / "worker-log.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _replace_admission(caller: str, **changes: object) -> Callable[[list[dict[str, Any]]], None]:
    def mutate(rows: list[dict[str, Any]]) -> None:
        for index, row in enumerate(rows):
            if row.get("event") == "admission.end" and row.get("caller") == caller:
                rows[index] = _admission(caller, **changes)

    return mutate


class TestK4:
    def test_shared_rows_reading_nothing_from_disk_pass(self, tmp_path: Path) -> None:
        result = check_read_floor([_worker_log(tmp_path)])
        _passes(result)
        fresh = result.facts["logs"][0]["fresh_walks"]
        assert {row["caller"] for row in fresh} == {"completion", "native_verification"}

    def test_memo_hits_are_not_disk_reads(self, tmp_path: Path) -> None:
        log = _worker_log(tmp_path, _replace_admission("generation", evaluations_memo_hits=9))
        _passes(check_read_floor([log]))

    def test_a_shared_row_reading_from_disk_fails_naming_the_stage(self, tmp_path: Path) -> None:
        log = _worker_log(tmp_path, _replace_admission("trace_capture", evaluations=9))
        _fails_with(check_read_floor([log]), "trace_capture read 9 evaluations from disk")

    def test_a_missing_admission_end_for_a_declared_stage_fails(self, tmp_path: Path) -> None:
        def drop(rows: list[dict[str, Any]]) -> None:
            rows[:] = [
                r
                for r in rows
                if not (r.get("event") == "admission.end" and r.get("caller") == "generation")
            ]

        _fails_with(check_read_floor([_worker_log(tmp_path, drop)]), "generation")

    def test_a_missing_count_fails_loud(self, tmp_path: Path) -> None:
        def drop_count(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                if row.get("event") == "admission.end" and row.get("caller") == "span_diagnostics":
                    del row["evaluations"]

        _fails_with(check_read_floor([_worker_log(tmp_path, drop_count)]), "does not project")

    def test_a_non_string_stage_is_a_malformed_row_not_a_traceback(self, tmp_path: Path) -> None:
        def bad_stage(rows: list[dict[str, Any]]) -> None:
            rows.append(_log("stage.end", stage=["completion"], seconds=1.0))

        _fails_with(check_read_floor([_worker_log(tmp_path, bad_stage)]), "malformed stage")

    def test_non_finite_seconds_fail_loud(self, tmp_path: Path) -> None:
        log = _worker_log(tmp_path, _replace_admission("generation", seconds=None))
        _fails_with(check_read_floor([log]), "does not project")

    def test_a_stopped_pod_needs_only_its_completion_admission(self, tmp_path: Path) -> None:
        def stopped(rows: list[dict[str, Any]]) -> None:
            rows[:] = rows[:2]

        _passes(check_read_floor([_worker_log(tmp_path, stopped)]))


# ------------------------------------------------------------------ K5 generation outputs
def _receipt(**summary: object) -> dict[str, object]:
    base: dict[str, object] = {
        "outcome": "sentence_budget",
        "sentences": ["ab", "c"],
        "observations": [97, 98, 256, 99, 256],
        "pending": [],
        "log_scores": [-1.0, -1.0, -1.0, -1.0, -1.0],
        "prompt": ["p"],
    }
    base.update(summary)
    return {
        "schema": "h-generation-receipt-v2",
        "run_identity": IDENTITY_A,
        "scientific_authority": "none",
        "status": "generation_complete",
        "summary": base,
    }


def _sampling(continuations: int) -> dict[str, object]:
    return {
        "sampler": "multinomial",
        "temperature": 1.0,
        "rng_digest": "8" * 64,
        "continuations": continuations,
    }


class TestK5:
    def test_boundary_terminated_receipt_v2_is_unproven_not_pass(self) -> None:
        result = check_generation([_receipt()])
        assert result.verdict == "UNPROVEN", result.to_json()
        assert result.facts["receipts"][0]["boundary"] == "PASS"
        assert result.facts["receipts"][0]["continuations"] == 1

    def test_a_document_end_outcome_passes_the_boundary_arm(self) -> None:
        receipt = _receipt(outcome="document_end", observations=[97, 98, 256, 99, 257])
        assert check_generation([receipt]).facts["receipts"][0]["boundary"] == "PASS"

    def test_sampling_law_with_several_continuations_proves_stochasticity(self) -> None:
        _passes(check_generation([{**_receipt(), "sampling": _sampling(3)}]))

    def test_one_continuation_with_a_sampling_law_is_still_unproven(self) -> None:
        receipt = {**_receipt(), "sampling": _sampling(1)}
        assert check_generation([receipt]).verdict == "UNPROVEN"

    def test_a_paused_outcome_fails(self) -> None:
        _fails_with(check_generation([_receipt(outcome="paused")]), "paused")

    def test_an_active_outcome_fails(self) -> None:
        _fails_with(check_generation([_receipt(outcome="active")]), "active")

    def test_a_sentence_budget_ending_on_a_byte_fails(self) -> None:
        receipt = _receipt(observations=[97, 98, 256, 99, 256, 100], pending=[100])
        _fails_with(check_generation([receipt]), "boundary")

    def test_boundaries_that_disagree_with_the_sentences_fail(self) -> None:
        _fails_with(check_generation([_receipt(sentences=["ab"])]), "boundary")

    def test_an_incomplete_receipt_fails(self) -> None:
        receipt = {**_receipt(), "status": "paused"}
        _fails_with(check_generation([receipt]), "generation_complete")


# FOR-1674 (receipt v3): per-continuation entries beside the aggregate summary. The boundary
# arm runs once per continuation over its own observation stream; stochasticity is PROVEN
# only with the top-level block, N > 1, a matching entry count and a per-entry law.
_V3_ROWS: list[tuple[str, list[str], list[int]]] = [
    ("sentence_budget", ["a"], [97, 256]),
    ("empty_sentence", [], [256]),
    ("length_limit", [], [97, 98, 99]),
    ("invalid_utf8", [], [200, 256]),
    ("document_end", ["b"], [98, 257]),
]


def _entry_v3(index: int, row: tuple[str, list[str], list[int]]) -> dict[str, object]:
    outcome, sentences, observations = row
    return {
        "index": index,
        "seed": 1000 + index,
        "sampling_law": {
            "sampler": "multinomial_temperature_1",
            "latent": "rsample",
            "seed": 1000 + index,
            "rng_digest": f"{index}" * 64,
        },
        "sentences": sentences,
        "log_scores": [-1.0] * len(observations),
        "outcome": outcome,
        "boundary": observations[-1] if observations[-1] in (256, 257) else None,
    }


def _receipt_v3(count: int = 5) -> dict[str, object]:
    rows = _V3_ROWS[:count]
    observations = [o for _, _, obs in rows for o in obs]
    return {
        "schema": "h-generation-receipt-v3",
        "run_identity": IDENTITY_A,
        "scientific_authority": "none",
        "status": "generation_complete",
        "summary": {
            "outcome": rows[-1][0],
            "sentences": [s for _, sentences, _ in rows for s in sentences],
            "observations": observations,
            "pending": [],
            "log_scores": [-1.0] * len(observations),
            "prompt": ["p"],
        },
        "continuations": [_entry_v3(i, row) for i, row in enumerate(rows)],
        "sampling": {
            "sampler": "multinomial_temperature_1",
            "temperature": 1.0,
            "continuations": count,
            "seed": 42,
            "rng_digest": "9" * 64,
        },
    }


class TestK5V3:
    def test_five_continuations_with_heterogeneous_outcomes_pass(self) -> None:
        # one sentence_budget, one empty_sentence, one length_limit, one invalid_utf8 and
        # one document_end: the boundary arm on the CONCATENATION would FAIL this receipt
        result = check_generation([_receipt_v3(5)])
        _passes(result)
        assert result.facts["receipts"][0]["continuations"] == 5

    def test_four_continuations_pass(self) -> None:
        _passes(check_generation([_receipt_v3(4)]))

    def test_the_aggregate_alone_would_fail_the_same_receipt(self) -> None:
        receipt = _receipt_v3(5)
        flat = {**receipt, "schema": "h-generation-receipt-v2"}
        assert check_generation([flat]).verdict == "FAIL"  # control: tiling is load-bearing

    def test_without_the_top_level_sampling_block_it_is_unproven(self) -> None:
        receipt = _receipt_v3(5)
        del receipt["sampling"]
        result = check_generation([receipt])
        assert result.verdict == "UNPROVEN", result.to_json()

    def test_dropping_the_per_continuation_law_is_unproven(self) -> None:
        receipt = _receipt_v3(5)
        for entry in receipt["continuations"]:  # type: ignore[attr-defined]
            del entry["sampling_law"]
        assert check_generation([receipt]).verdict == "UNPROVEN"

    def test_one_missing_per_continuation_law_is_unproven(self) -> None:
        receipt = _receipt_v3(5)
        receipt["continuations"][2]["sampling_law"] = None  # type: ignore[index]
        assert check_generation([receipt]).verdict == "UNPROVEN"

    def test_an_entry_count_that_differs_from_the_declared_count_is_never_pass(self) -> None:
        receipt = _receipt_v3(5)
        receipt["sampling"]["continuations"] = 4  # type: ignore[index]
        assert check_generation([receipt]).verdict != "PASS"

    def test_one_continuation_is_unproven(self) -> None:
        assert check_generation([_receipt_v3(1)]).verdict == "UNPROVEN"

    def test_log_scores_that_do_not_tile_the_observations_fail(self) -> None:
        receipt = _receipt_v3(5)
        receipt["summary"]["observations"].append(97)  # type: ignore[index]
        _fails_with(check_generation([receipt]), "tile")

    def test_a_continuation_ending_on_the_wrong_token_fails_by_index(self) -> None:
        receipt = _receipt_v3(5)
        receipt["continuations"][4]["outcome"] = "length_limit"  # type: ignore[index]
        _fails_with(check_generation([receipt]), "continuation 4")

    def test_a_v2_receipt_is_still_read(self) -> None:
        _passes(check_generation([{**_receipt(), "sampling": _sampling(3)}]))


# ------------------------------------------------------------------ K6 importance curve
def _with_counts(counts: dict[str, float]) -> Callable[[list[dict[str, Any]]], None]:
    """Rewrite every evaluation example to carry exactly ``counts`` (k -> ESS)."""

    def mutate(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("schema") == "h-predictive-evaluation-v1":
                entry = {"gap": None, "likelihood_relative_mcse": 0.5, "log_estimate": -10.0}
                row["examples"][0]["by_sample_count"] = {
                    k: [{**entry, "effective_sample_size": ess}] for k, ess in counts.items()
                }

    return mutate


class TestK6:
    def test_a_non_ascii_digit_key_is_a_malformed_entry_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        metrics, _ = write_h_run(tmp_path, mutate_metrics=_with_counts({"\u00b2": 1.5}))
        _fails_with(
            check_importance_curve([metrics], (2,)), "malformed sample-count entry '\u00b2'"
        )

    def test_an_over_long_digit_key_is_a_malformed_entry_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        metrics, _ = write_h_run(tmp_path, mutate_metrics=_with_counts({"9" * 5000: 1.5}))
        _fails_with(check_importance_curve([metrics], (2,)), "malformed sample-count entry")

    def test_an_empty_sample_count_entry_fails(self, tmp_path: Path) -> None:
        # the reader refuses an empty entry (h_aggregate "empty sample-count entry")
        def empty(rows: list[dict[str, Any]]) -> None:
            _with_counts({"2": 1.5})(rows)
            for row in rows:
                if row.get("schema") == "h-predictive-evaluation-v1":
                    row["examples"][0]["by_sample_count"] = {"2": []}

        metrics, _ = write_h_run(tmp_path, mutate_metrics=empty)
        _fails_with(check_importance_curve([metrics], (2,)), "empty sample-count entry")

    def test_the_admitted_gap_matches_the_readers(self) -> None:
        from entropic_substrate.ladder.analyze import h_aggregate, h_final_probe_checks

        assert h_final_probe_checks._ZERO_MASS_GAP == h_aggregate._ZERO_MASS_GAP  # pyright: ignore[reportPrivateUsage]

    def test_an_unknown_gap_value_fails(self, tmp_path: Path) -> None:
        # the reader admits only zero_importance_weight_mass (h_aggregate "unknown gap")
        def unknown(rows: list[dict[str, Any]]) -> None:
            _with_counts({"2": 1.5})(rows)
            for row in rows:
                if row.get("schema") == "h-predictive-evaluation-v1":
                    row["examples"][0]["by_sample_count"]["2"][0]["gap"] = "made_up_gap"

        metrics, _ = write_h_run(tmp_path, mutate_metrics=unknown)
        _fails_with(check_importance_curve([metrics], (2,)), "unknown gap 'made_up_gap'")

    def test_the_profile_k_set_passes(self, tmp_path: Path) -> None:
        metrics, _ = write_h_run(
            tmp_path, mutate_metrics=_with_counts({"2": 1.5, "4": 2.0, "8": 3.0})
        )
        result = check_importance_curve([metrics], (2, 4, 8))
        _passes(result)
        assert result.facts["rows"] == 2

    def test_a_missing_k_fails(self, tmp_path: Path) -> None:
        metrics, _ = write_h_run(tmp_path, mutate_metrics=_with_counts({"2": 1.5, "4": 2.0}))
        _fails_with(check_importance_curve([metrics], (2, 4, 8)), "k-set")

    def test_an_extra_k_fails(self, tmp_path: Path) -> None:
        metrics, _ = write_h_run(
            tmp_path, mutate_metrics=_with_counts({"2": 1.5, "4": 2.0, "8": 3.0, "16": 4.0})
        )
        _fails_with(check_importance_curve([metrics], (2, 4, 8)), "k-set")

    def test_an_ess_beyond_k_fails(self, tmp_path: Path) -> None:
        metrics, _ = write_h_run(
            tmp_path, mutate_metrics=_with_counts({"2": 2.5, "4": 2.0, "8": 3.0})
        )
        _fails_with(check_importance_curve([metrics], (2, 4, 8)), "outside [1, 2]")

    def test_the_expected_k_set_comes_from_the_profile_manifest(self) -> None:
        manifest = _remote_probe_request().profile.manifest
        assert evaluated_samples_from_manifest(manifest) == (2, 4, 8)


# ------------------------------------------------------------------ K7 analysis surface
class TestK7:
    def test_two_single_pod_arms_pass(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        b = write_h_run(tmp_path, IDENTITY_B, REQUEST_B, tag="b")
        _passes(check_analysis_surface({"magnitude": [a], "sphere": [b]}))

    def test_a_resumed_arm_is_admitted_through_the_stitch_reader(self, tmp_path: Path) -> None:
        resumed = _resumed_pair(tmp_path / "resumed")
        other = write_h_run(tmp_path, IDENTITY_B, REQUEST_C, tag="c")
        result = check_analysis_surface({"magnitude": resumed, "sphere": [other]})
        _passes(result)
        assert result.facts["paired_training_steps"] == 6

    def test_a_truncated_metrics_file_fails_naming_it(self, tmp_path: Path) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        text = a[0].read_text(encoding="utf-8")
        a[0].write_text(text[: len(text) // 2], encoding="utf-8")
        _fails_with(check_analysis_surface({"magnitude": [a]}), str(a[0]))


# ------------------------------------------------------------------ the script
def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("h0_final_probe_checks_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(directory: Path, plan: dict[str, object]) -> Path:
    path = directory / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


class TestScript:
    def test_all_requested_checks_pass_exits_zero_one_sorted_document_each(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        plan = {
            "k6": {"metrics": [str(a[0])], "evaluated_samples": [2, 4]},
            "k3": {"metrics": [str(a[0])], "request_digests": [REQUEST_A]},
        }
        code = _script().main(["--plan", str(_plan(tmp_path, plan))])
        lines = capsys.readouterr().out.splitlines()
        assert code == 0
        assert [json.loads(line)["check"] for line in lines] == ["k3", "k6"]
        for line in lines:
            assert line == json.dumps(json.loads(line), sort_keys=True)

    def test_unproven_is_not_a_pass(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        receipt = tmp_path / "generation.json"
        receipt.write_text(json.dumps(_receipt()), encoding="utf-8")
        plan = _plan(tmp_path, {"k5": {"receipts": [str(receipt)]}})
        code = _script().main(["--plan", str(plan)])
        assert json.loads(capsys.readouterr().out)["verdict"] == "UNPROVEN"
        assert code != 0

    def test_a_fail_exits_non_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        plan = {"k3": {"metrics": [str(a[0])], "request_digests": [REQUEST_C]}}
        code = _script().main(["--plan", str(_plan(tmp_path, plan))])
        assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"
        assert code != 0

    def test_an_unknown_check_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="k9"):
            _script().main(["--plan", str(_plan(tmp_path, {"k9": {}}))])

    @pytest.mark.parametrize(
        ("bad", "named"), [(8.9, "8.9"), (True, "True"), ("4", "'4'"), (None, "None")]
    )
    def test_k6_a_non_integer_sample_count_is_refused_naming_it(
        self, tmp_path: Path, bad: object, named: str
    ) -> None:
        a = write_h_run(tmp_path, IDENTITY_A, REQUEST_A, tag="a")
        plan = {"k6": {"metrics": [str(a[0])], "evaluated_samples": [2, bad]}}
        with pytest.raises(ValueError, match="evaluated_samples") as excinfo:
            _script().main(["--plan", str(_plan(tmp_path, plan))])
        assert named in str(excinfo.value)
