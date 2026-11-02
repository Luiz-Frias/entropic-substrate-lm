"""Graceful operator stop: R2 marker, `stopped` terminal, collection (FOR-1617).

Every test drives the REAL training/worker/collection pipeline; nothing here
replays a hand-built event. `_ObjectStore` subclasses below only control what
the CONTROL-KEY read returns or how many times it is called -- every other
key behaves exactly like the shared fake object store used across this
module's siblings.
"""

import io
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_probe_controller import Guard, Provider
from tests.test_h_probe_launch import launch
from tests.test_sentence_profile import profile
from tests.test_span_state_collection import model

from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.core.sentence.profile import RunSchedule
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.h_probe_collection import collect_probe_terminal
from entropic_substrate.ladder.run.h_probe_events import read_probe_event
from entropic_substrate.ladder.run.h_probe_process import ProbeWatchdogProcess
from entropic_substrate.ladder.run.journal import artifact_reference, write_json
from entropic_substrate.ladder.run.object_worker import (
    STOP_MARKER_SCHEMA,
    _event,
    classify_stop_marker,
    encode_stop_marker,
    run_object_worker,
    stop_marker_hits,
    stop_marker_key,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker.h_probe_controller import run_probe_controller
from entropic_substrate.ladder.run.worker.h_probe_stop import request_stop, run_probe_stop
from entropic_substrate.ladder.run.worker_request import ProbeRequest, TransferLimits
from entropic_substrate.ladder.run.worker_snapshot import publish_worker_snapshot
from entropic_substrate.ladder.run.worker_transport import (
    materialize_probe_request,
    publish_probe_request,
)

pytestmark = pytest.mark.unit

# warmup=2, prediction steps=4, checkpoint/eval every 2 -- boundaries at
# warmup step 2 (phase-final), prediction step 2 (mid-phase) and prediction
# step 4 (final + complete).
_DEFAULT = RunSchedule(2, 4, 2, 2, 2, 1, (2, 4), 1, 2, 2, diagnostic_samples=8)
# warmup=4 so warmup step 2 is a boundary strictly BEFORE warmup's own final
# step (4): the mid-warmup-phase case C13(f) needs.
_WIDE_WARMUP = RunSchedule(4, 4, 2, 2, 2, 1, (2, 4), 1, 2, 2, diagnostic_samples=8)
# checkpoint_interval=1: every committed step is a boundary, including step 1.
_EVERY_STEP = RunSchedule(2, 4, 2, 1, 2, 1, (2, 4), 1, 2, 2, diagnostic_samples=8)


def _prepared(
    root: Path, *, schedule: RunSchedule = _DEFAULT, nonce: str = "1" * 64
) -> tuple[ProbeRequest, Path]:
    template = model(root)
    profile = replace(template.profile, schedule=schedule)
    source = ArtifactIdentity(
        "source/prepared-corpus.json", ContentHash("sha256", profile.source_digest)
    )
    request = ProbeRequest(
        profile,
        (source,),
        nonce,
        TransferLimits(10_000_000, 50_000_000, 100),
        ProbeConfig(train_positions=8, eval_positions=8, updates=2, batch_size=4),
        None,
        None,
        None,
        None,
        None,
    )
    store = _ObjectStore()
    marker = publish_probe_request(store, request, root)
    worker = root / "worker"
    worker.mkdir()
    materialize_probe_request(store, marker, worker, max_request_bytes=1_000_000)
    return request, worker


def _remote_probe_request(
    *, schedule: RunSchedule = _DEFAULT, nonce: str = "3" * 64
) -> ProbeRequest:
    """A validly-shaped remote_tiny request; never used to run real numerics.

    ``HProfile`` requires CUDA/bf16 for any non-``local_tiny`` run class, so
    this fixture is only ever fed to :func:`_synthetic_stopped_terminal`
    (which never constructs a ``SentenceExecution``), not to the real
    training pipeline -- there is no CUDA device on this host.
    """
    base = replace(
        profile(), run_class="remote_tiny", device="cuda", precision="bf16", schedule=schedule
    )
    source = ArtifactIdentity(
        "source/prepared-corpus.json", ContentHash("sha256", base.source_digest)
    )
    bundle = ArtifactIdentity("code.bundle", ContentHash("sha256", "b" * 64))
    return ProbeRequest(
        base,
        (source, bundle),
        nonce,
        TransferLimits(10_000_000, 50_000_000, 100),
        ProbeConfig(train_positions=8, eval_positions=8, updates=2, batch_size=4),
        "c" * 40,
        bundle,
        "ghcr.io/luiz-frias/entropic-substrate-lm@sha256:" + "d" * 64,
        None,
        None,
    )


def _synthetic_stopped_terminal(
    objects: _ObjectStore,
    request: ProbeRequest,
    root: Path,
    *,
    steps: int,
    pod_id: str,
    receipt_schema: str = "h-probe-training-progress-v2",
    result_schema: str = "h-probe-worker-result-v1",
) -> None:
    """Publish a `stopped` event + snapshot shaped exactly as the worker would.

    Built with the SAME encoding/publication functions the worker uses
    (``write_json``, :func:`publish_worker_snapshot`, ``object_worker._event``)
    rather than hand-rolled JSON, so byte shape and canonical encoding match
    production -- only the (cheap, CPU-free) training receipt and result
    bodies are synthesized, never a real ``SentenceExecution``.
    """
    maximum = request.limits.max_object_bytes
    (root / "run").mkdir()
    (root / "worker").mkdir()
    body = {
        # FOR-1693 (house rule 4): this hand-written receipt was progress v1; the reader
        # now refuses v1 by name, so the fixture carries the v2 key set (null binding: a
        # single smoke cell, no hand-off).
        "schema": "h-probe-training-progress-v2",
        "run_identity": request.profile.identity,
        "run_class": request.profile.run_class,
        "scientific_authority": "none",
        "profile": None,
        "status": "paused",
        "phase": "prediction",
        "step": steps,
        "successful_optimizer_steps": steps,
        "own_optimizer_steps": steps,
        "inherited_warmup_steps": 0,
        "block": None,
        "pair": None,
        "cell": None,
        "authority_digest": None,
        "handoff": None,
        "last_seal": None,
        "evaluations": [],
        "parent_receipt": None,
        "inventory": None,
    }
    if receipt_schema == "h-probe-training-progress-v1":
        for key in (
            "own_optimizer_steps",
            "inherited_warmup_steps",
            "block",
            "pair",
            "cell",
            "authority_digest",
            "handoff",
        ):
            del body[key]
        body["schema"] = receipt_schema
    receipt = write_json(
        root,
        "run/progress.json",
        body,
        max_bytes=maximum,
    )
    runtime = {
        "kind": "remote_tiny",
        "pod_id": pod_id,
        "source_attestation": "exact_git_tree_match",
        "code_commit": request.code_commit,
        "requested_image": request.image,
        "accelerator": {"model": "fixture-gpu", "count": 1},
        "precision": "bf16",
    }
    result = write_json(
        root,
        "worker/result.json",
        {
            "schema": result_schema,
            "request_sha256": request.digest,
            "run_identity": request.profile.identity,
            "scientific_authority": "none",
            "status": "stopped_by_operator",
            "runtime": runtime,
            "successful_optimizer_steps": steps,
            "training": artifact_reference(root, receipt),
            "generation": None,
            "diagnostics": None,
            "capture": None,
            "verification": None,
            "wall_seconds": 0.01,
        },
        max_bytes=maximum,
    )
    snapshot = publish_worker_snapshot(
        objects, request, root, ("run/progress.json", "worker/result.json"), name="stopped"
    )
    _event(
        objects,
        request,
        "stopped",
        {
            "status": "stopped_by_operator",
            "result": artifact_reference(root, result),
            "snapshot": asdict(snapshot),
            "successful_optimizer_steps": steps,
        },
    )


@dataclass
class _MarkerStore(_ObjectStore):
    """Controls only what a read of ``control_key`` returns or does.

    ``control_payloads`` is consumed one entry per successful (non-raising)
    read; the last entry repeats once exhausted. ``raises`` counts down on
    each read of the control key before any payload is consulted, so a test
    can exhaust the whole :class:`VerificationReadRetry` budget.
    """

    control_key: str = ""
    control_payloads: tuple[bytes | None, ...] = ()
    raises: int = 0
    control_reads: int = 0

    def read(self, key: str) -> D1TrainerMailboxObjectRead | None:
        if key != self.control_key:
            return super().read(key)
        self.control_reads += 1
        if self.raises > 0:
            self.raises -= 1
            raise RuntimeError("injected transient control-key read failure")
        if not self.control_payloads:
            return None
        index = min(self.control_reads - 1, len(self.control_payloads) - 1)
        payload = self.control_payloads[index]
        if payload is None:
            return None
        return D1TrainerMailboxObjectRead(self._identity(key, payload), payload)


def _seed_marker(request: ProbeRequest, *, appear_after: int) -> _MarkerStore:
    """A store whose control-key read returns absent ``appear_after`` times,
    then the request's own genuine marker forever after."""
    payloads: tuple[bytes | None, ...] = (None,) * appear_after + (encode_stop_marker(request),)
    return _MarkerStore(control_key=stop_marker_key(request), control_payloads=payloads)


# ---------------------------------------------------------------------------
# Marker encode/classify (unit-level, no object store)
# ---------------------------------------------------------------------------


def test_stop_marker_is_a_pure_function_of_schema_and_digest(tmp_path: Path) -> None:
    request, _ = _prepared(tmp_path)
    payload = encode_stop_marker(request)
    row = json.loads(payload)
    assert set(row) == {"schema", "request_sha256"}
    assert row["schema"] == STOP_MARKER_SCHEMA
    assert row["request_sha256"] == request.digest
    assert encode_stop_marker(request) == payload  # deterministic, no `at`


def test_classify_stop_marker_ignores_a_foreign_request_and_fails_loud_otherwise(
    tmp_path: Path,
) -> None:
    request, _ = _prepared(tmp_path)
    assert classify_stop_marker(encode_stop_marker(request), request) is True
    other = replace(request, nonce="9" * 64)
    assert classify_stop_marker(encode_stop_marker(other), request) is False
    with pytest.raises(ValueError, match="schema"):
        classify_stop_marker(
            json.dumps({"schema": "h-probe-stop-v1", "request_sha256": "x"}).encode(), request
        )
    with pytest.raises(ValueError):  # json.JSONDecodeError is a ValueError
        classify_stop_marker(b"not json", request)


# ---------------------------------------------------------------------------
# Mid-phase stop, both phases (acceptance + C13(f))
# ---------------------------------------------------------------------------


def test_marker_mid_prediction_phase_stops_at_the_next_boundary_not_at_completion(
    tmp_path: Path,
) -> None:
    request, root = _prepared(tmp_path)
    # First boundary is warmup step 2; the marker appears only after that one
    # is read absent, so the run stops at prediction step 2 -- strictly
    # before prediction's own final step (4).
    objects = _seed_marker(request, appear_after=1)
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    terminal = run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    assert terminal.key == f"{request.prefix}/stopped.json"
    row = json.loads(objects.values[terminal.key])
    assert row["status"] == "stopped_by_operator"
    assert row["successful_optimizer_steps"] == 4  # 2 warmup + 2 prediction
    assert f"{request.prefix}/complete.json" not in objects.values
    event = read_probe_event(objects, request, "stopped", expected_pod_id=None, max_bytes=1_000_000)
    assert event is not None and event.reported_optimizer_steps == 4
    assert objects.control_reads == 2  # warmup-2 (absent), prediction-2 (present)

    # ACCEPTANCE bullet 1: the `stopped` receipt names the last committed
    # global step AND every sealed artifact key -- journal, seal, progress,
    # checkpoint .pt for the last committed step (both phases this run
    # passed through: warmup sealed once at its own final step, prediction
    # sealed at ITS last committed step), the run profile, the worker
    # start/result records, and the evaluation report produced at the stop
    # boundary. `runner.py` mints a fresh `uuid4().hex` receipt prefix once
    # per run and reuses it across every one of that run's filenames, so it
    # cannot be hardcoded here -- it is recovered from the snapshot's own
    # `codec_warmup` seal path, the one path shape unique to a single file.
    assert event.snapshot is not None
    manifest = json.loads(objects.values[event.snapshot.key])
    paths = {entry["path"] for entry in manifest["files"]}
    receipt_id = next(
        path.removeprefix("run/").removesuffix("-codec_warmup-00000002-seal.json")
        for path in paths
        if path.endswith("-codec_warmup-00000002-seal.json")
    )
    expected_paths = {
        f"run/{receipt_id}-codec_warmup-00000002-journal.json",
        f"run/{receipt_id}-codec_warmup-00000002-progress.json",
        f"run/{receipt_id}-codec_warmup-00000002-seal.json",
        f"run/{receipt_id}-codec_warmup-00000002.pt",
        f"run/{receipt_id}-prediction-00000002-journal.json",
        f"run/{receipt_id}-prediction-00000002-progress.json",
        f"run/{receipt_id}-prediction-00000002-seal.json",
        f"run/{receipt_id}-prediction-00000002.pt",
        f"run/{receipt_id}-00000002-evaluation.json",  # evaluation sealed at step 2
        "run/profile.json",
        "worker/result.json",
        "worker/start.json",
    }
    assert paths == expected_paths  # exactly 12 keys, no more, no fewer
    keys = {entry["key"] for entry in manifest["files"]}
    assert keys == {f"{request.prefix}/outputs/{path}" for path in expected_paths}


def test_marker_mid_warmup_phase_stops_strictly_before_warmups_final_step(
    tmp_path: Path,
) -> None:
    request, root = _prepared(tmp_path, schedule=_WIDE_WARMUP)
    # Boundaries: warmup step 2 (mid-phase, not final -- warmup_steps=4).
    objects = _seed_marker(request, appear_after=0)
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    row = json.loads(objects.values[f"{request.prefix}/stopped.json"])
    assert row["successful_optimizer_steps"] == 2
    receipt_key = next(
        k for k in objects.values if k.endswith("-codec_warmup-00000002-progress.json")
    )
    assert receipt_key  # sealed at warmup step 2, never reached step 4


def test_marker_present_before_step_one_stops_after_sealing_step_one(tmp_path: Path) -> None:
    request, root = _prepared(tmp_path, schedule=_EVERY_STEP)
    objects = _seed_marker(request, appear_after=0)
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    row = json.loads(objects.values[f"{request.prefix}/stopped.json"])
    assert row["successful_optimizer_steps"] == 1
    assert objects.control_reads == 1


def test_marker_at_a_checkpoint_step_seals_once_not_twice(tmp_path: Path) -> None:
    request, root = _prepared(tmp_path)
    objects = _seed_marker(request, appear_after=1)  # fires at prediction step 2
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    # `objects.values` is a dict: a key it holds can never count more than
    # once, so counting keys there cannot distinguish "sealed once" from
    # "sealed twice, second write collapsed the first". `objects.publishes`
    # is the append-only call log (test_d1_trainer_object_transport.py:73)
    # and is the only thing that can actually witness a double publish.
    assert objects.publishes.count(f"{request.prefix}/checkpoint-00000004.json") == 1
    assert (
        json.loads(objects.values[f"{request.prefix}/checkpoint-00000004.json"])[
            "successful_optimizer_steps"
        ]
        == 4
    )
    # Never reached the checkpoint the completing run would have sealed.
    assert f"{request.prefix}/checkpoint-00000006.json" not in objects.values


def test_marker_for_another_request_is_ignored_run_completes(tmp_path: Path) -> None:
    request, root = _prepared(tmp_path)
    other = replace(request, nonce="9" * 64)
    objects = _MarkerStore(
        control_key=stop_marker_key(request), control_payloads=(encode_stop_marker(other),)
    )
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    terminal = run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    assert terminal.key == f"{request.prefix}/complete.json"
    assert json.loads(objects.values[terminal.key])["status"] == "probe_workload_complete"
    assert f"{request.prefix}/stopped.json" not in objects.values


# ---------------------------------------------------------------------------
# Fail-loud paths (C4, C13(c)/(d))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [b"not json at all", json.dumps({"schema": "h-probe-stop-v1", "request_sha256": "x"}).encode()],
    ids=["unparseable_bytes", "foreign_schema"],
)
def test_unparseable_or_foreign_schema_marker_fails_loud_with_a_failure_snapshot(
    tmp_path: Path, payload: bytes
) -> None:
    request, root = _prepared(tmp_path)
    objects = _MarkerStore(control_key=stop_marker_key(request), control_payloads=(payload,))
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    with pytest.raises(ValueError):  # json.JSONDecodeError is a ValueError
        run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["status"] == "execution_defect"
    assert failure["snapshot"] is not None
    assert f"{request.prefix}/stopped.json" not in objects.values
    assert f"{request.prefix}/complete.json" not in objects.values


def test_control_key_read_error_retries_then_fails_loud_after_the_budget(tmp_path: Path) -> None:
    request, root = _prepared(tmp_path)
    objects = _MarkerStore(control_key=stop_marker_key(request), raises=10_000)
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    with pytest.raises(RuntimeError, match="injected transient control-key read failure"):
        run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    assert objects.control_reads == 3  # VerificationReadRetry().attempts, no numeric literal here
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["status"] == "execution_defect" and failure["error_type"] == "RuntimeError"
    assert failure["snapshot"] is not None


def test_reads_of_the_control_key_equal_seal_boundaries_reached(tmp_path: Path) -> None:
    request, root = _prepared(tmp_path)  # boundaries: warmup-2, prediction-2, prediction-4
    objects = _seed_marker(request, appear_after=1_000)  # never present
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    assert objects.control_reads == 3


# ---------------------------------------------------------------------------
# Evaluation report present in its own seal's snapshot (scope item 5 -- pin
# "already true at the epoch")
# ---------------------------------------------------------------------------


def test_evaluation_report_is_present_in_the_snapshot_published_by_its_own_seal(
    tmp_path: Path,
) -> None:
    request, root = _prepared(tmp_path)
    objects = _ObjectStore()
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    snapshot = json.loads(objects.values[f"{request.prefix}/snapshots/checkpoint-00000004.json"])
    paths = {entry["path"] for entry in snapshot["files"]}
    assert any(path.startswith("run/") and path.endswith("-evaluation.json") for path in paths)


# ---------------------------------------------------------------------------
# Collection (acceptance + guard-teeth: "collect stopped with workload_verified true")
# ---------------------------------------------------------------------------


def test_collection_of_a_stopped_terminal_has_workload_verified_false(tmp_path: Path) -> None:
    request, root = _prepared(tmp_path)
    objects = _seed_marker(request, appear_after=1)
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    destination = tmp_path / "collected"
    destination.mkdir()
    report = collect_probe_terminal(
        objects, request, destination, expected_pod_id=None, max_marker_bytes=1_000_000
    )
    assert report is not None
    assert report["workload_verified"] is False
    assert report["reported_outcome"] == "stopped_by_operator"
    assert report["reported_optimizer_steps"] == 4
    assert report["native_validation"] == "not_replayed_by_host"
    assert report["capture_status"] == "stop_bytes_verified"


def test_collection_refuses_a_v1_progress_receipt_by_name_and_a_result_v2_for_a_single_profile(
    tmp_path: Path,
) -> None:
    """FOR-1693: the collection of a stopped terminal reads progress v2; a v1 receipt is refused
    BY NAME (the minimum supported version is named), and a pair's result v2 is refused for a
    single-profile request."""
    request = _remote_probe_request()
    for tag, options, message in (
        (
            "v1-receipt",
            {"receipt_schema": "h-probe-training-progress-v1"},
            "minimum supported version is h-probe-training-progress-v2",
        ),
        (
            "pair-result-v2",
            {"result_schema": "h-probe-worker-result-v2"},
            r"names a pair's cells.*admits h-probe-worker-result-v1",
        ),
    ):
        objects = _ObjectStore()
        fixture_root = tmp_path / tag
        fixture_root.mkdir()
        _synthetic_stopped_terminal(
            objects, request, fixture_root, steps=4, pod_id="owned-pod", **options
        )
        destination = tmp_path / f"collected-{tag}"
        destination.mkdir()
        with pytest.raises(ValueError, match=message):
            collect_probe_terminal(
                objects,
                request,
                destination,
                expected_pod_id="owned-pod",
                max_marker_bytes=1_000_000,
            )
    ok = _ObjectStore()
    good_root = tmp_path / "good"
    good_root.mkdir()
    _synthetic_stopped_terminal(ok, request, good_root, steps=4, pod_id="owned-pod")
    good_destination = tmp_path / "collected-good"
    good_destination.mkdir()
    report = collect_probe_terminal(
        ok, request, good_destination, expected_pod_id="owned-pod", max_marker_bytes=1_000_000
    )
    assert report is not None and report["capture_status"] == "stop_bytes_verified"


# ---------------------------------------------------------------------------
# Controller: a stopped terminal is collected as terminated (C10)
# ---------------------------------------------------------------------------


def test_controller_collects_a_stopped_terminal_as_terminated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _remote_probe_request()
    objects = _ObjectStore()
    fixture_root = tmp_path / "worker-fixture"
    fixture_root.mkdir()
    _synthetic_stopped_terminal(objects, request, fixture_root, steps=4, pod_id="owned-pod")
    assert f"{request.prefix}/stopped.json" in objects.values

    guard = Guard(tmp_path)
    provider = Provider(guard)
    guard.launch = SimpleNamespace(
        request=request,
        digest=guard.launch.digest,
        pod_name=guard.launch.pod_name,
        # Rung B: the controller asks every launch to bound its rate by the run
        # authority; the real smoke launch's answer (no authority, no cap) is delegated.
        require_hourly_rate_within_cap=guard.launch.require_hourly_rate_within_cap,
        # FOR-1694: and the one ceiling its price check reads (the flag, for a smoke launch).
        hourly_ceiling=guard.launch.hourly_ceiling,
    )
    calls: list[str] = []

    def allocate(*_args: object) -> object:
        calls.append("create")
        guard.control.records["create-intent"] = {"guard_pid": guard.pid}
        guard.control.records["allocation"] = {"pod_id": provider.pod.pod_id}
        return provider.pod

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.allocate_probe_once", allocate
    )
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_controller.read_probe_control",
        lambda _root, _spec, _role, name, **_kw: guard.control.records.get(name),
    )
    destination = tmp_path / "capture"
    destination.mkdir()
    clock = [0.0]
    result = run_probe_controller(
        cast(ProbeWatchdogProcess, guard),
        provider,
        cast(D1TrainerMailboxObjectPort, objects),
        destination,
        poll_seconds=1,
        max_hourly_rate=1,
        monotonic=lambda: clock[0],
        wait=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    assert calls == ["create"]
    assert result["status"] == "terminal_collected"
    assert result["workload_verified"] is False
    assert result["termination_verified"] is True
    assert result["collection"]["reported_outcome"] == "stopped_by_operator"
    assert guard.control.records["stop"]["reason"] == "partial_capture"


# ---------------------------------------------------------------------------
# The verb (C11 + guard-teeth: "return from the verb before the event appears")
# ---------------------------------------------------------------------------


def test_verb_publishes_the_marker_once_and_waits_for_the_stopped_event(
    tmp_path: Path,
) -> None:
    request, root = _prepared(tmp_path)
    objects = _ObjectStore()
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()

    clock = [0.0]
    calls: list[float] = []

    def monotonic() -> float:
        return clock[0]

    def sleep(seconds: float) -> None:
        # Run the worker to completion on the FIRST poll tick, exactly once
        # -- mirrors an operator's stop arriving mid-run.
        calls.append(seconds)
        if len(calls) == 1:
            run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
        clock[0] += seconds

    keys = request_stop(
        objects, request, wait_seconds=60.0, poll_seconds=5.0, monotonic=monotonic, sleep=sleep
    )
    assert keys is not None
    assert f"{request.prefix}/snapshots/stopped.json" in keys
    assert any(key.endswith("worker/result.json") for key in keys)
    # Idempotent republish: a second call with the SAME payload is not an error.
    second = request_stop(
        objects, request, wait_seconds=1.0, poll_seconds=1.0, monotonic=monotonic, sleep=sleep
    )
    assert second == keys


def test_verb_returns_none_at_its_bound_when_the_event_never_appears(tmp_path: Path) -> None:
    request, _ = _prepared(tmp_path)
    objects = _ObjectStore()
    clock = [0.0]
    ticks: list[float] = []
    result = request_stop(
        objects,
        request,
        wait_seconds=10.0,
        poll_seconds=1.0,
        monotonic=lambda: clock[0],
        sleep=lambda seconds: (ticks.append(seconds), clock.__setitem__(0, clock[0] + seconds)),
    )
    assert result is None
    assert ticks  # it actually polled, never returned before the bound
    assert json.loads(objects.values[stop_marker_key(request)]) == json.loads(
        encode_stop_marker(request)
    )


# ---------------------------------------------------------------------------
# The verb's CLI entrypoint (ACCEPTANCE bullet 4: exits non-zero at its bound
# without the event; exits zero with the keys once the event exists).
# `build_worker_objects` builds a live R2 client from process credentials, so
# every test here substitutes it -- never lets it run against production R2.
# ---------------------------------------------------------------------------


def _stop_argv(tmp_path: Path, *, wait_seconds: str, poll_seconds: str) -> list[str]:
    (tmp_path / "launch.json").write_bytes(launch().payload)
    return [
        "--launch-root",
        str(tmp_path),
        "--launch-file",
        "launch.json",
        "--repo-root",
        str(tmp_path),
        "--max-launch-bytes",
        "1000000",
        "--wait-seconds",
        wait_seconds,
        "--poll-seconds",
        poll_seconds,
    ]


def test_verb_cli_exits_nonzero_without_the_event_at_its_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = _ObjectStore()
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_stop.build_worker_objects",
        lambda *_a, **_kw: objects,
    )
    argv = _stop_argv(tmp_path, wait_seconds="0.05", poll_seconds="0.01")
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_probe_stop(argv, environ={}, stdout=stdout, stderr=stderr)
    assert code == 1
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "status": "stop_not_confirmed",
        "scientific_authority": "none",
    }
    # The marker publish is a single call, and the fake store's own PUT is
    # idempotent-but-tracked, so its append-only call log is the only witness
    # that the verb published the marker exactly once (never per poll tick).
    assert objects.publishes.count(stop_marker_key(launch().request)) == 1


def test_verb_cli_exits_zero_and_prints_keys_once_the_stopped_event_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = launch().request
    objects = _ObjectStore()
    fixture_root = tmp_path / "worker-fixture"
    fixture_root.mkdir()
    _synthetic_stopped_terminal(objects, request, fixture_root, steps=4, pod_id="owned-pod")
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_stop.build_worker_objects",
        lambda *_a, **_kw: objects,
    )
    argv = _stop_argv(tmp_path, wait_seconds="5", poll_seconds="1")
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_probe_stop(argv, environ={}, stdout=stdout, stderr=stderr)
    assert code == 0
    assert stderr.getvalue() == ""
    printed = json.loads(stdout.getvalue())
    assert printed["status"] == "stopped" and printed["scientific_authority"] == "none"
    manifest = json.loads(objects.values[f"{request.prefix}/snapshots/stopped.json"])
    expected_keys = sorted(str(entry["key"]) for entry in manifest["files"])
    assert sorted(printed["keys"]) == sorted(
        (*expected_keys, f"{request.prefix}/snapshots/stopped.json")
    )


# ---------------------------------------------------------------------------
# Target seal (the marker names WHICH seal ends the run)
# ---------------------------------------------------------------------------


def test_untargeted_marker_bytes_are_unchanged_and_read_old_bytes_still_classify(
    tmp_path: Path,
) -> None:
    request, _ = _prepared(tmp_path)
    legacy = json.dumps(
        {"request_sha256": request.digest, "schema": STOP_MARKER_SCHEMA},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert encode_stop_marker(request) == legacy
    assert classify_stop_marker(legacy, request) is True
    assert stop_marker_hits(legacy, request, ("warmup_phase_any", 1)) is True


def test_targeted_marker_hits_only_at_its_seal(tmp_path: Path) -> None:
    request, _ = _prepared(tmp_path)
    payload = encode_stop_marker(request, target=("prediction", 4))
    assert json.loads(payload)["target"] == {"phase": "prediction", "step": 4}
    assert classify_stop_marker(payload, request) is True
    assert stop_marker_hits(payload, request, ("prediction", 4)) is True
    assert stop_marker_hits(payload, request, ("prediction", 2)) is False
    assert stop_marker_hits(payload, request, ("codec_warmup", 4)) is False
    other = replace(request, nonce="9" * 64)
    assert (
        stop_marker_hits(
            encode_stop_marker(other, target=("prediction", 4)), request, ("prediction", 4)
        )
        is False
    )


@pytest.mark.parametrize(
    "target",
    [
        ("", 2),
        ("prediction", 0),
        ("prediction", -1),
        ("prediction", True),
        ("prediction", "2"),
        (7, 2),
    ],
    ids=["empty_phase", "zero_step", "negative_step", "bool_step", "str_step", "int_phase"],
)
def test_malformed_target_is_refused_at_encode_and_at_classify(
    tmp_path: Path, target: tuple[object, object]
) -> None:
    request, _ = _prepared(tmp_path)
    with pytest.raises(ValueError, match="target"):
        encode_stop_marker(request, target=cast("tuple[str, int]", target))
    for bad in (
        {"phase": target[0], "step": target[1]},
        {"phase": "prediction"},
        {"phase": "prediction", "step": 2, "extra": 1},
        "prediction:2",
    ):
        raw = json.dumps(
            {"schema": STOP_MARKER_SCHEMA, "request_sha256": request.digest, "target": bad},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        with pytest.raises(ValueError, match="target"):
            classify_stop_marker(raw, request)


def test_targeted_marker_present_from_the_start_skips_earlier_seals(tmp_path: Path) -> None:
    request, root = _prepared(tmp_path)
    payload = encode_stop_marker(request, target=("prediction", 2))
    objects = _MarkerStore(control_key=stop_marker_key(request), control_payloads=(payload,))
    marker_id = publish_probe_request(objects, request, root)
    worker = tmp_path / "worker-run"
    worker.mkdir()
    run_object_worker(objects, marker_id, worker, max_request_bytes=1_000_000)
    row = json.loads(objects.values[f"{request.prefix}/stopped.json"])
    assert row["status"] == "stopped_by_operator"
    # warmup-2 seal was polled and passed; the stop landed at prediction-2.
    assert row["successful_optimizer_steps"] == 4
    assert objects.control_reads == 2


def test_verb_publishes_the_targeted_marker(tmp_path: Path) -> None:
    request, _ = _prepared(tmp_path)
    objects = _ObjectStore()
    clock = [0.0]
    result = request_stop(
        objects,
        request,
        target=("prediction", 2),
        wait_seconds=1.0,
        poll_seconds=1.0,
        monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    assert result is None
    assert objects.values[stop_marker_key(request)] == encode_stop_marker(
        request, target=("prediction", 2)
    )


def test_cli_requires_both_or_neither_of_at_phase_and_at_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = _ObjectStore()
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_stop.build_worker_objects",
        lambda *_a, **_kw: objects,
    )
    base = _stop_argv(tmp_path, wait_seconds="0.05", poll_seconds="0.01")
    for extra in (["--at-phase", "prediction"], ["--at-step", "2"]):
        code = run_probe_stop(
            [*base, *extra], environ={}, stdout=io.StringIO(), stderr=io.StringIO()
        )
        assert code == 2  # argparse refusal, before any object is published
    assert objects.publishes == []
    code = run_probe_stop(
        [*base, "--at-phase", "prediction", "--at-step", "2"],
        environ={},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert code == 1  # unconfirmed at the bound; the marker WAS published
    request = launch().request
    assert objects.values[stop_marker_key(request)] == encode_stop_marker(
        request, target=("prediction", 2)
    )


@pytest.mark.parametrize(
    ("phase", "step_from"),
    [
        ("prediction", "steps+1"),
        ("codec_warmup", "warmup+1"),
        ("predicton", "one"),
        ("evaluation", "one"),
    ],
)
def test_target_outside_the_request_schedule_is_refused_at_encode(
    tmp_path: Path, phase: str, step_from: str
) -> None:
    """A target the runner can never reach must fail at the verb, not time out.

    The marker is immutable once published (byte-identical republish only), so a
    bad target cannot be corrected for the same request; the rule mirrors
    ``ProbeRequest.pause``: phase in {codec_warmup, prediction}, 1 <= step <= that
    phase's budget. The worker-side reader stays unchanged (never raises).
    """
    request, _ = _prepared(tmp_path)
    schedule = request.profile.schedule
    step = {
        "steps+1": schedule.steps + 1,
        "warmup+1": schedule.warmup_steps + 1,
        "one": 1,
    }[step_from]
    with pytest.raises(ValueError, match="outside the request schedule"):
        encode_stop_marker(request, target=(phase, step))
    # Control: the largest in-range step of each real phase still encodes.
    encode_stop_marker(request, target=("prediction", schedule.steps))
    encode_stop_marker(request, target=("codec_warmup", schedule.warmup_steps))


def test_in_budget_step_that_is_not_a_seal_is_refused_at_encode(tmp_path: Path) -> None:
    """Review round 2: step 1 is inside the budget but never polled; refuse it at the verb."""
    request, _ = _prepared(tmp_path)
    schedule = request.profile.schedule
    assert schedule.checkpoint_interval > 1  # the fixture must have a non-seal step to test
    with pytest.raises(ValueError, match="outside the request schedule"):
        encode_stop_marker(request, target=("prediction", 1))


def test_target_rule_equals_the_pause_rule_on_every_coordinate(tmp_path: Path) -> None:
    """Parity pin: the marker target accepts exactly the coordinates ``ProbeRequest.pause``
    accepts, over every (phase, step) up to one past each budget plus an unknown phase."""
    request, _ = _prepared(tmp_path)
    schedule = request.profile.schedule
    grid = [
        (phase, step)
        for phase, budget in (
            ("codec_warmup", schedule.warmup_steps),
            ("prediction", schedule.steps),
            ("evaluation", schedule.steps),
        )
        for step in range(1, budget + 2)
    ]
    assert len(grid) >= 6
    for coordinate in grid:
        try:
            replace(request, pause=coordinate)
            pause_ok = True
        except ValueError:
            pause_ok = False
        try:
            encode_stop_marker(request, target=coordinate)
            target_ok = True
        except ValueError:
            target_ok = False
        assert target_ok is pause_ok, coordinate
