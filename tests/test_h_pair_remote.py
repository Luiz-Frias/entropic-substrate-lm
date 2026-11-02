"""FOR-1694 (h1-l4c): carry a pair through the remote path, on the L4b pair fixture.

The scientific pair (a ``pilot`` launch under one sealed authority) cannot EXECUTE on a CPU, so
the launch half runs over a pilot pair built from the same two profiles and is proven at the
construction, parse and digest layers; the host half (collection, resume, wave monitor, stop
verb, K3, the host verifier) runs over the real ``local_tiny`` pair the L4b worker executes on
CPU, read through a fake object store whose every publish and read is counted at the port.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_codec_handoff import Pair, make_pair
from tests.test_h_pair_request import (
    BLOCK,
    LIMITS,
    PROBE,
    pair_world,
    run_command,
)
from tests.test_h_probe_launch import PILOT_CEILING, launch
from tests.test_run_authority import authority as sealed_authority

from entropic_substrate.core.sentence.profile import (
    DEVELOPMENT_DIAGNOSTIC_SAMPLES,
    DEVELOPMENT_GENERATION_CONTINUATIONS,
    DEVELOPMENT_IMPORTANCE_SAMPLES,
    HProfile,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.h_probe_collection import collect_probe_terminal
from entropic_substrate.ladder.run.h_probe_launch import (
    LAUNCH_SCHEMA_V3,
    LAUNCH_SCHEMA_V4,
    LAUNCH_SCHEMA_V5,
    H0ProbeLaunch,
    parse_probe_launch,
)
from entropic_substrate.ladder.run.worker_request import (
    BlockBinding,
    PairCell,
    PairProbeRequest,
    pair_identity,
)

pytestmark = pytest.mark.unit

MAX_BYTES = 10_000_000


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    return make_pair(tmp_path)


def pilot_cell(profile: HProfile) -> HProfile:
    """The cell as a ``pilot`` profile: cuda/bf16 and the development sample counts."""
    return replace(
        profile,
        name=profile.name.removesuffix("-tiny") + "-pilot",  # a smoke name ends in -tiny
        run_class="pilot",
        device="cuda",
        precision="bf16",
        schedule=replace(
            profile.schedule,
            importance_samples=DEVELOPMENT_IMPORTANCE_SAMPLES,
            diagnostic_samples=DEVELOPMENT_DIAGNOSTIC_SAMPLES,
            generation_continuations=DEVELOPMENT_GENERATION_CONTINUATIONS,
        ),
    )


def pilot_pair_request(pair: Pair, **authority_changes: object) -> PairProbeRequest:
    """A valid H1 pilot PAIR request under one sealed authority (constructed, never executed)."""
    donor, recipient = pilot_cell(pair.donor), pilot_cell(pair.recipient)
    source = ArtifactIdentity(
        "source/prepared-corpus.json", ContentHash("sha256", donor.source_digest)
    )
    bundle = ArtifactIdentity("code.bundle", ContentHash("sha256", "b" * 64))
    return PairProbeRequest(
        cells=(PairCell(donor, None, None), PairCell(recipient, None, None)),
        inputs=(source, bundle),
        nonce="1" * 64,
        limits=LIMITS,
        probe=PROBE,
        code_commit="c" * 40,
        code_bundle=bundle,
        image="ghcr.io/luiz-frias/entropic-substrate-lm@sha256:" + "d" * 64,
        binding=BlockBinding(
            BLOCK, pair_identity(donor, recipient), donor.source_digest, donor.seed
        ),
        authority=sealed_authority(**authority_changes),
    )


def pilot_pair_launch(base: Path, **authority_changes: object) -> H0ProbeLaunch:
    """A v5 launch of a pilot pair; ``base`` must be an empty directory the fixture may fill."""
    fixture = base / "pilot-pair"
    fixture.mkdir(parents=True)
    return replace(
        launch(),
        request=pilot_pair_request(make_pair(fixture), **authority_changes),
        cost_guard="host_watchdog_budgeted",
        hourly_price_ceiling=PILOT_CEILING,
    )


# ------------------------------------------------------------------------------ the tracer
def test_a_v5_launch_seals_the_ceiling_and_the_host_collects_both_cells_of_a_pair(
    tmp_path: Path, pair: Pair
) -> None:
    # ---- the launch half: a v5 launch of a pair round-trips and its digest binds everything
    scientific = replace(
        launch(),
        request=pilot_pair_request(pair),
        cost_guard="host_watchdog_budgeted",
        hourly_price_ceiling=PILOT_CEILING,
    )
    request = scientific.request
    assert isinstance(request, PairProbeRequest)
    assert parse_probe_launch(scientific.payload, max_bytes=1_000_000) == scientific
    manifest = scientific.manifest
    assert manifest["schema"] == LAUNCH_SCHEMA_V5
    assert manifest["hourly_price_ceiling"] == PILOT_CEILING and manifest["hours_cap"] == 6.0
    assert manifest["pair"] is True
    assert manifest["cells"] == [request.donor.identity, request.recipient.identity]
    assert replace(scientific, hourly_price_ceiling=30.0).digest != scientific.digest
    assert scientific.pod_name == f"h1-pilot-{scientific.digest[:24]}"
    assert scientific.hourly_ceiling(None) == PILOT_CEILING
    with pytest.raises(ValueError, match="differs from the launch's sealed hourly price"):
        scientific.hourly_ceiling(PILOT_CEILING + 1)
    for schema in (LAUNCH_SCHEMA_V3, LAUNCH_SCHEMA_V4):
        with pytest.raises(ValueError, match="a pair launch requires h-probe-launch-v5"):
            replace(scientific, manifest_schema=schema, hourly_price_ceiling=None)

    # ---- the host half: the L4b worker ran both cells; the host collects ONE v2 collection
    world, objects, marker = pair_world(tmp_path, pair)
    status, error, _workspace, _observability = run_command(tmp_path, objects, marker)
    assert status == 0, error
    destination = tmp_path / "collected"
    destination.mkdir()
    report = collect_probe_terminal(
        objects, world, destination, expected_pod_id=None, max_marker_bytes=MAX_BYTES
    )
    assert report is not None
    assert report["schema"] == "h-probe-terminal-collection-v2"
    assert report["terminal_name"] == "complete"
    assert report["verdict"] == {"verdict": "complete", "reason": None, "cell": None}
    assert report["workload_verified"] is True and report["pair_id"] == world.binding.pair_id
    cells = report["cells"]
    assert [entry["cell"] for entry in cells] == ["donor", "recipient"]
    for entry, profile in zip(cells, (world.donor, world.recipient), strict=True):
        assert entry["run_identity"] == profile.identity
        assert entry["outcome"] == "complete" and entry["successful_optimizer_steps"] == 6
        assert entry["receipt"]["path"].startswith(f"run/{entry['cell']}/")
        assert (destination / entry["capture_root"]).is_dir()
        assert (destination / entry["verification_root"]).is_dir()
    assert json.loads((destination / "collection.json").read_text()) == report


# ------------------------------------------------------------------ the pair collection (v2)
def collect(
    tmp_path: Path, objects: Any, request: PairProbeRequest, name: str = "collected"
) -> dict[str, Any] | None:
    destination = tmp_path / name
    destination.mkdir()
    return collect_probe_terminal(  # type: ignore[return-value]
        objects, request, destination, expected_pod_id=None, max_marker_bytes=MAX_BYTES
    )


def test_a_pair_with_no_terminal_collects_as_none_and_touches_nothing(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects, _marker = pair_world(tmp_path, pair)
    assert collect(tmp_path, objects, request) is None
    assert not list((tmp_path / "collected").iterdir())


def test_an_excluded_pair_is_not_a_verified_workload_even_though_completion_verifies_its_bytes(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verdict, never the status: completion v2 reports ``probe_completion_verified`` for an
    excluded pair too (it proves the bytes), so a collection that keyed on that status would call
    a scientific exclusion a verified workload and the controller would stop it as
    ``capture_verified``."""
    from tests.test_h_pair_request import Fault

    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    request, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "prediction", 1, UndefinedDirection(2))  # the donor diverges after warm-up
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    report = collect(tmp_path, objects, request)
    assert report is not None
    assert report["terminal_name"] == "excluded"
    assert report["reported_outcome"] == "pair_excluded"
    assert report["completion"]["status"] == "probe_completion_verified"  # the trap
    assert report["workload_verified"] is False
    assert report["capture_status"] == "pair_exclusion_bytes_verified"
    assert report["verdict"] == {
        "verdict": "excluded",
        "reason": "nonfinite_endpoint",
        "cell": "donor",
    }
    donor, recipient = report["cells"]
    assert (donor["outcome"], recipient["outcome"]) == ("scientific_gap", "complete")
    assert donor["capture_root"] is None and recipient["capture_root"] == "capture/recipient"
    assert donor["receipt"]["path"].startswith("run/donor/")  # a cell's receipt, its own root


def test_result_v2_is_read_under_the_failure_terminal_and_keyed_on_its_verdict(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DH-47: a training-phase execution stop leaves result v2 beside the v1 failure record in the
    FAILURE snapshot; the host reads it there and records the verdict, never a status."""
    from tests.test_h_pair_request import Fault

    request, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "warmup", 1, RuntimeError("injected reader refusal"))
    status, _, _, _ = run_command(tmp_path, objects, marker)
    assert status == 2
    report = collect(tmp_path, objects, request)
    assert report is not None
    assert report["terminal_name"] == "failure"
    assert report["reported_outcome"] == "execution_defect" and report["workload_verified"] is False
    assert report["capture_status"] == "failure_bytes_verified"
    assert report["verdict"] == {"verdict": "excluded", "reason": "reader_refusal", "cell": "donor"}
    assert [c["outcome"] for c in report["cells"]] == ["execution_defect", "not_run"]
    assert report["cells"][1]["receipt"] is None and report["cells"][1]["capture_root"] is None


def test_a_post_training_failure_keeps_the_v1_path_and_names_no_verdict(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DH-45: a post-training failure of a cell stops the pair through the v1 failure record, with
    no result v2 and no exclusion member; the host records evidence it has, never a cell outcome."""
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.run import worker_execution

    request, objects, marker = pair_world(tmp_path, pair)
    real = worker_execution.run_probe_generation
    calls: list[str] = []

    def generation(*args: Any, **kwargs: Any) -> Any:
        calls.append("generation")
        if len(calls) == 2:
            raise UndefinedDirection(3)
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_execution, "run_probe_generation", generation)
    status, _, _, _ = run_command(tmp_path, objects, marker)
    assert status == 2
    report = collect(tmp_path, objects, request)
    assert report is not None and report["terminal_name"] == "failure"
    assert report["reported_outcome"] == "scientific_gap"
    assert report["capture_status"] == "failure_bytes_verified"
    assert report["verdict"] is None and report["workload_verified"] is False
    assert [c["outcome"] for c in report["cells"]] == ["unreported", "unreported"]
    assert report["cells"][0]["capture_root"] == "capture/donor"  # the donor's evidence is there


def test_a_pair_that_died_before_its_first_cell_collects_as_pair_not_started(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run import worker_execution

    request, objects, marker = pair_world(tmp_path, pair)

    def refused(*_: Any, **__: Any) -> Any:
        raise RuntimeError("runtime admission refused before any cell began")

    monkeypatch.setattr(worker_execution, "_runtime", refused)
    status, _, _, observability = run_command(tmp_path, objects, marker)
    assert status == 2 and not observability.exists()  # no sink, no MLflow run, no worker.end
    report = collect(tmp_path, objects, request)
    assert report is not None and report["terminal_name"] == "failure"
    assert report["capture_status"] == "pair_not_started"
    assert report["workload_verified"] is False and report["verdict"] is None
    assert all(c["outcome"] == "unreported" and c["capture_root"] is None for c in report["cells"])
    assert (tmp_path / "collected" / "worker" / "failure.json").is_file()


def test_pair_not_started_needs_positive_evidence_a_failed_delivery_is_a_missing_snapshot(
    tmp_path: Path, pair: Pair
) -> None:
    request, _objects, _marker = pair_world(tmp_path, pair)
    failed = {
        "schema": "h-worker-event-v1",
        "request_sha256": request.digest,
        "run_identity": request.run_identity,
        "scientific_authority": "none",
        "at": "2026-11-02T12:00:00+00:00",
        "status": "execution_defect",
        "stage": "execution",
        "error_type": "RuntimeError",
        "error_code": "fixture",
        "evidence_delivery": "failed",
        "snapshot": None,
        "delivery_error_type": "OSError",
    }
    store = _ObjectStore()
    store.values[f"{request.prefix}/failure.json"] = json.dumps(
        failed, sort_keys=True, separators=(",", ":")
    ).encode()
    report = collect(tmp_path, store, request)
    assert report is not None
    assert report["capture_status"] == "missing_failure_snapshot"  # NOT pair_not_started


@pytest.mark.parametrize("terminal", ["paused", "stopped"])
def test_a_paused_or_stopped_pair_hands_back_each_cells_resume_receipt(
    tmp_path: Path, pair: Pair, terminal: str
) -> None:
    from tests.test_h_pair_request import marker_store, pair_request

    from entropic_substrate.ladder.run.object_worker import encode_stop_marker
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    if terminal == "paused":
        request = pair_request(
            pair,
            cells=(
                PairCell(pair.donor, None, None),
                PairCell(pair.recipient, None, ("prediction", 2)),
            ),
        )
        objects: Any = _ObjectStore()
    else:
        request = pair_request(pair)
        objects = marker_store(request, encode_stop_marker(request, ("prediction", 2, "recipient")))
    marker = publish_probe_request(objects, request, tmp_path)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    report = collect(tmp_path, objects, request)
    assert report is not None and report["terminal_name"] == terminal
    assert (
        report["capture_status"] == f"{'pause' if terminal == 'paused' else 'stop'}_bytes_verified"
    )
    assert report["workload_verified"] is False and report["verdict"]["verdict"] == "none"
    donor, recipient = report["cells"]
    assert donor["outcome"] == "complete"
    assert recipient["outcome"] == ("paused" if terminal == "paused" else "stopped_by_operator")
    assert recipient["successful_optimizer_steps"] == 4
    assert recipient["receipt"]["path"].startswith("run/recipient/")
    assert (tmp_path / "collected" / recipient["receipt"]["path"]).is_file()


def test_a_collection_of_one_cell_missing_is_refused_not_collected(
    tmp_path: Path, pair: Pair
) -> None:
    """A completion that omits one cell's capture is not a complete pair: the host re-verifies each
    cell against its own capture and verification roots."""
    request, objects, marker = pair_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    snapshot_key = f"{request.prefix}/snapshots/complete.json"
    manifest = json.loads(objects.values[snapshot_key])
    manifest["files"] = [f for f in manifest["files"] if not f["path"].startswith("capture/donor/")]
    manifest["artifact_count"] = len(manifest["files"])
    manifest["artifact_bytes"] = sum(f["size_bytes"] for f in manifest["files"])
    forged = json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    objects.values[snapshot_key] = forged
    event_key = f"{request.prefix}/complete.json"
    event = json.loads(objects.values[event_key])
    event["snapshot"] = {
        "key": snapshot_key,
        "size_bytes": len(forged),
        "sha256": __import__("hashlib").sha256(forged).hexdigest(),
    }
    objects.values[event_key] = json.dumps(
        event, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    with pytest.raises(ValueError, match="could not be opened beneath its root"):
        collect(tmp_path, objects, request)


def test_a_pair_collection_presented_as_v1_is_refused_by_name_with_the_minimum_version(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.h_probe_collection import admit_pair_collection

    request, objects, marker = pair_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    report = collect(tmp_path, objects, request)
    assert report is not None
    assert admit_pair_collection(report, request) == report
    with pytest.raises(ValueError) as raised:
        admit_pair_collection({**report, "schema": "h-probe-terminal-collection-v1"}, request)
    assert "h-probe-terminal-collection-v1" in str(raised.value)
    assert "minimum supported version is h-probe-terminal-collection-v2" in str(raised.value)
    for tamper in (
        {"request_sha256": "0" * 64},
        {"pair_id": "0" * 64},
        {"cells": list(reversed(report["cells"]))},
        {"cells": report["cells"][:1]},
        {"schema": "h-probe-terminal-collection-v3"},
    ):
        with pytest.raises(ValueError, match="not a pair terminal collection of this request"):
            admit_pair_collection({**report, **tamper}, request)
    with pytest.raises(ValueError, match="must be an object"):
        admit_pair_collection([report], request)


def test_a_single_profile_collection_stays_v1_with_exactly_its_key_set(tmp_path: Path) -> None:
    from tests.test_sentence_worker_execution import prepared

    from entropic_substrate.ladder.run import object_worker
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "executed"
    worker.mkdir()
    object_worker.run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    destination = tmp_path / "collected"
    destination.mkdir()
    report = collect_probe_terminal(
        objects, request, destination, expected_pod_id=None, max_marker_bytes=1_000_000
    )
    assert report is not None
    assert report["schema"] == "h-probe-terminal-collection-v1"
    assert set(report) == {
        "schema",
        "scientific_authority",
        "request_sha256",
        "terminal",
        "reported_outcome",
        "workload_verified",
        "reported_optimizer_steps",
        "native_validation",
        "capture_status",
        "completion",
        "terminal_artifact",
    }
    assert "cells" not in report and "verdict" not in report and "pair_id" not in report


# ----------------------------------------------------------- the controller, on a pair launch
def drive_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, objects: Any, request: PairProbeRequest
) -> tuple[dict[str, object], Any]:
    """Run ``run_probe_controller`` over a pair world with the fake provider and guardian.

    The pair that EXECUTES on a CPU is ``local_tiny``, whose runtime names no pod, while the
    controller admits a collection against its allocated pod id; the two runtime-admission
    entry points are therefore replaced by ONE stub that asserts the pod id it was handed and
    returns the runtime unchanged. Everything else (the sealed v5 ceiling, the pair event reader,
    the pair collection, the verdict-keyed ``workload_verified`` and the stop reason) is real.
    """
    from types import SimpleNamespace
    from typing import cast

    from tests.test_h_probe_controller import Guard, Provider

    from entropic_substrate.ladder.run import h_probe_collection, worker_completion
    from entropic_substrate.ladder.run.h_probe_process import ProbeWatchdogProcess
    from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
        D1TrainerMailboxObjectPort,
    )
    from entropic_substrate.ladder.run.worker.h_probe_controller import run_probe_controller

    seen: list[object] = []

    def admit(runtime: object, _request: object, expected_pod_id: object) -> object:
        seen.append(expected_pod_id)
        return runtime

    monkeypatch.setattr(worker_completion, "admit_worker_runtime", admit)
    monkeypatch.setattr(h_probe_collection, "admit_worker_runtime", admit)
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_events.admit_worker_runtime", admit)
    sealed = pilot_pair_launch(tmp_path / "ctl")
    guard = Guard(tmp_path)
    provider = Provider(guard)
    guard.launch = SimpleNamespace(  # type: ignore[assignment]
        request=request,
        digest=sealed.digest,
        pod_name=sealed.pod_name,
        require_hourly_rate_within_cap=sealed.require_hourly_rate_within_cap,
        hourly_ceiling=sealed.hourly_ceiling,
    )
    provider.pod = replace(provider.pod, name=sealed.pod_name)
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
    destination = tmp_path / "controller-capture"
    destination.mkdir()
    clock = [0.0]
    result = run_probe_controller(
        cast(ProbeWatchdogProcess, guard),
        provider,
        cast(D1TrainerMailboxObjectPort, objects),
        destination,
        poll_seconds=1,
        max_hourly_rate=None,  # the sealed ceiling supplies it
        monotonic=lambda: clock[0],
        wait=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    assert calls == ["create"] and seen and set(seen) == {"owned-pod"}
    return result, guard


def test_a_pair_launch_is_driven_through_the_controller_to_terminal_collected(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, objects, marker = pair_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    result, guard = drive_controller(tmp_path, monkeypatch, objects, request)
    assert result["status"] == "terminal_collected" and result["workload_verified"] is True
    collection = result["collection"]
    assert collection["schema"] == "h-probe-terminal-collection-v2"
    assert collection["terminal_name"] == "complete" and len(collection["cells"]) == 2
    assert guard.control.records["stop"]["reason"] == "capture_verified"


def test_an_excluded_pair_is_terminal_collected_but_stops_as_a_partial_capture(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop reason the controller chooses reads ``collection['workload_verified']``: a wrong
    value there (the status-keyed trap) would write a false ``capture_verified``."""
    from tests.test_h_pair_request import Fault

    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    request, objects, marker = pair_world(tmp_path, pair)
    with monkeypatch.context() as patch:
        Fault(patch, "prediction", 1, UndefinedDirection(2))
        status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    result, guard = drive_controller(tmp_path, monkeypatch, objects, request)
    assert result["status"] == "terminal_collected" and result["workload_verified"] is False
    assert result["collection"]["terminal_name"] == "excluded"
    assert guard.control.records["stop"]["reason"] == "partial_capture"


# ------------------------------------------------------------- the advisory progress tick
class CountingStore(_ObjectStore):
    """The port's every ``read`` and ``publish``, counted at the boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] = []

    def read(self, key: str) -> Any:
        self.reads.append(key)
        return super().read(key)


def counting_copy(store: _ObjectStore) -> CountingStore:
    copy = CountingStore()
    copy.values.update(store.values)
    return copy


def test_the_pair_progress_tick_names_the_event_it_read_per_cell_and_claims_nothing_else(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.worker.h_probe_controller import _CheckpointProgress

    request, objects, marker = pair_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    prefix = request.prefix
    # a store holding only the donor's events: its newest, and the donor's own name
    donor_only = CountingStore()
    donor_only.values.update(
        {
            key: value
            for key, value in objects.values.items()
            if "checkpoint-recipient" not in key
            and key.startswith(prefix)
            and not key.endswith(("/complete.json", "/snapshots/complete.json"))
        }
    )
    progress = _CheckpointProgress(request)
    assert progress.total == 6
    assert progress.advance(donor_only) == 6 and progress.last_event == "checkpoint-donor-00000006"
    # the whole run: the recipient is as far along, and wins a tie by being the later cell
    full = counting_copy(objects)
    again = _CheckpointProgress(request)
    assert again.advance(full) == 6 and again.last_event == "checkpoint-recipient-00000006"
    # nothing published: no step, no event, and no claim
    empty = CountingStore()
    nothing = _CheckpointProgress(request)
    assert nothing.advance(empty) is None and nothing.last_event is None
    # reads are bounded by the ladders, not by what sits under the prefix
    assert 0 < len(empty.reads) <= 2 and len(full.reads) <= 2 * 6


def test_the_pair_progress_tick_never_relies_on_an_attribute_error_being_swallowed(
    tmp_path: Path, pair: Pair
) -> None:
    """A pair has no ``profile`` or ``resume``; the tick must read its cells, so it reaches the
    advisory ``except`` of ``_progress_tick`` for no reason of its own: no AttributeError."""
    from entropic_substrate.ladder.run.worker.h_probe_controller import _CheckpointProgress

    request, _objects, _marker = pair_world(tmp_path, pair)
    progress = _CheckpointProgress(request)
    assert progress.advance(CountingStore()) is None  # would raise AttributeError on the parent


# ----------------------------------------------------------------------------- the stop verb
def stop_kwargs() -> dict[str, Any]:
    ticks = iter(range(1_000_000))  # the clock advances: a dropped bound must time out
    return {
        "wait_seconds": 1.0,
        "poll_seconds": 0.1,
        "monotonic": lambda: float(next(ticks)),
        "sleep": lambda _: None,
    }


def test_a_pair_stop_names_its_cell_publishes_a_v2_marker_and_stops_the_recipient_only(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.object_worker import (
        STOP_MARKER_SCHEMA_V2,
        encode_stop_marker,
        stop_marker_key,
    )
    from entropic_substrate.ladder.run.worker.h_probe_stop import request_stop

    request, objects, marker = pair_world(tmp_path, pair)
    assert (
        request_stop(objects, request, target=("prediction", 2), cell="recipient", **stop_kwargs())
        is None
    )  # no worker has run: the marker is published, the receipt is not there yet
    published = objects.values[stop_marker_key(request)]
    assert published == encode_stop_marker(request, ("prediction", 2, "recipient"))
    row = json.loads(published)
    assert row["schema"] == STOP_MARKER_SCHEMA_V2 == "h-probe-stop-marker-v2"
    assert row["target"] == {"phase": "prediction", "step": 2, "cell": "recipient"}
    status, error, workspace, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    assert result["status"] == "stopped_by_operator"
    assert [c["outcome"] for c in result["cells"]] == ["complete", "stopped_by_operator"]
    # the same verb, idempotent, now finds the pair's `stopped` receipt and lists its keys
    keys = request_stop(
        objects, request, target=("prediction", 2), cell="recipient", **stop_kwargs()
    )
    assert keys is not None and f"{request.prefix}/snapshots/stopped.json" in keys
    assert any("/outputs/run/recipient/" in key for key in keys)


@pytest.mark.parametrize(
    ("target", "cell", "message"),
    [
        (("prediction", 2), None, "a pair stop names its cell: pass --at-cell"),
        (None, "recipient", "a pair stop names its seal"),
        (("prediction", 2), "observer", "must be .*phase, positive step, cell"),
        (("prediction", 3), "recipient", "outside the request schedule"),
        (("codec_warmup", 99), "donor", "outside the request schedule"),
    ],
)
def test_a_pair_stop_that_cannot_name_one_cell_at_one_seal_publishes_nothing(
    tmp_path: Path, pair: Pair, target: Any, cell: Any, message: str
) -> None:
    from entropic_substrate.ladder.run.object_worker import stop_marker_key
    from entropic_substrate.ladder.run.worker.h_probe_stop import request_stop

    request, objects, _marker = pair_world(tmp_path, pair)
    before = list(objects.publishes)
    with pytest.raises(ValueError, match=message):
        request_stop(objects, request, target=target, cell=cell, **stop_kwargs())
    assert stop_marker_key(request) not in objects.values and objects.publishes == before


def test_the_stop_command_takes_a_cell_for_a_pair_launch_and_refuses_it_for_a_single_one(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command parses ``--at-cell`` into the verb for a launch that carries a pair, and the
    host event it narrates names the cell beside the seal."""
    import io

    from entropic_substrate.ladder.run.worker import h_probe_stop

    value = pilot_pair_launch(tmp_path)
    (tmp_path / "launch.json").write_bytes(value.payload)
    seen: dict[str, Any] = {}

    def verb(objects: object, request: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        seen["pair"] = isinstance(request, PairProbeRequest)

    monkeypatch.setattr(h_probe_stop, "request_stop", verb)
    monkeypatch.setattr(h_probe_stop, "build_worker_objects", lambda *_a, **_k: object())
    code = h_probe_stop.run_probe_stop(
        [
            "--launch-root",
            str(tmp_path),
            "--launch-file",
            "launch.json",
            "--repo-root",
            str(tmp_path),
            "--max-launch-bytes",
            "1000000",
            "--wait-seconds",
            "5",
            "--poll-seconds",
            "1",
            "--at-phase",
            "prediction",
            "--at-step",
            "2",
            "--at-cell",
            "recipient",
        ],
        environ={},
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert code == 1  # the verb returned None: the stop is unconfirmed, as it should be here
    assert seen["pair"] is True and seen["cell"] == "recipient"
    assert seen["target"] == ("prediction", 2)
