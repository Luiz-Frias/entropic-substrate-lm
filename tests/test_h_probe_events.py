"""Host observation preserves producer stages without inventing completion authority."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.ladder.run.h_probe_events import read_probe_event
from entropic_substrate.ladder.run.object_worker import run_object_worker
from entropic_substrate.ladder.run.worker_request import ProbeRequest
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit


def _run_and_fetch_checkpoint_key(tmp_path: Path) -> tuple[_ObjectStore, ProbeRequest, str]:
    request, inputs = prepared(tmp_path, pause=None)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    root = tmp_path / "run-worker"
    root.mkdir()
    run_object_worker(objects, marker, root, max_request_bytes=1_000_000)
    key = f"{request.prefix}/checkpoint-00000002.json"
    assert key in objects.values
    return objects, request, key


@pytest.fixture(scope="module")
def _checkpoint_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ProbeRequest, str, dict[str, bytes]]:
    """Run the REAL worker pipeline exactly ONCE for this whole module: every
    publication-block test below mutates its OWN in-memory copy of the
    resulting object values, never re-running the (slow) training pipeline."""
    objects, request, key = _run_and_fetch_checkpoint_key(tmp_path_factory.mktemp("h-probe-pub"))
    return request, key, dict(objects.values)


def _store_copy(values: dict[str, bytes]) -> _ObjectStore:
    return _ObjectStore(values=dict(values))


def test_the_worker_always_writes_a_publication_block_on_checkpoint_events(
    _checkpoint_fixture: tuple[ProbeRequest, str, dict[str, bytes]],
) -> None:
    """Write-new (FOR-1567/DG-C): every checkpoint-* event the worker emits
    carries a well-shaped publication block, unconditionally."""
    request, key, values = _checkpoint_fixture
    objects = _store_copy(values)
    row = json.loads(objects.values[key])
    publication = row["publication"]
    assert set(publication) == {
        "l1_seconds",
        "l2_seconds",
        "l3_seconds",
        "objects_put",
        "objects_skipped",
        "verification_reads",
        "retries",
    }
    for seconds_field in ("l1_seconds", "l2_seconds", "l3_seconds"):
        assert isinstance(publication[seconds_field], (int, float))
        assert publication[seconds_field] >= 0
    for count_field in ("objects_put", "objects_skipped", "verification_reads", "retries"):
        assert type(publication[count_field]) is int and publication[count_field] >= 0
    event = read_probe_event(
        objects, request, "checkpoint-00000002", expected_pod_id=None, max_bytes=1_000_000
    )
    assert event is not None and event.reported_status == "checkpoint"


def test_a_checkpoint_event_is_admitted_with_or_without_a_publication_block(
    _checkpoint_fixture: tuple[ProbeRequest, str, dict[str, bytes]],
) -> None:
    """Read-old (FOR-1567/DG-C): a checkpoint event recorded BEFORE this door
    existed -- no ``publication`` key at all -- is still admitted."""
    request, key, values = _checkpoint_fixture
    objects = _store_copy(values)
    row = json.loads(objects.values[key])
    assert "publication" in row  # write-new sanity: the fixture really has one
    del row["publication"]
    old_payload = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    objects.values[key] = old_payload
    event = read_probe_event(
        objects, request, "checkpoint-00000002", expected_pod_id=None, max_bytes=1_000_000
    )
    assert event is not None and event.reported_status == "checkpoint"


@pytest.mark.parametrize(
    "corruption",
    [
        lambda pub: {**pub, "retries": -1},
        lambda pub: {**pub, "retries": 1.5},
        lambda pub: {**pub, "l1_seconds": -0.1},
        lambda pub: {k: v for k, v in pub.items() if k != "retries"},
    ],
)
def test_a_malformed_publication_block_is_refused(
    _checkpoint_fixture: tuple[ProbeRequest, str, dict[str, bytes]], corruption
) -> None:
    request, key, values = _checkpoint_fixture
    objects = _store_copy(values)
    row = json.loads(objects.values[key])
    row["publication"] = corruption(row["publication"])
    payload = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    objects.values[key] = payload
    with pytest.raises(ValueError):
        read_probe_event(
            objects, request, "checkpoint-00000002", expected_pod_id=None, max_bytes=1_000_000
        )


def test_a_nonfinite_publication_second_is_refused(
    _checkpoint_fixture: tuple[ProbeRequest, str, dict[str, bytes]],
) -> None:
    """``allow_nan=False`` on the WRITE side (object_worker/_event) already
    forbids a NaN in a fresh payload; this pins the READER's own defense for
    a payload that reached it by some other route (e.g. read-old storage)."""
    request, key, values = _checkpoint_fixture
    objects = _store_copy(values)
    row = json.loads(objects.values[key])
    row["publication"]["l1_seconds"] = float("nan")
    # allow_nan=True here deliberately: constructing the adversarial payload
    # must not itself be blocked by the same guard under test.
    payload = json.dumps(row, sort_keys=True, allow_nan=True, separators=(",", ":")).encode()
    objects.values[key] = payload
    with pytest.raises(ValueError):
        read_probe_event(
            objects, request, "checkpoint-00000002", expected_pod_id=None, max_bytes=1_000_000
        )


@pytest.mark.parametrize("pause", [None, ("prediction", 2)])
def test_real_worker_events_share_one_lineage_and_remain_observations(
    tmp_path: Path, pause: tuple[str, int] | None
) -> None:
    request, inputs = prepared(tmp_path, pause=pause)
    objects = _ObjectStore()
    assert (
        read_probe_event(objects, request, "started", expected_pod_id=None, max_bytes=1_000_000)
        is None
    )
    marker = publish_probe_request(objects, request, inputs)
    root = tmp_path / "run-worker"
    root.mkdir()
    run_object_worker(objects, marker, root, max_request_bytes=1_000_000)
    for name, status in (
        ("started", "inputs_admitted"),
        ("checkpoint-00000002", "checkpoint"),
        ("complete", "probe_workload_complete") if pause is None else ("paused", "paused"),
    ):
        event = read_probe_event(objects, request, name, expected_pod_id=None, max_bytes=1_000_000)
        assert event is not None and event.reported_status == status
        assert event.snapshot is not None
        assert event.snapshot.key == f"{request.prefix}/snapshots/{name}.json"
        assert event.scientific_authority == "none" and event.completion_verified is False
        assert event.reported_optimizer_steps == (2 if status == "checkpoint" else None)


def event(request: ProbeRequest, name: str, **fields: object) -> dict[str, object]:
    return {
        "schema": "h-worker-event-v1",
        "request_sha256": request.digest,
        "run_identity": request.profile.identity,
        "scientific_authority": "none",
        "at": "2026-09-16T00:00:00+00:00",
        **fields,
        "snapshot": {
            "key": f"{request.prefix}/snapshots/{name}.json",
            "size_bytes": 100,
            "sha256": "2" * 64,
        },
    }


def store(request: ProbeRequest, name: str, value: object) -> _ObjectStore:
    objects = _ObjectStore()
    raw = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    objects.publish(
        f"{request.prefix}/{name}.json", raw, expected_sha256=hashlib.sha256(raw).hexdigest()
    )
    return objects


def _publication_block(**overrides: object) -> dict[str, object]:
    block: dict[str, object] = {
        "l1_seconds": 0.0,
        "l2_seconds": 0.0,
        "l3_seconds": 0.01,
        "objects_put": 2,
        "objects_skipped": 0,
        "verification_reads": 2,
        "retries": 0,
    }
    block.update(overrides)
    return block


# FOR-1567/pre-for-1567c, DN-1567-C: `publication` is admitted on
# checkpoint/complete/paused/stopped -- the fields each terminal kind
# otherwise requires, minus `snapshot` (added by `event()`) and
# `publication` itself (added per-test below).
_TERMINAL_FIELDS: dict[str, dict[str, object]] = {
    "complete": {
        "status": "probe_workload_complete",
        "result": {"path": "worker/result.json", "sha256": "3" * 64},
    },
    "paused": {
        "status": "paused",
        "result": {"path": "worker/result.json", "sha256": "3" * 64},
    },
    "stopped": {
        "status": "stopped_by_operator",
        "result": {"path": "worker/result.json", "sha256": "3" * 64},
        "successful_optimizer_steps": 4,
    },
}


@pytest.mark.parametrize("kind", ["complete", "paused", "stopped"])
def test_a_terminal_event_is_admitted_with_or_without_a_publication_block(
    tmp_path: Path, kind: str
) -> None:
    """Write-new/read-old (DN-1567-C): every terminal kind the final-snapshot
    call site can emit admits its event both with and without `publication`."""
    request, _ = prepared(tmp_path)
    with_block = event(request, kind, **_TERMINAL_FIELDS[kind], publication=_publication_block())
    observed = read_probe_event(
        store(request, kind, with_block), request, kind, expected_pod_id=None, max_bytes=1_000_000
    )
    assert observed is not None and observed.reported_status == _TERMINAL_FIELDS[kind]["status"]
    without_block = event(request, kind, **_TERMINAL_FIELDS[kind])
    observed = read_probe_event(
        store(request, kind, without_block),
        request,
        kind,
        expected_pod_id=None,
        max_bytes=1_000_000,
    )
    assert observed is not None and observed.reported_status == _TERMINAL_FIELDS[kind]["status"]


@pytest.mark.parametrize("kind", ["complete", "paused", "stopped"])
@pytest.mark.parametrize(
    "corruption",
    [
        lambda pub: {**pub, "retries": -1},
        lambda pub: {**pub, "l1_seconds": -0.1},
        lambda pub: {k: v for k, v in pub.items() if k != "retries"},
    ],
)
def test_a_malformed_publication_block_on_a_terminal_event_is_refused(
    tmp_path: Path, kind: str, corruption: Callable[[dict[str, object]], dict[str, object]]
) -> None:
    request, _ = prepared(tmp_path)
    value = event(
        request, kind, **_TERMINAL_FIELDS[kind], publication=corruption(_publication_block())
    )
    with pytest.raises(ValueError):
        read_probe_event(
            store(request, kind, value), request, kind, expected_pod_id=None, max_bytes=1_000_000
        )


@pytest.mark.parametrize(
    "kind,fields",
    [
        (
            "started",
            {
                "status": "inputs_admitted",
                "runtime": {"kind": "local_tiny", "pod_id": None},
                "record": {"path": "worker/start.json", "sha256": "3" * 64},
            },
        ),
        (
            "failure",
            {
                "status": "execution_defect",
                "stage": "execution",
                "error_type": "FixtureError",
                "error_code": "fixture",
                "evidence_delivery": "captured",
            },
        ),
    ],
)
def test_a_publication_block_on_started_or_failure_is_refused(
    tmp_path: Path, kind: str, fields: dict[str, object]
) -> None:
    """Negative arm (DN-1567-C): `started` and `failure` are NOT in the
    publication door -- an event of either kind carrying the block still
    raises, enforced by exact-key admission (no `extra` branch for them)."""
    request, _ = prepared(tmp_path)
    value = event(request, kind, **fields, publication=_publication_block())
    with pytest.raises(ValueError):
        read_probe_event(
            store(request, kind, value), request, kind, expected_pod_id=None, max_bytes=1_000_000
        )


@pytest.mark.parametrize("status", ["execution_defect", "scientific_gap", "diagnostic_gap"])
@pytest.mark.parametrize("delivery", ["captured", "failed"])
def test_failure_kinds_and_missing_delivery_remain_explicit(
    tmp_path: Path, status: str, delivery: str
) -> None:
    request, _ = prepared(tmp_path)
    value = event(
        request,
        "failure",
        status=status,
        stage="execution",
        error_type="FixtureError",
        error_code="fixture",
        evidence_delivery=delivery,
    )
    if delivery == "failed":
        value.update(snapshot=None, delivery_error_type="OSError")
    observed = read_probe_event(
        store(request, "failure", value),
        request,
        "failure",
        expected_pod_id=None,
        max_bytes=1_000_000,
    )
    assert observed is not None
    assert observed.reported_status == status
    assert (observed.snapshot is None) is (delivery == "failed")
    assert observed.completion_verified is False


@pytest.mark.parametrize(
    "mutation", ["request", "run", "authority", "time", "snapshot", "count", "extra"]
)
def test_rehashed_foreign_or_malformed_event_is_refused(tmp_path: Path, mutation: str) -> None:
    request, _ = prepared(tmp_path)
    name = "checkpoint-00000002"
    value = event(
        request,
        name,
        status="checkpoint",
        successful_optimizer_steps=2,
        training={"path": "run/progress.json", "sha256": "3" * 64},
    )
    if mutation == "request":
        value["request_sha256"] = "0" * 64
    elif mutation == "run":
        value["run_identity"] = "0" * 64
    elif mutation == "authority":
        value["scientific_authority"] = "authorized"
    elif mutation == "time":
        value["at"] = "2026-09-16T00:00:00"
    elif mutation == "snapshot":
        value["snapshot"]["key"] = "foreign/snapshot.json"
    elif mutation == "count":
        value["successful_optimizer_steps"] = 3
    else:
        value["fallback"] = True
    with pytest.raises(ValueError):
        read_probe_event(
            store(request, name, value), request, name, expected_pod_id=None, max_bytes=1_000_000
        )


def test_started_event_cannot_claim_a_foreign_allocation(tmp_path: Path) -> None:
    request, _ = prepared(tmp_path)
    value = event(
        request,
        "started",
        status="inputs_admitted",
        runtime={"kind": "local_tiny", "pod_id": None},
        record={"path": "worker/start.json", "sha256": "3" * 64},
    )
    with pytest.raises(ValueError, match="allocation"):
        read_probe_event(
            store(request, "started", value),
            request,
            "started",
            expected_pod_id="foreign-pod",
            max_bytes=1_000_000,
        )


@pytest.mark.parametrize("name", ["", "checkpoint", "checkpoint-2", "../complete", "approved"])
def test_unsupported_event_names_fail_even_when_objects_are_absent(
    tmp_path: Path, name: str
) -> None:
    request, _ = prepared(tmp_path)
    with pytest.raises(ValueError, match="name"):
        read_probe_event(_ObjectStore(), request, name, expected_pod_id=None, max_bytes=1_000_000)


@pytest.mark.parametrize("maximum", [0, -1, True, 10_000_001])
def test_invalid_read_budgets_fail_before_polling(tmp_path: Path, maximum: int) -> None:
    request, _ = prepared(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        read_probe_event(
            _ObjectStore(), request, "complete", expected_pod_id=None, max_bytes=maximum
        )


@pytest.mark.parametrize("count", [0, -1, True, 7])
def test_checkpoint_count_cannot_escape_the_requested_workload(tmp_path: Path, count: int) -> None:
    request, _ = prepared(tmp_path)
    name = f"checkpoint-{count:08d}"
    value = event(
        request,
        name,
        status="checkpoint",
        successful_optimizer_steps=count,
        training={"path": "run/progress.json", "sha256": "3" * 64},
    )
    with pytest.raises(ValueError):
        read_probe_event(
            store(request, name, value), request, name, expected_pod_id=None, max_bytes=1_000_000
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "status",
        "stage",
        "delivery",
        "missing_snapshot",
        "empty_error",
        "wrong_section",
        "oversize",
        "duplicate_field",
    ],
)
def test_invalid_terminal_claims_are_not_completion(tmp_path: Path, mutation: str) -> None:
    request, _ = prepared(tmp_path)
    name = (
        "failure"
        if mutation in ("stage", "delivery", "missing_snapshot", "empty_error")
        else "complete"
    )
    value = event(
        request,
        name,
        **(
            {
                "status": "execution_defect",
                "stage": "execution",
                "error_type": "ValueError",
                "error_code": "fixture",
                "evidence_delivery": "captured",
            }
            if name == "failure"
            else {
                "status": "probe_workload_complete",
                "result": {"path": "worker/result.json", "sha256": "3" * 64},
            }
        ),
    )
    if mutation == "status":
        value["status"] = "checkpoint"
    elif mutation == "stage":
        value["stage"] = "arbitrary"
    elif mutation == "delivery":
        value["evidence_delivery"] = "ignored"
    elif mutation == "missing_snapshot":
        value["snapshot"] = None
    elif mutation == "empty_error":
        value["error_code"] = " "
    elif mutation == "wrong_section":
        value["result"] = {"path": "run/result.json", "sha256": "3" * 64}
    objects = store(request, name, value)
    if mutation == "duplicate_field":
        key = f"{request.prefix}/{name}.json"
        objects.values[key] = b'{"schema":"h-worker-event-v1",' + objects.values[key][1:]
    with pytest.raises(ValueError):
        read_probe_event(
            objects,
            request,
            name,
            expected_pod_id=None,
            max_bytes=1 if mutation == "oversize" else 1_000_000,
        )
