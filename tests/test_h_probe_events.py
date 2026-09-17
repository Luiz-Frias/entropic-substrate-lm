"""Host observation preserves producer stages without inventing completion authority."""

import hashlib
import json
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.h_probe_events import read_probe_event
from entropic_substrate.sentence_prediction.object_worker import run_object_worker
from entropic_substrate.sentence_prediction.worker_request import ProbeRequest
from entropic_substrate.sentence_prediction.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit


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
