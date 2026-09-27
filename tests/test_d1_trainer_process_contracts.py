"""Canonical admission for raw D1 trainer process reports."""

from __future__ import annotations

import json
import traceback
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from d1_resume_fixtures import resume_start
from d1_trainer_fixtures import ORIGIN_NS, micro_plan, run_start
from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.register.d1_plan import SmokeD1RunPlan
from entropic_substrate.ladder.run.campaign.d1_operator_evidence import (
    d1_runtime_observation_sha256,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1RuntimeObservation
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.ladder.run.transport.d1_trainer_process_contracts import (
    D1TrainerStartReport,
    parse_d1_runtime_report,
    parse_d1_start_report,
    serialize_d1_runtime_report,
    serialize_d1_start_report,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    D1TrainerRequest,
    build_d1_trainer_request,
    d1_trainer_command_sha256,
    d1_trainer_request_sha256,
)

pytestmark = pytest.mark.unit
POD, ATTEMPT = "pod-123", "a" * 64


def canonical_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def runtime_value(start: RunStartProvenance, pod: str = POD) -> D1RuntimeObservation:
    return D1RuntimeObservation(
        pod,
        start.trainer_started_at,
        start.cuda_version,
        start.driver_version,
        start.pytorch_version,
        start.compiler_enabled,
        start.compiler_version,
        ORIGIN_NS,
    )


def runtime_report(start: RunStartProvenance) -> dict[str, object]:
    runtime = runtime_value(start)
    return {
        "attempt_sha256": ATTEMPT,
        "compiler_enabled": runtime.compiler_enabled,
        "compiler_version": runtime.compiler_version,
        "cuda_version": runtime.cuda_version,
        "driver_version": runtime.driver_version,
        "pod_id": runtime.pod_id,
        "pytorch_version": runtime.pytorch_version,
        "schema_version": "d1-trainer-runtime-report-v1",
        "trainer_monotonic_origin_ns": runtime.trainer_monotonic_origin_ns,
        "trainer_started_at": runtime.trainer_started_at.isoformat().replace("+00:00", "Z"),
    }


def trainer_request(plan: SmokeD1RunPlan, start: RunStartProvenance) -> D1TrainerRequest:
    return build_d1_trainer_request(
        plan, start, D1TrainerWallOrigin(start.trainer_started_at, ORIGIN_NS)
    )


def start_report(request: D1TrainerRequest, start: RunStartProvenance) -> dict[str, object]:
    return {
        "attempt_sha256": ATTEMPT,
        "command_sha256": d1_trainer_command_sha256(request),
        "pod_id": POD,
        "process_id": "trainer-123",
        "request_sha256": d1_trainer_request_sha256(request),
        "runtime_sha256": d1_runtime_observation_sha256(runtime_value(start)),
        "schema_version": "d1-trainer-start-report-v1",
        "trainer_started_at": start.trainer_started_at.isoformat().replace("+00:00", "Z"),
    }


def admit_start(
    payload: bytes,
    request: D1TrainerRequest,
    runtime: D1RuntimeObservation,
    started_at: datetime,
    *,
    pod: object = POD,
    command_sha: str | None = None,
) -> D1TrainerStartReport:
    return parse_d1_start_report(
        payload,
        pod_id=pod,
        attempt_sha256=ATTEMPT,  # type: ignore[arg-type]
        request_sha256=d1_trainer_request_sha256(request),
        runtime_sha256=d1_runtime_observation_sha256(runtime),
        trainer_started_at=started_at,
        command_sha256=command_sha or d1_trainer_command_sha256(request),
    )


def test_canonical_reports_admit_every_field_with_known_command_digest() -> None:
    plan = micro_plan(D1Arm.NONE)
    start, runtime = run_start(plan), runtime_value(run_start(plan))
    request = trainer_request(plan, start)
    command_sha = "68cc33a28bad1e3abbaee23d945c31ba59bc6c6707bcfcbc0db56fa0138a4101"  # pragma: allowlist secret
    assert d1_trainer_command_sha256(request) == command_sha
    resume = trainer_request(plan, resume_start(plan, "9" * 64))
    resume_sha = d1_trainer_command_sha256(resume)
    resume_command_sha = "b6633fd7c401f2b94fe477a135a23703f2b4f8c10dcd1250dda6b0821064ae30"  # pragma: allowlist secret
    assert resume_sha == resume_command_sha
    observed_runtime = parse_d1_runtime_report(
        canonical_json(runtime_report(start)), pod_id=POD, attempt_sha256=ATTEMPT
    )
    value = start_report(request, start)
    value["command_sha256"] = command_sha
    observed_start = admit_start(
        canonical_json(value),
        request,
        runtime,
        start.trainer_started_at,
        command_sha=command_sha,
    )
    assert observed_runtime == runtime
    expected = value | {"trainer_started_at": start.trainer_started_at}
    del expected["schema_version"]
    assert observed_start == D1TrainerStartReport(**expected)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        d1_trainer_command_sha256(object())  # type: ignore[arg-type]


def test_report_serializers_are_the_canonical_parser_inverse() -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    runtime = runtime_value(start)
    request = trainer_request(plan, start)
    expected_start = start_report(request, start)
    report = D1TrainerStartReport(
        pod_id=POD,
        attempt_sha256=ATTEMPT,
        request_sha256=d1_trainer_request_sha256(request),
        runtime_sha256=d1_runtime_observation_sha256(runtime),
        trainer_started_at=start.trainer_started_at,
        process_id="trainer-123",
        command_sha256=d1_trainer_command_sha256(request),
    )

    runtime_payload = serialize_d1_runtime_report(runtime, attempt_sha256=ATTEMPT)
    start_payload = serialize_d1_start_report(report)

    assert runtime_payload == canonical_json(runtime_report(start))
    assert start_payload == canonical_json(expected_start)
    assert (
        parse_d1_runtime_report(
            runtime_payload,
            pod_id=POD,
            attempt_sha256=ATTEMPT,
        )
        == runtime
    )
    assert admit_start(start_payload, request, runtime, start.trainer_started_at) == report


def test_report_serializers_reject_unadmitted_values() -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    request = trainer_request(plan, start)
    report = D1TrainerStartReport(
        POD,
        ATTEMPT,
        d1_trainer_request_sha256(request),
        d1_runtime_observation_sha256(runtime_value(start)),
        start.trainer_started_at,
        "trainer-123",
        d1_trainer_command_sha256(request),
    )

    with pytest.raises(TypeError):
        serialize_d1_runtime_report(object(), attempt_sha256=ATTEMPT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        serialize_d1_runtime_report(runtime_value(start), attempt_sha256="invalid")
    with pytest.raises(ValueError):
        serialize_d1_runtime_report(runtime_value(start), attempt_sha256=ATTEMPT.upper())
    with pytest.raises(TypeError):
        serialize_d1_start_report(object())  # type: ignore[arg-type]
    subclass = type("DatetimeSubclass", (datetime,), {}).fromtimestamp(
        start.trainer_started_at.timestamp(),
        tz=UTC,
    )
    invalid_reports = (
        replace(report, process_id=" "),
        replace(report, pod_id=True),  # type: ignore[arg-type]
        replace(report, attempt_sha256="invalid"),
        replace(report, attempt_sha256=ATTEMPT.upper()),
        replace(report, request_sha256=report.request_sha256.upper()),
        replace(report, runtime_sha256=report.runtime_sha256.upper()),
        replace(report, command_sha256=report.command_sha256.upper()),
        replace(report, trainer_started_at="not-a-datetime"),  # type: ignore[arg-type]
        replace(report, trainer_started_at=start.trainer_started_at.replace(tzinfo=None)),
        replace(
            report,
            trainer_started_at=start.trainer_started_at.astimezone(timezone(timedelta(hours=1))),
        ),
        replace(report, trainer_started_at=subclass),
    )
    for invalid in invalid_reports:
        with pytest.raises(ValueError):
            serialize_d1_start_report(invalid)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"secret-sentinel":1,"secret-sentinel":2}',
        b'{"secret-sentinel":',
    ],
)
def test_report_parser_errors_never_retain_raw_payload(payload: bytes) -> None:
    with pytest.raises(ValueError) as caught:
        parse_d1_runtime_report(payload, pod_id=POD, attempt_sha256=ATTEMPT)

    rendered = "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret-sentinel" not in rendered


@pytest.mark.parametrize(
    "drift", "duplicate extra type pod attempt compiler time nonfinite noncanonical".split()
)
def test_runtime_report_drift_is_rejected(drift: str) -> None:
    plan = micro_plan(D1Arm.NONE)
    value = runtime_report(run_start(plan))
    if drift == "duplicate":
        payload = canonical_json(value).replace(
            b'{"attempt_sha256":', b'{"attempt_sha256":"b","attempt_sha256":'
        )
    elif drift == "nonfinite":
        payload = canonical_json(value).replace(
            b'"trainer_monotonic_origin_ns":10000', b'"trainer_monotonic_origin_ns":NaN'
        )
    elif drift == "noncanonical":
        payload = canonical_json(value) + b"\n"
    else:
        key, mutation = {
            "extra": ("extra", True),
            "type": ("trainer_monotonic_origin_ns", True),
            "pod": ("pod_id", "other"),
            "attempt": ("attempt_sha256", "b" * 64),
            "compiler": ("compiler_enabled", 1),
            "time": ("trainer_started_at", "+00:00"),
        }[drift]
        value[key] = mutation
        payload = canonical_json(value)
    with pytest.raises(ValueError):
        parse_d1_runtime_report(payload, pod_id=POD, attempt_sha256=ATTEMPT)


@pytest.mark.parametrize(
    "drift", "pod attempt request runtime time process command nonfinite".split()
)
def test_start_report_drift_is_rejected(drift: str) -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    request, runtime = trainer_request(plan, start), runtime_value(start)
    value = start_report(request, start)
    key = {
        "pod": "pod_id",
        "attempt": "attempt_sha256",
        "request": "request_sha256",
        "runtime": "runtime_sha256",
        "time": "trainer_started_at",
        "process": "process_id",
        "command": "command_sha256",
        "nonfinite": "process_id",
    }[drift]
    value[key] = " " if drift == "process" else "b" * 64
    payload = canonical_json(value)
    if drift == "nonfinite":
        payload = payload.replace(
            b'"process_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"',
            b'"process_id":NaN',
        )
    with pytest.raises(ValueError):
        parse_d1_start_report(
            payload,
            pod_id=POD,
            attempt_sha256=ATTEMPT,
            request_sha256=d1_trainer_request_sha256(request),
            runtime_sha256=d1_runtime_observation_sha256(runtime),
            trainer_started_at=start.trainer_started_at,
            command_sha256=d1_trainer_command_sha256(request),
        )


@pytest.mark.parametrize("invalid_pod", [True, " "])
def test_start_report_rejects_agreeing_invalid_pod(invalid_pod: object) -> None:
    plan = micro_plan(D1Arm.NONE)
    start, request = run_start(plan), trainer_request(plan, run_start(plan))
    value = start_report(request, start)
    value["pod_id"] = invalid_pod
    with pytest.raises(ValueError):
        admit_start(
            canonical_json(value),
            request,
            runtime_value(start),
            start.trainer_started_at,
            pod=invalid_pod,
        )


@pytest.mark.parametrize("kind", ["naive", "non_utc", "subclass"])
def test_start_report_rejects_invalid_expected_datetime(kind: str) -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    request, expected = trainer_request(plan, start), start.trainer_started_at
    if kind == "naive":
        expected = expected.replace(tzinfo=None)
    elif kind == "non_utc":
        expected = expected.astimezone(timezone(timedelta(hours=1)))
    else:
        expected = datetime.fromtimestamp(expected.timestamp(), tz=UTC)
        expected = type("DatetimeSubclass", (datetime,), {}).fromtimestamp(
            expected.timestamp(), tz=UTC
        )
    with pytest.raises(ValueError):
        admit_start(
            canonical_json(start_report(request, start)),
            request,
            runtime_value(start),
            expected,
        )


@pytest.mark.parametrize("fraction", [".0Z", ".000000Z", ".1Z"])
def test_start_report_rejects_noncanonical_fraction_spelling(fraction: str) -> None:
    plan = micro_plan(D1Arm.NONE)
    start = run_start(plan)
    expected = start.trainer_started_at.replace(microsecond=100_000 if fraction == ".1Z" else 0)
    request, value = trainer_request(plan, start), start_report(trainer_request(plan, start), start)
    value["trainer_started_at"] = expected.strftime("%Y-%m-%dT%H:%M:%S") + fraction
    with pytest.raises(ValueError):
        admit_start(
            canonical_json(value),
            request,
            runtime_value(start),
            expected,
        )
