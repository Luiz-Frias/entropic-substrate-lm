"""Allocation hands ownership to the live guardian, never to a guessed pod name."""

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_probe_control import root
from tests.test_h_probe_launch import launch
from tests.test_h_probe_process import config, fake_policy

from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.run.h_probe_control import (
    own_probe_role,
    read_probe_control,
)
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_probe_process import (
    ProbeWatchdogProcess,
    supervise_probe_watchdog,
)
from entropic_substrate.ladder.run.h_probe_watchdog import _allocation, _allocation_message
from entropic_substrate.ladder.run.h_runpod_transport import (
    ProbeAccount,
    ProbePod,
    ProbeProviderError,
)
from entropic_substrate.ladder.run.worker.h_probe_allocation import (
    ProbeAllocationHandoffError,
    _bounded_provider_detail,
    allocate_probe_once,
    provider_error_code,
)
from entropic_substrate.ladder.run.worker_transport import publish_verified_object

pytestmark = pytest.mark.unit


class Provider:
    def __init__(self, guard: ProbeWatchdogProcess) -> None:
        self.guard = guard
        self.account_id = "test-account"
        self.error: Exception | None = None
        self.calls = 0
        self.template_ok = True

    def account(self) -> ProbeAccount:
        return ProbeAccount(self.account_id, 20, 0)

    def check_template(self, spec: H0ProbeLaunch) -> None:
        assert spec == self.guard.launch
        if not self.template_ok:
            raise ValueError("template mismatch")

    def create_once(self, spec: H0ProbeLaunch) -> ProbePod:
        assert read_probe_control(
            self.guard.root, spec, "controller", "create-intent", max_bytes=10_000
        ) == {"schema": "h-probe-create-intent-v1", "guard_pid": self.guard.pid}
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ProbePod(
            "owned-pod", spec.pod_name, str(spec.request.image), 1, "RUNNING", None, 1.0, "a" * 64
        )


@contextmanager
def rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[ProbeWatchdogProcess, Provider, _ObjectStore]]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    monkeypatch.setattr("entropic_substrate.ladder.run.h_probe_process._run_policy", fake_policy)
    spec, directory = launch(), root(tmp_path)
    objects = _ObjectStore()
    publish_verified_object(objects, f"{spec.request.prefix}/request.json", spec.request.payload)
    with supervise_probe_watchdog(
        directory, spec, config(tmp_path), expected_account_id="test-account"
    ) as guard:
        try:
            yield guard, Provider(guard), objects
        finally:
            guard.control.write("stop", {"reason": "test_only_cleanup"}, max_bytes=10_000)
            guard.wait_result(timeout_seconds=5)


def record(guard: ProbeWatchdogProcess, name: str) -> dict[str, object] | None:
    return read_probe_control(guard.root, guard.launch, "controller", name, max_bytes=10_000)


def test_success_hands_response_identity_to_watchdog_and_refuses_second_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with rig(tmp_path, monkeypatch) as (guard, provider, objects):
        pod = allocate_probe_once(guard, provider, objects)
        assert record(guard, "allocation") == {
            "schema": "h-probe-allocation-v2",
            "guard_pid": guard.pid,
            "outcome": "created",
            "pod_id": pod.pod_id,
            "response_sha256": "a" * 64,
            "provider_error_code": None,
            "allocation_uncertain": False,
            "possible_pod_id": None,
            "provider_detail": None,
            "provider_detail_sha256": None,
        }
        with pytest.raises(ValueError):
            allocate_probe_once(guard, provider, objects)
        assert provider.calls == 1


@pytest.mark.parametrize("fault", ["missing_request", "foreign_request", "account", "template"])
def test_preflight_defects_never_record_intent_or_call_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    with rig(tmp_path, monkeypatch) as (guard, provider, objects):
        if fault == "missing_request":
            objects.values.clear()
        elif fault == "foreign_request":
            objects.values[f"{guard.launch.request.prefix}/request.json"] = b"{}"
        elif fault == "account":
            provider.account_id = "different-account"
        else:
            provider.template_ok = False
        with pytest.raises(ValueError):
            allocate_probe_once(guard, provider, objects)
        assert provider.calls == 0 and record(guard, "create-intent") is None


@pytest.mark.parametrize("pod_id", [None, "owned-pod"])
def test_ambiguous_provider_error_preserves_attribution_and_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pod_id: str | None
) -> None:
    with rig(tmp_path, monkeypatch) as (guard, provider, objects):
        provider.error = ProbeProviderError(
            "network_error", allocation_uncertain=True, possible_pod_id=pod_id
        )
        with pytest.raises(ProbeProviderError) as caught:
            allocate_probe_once(guard, provider, objects)
        assert caught.value is provider.error
        detail = str(provider.error)
        assert record(guard, "allocation") == {
            "schema": "h-probe-allocation-v2",
            "guard_pid": guard.pid,
            "outcome": "uncertain",
            "pod_id": pod_id,
            "response_sha256": None,
            "provider_error_code": "network_error",
            "allocation_uncertain": True,
            "possible_pod_id": pod_id,
            "provider_detail": detail,
            "provider_detail_sha256": hashlib.sha256(detail.encode()).hexdigest(),
        }
        with pytest.raises(ValueError):
            allocate_probe_once(guard, provider, objects)
        assert provider.calls == 1


def test_unexpected_create_exception_is_not_relabelled_as_no_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with rig(tmp_path, monkeypatch) as (guard, provider, objects):
        provider.error = RuntimeError("injected crash")
        with pytest.raises(RuntimeError, match="injected crash"):
            allocate_probe_once(guard, provider, objects)
        row = record(guard, "allocation")
        assert row is not None and row["outcome"] == "uncertain" and row["pod_id"] is None
        # An exception outside the provider's own contract is conservatively
        # treated as possibly having created a pod, and its type name stands in
        # for a structured code.
        assert row["provider_error_code"] == "RuntimeError"
        assert row["allocation_uncertain"] is True
        assert row["possible_pod_id"] is None
        assert row["provider_detail"] == "injected crash"
        assert row["provider_detail_sha256"] == hashlib.sha256(b"injected crash").hexdigest()


def test_provider_error_code_prefers_the_structured_code_over_the_type_name() -> None:
    assert provider_error_code(None) is None
    assert provider_error_code(RuntimeError("boom")) == "RuntimeError"
    assert provider_error_code(ProbeProviderError("network_error")) == "network_error"


def test_provider_detail_is_bounded_to_512_bytes_with_a_full_text_digest() -> None:
    oversized = "x" * 600
    expected_digest = hashlib.sha256(oversized.encode()).hexdigest()

    detail, digest = _bounded_provider_detail(oversized)
    assert len(detail.encode("utf-8")) <= 512
    assert digest == expected_digest
    # The full text's digest, not the truncated text's own digest.
    assert digest != hashlib.sha256(detail.encode()).hexdigest()


def test_watchdog_allocation_reader_accepts_old_v1_and_new_v2_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-old, write-new: both schema shapes must round-trip through the reader
    that the watchdog actually uses (_allocation_message + _allocation)."""
    directory, spec = root(tmp_path), launch()
    with own_probe_role(directory, spec, "controller") as controller:
        controller.write(
            "allocation",
            {
                "schema": "h-probe-allocation-v1",
                "guard_pid": 4242,
                "outcome": "uncertain",
                "pod_id": None,
                "response_sha256": None,
            },
            max_bytes=10_000,
        )
    monkeypatch.setattr("os.getpid", lambda: 4242)
    row = _allocation_message(directory, spec, 10_000)
    assert row is not None
    pod_id, uncertain = _allocation(row)
    assert pod_id is None and uncertain is True

    (tmp_path / "v2").mkdir(mode=0o700)
    directory2 = root(tmp_path / "v2")
    with own_probe_role(directory2, spec, "controller") as controller:
        controller.write(
            "allocation",
            {
                "schema": "h-probe-allocation-v2",
                "guard_pid": 4242,
                "outcome": "uncertain",
                "pod_id": None,
                "response_sha256": None,
                "provider_error_code": "network_error",
                "allocation_uncertain": True,
                "possible_pod_id": None,
                "provider_detail": "H probe provider: network_error",
                "provider_detail_sha256": hashlib.sha256(
                    b"H probe provider: network_error"
                ).hexdigest(),
            },
            max_bytes=10_000,
        )
    row2 = _allocation_message(directory2, spec, 10_000)
    assert row2 is not None
    pod_id2, uncertain2 = _allocation(row2)
    assert pod_id2 is None and uncertain2 is True


@pytest.mark.parametrize("provider_fails", [False, True])
def test_failed_handoff_exposes_attributed_pod_without_claiming_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_fails: bool
) -> None:
    with rig(tmp_path, monkeypatch) as (guard, provider, objects):
        if provider_fails:
            provider.error = ProbeProviderError(
                "network_error", allocation_uncertain=True, possible_pod_id="owned-pod"
            )
        original = guard.control.write

        def fail(name: str, payload: dict[str, object], *, max_bytes: int) -> ArtifactIdentity:
            if name == "allocation":
                raise OSError("injected write failure")
            return original(name, payload, max_bytes=max_bytes)

        monkeypatch.setattr(guard.control, "write", fail)
        with pytest.raises(ProbeAllocationHandoffError) as caught:
            allocate_probe_once(guard, provider, objects)
        assert caught.value.pod_id == "owned-pod"
        assert caught.value.provider_error is provider.error
        assert isinstance(caught.value.__cause__, OSError)
        assert record(guard, "allocation") is None and guard.alive and provider.calls == 1


def test_liveness_failure_after_handoff_still_exposes_the_allocated_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with rig(tmp_path, monkeypatch) as (guard, provider, objects):
        original = guard.require_alive

        def fail_after_handoff() -> None:
            original()
            if record(guard, "allocation") is not None:
                raise ValueError("injected guardian loss after response handoff")

        monkeypatch.setattr(guard, "require_alive", fail_after_handoff)
        with pytest.raises(ProbeAllocationHandoffError) as caught:
            allocate_probe_once(guard, provider, objects)
        assert caught.value.pod_id == "owned-pod" and caught.value.provider_error is None
        assert isinstance(caught.value.__cause__, ValueError)
        assert record(guard, "allocation") is not None and provider.calls == 1
