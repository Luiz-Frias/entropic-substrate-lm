"""Provider-neutral objects satisfy the raw D1 trainer transport contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.ladder.run.transport.d1_trainer_evidence_adapter import (
    D1TrainerEvidenceAdapter,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxErrorReport,
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
    d1_trainer_mailbox_attempt_prefix,
    d1_trainer_mailbox_error_path,
    serialize_d1_trainer_mailbox_error,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1ObjectMailboxTrainerTransport,
    D1ObjectMailboxTransportError,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    d1_trainer_request_sha256,
)

from .test_d1_trainer_process_worker import (
    _ATTEMPT,
    _POD,
    _fresh_request,
    _stage,
    _worker,
)

pytestmark = pytest.mark.unit


@dataclass
class _ObjectStore:
    values: dict[str, bytes] = field(default_factory=dict)
    publishes: list[str] = field(default_factory=list)
    foreign_listing: bool = False

    @staticmethod
    def _identity(key: str, payload: bytes) -> D1TrainerMailboxObjectIdentity:
        return D1TrainerMailboxObjectIdentity(
            key,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )

    def seed(self, key: str, payload: bytes) -> None:
        assert key not in self.values
        self.values[key] = payload

    def publish(
        self,
        key: str,
        payload: bytes,
        *,
        expected_sha256: str,
    ) -> D1TrainerMailboxObjectIdentity:
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        if key in self.values and self.values[key] != payload:
            raise ValueError("immutable object collision")
        self.values[key] = payload
        self.publishes.append(key)
        return self._identity(key, payload)

    def read(self, key: str) -> D1TrainerMailboxObjectRead | None:
        payload = self.values.get(key)
        if payload is None:
            return None
        return D1TrainerMailboxObjectRead(self._identity(key, payload), payload)

    def list(self, prefix: str) -> tuple[D1TrainerMailboxObjectIdentity, ...]:
        values = tuple(
            self._identity(key, payload)
            for key, payload in sorted(self.values.items())
            if key.startswith(prefix)
        )
        if self.foreign_listing:
            return (
                *values,
                D1TrainerMailboxObjectIdentity(
                    "foreign/object", 1, hashlib.sha256(b"x").hexdigest()
                ),
            )
        return values


def _seed_real_output(store: _ObjectStore, source: Path, prefix: str) -> object:
    request = _fresh_request(source)
    _stage(source, request)
    worker = _worker(source, request)
    worker.bootstrap(_POD, _ATTEMPT)
    worker.start_trainer(_POD, _ATTEMPT, d1_trainer_request_sha256(request))
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        if relative.startswith("requests/"):
            continue
        store.seed(f"{prefix}/{relative}", path.read_bytes())
    return request


def _transport(
    tmp_path: Path,
    store: _ObjectStore,
    **kwargs: object,
) -> D1ObjectMailboxTrainerTransport:
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    return D1ObjectMailboxTrainerTransport(
        objects=store,
        materialization_root=materialized,
        wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
        **kwargs,
    )


def test_object_transport_drives_existing_evidence_adapter_and_real_export(
    tmp_path: Path,
) -> None:
    store = _ObjectStore()
    prefix = d1_trainer_mailbox_attempt_prefix(_POD, _ATTEMPT)
    source = tmp_path / "source"
    source.mkdir()
    request = _seed_real_output(store, source, prefix)
    transport = _transport(tmp_path, store)
    adapter = D1TrainerEvidenceAdapter(
        transport=transport,
        catalog=(request.plan.profile,),
    )

    runtime = adapter.bootstrap_runtime(_POD, _ATTEMPT)
    staged = adapter.stage_inputs(_POD, request=request)
    started = adapter.start_trainer(_POD)
    candidate = adapter.await_exported_handoff(_POD)

    assert runtime.pod_id == _POD
    assert staged.request_sha256 == d1_trainer_request_sha256(request)
    assert started.request_sha256 == staged.request_sha256
    assert candidate.handoff.successful_step == 2
    assert candidate.journal_path.is_relative_to(tmp_path / "materialized")
    assert store.publishes == [
        f"{prefix}/commands/bootstrap.json",
        f"{prefix}/requests/{_ATTEMPT}.json",
        f"{prefix}/commands/start-{staged.request_sha256}.json",
    ]


def test_object_transport_surfaces_typed_worker_error_without_detail(tmp_path: Path) -> None:
    store = _ObjectStore()
    report = D1TrainerMailboxErrorReport(_POD, _ATTEMPT, "bootstrap", "ValueError")
    store.seed(
        d1_trainer_mailbox_error_path(_POD, _ATTEMPT),
        serialize_d1_trainer_mailbox_error(report),
    )
    transport = _transport(tmp_path, store)
    transport.bootstrap(_POD, _ATTEMPT)

    with pytest.raises(D1ObjectMailboxTransportError) as raised:
        transport.read_runtime_report(_POD, _ATTEMPT)

    assert raised.value.stage == "runtime"
    assert raised.value.code == "worker_bootstrap_ValueError"
    assert "detail" not in str(raised.value)


@dataclass
class _FakeTime:
    value: float = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_object_transport_timeout_is_bounded_and_does_not_republish(tmp_path: Path) -> None:
    store = _ObjectStore()
    timer = _FakeTime()
    transport = _transport(
        tmp_path,
        store,
        monotonic=timer.monotonic,
        sleep=timer.sleep,
    )
    transport.bootstrap(_POD, _ATTEMPT)

    with pytest.raises(D1ObjectMailboxTransportError, match="timed_out"):
        transport.read_runtime_report(_POD, _ATTEMPT)

    assert len(store.publishes) == 1
    assert timer.value >= 1.0


def test_object_transport_rejects_wrong_order_and_identity_without_effects(
    tmp_path: Path,
) -> None:
    store = _ObjectStore()
    transport = _transport(tmp_path, store)

    with pytest.raises(D1ObjectMailboxTransportError, match="session_missing"):
        transport.read_request(_POD, _ATTEMPT, f"requests/{_ATTEMPT}.json")
    transport.bootstrap(_POD, _ATTEMPT)
    with pytest.raises(D1ObjectMailboxTransportError, match="session_exists"):
        transport.bootstrap(_POD, _ATTEMPT)
    with pytest.raises(D1ObjectMailboxTransportError, match="request_path"):
        transport.write_request(_POD, _ATTEMPT, "requests/foreign.json", b"request")
    with pytest.raises(D1ObjectMailboxTransportError, match="request_missing"):
        transport.start_trainer(_POD, _ATTEMPT, "b" * 64)
    assert len(store.publishes) == 1
