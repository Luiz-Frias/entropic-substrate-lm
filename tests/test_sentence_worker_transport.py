"""Byte transfer ordering is tested separately from native model admission."""

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_request import request

from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest
from entropic_substrate.ladder.run.worker_transport import (
    VerificationReadRetry,
    materialize_probe_request,
    publish_probe_request,
    publish_verified_object,
)

pytestmark = pytest.mark.unit


class _ReadFailure(RuntimeError):
    pass


@dataclass
class _FlakyReadStore(_ObjectStore):
    """An ``_ObjectStore`` whose ``read`` fails N times per key, then succeeds."""

    fail_reads_remaining: dict[str, int] = field(default_factory=dict)
    read_calls: list[str] = field(default_factory=list)
    publish_calls: list[str] = field(default_factory=list)

    def publish(
        self, key: str, payload: bytes, *, expected_sha256: str
    ) -> D1TrainerMailboxObjectIdentity:
        self.publish_calls.append(key)
        return super().publish(key, payload, expected_sha256=expected_sha256)

    def read(self, key: str) -> D1TrainerMailboxObjectRead | None:
        self.read_calls.append(key)
        remaining = self.fail_reads_remaining.get(key, 0)
        if remaining > 0:
            self.fail_reads_remaining[key] = remaining - 1
            raise _ReadFailure("transient provider failure on a verification read")
        return super().read(key)


def test_default_retry_is_none_and_behaves_byte_for_byte_as_before(tmp_path: Path) -> None:
    store = _FlakyReadStore()
    identity = publish_verified_object(store, "k/one.json", b"payload")
    assert identity.key == "k/one.json"
    assert store.publish_calls == ["k/one.json"]
    assert store.read_calls == ["k/one.json"]


def test_two_transient_read_failures_then_success_records_two_retries(tmp_path: Path) -> None:
    store = _FlakyReadStore(fail_reads_remaining={"k/two.json": 2})
    retry = VerificationReadRetry(sleep=lambda seconds: None)
    identity = publish_verified_object(store, "k/two.json", b"payload", retry=retry)
    assert identity.key == "k/two.json"
    assert store.publish_calls == ["k/two.json"]
    assert retry.failures == 2


def test_three_transient_read_failures_raise_the_original_class(tmp_path: Path) -> None:
    store = _FlakyReadStore(fail_reads_remaining={"k/three.json": 3})
    retry = VerificationReadRetry(sleep=lambda seconds: None)
    with pytest.raises(_ReadFailure):
        publish_verified_object(store, "k/three.json", b"payload", retry=retry)
    assert store.publish_calls == ["k/three.json"]
    assert retry.failures == 3


def test_a_read_mismatch_is_never_retried(tmp_path: Path) -> None:
    class MismatchedStore(_FlakyReadStore):
        def read(self, key: str) -> D1TrainerMailboxObjectRead | None:
            self.read_calls.append(key)
            return D1TrainerMailboxObjectRead(
                D1TrainerMailboxObjectIdentity(key, 1, sha256(b"x").hexdigest()), b"x"
            )

    store = MismatchedStore()
    retry = VerificationReadRetry(sleep=lambda seconds: None)
    with pytest.raises(ValueError, match="readback"):
        publish_verified_object(store, "k/four.json", b"payload", retry=retry)
    assert len(store.read_calls) == 1
    assert retry.failures == 0


def test_retry_never_re_issues_the_put(tmp_path: Path) -> None:
    store = _FlakyReadStore(fail_reads_remaining={"k/five.json": 2})
    retry = VerificationReadRetry(sleep=lambda seconds: None)
    publish_verified_object(store, "k/five.json", b"payload", retry=retry)
    assert store.publish_calls == ["k/five.json"]


def _real_adapter(client: object) -> object:
    from entropic_substrate.ladder.run.transport.d1_trainer_s3_object_adapter import (
        Boto3D1TrainerMailboxObjectAdapter,
    )

    return Boto3D1TrainerMailboxObjectAdapter(
        client=client, bucket="fixture-bucket", max_objects=64
    )


def test_real_adapters_own_post_put_readback_failure_recovers_through_retry() -> None:
    """C6/C7: the FIRST get_object call(s) after put_object are the adapter's
    OWN internal post-PUT readback inside ``publish()`` (:263), not the
    explicit ``objects.read`` this module calls afterwards. A single
    injected failure there must still be recovered by this module's retry,
    without re-issuing ``put_object`` and without importing the adapter."""
    from tests.test_d1_trainer_s3_object_adapter import _S3Client

    client = _S3Client(fail_get_times=1)
    adapter = _real_adapter(client)
    retry = VerificationReadRetry(sleep=lambda seconds: None)
    identity = publish_verified_object(adapter, "k/one.json", b"payload", retry=retry)
    assert identity.size_bytes == len(b"payload")
    assert len(client.puts) == 1
    assert retry.failures == 1


def test_real_adapter_two_failures_then_success_records_two_and_puts_once() -> None:
    from tests.test_d1_trainer_s3_object_adapter import _S3Client

    client = _S3Client(fail_get_times=2)
    adapter = _real_adapter(client)
    retry = VerificationReadRetry(sleep=lambda seconds: None)
    identity = publish_verified_object(adapter, "k/two.json", b"payload", retry=retry)
    assert identity.size_bytes == len(b"payload")
    assert len(client.puts) == 1
    assert retry.failures == 2


def test_real_adapter_exhaustion_raises_the_original_d1_s3_object_error_class() -> None:
    from tests.test_d1_trainer_s3_object_adapter import _S3Client

    from entropic_substrate.ladder.run.transport.d1_trainer_s3_object_adapter import (
        D1TrainerS3ObjectError,
    )

    client = _S3Client(fail_get_times=3)
    adapter = _real_adapter(client)
    retry = VerificationReadRetry(sleep=lambda seconds: None)
    with pytest.raises(D1TrainerS3ObjectError, match="provider_failure"):
        publish_verified_object(adapter, "k/three.json", b"payload", retry=retry)
    assert len(client.puts) == 1
    assert retry.failures == 3


def inputs(root: Path) -> ProbeRequest:
    (root / "source").mkdir()
    data = b"byte-transfer fixture; not a native corpus"
    publish_d1_atomic_bytes(root, "source/prepared-corpus.json", data, label="test input")
    digest = sha256(data).hexdigest()
    value = request()
    return replace(
        value,
        profile=replace(value.profile, source_digest=digest),
        inputs=(ArtifactIdentity("source/prepared-corpus.json", ContentHash("sha256", digest)),),
    )


def test_inputs_precede_request_marker_and_materialize_exactly(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    store = _ObjectStore()
    marker = publish_probe_request(store, value, tmp_path)
    assert store.publishes[-1] == f"{value.prefix}/request.json"
    target = tmp_path / "worker"
    target.mkdir()
    observed = materialize_probe_request(store, marker, target, max_request_bytes=1_000_000)
    assert observed == value
    assert (target / "source/prepared-corpus.json").read_bytes() == (
        tmp_path / "source/prepared-corpus.json"
    ).read_bytes()
    assert (target / "request.json").read_bytes() == value.payload
    with pytest.raises(ValueError, match="empty"):
        materialize_probe_request(store, marker, target, max_request_bytes=1_000_000)


def test_tampering_cannot_publish_a_materialization_marker(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    store = _ObjectStore()
    marker = publish_probe_request(store, value, tmp_path)
    store.values[f"{value.prefix}/inputs/source/prepared-corpus.json"] = b"different"
    target = tmp_path / "worker"
    target.mkdir()
    with pytest.raises(ValueError, match="identity"):
        materialize_probe_request(store, marker, target, max_request_bytes=1_000_000)
    assert not (target / "request.json").exists()


def test_budget_failure_precedes_all_remote_writes(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    store = _ObjectStore()
    value = replace(value, limits=replace(value.limits, max_total_bytes=1))
    with pytest.raises(ValueError, match="budget"):
        publish_probe_request(store, value, tmp_path)
    assert store.publishes == []


def test_local_input_hash_is_checked_before_publication(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    (tmp_path / "source/prepared-corpus.json").write_bytes(b"replaced")
    store = _ObjectStore()
    with pytest.raises(ValueError, match="identity"):
        publish_probe_request(store, value, tmp_path)
    assert store.publishes == []


def test_missing_readback_never_publishes_the_request_marker(tmp_path: Path) -> None:
    class DroppingStore(_ObjectStore):
        def publish(
            self, key: str, payload: bytes, *, expected_sha256: str
        ) -> D1TrainerMailboxObjectIdentity:
            result = super().publish(key, payload, expected_sha256=expected_sha256)
            del self.values[key]
            return result

    value = inputs(tmp_path)
    store = DroppingStore()
    with pytest.raises(ValueError, match="readback"):
        publish_probe_request(store, value, tmp_path)
    assert f"{value.prefix}/request.json" not in store.publishes


def test_request_namespace_cannot_be_relabelled(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    store = _ObjectStore()
    wrong = store.publish("other/request.json", value.payload, expected_sha256=value.digest)
    target = tmp_path / "worker"
    target.mkdir()
    with pytest.raises(ValueError, match="prefix"):
        materialize_probe_request(store, wrong, target, max_request_bytes=1_000_000)
    assert not list(target.iterdir())
