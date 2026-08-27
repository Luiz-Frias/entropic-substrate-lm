"""D1 object export materialization rejects foreign or unsafe custody."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.d1_trainer_mailbox_contracts import (
    d1_trainer_mailbox_attempt_prefix,
)
from entropic_substrate.d1_trainer_mailbox_worker import D1TrainerMailboxWaitPolicy
from entropic_substrate.d1_trainer_object_transport import (
    D1ObjectMailboxTrainerTransport,
    D1ObjectMailboxTransportError,
)
from entropic_substrate.d1_trainer_request_codec import d1_trainer_request_sha256

from .test_d1_trainer_object_transport import (
    _ObjectStore,
    _seed_real_output,
    _transport,
)
from .test_d1_trainer_process_worker import _ATTEMPT, _POD

pytestmark = pytest.mark.unit


def test_export_materialization_refuses_foreign_listing_before_marker_publish(
    tmp_path: Path,
) -> None:
    store = _ObjectStore(foreign_listing=True)
    prefix = d1_trainer_mailbox_attempt_prefix(_POD, _ATTEMPT)
    source = tmp_path / "source"
    source.mkdir()
    request = _seed_real_output(store, source, prefix)
    transport = _transport(tmp_path, store)
    transport.bootstrap(_POD, _ATTEMPT)
    payload = (source / f"requests/{_ATTEMPT}.json").read_bytes()
    transport.write_request(
        _POD,
        _ATTEMPT,
        f"requests/{_ATTEMPT}.json",
        payload,
    )
    transport.start_trainer(_POD, _ATTEMPT, d1_trainer_request_sha256(request))

    with pytest.raises(D1ObjectMailboxTransportError, match="foreign_object"):
        transport.await_export(_POD, _ATTEMPT, request.locations.checkpoint_path)

    destination = tmp_path / "materialized" / _POD / _ATTEMPT
    assert not destination.exists()
    assert not (destination / f"{request.locations.checkpoint_path}.complete.json").exists()


def test_object_transport_requires_exact_port_and_real_materialization_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="object port"):
        D1ObjectMailboxTrainerTransport(
            objects=object(),
            materialization_root=tmp_path,
            wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.1),
        )
    linked = tmp_path / "linked"
    target = tmp_path / "target"
    target.mkdir()
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="materialization"):
        D1ObjectMailboxTrainerTransport(
            objects=_ObjectStore(),
            materialization_root=linked,
            wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.1),
        )
