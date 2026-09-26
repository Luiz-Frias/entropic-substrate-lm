"""Shared atomic bytes beneath D1 content-addressed custody adapters."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.custody.d1_content_addressed_custody import (
    D1ContentAddressedCustodyError,
    D1ContentAddressedCustodyStore,
)

pytestmark = pytest.mark.unit


def test_store_publishes_reads_and_reuses_exact_bytes(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    payload = b'{"canonical":true}'
    digest = hashlib.sha256(payload).hexdigest()
    store = D1ContentAddressedCustodyStore(root)

    first = store.publish(payload, expected_sha256=digest)
    second = store.publish(payload, expected_sha256=digest)

    assert first == second
    assert first.path == str(root / f"{digest}.json")
    assert first.content_hash == ContentHash("sha256", digest)
    assert store.read(first) == payload


def test_store_refuses_wrong_expected_hash_and_substituted_artifact(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    payload = b"bytes"
    digest = hashlib.sha256(payload).hexdigest()
    store = D1ContentAddressedCustodyStore(root)

    with pytest.raises(D1ContentAddressedCustodyError, match="hash"):
        store.publish(payload, expected_sha256="0" * 64)
    artifact = store.publish(payload, expected_sha256=digest)
    for candidate in (
        replace(artifact, content_hash=ContentHash("sha256", "0" * 64)),
        ArtifactIdentity(str(tmp_path / "elsewhere.json"), artifact.content_hash),
    ):
        with pytest.raises(D1ContentAddressedCustodyError, match="artifact"):
            store.read(candidate)


def test_closed_store_refuses_new_reads_or_publications(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    store = D1ContentAddressedCustodyStore(root)
    store.close()

    with pytest.raises(D1ContentAddressedCustodyError, match="closed"):
        store.publish(b"x", expected_sha256=hashlib.sha256(b"x").hexdigest())
    with pytest.raises(D1ContentAddressedCustodyError, match="closed"):
        store.read(
            ArtifactIdentity(str(root / f"{'0' * 64}.json"), ContentHash("sha256", "0" * 64))
        )
