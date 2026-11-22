"""``read_json_artifact`` is an evidence-reading path: it admits only finite, unique JSON.

The stdlib decoder accepts the ``NaN``, ``Infinity`` and ``-Infinity`` literals and
decodes an overflowing literal such as ``1e999`` to ``inf``; a ladder artifact carrying
either would reach a gate as a non-finite measurement. The refusal lives inside
``kernel.artifacts`` itself, a hashed member of both H reader identities, so the
policy is bound by the identity that pins the reader.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes, read_json_artifact
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash

pytestmark = pytest.mark.unit


def _published(root: Path, payload: bytes) -> ArtifactIdentity:
    identity = publish_d1_atomic_bytes(root, "artifact.json", payload, label="test artifact")
    assert identity.content_hash == ContentHash("sha256", hashlib.sha256(payload).hexdigest())
    return identity


def test_a_finite_unique_object_is_admitted(tmp_path: Path) -> None:
    artifact = _published(tmp_path, b'{"loss": 1.5, "step": 2, "rows": [0.0, -3e-7]}')
    assert read_json_artifact(tmp_path, artifact) == {
        "loss": 1.5,
        "step": 2,
        "rows": [0.0, -3e-7],
    }


@pytest.mark.parametrize(
    "payload",
    [
        b'{"loss": NaN}',
        b'{"loss": Infinity}',
        b'{"loss": -Infinity}',
        b'{"rows": [{"loss": NaN}]}',
        b'{"loss": 1e999}',
        b'{"loss": -1e999}',
    ],
)
def test_a_non_finite_value_is_refused(tmp_path: Path, payload: bytes) -> None:
    artifact = _published(tmp_path, payload)
    with pytest.raises(ValueError, match="non-finite"):
        read_json_artifact(tmp_path, artifact)


def test_a_duplicate_key_is_still_refused(tmp_path: Path) -> None:
    artifact = _published(tmp_path, b'{"loss": 1.0, "loss": 2.0}')
    with pytest.raises(ValueError, match="duplicate ladder JSON key"):
        read_json_artifact(tmp_path, artifact)
