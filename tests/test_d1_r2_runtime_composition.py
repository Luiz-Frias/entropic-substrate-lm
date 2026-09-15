"""D1 R2 runtime composition joins committed topology to environment credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.d1_r2_runtime_composition import (
    D1R2RuntimeCompositionError,
    build_d1_r2_runtime_composition,
)

pytestmark = pytest.mark.unit

_ACCESS = "r2-access-id"
_PRIVATE = "r2-private-value"


@dataclass
class _Client:
    def put_object(self, **_kwargs: object) -> dict[str, object]:
        return {}

    def get_object(self, **_kwargs: object) -> dict[str, object]:
        return {}

    def head_object(self, **_kwargs: object) -> dict[str, object]:
        return {}

    def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
        return {"Contents": [], "IsTruncated": False}


@dataclass
class _ClientFactory:
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    fail: bool = False

    def __call__(self, service: str, **kwargs: object) -> _Client:
        self.calls.append((service, dict(kwargs)))
        if self.fail:
            raise RuntimeError(_PRIVATE)
        return _Client()


class _Dvc:
    def add(self, **_kwargs: object) -> None:
        pass

    def push(self, **_kwargs: object) -> None:
        pass

    def pull(self, **_kwargs: object) -> None:
        pass


class _Cloud:
    def object_is_present(self, **_kwargs: object) -> bool:
        return True


def _build(repo: Path, factory: _ClientFactory, environ: dict[str, str]):
    return build_d1_r2_runtime_composition(
        repo_root=repo,
        environ=environ,
        client_factory=factory,
        dvc=_Dvc(),
        cloud=_Cloud(),
        max_objects=128,
    )


def test_composition_uses_committed_topology_and_environment_credentials() -> None:
    repo = Path(__file__).resolve().parents[1]
    factory = _ClientFactory()

    composition = _build(
        repo,
        factory,
        {
            "AWS_ACCESS_KEY_ID": _ACCESS,
            "AWS_SECRET_ACCESS_KEY": _PRIVATE,
        },
    )

    assert composition.topology.manifest == {
        "bucket": "entropic-substrate-lm",
        "endpoint_url": ("https://62c7fd87a1874f1dee6b149ad40ac657.r2.cloudflarestorage.com"),
        "remote_name": "r2-entropic-substrate-lm",
    }
    assert factory.calls == [
        (
            "s3",
            {
                "aws_access_key_id": _ACCESS,
                "aws_secret_access_key": _PRIVATE,
                "endpoint_url": composition.topology.endpoint_url,
                "region_name": "auto",
            },
        )
    ]
    assert _ACCESS not in repr(composition) and _PRIVATE not in repr(composition)
    assert callable(composition.checkpoint_recovery.latest)
    assert callable(composition.checkpoint_recovery.restore)
    bound = composition.bind_checkpoint_sync(
        pod_id="pod-r2-123",
        attempt_sha256="a" * 64,
        logical_run_id="d1-confirmatory-both-s101",
    )
    assert callable(bound.sync)
    assert _ACCESS not in repr(bound) and _PRIVATE not in repr(bound)


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"AWS_ACCESS_KEY_ID": _ACCESS},
        {"AWS_SECRET_ACCESS_KEY": _PRIVATE},
        {"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": _PRIVATE},
        {"AWS_ACCESS_KEY_ID": _ACCESS, "AWS_SECRET_ACCESS_KEY": ""},
    ],
)
def test_composition_requires_both_nonempty_environment_credentials_before_client(
    environ: dict[str, str],
) -> None:
    factory = _ClientFactory()
    with pytest.raises(D1R2RuntimeCompositionError, match="credentials"):
        _build(Path(__file__).resolve().parents[1], factory, environ)
    assert factory.calls == []


def test_composition_sanitizes_client_factory_failure() -> None:
    factory = _ClientFactory(fail=True)
    with pytest.raises(D1R2RuntimeCompositionError) as raised:
        _build(
            Path(__file__).resolve().parents[1],
            factory,
            {
                "AWS_ACCESS_KEY_ID": _ACCESS,
                "AWS_SECRET_ACCESS_KEY": _PRIVATE,
            },
        )
    assert raised.value.code == "client"
    assert _PRIVATE not in str(raised.value)


def test_composition_rejects_invalid_ports_and_object_bound_without_client() -> None:
    factory = _ClientFactory()
    kwargs = {
        "repo_root": Path(__file__).resolve().parents[1],
        "environ": {
            "AWS_ACCESS_KEY_ID": _ACCESS,
            "AWS_SECRET_ACCESS_KEY": _PRIVATE,
        },
        "client_factory": factory,
        "dvc": object(),
        "cloud": _Cloud(),
        "max_objects": 0,
    }
    with pytest.raises((TypeError, ValueError)):
        build_d1_r2_runtime_composition(**kwargs)  # type: ignore[arg-type]
    assert factory.calls == []
