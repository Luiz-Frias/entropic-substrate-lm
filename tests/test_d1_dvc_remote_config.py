"""D1 R2 topology comes only from the committed DVC remote configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.ladder.run.transport.d1_dvc_remote_config import (
    D1DvcRemoteConfigError,
    load_d1_dvc_s3_remote,
)

pytestmark = pytest.mark.unit


def _repo(tmp_path: Path, body: str) -> Path:
    repo = tmp_path / "repo"
    config = repo / ".dvc/config"
    config.parent.mkdir(parents=True)
    config.write_text(body, encoding="utf-8")
    return repo


def test_repository_dvc_config_resolves_cloudflare_r2_without_credentials() -> None:
    repo = Path(__file__).resolve().parents[1]

    remote = load_d1_dvc_s3_remote(repo)

    assert remote.remote_name == "r2-entropic-substrate-lm"
    assert remote.bucket == "entropic-substrate-lm"
    assert remote.endpoint_url == (
        "https://62c7fd87a1874f1dee6b149ad40ac657.r2.cloudflarestorage.com"
    )
    assert set(remote.manifest) == {"bucket", "endpoint_url", "remote_name"}


@pytest.mark.parametrize(
    "body",
    [
        "",
        "[core]\nremote = missing\n",
        "[core]\nremote = x\n['remote \"x\"']\nurl = s3://bucket\n",
        "[core]\nremote = x\n['remote \"x\"']\nurl = gs://bucket\nendpointurl = https://x.example\n",
        "[core]\nremote = x\n['remote \"x\"']\nurl = s3://bucket/prefix\nendpointurl = https://x.example\n",
        "[core]\nremote = x\n['remote \"x\"']\nurl = s3://bucket\nendpointurl = http://x.example\n",
        (
            "[core]\nremote = x\n['remote \"x\"']\nurl = s3://bucket\n"
            "endpointurl = https://x.example\naccess_key_id = forbidden\n"
        ),
    ],
)
def test_dvc_remote_reader_refuses_missing_foreign_or_secret_bearing_config(
    tmp_path: Path,
    body: str,
) -> None:
    with pytest.raises(D1DvcRemoteConfigError):
        load_d1_dvc_s3_remote(_repo(tmp_path, body))


def test_dvc_remote_reader_refuses_symlinked_config(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    target = tmp_path / "outside-config"
    target.write_text("[core]\nremote = x\n", encoding="utf-8")
    config = repo / ".dvc/config"
    config.parent.mkdir(parents=True)
    config.symlink_to(target)

    with pytest.raises(D1DvcRemoteConfigError, match="regular"):
        load_d1_dvc_s3_remote(repo)
