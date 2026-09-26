"""Strict credential-free topology reader for the committed D1 DVC S3 remote."""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["D1DvcRemoteConfigError", "D1DvcS3Remote", "load_d1_dvc_s3_remote"]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUCKET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REMOTE_KEYS = frozenset({"endpointurl", "url"})
_MAX_CONFIG_BYTES = 1024 * 1024


class D1DvcRemoteConfigError(ValueError):
    """The committed DVC remote is absent, ambiguous, unsafe, or secret-bearing."""


@dataclass(frozen=True, slots=True)
class D1DvcS3Remote:
    remote_name: str
    bucket: str
    endpoint_url: str

    def __post_init__(self) -> None:
        if _NAME.fullmatch(self.remote_name) is None:
            raise ValueError("DVC remote name is unsafe")
        if _BUCKET.fullmatch(self.bucket) is None:
            raise ValueError("DVC S3 bucket is unsafe")
        endpoint = urlsplit(self.endpoint_url)
        if (
            endpoint.scheme != "https"
            or not endpoint.netloc
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path not in {"", "/"}
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("DVC S3 endpoint must be one credential-free HTTPS origin")

    @property
    def manifest(self) -> dict[str, str]:
        return {
            "bucket": self.bucket,
            "endpoint_url": self.endpoint_url,
            "remote_name": self.remote_name,
        }


def _config_path(repo_root: object) -> Path:
    if not isinstance(repo_root, Path):
        raise TypeError("DVC remote repo_root must be a Path")
    root = Path(os.path.abspath(repo_root))
    if root.is_symlink() or not root.is_dir():
        raise D1DvcRemoteConfigError("DVC remote repository root must be a real directory")
    config = root / ".dvc/config"
    if config.is_symlink() or not config.is_file():
        raise D1DvcRemoteConfigError("DVC remote config must be a regular non-symlink file")
    if config.stat().st_size > _MAX_CONFIG_BYTES:
        raise D1DvcRemoteConfigError("DVC remote config exceeds its byte bound")
    return config


def load_d1_dvc_s3_remote(repo_root: Path) -> D1DvcS3Remote:
    """Resolve only bucket and endpoint topology; credentials remain environment-owned."""
    config = _config_path(repo_root)
    parser = configparser.RawConfigParser(interpolation=None, strict=True)
    try:
        with config.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, UnicodeError, configparser.Error) as error:
        raise D1DvcRemoteConfigError("DVC remote config is invalid UTF-8 INI") from error
    if not parser.has_section("core"):
        raise D1DvcRemoteConfigError("DVC remote config lacks core")
    remote_name = parser.get("core", "remote", fallback="")
    if _NAME.fullmatch(remote_name) is None:
        raise D1DvcRemoteConfigError("DVC default remote name is absent or unsafe")
    section = f"'remote \"{remote_name}\"'"
    if not parser.has_section(section):
        raise D1DvcRemoteConfigError("DVC default remote section is absent")
    values = dict(parser[section])
    if frozenset(values) != _REMOTE_KEYS:
        raise D1DvcRemoteConfigError(
            "DVC remote fields must be exactly credential-free url and endpointurl"
        )
    remote_url = urlsplit(values["url"])
    if (
        remote_url.scheme != "s3"
        or _BUCKET.fullmatch(remote_url.netloc) is None
        or remote_url.path not in {"", "/"}
        or remote_url.query
        or remote_url.fragment
    ):
        raise D1DvcRemoteConfigError("DVC remote url must name one unprefixed S3 bucket")
    try:
        return D1DvcS3Remote(remote_name, remote_url.netloc, values["endpointurl"])
    except ValueError as error:
        raise D1DvcRemoteConfigError(str(error)) from error
