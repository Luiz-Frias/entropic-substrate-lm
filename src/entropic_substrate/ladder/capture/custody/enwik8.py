"""enwik8 (first 100MB of English Wikipedia) byte-level loader.

The raw file is fetched by ``scripts/download_enwik8.py`` into ``data/enwik8``
and then version-tracked with ``dvc add data/enwik8``; this module only reads
whatever is on disk, memory-mapped so a 100MB corpus costs no RSS.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

__all__ = [
    "ENWIK8_RAW_BYTES",
    "ENWIK8_URL",
    "ENWIK8_ZIP_SHA256",
    "download_enwik8",
    "load_enwik8_bytes",
]

ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
ENWIK8_ZIP_SHA256 = (
    "547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc"  # pragma: allowlist secret
)
# enwik8 is defined as exactly the first 10^8 bytes of an English Wikipedia
# dump, so the uncompressed size is the integrity check we can assert on.
ENWIK8_RAW_BYTES = 100_000_000


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download_enwik8(dest_dir: str | os.PathLike[str] = "data/enwik8") -> Path:
    """Download + unzip enwik8 into ``dest_dir``; returns the raw file path.

    Idempotent: an existing, non-empty ``enwik8`` file short-circuits the
    download.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    raw = dest / "enwik8"
    if raw.exists() and raw.stat().st_size > 0:
        return raw

    archive = dest / "enwik8.zip"
    if not archive.exists() or archive.stat().st_size == 0:
        tmp = archive.with_suffix(".zip.part")
        # The URL is a fixed HTTPS constant and the downloaded bytes are
        # authenticated against ENWIK8_ZIP_SHA256 immediately below.
        with urllib.request.urlopen(ENWIK8_URL) as response, tmp.open("wb") as out:  # nosec B310
            shutil.copyfileobj(response, out)
        tmp.replace(archive)

    archive_sha256 = _sha256(archive)
    if archive_sha256 != ENWIK8_ZIP_SHA256:
        msg = f"enwik8 archive SHA-256 mismatch: expected {ENWIK8_ZIP_SHA256}, got {archive_sha256}"
        raise RuntimeError(msg)

    with zipfile.ZipFile(archive) as zf:
        with zf.open("enwik8") as src, raw.open("wb") as out:
            shutil.copyfileobj(src, out)

    size = raw.stat().st_size
    if size != ENWIK8_RAW_BYTES:
        msg = f"expected {ENWIK8_RAW_BYTES} bytes of enwik8, extracted {size}"
        raise RuntimeError(msg)
    # Recorded so `dvc add data/enwik8` output can be cross-checked by hand.
    (dest / "enwik8.sha256").write_text(f"{_sha256(raw)}  enwik8\n", encoding="utf-8")
    return raw


def load_enwik8_bytes(
    path: str | os.PathLike[str] = "data/enwik8/enwik8",
    max_bytes: int | None = None,
) -> np.ndarray:
    """Memory-map the enwik8 byte stream as ``uint8``."""
    file_path = Path(path)
    if not file_path.exists():
        msg = (
            f"{file_path} not found - run `python scripts/download_enwik8.py` "
            "(or `just download-data`) first"
        )
        raise FileNotFoundError(msg)
    data = np.memmap(file_path, dtype=np.uint8, mode="r")
    if max_bytes is not None:
        data = data[:max_bytes]
    return data
