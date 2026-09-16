"""Base-image source bootstrap: stdlib only, no imports from the incoming package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import selectors
import signal
import stat
import subprocess  # nosec B404 -- bounded fixed Git argv; no shell or checkout hooks
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

PREFIXES = (
    "src/entropic_substrate",
    "configs",
    "scripts",
    "pyproject.toml",
    "uv.lock",
    ".dvc/config",
)
REQUIRED = {
    "pyproject.toml",
    "uv.lock",
    ".dvc/config",
    "src/entropic_substrate/sentence_prediction/worker_command.py",
}


def _directory(path: Path) -> Path:
    selected = Path(os.path.abspath(path))
    if any(item.is_symlink() for item in (selected, *selected.parents)) or not selected.is_dir():
        raise ValueError("bootstrap directory must exist without symlinks")
    return selected


def _write(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".h-bootstrap-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def _git(root: Path, args: tuple[str, ...], *, timeout: float, max_bytes: int) -> bytes:
    """Bound Git output and the entire process group before any package is available."""
    argv = (
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.attributesFile=/dev/null",
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.file.allow=always",
        *args,
    )
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    process = subprocess.Popen(  # nosec B603 -- fixed Git operation, validated operands, no shell
        argv,
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        if process.stdout is None:
            raise RuntimeError("bootstrap Git pipe unavailable")
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("bootstrap Git deadline exceeded")
                for key, _ in selector.select(min(remaining, 0.25)):
                    chunk = os.read(key.fd, min(65536, max_bytes - len(output) + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output.extend(chunk)
                    if len(output) > max_bytes:
                        raise RuntimeError("bootstrap Git output limit exceeded")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or process.wait(timeout=remaining) != 0:
                raise RuntimeError("bootstrap Git operation failed")
        return bytes(output)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)
        if process.stdout is not None:
            process.stdout.close()


def materialize_source(
    bundle: Path,
    root: Path,
    commit: str,
    expected_sha256: str,
    *,
    max_bundle_bytes: int,
    max_source_bytes: int,
    max_files: int,
    timeout_seconds: float,
) -> dict[str, object]:
    """Verify a bundle and export its runtime blobs into a new owned workspace.

    No checkout, install, training or network occurs here. Failed extractions
    retain partial evidence without a readiness marker; retry in a new root.
    """
    if (
        any(type(v) is not int or v < 1 for v in (max_bundle_bytes, max_source_bytes, max_files))
        or type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("bootstrap limits must be finite positive numbers")
    if (
        type(commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None
        or type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("bootstrap requires immutable commit and SHA-256 identities")
    root = _directory(root)
    if any(root.iterdir()):
        raise ValueError("bootstrap requires an empty owned directory")
    _directory(bundle.parent)
    descriptor = os.open(bundle, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max_bundle_bytes:
            raise ValueError("bootstrap bundle is not a bounded regular file")
        payload = stream.read(max_bundle_bytes + 1)
    if len(payload) != info.st_size or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("bootstrap bundle size or digest differs")
    _write(root / "code.bundle", payload)
    deadline = time.monotonic() + timeout_seconds

    def git(cwd: Path, *args: str, bound: int = 1_048_576) -> bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("bootstrap extraction deadline exceeded")
        return _git(cwd, args, timeout=remaining, max_bytes=bound)

    git(root, "clone", "--bare", "--no-local", "--quiet", "--", "code.bundle", "proof.git")
    proof = root / "proof.git"
    if git(proof, "cat-file", "-t", commit) != b"commit\n":
        raise ValueError("bootstrap identity does not name a commit")
    inventory = git(proof, "ls-tree", "-rzl", "--full-tree", commit, "--", *PREFIXES)
    if not inventory or not inventory.endswith(b"\0"):
        raise ValueError("bootstrap runtime inventory is empty or malformed")
    entries: list[tuple[str, str, int, str]] = []
    total = 0
    for record in inventory[:-1].split(b"\0"):
        metadata, encoded_path = record.split(b"\t", 1)
        mode, kind, blob, encoded_size = metadata.decode("ascii").split()
        relative = encoded_path.decode("utf-8")
        path = PurePosixPath(relative)
        size = int(encoded_size)
        if (
            mode not in ("100644", "100755")
            or kind != "blob"
            or size < 0
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or "\\" in relative
        ):
            raise ValueError("bootstrap runtime must contain canonical regular blobs only")
        total += size
        entries.append((mode, blob, size, relative))
        if len(entries) > max_files or total > max_source_bytes:
            raise ValueError("bootstrap runtime exceeds extraction limits")
    paths = {entry[3] for entry in entries}
    if len(paths) != len(entries) or not REQUIRED <= paths:
        raise ValueError("bootstrap runtime is duplicate or lacks required H inputs")
    source = root / "source"
    source.mkdir()
    hashes: dict[str, str] = {}
    for mode, blob, size, relative in entries:
        payload = git(proof, "cat-file", "blob", blob, bound=size + 1)
        if len(payload) != size:
            raise ValueError("bootstrap blob size differs from inventory")
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _write(target, payload)
        target.chmod(0o755 if mode == "100755" else 0o644)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    result: dict[str, object] = {
        "schema": "h-bootstrap-source-v1",
        "scientific_authority": "none",
        "code_commit": commit,
        "bundle_sha256": expected_sha256,
        "file_count": len(entries),
        "source_bytes": total,
        "files": hashes,
    }
    if time.monotonic() >= deadline:
        raise RuntimeError("bootstrap extraction deadline exceeded before readiness")
    _write(root / "source-ready.json", json.dumps(result, sort_keys=True).encode())
    return result


def main(argv: list[str] | None = None) -> int:
    """Materialize a locally delivered bundle; neither allocate nor start a trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--max-bundle-bytes", type=int, required=True)
    parser.add_argument("--max-source-bytes", type=int, required=True)
    parser.add_argument("--max-files", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_source(
            args.bundle,
            args.root,
            args.commit,
            args.sha256,
            max_bundle_bytes=args.max_bundle_bytes,
            max_source_bytes=args.max_source_bytes,
            max_files=args.max_files,
            timeout_seconds=args.timeout_seconds,
        )
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(
            json.dumps(
                {
                    "code": "bootstrap_source_refused",
                    "error_type": type(error).__name__,
                    "scientific_authority": "none",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
