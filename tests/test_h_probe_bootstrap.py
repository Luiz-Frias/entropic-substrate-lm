"""Bootstrap uses real Git objects without importing the not-yet-installed H package."""

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "h_probe_bootstrap", Path(__file__).resolve().parents[1] / "scripts/h_probe_bootstrap.py"
)
assert _SPEC is not None and _SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap)
materialize_source = bootstrap.materialize_source

pytestmark = pytest.mark.unit


def bundle(
    root: Path, *, symlink: bool = False, missing_worker: bool = False
) -> tuple[Path, str, str]:
    repo = root / "repository"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=repo, text=True).strip()

    git("init", "-b", "fixture")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.test")
    for name, body in {
        "pyproject.toml": "[project]\nname='fixture'\nversion='0.1.0'\n",
        "uv.lock": "version = 1\n",
        ".dvc/config": "[core]\nremote=r2\n",
        "src/entropic_substrate/sentence_prediction/worker_command.py": "raise SystemExit(0)\n",
        "scripts/example.py": "print('never execute during extraction')\n",
        "private-not-runtime.txt": "must not be exported\n",
    }.items():
        if missing_worker and name.endswith("worker_command.py"):
            continue
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (repo / "scripts/example.py").chmod(0o755)
    if symlink:
        (repo / "scripts/escape").symlink_to("/etc/passwd")
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-m", "fixture")
    commit = git("rev-parse", "HEAD")
    path = root / "source.bundle"
    git("bundle", "create", str(path), "fixture")
    return path, commit, hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, artifact: tuple[Path, str, str], **changes: object) -> dict[str, object]:
    source, commit, digest = artifact
    options = {
        "max_bundle_bytes": 1_000_000,
        "max_source_bytes": 1_000_000,
        "max_files": 100,
        "timeout_seconds": 10.0,
    } | changes
    return materialize_source(source, root, commit, digest, **options)


def test_real_bundle_exports_only_runtime_files_and_commits_readiness_last(tmp_path: Path) -> None:
    artifact = bundle(tmp_path)
    root = tmp_path / "bootstrap"
    root.mkdir()
    result = run(root, artifact)
    assert result["schema"] == "h-bootstrap-source-v1"
    assert result["code_commit"] == artifact[1]
    assert result["file_count"] == 5
    assert result["scientific_authority"] == "none"
    assert json.loads((root / "source-ready.json").read_bytes()) == result
    source = root / "source"
    assert not (source / "private-not-runtime.txt").exists()
    assert not (source / ".git").exists()
    assert (source / "scripts/example.py").stat().st_mode & 0o111
    assert (source / "uv.lock").read_text() == "version = 1\n"


@pytest.mark.parametrize(
    "mutation", ["digest", "size", "commit", "symlink", "occupied", "files", "bytes", "required"]
)
def test_invalid_bundle_or_envelope_never_produces_ready_evidence(
    tmp_path: Path, mutation: str
) -> None:
    artifact = bundle(
        tmp_path, symlink=mutation == "symlink", missing_worker=mutation == "required"
    )
    root = tmp_path / "bootstrap"
    root.mkdir()
    options = {}
    if mutation == "digest":
        artifact = (artifact[0], artifact[1], "0" * 64)
    if mutation == "commit":
        artifact = (artifact[0], "0" * 40, artifact[2])
    if mutation == "size":
        options["max_bundle_bytes"] = 1
    if mutation == "files":
        options["max_files"] = 1
    if mutation == "bytes":
        options["max_source_bytes"] = 1
    if mutation == "occupied":
        (root / "existing").write_bytes(b"preserved")
    with pytest.raises((ValueError, RuntimeError)):
        run(root, artifact, **options)
    assert not (root / "source-ready.json").exists()
    if mutation == "occupied":
        assert (root / "existing").read_bytes() == b"preserved"


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_limits_fail_before_any_git_or_disk_output(tmp_path: Path, limit: int) -> None:
    artifact = (tmp_path / "absent.bundle", "1" * 40, "2" * 64)
    with pytest.raises(ValueError, match="limits"):
        run(tmp_path, artifact, max_files=limit)
    assert not tuple(tmp_path.iterdir())


def test_standalone_command_needs_no_installed_project(tmp_path: Path) -> None:
    artifact = bundle(tmp_path)
    root = tmp_path / "bootstrap"
    root.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(bootstrap.__file__)),
            "--bundle",
            str(artifact[0]),
            "--root",
            str(root),
            "--commit",
            artifact[1],
            "--sha256",
            artifact[2],
            "--max-bundle-bytes",
            "1000000",
            "--max-source-bytes",
            "1000000",
            "--max-files",
            "100",
            "--timeout-seconds",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["file_count"] == 5


@pytest.mark.parametrize("mode", ["timeout", "overflow"])
def test_git_subprocess_guards_reap_the_owned_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    real = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def launch(
        _args: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        stdin: int,
        stdout: int,
        stderr: int,
        start_new_session: bool,
    ) -> subprocess.Popen[bytes]:
        code = "import time; time.sleep(30)" if mode == "timeout" else "print('x' * 10000)"
        process = real(
            (sys.executable, "-c", code),
            cwd=cwd,
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=start_new_session,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(bootstrap.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match=r"deadline|output limit"):
        bootstrap._git(tmp_path, ("version",), timeout=0.2, max_bytes=10)
    assert len(processes) == 1 and processes[0].poll() is not None


@pytest.mark.parametrize("refusal", [PermissionError, ProcessLookupError])
def test_git_guard_cleanup_survives_a_refused_or_missing_group_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refusal: type[OSError],
) -> None:
    """FOR-1537: the cleanup's killpg may be answered EPERM (PGID reused by a
    session we do not own) as well as ESRCH; both mean "not ours", neither may
    escape _git. ProcessLookupError is the control arm (already handled)."""
    reaped: list[int] = []
    real_killpg = bootstrap.os.killpg

    def refuse(pgid: int, sig: int) -> None:
        reaped.append(pgid)
        raise refusal(1, "refused by the kernel")

    monkeypatch.setattr(bootstrap.os, "killpg", refuse)
    try:
        output = bootstrap._git(tmp_path, ("--version",), timeout=10, max_bytes=65536)
    finally:
        monkeypatch.setattr(bootstrap.os, "killpg", real_killpg)
    assert output.startswith(b"git version")
    assert len(reaped) == 1


def test_bootstrap_inventory_matches_native_worker_verification() -> None:
    from entropic_substrate.historical.d_ladder.d_ladder_worker_runtime import SOURCE_PREFIXES

    assert bootstrap.PREFIXES == SOURCE_PREFIXES


def test_failed_atomic_readiness_publication_keeps_marker_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = bundle(tmp_path)
    root = tmp_path / "bootstrap"
    root.mkdir()
    original = bootstrap.os.link

    def link(source: Path, destination: Path) -> None:
        if Path(destination).name == "source-ready.json":
            raise OSError("simulated readiness publication failure")
        return original(source, destination)

    monkeypatch.setattr(bootstrap.os, "link", link)
    with pytest.raises(OSError, match="simulated"):
        run(root, artifact)
    assert not (root / "source-ready.json").exists()
    assert not list(root.glob(".h-bootstrap-*"))


@pytest.mark.parametrize("boundary", ["foreign_owner", "group_writable", "world_writable"])
def test_bootstrap_enforces_workspace_ownership_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    artifact = bundle(tmp_path)
    root = tmp_path / "bootstrap"
    root.mkdir()
    if boundary == "foreign_owner":
        foreign = root.stat().st_uid + 1
        monkeypatch.setattr(bootstrap.os, "geteuid", lambda: foreign)
    else:
        root.chmod(0o775 if boundary == "group_writable" else 0o757)
    with pytest.raises(ValueError, match="owned"):
        run(root, artifact)
    assert not tuple(root.iterdir())


def test_internal_workspaces_remain_private_under_shared_umask(tmp_path: Path) -> None:
    artifact = bundle(tmp_path)
    root = tmp_path / "bootstrap"
    root.mkdir(mode=0o755)
    original = os.umask(0o002)
    try:
        run(root, artifact)
    finally:
        os.umask(original)
    assert stat.S_IMODE((root / "source").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "proof.git").stat().st_mode) == 0o700
