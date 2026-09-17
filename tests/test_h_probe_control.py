"""Durable single-writer control custody is not an execution authorization."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tests.test_h_probe_launch import launch

from entropic_substrate.d1_command_runner import BoundedSubprocessD1CommandRunner
from entropic_substrate.d1_operator_ports import D1CommandResult
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.h_probe_control import own_probe_role, read_probe_control

pytestmark = pytest.mark.unit


def root(tmp_path: Path) -> Path:
    directory = tmp_path / "probe"
    directory.mkdir(mode=0o700)
    return directory


def test_roles_are_independent_but_each_role_is_single_writer(tmp_path: Path) -> None:
    directory, spec = root(tmp_path), launch()
    with own_probe_role(directory, spec, "controller") as controller:
        with pytest.raises(ValueError, match="already owned"):
            with own_probe_role(directory, spec, "controller"):
                pytest.fail("second owner admitted")
        with own_probe_role(directory, spec, "watchdog") as guard:
            guard.write("ready", {"pid": os.getpid()}, max_bytes=10_000)
        assert read_probe_control(directory, spec, "watchdog", "ready", max_bytes=10_000) == {
            "pid": os.getpid()
        }
        controller.write("create-intent", {"attempt": "one"}, max_bytes=10_000)
    with pytest.raises(ValueError, match="released"):
        controller.write("another", {}, max_bytes=10_000)


def test_recorded_intent_survives_owner_exit_and_cannot_be_replaced(tmp_path: Path) -> None:
    directory, spec = root(tmp_path), launch()
    with own_probe_role(directory, spec, "controller") as controller:
        reference = controller.write("create-intent", {"attempt": "one"}, max_bytes=10_000)
    with own_probe_role(directory, spec, "controller") as resumed:
        assert read_probe_control(
            directory, spec, "controller", "create-intent", max_bytes=10_000
        ) == {"attempt": "one"}
        assert Path(reference.path).is_file()
        with pytest.raises(ValueError):
            resumed.write("create-intent", {"attempt": "one"}, max_bytes=10_000)
        with pytest.raises(ValueError):
            resumed.write("create-intent", {"attempt": "two"}, max_bytes=10_000)


def test_second_launch_cannot_adopt_the_same_control_directory(tmp_path: Path) -> None:
    from dataclasses import replace

    directory, spec = root(tmp_path), launch()
    with own_probe_role(directory, spec, "controller"):
        pass
    with pytest.raises(ValueError):
        with own_probe_role(directory, replace(spec, container_disk_gb=81), "controller"):
            pytest.fail("foreign launch adopted")


@pytest.mark.parametrize(
    "kind", ["public_root", "root_link", "lock_link", "lock_public", "init_lock_link"]
)
def test_unsafe_control_paths_fail_without_repairing_them(tmp_path: Path, kind: str) -> None:
    directory = root(tmp_path)
    if kind == "public_root":
        directory.chmod(0o755)
    elif kind == "root_link":
        alias = tmp_path / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        directory = alias
    else:
        target = directory / ("launch.lock" if kind == "init_lock_link" else "controller.lock")
        if kind in ("lock_link", "init_lock_link"):
            target.symlink_to(tmp_path / "not-created")
        else:
            target.write_text("")
            target.chmod(0o666)
    with pytest.raises(ValueError):
        with own_probe_role(directory, launch(), "controller"):
            pytest.fail("unsafe path admitted")
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("mutation", ["launch", "role", "name", "authority", "extra", "symlink"])
def test_relabelled_control_evidence_is_rejected(tmp_path: Path, mutation: str) -> None:
    directory, spec = root(tmp_path), launch()
    with own_probe_role(directory, spec, "controller") as controller:
        controller.write("create-intent", {"attempt": "one"}, max_bytes=10_000)
    path = directory / "controller-create-intent.json"
    row = json.loads(path.read_bytes())
    if mutation == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "missing")
    else:
        if mutation == "launch":
            row["launch_sha256"] = "0" * 64
        elif mutation == "role":
            row["role"] = "watchdog"
        elif mutation == "name":
            row["name"] = "allocation"
        elif mutation == "authority":
            row["authority"] = "spending"
        else:
            row["fallback"] = True
        path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError):
        read_probe_control(directory, spec, "controller", "create-intent", max_bytes=10_000)


def test_missing_control_is_absent_not_ready_and_limits_are_explicit(tmp_path: Path) -> None:
    directory, spec = root(tmp_path), launch()
    assert read_probe_control(directory, spec, "watchdog", "ready", max_bytes=10_000) is None
    with own_probe_role(directory, spec, "controller") as controller:
        with pytest.raises(ValueError):
            controller.write("../escape", {}, max_bytes=10_000)
        with pytest.raises(ValueError):
            controller.write("large", {"payload": "x" * 100}, max_bytes=1)
    for maximum in (0, True, -1):
        with pytest.raises(ValueError):
            read_probe_control(directory, spec, "controller", "missing", max_bytes=maximum)


def test_separate_process_cannot_take_an_owned_role_and_can_take_a_released_role(
    tmp_path: Path,
) -> None:
    directory, spec = root(tmp_path), launch()
    code = """from pathlib import Path
import sys
from entropic_substrate.h_probe_launch import parse_probe_launch
from entropic_substrate.h_probe_control import own_probe_role
root = Path(sys.argv[1])
launch = parse_probe_launch((root / 'launch.json').read_bytes(), max_bytes=10000)
with own_probe_role(root, launch, 'controller'):
    print('role_acquired')
"""
    runner = BoundedSubprocessD1CommandRunner(timeout_seconds=30, max_output_bytes=10_000)
    arguments = (sys.executable, "-c", code, str(directory))
    with own_probe_role(directory, spec, "controller"):
        denied = runner.run(arguments, cwd=Path.cwd(), stdin=None)
        assert denied.returncode != 0 and "already owned" in denied.stderr
        assert "role_acquired" not in denied.stdout
    admitted = runner.run(arguments, cwd=Path.cwd(), stdin=None)
    assert admitted.returncode == 0 and admitted.stdout.strip() == "role_acquired"


def test_replaced_manifest_invalidates_existing_control_records(tmp_path: Path) -> None:
    directory, spec = root(tmp_path), launch()
    with own_probe_role(directory, spec, "controller") as controller:
        controller.write("create-intent", {}, max_bytes=10_000)
        (directory / "launch.json").write_bytes(b"{}")
        with pytest.raises(ValueError, match="another launch"):
            read_probe_control(directory, spec, "controller", "create-intent", max_bytes=10_000)
        with pytest.raises(ValueError, match="another launch"):
            controller.write("allocation", {}, max_bytes=10_000)


def test_manifest_publication_holds_a_shared_cross_process_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate import h_probe_control as control

    directory, spec = root(tmp_path), launch()
    original = control.publish_d1_atomic_bytes
    runner = BoundedSubprocessD1CommandRunner(timeout_seconds=30, max_output_bytes=10_000)
    code = """import fcntl, os, sys
descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('blocked')
else:
    print('unlocked')
finally:
    os.close(descriptor)
"""

    def checked(root: Path, name: object, payload: bytes, *, label: str) -> ArtifactIdentity:
        if name == "launch.json":
            observed = runner.run(
                (sys.executable, "-c", code, str(directory / "launch.lock")),
                cwd=Path.cwd(),
                stdin=None,
            )
            assert observed.returncode == 0 and observed.stdout.strip() == "blocked"
        return original(root, name, payload, label=label)

    monkeypatch.setattr(control, "publish_d1_atomic_bytes", checked)
    with own_probe_role(directory, spec, "controller"):
        pass


def test_two_processes_can_initialize_different_roles_in_one_fresh_directory(
    tmp_path: Path,
) -> None:
    directory, spec = root(tmp_path), launch()
    code = """import os, sys
from pathlib import Path
from entropic_substrate.h_probe_launch import parse_probe_launch
from entropic_substrate.h_probe_control import own_probe_role
launch = parse_probe_launch(sys.stdin.buffer.read(), max_bytes=10000)
with own_probe_role(Path(sys.argv[1]), launch, sys.argv[2]) as role:
    role.write('entered', {'pid': os.getpid()}, max_bytes=10000)
"""

    def enter(role: str) -> D1CommandResult:
        runner = BoundedSubprocessD1CommandRunner(timeout_seconds=30, max_output_bytes=10_000)
        return runner.run(
            (sys.executable, "-c", code, str(directory), role),
            cwd=Path.cwd(),
            stdin=spec.payload.decode(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(enter, ("controller", "watchdog")))
    assert all(result.returncode == 0 for result in results), [r.stderr for r in results]
    first = read_probe_control(directory, spec, "controller", "entered", max_bytes=10_000)
    second = read_probe_control(directory, spec, "watchdog", "entered", max_bytes=10_000)
    assert first is not None and second is not None and first["pid"] != second["pid"]


def test_abrupt_owner_exit_releases_lock_but_preserves_intent(tmp_path: Path) -> None:
    directory, spec = root(tmp_path), launch()
    code = """import os, sys
from pathlib import Path
from entropic_substrate.h_probe_launch import parse_probe_launch
from entropic_substrate.h_probe_control import own_probe_role
launch = parse_probe_launch(sys.stdin.buffer.read(), max_bytes=10000)
with own_probe_role(Path(sys.argv[1]), launch, 'controller') as writer:
    writer.write('create-intent', {'case': 'crash-fixture'}, max_bytes=10000)
    os._exit(7)
"""
    runner = BoundedSubprocessD1CommandRunner(timeout_seconds=30, max_output_bytes=10_000)
    result = runner.run(
        (sys.executable, "-c", code, str(directory)), cwd=Path.cwd(), stdin=spec.payload.decode()
    )
    assert result.returncode == 7
    with own_probe_role(directory, spec, "controller"):
        assert read_probe_control(
            directory, spec, "controller", "create-intent", max_bytes=10_000
        ) == {"case": "crash-fixture"}
