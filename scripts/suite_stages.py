#!/usr/bin/env python3
"""Which tests each stage runs, and the record that lets a late stage gate a push.

The folder says what KIND a test is; a marker says what it COSTS. A stage is
a set of folders plus an optional marker filter:

    pre-commit   tests/unit, fast, SCOPED to the        blocks the commit
                 folders the staged paths mirror
    post-commit  repo integration architecture e2e      cannot block (git ignores
                 fault, fast                            its exit), so it writes a
                                                        RECORD keyed by tree
    pre-push     tests/unit tests/e2e, including slow   blocks the push, and reads
                                                        the record
    post-ship    repo integration architecture fault,   after `gt submit`
                 including slow

Usage:

    python scripts/suite_stages.py run pre-commit   # the whole stage
    python scripts/suite_stages.py pre-commit       # the hook: staged scope
    python scripts/suite_stages.py post-commit
    python scripts/suite_stages.py pre-push

The post-commit stage runs against a throwaway ``git worktree`` of the commit
in a temp directory, never the live tree, which may already hold the next
edit. Records live under the git common directory, so they are never tracked;
they are keyed by TREE so a reword or restack keeps its result, and pre-push
prunes every record whose tree no branch points at any more.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec B404 -- fixed git and pytest argv, no shell
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

RECORD_DIR_NAME = "suite-records"
TREE_HEX = 40
# Folders no local stage runs: they need the live MLflow stack, which only
# `just ship-it-full` (scripts/run_container_test.sh) brings up.
FULL_TIER_ONLY = ("live",)
PARALLEL = ("-q", "--maxfail=1", "--disable-warnings", "-n", "auto", "--dist", "loadfile")


@dataclass(frozen=True)
class Stage:
    """Folders to collect and an optional marker expression to filter by."""

    paths: tuple[str, ...]
    marker: str | None


@dataclass(frozen=True)
class Record:
    """The outcome of one stage for one tree."""

    tree: str
    stage: str
    passed: bool
    summary: str


STAGES: dict[str, Stage] = {
    "pre-commit": Stage(("tests/unit",), "not slow"),
    "post-commit": Stage(
        ("tests/repo", "tests/integration", "tests/architecture", "tests/e2e", "tests/fault"),
        "not slow",
    ),
    "pre-push": Stage(("tests/unit", "tests/e2e"), None),
    "post-ship": Stage(
        ("tests/repo", "tests/integration", "tests/architecture", "tests/fault"), None
    ),
}


# --- the commit stage is scoped by path ------------------------------------
#
# MEASURED 2026-11-21, 8 GB host, 9,537 fast unit tests: six xdist settings
# (4, 6, 8 workers x loadfile, worksteal) spanned 193.0 to 219.0 s with the
# current one fastest, so worker count is not the lever; tests/unit/ladder/run
# holds 816 of the 1,122 summed test-seconds. tests/unit mirrors
# src/entropic_substrate, so the commit stage runs only the folders the staged
# paths name. What it leaves out is NOT moved to post-commit (git waits for
# that hook, so the wall time would come straight back); it runs, in full and
# blocking, at pre-push.
PACKAGE_ROOT = "src/entropic_substrate"
UNIT_ROOT = "tests/unit"
FULL_UNIT = (UNIT_ROOT,)
SCOPE_DEPTH = 2
# Paths that add nothing to the commit stage: a later stage owns them, or no
# fast unit test is expected to depend on them. Anything not named here and
# not under the two mirrored roots WIDENS to the whole unit folder, so a new
# top-level path is tested in full until someone classifies it.
DEFERRED_PREFIXES = (
    "docs/",
    "scripts/",
    "tools/",
    "notebooks/",
    "references/",
    ".github/",
    "tests/repo/",
    "tests/integration/",
    "tests/architecture/",
    "tests/e2e/",
    "tests/fault/",
    "tests/live/",
)
DEFERRED_NAMES = frozenset({"justfile", ".pre-commit-config.yaml", "LICENSE", ".gitignore"})


def _is_deferred(path: str) -> bool:
    return path in DEFERRED_NAMES or path.endswith(".md") or path.startswith(DEFERRED_PREFIXES)


def _mirror(path: str, *, root: Path) -> str | None:
    """The deepest existing unit folder covering ``path``, or None to widen."""
    for base in (PACKAGE_ROOT, UNIT_ROOT):
        if not path.startswith(base + "/"):
            continue
        folders = path[len(base) + 1 :].split("/")[:-1][:SCOPE_DEPTH]
        while folders and not (root / UNIT_ROOT / "/".join(folders)).is_dir():
            folders.pop()
        return f"{UNIT_ROOT}/{'/'.join(folders)}" if folders else None
    return None


def unit_scope(changed: Iterable[str], *, root: Path) -> tuple[str, ...]:
    """The unit folders a change set names: sorted, never nested, () for none."""
    chosen: set[str] = set()
    for path in changed:
        if _is_deferred(path):
            continue
        folder = _mirror(path, root=root)
        if folder is None:
            return FULL_UNIT
        chosen.add(folder)
    return tuple(
        sorted(
            folder
            for folder in chosen
            if not any(folder.startswith(other + "/") for other in chosen)
        )
    )


def scoped_commit_stage(scope: tuple[str, ...]) -> Stage | None:
    """The commit stage narrowed to ``scope``; None when there is nothing to run."""
    return Stage(scope, STAGES["pre-commit"].marker) if scope else None


def pytest_argv(stage: Stage) -> list[str]:
    """The pytest arguments for a stage, parallel flags included."""
    marker = ["-m", stage.marker] if stage.marker else []
    return [*stage.paths, *marker, *PARALLEL]


def _record_path(store: Path, tree: str) -> Path:
    return store / f"{tree}.json"


def write_record(store: Path, record: Record) -> None:
    store.mkdir(parents=True, exist_ok=True)
    _record_path(store, record.tree).write_text(
        json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8"
    )


def read_record(store: Path, tree: str) -> Record | None:
    """The record for ``tree``; anything unreadable or mismatched is absent."""
    try:
        loaded = json.loads(_record_path(store, tree).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict) or set(loaded) != {"tree", "stage", "passed", "summary"}:
        return None
    if loaded["tree"] != tree or not isinstance(loaded["passed"], bool):
        return None
    if not isinstance(loaded["stage"], str) or not isinstance(loaded["summary"], str):
        return None
    return Record(tree, loaded["stage"], loaded["passed"], loaded["summary"])


def verify(store: Path, tree: str, *, rerun: Callable[[], bool]) -> bool:
    """A recorded pass admits, a recorded failure refuses, no record runs the stage now."""
    record = read_record(store, tree)
    if record is None:
        return rerun()
    return record.passed


def prune(store: Path, *, keep: set[str]) -> int:
    """Delete every record whose tree is not in ``keep``; return how many."""
    if not store.is_dir():
        return 0
    removed = 0
    for path in store.glob("*.json"):
        if len(path.stem) == TREE_HEX and path.stem not in keep:
            path.unlink()
            removed += 1
    return removed


def _clean_env() -> dict[str, str]:
    # A hook runs with GIT_DIR / GIT_INDEX_FILE pointing at the caller's
    # repository; a child git or pytest must not inherit them.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(  # nosec B603 B607 -- git from PATH by design, fixed argv
        ["git", *args], cwd=repo, env=_clean_env(), check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def _store(repo: Path) -> Path:
    return (repo / _git(repo, "rev-parse", "--git-common-dir")).resolve() / RECORD_DIR_NAME


def _staged(repo: Path) -> list[str]:
    # --no-renames lists both sides of a move, so the folder a file LEFT is
    # tested as well as the one it arrived in.
    return _git(repo, "diff", "--cached", "--name-only", "--no-renames").splitlines()


def _live_trees(repo: Path) -> set[str]:
    heads = _git(repo, "for-each-ref", "--format=%(objectname)^{tree}", "refs/heads").split()
    return {_git(repo, "rev-parse", ref) for ref in [*heads, "HEAD^{tree}"]}


def run_stage(stage: Stage, *, cwd: Path, src: Path) -> tuple[bool, str]:
    """Run one stage's tests in ``cwd`` against the package under ``src``."""
    env = _clean_env()
    env["PYTHONPATH"] = str(src)
    done = subprocess.run(  # nosec B603 -- this interpreter, fixed pytest argv
        [sys.executable, "-m", "pytest", *pytest_argv(stage), "-p", "no:cacheprovider"],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else f"pytest exit {done.returncode}, no output"
    if done.returncode:
        sys.stderr.write(done.stdout[-6000:] + done.stderr[-2000:])
    return done.returncode == 0, summary


def link_dvc_data(repo: Path, export: Path) -> list[str]:
    """Symlink DVC-tracked data into the export where its pointer is unchanged.

    DVC data is gitignored, so a fresh worktree has the ``*.dvc`` pointers and
    none of the payloads. A payload in the live checkout belongs to the LIVE
    pointer; it is linked only when the commit's pointer is byte-identical,
    so the export never reads data the commit does not name.
    """
    linked: list[str] = []
    for pointer in sorted(export.rglob("*.dvc")):
        relative = pointer.relative_to(export)
        payload = relative.with_suffix("")
        live_pointer, source, target = repo / relative, repo / payload, export / payload
        if not live_pointer.is_file() or not source.exists() or target.exists():
            continue
        if live_pointer.read_bytes() != pointer.read_bytes():
            continue
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        linked.append(payload.as_posix())
    return linked


def run_in_export(repo: Path, stage_name: str) -> Record:
    """Run a stage against a throwaway worktree of HEAD and record the result."""
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    commit = _git(repo, "rev-parse", "HEAD")
    scratch = Path(tempfile.mkdtemp(prefix="suite-export-"))
    export = scratch / "tree"
    try:
        _git(repo, "worktree", "add", "--detach", str(export), commit)
        link_dvc_data(repo, export)
        passed, summary = run_stage(STAGES[stage_name], cwd=export, src=export / "src")
    finally:
        subprocess.run(  # nosec B603 B607 -- git from PATH by design, fixed argv
            ["git", "worktree", "remove", "--force", str(export)],
            cwd=repo,
            env=_clean_env(),
            check=False,
            capture_output=True,
        )
        shutil.rmtree(scratch, ignore_errors=True)
        subprocess.run(  # nosec B603 B607 -- git from PATH by design, fixed argv
            ["git", "worktree", "prune"],
            cwd=repo,
            env=_clean_env(),
            check=False,
            capture_output=True,
        )
    record = Record(tree, stage_name, passed, summary)
    write_record(_store(repo), record)
    return record


def _say(message: str) -> None:
    print(f"suite-stages: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run").add_argument("stage", choices=sorted(STAGES))
    commands.add_parser("pre-commit")
    commands.add_parser("post-commit")
    commands.add_parser("pre-push")
    args = parser.parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))

    if args.command == "run":
        passed, summary = run_stage(STAGES[args.stage], cwd=repo, src=repo / "src")
        _say(f"{args.stage}: {summary}")
        return 0 if passed else 1

    if args.command == "pre-commit":
        scope = unit_scope(_staged(repo), root=repo)
        stage = scoped_commit_stage(scope)
        if stage is None:
            _say("pre-commit: no staged path maps to tests/unit; pre-push runs it in full")
            return 0
        passed, summary = run_stage(stage, cwd=repo, src=repo / "src")
        _say(f"pre-commit [{' '.join(scope)}]: {summary}")
        return 0 if passed else 1

    if args.command == "post-commit":
        record = run_in_export(repo, "post-commit")
        verdict = "PASS" if record.passed else "FAIL (the next push will refuse this tree)"
        _say(f"post-commit {verdict} for tree {record.tree[:12]}: {record.summary}")
        return 0

    passed, summary = run_stage(STAGES["pre-push"], cwd=repo, src=repo / "src")
    _say(f"pre-push: {summary}")
    if not passed:
        return 1
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    store = _store(repo)
    admitted = verify(store, tree, rerun=lambda: run_in_export(repo, "post-commit").passed)
    record = read_record(store, tree)
    _say(
        f"post-commit record for tree {tree[:12]}: {'PASS' if admitted else 'FAIL'}: {record.summary if record else 'absent'}"
    )
    if admitted:
        _say(f"pruned {prune(store, keep=_live_trees(repo))} stale record(s)")
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
