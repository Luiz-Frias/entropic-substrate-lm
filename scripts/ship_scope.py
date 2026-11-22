#!/usr/bin/env python3
"""Decide whether a change may ship through slim ``just ship-it``.

Slim ``ship-it`` does not bring up the live MLflow stack. This guard reads
the files changed against the base and exits 3, naming them and
``just ship-it-full``, when any of them can change how the live stack
behaves: the compose project, the image definition, the dependency set, the
MLflow adapter and sinks, the container gate itself, or the live tests.

    python scripts/ship_scope.py                # base: origin/main
    python scripts/ship_scope.py --base main

MEASURED reason this is mechanical and not a habit: dependency commit 81bd4dd
(mlflow -> mlflow-skinny) turned the live gate red under a green CI, because
the live tests skip without a stack and nothing else runs them.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess  # nosec B404 -- fixed git argv, no shell
import sys
from collections.abc import Sequence
from pathlib import Path

EXIT_NEEDS_FULL = 3
EXIT_NOTHING_EXAMINED = 2

INFRA_PATTERNS: tuple[str, ...] = (
    "compose.yaml",
    "Containerfile*",
    ".containerignore",
    ".dockerignore",
    "Caddyfile",
    "pyproject.toml",
    "uv.lock",
    "scripts/run_container_test.sh",
    "scripts/derive_compose_project.sh",
    "src/entropic_substrate/ladder/capture/adapters/mlflow*.py",
    "src/entropic_substrate/ladder/capture/reader/sinks.py",
    "tests/live/*",
)


def requires_full(changed: Sequence[str], *, patterns: Sequence[str] = INFRA_PATTERNS) -> list[str]:
    """The changed paths that can alter live-stack behaviour, sorted."""
    return sorted(
        {
            path
            for path in changed
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
        }
    )


def decide(changed: Sequence[str]) -> tuple[int, str]:
    """Exit code and message for a change set."""
    if not changed:
        return EXIT_NOTHING_EXAMINED, "no changed files against the base: nothing examined"
    hits = requires_full(changed)
    if hits:
        listed = "\n".join(f"    {path}" for path in hits)
        return (
            EXIT_NEEDS_FULL,
            f"this change touches the live stack:\n{listed}\n  publish it with: just ship-it-full",
        )
    return 0, f"{len(changed)} changed file(s), none touch the live stack"


def changed_files(repo: Path, base: str) -> list[str]:
    if not base or base.startswith("-"):
        message = f"not a revision: {base!r}"
        raise ValueError(message)

    def git(*args: str) -> str:
        done = subprocess.run(  # nosec B603 B607 -- git from PATH by design, fixed argv plus one validated revision
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
        return done.stdout.strip()

    merge_base = git("merge-base", "HEAD", base)
    return [line for line in git("diff", "--name-only", f"{merge_base}..HEAD").splitlines() if line]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--base", default="origin/main", help="revision the stack is measured against"
    )
    args = parser.parse_args(argv)
    code, message = decide(changed_files(Path.cwd(), args.base))
    print(f"ship-scope: {message}", file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
