#!/usr/bin/env python3
"""``commitizen-branch`` pre-push hook that tolerates an empty range (FOR-1541).

MEASURED 2026-09-22 and 2026-10-05: ``just ship-it`` pushes with
``gt submit --stack --always``. When the bottom branch of the stack is already
up to date on the remote, ``--always`` still pushes it and prek hands the
pre-push hooks ``PRE_COMMIT_FROM_REF == PRE_COMMIT_TO_REF`` for that ref. The
upstream hook (``cz check --rev-range FROM..TO``) then finds no commits, prints
``No commit found with range: 'X..X'`` and exits 3, aborting the whole atomic
push. ``--always`` stays in the justfile on purpose; this wrapper is the fix.

Contract:

* ``PRE_COMMIT_FROM_REF`` / ``PRE_COMMIT_TO_REF`` missing or empty: error on
  stderr, exit 2 (unexpected invocation, fail loud).
* ``FROM == TO``, or ``git rev-list --count FROM..TO`` prints ``0``: nothing to
  lint, one-line message on stdout, exit 0.
* Anything else, including a range git cannot count: run
  ``cz check --rev-range FROM..TO`` (argument order and text identical to the
  upstream hook) and exit with cz's return code.

``CZ_CHECK_BIN`` overrides the ``cz`` executable. It exists for tests only;
the hook configuration never sets it.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - fixed argv lists only, no shell
import sys

_USAGE_ERROR = 2


def _empty_range(from_ref: str, to_ref: str) -> bool:
    if from_ref == to_ref:
        return True
    counted = subprocess.run(  # nosec B603 B607 - fixed argv, refs are a single argv element each
        ["git", "rev-list", "--count", f"{from_ref}..{to_ref}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return counted.returncode == 0 and counted.stdout.strip() == "0"


def main() -> int:
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF", "")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF", "")
    if not from_ref or not to_ref:
        print(
            "commitizen-branch: PRE_COMMIT_FROM_REF and PRE_COMMIT_TO_REF must both be set and non-empty "
            "(this hook runs at the pre-push stage under prek/pre-commit)",
            file=sys.stderr,
        )
        return _USAGE_ERROR
    if _empty_range(from_ref, to_ref):
        print(
            f"commitizen-branch: empty range {from_ref}..{to_ref} (ref already up to date), nothing to lint"
        )
        return 0
    cz = os.environ.get("CZ_CHECK_BIN") or "cz"
    try:
        done = subprocess.run([cz, "check", "--rev-range", f"{from_ref}..{to_ref}"], check=False)  # nosec B603 B607
    except FileNotFoundError:
        print(
            f"commitizen-branch: cannot execute {cz!r}; is commitizen installed in the hook environment?",
            file=sys.stderr,
        )
        return _USAGE_ERROR
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
