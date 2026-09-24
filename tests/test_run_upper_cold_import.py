"""ACCEPTANCE instrument for the p10-run-upper lane (brief ACCEPTANCE: "Cold
import, fresh interpreter per moved leaf, entered FIRST, stdout marker not
exit code" -- mirrors ``tests/test_run_lower_cold_import.py``).

RED-first (house rule 5): at commit 1 this file's own subject set is derived
live from the p10 fragment's ``landed: true`` rows, which do not exist before
the first ``git mv`` lands. Extended per landing commit.

COLD-IMPORT TEST SUBJECT SET (architect constraint, design gate 2026-09-24):
the filesystem side enumerates ``ladder/run/{campaign,closure,ckptsync}/*.py``
(excluding ``__init__.py``) PLUS the one D1 worker leaf,
``ladder/run/worker/d1_predecessor_loader.py``, named explicitly -- never a
glob of ``ladder/run/worker/*.py``, because that package also holds p4's H
leaves and its ``hardware.py`` split leaf, none of which are p10's. The split
leaf ``ladder.run.worker.resource_registration`` has no fragment row (it is a
``split_consumption``, not a ``modules`` entry) and is therefore declared as
its own explicit, non-fragment subject, parametrized and cold-imported
separately, with an existence check -- never derived from a ``worker/*.py``
glob, which would silently fold it in with modules that are not p10's.

The four pod-facing entrypoints (the pod rows of ``scripts/entrypoints.env``)
are cold-imported as a fixed, independent sanity arm on every extension of
this file: ``entropic_substrate.sentence_prediction.worker_command``,
``entropic_substrate.d1_trainer_mailbox_command``,
``entropic_substrate.d_ladder_worker_command``,
``entropic_substrate.analysis.d1_external_worker``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_PATH = REPO_ROOT / ".claude" / "workflows" / "landed-set" / "p10-run-upper.json"
RUN_ROOT = REPO_ROOT / "src" / "entropic_substrate" / "ladder" / "run"
SUBPACKAGES = ("campaign", "closure", "ckptsync")
WORKER_D1_LEAF = RUN_ROOT / "worker" / "d1_predecessor_loader.py"
WORKER_D1_DOTTED = "entropic_substrate.ladder.run.worker.d1_predecessor_loader"
SPLIT_LEAF_DOTTED = "entropic_substrate.ladder.run.worker.resource_registration"
SPLIT_LEAF_PATH = RUN_ROOT / "worker" / "resource_registration.py"

POD_ENTRYPOINTS = (
    "entropic_substrate.sentence_prediction.worker_command",
    "entropic_substrate.d1_trainer_mailbox_command",
    "entropic_substrate.d_ladder_worker_command",
    "entropic_substrate.analysis.d1_external_worker",
)

pytestmark = pytest.mark.unit

_MARKER = "COLD_IMPORT_OK"

_COLD_IMPORT_TEMPLATE = """
import importlib
importlib.import_module({dotted!r})
print({marker!r})
"""


def _fragment_landed_targets() -> list[str]:
    """Side (b): the p10 fragment's own ``landed: true`` target rows for the
    27 full-move modules (excludes the split leaf, which is a
    ``split_consumption`` note, not a ``modules`` row)."""
    if not FRAGMENT_PATH.exists():
        return []
    data = json.loads(FRAGMENT_PATH.read_text())
    return sorted(
        entry["target"]
        for entry in (data.get("modules") or {}).values()
        if entry.get("landed", False)
    )


def _filesystem_landed_targets() -> list[str]:
    """Side (a): a live directory listing of the three new p10 sub-packages
    PLUS the one named worker D1 leaf -- never a glob of ``worker/*.py``
    (see module docstring)."""
    out: list[str] = []
    for sub in SUBPACKAGES:
        d = RUN_ROOT / sub
        if not d.is_dir():
            continue
        for f in d.glob("*.py"):
            if f.name == "__init__.py":
                continue
            out.append(f"entropic_substrate.ladder.run.{sub}.{f.stem}")
    if WORKER_D1_LEAF.is_file():
        out.append(WORKER_D1_DOTTED)
    return sorted(out)


def _run_cold_import(dotted: str) -> subprocess.CompletedProcess[str]:
    code = _COLD_IMPORT_TEMPLATE.format(dotted=dotted, marker=_MARKER)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )


def _assert_cold_import_ok(result: subprocess.CompletedProcess[str], dotted: str) -> None:
    """Shared positive-arm assertion (CodeRabbit local review, p10-run-upper phase
    diff): the cold-import subprocess must exit 0 AND print the marker -- not
    merely exit 0. Used only by the three subject-driven tests below; the
    guard-teeth triad at the bottom of this file stays independent of this
    helper on purpose (each proves the instrument itself, not a subject)."""
    assert result.returncode == 0, f"cold import of {dotted} failed:\n{result.stderr}"
    assert _MARKER in result.stdout, (
        f"child process for {dotted} did not report success; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )


def test_filesystem_and_fragment_derivations_of_the_landed_set_agree() -> None:
    """The C7 equality arm: two INDEPENDENT derivations of "what p10 has
    landed so far" must agree, with a non-vacuity guard so an empty==empty
    pass does not satisfy this test."""
    fs_side = _filesystem_landed_targets()
    fragment_side = _fragment_landed_targets()
    assert fs_side, "no p10 destination modules found on disk -- has the first git mv landed?"
    assert fragment_side, (
        "p10 fragment has no landed:true rows yet -- RED-first: this is expected "
        "before the first landing commit, and must go GREEN once it lands"
    )
    assert fs_side == fragment_side, (
        f"filesystem enumeration {fs_side} != fragment landed targets {fragment_side}"
    )


@pytest.mark.parametrize("dotted", _fragment_landed_targets())
def test_cold_import_of_landed_run_upper_target_succeeds_with_nothing_warming_it(
    dotted: str,
) -> None:
    """Fresh interpreter, NOTHING imported first: importing ``dotted``
    directly must succeed and print the marker -- not merely exit 0."""
    _assert_cold_import_ok(_run_cold_import(dotted), dotted)


def test_split_leaf_exists_and_is_declared_explicitly_not_via_worker_glob() -> None:
    """The split leaf has no fragment row (SPLIT_CONSUMPTION, not a
    ``modules`` entry) -- this test asserts its existence directly rather
    than relying on a ``worker/*.py`` glob, which would also match p4's
    unrelated leaves."""
    assert SPLIT_LEAF_PATH.is_file(), (
        "ladder/run/worker/resource_registration.py does not exist yet -- "
        "RED-first before commit 1 lands the worker split"
    )


def test_cold_import_of_the_worker_split_leaf_succeeds_with_nothing_warming_it() -> None:
    _assert_cold_import_ok(_run_cold_import(SPLIT_LEAF_DOTTED), SPLIT_LEAF_DOTTED)


@pytest.mark.parametrize("dotted", POD_ENTRYPOINTS)
def test_cold_import_of_pod_entrypoint_succeeds_entered_first(dotted: str) -> None:
    """The four pod-facing entrypoints (``scripts/entrypoints.env`` pod rows)
    must keep cold-importing cleanly, entered FIRST in a fresh interpreter,
    through every p10 landing commit."""
    _assert_cold_import_ok(_run_cold_import(dotted), dotted)


def test_cold_import_guard_accepts_a_real_import() -> None:
    """Positive arm: the instrument reports the marker for an importable module."""
    result = _run_cold_import("json")
    assert result.returncode == 0
    assert _MARKER in result.stdout


def test_cold_import_guard_rejects_a_failed_import() -> None:
    """Negative arm: a failed import never prints the marker."""
    result = _run_cold_import("entropic_substrate.__no_such_module__")
    assert result.returncode != 0
    assert _MARKER not in result.stdout


def test_cold_import_guard_is_not_fooled_by_a_process_that_silently_exits_0() -> None:
    """A child that exits 0 without printing the marker fails the marker check.

    This is the GREEN control for the instrument itself, distinct from the two
    tests above: those exercise ``_run_cold_import`` end to end (the real guard
    template), while this one shows the marker-presence assertion alone is not
    fooled by an unrelated zero-exit child that never ran the template at all."""
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert _MARKER not in result.stdout
