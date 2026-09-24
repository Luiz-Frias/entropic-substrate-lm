"""RED-first guard-teeth for the cold-import hazard rewrite-on-arrival closes
(p8-capture brief ACCEPTANCE: "Cold import at the NEW address, fresh
interpreter per moved leaf"). Same discipline as ``tests/test_shim_layer.py``
and p6's ``tests/test_core_cold_import.py``: a subprocess-plus-stdout-marker
check, never exit code alone -- a child that silently no-ops still exits 0.

Subjects are derived LIVE from THIS phase's own fragment
(``.claude/workflows/landed-set/p8-capture.json``), never from a hardcoded
list and never from a generic ``ladder.capture.*`` prefix (architect
constraint 2026-09-23): p3b and p4 already land other ``ladder.capture.*``
targets (the ports/adapters rail, ``capture.py``), so a prefix-based
collector would silently pull in modules this phase does not own. Reading
only this phase's own ``modules`` dict with ``landed: true`` is exact by
construction and satisfies the ACCEPTANCE bullet "collected count == that
fragment's landed-row count" trivially.

At commit 1 (this file's own commit) ``p8-capture.json`` does not exist yet
-- no p8 module has landed -- so the parametrization is empty and this file
introduces no failing collection. The RED arm was exhibited manually before
this commit (never committed): a scratch fragment seeding ONE row for a
not-yet-moved module made the cold import of its NEW address fail
(``ModuleNotFoundError``), then the seed was deleted. From commit 2 onward,
once ``p8-capture.json`` exists, an EMPTY collected set FAILS rather than
skips (a fragment that exists but names nothing landed is a regression, not
an absence of subjects) -- see ``test_p8_landed_targets_is_nonempty_once_
fragment_exists``.

NO static ``entropic_substrate.__legacy`` import appears anywhere in this
file (architect constraint 2026-09-23): ``tools/gen_shims.py``'s
``collect_legacy_importers`` scans ``tests/`` as well as ``src/``, and a
static ``__legacy`` import here would create a permanent forwarder
obligation this file does not need -- it only ever imports the NEW address,
in a child process, and reads its own fragment file directly as JSON.

PYTHONPATH is pinned to THIS worktree's absolute ``src``, derived live from
``__file__`` -- never inherited from the ambient environment (nested-worktree
Lens: the second PATH entry under ``mise env`` is the MAIN checkout's
``.venv/bin``, whose editable install targets the unmoved ``src``). The
child prints its own resolved ``entropic_substrate.__file__`` alongside the
success marker so the parent can assert it imported THIS worktree's tree,
not the main checkout's.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FRAGMENT_PATH = REPO_ROOT / ".claude" / "workflows" / "landed-set" / "p8-capture.json"

pytestmark = pytest.mark.unit

_MARKER = "COLD_IMPORT_IDENTICAL"

_COLD_IMPORT_TEMPLATE = """
import importlib
import entropic_substrate
mod = importlib.import_module({dotted!r})
print("ES_FILE=" + entropic_substrate.__file__)
print({marker!r})
"""


def _p8_landed_targets() -> list[str]:
    """Every address THIS phase's own fragment currently marks ``landed:
    true``, sorted, read directly off ``p8-capture.json`` -- never merged
    with sibling fragments and never derived from a ``ladder.capture.*``
    prefix (see module docstring). An absent fragment (before commit 2)
    yields an empty list."""
    if not FRAGMENT_PATH.exists():
        return []
    doc = json.loads(FRAGMENT_PATH.read_text())
    return sorted(
        entry["target"] for entry in doc.get("modules", {}).values() if entry.get("landed", False)
    )


def _run_cold_import(dotted: str) -> subprocess.CompletedProcess[str]:
    code = _COLD_IMPORT_TEMPLATE.format(dotted=dotted, marker=_MARKER)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )


@pytest.mark.parametrize("dotted", _p8_landed_targets())
def test_cold_import_of_landed_capture_target_succeeds_with_nothing_warming_it(
    dotted: str,
) -> None:
    """Fresh interpreter, NOTHING imported first: importing ``dotted``
    directly must succeed and print the marker -- not merely exit 0 -- and
    the resolved ``entropic_substrate.__file__`` must live under THIS
    worktree's ``src``, never a sibling worktree's or the main checkout's."""
    result = _run_cold_import(dotted)
    assert result.returncode == 0, f"cold import of {dotted} failed:\n{result.stderr}"
    assert _MARKER in result.stdout, (
        f"child process for {dotted} did not report success; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )
    es_file_lines = [line for line in result.stdout.splitlines() if line.startswith("ES_FILE=")]
    assert es_file_lines, f"child for {dotted} never printed its resolved __file__"
    resolved_file = Path(es_file_lines[0][len("ES_FILE=") :]).resolve()
    assert resolved_file.is_relative_to(SRC_ROOT), (
        f"child for {dotted} imported entropic_substrate from {resolved_file}, "
        f"not this worktree's src ({SRC_ROOT})"
    )


def test_cold_import_guard_is_not_fooled_by_a_process_that_silently_exits_0() -> None:
    """GREEN control for the instrument itself (mirrors
    tests/test_shim_layer.py's own control): a child that exits 0 WITHOUT
    printing the marker must not satisfy this test's own marker check."""
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert _MARKER not in result.stdout


def test_cold_import_guard_rejects_an_unmoved_address() -> None:
    """RED arm, committed (house rule 5, both arms): ``_run_cold_import``
    itself must fail closed for an address nothing has landed at, not just
    succeed for addresses this phase moved. The module docstring above
    already records that this arm was exhibited manually pre-commit --
    this test is that exhibition, committed, using the same helper the
    GREEN parametrization above calls, so a helper that silently swallowed
    an import error would be caught here as well as there."""
    result = _run_cold_import("entropic_substrate.ladder.capture.__definitely_not_landed__")
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert _MARKER not in result.stdout


def test_p8_landed_targets_is_nonempty_once_fragment_exists() -> None:
    """Once ``p8-capture.json`` exists (commit 2 onward), an EMPTY collected
    set is a fragments regression, not a vacuous pass, and FAILS rather than
    skips. Before commit 2 (this file's own commit) the fragment is absent
    and there is nothing to collect yet, so this test is skipped -- not
    failed -- exactly once, in the commit that introduces this file."""
    if not FRAGMENT_PATH.exists():
        pytest.skip("p8-capture.json does not exist yet (pre-landing-commit state)")
    assert _p8_landed_targets(), "p8-capture.json exists but names zero landed rows"


def test_p8_landed_targets_count_equals_fragment_landed_row_count() -> None:
    """The ACCEPTANCE bullet, encoded directly: the collected subject count
    must equal the fragment's own landed-row count -- not merely be
    non-empty. Skipped before the fragment exists, same as above."""
    if not FRAGMENT_PATH.exists():
        pytest.skip("p8-capture.json does not exist yet (pre-landing-commit state)")
    doc = json.loads(FRAGMENT_PATH.read_text())
    landed_row_count = sum(
        1 for entry in doc.get("modules", {}).values() if entry.get("landed", False)
    )
    assert len(_p8_landed_targets()) == landed_row_count
