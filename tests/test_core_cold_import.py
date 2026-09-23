"""RED-first guard-teeth for the cold-import hazard rewrite-on-arrival closes
(p6-core brief ACCEPTANCE: "Cold import at the NEW address, fresh interpreter
per moved leaf"). MEASURED at 6d35ddd (pre-move): importing the explicit
``__legacy`` address FIRST failed for NINE of the 27 p6-core targets
(``models.backbone``, ``models.field_encoder``, ``observation.byte_codec``,
``observation.modules``, ``predictive_state.{future_law,stabilization,
target_updates,target_values,targets}``) with ``ImportError: cannot import
name ... from partially initialized module``, because the leaf imported a
sibling by its OLD dotted name and the live old-path facade ran first,
re-entering the still-initializing ``__legacy`` leaf.
``tests/test_shim_layer.py`` cannot see this: it imports the OLD prefix
FIRST, which warms the whole facade chain before the ``__legacy`` import
ever runs. This file imports each of the 27 destination ``core.*`` addresses
DIRECTLY, in a fresh interpreter, with NOTHING warming the chain first --
rewrite-on-arrival closes the hazard by construction, because after the
move every intra-set import is an explicit ``core.*`` address, never an old
dotted name.

INSTRUMENT (same discipline as ``tests/test_shim_layer.py``, architect
constraint 4): a subprocess-plus-stdout-marker check, never exit code alone
-- a child that silently no-ops still exits 0. Addresses are derived LIVE
from the landed set via ``gen_shims.load_fragments -> merge_fragments ->
resolve_all_modules``, never a hardcoded list: a literal committed before
all 27 destinations exist would name addresses that do not yet resolve at
intermediate commits (architect constraint, p6-core design gate
2026-09-23). At any given commit this test exercises exactly the rows the
landed set currently marks ``landed: true`` under ``entropic_substrate.core``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:  # tools/ is not a package; import the measurement oracle directly
    sys.path.insert(0, str(TOOLS))
import gen_shims  # noqa: E402

pytestmark = pytest.mark.unit

_MARKER = "COLD_IMPORT_OK"

_COLD_IMPORT_TEMPLATE = """
import importlib
importlib.import_module({dotted!r})
print({marker!r})
"""


def _landed_core_targets() -> list[str]:
    """Every ``core.*`` address the landed set currently marks ``landed:
    true``, sorted -- the live, never-hardcoded address list this file's
    parametrization reads."""
    mapping = gen_shims.load_mapping(REPO_ROOT / "tools" / "mapping.json")
    fragments = gen_shims.load_fragments(REPO_ROOT / ".claude" / "workflows" / "landed-set")
    merged = gen_shims.merge_fragments(fragments, mapping)
    resolved = gen_shims.resolve_all_modules(merged, mapping)
    return sorted(
        entry["target"]
        for entry in resolved.values()
        if entry.get("landed", False) and entry["target"].startswith("entropic_substrate.core")
    )


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


@pytest.mark.parametrize("dotted", _landed_core_targets())
def test_cold_import_of_landed_core_target_succeeds_with_nothing_warming_it(
    dotted: str,
) -> None:
    """Fresh interpreter, NOTHING imported first: importing ``dotted``
    directly must succeed and print the marker -- not merely exit 0."""
    result = _run_cold_import(dotted)
    assert result.returncode == 0, f"cold import of {dotted} failed:\n{result.stderr}"
    assert _MARKER in result.stdout, (
        f"child process for {dotted} did not report success; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
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


def test_landed_core_targets_is_nonempty() -> None:
    """Guard against a fragments/mapping regression silently emptying the
    parametrization above (an empty parametrize list is a vacuous PASS,
    not a real check)."""
    assert _landed_core_targets(), "no landed core.* targets found -- fragments regression?"
