"""ACCEPTANCE instrument, not a guard-teeth row (brief Q-3 ruling, p11 form):
cold import at every landed address, fresh interpreter per leaf (p11-analyze-
review-file brief ACCEPTANCE: "Cold import at the NEW address, fresh
interpreter per moved leaf"). Same discipline as p6-core's
``tests/test_core_cold_import.py`` and p7's ``tests/test_register_gate_cold_import.py``:
this file imports each ``entropic_substrate.ladder.{analyze,review,file}.*``
destination address directly, in a fresh interpreter, with NOTHING warming
the chain first.

This arm is also the load-bearing guard for the ``codec_equivariance.py:21``
reclassification (house rule 4 retraction): ``entropic_substrate.__legacy.observation``
exports nothing, so a literal reading of the old lane pack's bucketing fails
this test, while the ruled ``core.observation.contracts`` form imports clean.

INSTRUMENT (same discipline as ``tests/test_shim_layer.py`` and
``tests/test_core_cold_import.py``, architect constraint 3): a
subprocess-plus-stdout-marker check, never exit code alone -- a child that
silently no-ops still exits 0. Addresses are derived LIVE from the landed
set via ``gen_shims.load_fragments -> merge_fragments -> resolve_all_modules``,
never a hardcoded 47-address literal. At any given commit this file
exercises exactly the rows the landed set currently marks ``landed: true``
under ``entropic_substrate.ladder.analyze``, ``.review`` or ``.file``.
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

_TARGET_PREFIXES = (
    "entropic_substrate.ladder.analyze",
    "entropic_substrate.ladder.review",
    "entropic_substrate.ladder.file",
)


def _landed_analyze_review_file_targets() -> list[str]:
    """Every ``ladder.analyze.*`` / ``ladder.review.*`` / ``ladder.file.*``
    address the landed set currently marks ``landed: true``, sorted -- the
    live, never-hardcoded address list this file's parametrization reads."""
    mapping = gen_shims.load_mapping(REPO_ROOT / "tools" / "mapping.json")
    fragments = gen_shims.load_fragments(REPO_ROOT / ".claude" / "workflows" / "landed-set")
    merged = gen_shims.merge_fragments(fragments, mapping)
    resolved = gen_shims.resolve_all_modules(merged, mapping)
    return sorted(
        entry["target"]
        for entry in resolved.values()
        if entry.get("landed", False) and entry["target"].startswith(_TARGET_PREFIXES)
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


@pytest.mark.parametrize("dotted", _landed_analyze_review_file_targets())
def test_cold_import_of_landed_analyze_review_file_target_succeeds_with_nothing_warming_it(
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
    tests/test_shim_layer.py's and tests/test_core_cold_import.py's own
    controls): a child that exits 0 WITHOUT printing the marker must not
    satisfy this test's own marker check."""
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert _MARKER not in result.stdout


def test_landed_analyze_review_file_targets_is_nonempty() -> None:
    """Guard against a fragments/mapping regression silently emptying the
    parametrization above (an empty parametrize list is a vacuous PASS,
    not a real check)."""
    assert _landed_analyze_review_file_targets(), (
        "no landed ladder.analyze/review/file targets found -- fragments regression?"
    )
