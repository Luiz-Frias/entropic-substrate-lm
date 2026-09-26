"""RED-first guard for the ``__legacy/data/build.py`` import cycle (conductor rung
DATA-BUILD-RUNG, 2026-09-23; measured at the p9 design gate of wave w2, second launch).

MEASURED at e01867a: importing ``entropic_substrate.__legacy.data.build`` FIRST in a fresh
interpreter raised ``ImportError: cannot import name 'build_corpus' from partially
initialized module 'entropic_substrate.__legacy.data.build'``. Mechanism: build.py imported
``entropic_substrate.data.batching`` by its OLD name, which ran the verbatim old-path facade
``entropic_substrate/data/__init__.py``, whose line 11 imports the ``data.build`` shim back
into the module still initialising. Today every consumer happens to enter through the old
name first, so the cycle is latent; the first landed consumer that rewrites its own import
of ``data.batching`` onto ``core.corpus.batching`` (p9's ``d1_trainer_runtime.py:15``) and
keeps ``data.build`` at the explicit ``__legacy`` address (FOR-1547) trips it, and a
scratch simulation of the p9 move measured 41 of 101 cold probes failing, including two
h-checkpoint-reader-v2 members and all three pod ``python -m`` entrypoints.

The fix rewrites build.py's five first-party import lines to explicit addresses (landed
ones where the target landed, ``__legacy`` ones for its still-quarantined siblings), so the
facade is never entered from inside the package. Both arms: this test is RED at e01867a
and GREEN after the rewrite; restoring one old-name line makes it RED again.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
import gen_shims  # noqa: E402

pytestmark = pytest.mark.unit

_SRC = REPO_ROOT / "src"
_OLD = "entropic_substrate.data.build"


def _current_address(old: str) -> str:
    """The module's address TODAY, derived live from the landed set (never typed): its
    ``__legacy`` address while quarantined, its landed target once a phase moves it (p8
    moves build.py; ruled P8 gate D-4, 2026-09-23: this test must follow the move, not
    pin the quarantine address)."""
    mapping = gen_shims.load_mapping(REPO_ROOT / "tools" / "mapping.json")
    fragments = gen_shims.load_fragments(REPO_ROOT / ".claude" / "workflows" / "landed-set")
    merged = gen_shims.merge_fragments(fragments, mapping)
    resolved = gen_shims.resolve_all_modules(merged, mapping)
    entry = resolved[old]
    return entry["target"] if entry.get("landed") else gen_shims.legacy_address(old)


def _cold_import_first(dotted: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 -- fixed argv, this interpreter
        [sys.executable, "-c", f"import {dotted} as m; print(m.__name__)"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": ""},
        check=False,
    )


def test_legacy_data_build_imports_cold_when_entered_first() -> None:
    """The quarantined module is importable at its own address with nothing warming the
    old-path facade first -- the exact entry order a rewritten landed consumer produces."""
    address = _current_address(_OLD)
    result = _cold_import_first(address)
    assert result.returncode == 0, result.stderr[-1200:]
    assert result.stdout.strip() == address


def test_old_name_entry_still_resolves_to_the_same_module_GREEN_control() -> None:
    """Control: the shim path is untouched by the rewrite."""
    result = _cold_import_first(_OLD)
    assert result.returncode == 0, result.stderr[-1200:]
    assert result.stdout.strip() == _current_address(_OLD)
