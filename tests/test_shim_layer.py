"""Pytest for the p3-quarantine cold-import guard and the facade-retraction
guard tooth (ACCEPTANCE: "Cold-import guard ... a test spawns a FRESH
interpreter (subprocess) per facade package and imports
entropic_substrate.<facade>.<child> for one child of each of the seven
facades, old prefix first, asserting success and sys.modules identity with
the __legacy module; the in-process sweep's sorted() order can warm the
chain and must not be read as evidence on its own").

INSTRUMENT (architect constraint, design gate 2026-09-20): one fresh
subprocess per facade, launched with `sys.executable` (never a hardcoded
`python`, which would silently test the ambient interpreter instead of
THIS venv's) and the parent test's own environment, importing the
old-prefix dotted path FIRST -- nothing warms `__legacy` ahead of it. The
CHILD process itself evaluates the `sys.modules` identity comparison and
prints an explicit marker; the parent asserts on that stdout text AND the
return code, never on the exit code alone (a child that silently skips
the comparison still exits 0, which is the exact defect the ruling was
written against -- p0's own sorted() sweep warms `__legacy` first because
`"_"` sorts before every letter, so an exit-code-only guard would stay
green through the real hazard)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:  # tools/ is not a package; import the measurement oracle directly
    sys.path.insert(0, str(TOOLS))
import gen_shims  # noqa: E402

pytestmark = pytest.mark.unit

# facade package -> one real child module each restored __init__.py imports
# from (see each facade's docstring/import list at src/entropic_substrate/
# <facade>/__init__.py, restored verbatim after the git mv).
FACADE_CHILDREN = {
    "data": "batching",
    "geometry": "kappa",
    "init": "fisher",
    "layers": "kappa_norm",
    "models": "backbone",
    "observation": "byte_codec",
    "predictive_state": "future_law",
}


def _resolve_child_targets() -> dict[tuple[str, str], str]:
    """The LANDED target address for each facade child, derived from the
    landed set + mapping.json (p6-core Q-1 ruling / architect constraint,
    design gate 2026-09-23: never a hardcoded ``entropic_substrate.__legacy.
    <facade>.<child>`` literal here -- as each child moves out of quarantine
    that address stops existing, and a literal would go stale at exactly the
    commit that lands it). Before a child moves this resolves to its
    ``__legacy`` address (the derived default `resolve_all_modules` fills in
    for any module no fragment names yet); after it moves this resolves to
    its real ``core.*``/``kernel.*`` landing address -- both read live, never
    typed."""
    mapping = gen_shims.load_mapping(REPO_ROOT / "tools" / "mapping.json")
    fragments = gen_shims.load_fragments(REPO_ROOT / ".claude" / "workflows" / "landed-set")
    merged = gen_shims.merge_fragments(fragments, mapping)
    resolved = gen_shims.resolve_all_modules(merged, mapping)
    return {
        (facade, child): resolved[f"entropic_substrate.{facade}.{child}"]["target"]
        for facade, child in FACADE_CHILDREN.items()
    }


FACADE_CHILD_TARGETS = _resolve_child_targets()

_MARKER = "COLD_IMPORT_IDENTICAL"

_COLD_IMPORT_CHILD_TEMPLATE = """
import sys
old_dotted = {old_dotted!r}
target_dotted = {target_dotted!r}
import importlib
old = importlib.import_module(old_dotted)
target = importlib.import_module(target_dotted)
identical = sys.modules[old_dotted] is sys.modules[target_dotted]
print({marker!r} if identical else "COLD_IMPORT_DIFFERENT")
"""


def _run_cold_import(
    facade: str, child: str, *, package_root: Path | None = None
) -> subprocess.CompletedProcess[str]:
    old_dotted = f"entropic_substrate.{facade}.{child}"
    target_dotted = FACADE_CHILD_TARGETS[(facade, child)]
    code = _COLD_IMPORT_CHILD_TEMPLATE.format(
        old_dotted=old_dotted, target_dotted=target_dotted, marker=_MARKER
    )
    env = os.environ.copy()
    if package_root is not None:
        # PYTHONPATH is consulted before site-packages (and so before the
        # editable install's __editable__*.pth entry for src/), so this root
        # SHADOWS the committed tree for the child only. Bytecode writing is
        # off so the overlay's mutated module can never deposit a .pyc in a
        # __pycache__ that the committed tree shares.
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{package_root}{os.pathsep}{existing}" if existing else str(package_root)
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )


def _overlay_with_planted_facade_body(root: Path, facade: str, body: str) -> Path:
    """Mirror `src/` as a symlink farm in which ONE file is a real, rewritten
    file: `entropic_substrate/__legacy/<facade>/__init__.py`.

    Only the directories on the path to that file are materialised; every
    other entry at every level is a symlink to the committed tree, so the
    overlay costs a handful of symlinks rather than a copy of the package.
    `__pycache__` is never linked, so nothing the child imports can write
    bytecode into the committed tree through this farm.
    """
    overlay = root / "overlay"
    overlay.mkdir()
    src_dir, dst_dir = SRC, overlay
    for name in ("entropic_substrate", "__legacy", facade):
        for entry in src_dir.iterdir():
            if entry.name not in (name, "__pycache__"):
                (dst_dir / entry.name).symlink_to(entry)
        src_dir, dst_dir = src_dir / name, dst_dir / name
        dst_dir.mkdir()
    for entry in src_dir.iterdir():
        if entry.name not in ("__init__.py", "__pycache__"):
            (dst_dir / entry.name).symlink_to(entry)
    (dst_dir / "__init__.py").write_text(body)
    return overlay


@pytest.fixture
def committed_legacy_facades_untouched() -> Generator[None, None, None]:
    """Regression pin (FOR-1542): no test in this file may write into the
    committed tree.

    MEASURED 2026-09-22 at 1c10969: the facade-retraction guard tooth below
    used to plant the original re-export body directly in
    `src/entropic_substrate/__legacy/data/__init__.py` and revert it in a
    `finally`. Because `pytest -m unit -n auto --dist loadfile` gives every
    xdist worker the SAME working tree, any cold import in another worker
    that landed inside that window hit the deliberately planted cycle:
    `ImportError: cannot import name 'Corpus' from partially initialized
    module 'entropic_substrate.data.batching'`, reproduced frame-for-frame
    from `entropic_substrate.d1_billing_finalization_values`.

    Bytes alone cannot see that defect -- the write was reverted -- so this
    pins `st_mtime_ns` as well, which the write-then-revert pair does change.
    """
    watched = sorted(
        SRC / "entropic_substrate" / "__legacy" / facade / "__init__.py"
        for facade in FACADE_CHILDREN
    )
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in watched}
    yield
    for path, (payload, mtime_ns) in before.items():
        assert path.read_bytes() == payload, (
            f"{path} was rewritten by this test; every xdist worker shares this "
            "one working tree, so a concurrent cold import sees the mutation"
        )
        assert path.stat().st_mtime_ns == mtime_ns, (
            f"{path} was written and reverted by this test; the bytes match again "
            "but the window was open to every other xdist worker meanwhile"
        )


@pytest.mark.parametrize("facade,child", sorted(FACADE_CHILDREN.items()))
def test_cold_import_of_facade_child_is_sys_modules_identical_to_legacy(
    facade: str, child: str
) -> None:
    """Fresh interpreter, old prefix imported FIRST (nothing else in this
    one-line child script touches __legacy ahead of it): the shim's
    sys.modules rebind must make the old and __legacy module objects
    IDENTICAL, asserted from the child's own stdout marker AND its
    return code -- not the return code alone."""
    result = _run_cold_import(facade, child)
    assert result.returncode == 0, (
        f"cold import of entropic_substrate.{facade}.{child} failed:\n{result.stderr}"
    )
    assert _MARKER in result.stdout, (
        f"child process did not report identity for {facade}.{child}; "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )


def test_cold_import_guard_is_not_fooled_by_a_process_that_silently_exits_0() -> None:
    """GREEN control for the instrument itself: a child process that exits 0
    WITHOUT printing the marker (the defect class the ruling is written
    against -- an exit-code-only guard cannot tell this apart from a real
    identity pass) must NOT satisfy this test's own marker assertion."""
    result = subprocess.run(
        [sys.executable, "-c", "pass"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert _MARKER not in result.stdout


# RETRACTION (house rule 4, ruling P8-SHIM-LAYER-RETIRE, conductor
# 2026-09-23), 2026-09-23: `test_restoring_one_facade_original_bytes_
# breaks_cold_import_RED_case` (facade "data", child "batching") planted a
# cycle that was reachable only while the `data/{build,enwik8,fineweb,
# synthetic}.py` shims imported quarantined `__legacy/data/*` leaves. p8's
# SCC commit moves all four into `ladder/capture/custody/`, so
# `__legacy/data/` now holds no leaf module and the cycle cannot form.
# MEASURED at this lane's tip: the RED case's own RED arm now fails --
# `result.returncode` is 0 (not the expected nonzero) and stdout reads
# `COLD_IMPORT_IDENTICAL`, where the control arm (pre-move) recorded rc=1
# with "partially initialized module 'entropic_substrate.data.build'" and
# the post-move emulation recorded rc=0 with the same marker this lane now
# measures directly. The hazard the test proved load-bearing is therefore
# unreachable, and the test is retired here rather than re-homed: the only
# other `FACADE_CHILDREN` facade with quarantined leaves,
# `predictive_state` (8, all mapped to `ladder.run`), would re-break the
# same way once that lane moves them, so re-homing would only relocate a
# test destined to go RED again. `_overlay_with_planted_facade_body` (this
# file) and the fixture `committed_legacy_facades_untouched` lose their
# only consumer here and are reported as orphaned for the p13 sweep, not
# deleted.
