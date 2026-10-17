"""Observability conformance -- the rail p3b authors, frozen_after p13.

Per the PRIORITY AMENDMENT (ruling P-1): observability wiring is the ONE
behaviour-change class permitted inside a structural phase, scoped to log
routing, metric routing and entry-point configuration. This module is the
contract every later phase conforms to as it moves more of the tree; it is
NOT edited again until p13 (the eventual sweep), except that DN-OBS-CONF
(operator 2026-10-13, applied 2026-10-16 by FOR-1645) unfroze its entry-point
universe assertions, and only those, for four H host entry points. Per-phase editable state
(the entry-point partition, the print allowlist, the sink-construction
seam) lives in ``tests/observability_conformance_config.py`` beside it
(ENTRY-POINT ARM RULING, conductor 2026-09-21, wf_3b1acf4d-36f) -- never
inline here.

Three arms, each AST-based (never a source-text grep -- MEASURED,
architect constraint 2026-09-21: ``entropic_substrate.kernel.runtime.logging``'s
own module docstring contains the literal token ``print()`` twice, in prose;
a text grep scoped to landed packages would go RED on the substrate's own
documentation on day one) and each asserting the SIZE of its own subject set
so a zero-subject pass reads as a bug in the arm, not a green contract
(ACCEPTANCE):

1. Zero bare ``print(`` calls anywhere under a LANDED package, outside the
   two pinned machine-consumed stdout sites (named in
   ``observability_conformance_config.PRINT_ALLOWLIST``).
2. Every one of the seven ``entrypoint: true`` rows in ``tools/mapping.json``
   is accounted for by the CONFORMING/NOT-YET-CONFORMED partition, and every
   CONFORMING entry point's ``main()`` calls ``configure_logging`` as its
   first statement.
3. Every construction site of a ``MetricSink``-implementing class inside a
   landed package resolves to the run-descriptor door's single construction
   seam (``ladder.capture.adapters.composite``).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import observability_conformance_config as occ  # noqa: E402

pytestmark = pytest.mark.unit


def _landed_py_files() -> list[Path]:
    src = REPO_ROOT / "src" / "entropic_substrate"
    files: list[Path] = []
    for prefix in occ.LANDED_LAYER_PREFIXES:
        rel = Path(*prefix.split(".")[1:])  # strip the "entropic_substrate" root
        root = src / rel
        if root.is_dir():
            files.extend(sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts))
        elif root.with_suffix(".py").is_file():
            files.append(root.with_suffix(".py"))
    return files


def _module_dotted_name(path: Path) -> str:
    src = REPO_ROOT / "src"
    rel = path.relative_to(src).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Arm 1: zero bare print( outside the pinned allowlist, scoped to landed
# packages, AST-based.
# ---------------------------------------------------------------------------


def _find_print_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            lines.append(node.lineno)
    return lines


def test_print_allowlist_is_the_exact_frozen_ten_name_set() -> None:
    assert len(occ.PRINT_ALLOWLIST) == 10
    assert occ.PRINT_ALLOWLIST == {
        "entropic_substrate.composition.capture_cli",
        "entropic_substrate.composition.cli",
        "entropic_substrate.composition.entrypoints.cli",
        "entropic_substrate.composition.source_cli",
        "entropic_substrate.composition.span_cli",
        "entropic_substrate.composition.wiring.d1_probe_cli",
        "entropic_substrate.composition.wiring.d_ladder_cli",
        "entropic_substrate.composition.wiring.d_ladder_probe_cli",
        "entropic_substrate.ladder.capture.reader.telemetry",
        "entropic_substrate.ladder.review.d1_external_worker",
    }


def test_zero_bare_print_calls_in_landed_packages_outside_the_allowlist() -> None:
    files = _landed_py_files()
    assert files, (
        "zero landed .py files scanned -- the print-ban arm's subject set is "
        "empty, which is a bug in the scan (or the classifier), not a passing gate"
    )
    offenders: list[str] = []
    for path in files:
        mod = _module_dotted_name(path)
        if mod in occ.PRINT_ALLOWLIST:
            continue
        for lineno in _find_print_calls(path):
            offenders.append(f"{mod}:{lineno}")
    assert not offenders, f"bare print( found outside the allowlist: {offenders}"


def test_print_allowlist_orphan_detection_every_allowlisted_module_has_a_print() -> None:
    """Mutation A guard (P13-OBS): an allowlisted module with zero print(
    sites is an orphan -- the allowlist must never carry dead entries."""
    files = {_module_dotted_name(p): p for p in _landed_py_files()}
    orphans = [
        mod for mod in occ.PRINT_ALLOWLIST if mod not in files or not _find_print_calls(files[mod])
    ]
    assert not orphans, f"allowlisted module(s) with no print( site: {orphans}"


def test_print_ban_arm_subject_set_scans_every_landed_file_GREEN_control() -> None:
    """Guard-tooth for the scan itself: a file with a real print() call, read
    through the SAME AST walk, is detected -- the arm above is discriminating,
    not vacuously green because nothing it looks at ever has a print()."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "probe.py"
        p.write_text("def f():\n    print('hello')\n")
        assert _find_print_calls(p) == [2]


# ---------------------------------------------------------------------------
# Arm 2: entry-point partition + CONFORMING body inspection.
# ---------------------------------------------------------------------------


def _mapping_entrypoints() -> frozenset[str]:
    mapping = json.loads((REPO_ROOT / "tools" / "mapping.json").read_text())
    return frozenset(
        old for old, meta in mapping["modules"].items() if meta.get("entrypoint", False)
    )


def test_entrypoint_partition_covers_all_seven_rows_with_at_least_two_conforming() -> None:
    """P13-OBS (addendum 2), 2026-09-26: the `:139` equality against
    ``tools/mapping.json``'s seven OLD-named rows is replaced by an
    exact-name assertion on the two inlined sets -- `tools/mapping.json`
    keeps its old-name keys as history (P13-MAPPING), never live addresses,
    so it can no longer be compared directly against the LANDED partition.
    The universe-size cross-check against the frozen file stays: it still
    proves the partition below covers exactly as many entry points as the
    frozen map ever named."""
    # RETRACTED 2026-10-16 (DN-OBS-CONF, FOR-1645): this test's name and the "seven"
    # wording above (and in the banner) describe the universe as it stood at p13. Four H
    # host entry points (h_probe_command, h_probe_stop, h_tiny_tracer, h_wave_monitor)
    # joined as CONFORMING and tools/mapping.json gained their entrypoint:true rows, so the
    # universe is ELEVEN. The function name and that wording are retracted by this reference,
    # not renamed (the contract file is otherwise frozen).
    mapping_rows = _mapping_entrypoints()
    assert len(mapping_rows) == 11, (
        f"tools/mapping.json names {len(mapping_rows)} entrypoint:true row(s), expected 11 "
        "-- the partition below is defined over exactly eleven rows"
    )
    assert occ.CONFORMING_ENTRYPOINTS == {
        "entropic_substrate.composition.entrypoints.cli",
        "entropic_substrate.ladder.run.worker_command",
        "entropic_substrate.ladder.capture.reader.telemetry",
        "entropic_substrate.ladder.run.worker.h_probe_command",
        "entropic_substrate.ladder.run.worker.h_probe_stop",
        "entropic_substrate.composition.wiring.h_tiny_tracer",
        "entropic_substrate.composition.entrypoints.h_wave_monitor",
    }
    assert occ.NOT_YET_CONFORMED_ENTRYPOINTS == {
        "entropic_substrate.composition.entrypoints.__main__",
        "entropic_substrate.ladder.review.d1_external_worker",
        "entropic_substrate.composition.entrypoints.d1_trainer_mailbox_command",
        "entropic_substrate.historical.d_ladder.d_ladder_worker_command",
    }
    assert set(occ.NOT_YET_CONFORMED_REASONS) == occ.NOT_YET_CONFORMED_ENTRYPOINTS
    assert all(occ.NOT_YET_CONFORMED_REASONS.values()), "every NOT_YET reason must be non-empty"
    assert not (occ.CONFORMING_ENTRYPOINTS & occ.NOT_YET_CONFORMED_ENTRYPOINTS)
    assert len(occ.CONFORMING_ENTRYPOINTS) + len(occ.NOT_YET_CONFORMED_ENTRYPOINTS) == 11
    assert len(occ.CONFORMING_ENTRYPOINTS) >= 2


def _main_function(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            return node
    return None


def _first_statement_calls(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """True iff the first REAL statement of ``node`` (skipping a leading
    docstring, which is its own ``ast.Expr(Constant(str))`` statement and
    not the call this checks for) is a bare call to ``name``."""
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if not body:
        return False
    first = body[0]
    call = first.value if isinstance(first, ast.Expr) else None
    return bool(call is not None and isinstance(call, ast.Call) and _call_name(call) == name)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


@pytest.mark.parametrize("dotted", sorted(occ.CONFORMING_ENTRYPOINTS))
def test_every_conforming_entrypoint_calls_configure_logging_first_in_main(dotted: str) -> None:
    """Resolves the CONFORMING entry point's real source via its imported
    module's own ``__file__`` -- never assumed at ``src/entropic_substrate/<name>.py``
    (design-gate architect constraint, 2026-09-21): today's shims for both
    CONFORMING modules still point at ``entropic_substrate.__legacy.<x>``,
    and the day either lands, its ``__file__`` moves with it -- this test
    keeps working across that move with zero edit."""
    import importlib

    module = importlib.import_module(dotted)
    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    main_fn = _main_function(tree)
    assert main_fn is not None, f"{dotted} ({source_path}) defines no main() function"
    assert _first_statement_calls(main_fn, "configure_logging"), (
        f"{dotted}'s main() ({source_path}:{main_fn.lineno}) does not call "
        "configure_logging() as its first statement"
    )


# ---------------------------------------------------------------------------
# Arm 3: MetricSink-implementing class construction sites resolve to the
# run-descriptor door's single construction seam.
# ---------------------------------------------------------------------------


def _find_sink_construction_sites(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in occ.SINK_IMPLEMENTATION_CLASS_NAMES:
            sites.append((node.lineno, name))
    return sites


def test_allowed_sink_construction_modules_is_the_exact_frozen_three_name_set() -> None:
    assert len(occ.ALLOWED_SINK_CONSTRUCTION_MODULES) == 3
    assert occ.ALLOWED_SINK_CONSTRUCTION_MODULES == {
        "entropic_substrate.historical.e_series.field_pretrain",
        "entropic_substrate.historical.e_series.train",
        "entropic_substrate.ladder.capture.adapters.composite",
    }


def test_sink_construction_sites_resolve_to_the_single_construction_seam() -> None:
    files = _landed_py_files()
    all_sites: list[str] = []
    offenders: list[str] = []
    for path in files:
        mod = _module_dotted_name(path)
        for lineno, name in _find_sink_construction_sites(path):
            entry = f"{mod}:{lineno} ({name})"
            all_sites.append(entry)
            if mod not in occ.ALLOWED_SINK_CONSTRUCTION_MODULES:
                offenders.append(entry)
    assert all_sites, (
        "zero MetricSink-implementing construction sites found in any landed package -- "
        "the sink-construction arm's subject set is empty, which is a bug in the scan, "
        "not a passing gate"
    )
    assert not offenders, (
        f"MetricSink-implementing class constructed outside "
        f"{sorted(occ.ALLOWED_SINK_CONSTRUCTION_MODULES)!r}: {offenders}"
    )


def test_sink_construction_arm_detects_a_construction_site_outside_the_seam_GREEN_control() -> None:
    """Guard-tooth for the scan itself: a synthetic file constructing
    ``JsonlSink`` outside the allowed module is detected -- proves the scan
    is discriminating, not vacuously green because nothing it looks at ever
    constructs a sink."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "probe.py"
        p.write_text("sink = JsonlSink('x.jsonl')\n")
        sites = _find_sink_construction_sites(p)
        assert sites == [(1, "JsonlSink")]
