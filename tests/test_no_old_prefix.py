"""The retired-namespace door: post-p13 an "old address" is defined
STRUCTURALLY (P13-NAMESPACE-RULE, amended three times), never by a
retired-address list. A dotted reference ``entropic_substrate.<seg>...`` is
old iff ``<seg>`` is not one of the five landed layer roots or the root
entry ``__main__``, and is not a dunder segment (``__[a-z]+__``:
``entropic_substrate.__file__`` is an attribute, never an address).

This module IS the durable definition of "old" -- the one-shot rewrite tool
(``tools/rewrite_test_addresses.py``) that used to carry this predicate is
deleted in this same commit (B2), one-shot by design (its input class, the
generated shim tree, no longer exists). This file therefore also becomes
the ONE source of ``LANDED_LAYER_PREFIXES``, imported by
``tests/observability_conformance_config.py`` (P13-NAMESPACE-RULE, B2 row 9).

PREDICATE PARITY (architect constraint): this scanner implements every form
the certified B1-tip tool's ``verify`` implemented: AST ``Import``,
absolute ``ImportFrom`` including the root ``from entropic_substrate import
<seg>`` form, non-docstring string ``Constant`` values (including the parts
of an f-string / implicitly concatenated literal, which ``ast.walk`` visits
like any other ``Constant``) searched with the SAME unanchored patterns, and
a line-based, comment-skipping scan of the six non-Python surfaces.

REGIME-INDEPENDENT (architect constraint): this scan reads TRACKED content
only (``ast.parse`` on ``Path.read_text()`` of files this module itself
enumerates via ``.rglob("*.py")`` / literal surface paths) -- it never asks
whether a retired package is *importable*, so it behaves identically in the
lane checkout (which may carry ignored ``__pycache__`` residue for a
deleted package) and in a clean clone.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

# ONE source of the five landed prefixes (moved in from the retired
# tests/test_legacy_coupling.py, B2 row 9).
LANDED_LAYER_PREFIXES = (
    "entropic_substrate.kernel",
    "entropic_substrate.core",
    "entropic_substrate.ladder",
    "entropic_substrate.historical",
    "entropic_substrate.composition",
)

LANDED_TOP_SEGMENTS = frozenset({"kernel", "core", "ladder", "historical", "composition"})
LANDED_OR_ROOT_SEGMENTS = LANDED_TOP_SEGMENTS | {"__main__"}
_DUNDER_RE = re.compile(r"^__[a-z]+__$")

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
CHAIN_RE = re.compile(rf"(?<![A-Za-z0-9_.])entropic_substrate(?:\.{_IDENT})*")
ROOT_FROM_IMPORT_RE = re.compile(
    rf"(?<![A-Za-z0-9_.])from\s+entropic_substrate\s+import\s+({_IDENT})"
)

PY_SUBJECT_DIRS = ("src", "tests", "scripts", "tools", "notebooks")
NONPY_SUBJECTS = (
    "justfile",
    "Containerfile",
    "compose.yaml",
    "scripts/run_ladder.sh",
    "scripts/entrypoints.env",
    "pyproject.toml",
)

# By-name exclusions (P13-NAMESPACE-RULE): whole files excluded from the
# scan entirely, each with its recorded reason.
NAMESPACE_EXCLUSIONS: dict[str, str] = {
    "tests/test_identity_epoch.py": "frozen identity fixture (P13-RPIN)",
    "tools/mapping.json": "the recorded map itself, JSON, byte-identical (P13-MAPPING)",
}

# Wire-format manifest VALUES, excluded by their exact (relative path,
# matched chain) content -- never by line, because an upstream edit in the
# same file can shift the line a value sits on. Every OTHER permanently-
# old-addressed test site (banned-prefix tuples, absence assertions,
# retired-module probes) is composed at runtime instead (never a literal
# chain), so this table stays exactly the three wire-format sites.
WIRE_VALUE_EXCLUSIONS: dict[str, frozenset[str]] = {
    "src/entropic_substrate/ladder/capture/custody/d1_capture_aggregate.py": frozenset(
        {
            "entropic_substrate.telemetry.read_d1_run_evidence",
            "entropic_substrate.telemetry.read_d1_terminal_defect",
        }
    ),
    "src/entropic_substrate/ladder/capture/custody/d1_capture_bundle.py": frozenset(
        {"entropic_substrate.telemetry.read_d1_run_evidence"}
    ),
    "tests/test_d1_capture_validation.py": frozenset(
        {"entropic_substrate.telemetry.read_d1_run_evidence"}
    ),
}


def is_old_seg(seg: str) -> bool:
    if seg in LANDED_OR_ROOT_SEGMENTS:
        return False
    if _DUNDER_RE.fullmatch(seg):
        return False
    return True


def _docstring_node_ids(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
        ):
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


def _self_excluded(rel: str, chain: str) -> bool:
    if rel in WIRE_VALUE_EXCLUSIONS and chain in WIRE_VALUE_EXCLUSIONS[rel]:
        return True
    return False


# EXCLUSION-TABLE IDENTIFIED STRUCTURALLY (architect constraint): this
# module's own WIRE_VALUE_EXCLUSIONS / NAMESPACE_EXCLUSIONS table
# definitions above are the only place a literal old chain may legitimately
# appear in this file's own source. Their defining nodes are found once,
# structurally (by assignment TARGET name, exactly as the retired tool's
# ``_exclusion_def_node_ids`` did), and skipped ONLY when this file scans
# ITSELF -- never a whole-file skip, never a line-number list maintained by
# hand.
_EXCLUSION_TABLE_NAMES = frozenset({"NAMESPACE_EXCLUSIONS", "WIRE_VALUE_EXCLUSIONS"})
_OWN_PATH = Path(__file__).resolve()
_OWN_REL = _OWN_PATH.relative_to(REPO_ROOT).as_posix()


def _exclusion_def_line_numbers(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in _EXCLUSION_TABLE_NAMES for t in targets):
                for sub in ast.walk(node.value):
                    if hasattr(sub, "lineno"):
                        lines.add(sub.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in _EXCLUSION_TABLE_NAMES and node.value is not None:
                for sub in ast.walk(node.value):
                    if hasattr(sub, "lineno"):
                        lines.add(sub.lineno)
    return lines


_OWN_EXCLUSION_DEF_LINES = _exclusion_def_line_numbers(
    ast.parse(_OWN_PATH.read_text(encoding="utf-8"), filename=_OWN_REL)
)


def scan_python_source(rel: str, text: str) -> list[tuple[int, str]]:
    """Returns [(lineno, chain), ...] for every UNEXCLUDED old-address site
    in one Python file's source. Raises SyntaxError to the caller (never
    swallowed -- a subject file that fails to parse is a defect to see)."""
    tree = ast.parse(text, filename=rel)
    doc_ids = _docstring_node_ids(tree)
    exclusion_def_lines = _OWN_EXCLUSION_DEF_LINES if rel == _OWN_REL else frozenset()
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) in exclusion_def_lines:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "entropic_substrate" or not name.startswith("entropic_substrate."):
                    continue
                if not is_old_seg(name.split(".")[1]):
                    continue
                if _self_excluded(rel, name):
                    continue
                hits.append((alias.lineno, name))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            module = node.module or ""
            if module == "entropic_substrate":
                for alias in node.names:
                    if alias.name == "*" or is_old_seg(alias.name):
                        chain = f"entropic_substrate import {alias.name}"
                        if not _self_excluded(rel, chain):
                            hits.append((node.lineno, chain))
            elif module.startswith("entropic_substrate."):
                if is_old_seg(module.split(".")[1]) and not _self_excluded(rel, module):
                    hits.append((node.lineno, module))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in doc_ids:
                continue
            for m in CHAIN_RE.finditer(node.value):
                chain = m.group(0)
                parts = chain.split(".")
                if len(parts) < 2 or not is_old_seg(parts[1]):
                    continue
                if _self_excluded(rel, chain):
                    continue
                hits.append((node.lineno, chain))
            for m in ROOT_FROM_IMPORT_RE.finditer(node.value):
                seg = m.group(1)
                if not is_old_seg(seg):
                    continue
                chain = m.group(0)
                if _self_excluded(rel, chain):
                    continue
                hits.append((node.lineno, chain))
    return hits


def scan_nonpython_source(rel: str, text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        for m in CHAIN_RE.finditer(line):
            chain = m.group(0)
            parts = chain.split(".")
            if len(parts) < 2 or not is_old_seg(parts[1]):
                continue
            if _self_excluded(rel, chain):
                continue
            hits.append((lineno, chain))
        for m in ROOT_FROM_IMPORT_RE.finditer(line):
            seg = m.group(1)
            if not is_old_seg(seg):
                continue
            chain = m.group(0)
            if _self_excluded(rel, chain):
                continue
            hits.append((lineno, chain))
    return hits


def _iter_py_subject_files() -> list[Path]:
    files: list[Path] = []
    for d in PY_SUBJECT_DIRS:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel in NAMESPACE_EXCLUSIONS:
                continue
            files.append(p)
    return files


def find_all_old_address_sites() -> list[tuple[str, int, str]]:
    """Every UNEXCLUDED old-address site over the whole tree: (rel path,
    line, matched chain). Empty iff the retired namespace is truly gone."""
    sites: list[tuple[str, int, str]] = []
    for p in _iter_py_subject_files():
        rel = p.relative_to(REPO_ROOT).as_posix()
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, chain in scan_python_source(rel, text):
            sites.append((rel, lineno, chain))
    for surface in NONPY_SUBJECTS:
        p = REPO_ROOT / surface
        if not p.is_file():
            continue
        rel = surface
        if rel in NAMESPACE_EXCLUSIONS:
            continue
        text = p.read_text(encoding="utf-8")
        for lineno, chain in scan_nonpython_source(rel, text):
            sites.append((rel, lineno, chain))
    return sites


# ---------------------------------------------------------------------------
# Predicate self-tests (guard-teeth for the classifier itself)
#
# NO NEW SKIP MECHANISM (architect constraint): every old-address SAMPLE
# below is composed at runtime via ``_addr(*segs)`` / ``.join()`` rather
# than written as one literal chain -- this file is itself a scanned
# subject (tests/), so a literal sample chain here would be a real
# violation the whole-tree acceptance test below would (correctly) catch.
# Composing at runtime means no single source-level string constant ever
# spells the full sample chain; only the WIRE_VALUE_EXCLUSIONS /
# NAMESPACE_EXCLUSIONS tables above carry literal old chains, and those are
# EXCLUSION_DEF sites (see ``_exclusion_def_line_numbers`` below), never
# ordinary subjects.
# ---------------------------------------------------------------------------


def _addr(*segs: str) -> str:
    return ".".join(("entropic_substrate", *segs))


def test_is_old_seg_accepts_landed_and_root_rejects_everything_else() -> None:
    for seg in ("kernel", "core", "ladder", "historical", "composition", "__main__"):
        assert is_old_seg(seg) is False
    for seg in ("analysis", "evidence", "sentence_prediction", "cli", "telemetry", "sinks"):
        assert is_old_seg(seg) is True


def test_dunder_segments_are_never_old_but_double_underscore_prefix_alone_is() -> None:
    """P13-NAMESPACE-RULE addendum 3: a segment must fullmatch __[a-z]+__ to
    be exempt as a dunder. __legacy and __no_such_module__ do NOT fullmatch
    (no trailing double underscore / uppercase-free-but-wrong-shape) and are
    flagged; __file__ and __version__ DO fullmatch and are exempt."""
    assert is_old_seg("__legacy") is True
    assert is_old_seg("__no_such_module__") is True
    assert is_old_seg("__file__") is False
    assert is_old_seg("__version__") is False


def test_boundary_the_kernelling_sibling_is_never_mistaken_for_kernel() -> None:
    """RED-first boundary self-test: the unanchored CHAIN_RE finds whole
    dot-joined chains, and is_old_seg checks the WHOLE first segment
    against LANDED_OR_ROOT_SEGMENTS -- a hypothetical
    entropic_substrate.kernelling sibling must be flagged old (it is not
    literally "kernel"), never silently accepted as a landed descendant."""
    assert is_old_seg("kernelling") is True
    assert is_old_seg("kernel") is False
    sample = _addr("kernelling", "thing")
    hits = scan_python_source("probe.py", f"x = {sample!r}\n")
    assert hits == [(1, sample)]


def test_docstrings_and_comments_are_never_subjects() -> None:
    sample = _addr("telemetry")
    py_source = f'"""{sample} mentioned in a module docstring."""\nx = 1\n'
    assert scan_python_source("probe.py", py_source) == []
    nonpy_source = f"# {sample} mentioned in a comment\nkey=value\n"
    assert scan_nonpython_source("probe.env", nonpy_source) == []


def test_unanchored_scan_finds_a_mid_string_address_in_subprocess_source() -> None:
    """P13-NAMESPACE-RULE addendum 2: mid-string addresses (e.g. inside a
    subprocess source string) are subjects, not just whole-literal chains."""
    sample = _addr("telemetry")
    py_source = f'code = "import {sample}\\nrun()"\n'
    hits = scan_python_source("probe.py", py_source)
    assert (1, sample) in hits


def test_root_from_import_form_is_a_subject_in_code_and_in_text() -> None:
    stmt = "from entropic_substrate import " + "telemetry"
    hits = scan_python_source("probe.py", stmt + "\n")
    assert hits == [(1, "entropic_substrate import telemetry")]
    hits = scan_nonpython_source("probe.sh", stmt + "\n")
    assert hits == [(1, stmt)]


def test_root_from_import_of_a_landed_segment_is_not_flagged() -> None:
    stmt = "from entropic_substrate import " + "kernel"
    hits = scan_python_source("probe.py", stmt + "\n")
    assert hits == []


def test_by_name_exclusions_and_wire_value_exclusions_suppress_their_sites() -> None:
    # NAMESPACE_EXCLUSIONS is a whole-FILE exclusion, applied at the
    # directory-walk level (_iter_py_subject_files), not inside the scanner
    # itself -- this arm only proves the wire-value table works.
    wire = _addr("telemetry", "read_d1_run_evidence")
    assert _self_excluded("tests/test_d1_capture_validation.py", wire)
    assert not _self_excluded("tests/test_d1_capture_validation.py", _addr("telemetry"))
    assert _self_excluded("tests/test_identity_epoch.py", _addr("telemetry")) is False


def test_injected_old_import_in_an_unexcluded_subject_is_detected_RED() -> None:
    """Guard-teeth (house rule 5): a real old import anywhere must be
    detected -- this is the mutation the ACCEPTANCE table's first row
    names, demonstrated here directly (not merely asserted)."""
    sample = _addr("analysis", "foo")
    hits = scan_python_source("probe.py", f"import {sample} as x\n")
    assert hits == [(1, sample)]


def test_entrypoints_env_line_based_arm_sees_key_value_lines() -> None:
    """The unanchored pattern sees KEY=value lines in scripts/entrypoints.env
    directly (no assignment-expression AST here -- it is not Python)."""
    sample = _addr("cli")
    hits = scan_nonpython_source("scripts/entrypoints.env", f"ENTRYPOINT_CLI={sample}\n")
    assert hits == [(1, sample)]


# ---------------------------------------------------------------------------
# ACCEPTANCE: the whole tree is clean
# ---------------------------------------------------------------------------


def test_zero_old_address_sites_over_the_whole_tree() -> None:
    sites = find_all_old_address_sites()
    assert sites == [], (
        f"{len(sites)} old-address site(s) found outside the by-name exclusions: "
        f"{sites[:20]}{'...' if len(sites) > 20 else ''}"
    )
