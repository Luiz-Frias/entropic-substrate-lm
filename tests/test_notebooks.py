"""Static contracts for Lane B's marimo notebook(s) (notebooks/*.py).

The telemetry-only-reader invariant binds Lane B exactly as it binds Lane A's
CLI/analysis code: a notebook that parsed JSONL itself would defeat the point
of having a single admissibility predicate. These are STATIC checks (source
text + AST), not a live marimo run — a real headless execution smoke check
lives in the PR description / manual verification, since it needs a marimo
runtime round-trip this unit suite does not want as a hard dependency of
`pytest -m unit`.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "interpretation_charts.py"


def test_notebook_exists() -> None:
    assert NOTEBOOK.exists(), f"expected a Lane B notebook at {NOTEBOOK}"


def test_notebook_is_valid_python() -> None:
    ast.parse(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_a_marimo_app() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "import marimo" in source
    assert "app = marimo.App(" in source
    assert "@app.cell" in source


def test_notebook_renders_with_plotly() -> None:
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "import plotly.graph_objects as go" in source or "import plotly" in source


def test_notebook_reads_only_through_entropic_substrate_analysis_or_telemetry() -> None:
    """The invariant this file exists to pin: no ad-hoc JSONL parsing in Lane B.

    ``entropic_substrate.analysis.series`` (imported here) is itself layered
    exclusively on ``entropic_substrate.telemetry`` — see
    tests/test_analysis_series.py — so importing it is sufficient; the
    notebook must never additionally import/call ``json`` on a results path.
    """
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "from entropic_substrate.analysis.series import" in source
    assert "import json" not in source
    # Calls, not prose: the module docstring/markdown cell mentions
    # "json.loads" BY NAME to explain the invariant, so this checks for an
    # actual invocation (the trailing paren) rather than the bare substring.
    assert "json.loads(" not in source
    assert "json.load(" not in source


def test_notebook_documents_dvc_primary_mlflow_fallback_ordering() -> None:
    """Scope: 'results/captured/ via DVC checkout, with the MLflow artifacts
    API as documented fallback' — pin that the fallback is ACTUALLY gated on
    the local glob coming back empty, not queried unconditionally."""
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "mlflow.artifacts" in source
    assert "local = sorted(dest.glob(glob_pattern))" in source
    assert "if local:\n            return local" in source


def test_notebook_is_tracked_by_git_despite_the_notebooks_scratch_gitignore() -> None:
    """notebooks/* is gitignored as a scratch dir (.gitignore) — this specific
    file must be the explicit exception, or `git add` silently drops it and
    the deliverable never reaches a commit."""
    result = subprocess.run(  # nosec B603 B607 -- fixed argv, no shell
        ["git", "check-ignore", "--quiet", str(NOTEBOOK)],
        cwd=REPO_ROOT,
        check=False,
    )
    # `git check-ignore` exits 0 when the path IS ignored — we need the
    # opposite, so a non-zero exit here is the passing case.
    assert result.returncode != 0, f"{NOTEBOOK} is still gitignored — check .gitignore"


def test_marimo_check_reports_no_issues() -> None:
    """A real ``marimo check --strict`` run — catches structural notebook
    defects (undefined cell references, cyclic deps) static text checks
    above cannot see."""
    result = subprocess.run(  # nosec B603 B607 -- fixed argv, no shell
        ["uv", "run", "marimo", "check", "--strict", str(NOTEBOOK)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"marimo check found issues:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
