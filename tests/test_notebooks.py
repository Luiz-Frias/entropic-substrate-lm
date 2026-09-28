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
import sys
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
    assert "from entropic_substrate.ladder.analyze.series import" in source
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

    # "not ignored" and "actually tracked" are different claims — a file can
    # be un-ignored yet never have reached `git add` at all. `git ls-files
    # --error-unmatch` exits 0 ONLY when the path is in git's index.
    tracked = subprocess.run(  # nosec B603 B607 -- fixed argv, no shell
        ["git", "ls-files", "--error-unmatch", "--", str(NOTEBOOK.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        check=False,
    )
    assert tracked.returncode == 0, f"{NOTEBOOK} is not tracked by git"


def test_notebook_covers_all_six_lane_a_chart_titles() -> None:
    """Chart-set parity guard: Lane B must render the same six charts Lane A
    (entropic_substrate.analysis.charts) does. MEASURED regression this pins:
    the notebook shipped with only 5 of 6 (chart 2, the E1 log-steps washout
    view, was entirely absent) and nothing in this suite caught it — the
    other Lane B tests above check structure/plumbing, not chart-set
    completeness. Titles are the exact strings each Lane A ``fig_*`` function
    sets (``entropic_substrate.analysis.charts``); Lane B's plotly titles
    mirror them so this string match is meaningful, not coincidental.
    """
    source = NOTEBOOK.read_text(encoding="utf-8")
    expected_chart_titles = [
        "E1: fisher vs control — paired loss curves",  # chart 1
        "E1: log-step washout view",  # chart 2
        "E2: per-step train loss — spike/variance texture",  # chart 3
        "E3 vs E2: loss separation",  # chart 4
        "per-head κ trajectory",  # chart 5
        "κ-divergence timing vs loss-gap timing",  # chart 6
    ]
    for title in expected_chart_titles:
        assert title in source, f"Lane B notebook is missing chart: {title!r}"


def test_notebook_shares_one_e1_trace_builder_across_both_e1_charts() -> None:
    """Chart 1 (paired loss curves) and chart 2 (log-step washout view) differ
    only in LAYOUT — the trace loops themselves used to be duplicated
    byte-for-byte, which let the two E1 charts drift apart in the
    color/dash/marker encoding (a redundant non-color channel, not
    decoration). Pins that both chart cells call one shared builder instead
    of rebuilding the trace list."""
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "def e1_traces(" in source
    assert source.count("e1_traces(") >= 3  # one definition + two call sites


def test_notebook_computes_kappa_spread_through_telemetry() -> None:
    """Lane A (entropic_substrate.analysis.charts) and Lane B must produce
    identical kappa spreads for the same run, so the arithmetic lives once
    in entropic_substrate.telemetry rather than being re-derived here."""
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert (
        "from entropic_substrate.ladder.capture.reader.telemetry import max_kappa_spread" in source
    )
    assert "max(max(row) - min(row)" not in source


def test_marimo_check_reports_no_issues() -> None:
    """A real ``marimo check --strict`` run — catches structural notebook
    defects (undefined cell references, cyclic deps) static text checks
    above cannot see.

    Runs marimo through ``sys.executable -m marimo``, never ``uv run``:
    a bare ``uv run`` re-syncs the project ``.venv`` from ``uv.lock`` while
    the suite is executing in it. MEASURED 2026-09-28 (FOR-1618): 26.57 s
    and 55.72 s in CI for this one test (runs 36486275734, 36486548073)
    against 1.4 s for ``python -m marimo check`` on the same notebook, and
    under two xdist workers the re-sync raced ``test_cold_import.py`` into
    ``ModuleNotFoundError`` three runs out of three (36489145501,
    36489147202, 36489150096). ``tests/test_subprocess_env_hygiene.py``
    pins the rule.
    """
    result = subprocess.run(  # nosec B603 -- fixed argv, no shell
        [sys.executable, "-m", "marimo", "check", "--strict", str(NOTEBOOK)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"marimo check found issues:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
