"""Contracts for the PR-authoring rails: title inheritance and the body template.

A PR has two independently-sourced fields, and GitHub bakes exactly one of them
into permanent history: the squash-merge subject is composed from the PR
*title* as ``<title> (#N)``. Graphite documents two title sources — the first
commit's subject (default), or AI generation when ``gt submit --ai`` is passed,
which overrides it.

This repo already pays for a rule-aware title: ``just stack`` writes the commit
subject via its own generator under gitlint (title <= 72, conventional) and the
commitizen gate. ``--ai`` re-generated the title from the diff, blind to those
rules, and twice produced code-fenced titles that GitHub then baked into
squash subjects failing ``cz check`` at pre-push (merged PRs #32/#33 — the
artifacts ``tests/test_commit_schema.py`` now has to admit). Dropping ``--ai``
makes the title deterministically inherit the checked commit subject; there is
no title left for a rule-blind generator to write.

The template is the body-side rail. Graphite registers PR templates by file
discovery (repo root, ``.github/``, or ``.docs/``), mimicking GitHub — the
``gt config`` picker only chooses among discovered files, so the file's
existence IS the registration. The configured default body is commit messages
*plus* template, so the template must not duplicate the commit rail (no
Summary section — the commit body is the summary); it carries only the
checklist the commits cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ship_it_recipe() -> str:
    """The ship-it recipe body: from its header line to the next rule."""
    justfile = (REPO_ROOT / "justfile").read_text(encoding="utf-8")
    match = re.search(r"^ship-it:.*\n(?:^[ \t]+.*\n?)*", justfile, flags=re.MULTILINE)
    assert match is not None, "justfile no longer defines a ship-it recipe"
    return match.group(0)


def test_ship_it_does_not_regenerate_pr_titles_with_ai() -> None:
    """``--ai`` overrides the PR title; its absence makes the title inherit."""
    recipe = _ship_it_recipe()
    assert "gt submit" in recipe, "ship-it no longer submits via gt"
    assert "--ai" not in recipe, (
        "ship-it passes --ai to gt submit, which REPLACES the PR title the "
        "gitlint/commitizen-checked commit subject would otherwise become. A "
        "rule-blind title reaches the permanent squash-merge subject: this is "
        "how PRs #32/#33 landed code-fenced subjects that turned the pre-push "
        "gate red. Without --ai, gt uses the first commit's subject — the one "
        "artifact in the pipeline that is already generated under the rules."
    )


def test_ship_it_still_submits_the_whole_stack_noninteractively() -> None:
    """Dropping ``--ai`` must not take neighbouring flags with it."""
    recipe = _ship_it_recipe()
    for flag in ("--stack", "--no-edit", "--no-interactive", "--always"):
        assert flag in recipe, f"ship-it lost {flag} while --ai was removed"


def test_pr_template_is_registered_by_existing() -> None:
    """Graphite/GitHub discover the template by path — existence is registration."""
    template = REPO_ROOT / ".github" / "pull_request_template.md"
    assert template.is_file(), (
        ".github/pull_request_template.md is missing. Graphite's gt config "
        "template picker only selects among files it discovers at the GitHub "
        "locations (repo root, .github/, .docs/); with no file there is "
        "nothing to register and the PR body loses its checklist rail."
    )


def test_pr_template_complements_the_commit_rail_not_duplicates_it() -> None:
    """Commit messages are pasted into the body ABOVE the template."""
    text = (REPO_ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    assert "ship-it" in text, (
        "the template's reason to exist is the gate-evidence checklist; its "
        "first item is the live gate (house rule 1)"
    )
    assert not re.search(r"^#+\s*Summary\b", text, flags=re.MULTILINE | re.IGNORECASE), (
        "the template must not carry a Summary section: gt pastes the commit "
        "message(s) into the body above the template, and the commit body IS "
        "the summary — a Summary heading invites re-writing it by hand"
    )
