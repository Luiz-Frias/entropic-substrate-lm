"""Commitizen's schema, and why it is a fork rather than a one-line setting.

GitHub composes a squash-merge subject as ``<PR title> (#N)``. When the PR title
was written by ``gt submit --ai`` it can arrive wrapped in a code fence, so the
subject becomes ``` `docs(x): y` (#33) ``` — conventional on the inside, but no
longer matching at position 0, which is where ``cz check`` anchors. Two such
commits are already merged onto shared branches (#32, #33), authored by GitHub
rather than by us. They are not rewritable, so the schema has to admit them.

The admission is deliberately narrow: a fenced subject is valid *only* when it
also carries the trailing ``(#N)`` that identifies it as forge output. A human
(or an AI titler) writing a fenced subject locally still fails — which matters
here, because the local authoring path is an AI generator with a demonstrated
taste for code fences, and commitizen is the only conventional-commit gate in
the repo (gitlint runs its default rules, which do not include CT1).

MEASURED 2026-08-16, commitizen 4.11.6: a top-level ``[tool.commitizen]
schema_pattern`` key is parsed and then **ignored** — ``ConventionalCommitsCz``
returns a hardcoded regex and never reads the setting. The failure output under
that config is byte-identical to the no-config control. Only ``cz_customize``
consults it. ``test_pattern_is_read_from_the_key_commitizen_honours`` pins that,
because the ignored spelling is the one that looks correct.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

# ``ConventionalCommitsCz.schema_pattern()`` as of commitizen 4.11.6, read out of
# the installed plugin on 2026-08-16. Our pattern is a fork of this one, so it
# does not inherit new commit types added upstream: re-check on a hook rev bump.
STOCK_PATTERN = (
    r"(?s)(build|bump|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\(\S+\))?!?: ([^\n\r]+)((\n\n.*)|(\s*))?$"
)

# Verbatim merge artifacts, the two commits that turned the pre-push gate red.
SQUASH_33 = (
    "`docs(claude): document MLFLOW_S3_ENDPOINT_URL requirement and "
    "container-test preflight abort` (#33)\n"
    "\n"
    "Documents the `MLFLOW_S3_ENDPOINT_URL` requirement in `.env`, noting that "
    "`just container-test` now performs a preflight check.\n"
)
SQUASH_32 = (
    "`docs(justfile): add inline comments for container-build and "
    "container-test recipes` (#32)\n"
    "\n"
    "Adds doc comments to the `container-build` and `container-test` justfile "
    "recipes so they appear in `just --list` output.\n"
)
# 1cb46b5 — internal backticks in a subject that was *not* fenced. Already legal
# under the stock pattern; pinned so the fork cannot regress it.
INTERNAL_BACKTICKS = (
    "chore(dx): add `just ship-it` live gate, pin Graphite 1.8.6 to toolchain, "
    "document house rules in CLAUDE.md (#31)"
)

ACCEPTED = [
    pytest.param("feat: add a thing", id="plain"),
    pytest.param("fix(scope): repair a thing", id="scoped"),
    pytest.param("feat(api)!: drop the old port", id="breaking"),
    pytest.param("docs: a title\n\nA body paragraph.\n", id="with-body"),
    pytest.param(INTERNAL_BACKTICKS, id="internal-backticks-unfenced"),
    pytest.param("`fix(x): y` (#7)", id="fenced-squash-no-body"),
    pytest.param(SQUASH_32, id="squash-32"),
    pytest.param(SQUASH_33, id="squash-33"),
]

REJECTED = [
    # The tooth: a fence is admitted only alongside the forge's PR reference.
    pytest.param("`docs(claude): fenced with no pr reference`", id="fenced-no-pr"),
    # The reference must sit outside the fence, as GitHub writes it.
    pytest.param("`docs(claude): pr reference inside the fence (#9)`", id="pr-inside"),
    pytest.param("`updated stuff` (#33)", id="fenced-nonconventional"),
    pytest.param("updated stuff", id="nonconventional"),
    pytest.param("Feat: capitalised type", id="capitalised-type"),
    pytest.param("feat add a thing", id="no-colon"),
    pytest.param("feat:no space after colon", id="no-space"),
    # Stock rejects a body that is not separated by a blank line; so must we.
    pytest.param("feat: a title\nnot a real body", id="single-newline-body"),
    pytest.param("", id="empty"),
]


def _commitizen_config() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["commitizen"]


def _schema_pattern() -> str:
    pattern = _commitizen_config()["customize"]["schema_pattern"]
    assert isinstance(pattern, str)
    return pattern


def _matches(pattern: str, message: str) -> bool:
    """Mirror ``commitizen.commands.check`` exactly: ``re.match``, not fullmatch."""
    return bool(re.match(pattern, message))


def test_pattern_is_read_from_the_key_commitizen_honours() -> None:
    """A top-level ``schema_pattern`` parses and is then ignored — see docstring."""
    config = _commitizen_config()
    assert config["name"] == "cz_customize", (
        "cz_conventional_commits hardcodes its regex; only cz_customize reads "
        "schema_pattern from config"
    )
    assert "schema_pattern" not in config, (
        "a top-level schema_pattern is a silent no-op; it belongs under [tool.commitizen.customize]"
    )
    assert "schema_pattern" in config["customize"]


@pytest.mark.parametrize("message", ACCEPTED)
def test_accepted(message: str) -> None:
    assert _matches(_schema_pattern(), message)


@pytest.mark.parametrize("message", REJECTED)
def test_rejected(message: str) -> None:
    assert not _matches(_schema_pattern(), message)


@pytest.mark.parametrize("message", ACCEPTED + REJECTED)
def test_pattern_is_a_strict_superset_of_stock(message: str) -> None:
    """Only the forge artifact is newly admitted; nothing stock allows is lost."""
    pattern = _schema_pattern()
    if _matches(STOCK_PATTERN, message):
        assert _matches(pattern, message), "the fork narrowed the stock schema"


def test_the_fork_admits_exactly_the_forge_artifact() -> None:
    """Guard teeth: the *only* behavioural delta from stock is the fenced form."""
    pattern = _schema_pattern()
    delta = [
        m.values[0]
        for m in ACCEPTED + REJECTED
        if _matches(pattern, str(m.values[0])) != _matches(STOCK_PATTERN, str(m.values[0]))
    ]
    assert delta == ["`fix(x): y` (#7)", SQUASH_32, SQUASH_33]
