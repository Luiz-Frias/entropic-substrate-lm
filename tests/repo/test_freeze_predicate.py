"""The H1 freeze of the platform refactor, as a predicate over tracked files.

Ruling R12 (2026-11-27) runs the platform refactor before H1 resumes, and
ruling R25 makes the freeze full: while any reader-identity member is in
motion, no H1 registration is sealed and no H capture is admitted
(`docs/plans/series-plugin-topology.plan.md`, "Rung seven"). Evidence sealed
under an identity that the next horizon re-pins could never be read again.

The freeze is in force while the refactor's sprint manifest is open. It is
found by CONTENT, the manifest whose ``trunk`` is the refactor trunk, and not
by file name: the rotation protocol renames a finished manifest to
``sprint-manifest.<slug>.json``, and a guard keyed on the name would then
read the next sprint's manifest and bind or release the wrong work.

Three observables turn the predicate red while the manifest is open:

* an H1 result pointer, ``results/**/h1-*.dvc``;
* a filled identity row in ``docs/h1/preregistration-skeleton.md``, a table
  row named "Reader identity ..." or "Regime rows" that carries a pinned hex
  value;
* that skeleton without its "NOT SEALED" banner.

The identity rows are found by their instrument name and judged by whether
they carry a pinned value. Nothing here reads a reader family name such as
the ones those rows quote today, because horizon M2 renames every one of
them (ruling R18) and this guard has to survive that rename unchanged.

A state the predicate cannot read is an error, never a pass: no refactor
manifest, two that disagree, a missing skeleton, or a skeleton with no
identity row all raise ``FreezeUnobservable``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import repo_paths

pytestmark = pytest.mark.unit

REFACTOR_TRUNK = "refactor/platform-topology"
WORKFLOWS = Path(".claude") / "workflows"
RESULTS = Path("results")
SKELETON = Path("docs") / "h1" / "preregistration-skeleton.md"
CLOSED = "archived"
BANNER = "NOT SEALED"
BANNER_WINDOW = 12
IDENTITY_ROW_NAMES = ("reader identity", "regime rows")
POINTER_PREFIX = "h1-"
POINTER_SUFFIX = ".dvc"
# A pinned value is a commit or a digest: a short SHA at the least (7 hex
# characters holding a digit and a letter), or any 40-character hex run.
_HEX_RUN = re.compile(r"(?<![0-9A-Za-z])[0-9A-Fa-f]{7,}(?![0-9A-Za-z])")
FULL_SHA = 40


class FreezeUnobservable(RuntimeError):
    """The tree does not let the freeze be evaluated; that is never a pass."""


def refactor_manifests(root: Path) -> list[tuple[Path, str]]:
    """Every sprint manifest of the refactor trunk under ``root``, with its lifecycle."""
    found: list[tuple[Path, str]] = []
    for path in sorted((root / WORKFLOWS).glob("sprint-manifest*.json")):
        if path.name.endswith(".schema.json"):
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("trunk") == REFACTOR_TRUNK:
            found.append((path, str(manifest.get("lifecycle"))))
    return found


def freeze_in_force(root: Path) -> bool:
    """True while the refactor's manifest is open; raises when that cannot be read."""
    manifests = refactor_manifests(root)
    if not manifests:
        message = (
            f"no sprint manifest with trunk {REFACTOR_TRUNK!r} under {root / WORKFLOWS}: "
            "the freeze cannot be evaluated"
        )
        raise FreezeUnobservable(message)
    lifecycles = {lifecycle for _path, lifecycle in manifests}
    if len(lifecycles) > 1:
        names = ", ".join(f"{path.name}={lifecycle}" for path, lifecycle in manifests)
        message = f"refactor manifests disagree on lifecycle: {names}"
        raise FreezeUnobservable(message)
    return lifecycles != {CLOSED}


def h1_result_pointers(root: Path) -> list[Path]:
    """Every ``h1-*.dvc`` pointer under ``results``, at any depth, any letter case."""
    results = root / RESULTS
    if not results.is_dir():
        return []
    return sorted(
        path
        for path in results.rglob(f"*{POINTER_SUFFIX}")
        if path.name.lower().startswith(POINTER_PREFIX)
    )


def identity_rows(skeleton: str) -> list[str]:
    """Table rows whose first cell names a reader identity or the regime rows."""
    rows: list[str] = []
    for line in skeleton.splitlines():
        if not line.startswith("|"):
            continue
        first_cell = line.split("|")[1].strip().lower()
        if first_cell.startswith(IDENTITY_ROW_NAMES):
            rows.append(line)
    return rows


def pinned_values(row: str) -> list[str]:
    """Hex runs in ``row`` that read as a commit or a digest."""
    pinned: list[str] = []
    for token in _HEX_RUN.findall(row):
        mixed = any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)
        if mixed or len(token) >= FULL_SHA:
            pinned.append(token)
    return pinned


def freeze_violations(root: Path) -> list[str]:
    """What breaks the freeze under ``root``; empty when it holds or is lifted."""
    if not freeze_in_force(root):
        return []
    violations = [
        f"H1 result pointer under the freeze: {path.relative_to(root)}"
        for path in h1_result_pointers(root)
    ]
    skeleton_path = root / SKELETON
    if not skeleton_path.is_file():
        message = f"{SKELETON} is absent: the registration state cannot be read"
        raise FreezeUnobservable(message)
    skeleton = skeleton_path.read_text(encoding="utf-8")
    rows = identity_rows(skeleton)
    if not rows:
        message = f"{SKELETON} has no identity row: the predicate would examine nothing"
        raise FreezeUnobservable(message)
    if BANNER not in "\n".join(skeleton.splitlines()[:BANNER_WINDOW]):
        violations.append(f"{SKELETON} lost its {BANNER!r} banner under the freeze")
    for row in rows:
        pinned = pinned_values(row)
        if pinned:
            name = row.split("|")[1].strip()
            violations.append(
                f"{SKELETON} identity row {name!r} is filled under the freeze: {pinned[0]}"
            )
    return violations


# --- the tree as committed ---------------------------------------------------


def test_the_freeze_holds_in_this_tree() -> None:
    assert freeze_violations(repo_paths.REPO_ROOT) == []


def test_the_predicate_reads_real_identity_rows() -> None:
    # Guard against a predicate that passes by examining nothing: the real
    # skeleton must yield identity rows whether or not the freeze is in force.
    skeleton = (repo_paths.REPO_ROOT / SKELETON).read_text(encoding="utf-8")
    assert len(identity_rows(skeleton)) >= 2


def test_renaming_the_reader_families_does_not_move_the_predicate() -> None:
    # M2 renames every reader family under ruling R18. Rewrite each backticked
    # family name in the real identity rows and the verdict must not change.
    skeleton = (repo_paths.REPO_ROOT / SKELETON).read_text(encoding="utf-8")
    renamed = re.sub(r"`[a-z0-9-]+-v\d+`", "`reader-identity-v1`", skeleton)
    assert renamed != skeleton
    assert identity_rows(renamed) != identity_rows(skeleton)
    assert len(identity_rows(renamed)) == len(identity_rows(skeleton))
    assert [pinned_values(row) for row in identity_rows(renamed)] == [
        pinned_values(row) for row in identity_rows(skeleton)
    ]


# --- both arms, on a synthetic tree --------------------------------------------

SKELETON_TEXT = """\
# H1 preregistration skeleton

**DRAFT / NOT SEALED.** This file seals nothing.

## 8. Instruments

| instrument | what it measures | status at base | label |
| --- | --- | --- | --- |
| Importance-weighted log-marginal | `scoring.py:79` | landed | MEASURED |
| Reader identity, checkpoint | `some-reader-v2` (`checkpoint.py:53`) | pin at the sealing SHA; BLANK | PLAN |
| Reader identity, source | `other-reader-v3` (`capture.py:51`) | pin at the sealing SHA; BLANK | PLAN |
| Regime rows | `tools/regime_rows.py` | pin at the sealing SHA | PLAN |
"""
# Derived, not written out: a 64-character hex literal reads as a secret to the
# repository's scanners, and no real digest belongs in this file anyway.
DIGEST = hashlib.sha256(b"freeze-predicate-fixture").hexdigest()
OPEN = "active"


def _tree(
    root: Path,
    *,
    lifecycle: str = OPEN,
    skeleton: str | None = SKELETON_TEXT,
    manifest_name: str = "sprint-manifest.json",
    trunk: str = REFACTOR_TRUNK,
) -> Path:
    (root / WORKFLOWS).mkdir(parents=True)
    (root / WORKFLOWS / manifest_name).write_text(
        json.dumps({"trunk": trunk, "lifecycle": lifecycle}), encoding="utf-8"
    )
    (root / WORKFLOWS / "sprint-manifest.schema.json").write_text(
        json.dumps({"trunk": REFACTOR_TRUNK, "lifecycle": OPEN}), encoding="utf-8"
    )
    (root / RESULTS).mkdir()
    (root / RESULTS / "h0-final-probe.dvc").write_text("outs: []\n", encoding="utf-8")
    if skeleton is not None:
        (root / SKELETON).parent.mkdir(parents=True)
        (root / SKELETON).write_text(skeleton, encoding="utf-8")
    return root


def _filled(row_name: str, value: str) -> str:
    lines = SKELETON_TEXT.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(f"| {row_name}"))
    lines[index] = lines[index].replace("pin at the sealing SHA", f"pinned {value}")
    return "\n".join(lines) + "\n"


def test_a_clean_open_tree_holds(tmp_path: Path) -> None:
    assert freeze_violations(_tree(tmp_path)) == []


@pytest.mark.parametrize(
    "pointer", ["h1-pilot-v1.abc123.dvc", "H1-pilot.dvc", "nested/h1-dev-quote-v4.499aca.dvc"]
)
def test_an_h1_pointer_is_red_while_the_manifest_is_open(tmp_path: Path, pointer: str) -> None:
    root = _tree(tmp_path)
    target = root / RESULTS / pointer
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("outs: []\n", encoding="utf-8")
    violations = freeze_violations(root)
    assert len(violations) == 1
    assert Path(pointer).name in violations[0]


def test_an_h1_pointer_is_admitted_once_the_manifest_is_closed(tmp_path: Path) -> None:
    root = _tree(tmp_path, lifecycle=CLOSED)
    (root / RESULTS / "h1-pilot-v1.abc123.dvc").write_text("outs: []\n", encoding="utf-8")
    assert freeze_violations(root) == []


@pytest.mark.parametrize("name", ["h0-probe.dvc", "h10-probe.dvc", "h1-notes.md", "ah1-x.dvc"])
def test_other_results_files_are_not_h1_pointers(tmp_path: Path, name: str) -> None:
    root = _tree(tmp_path)
    (root / RESULTS / name).write_text("outs: []\n", encoding="utf-8")
    assert freeze_violations(root) == []


@pytest.mark.parametrize(
    "row_name", ["Reader identity, checkpoint", "Reader identity, source", "Regime rows"]
)
@pytest.mark.parametrize(
    "value", [DIGEST, DIGEST.upper(), DIGEST * 2, DIGEST[:40], DIGEST[:12], "9d7917c", "1" * 40]
)
def test_a_filled_identity_row_is_red_while_the_manifest_is_open(
    tmp_path: Path, row_name: str, value: str
) -> None:
    violations = freeze_violations(_tree(tmp_path, skeleton=_filled(row_name, value)))
    assert len(violations) == 1
    assert row_name in violations[0]
    assert value in violations[0]


def test_a_filled_identity_row_is_admitted_once_the_manifest_is_closed(tmp_path: Path) -> None:
    root = _tree(
        tmp_path, lifecycle=CLOSED, skeleton=_filled("Reader identity, checkpoint", DIGEST)
    )
    assert freeze_violations(root) == []


@pytest.mark.parametrize("value", ["2026-11-27", "20261127", "sealing", "deadbee", "v2", "abc12"])
def test_dates_words_and_short_tokens_are_not_pinned_values(tmp_path: Path, value: str) -> None:
    root = _tree(tmp_path, skeleton=_filled("Regime rows", value))
    assert freeze_violations(root) == []


def test_a_hex_value_in_another_row_is_not_an_identity_fill(tmp_path: Path) -> None:
    skeleton = SKELETON_TEXT.replace("| landed |", f"| landed at {DIGEST} |")
    assert skeleton != SKELETON_TEXT
    assert freeze_violations(_tree(tmp_path, skeleton=skeleton)) == []


def test_a_missing_banner_is_red_while_the_manifest_is_open(tmp_path: Path) -> None:
    skeleton = SKELETON_TEXT.replace("**DRAFT / NOT SEALED.** ", "**SEALED 2026-12-01.** ")
    violations = freeze_violations(_tree(tmp_path, skeleton=skeleton))
    assert len(violations) == 1
    assert BANNER in violations[0]


def test_a_banner_below_the_head_of_the_file_does_not_count(tmp_path: Path) -> None:
    skeleton = SKELETON_TEXT.replace("**DRAFT / NOT SEALED.** ", "") + "\nwas NOT SEALED once\n"
    assert len(freeze_violations(_tree(tmp_path, skeleton=skeleton))) == 1


def test_every_violation_is_reported_together(tmp_path: Path) -> None:
    skeleton = _filled("Reader identity, source", DIGEST).replace("NOT SEALED", "SEALED")
    root = _tree(tmp_path, skeleton=skeleton)
    (root / RESULTS / "h1-pilot.dvc").write_text("outs: []\n", encoding="utf-8")
    assert len(freeze_violations(root)) == 3


def test_a_rotated_manifest_still_decides(tmp_path: Path) -> None:
    # After rotation the refactor's manifest carries a slug and another sprint
    # owns the active file name. The refactor's own lifecycle still decides.
    root = _tree(tmp_path, lifecycle=CLOSED, manifest_name="sprint-manifest.platform.json")
    (root / WORKFLOWS / "sprint-manifest.json").write_text(
        json.dumps({"trunk": "main", "lifecycle": OPEN}), encoding="utf-8"
    )
    (root / RESULTS / "h1-pilot.dvc").write_text("outs: []\n", encoding="utf-8")
    assert freeze_violations(root) == []
    (root / WORKFLOWS / "sprint-manifest.platform.json").write_text(
        json.dumps({"trunk": REFACTOR_TRUNK, "lifecycle": "queued"}), encoding="utf-8"
    )
    assert len(freeze_violations(root)) == 1


def test_no_refactor_manifest_is_an_error_not_a_pass(tmp_path: Path) -> None:
    root = _tree(tmp_path, trunk="main")
    (root / RESULTS / "h1-pilot.dvc").write_text("outs: []\n", encoding="utf-8")
    with pytest.raises(FreezeUnobservable, match="no sprint manifest"):
        freeze_violations(root)


def test_disagreeing_refactor_manifests_are_an_error(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / WORKFLOWS / "sprint-manifest.platform.json").write_text(
        json.dumps({"trunk": REFACTOR_TRUNK, "lifecycle": CLOSED}), encoding="utf-8"
    )
    with pytest.raises(FreezeUnobservable, match="disagree"):
        freeze_violations(root)


def test_a_missing_skeleton_is_an_error_while_the_manifest_is_open(tmp_path: Path) -> None:
    with pytest.raises(FreezeUnobservable, match="absent"):
        freeze_violations(_tree(tmp_path, skeleton=None))


def test_a_skeleton_without_identity_rows_is_an_error(tmp_path: Path) -> None:
    skeleton = "\n".join(
        line
        for line in SKELETON_TEXT.splitlines()
        if not line.lower().startswith(("| reader identity", "| regime rows"))
    )
    with pytest.raises(FreezeUnobservable, match="no identity row"):
        freeze_violations(_tree(tmp_path, skeleton=skeleton))
