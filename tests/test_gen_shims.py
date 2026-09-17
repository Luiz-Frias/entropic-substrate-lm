"""Pytest for gen_shims.py. House rule 5 (TDD non-negotiable, guard-teeth prove
both arms): every RED-first case below is paired with a GREEN control proving
the same check does NOT fire on the thing it must let through.

RETRACTION HISTORY (2026-09-17, same session):
1. An early draft asserted "a landed module gets no shim" -- wrong, inverted
   the strangler design. Corrected: a module keeps its shim while listed;
   `landed` controls where it points, not whether it exists.
2. That corrected draft read a single `landed-set.json` manifest and emitted
   an import-linter `ignore_imports` section. Superseded by ruling DN-P1
   (docs/plans/src-hexagonal-refactor.plan.md, "The mechanism rule that makes phases
   disjoint", verified 2026-09-17): a single shared manifest, and a shared
   ignore list, are both hubs that parallel wave lanes would collide on.
   Replaced by a DIRECTORY of disjoint per-phase fragments
   (`.claude/workflows/landed-set/<phase-id>.json`) merged by this tool
   against `mapping.json`, with NO ignore-list output anywhere, plus a
   per-phase `legacy_edge_budget` this tool can report via `--print-budget`.
   This file's tests below are for THIS (current, verified) design; the two
   retracted designs' tests are not reproduced here (git history has them).

Run standalone (no repo pytest config picked up):
    uv run --project <repo> --no-sync pytest tools/test_gen_shims.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:  # tools/ is not a package; import the generator directly
    sys.path.insert(0, str(TOOLS))
import gen_shims  # noqa: E402

pytestmark = pytest.mark.unit


def write_mapping(tmp_path: Path, module_names: list[str]) -> Path:
    """A minimal but structurally-real mapping.json: just needs the module
    names to exist as keys, matching the real mapping.json's shape enough
    for load_mapping()/merge_fragments() to work against it."""
    mapping = {
        "layers": ["kernel", "core", "ladder", "__legacy", "composition"],
        "modules": {
            m: {"dest": "entropic_substrate.somewhere", "layer": "ladder", "action": "move"}
            for m in module_names
        },
    }
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps(mapping))
    return p


def write_fragment(
    fragments_dir: Path, phase: str, modules: dict, legacy_edge_budget: int = 0
) -> Path:
    fragments_dir.mkdir(parents=True, exist_ok=True)
    doc = {"phase": phase, "modules": modules, "legacy_edge_budget": legacy_edge_budget}
    p = fragments_dir / f"{phase}.json"
    p.write_text(json.dumps(doc))
    return p


# ---------------------------------------------------------------------------
# Shim-form AST checks (unaffected by the fragment-directory redesign).
# ---------------------------------------------------------------------------


def test_star_import_shim_is_rejected_by_verify(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    bad = out / "bad_shim.py"
    bad.write_text("from entropic_substrate.__legacy.thing import *\n")
    failures = gen_shims.verify_dir(out)
    assert len(failures) == 1
    assert failures[0][0] == bad


def test_exact_form_shim_passes_verify_GREEN_control(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    good = out / "good_shim.py"
    good.write_text(
        gen_shims.shim_source(
            "entropic_substrate.analysis.charts",
            "entropic_substrate.__legacy.analysis.charts",
            "fragments/",
        )
    )
    assert gen_shims.verify_dir(out) == []


def test_verify_rejects_importlib_string_shim(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    bad = out / "importlib_shim.py"
    bad.write_text(
        "import importlib, sys\n"
        "_target = importlib.import_module('entropic_substrate.__legacy.thing')\n"
        "sys.modules[__name__] = _target\n"
    )
    assert len(gen_shims.verify_dir(out)) == 1


# ---------------------------------------------------------------------------
# RED-first: fragment disjointness. A module named in two fragments is a
# fail-loud merge error, not a silent last-writer-wins.
# ---------------------------------------------------------------------------


def test_module_named_in_two_fragments_is_a_disjointness_error(tmp_path: Path) -> None:
    mapping = json.loads(write_mapping(tmp_path, ["entropic_substrate.shared_thing"]).read_text())
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.shared_thing": {
                "target": "entropic_substrate.kernel.runtime.shared_thing",
                "landed": True,
            }
        },
    )
    write_fragment(
        frags_dir,
        "p6-core",
        {
            "entropic_substrate.shared_thing": {
                "target": "entropic_substrate.core.geometry.shared_thing",
                "landed": True,
            }
        },
    )
    fragments = gen_shims.load_fragments(frags_dir)
    with pytest.raises(gen_shims.ManifestError, match="shared_thing"):
        gen_shims.merge_fragments(fragments, mapping)


def test_disjoint_fragments_merge_cleanly_GREEN_control(tmp_path: Path) -> None:
    """Control for the case above: two fragments naming DIFFERENT modules
    must merge without error, or the disjointness check would reject every
    multi-fragment merge, not just colliding ones."""
    mapping = json.loads(
        write_mapping(
            tmp_path, ["entropic_substrate.a_thing", "entropic_substrate.b_thing"]
        ).read_text()
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.a_thing": {
                "target": "entropic_substrate.kernel.runtime.a_thing",
                "landed": True,
            }
        },
    )
    write_fragment(
        frags_dir,
        "p6-core",
        {
            "entropic_substrate.b_thing": {
                "target": "entropic_substrate.core.geometry.b_thing",
                "landed": True,
            }
        },
    )
    fragments = gen_shims.load_fragments(frags_dir)
    merged = gen_shims.merge_fragments(fragments, mapping)
    assert set(merged) == {"entropic_substrate.a_thing", "entropic_substrate.b_thing"}


def test_verify_reports_disjointness_violation_via_main(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.shared_thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.shared_thing": {
                "target": "entropic_substrate.kernel.runtime.shared_thing",
                "landed": True,
            }
        },
    )
    write_fragment(
        frags_dir,
        "p6-core",
        {
            "entropic_substrate.shared_thing": {
                "target": "entropic_substrate.core.geometry.shared_thing",
                "landed": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--verify"])
    assert rc == 1


def test_module_unknown_to_mapping_is_a_fail_loud_error(tmp_path: Path) -> None:
    mapping = json.loads(write_mapping(tmp_path, ["entropic_substrate.known_thing"]).read_text())
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.typo_thing": {
                "target": "entropic_substrate.kernel.runtime.typo_thing",
                "landed": True,
            }
        },
    )
    fragments = gen_shims.load_fragments(frags_dir)
    with pytest.raises(gen_shims.ManifestError, match="typo_thing"):
        gen_shims.merge_fragments(fragments, mapping)


# ---------------------------------------------------------------------------
# landed/retired semantics: unchanged from the prior correction, re-verified
# against the merge path.
# ---------------------------------------------------------------------------


def test_unlanded_module_gets_a_shim_to_legacy(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.old_thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.old_thing": {
                "target": "entropic_substrate.__legacy.old_thing",
                "landed": False,
            }
        },
    )
    out_dir = tmp_path / "out"
    fragments = gen_shims.load_fragments(frags_dir)
    mapping = gen_shims.load_mapping(mapping_path)
    merged = gen_shims.merge_fragments(fragments, mapping)
    written = gen_shims.generate(merged, str(frags_dir), out_dir)
    out = out_dir / "entropic_substrate" / "old_thing.py"
    assert written == [out]
    assert "entropic_substrate.__legacy.old_thing" in out.read_text()


def test_landed_module_still_gets_a_shim_at_new_home(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.old_thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p6-core",
        {
            "entropic_substrate.old_thing": {
                "target": "entropic_substrate.ladder.run.new_thing",
                "landed": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    fragments = gen_shims.load_fragments(frags_dir)
    mapping = gen_shims.load_mapping(mapping_path)
    merged = gen_shims.merge_fragments(fragments, mapping)
    written = gen_shims.generate(merged, str(frags_dir), out_dir)
    out = out_dir / "entropic_substrate" / "old_thing.py"
    assert written == [out]
    assert "entropic_substrate.ladder.run.new_thing" in out.read_text()
    assert "__legacy" not in out.read_text()


def test_retired_module_gets_no_shim(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.old_thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p13-sweep",
        {
            "entropic_substrate.old_thing": {
                "target": "entropic_substrate.ladder.run.new_thing",
                "landed": True,
                "retired": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    fragments = gen_shims.load_fragments(frags_dir)
    mapping = gen_shims.load_mapping(mapping_path)
    merged = gen_shims.merge_fragments(fragments, mapping)
    written = gen_shims.generate(merged, str(frags_dir), out_dir)
    assert written == []
    assert not (out_dir / "entropic_substrate" / "old_thing.py").exists()


def test_landed_true_with_legacy_target_rejected(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.confused"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.confused": {
                "target": "entropic_substrate.__legacy.confused",
                "landed": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)])
    assert rc == 1


def test_landed_false_with_non_legacy_target_rejected_GREEN_mirror(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.confused2"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.confused2": {
                "target": "entropic_substrate.ladder.run.already_promoted",
                "landed": False,
            }
        },
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)])
    assert rc == 1


# ---------------------------------------------------------------------------
# Package __init__ handling (unchanged).
# ---------------------------------------------------------------------------


def test_init_shim_emitted_as_package_file(tmp_path: Path) -> None:
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.analysis", "entropic_substrate.analysis.charts"]
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.analysis": {
                "target": "entropic_substrate.__legacy.analysis",
                "landed": False,
                "package": True,
            },
            "entropic_substrate.analysis.charts": {
                "target": "entropic_substrate.__legacy.analysis.charts",
                "landed": False,
            },
        },
    )
    out_dir = tmp_path / "out"
    fragments = gen_shims.load_fragments(frags_dir)
    mapping = gen_shims.load_mapping(mapping_path)
    merged = gen_shims.merge_fragments(fragments, mapping)
    written = gen_shims.generate(merged, str(frags_dir), out_dir)
    init_path = out_dir / "entropic_substrate" / "analysis" / "__init__.py"
    child_path = out_dir / "entropic_substrate" / "analysis" / "charts.py"
    assert init_path in written
    assert child_path in written
    assert not (out_dir / "entropic_substrate" / "analysis.py").exists()


def test_unflagged_package_prefix_fails_loud(tmp_path: Path) -> None:
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.analysis", "entropic_substrate.analysis.charts"]
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.analysis": {
                "target": "entropic_substrate.__legacy.analysis",
                "landed": False,
            },
            "entropic_substrate.analysis.charts": {
                "target": "entropic_substrate.__legacy.analysis.charts",
                "landed": False,
            },
        },
    )
    out_dir = tmp_path / "out"
    fragments = gen_shims.load_fragments(frags_dir)
    mapping = gen_shims.load_mapping(mapping_path)
    merged = gen_shims.merge_fragments(fragments, mapping)
    with pytest.raises(gen_shims.ManifestError, match="analysis"):
        gen_shims.generate(merged, str(frags_dir), out_dir)


# ---------------------------------------------------------------------------
# RED-first: no ignore-list output anywhere. DN-P1 removed the ignore list
# entirely -- there is no flag, no function, and no file this tool writes
# that resembles one.
# ---------------------------------------------------------------------------


def test_no_ignore_list_function_exists() -> None:
    assert not hasattr(gen_shims, "build_ignore_list")


def test_cli_has_no_ignore_list_flag(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.thing": {
                "target": "entropic_substrate.kernel.runtime.thing",
                "landed": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit):
        gen_shims.main(
            [
                str(frags_dir),
                str(mapping_path),
                str(out_dir),
                "--ignore-list-out",
                str(tmp_path / "ignored.cfg"),
            ]
        )
    assert not (tmp_path / "ignored.cfg").exists()


def test_generate_writes_no_ini_or_cfg_files(tmp_path: Path) -> None:
    """GREEN-control-shaped sweep: after a normal generate() run, nothing
    under out_dir looks like an ignore-list artifact (.cfg/.ini), only *.py
    shims -- a direct behavioural check, not just an API-surface check."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.thing": {
                "target": "entropic_substrate.kernel.runtime.thing",
                "landed": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    fragments = gen_shims.load_fragments(frags_dir)
    mapping = gen_shims.load_mapping(mapping_path)
    merged = gen_shims.merge_fragments(fragments, mapping)
    gen_shims.generate(merged, str(frags_dir), out_dir)
    non_py = [p for p in out_dir.rglob("*") if p.is_file() and p.suffix != ".py"]
    assert non_py == []


# ---------------------------------------------------------------------------
# RED-first: --print-budget reports the highest-ordered present fragment's
# legacy_edge_budget.
# ---------------------------------------------------------------------------


def test_print_budget_reports_highest_ordered_fragment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.a", "entropic_substrate.b"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {"entropic_substrate.a": {"target": "entropic_substrate.kernel.runtime.a", "landed": True}},
        legacy_edge_budget=40,
    )
    write_fragment(
        frags_dir,
        "p6-core",
        {"entropic_substrate.b": {"target": "entropic_substrate.core.geometry.b", "landed": True}},
        legacy_edge_budget=16,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = gen_shims.main(
        [
            str(frags_dir),
            str(mapping_path),
            str(out_dir),
            "--print-budget",
            "--sequence",
            "p0,p5-kernel,p6-core,p7-register-gate",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "16"


def test_print_budget_uses_only_the_present_prefix_GREEN_control(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Control: with only p5-kernel present (p6-core not yet landed), the
    budget reported is p5-kernel's, not some later phase's -- proves the
    ranking genuinely depends on what's on disk, not just sequence position."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.a"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {"entropic_substrate.a": {"target": "entropic_substrate.kernel.runtime.a", "landed": True}},
        legacy_edge_budget=40,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = gen_shims.main(
        [
            str(frags_dir),
            str(mapping_path),
            str(out_dir),
            "--print-budget",
            "--sequence",
            "p0,p5-kernel,p6-core",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "40"


def test_print_budget_fails_loud_on_out_of_sequence_fragment(tmp_path: Path) -> None:
    """RED-first: a fragment present on disk whose phase id is absent from
    --sequence has undefined rank -- must fail loud, not silently ignore it
    or silently place it last/first."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.a"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p99-unknown-phase",
        {"entropic_substrate.a": {"target": "entropic_substrate.kernel.runtime.a", "landed": True}},
        legacy_edge_budget=5,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = gen_shims.main(
        [
            str(frags_dir),
            str(mapping_path),
            str(out_dir),
            "--print-budget",
            "--sequence",
            "p0,p5-kernel,p6-core",
        ]
    )
    assert rc == 1


def test_print_budget_via_sequence_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.a"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {"entropic_substrate.a": {"target": "entropic_substrate.kernel.runtime.a", "landed": True}},
        legacy_edge_budget=7,
    )
    seq_file = tmp_path / "sprint-manifest.json"
    seq_file.write_text(json.dumps({"sequence": ["p0", "p5-kernel", "p6-core"]}))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = gen_shims.main(
        [
            str(frags_dir),
            str(mapping_path),
            str(out_dir),
            "--print-budget",
            "--sequence-file",
            str(seq_file),
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "7"


# ---------------------------------------------------------------------------
# Loud failure on unreadable / malformed input (never a silent skip).
# ---------------------------------------------------------------------------


def test_missing_fragments_dir_fails_loud(tmp_path: Path) -> None:
    write_mapping(tmp_path, ["entropic_substrate.a"])
    with pytest.raises(gen_shims.ManifestError):
        gen_shims.load_fragments(tmp_path / "does_not_exist")


def test_missing_mapping_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(gen_shims.ManifestError):
        gen_shims.load_mapping(tmp_path / "nope.json")


def test_main_exits_1_on_missing_mapping(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {"entropic_substrate.a": {"target": "entropic_substrate.kernel.runtime.a", "landed": True}},
    )
    rc = gen_shims.main([str(frags_dir), str(tmp_path / "nope.json"), str(tmp_path / "out")])
    assert rc == 1
    assert "FATAL" in capsys.readouterr().err
