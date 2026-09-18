"""Pytest for gen_shims.py. House rule 5 (TDD non-negotiable, guard-teeth prove
both arms): every RED-first case below is paired with a GREEN control proving
the same check does NOT fire on the thing it must let through.

RETRACTION HISTORY (2026-09-17/18, same session):
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
3. Sprint-preflight (2026-09-18) found FOUR blocking defects in that design,
   ruled by the conductor:
   (a) every quarantined module had to be NAMED, explicitly, by some fragment
       -- p3 would have had to enumerate the entire unmoved tree, defeating
       "fragments only ever ADD". Corrected: a module absent from every
       fragment is now the DERIVED legacy default (landed:false, target
       entropic_substrate.__legacy.<old suffix>), no shim without it.
   (b/c) the generator used to write a package `__init__.py` shim (flagged
       `package: true`), racing the __path__ hazard the docstring itself
       warned about. Corrected: package keys (a dotted prefix of another
       mapping key, plus the root `entropic_substrate`) are NEVER shimmable
       -- a fragment naming one now fails loud. `test_init_shim_emitted_as_
       package_file` and `test_unflagged_package_prefix_fails_loud` below
       tested the RETRACTED behaviour and are replaced, not left to
       coincidentally error, by `test_naming_package_key_in_fragment_fails_
       loud` / `test_naming_root_key_in_fragment_fails_loud` /
       `test_no_package_init_ever_written_even_when_children_named`.
   (d) `--verify` used to `rglob` every `*.py` under `out_dir`, so a real
       production module living in the same tree (or a package `__init__.py`
       restored verbatim after generate() ran) was scored as a malformed
       shim. Corrected: generate() now writes an emitted-files manifest and
       `--verify` checks exactly those files.
4. Two items surfaced in that same report, ruled in scope by the conductor
   the same session:
   (e) ENTRYPOINT SHIMS. A plain shim is inert under `python -m <old>` --
       the target's own `if __name__ == "__main__":` guard never fires
       because the shim's rebind imports it under its OWN name, not
       `__main__`. mapping.json gains an optional per-module
       `"entrypoint": true` (confirmed 2026-09-18 against the 7 real `-m`
       targets, each either `__main__.py` or carrying its own guard); those
       modules get a longer shim form (`if __name__ == "__main__": import
       runpy; runpy.run_module(...); raise SystemExit(0)`, THEN the
       standard rebind) that `--verify` also accepts.
   (f) STALE-VERIFY. `--verify` now re-resolves fragments + mapping.json
       and fails loud, naming each, if the manifest's {path: target} set
       disagrees with the fresh resolution (extra / missing / re-pointed)
       -- so flipping `landed:true` in a fragment without regenerating can
       no longer verify green.

Run standalone (no repo pytest config picked up):
    uv run --project <repo> --no-sync pytest tools/test_gen_shims.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if not TOOLS.is_dir():
    # Staging-copy layout (this session's deliverables/ dir): gen_shims.py sits
    # alongside this file, not under a sibling tools/ one level up. Promoting
    # this file back to tests/test_gen_shims.py in the repo restores the
    # parents[1]/"tools" branch above without any further edit.
    TOOLS = Path(__file__).resolve().parent
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


def resolve(fragments_dir: Path, mapping_path: Path) -> dict:
    """Convenience: run the full merge -> derive-legacy-defaults pipeline the
    way main() does, for tests that want the fully resolved module set."""
    fragments = gen_shims.load_fragments(fragments_dir)
    mapping = gen_shims.load_mapping(mapping_path)
    merged = gen_shims.merge_fragments(fragments, mapping)
    return gen_shims.resolve_all_modules(merged, mapping)


# ---------------------------------------------------------------------------
# Shim-form AST checks, exercised directly against verify_shim_source (the
# per-file checker). `verify_dir`'s directory-rglob wrapper was deleted as
# part of DEFECT 8's fix (see RETRACTION HISTORY item 3(d)); these tests no
# longer need a directory or a written file to check the AST predicate.
# ---------------------------------------------------------------------------


def test_star_import_shim_is_rejected_by_verify() -> None:
    ok, reason = gen_shims.verify_shim_source("from entropic_substrate.__legacy.thing import *\n")
    assert ok is False
    assert reason


def test_exact_form_shim_passes_verify_GREEN_control() -> None:
    source = gen_shims.shim_source(
        "entropic_substrate.analysis.charts",
        "entropic_substrate.__legacy.analysis.charts",
        "fragments/",
    )
    ok, _ = gen_shims.verify_shim_source(source)
    assert ok is True


def test_verify_rejects_importlib_string_shim() -> None:
    ok, _ = gen_shims.verify_shim_source(
        "import importlib, sys\n"
        "_target = importlib.import_module('entropic_substrate.__legacy.thing')\n"
        "sys.modules[__name__] = _target\n"
    )
    assert ok is False


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
    merged = resolve(frags_dir, mapping_path)
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
    merged = resolve(frags_dir, mapping_path)
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
    merged = resolve(frags_dir, mapping_path)
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
# RED-first (DEFECT 1): a module absent from EVERY fragment is not an
# omission to flag -- it is the derived legacy default. "fragments only ever
# ADD" means p3's own fragment may list nothing at all.
# ---------------------------------------------------------------------------


def test_unnamed_module_gets_implicit_legacy_shim(tmp_path: Path) -> None:
    """p3 lists nothing for `old_thing` (its fragment only carries the
    phase's own budget) -- the module must still get a shim, pointed at the
    DERIVED legacy target, with no fragment naming it at all."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.old_thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {}, legacy_edge_budget=12)
    out_dir = tmp_path / "out"
    merged = resolve(frags_dir, mapping_path)
    written = gen_shims.generate(merged, str(frags_dir), out_dir)
    out = out_dir / "entropic_substrate" / "old_thing.py"
    assert written == [out]
    assert "entropic_substrate.__legacy.old_thing" in out.read_text()


def test_named_module_is_not_overridden_by_the_derived_default_GREEN_control(
    tmp_path: Path,
) -> None:
    """Control for the case above: a module a fragment DOES name keeps that
    fragment's explicit target, the derived default never shadows it."""
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
    merged = resolve(frags_dir, mapping_path)
    assert (
        merged["entropic_substrate.old_thing"]["target"]
        == "entropic_substrate.ladder.run.new_thing"
    )


# ---------------------------------------------------------------------------
# RED-first (DEFECTS 2/3/9): the generator NEVER writes a package shim. A
# package key is any mapping key that is a dotted prefix of another mapping
# key, plus the root `entropic_substrate` itself. Naming one in a fragment
# fails loud, naming the key.
# ---------------------------------------------------------------------------


def test_naming_package_key_in_fragment_fails_loud(tmp_path: Path) -> None:
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.analysis", "entropic_substrate.analysis.charts"]
    )
    mapping = gen_shims.load_mapping(mapping_path)
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.analysis": {
                "target": "entropic_substrate.__legacy.analysis",
                "landed": False,
            }
        },
    )
    fragments = gen_shims.load_fragments(frags_dir)
    with pytest.raises(gen_shims.ManifestError, match=r"entropic_substrate\.analysis"):
        gen_shims.merge_fragments(fragments, mapping)


def test_naming_root_key_in_fragment_fails_loud(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate", "entropic_substrate.thing"])
    mapping = gen_shims.load_mapping(mapping_path)
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate": {
                "target": "entropic_substrate.__legacy",
                "landed": False,
            }
        },
    )
    fragments = gen_shims.load_fragments(frags_dir)
    with pytest.raises(gen_shims.ManifestError, match="entropic_substrate"):
        gen_shims.merge_fragments(fragments, mapping)


def test_naming_a_leaf_module_still_works_GREEN_control(tmp_path: Path) -> None:
    """Control for the two cases above: naming an ordinary LEAF module (not
    a dotted prefix of anything else) must not trip the package guard."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.leaf_thing"])
    mapping = gen_shims.load_mapping(mapping_path)
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.leaf_thing": {
                "target": "entropic_substrate.kernel.runtime.leaf_thing",
                "landed": True,
            }
        },
    )
    fragments = gen_shims.load_fragments(frags_dir)
    merged = gen_shims.merge_fragments(fragments, mapping)
    assert set(merged) == {"entropic_substrate.leaf_thing"}


def test_no_package_init_ever_written_even_when_children_named(tmp_path: Path) -> None:
    """The package itself (`entropic_substrate.analysis`) is never named --
    only its leaf child is. No `__init__.py` may appear anywhere under
    out_dir; only the leaf shim is written."""
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.analysis", "entropic_substrate.analysis.charts"]
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.analysis.charts": {
                "target": "entropic_substrate.__legacy.analysis.charts",
                "landed": False,
            }
        },
    )
    out_dir = tmp_path / "out"
    merged = resolve(frags_dir, mapping_path)
    written = gen_shims.generate(merged, str(frags_dir), out_dir)
    init_paths = list(out_dir.rglob("__init__.py"))
    assert init_paths == []
    assert out_dir / "entropic_substrate" / "analysis" / "charts.py" in written


def test_real_init_left_in_out_dir_untouched_byte_for_byte(tmp_path: Path) -> None:
    """Simulates p3 restoring the ORIGINAL package __init__.py (real
    re-export content) after generate() has already written the sibling leaf
    shims into the same directory. generate() must never write to, or even
    look at, that path -- it is not a key in `modules` at all."""
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.analysis", "entropic_substrate.analysis.charts"]
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.analysis.charts": {
                "target": "entropic_substrate.__legacy.analysis.charts",
                "landed": False,
            }
        },
    )
    out_dir = tmp_path / "out"
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir)

    real_init = out_dir / "entropic_substrate" / "analysis" / "__init__.py"
    real_bytes = b'"""Real re-export content, restored verbatim by p3."""\n\nSENTINEL = 1\n'
    real_init.write_bytes(real_bytes)

    # A second generate() pass (e.g. a later phase re-running the generator)
    # must not perturb the restored file.
    gen_shims.generate(merged, str(frags_dir), out_dir)
    assert real_init.read_bytes() == real_bytes


# ---------------------------------------------------------------------------
# Package __init__ handling: OLD tests for the RETRACTED "package: true"
# design lived here (test_init_shim_emitted_as_package_file,
# test_unflagged_package_prefix_fails_loud). Removed per RETRACTION HISTORY
# item 3(b/c) -- superseded by the package-guard tests above, which assert
# the corrected (inverted) behaviour: naming a package fails loud, and no
# __init__.py is ever emitted, flagged or not.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RED-first (DEFECT 8): --verify checks exactly the emitted-files manifest
# generate() writes, never an rglob of out_dir -- out_dir may also hold real
# production modules that must never be scored as shims.
# ---------------------------------------------------------------------------


def test_verify_manifest_ignores_real_production_module_in_tree(tmp_path: Path) -> None:
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
    manifest_path = tmp_path / "emitted.json"
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir, manifest_path=manifest_path)

    # A real production module living in the SAME tree, not a shim, not in
    # the manifest -- rglob-based verify would score this and fail.
    real_module = out_dir / "entropic_substrate" / "not_a_shim.py"
    real_module.parent.mkdir(parents=True, exist_ok=True)
    real_module.write_text("def real_function():\n    return 42\n")

    rc = gen_shims.main(
        [
            str(frags_dir),
            str(mapping_path),
            str(out_dir),
            "--verify",
            "--manifest-out",
            str(manifest_path),
        ]
    )
    assert rc == 0


def test_verify_manifest_still_catches_a_broken_emitted_shim_GREEN_control(
    tmp_path: Path,
) -> None:
    """Control for the case above: --verify must still FAIL when a file the
    manifest actually lists has been hand-edited into a broken shim -- the
    manifest-scoping in the case above must not have gone blind entirely."""
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
    manifest_path = tmp_path / "emitted.json"
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir, manifest_path=manifest_path)

    (out_dir / "entropic_substrate" / "old_thing.py").write_text(
        "from entropic_substrate.__legacy.old_thing import *\n"
    )

    rc = gen_shims.main(
        [
            str(frags_dir),
            str(mapping_path),
            str(out_dir),
            "--verify",
            "--manifest-out",
            str(manifest_path),
        ]
    )
    assert rc == 1


def test_generate_default_manifest_path_matches_pinned_invocation(tmp_path: Path) -> None:
    """The module docstring pins `--manifest-out
    .claude/workflows/landed-set/.emitted.json` as the canonical invocation,
    i.e. a dotfile living INSIDE the fragments dir itself. The default (no
    --manifest-out) must resolve to that same place, and a second run must
    not choke trying to parse its own manifest as a fragment (pathlib's
    `Path.glob("*.json")` matches dotfiles, unlike a shell glob -- MEASURED
    2026-09-18)."""
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
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)])
    assert rc == 0
    default_manifest = frags_dir / ".emitted.json"
    assert default_manifest.is_file()

    # Re-running (generate again, then verify) must not treat the manifest
    # dotfile sitting in frags_dir as a bogus fragment.
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)])
    assert rc == 0
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--verify"])
    assert rc == 0


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
    """GREEN-control-shaped sweep: after a normal generate() run (called
    directly, no manifest_path given), nothing under out_dir looks like an
    ignore-list artifact (.cfg/.ini), only *.py shims -- a direct
    behavioural check, not just an API-surface check."""
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
    merged = resolve(frags_dir, mapping_path)
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


# ---------------------------------------------------------------------------
# RED-first: ENTRYPOINT SHIMS. mapping.json's optional per-module
# "entrypoint": true gets the longer if-guard + rebind form; --verify
# accepts EXACTLY the two forms (standard 3-statement, or this 4-statement
# one) and rejects anything else.
# ---------------------------------------------------------------------------


def write_mapping_with_flags(tmp_path: Path, entries: dict[str, dict]) -> Path:
    """Like write_mapping, but lets a caller set arbitrary per-module
    mapping.json metadata (e.g. `entrypoint: true`) instead of the fixed
    dest/layer/action triple."""
    mapping = {
        "layers": ["kernel", "core", "ladder", "__legacy", "composition"],
        "modules": {
            m: {"dest": "entropic_substrate.somewhere", "layer": "ladder", "action": "move", **meta}
            for m, meta in entries.items()
        },
    }
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps(mapping))
    return p


def test_generate_emits_entrypoint_form_for_flagged_module(tmp_path: Path) -> None:
    mapping_path = write_mapping_with_flags(
        tmp_path, {"entropic_substrate.worker_command": {"entrypoint": True}}
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.worker_command": {
                "target": "entropic_substrate.kernel.runtime.worker_command",
                "landed": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    merged = resolve(frags_dir, mapping_path)
    assert merged["entropic_substrate.worker_command"]["entrypoint"] is True
    gen_shims.generate(merged, str(frags_dir), out_dir)
    text = (out_dir / "entropic_substrate" / "worker_command.py").read_text()
    assert 'if __name__ == "__main__":' in text
    assert "import runpy" in text
    assert (
        'runpy.run_module("entropic_substrate.kernel.runtime.worker_command", '
        'run_name="__main__", alter_sys=True)' in text
    )
    assert "raise SystemExit(0)" in text
    assert "import entropic_substrate.kernel.runtime.worker_command as _target" in text
    ok, reason = gen_shims.verify_shim_source(text)
    assert ok is True, reason


def test_generate_emits_standard_form_for_non_entrypoint_module_GREEN_control(
    tmp_path: Path,
) -> None:
    """Control for the case above: a module with no `entrypoint` flag (the
    overwhelming majority of mapping.json) is completely unaffected -- the
    old plain 3-statement shim, byte for byte."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.ordinary_thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p5-kernel",
        {
            "entropic_substrate.ordinary_thing": {
                "target": "entropic_substrate.kernel.runtime.ordinary_thing",
                "landed": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    merged = resolve(frags_dir, mapping_path)
    assert merged["entropic_substrate.ordinary_thing"]["entrypoint"] is False
    gen_shims.generate(merged, str(frags_dir), out_dir)
    text = (out_dir / "entropic_substrate" / "ordinary_thing.py").read_text()
    assert 'if __name__ == "__main__":' not in text
    assert "import runpy" not in text
    ok, _ = gen_shims.verify_shim_source(text)
    assert ok is True


def test_verify_accepts_entrypoint_form(tmp_path: Path) -> None:
    source = gen_shims.shim_source(
        "entropic_substrate.cli",
        "entropic_substrate.kernel.runtime.cli",
        "fragments/",
        is_entrypoint=True,
    )
    ok, reason = gen_shims.verify_shim_source(source)
    assert ok is True, reason


def test_verify_rejects_entrypoint_form_missing_system_exit(tmp_path: Path) -> None:
    """A malformed near-miss of the entrypoint form: the if-guard body still
    has exactly 3 statements (so the generic statement-count checks are
    satisfied either way -- this isolates the THIRD statement's own shape),
    but the third one is `pass` instead of `raise SystemExit(0)`. Without
    that raise, falling through to the rebind after `-m` would double-bind
    `sys.modules[__name__]` in the same process. Must still be rejected."""
    source = (
        'if __name__ == "__main__":\n'
        "    import runpy\n\n"
        '    runpy.run_module("entropic_substrate.kernel.runtime.cli", run_name="__main__", alter_sys=True)\n'
        "    pass\n"
        "import entropic_substrate.kernel.runtime.cli as _target\n"
        "import sys\n"
        "sys.modules[__name__] = _target\n"
    )
    ok, reason = gen_shims.verify_shim_source(source)
    assert ok is False
    assert "SystemExit" in reason


def test_verify_rejects_entrypoint_form_wrong_alter_sys(tmp_path: Path) -> None:
    source = (
        'if __name__ == "__main__":\n'
        "    import runpy\n\n"
        '    runpy.run_module("entropic_substrate.kernel.runtime.cli", run_name="__main__", alter_sys=False)\n'
        "    raise SystemExit(0)\n"
        "import entropic_substrate.kernel.runtime.cli as _target\n"
        "import sys\n"
        "sys.modules[__name__] = _target\n"
    )
    ok, _ = gen_shims.verify_shim_source(source)
    assert ok is False


def test_verify_rejects_five_statement_shim(tmp_path: Path) -> None:
    """Neither of the two accepted shapes -- a stray 5th statement, however
    innocuous, is rejected rather than pattern-matched loosely."""
    source = (
        "import entropic_substrate.kernel.runtime.thing as _target\n"
        "import sys\n"
        "import os\n"
        "sys.modules[__name__] = _target\n"
        "os.environ.setdefault('X', '1')\n"
    )
    ok, reason = gen_shims.verify_shim_source(source)
    assert ok is False
    assert "5" in reason


def test_main_dash_m_runs_entrypoint_shim_target(tmp_path: Path) -> None:
    """The whole point of the entrypoint form: `python -m <old dotted>`
    must actually execute the target's own `if __name__ == "__main__":`
    body. Proven end to end with a real subprocess against a throwaway
    target module whose main() writes a marker file."""
    target_dir = tmp_path / "target_src"
    target_dir.mkdir()
    marker_path = tmp_path / "entrypoint_marker.txt"
    (target_dir / "real_target_module.py").write_text(
        "def main():\n"
        "    from pathlib import Path\n"
        f"    Path({str(marker_path)!r}).write_text('ran')\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    shim_dir = tmp_path / "shim_src"
    shim_dir.mkdir()
    (shim_dir / "old_entry_shim.py").write_text(
        gen_shims.shim_source(
            "old_entry_shim", "real_target_module", "test-fragment", is_entrypoint=True
        )
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(shim_dir), str(target_dir), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-m", "old_entry_shim"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert marker_path.exists()
    assert marker_path.read_text() == "ran"


def test_main_dash_m_on_standard_shim_is_unchanged_GREEN_control(tmp_path: Path) -> None:
    """Control for the case above: `-m` on an ORDINARY (non-entrypoint)
    shim behaves exactly as before -- it does NOT run the target's main()
    (the exact defect this feature was added to fix, now proven confined to
    entrypoint-flagged modules only, not silently applied everywhere)."""
    target_dir = tmp_path / "target_src2"
    target_dir.mkdir()
    marker_path = tmp_path / "standard_marker.txt"
    (target_dir / "real_target_module2.py").write_text(
        "def main():\n"
        "    from pathlib import Path\n"
        f"    Path({str(marker_path)!r}).write_text('ran')\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    shim_dir = tmp_path / "shim_src2"
    shim_dir.mkdir()
    (shim_dir / "old_plain_shim.py").write_text(
        gen_shims.shim_source("old_plain_shim", "real_target_module2", "test-fragment")
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(shim_dir), str(target_dir), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-m", "old_plain_shim"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert not marker_path.exists()


# ---------------------------------------------------------------------------
# RED-first: STALE-VERIFY. --verify re-resolves fragments+mapping fresh and
# fails loud, naming each, if the manifest's {path: target} set disagrees
# with that fresh resolution (extra / missing / re-pointed).
# ---------------------------------------------------------------------------


def test_find_manifest_drift_detects_repointed_target() -> None:
    manifest_shims = [
        {"path": "entropic_substrate/thing.py", "target": "entropic_substrate.__legacy.thing"}
    ]
    expected = {"entropic_substrate/thing.py": "entropic_substrate.kernel.runtime.thing"}
    problems = gen_shims.find_manifest_drift(manifest_shims, expected)
    assert len(problems) == 1
    assert "STALE" in problems[0]
    assert "entropic_substrate/thing.py" in problems[0]


def test_find_manifest_drift_detects_extra_and_missing() -> None:
    manifest_shims = [
        {
            "path": "entropic_substrate/retired_thing.py",
            "target": "entropic_substrate.ladder.run.retired_thing",
        }
    ]
    expected = {"entropic_substrate/new_thing.py": "entropic_substrate.__legacy.new_thing"}
    problems = gen_shims.find_manifest_drift(manifest_shims, expected)
    assert len(problems) == 2
    joined = " ".join(problems)
    assert "retired_thing" in joined
    assert "new_thing" in joined


def test_find_manifest_drift_empty_on_matching_sets_GREEN_control() -> None:
    manifest_shims = [
        {"path": "entropic_substrate/thing.py", "target": "entropic_substrate.kernel.runtime.thing"}
    ]
    expected = {"entropic_substrate/thing.py": "entropic_substrate.kernel.runtime.thing"}
    assert gen_shims.find_manifest_drift(manifest_shims, expected) == []


def test_stale_verify_fails_when_landed_flips_without_regenerate(tmp_path: Path) -> None:
    """The exact scenario the sprint-preflight named: a phase flips
    `landed:true` (re-pointing the module off __legacy) in its OWN fragment
    file, but forgets to re-run generate(). The manifest and the shim on
    disk both still show the OLD target -- --verify must fail loud, not
    pass because the shim file itself still happens to be well-formed."""
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
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)])
    assert rc == 0

    # The phase flips landed:true in its OWN fragment file -- without
    # regenerating.
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.old_thing": {
                "target": "entropic_substrate.ladder.run.old_thing",
                "landed": True,
            }
        },
    )
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--verify"])
    assert rc == 1


def test_stale_verify_passes_after_regenerate_GREEN_control(tmp_path: Path) -> None:
    """Control for the case above: the SAME landed-flip, but followed by a
    regenerate -- --verify must pass, proving the drift check isn't simply
    always failing after any fragment edit."""
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
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)])
    assert rc == 0

    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.old_thing": {
                "target": "entropic_substrate.ladder.run.old_thing",
                "landed": True,
            }
        },
    )
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)])
    assert rc == 0
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--verify"])
    assert rc == 0
