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


def plant(out_dir: Path, *dotted: str) -> list[Path]:
    """Creates a real (non-shim) module file at each dotted path under
    ``out_dir`` -- the on-disk state generate() now requires for every row it
    shims (the module must physically exist at its target; see
    find_missing_sources). Never writes a package ``__init__.py``: the guard
    keys on the module file alone, and several tests pin that no
    ``__init__.py`` ever appears under out_dir."""
    planted: list[Path] = []
    for name in dotted:
        f = out_dir.joinpath(*name.split(".")).with_suffix(".py")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f'"""planted fixture source for {name}"""\n\nPLANTED = True\n')
        planted.append(f)
    return planted


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
    plant(
        out_dir,
        "entropic_substrate.kernel.runtime.shared_thing",
        "entropic_substrate.core.geometry.shared_thing",
    )
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
    plant(out_dir, "entropic_substrate.__legacy.old_thing")
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
    plant(out_dir, "entropic_substrate.ladder.run.new_thing")
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
    plant(out_dir, "entropic_substrate.ladder.run.new_thing")
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
    plant(out_dir, "entropic_substrate.__legacy.confused")
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
    plant(out_dir, "entropic_substrate.ladder.run.already_promoted")
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
    plant(out_dir, "entropic_substrate.__legacy.old_thing")
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
    plant(out_dir, "entropic_substrate.__legacy.analysis.charts")
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
    plant(out_dir, "entropic_substrate.__legacy.analysis.charts")
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
    plant(out_dir, "entropic_substrate.__legacy.old_thing")
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
    plant(out_dir, "entropic_substrate.__legacy.old_thing")
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
    plant(out_dir, "entropic_substrate.__legacy.old_thing")
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
    plant(out_dir, "entropic_substrate.kernel.runtime.thing")
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
    ignore-list artifact (.cfg/.ini) -- only *.py shims and their *.pyi
    stub siblings (typed shim pair, RULED 2026-09-20; amended by this
    phase, which turned the pair on by construction) -- a direct
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
    plant(out_dir, "entropic_substrate.kernel.runtime.thing")
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir)
    non_py_non_pyi = [
        p for p in out_dir.rglob("*") if p.is_file() and p.suffix not in (".py", ".pyi")
    ]
    assert non_py_non_pyi == []
    assert (out_dir / "entropic_substrate" / "thing.pyi").is_file()


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
    plant(out_dir, "entropic_substrate.kernel.runtime.a", "entropic_substrate.core.geometry.b")
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
    plant(out_dir, "entropic_substrate.kernel.runtime.a")
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
    plant(out_dir, "entropic_substrate.kernel.runtime.a")
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
    plant(out_dir, "entropic_substrate.kernel.runtime.a")
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
    plant(out_dir, "entropic_substrate.kernel.runtime.worker_command")
    merged = resolve(frags_dir, mapping_path)
    assert merged["entropic_substrate.worker_command"]["entrypoint"] is True
    gen_shims.generate(merged, str(frags_dir), out_dir)
    text = (out_dir / "entropic_substrate" / "worker_command.py").read_text()
    assert 'if __name__ == "__main__":' in text
    assert "import runpy" in text
    # generate() now canonicalizes shim text through the same ruff pipeline
    # as the stub (constraint 3, design gate 2026-09-20: an uncanonicalized
    # shim is rewritten mid-gate by `ruff check . --fix`), so a long
    # `runpy.run_module(...)` call may be WRAPPED across lines by `ruff
    # format` -- collapse whitespace before comparing so this assertion
    # checks the call's substance, not its exact line-wrap.
    collapsed = " ".join(text.split())
    assert (
        'runpy.run_module( "entropic_substrate.kernel.runtime.worker_command",'
        ' run_name="__main__", alter_sys=True )'
        in collapsed
        or 'runpy.run_module("entropic_substrate.kernel.runtime.worker_command",'
        ' run_name="__main__", alter_sys=True)'
        in collapsed
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
    plant(out_dir, "entropic_substrate.kernel.runtime.ordinary_thing")
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
    plant(
        out_dir, "entropic_substrate.__legacy.old_thing", "entropic_substrate.ladder.run.old_thing"
    )
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
    plant(
        out_dir, "entropic_substrate.__legacy.old_thing", "entropic_substrate.ladder.run.old_thing"
    )
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


# ---------------------------------------------------------------------------
# Source-existence guard (FINDING 2026-09-20, sprint p2 -> p3 gap). Shim
# emission is keyed off mapping.json rows alone: resolve_all_modules()
# manufactures a `__legacy` default for EVERY non-package key, and generate()
# used to write a shim for each one without ever asking whether the module it
# points at exists. Measured on mythwave/p2-d1-closure: the row
# `entropic_substrate.analysis.d1_reader_source_closure` (deleted outright at
# 8cf5aae, ruling E-1 extended) would have become a shim importing a missing
# `__legacy` module at p3. The mapping-coverage test deliberately tolerates the
# stale row (swept at p13), so the generator itself must refuse. Both arms:
# the RED arm shows the refusal names the row and writes nothing; the GREEN
# control shows a planted target source is enough.
# ---------------------------------------------------------------------------


def test_generate_refuses_row_whose_module_has_no_source_on_disk(tmp_path: Path) -> None:
    """RED arm: nothing at the target dotted path, nothing at the old one ->
    ManifestError naming the row, and NO shim file is written (all-or-nothing:
    the refusal happens before the first write)."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.analysis.gone_closure"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    merged = resolve(frags_dir, mapping_path)
    with pytest.raises(gen_shims.ManifestError) as excinfo:
        gen_shims.generate(merged, str(frags_dir), out_dir)
    msg = str(excinfo.value)
    assert "entropic_substrate.analysis.gone_closure" in msg
    assert "entropic_substrate.__legacy.analysis.gone_closure" in msg
    assert not (out_dir / "entropic_substrate" / "analysis" / "gone_closure.py").exists()


def test_generate_refusal_is_all_or_nothing(tmp_path: Path) -> None:
    """One stale row among healthy ones must block the WHOLE emission: a
    partial shim set on disk would be worse than none (half the tree resolves,
    the other half raises ImportError somewhere far from the cause)."""
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.aaa_present", "entropic_substrate.zzz_gone"]
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.__legacy.aaa_present")
    merged = resolve(frags_dir, mapping_path)
    with pytest.raises(gen_shims.ManifestError, match="zzz_gone"):
        gen_shims.generate(merged, str(frags_dir), out_dir)
    assert not (out_dir / "entropic_substrate" / "aaa_present.py").exists()


def test_generate_emits_when_target_source_exists_GREEN_control(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.analysis.kept_module"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.__legacy.analysis.kept_module")
    merged = resolve(frags_dir, mapping_path)
    written = gen_shims.generate(merged, str(frags_dir), out_dir)
    assert written == [out_dir / "entropic_substrate" / "analysis" / "kept_module.py"]


def test_find_missing_sources_accepts_unmoved_real_source_at_old_path(tmp_path: Path) -> None:
    """Pre-move state (what --preflight sees before p3's git mv): the module
    still lives at its OLD path and no `__legacy` target exists yet. That is
    not a stale row; the pre-flight must stay quiet about it."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.still_here"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.still_here")
    merged = resolve(frags_dir, mapping_path)
    assert gen_shims.find_missing_sources(merged, out_dir) == []


def test_find_missing_sources_does_not_count_a_shim_at_old_path(tmp_path: Path) -> None:
    """A shim sitting at the old path is not source: if its target is gone,
    the row is stale even though a file exists at the old dotted name. This is
    the arm that keeps the guard load-bearing AFTER p3, when every old path
    holds a shim."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.shimmed_only"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    old_path = out_dir / "entropic_substrate" / "shimmed_only.py"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(
        gen_shims.shim_source(
            "entropic_substrate.shimmed_only", "entropic_substrate.__legacy.shimmed_only", "t"
        )
    )
    merged = resolve(frags_dir, mapping_path)
    missing = gen_shims.find_missing_sources(merged, out_dir)
    assert len(missing) == 1
    assert "entropic_substrate.shimmed_only" in missing[0]


def test_find_missing_sources_accepts_package_init_as_target(tmp_path: Path) -> None:
    """A leaf key whose module is a package directory (no mapped children, so
    compute_package_keys does not classify it) resolves through its
    `__init__.py`, exactly as the import system would."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.leaf_pkg"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    init = out_dir / "entropic_substrate" / "__legacy" / "leaf_pkg" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text('"""real package"""\n')
    merged = resolve(frags_dir, mapping_path)
    assert gen_shims.find_missing_sources(merged, out_dir) == []


def test_generate_refuses_to_overwrite_real_source_at_old_path(tmp_path: Path) -> None:
    """The sibling hazard the same walk exposes: running the generator BEFORE
    the git mv would replace every real module with a three-line shim. Real
    (non-shim) source at the old path is a refusal, and the file is untouched."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.not_yet_moved"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    (real,) = plant(out_dir, "entropic_substrate.not_yet_moved")
    plant(out_dir, "entropic_substrate.__legacy.not_yet_moved")  # target present too
    before = real.read_bytes()
    merged = resolve(frags_dir, mapping_path)
    with pytest.raises(gen_shims.ManifestError, match="not_yet_moved"):
        gen_shims.generate(merged, str(frags_dir), out_dir)
    assert real.read_bytes() == before


def test_generate_overwrites_its_own_earlier_shim_GREEN_control(tmp_path: Path) -> None:
    """Regeneration is the normal path (every later phase re-runs the
    generator): an exact-form shim at the old path is replaced, not refused."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.re_gen"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.__legacy.re_gen")
    merged = resolve(frags_dir, mapping_path)
    first = gen_shims.generate(merged, str(frags_dir), out_dir)
    second = gen_shims.generate(merged, str(frags_dir), out_dir)
    assert first == second == [out_dir / "entropic_substrate" / "re_gen.py"]


def test_preflight_names_stale_row_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--preflight` is the pre-move check p3 runs first: no fragment exists
    yet (the landed-set directory may not even exist), every row takes the
    derived `__legacy` default, and a row with no source anywhere is reported
    by name. Nothing is written."""
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.present_one", "entropic_substrate.analysis.deleted_one"]
    )
    frags_dir = tmp_path / "landed-set"  # deliberately never created
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.present_one")
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--preflight"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "entropic_substrate.analysis.deleted_one" in captured.err
    assert "present_one" not in captured.err
    assert list(out_dir.rglob("*.py")) == [out_dir / "entropic_substrate" / "present_one.py"]


def test_preflight_clean_tree_exits_0_GREEN_control(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.present_one"])
    frags_dir = tmp_path / "landed-set"
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.present_one")
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--preflight"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "1 row(s)" in captured.out
    assert list(out_dir.rglob("*.py")) == [out_dir / "entropic_substrate" / "present_one.py"]


def test_preflight_with_only_manifest_dotfile_present_is_treated_as_no_fragments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Root-cause fix (CodeRabbit 2026-09-21): a fragments directory holding
    ONLY a stray `.emitted.json` (no real phase fragment) used to fail
    `any(glob("*.json"))`'s dotfile-blind check in main()'s own probe, so
    --preflight handed it to load_fragments() instead of taking the pre-p3
    derived-default path -- load_fragments() already skips dotfiles itself,
    so it found zero real fragments and raised "no *.json fragments found",
    a FATAL where the derived-default path was the correct read. Both call
    sites now share `_fragment_json_files`."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.present_one"])
    frags_dir = tmp_path / "landed-set"
    frags_dir.mkdir(parents=True)
    (frags_dir / ".emitted.json").write_text("{}")  # stray manifest, not a fragment
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.present_one")
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--preflight"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "preflight ok: 1 row(s)" in captured.out
    assert "from 0 fragment(s)" in captured.out


def test_preflight_summary_excludes_retired_and_pending_from_checked_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Root-cause fix (CodeRabbit 2026-09-21): find_missing_sources and
    find_unmoved_sources both skip retired/pending rows internally -- they
    are never looked at against on-disk source -- so the preflight
    summary's "row(s) resolve to source on disk" / "at the target" counts
    must not silently credit them as checked-and-landed."""
    mapping_path = write_mapping(
        tmp_path,
        [
            "entropic_substrate.alpha",
            "entropic_substrate.beta_retired",
            "entropic_substrate.gamma_pending",
        ],
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.beta_retired": {
                "target": "entropic_substrate.__legacy.beta_retired",
                "landed": False,
                "retired": True,
            },
            "entropic_substrate.gamma_pending": {
                "target": "entropic_substrate.__legacy.gamma_pending",
                "landed": False,
                "pending": "p9-later",
            },
        },
    )
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.alpha")  # only the CHECKED row has real source
    rc = gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--preflight"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "preflight ok: 1 row(s)" in captured.out
    assert "2 retired/pending row(s) not checked" in captured.out


def test_verify_fails_when_target_source_vanishes_after_generate(tmp_path: Path) -> None:
    """`--verify` gains the same teeth: a module deleted AFTER its shim was
    emitted leaves an exact-form shim pointing at nothing, which the
    manifest-drift and form checks alone cannot see."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.old_thing"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(frags_dir, "p3-quarantine", {})
    out_dir = tmp_path / "out"
    (target,) = plant(out_dir, "entropic_substrate.__legacy.old_thing")
    assert gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)]) == 0
    assert gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--verify"]) == 0
    target.unlink()
    assert gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--verify"]) == 1


# ---------------------------------------------------------------------------
# `pending` -- the fourth fragment state the same measurement forced. The
# real-tree --preflight at the p2 tip listed THREE rows without source: the
# deleted closure, plus `d1_training_step_records` / `d1_calibration_records`,
# which mapping.json carries today but p5 only creates. Without a positive
# declaration the guard would (correctly) block p3's whole emission on them.
# ---------------------------------------------------------------------------


def test_pending_row_gets_no_shim_and_passes_the_guard(tmp_path: Path) -> None:
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.here_now", "entropic_substrate.d1_future_records"]
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.d1_future_records": {
                "target": "entropic_substrate.__legacy.d1_future_records",
                "landed": False,
                "pending": "p5-kernel",
            }
        },
    )
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.__legacy.here_now")
    # main() so the emitted-files manifest is written where --verify reads it.
    assert gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir)]) == 0
    assert (out_dir / "entropic_substrate" / "here_now.py").is_file()
    assert not (out_dir / "entropic_substrate" / "d1_future_records.py").exists()
    assert gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--verify"]) == 0


def test_pending_row_whose_module_exists_is_refused(tmp_path: Path) -> None:
    """The flag cannot linger: once the creating phase lands the module, a
    stale `pending` declaration is a loud refusal, not a silently missing shim."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.d1_future_records"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.d1_future_records": {
                "target": "entropic_substrate.__legacy.d1_future_records",
                "landed": False,
                "pending": "p5-kernel",
            }
        },
    )
    out_dir = tmp_path / "out"
    plant(out_dir, "entropic_substrate.__legacy.d1_future_records")
    merged = resolve(frags_dir, mapping_path)
    with pytest.raises(gen_shims.ManifestError, match="pending"):
        gen_shims.generate(merged, str(frags_dir), out_dir)
    assert gen_shims.main([str(frags_dir), str(mapping_path), str(out_dir), "--preflight"]) == 1


def test_pending_and_retired_together_is_refused(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.contradiction"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.contradiction": {
                "target": "entropic_substrate.__legacy.contradiction",
                "landed": False,
                "pending": "p5-kernel",
                "retired": True,
            }
        },
    )
    out_dir = tmp_path / "out"
    merged = resolve(frags_dir, mapping_path)
    with pytest.raises(gen_shims.ManifestError, match="both pending"):
        gen_shims.generate(merged, str(frags_dir), out_dir)


# ---------------------------------------------------------------------------
# RED-first: TYPED SHIM PAIR (RULED 2026-09-20, operator, option (b) of
# design-gate wf_7f27b290-e61). Every non-retired, non-pending row gets a
# `.pyi` stub alongside its `.py` shim, DERIVED from the target's own AST --
# never a star import. `--verify` re-derives and compares byte-for-byte,
# never crossing the stub check with the shim exact-form check.
# ---------------------------------------------------------------------------


def test_generate_emits_a_pyi_stub_sibling_for_every_shim(tmp_path: Path) -> None:
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
    target_file = out_dir / "entropic_substrate" / "__legacy" / "old_thing.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("PUBLIC_THING = 1\n\n\ndef helper():\n    return PUBLIC_THING\n")
    manifest_path = tmp_path / "emitted.json"
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir, manifest_path=manifest_path)

    stub = out_dir / "entropic_substrate" / "old_thing.pyi"
    assert stub.is_file()
    text = stub.read_text()
    assert "from entropic_substrate.__legacy.old_thing import PUBLIC_THING as PUBLIC_THING" in text
    assert "from entropic_substrate.__legacy.old_thing import helper as helper" in text
    assert "import *" not in text  # no star import anywhere

    manifest = json.loads(manifest_path.read_text())
    assert manifest["stubs"] == [
        {
            "path": "entropic_substrate/old_thing.pyi",
            "target": "entropic_substrate.__legacy.old_thing",
        }
    ]


def test_derived_stub_includes_underscore_names_with_pyright_ignore(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.private_haver"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.private_haver": {
                "target": "entropic_substrate.__legacy.private_haver",
                "landed": False,
            }
        },
    )
    out_dir = tmp_path / "out"
    target_file = out_dir / "entropic_substrate" / "__legacy" / "private_haver.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("_PRIVATE_LIMIT = 5\n\n\ndef _helper():\n    return _PRIVATE_LIMIT\n")
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir)
    text = (out_dir / "entropic_substrate" / "private_haver.pyi").read_text()
    # ruff format may wrap a long `from X import name as name  # comment`
    # onto several lines (parenthesized), so this checks substrings survive
    # the canonicalization pipeline rather than pinning one exact line shape.
    assert "_PRIVATE_LIMIT as _PRIVATE_LIMIT" in text
    assert "_helper as _helper" in text
    assert text.count("pyright: ignore[reportPrivateUsage]") == 2


def test_derived_stub_restates_a_literal_dunder_all_GREEN_control(tmp_path: Path) -> None:
    """Control: a target defining a literal __all__ gets that list restated
    verbatim in the stub, alongside every module-level binding (nothing an
    __all__-omitted name hides is invisible to the derivation -- only the
    literal restatement of __all__ itself is conditional)."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.curated"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.curated": {
                "target": "entropic_substrate.__legacy.curated",
                "landed": False,
            }
        },
    )
    out_dir = tmp_path / "out"
    target_file = out_dir / "entropic_substrate" / "__legacy" / "curated.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text('__all__ = ["Exposed"]\n\nExposed = 1\nHidden = 2\n')
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir)
    text = (out_dir / "entropic_substrate" / "curated.pyi").read_text()
    assert "__all__" in text and "Exposed" in text
    # Hidden is still re-exported (a star stub would have hidden it -- this
    # derivation never hides a binding the target actually has).
    assert "Hidden as Hidden" in text


def test_derived_stub_recurses_into_type_checking_block(tmp_path: Path) -> None:
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.tc_user"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.tc_user": {
                "target": "entropic_substrate.__legacy.tc_user",
                "landed": False,
            }
        },
    )
    out_dir = tmp_path / "out"
    target_file = out_dir / "entropic_substrate" / "__legacy" / "tc_user.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pathlib import Path as PathAlias\n"
    )
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir)
    text = (out_dir / "entropic_substrate" / "tc_user.pyi").read_text()
    assert "PathAlias as PathAlias" in text


def test_derived_stub_includes_pep695_type_alias_statements(tmp_path: Path) -> None:
    """RED-FIRST (found post-mass-emission, constraint 5 firing after the
    fact): `_bound_names_in_stmts` had no branch for `ast.TypeAlias` (the
    PEP 695 `type X = ...` statement, Python 3.12+), so a target module
    that defines its public type with `type FrozenJson = ...` -- the real,
    measured shape of `FrozenJson`/`JsonValue`/`InterfaceMode`/
    `D1TrainerRequest` and 6 other names across 9 real quarantined modules
    -- got NO re-export line in its stub. A consumer's OWN derived stub
    then imported that name from the target and basedpyright reported
    `reportAttributeAccessIssue` / `reportUnsupportedDunderAll` (83 errors
    measured against the real tree, all one root cause)."""
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.type_alias_user"])
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.type_alias_user": {
                "target": "entropic_substrate.__legacy.type_alias_user",
                "landed": False,
            }
        },
    )
    out_dir = tmp_path / "out"
    target_file = out_dir / "entropic_substrate" / "__legacy" / "type_alias_user.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(
        "type FrozenJsonScalar = str | int | float | bool | None\n"
        'type FrozenJson = FrozenJsonScalar | tuple["FrozenJson", ...]\n'
    )
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir)
    text = (out_dir / "entropic_substrate" / "type_alias_user.pyi").read_text()
    assert "FrozenJsonScalar as FrozenJsonScalar" in text
    assert "FrozenJson as FrozenJson" in text


def test_derive_stub_body_pins_single_all_binding(tmp_path: Path) -> None:
    """RED-FIRST (house rule 5 / SLICE-PROBE FINDING 2026-09-20): commit
    1e63ec1 ("fix(refactor): stop double-binding __all__ in derived stubs")
    filtered `__all__` from the generic name list when a literal `__all__`
    is derived, preventing BOTH a `from X import __all__` line AND a literal
    `__all__ = [...]` restatement in the same stub. The prior test suite's
    assertion `"__all__" in text` passes identically for both the buggy
    (double) and fixed (single) output. This test pins the exact invariant
    via AST:

    RED mutation (shows the bug):
        sed -i '650s/names = [n for n in names if n != "__all__"]/names = [n for n in names]  # BUGGY/' tools/gen_shims.py
    produces:
        from entropic_substrate.__legacy.curated import __all__ as __all__
        from entropic_substrate.__legacy.curated import Exposed as Exposed
        __all__ = ["Exposed"]  # The duplicate: two bindings, not one

    Tested two arms:
    - literal `__all__`: exactly one ast.Assign binding, zero ImportFrom aliases
    - computed `__all__`: exactly one ImportFrom alias binding, zero Assigns
    """
    import ast

    def count_all_bindings_in_stub(stub_body: str) -> tuple[int, int]:
        """Parse stub body AST and count:
        - ast.Assign statements that bind to __all__
        - ast.ImportFrom aliases named __all__ (or aliased to __all__)
        """
        tree = ast.parse(stub_body)
        assign_count = 0
        import_count = 0

        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        assign_count += 1
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if alias.name == "__all__" and (
                        alias.asname == "__all__" or alias.asname is None
                    ):
                        import_count += 1

        return assign_count, import_count

    # ARM 1: literal __all__ -- should have exactly one ast.Assign, zero ImportFroms
    target = "entropic_substrate.__legacy.curated"
    source_literal = '__all__ = ["Exposed", "Hidden"]\n\nExposed = 1\nHidden = 2\n'

    body = gen_shims.derive_stub_body(target, source_literal)
    assign_count, import_count = count_all_bindings_in_stub(body)

    assert assign_count == 1, f"literal __all__: expected 1 assignment, got {assign_count}"
    assert import_count == 0, f"literal __all__: expected 0 from-imports, got {import_count}"

    # ARM 2: computed __all__ (non-literal) -- should have zero Assigns, exactly one ImportFrom
    source_computed = '__all__ = sorted(["X", "Y"])\n\nX = 1\nY = 2\n'

    body2 = gen_shims.derive_stub_body(target, source_computed)
    assign_count2, import_count2 = count_all_bindings_in_stub(body2)

    assert assign_count2 == 0, (
        f"computed __all__: expected 0 assignments (not literal), got {assign_count2}"
    )
    assert import_count2 == 1, f"computed __all__: expected 1 from-import, got {import_count2}"


def test_verify_fails_loud_when_a_stub_is_deleted(tmp_path: Path) -> None:
    """Guard tooth: delete one emitted .pyi (leave its .py shim) -> --verify
    fails naming the missing stub."""
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
    plant(out_dir, "entropic_substrate.__legacy.old_thing")
    manifest_path = tmp_path / "emitted.json"
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir, manifest_path=manifest_path)
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
    assert rc == 0  # GREEN control arm: passes before the mutation

    (out_dir / "entropic_substrate" / "old_thing.pyi").unlink()
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


def test_verify_fails_loud_when_a_stub_is_repointed_at_a_different_target(tmp_path: Path) -> None:
    """Guard tooth: point one .pyi stub at a different target than its .py
    sibling -> --verify fails loud naming the stub (as manifest drift, since
    the manifest's own recorded target no longer matches the freshly
    resolved one)."""
    mapping_path = write_mapping(
        tmp_path, ["entropic_substrate.old_thing", "entropic_substrate.other_thing"]
    )
    frags_dir = tmp_path / "landed-set"
    write_fragment(
        frags_dir,
        "p3-quarantine",
        {
            "entropic_substrate.old_thing": {
                "target": "entropic_substrate.__legacy.old_thing",
                "landed": False,
            },
            "entropic_substrate.other_thing": {
                "target": "entropic_substrate.__legacy.other_thing",
                "landed": False,
            },
        },
    )
    out_dir = tmp_path / "out"
    plant(
        out_dir, "entropic_substrate.__legacy.old_thing", "entropic_substrate.__legacy.other_thing"
    )
    manifest_path = tmp_path / "emitted.json"
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir, manifest_path=manifest_path)

    manifest = json.loads(manifest_path.read_text())
    for stub in manifest["stubs"]:
        if stub["path"] == "entropic_substrate/old_thing.pyi":
            stub["target"] = "entropic_substrate.__legacy.other_thing"
    manifest_path.write_text(json.dumps(manifest))

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


def test_verify_fails_loud_when_a_derived_reexport_line_is_removed(tmp_path: Path) -> None:
    """Guard tooth: remove one derived re-export line from a stub whose name
    a consumer would import -> --verify fails loud (content no longer
    matches a fresh re-derivation)."""
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
    target_file = out_dir / "entropic_substrate" / "__legacy" / "old_thing.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("FIRST_NAME = 1\nSECOND_NAME = 2\n")
    manifest_path = tmp_path / "emitted.json"
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir, manifest_path=manifest_path)

    stub_path = out_dir / "entropic_substrate" / "old_thing.pyi"
    lines = [ln for ln in stub_path.read_text().splitlines() if "SECOND_NAME" not in ln]
    stub_path.write_text("\n".join(lines) + "\n")

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


def test_verify_passes_on_a_freshly_generated_pair_GREEN_control(tmp_path: Path) -> None:
    """Control for the three mutation arms above: an untouched, freshly
    generated shim+stub pair verifies clean."""
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
    target_file = out_dir / "entropic_substrate" / "__legacy" / "old_thing.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("FIRST_NAME = 1\nSECOND_NAME = 2\n")
    manifest_path = tmp_path / "emitted.json"
    merged = resolve(frags_dir, mapping_path)
    gen_shims.generate(merged, str(frags_dir), out_dir, manifest_path=manifest_path)
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


def test_shim_source_puts_import_sys_before_target_import(tmp_path: Path) -> None:
    """RULED 2026-09-20 (wf_43ef5138-4bc, 'SECOND'): `import sys` sits AHEAD
    of the static target import in the emitted shim."""
    source = gen_shims.shim_source(
        "entropic_substrate.old_thing", "entropic_substrate.__legacy.old_thing", "fragments/"
    )
    lines = [ln for ln in source.splitlines() if not ln.startswith("#")]
    assert lines[0] == "import sys"
    assert lines[1] == "import entropic_substrate.__legacy.old_thing as _target"
    assert lines[2] == "sys.modules[__name__] = _target"
    ok, reason = gen_shims.verify_shim_source(source)
    assert ok is True, reason


def test_verify_rejects_old_order_target_import_before_sys(tmp_path: Path) -> None:
    """RED-first mirror of the reordering: the OLD order (target import
    first, then `import sys`) must now be REJECTED -- proves the order
    check is load-bearing, not merely permissive of both."""
    source = (
        "import entropic_substrate.__legacy.old_thing as _target\n"
        "import sys\n"
        "sys.modules[__name__] = _target\n"
    )
    ok, reason = gen_shims.verify_shim_source(source)
    assert ok is False
    assert "sys" in reason


REPO_PYPROJECT = TOOLS.parent / "pyproject.toml"


def _ruff_diff(paths: list[Path]) -> str:
    """Runs this repo's OWN `ruff check --fix --diff` (isort included, per
    [tool.ruff.lint] select) then `ruff format --diff`, exactly the two
    passes `just lint` / the pre-commit hook apply, against real files on
    disk. Non-empty output means the hook would REWRITE the file mid-gate --
    the exact 'hook fixed point' failure mode constraint 5 (design gate
    2026-09-20) warns about for the stub pipeline, checked here for the
    SHIM `.py` side of the typed pair.

    Only returncode 0 (no changes) or 1 (`--diff` reports changes it would
    make -- MEASURED: both `ruff check --fix --diff` and `ruff format
    --diff` use 1 for this, never a raised error) are treated as a real
    answer; any other status is a fail-loud assertion naming the command and
    stderr, not a silently-empty diff read as "no changes" (root-cause fix,
    CodeRabbit 2026-09-21 -- a crashed/misconfigured ruff invocation must
    not be indistinguishable from a genuine fixed point)."""
    out = ""
    for args in (
        ["ruff", "check", "--fix", "--diff", "--config", str(REPO_PYPROJECT)],
        ["ruff", "format", "--diff", "--config", str(REPO_PYPROJECT)],
    ):
        cmd = [*args, *[str(p) for p in paths]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode in (0, 1), (
            f"{cmd} exited {r.returncode} (expected 0 or 1): stderr={r.stderr!r}"
        )
        out += r.stdout
    return out


def test_shim_py_is_a_ruff_fixed_point_against_the_real_repo_config(tmp_path: Path) -> None:
    """RED-FIRST (constraint 3, design gate 2026-09-20): `shim_source()`'s
    raw text is written to disk with NO ruff canonicalization pass, unlike
    the `.pyi` stub (`canonicalize_stub_texts`). Under THIS repo's own
    ruff config (`known-first-party = ["entropic_substrate"]`, `select`
    includes `I`), isort splits `import sys` (stdlib) and
    `import entropic_substrate.__legacy.x as _target` (first-party) into
    two groups and ruff format inserts a blank line after the import
    block -- `ruff check . --fix` (house rule: `just lint`, and the
    pre-commit hook `just check`/`just pc-ci` both run) REWRITES every
    emitted shim mid-gate, which is exactly the ACCEPTANCE bullet 'git
    status --porcelain is EMPTY after just check and after just pc-ci'
    failing. MEASURED (this test, before the fix): non-empty diff on a
    freshly emitted shim under the real repo config."""
    fragments_dir = tmp_path / "fragments"
    mapping_path = write_mapping(tmp_path, ["entropic_substrate.old_thing"])
    write_fragment(fragments_dir, "p3-quarantine", {})
    out_dir = tmp_path / "src"
    plant(out_dir, "entropic_substrate.__legacy.old_thing")
    resolved = resolve(fragments_dir, mapping_path)
    written = gen_shims.generate(resolved, str(fragments_dir), out_dir)
    shim_paths = [p for p in written if p.suffix == ".py"]
    assert shim_paths
    diff = _ruff_diff(shim_paths)
    assert diff == "", f"ruff would rewrite the emitted shim mid-gate:\n{diff}"


def test_canonicalize_reports_ruff_stdout_and_stderr_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root-cause fix (CodeRabbit 2026-09-21): `CalledProcessError`'s own
    `str()` carries only argv + returncode, never the captured streams this
    call already asks for (`capture_output=True`) -- the actual reason a
    canonicalization pass failed (a malformed derived text, a ruff config
    error) was invisible in the raised `ManifestError`."""

    def _fake_run(cmd: list[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, cmd, output="STDOUT-MARKER", stderr="STDERR-MARKER")

    monkeypatch.setattr(gen_shims.subprocess, "run", _fake_run)
    with pytest.raises(gen_shims.ManifestError, match="STDOUT-MARKER") as exc_info:
        gen_shims.canonicalize_python_texts({"x.py": "x = 1\n"}, tmp_path)
    assert "STDERR-MARKER" in str(exc_info.value)


def test_canonicalize_file_not_found_stays_concise_GREEN_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the fix above: a missing `ruff` binary (`FileNotFoundError`,
    no captured streams to report) keeps its original, concise message
    shape -- only the `CalledProcessError` arm gained stdout/stderr."""

    def _fake_run(cmd: list[str], **_kwargs: object) -> None:
        raise FileNotFoundError("ruff")

    monkeypatch.setattr(gen_shims.subprocess, "run", _fake_run)
    with pytest.raises(gen_shims.ManifestError, match="ruff stub-canonicalization pipeline failed"):
        gen_shims.canonicalize_python_texts({"x.py": "x = 1\n"}, tmp_path)
