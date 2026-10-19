# for-1609-review-packet-split: bring scripts/assemble_review_packet.py back under 1000 lines

DRAFT 2026-10-19 by the conductor (Bug Squashing wave `bugsquash-2026-10-19`; issue FOR-1609, thermo-nuclear
review presumptive blocker B6 at epoch `0e55d9d`, adversarially verified). Anchors measured at main `04ad7790`
with `wc -l`, `grep -n`, `sed -n`; the implementer re-verifies them in recon. The Linear ticket is authoritative.

## Branch

| field                | value                                                                       |
| -------------------- | --------------------------------------------------------------------------- |
| branch               | `fix/for-1609-review-packet-split`                                          |
| parent               | the wave base branch (main `04ad7790` + the briefs commit)                  |
| cascade `phase.mode` | `normal` (a pure split of one tool script; no door, no reader-tuple member) |
| publish              | none in the lane (`publishMode: defer`, CERTIFIED); the zip publishes       |

## Ruling (operator R2, 2026-10-19): scripts carve-out

The standing never-edit list forbids `scripts/*` to lanes because `scripts/run_exit_gate.py` and
`scripts/exit_gate.py` are gate instruments. For THIS lane only, `scripts/assemble_review_packet.py` and
any NEW sibling module(s) it is split into under `scripts/` are allowed; every other file under `scripts/`
stays forbidden (`run_exit_gate.py`, `exit_gate.py`, `entrypoints.env`, `run_ladder.sh`, `hooks/`, ...).

## Why (measured at `04ad7790`)

- `scripts/assemble_review_packet.py` is 1038 lines (948 at pre-refactor `8c7a7f7`; pushed past the 1000-line threshold by the refactor itself, 9.5 percent, with no stated reason).
- Structure: `ArmSpec` (`:66`), `RungSpec` (`:86`), the E1 and E1.5 preregistration TEXT blocks (`:137-253` and `:253-402`, markdown headings inside string constants), `RUNGS` table, then helpers `md5` (`:516`), `scrub_record` (`:521`), `rekey_attestation` (`:543`), `assert_attestation_present` (`:579`), `_DocstringStripper` (`:609`), `_is_shim_source` (`:644`), `build_reader` (`:693`), `banned_scan` (`:762`), `select_captures` (`:769`), `assert_blinded_parity` (`:804`), `main` (`:862-1036`, argparse + the per-packet salt at `:933`, the hook-stability normalization comment at `:964-968`, manifest at `:980`).
- Tests: `tests/test_review_packet.py` (29 tests) loads the script BY FILE PATH with `importlib.util.spec_from_file_location("assemble_review_packet", <repo>/scripts/assemble_review_packet.py)` (`:31-38`) and reads `_MOD.RUNGS["e1"]` etc.; `tests/test_cli_surface.py:96` pins two frozen literals that must stay in `scripts/assemble_review_packet.py` verbatim (`entropic_substrate.ladder.capture.reader.d1_`, `entropic_substrate.ladder.capture.records.d1_`). `scripts/` is NOT a package (no `__init__.py`).

## Scope

1. **C1, tests RED at the parent**: a size pin in `tests/test_review_packet.py` that `scripts/assemble_review_packet.py` is under 1000 lines (RED at 1038) and that every public name the existing tests read from `_MOD` (`RUNGS`, `scrub_record`, `rekey_attestation`, `assert_attestation_present`, `build_reader`, `banned_scan`, `select_captures`, `assert_blinded_parity`, the specs) is still reachable through the file-path loader after the split (GREEN at both, a non-regression guard for M1).
2. **C2, the split**: the implementer picks the boundary from the measured structure (the obvious seam is the two preregistration text blocks + `RUNGS` into a sibling data module, or the reader-assembly helpers `_DocstringStripper`/`_is_shim_source`/`build_reader` into a sibling), keeping `scripts/assemble_review_packet.py` as the CLI entrypoint with `main()`. Because the tests load by file path and `scripts/` is not a package, the entry script must import its siblings in a way that works both as `python scripts/assemble_review_packet.py` and under the file-path loader (e.g. a `sys.path` entry for its own directory, or a package directory under `scripts/` with explicit relative loading); the chosen mechanism is recorded in the module docstring with the measurement. Packet-assembly ORDER and the salt/normalize/hash sequence in `main()` stay byte-identical in behaviour (the 2026-08-18 hook-stability measurement at `:964-968` is kept verbatim).
3. **C3**: re-measure `wc -l` of the entry script and every sibling and record the numbers in the commit message (ticket re-validation checklist).

## Guard teeth by mutation

| id | mutation                                                | expected RED                                  |
| -- | ------------------------------------------------------- | --------------------------------------------- |
| M1 | revert the split (inline the sibling back)              | the <1000 size pin RED                        |
| M2 | drop one moved helper's re-export from the entry script | the reachability pin RED                      |
| M3 | remove one frozen literal from the entry script         | `tests/test_cli_surface.py` path-site pin RED |

## Allowed files

`scripts/assemble_review_packet.py`, new sibling module(s) under `scripts/` created by this lane,
`tests/test_review_packet.py`, `tests/test_cli_surface.py` ONLY if a new sibling must be added to its
frozen-literal path table (prefer not: keep the literals in the entry script). Never: any other file under
`scripts/`, `docs/*` (post-zip sweep), `justfile`, `.pre-commit-config.yaml`.

## Acceptance

C1 RED at the parent, GREEN at the tip; M1-M3 RED one at a time and restored; all 29 existing tests in
`tests/test_review_packet.py` and `tests/test_cli_surface.py` green unchanged in meaning; `just check` exit 0
and the exit-gate runner green under the suite lock; `python scripts/assemble_review_packet.py --help` exits 0
from the repo root; line counts recorded; diff confined to the Allowed files.
