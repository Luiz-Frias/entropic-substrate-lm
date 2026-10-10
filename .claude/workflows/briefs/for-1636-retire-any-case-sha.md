# for-1636-retire-any-case-sha: retire `require_sha256_hex_any_case` (zero callers after FOR-1626)

Authoritative brief for one `td-phase-lifecycle.cascade` run, written 2026-10-10 by the conductor from the operator's ruling (2026-10-10: run FOR-1636 as the fourth branch of the post-merge stack, then merge the stack in one sweep). Linear: FOR-1636. Premises measured at `ee2c13a` (the reworded, re-certified head of PR #335): `src` holds no reference to `require_sha256_hex_any_case` except its definition (`kernel/identity/d1_primitives.py:39-45`); outside `src` only `tests/test_sha256_caller_audit.py` exercises it (`test_any_case_accepts_and_rejects` and the live-scan definition control) and `tests/test_strict_json_adoption_analyze_review_file.py:15` names it in prose; `import string` in `d1_primitives.py` stays because `require_commit` uses `string.hexdigits` too.

## Branch

| field                | value                                                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| branch               | `mythwave/for-1636-retire-any-case-sha`                                                                                                                            |
| parent               | `mythwave/for-1626-sha-repoint` (PR #335, head `ee2c13a` after the DN-8DB6295-MSG reword); the operator merges #333 → #334 → #335 → this PR `--no-ff` in one sweep |
| cascade `phase.mode` | `arch`, always explicit (a public kernel name is removed, operator-ruled)                                                                                          |
| wave                 | serial, main checkout, no lanes                                                                                                                                    |
| `gates.liveRequired` | `false` (no MLflow adapter in the touch set)                                                                                                                       |
| gates                | `{"offline":"just check","exit":"just pc-ci","liveRequired":false}`                                                                                                |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                                                                             |
| publish              | the cascade's publish stage: `gt submit --draft --no-interactive` once, stacked on the parent                                                                      |

The branch EXISTS at launch: the conductor created it with `git checkout -b mythwave/for-1636-retire-any-case-sha ee2c13a` (start point passed explicitly) and committed this brief as its first commit; `gt track --parent mythwave/for-1626-sha-repoint` records the parent. Do not recreate it; verify `git merge-base --is-ancestor ee2c13a HEAD` and implement on top.

## Contract surface owned

| contract                                                                                                                                                                         | kind                                                      | frozen_after |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ------------ |
| `kernel.identity.d1_primitives.require_sha256_hex_any_case` is REMOVED; the name no longer resolves from the module (same shape as the FOR-1615 `require_sha256` retirement pin) | public kernel surface removal (operator-ruled 2026-10-10) | true         |
| `docs/plans/measurements/p14-sha-callers.txt` is FROZEN as history (dated header note; header counts and rows untouched)                                                         | measurement record                                        | true         |

## Contracts consumed

- `kernel.codec.strict_json.require_lowercase_sha256`: the one boundary digest predicate every former caller now uses (FOR-1615 rename, FOR-1626 repoint).
- `tests/test_sha256_caller_audit.py`: `test_old_name_no_longer_resolves` (the pin style to mirror), the call-counter controls (`test_call_counter_counts_a_known_call`, `test_live_scan_aggregates_a_known_caller_end_to_end`, both on synthetic trees, kept as they are), `_parse_audit` (header/row consistency, kept).

## Sacred invariants

| invariant                              | verbatim                                                             | tripwire                                                                                                                                                                                                                    | guard-teeth recipe    |
| -------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| House rule 5                           | "TDD is non-negotiable, and guard-teeth prove both arms."            | the new and restated pins are shown RED on the parent tree (where the name still resolves) before the deletion                                                                                                              | mutations M1-M3 below |
| Boundary lowercase contract (FOR-1615) | "every untrusted-boundary site requires lowercase"                   | `tests/test_strict_json_adoption_run.py::test_every_boundary_sha_site_refuses_a_mixed_case_digest_with_its_own_error` green unmodified                                                                                      | n/a (no site changes) |
| Reader identity (R-PIN)                | the 51 reader-tuple files are never edited outside the identity rail | `git diff --name-only ee2c13a...HEAD` has an empty intersection with the 51 files listed in `.claude/workflows/briefs/p14-t-verdict-vocab.md` under *Identity members* (`kernel/identity/d1_primitives.py` is NOT a member) | proven at p0          |
| `require_commit` behaviour             | unchanged; it keeps `string.hexdigits`                               | `tests/test_d1_primitives.py` and every test that constructs a commit identity stay green                                                                                                                                   | mutation M4           |
| Layer contract                         | `kernel < core < ladder < historical < composition`                  | `lint-imports` green                                                                                                                                                                                                        | proven at p0 and p13  |

## Identity members

None. `kernel/identity/d1_primitives.py`, the test file and the measurement file are outside the 51 reader-tuple files (measured). Editing a member is a HALT (DECISION), never a hand-off.

## Scope

One commit.

1. `src/entropic_substrate/kernel/identity/d1_primitives.py`: delete the `require_sha256_hex_any_case` definition (`:39-45`) and nothing else; `import string` stays (`require_commit` uses it; a reviewer confirms with `grep -n 'string\.'`).
2. `tests/test_sha256_caller_audit.py`: (a) replace `test_any_case_accepts_and_rejects` with `test_any_case_name_no_longer_resolves`, mirroring `test_old_name_no_longer_resolves` (`importlib.import_module(_MODULE_DOTTED)`, `assert not hasattr(module, _NEW_NAME)`); (b) in `test_caller_audit_is_committed_and_covers_the_live_tree`, the definition control flips to the end state: the live scan of `src` finds NO call site AND NO definition (`live_counts == {}` and the definition flag is `False`; rename the flag to `definition_found` where it reads better); the header/row consistency checks on the frozen audit file stay; (c) the two synthetic-tree counter controls stay unchanged (they build their own definition module); (d) the module docstring records the FOR-1636 end state (the SHA sequence FOR-1615 → FOR-1626 → FOR-1636 is closed; the name no longer resolves; the audit file is frozen history). RED-first against the parent tree: (a) and (b) both RED there.
3. `docs/plans/measurements/p14-sha-callers.txt`: add a dated `#` header note that FOR-1636 retired the helper and the rows are history of the migration; counts and rows untouched (`files: 33 / rows: 57 / boundary: 57 / in-process: 0`).

## ACCEPTANCE

- `grep -rn require_sha256_hex_any_case src` is empty; `python -c "import entropic_substrate.kernel.identity.d1_primitives as m; assert not hasattr(m, 'require_sha256_hex_any_case')"` passes; `grep -n 'string\.' src/entropic_substrate/kernel/identity/d1_primitives.py` still shows `require_commit`'s use.
- `tests/test_sha256_caller_audit.py` green at the tip with the new pin and the restated end state; both shown RED on the parent tree (`ee2c13a`), counts quoted.
- `tests/test_strict_json_adoption_run.py`, `tests/test_strict_json_adoption_analyze_review_file.py`, `tests/test_d1_primitives.py`, `tests/test_d1_initialization.py`, `tests/test_d1_registered_design.py` green unmodified.
- Guard teeth by mutation, both arms quoted (below).
- remap_check at the tip over the re-derived mapping: inversions <= 1, package cycles <= 4; mechanical counts at or under `da4d35c` (hook files 2, `def _mapping(` 23, separators 64, lowercase-SHA files 29, `type: ignore[arg-type]` 9, files > 1000 lines 2).
- `git diff --name-only ee2c13a...HEAD` minus the conductor's brief commit is a subset of *Allowed files*; the identity tripwire is empty.
- `just check` green at the tip under the host lock; exit gate certified by the runner JSON at HEAD; draft PR stacked on `mythwave/for-1626-sha-repoint` with a filled body (Linear ref FOR-1636, the ruling, RED/GREEN counts, mutation arms, remap stdout, the six counts).

## Non-goals

- Does NOT touch `require_lowercase_sha256`, any boundary site, `require_commit`, or the `_SHA256_LENGTH` constant if another function uses it (check; if it becomes unused, deleting it is in scope and said so in the commit body).
- Does NOT rewrite the audit rows or delete the measurement file.
- Does NOT edit `tests/test_strict_json_adoption_analyze_review_file.py` (its prose mention is historical and still true).
- Does NOT launch pods, spend, build containers or start an MLflow server.

## Allowed files

- `src/entropic_substrate/kernel/identity/d1_primitives.py`
- `tests/test_sha256_caller_audit.py`
- `docs/plans/measurements/p14-sha-callers.txt`
- `.secrets.baseline` (mechanical companion, regenerated by the prek hook only)

Never edit: `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/regime_rows.py`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `tools/mapping.json`, the manifest, the plan, briefs, `pyproject.toml`, `uv.lock`.

## Already ruled

- Operator 2026-10-10: FOR-1636 runs now as the fourth branch of the post-merge stack; the public-name removal is thereby ruled.
- DN-8DB6295-MSG (operator 2026-10-10, option a): the parent head is the reworded, re-certified `ee2c13a`; never rewrite published history without a ruling.
- FOR-1626 rulings R1-R6 stand; this phase closes the SHA sequence they left open (R4).
- Message lint before every commit (`uv run gitlint --msg-filename`); the commit-msg hook now also lints root commits (#333).

## Guard teeth by mutation

| id | mutation                                                           | expected RED                                                                                                                                                                           | expected GREEN after revert |
| -- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| M1 | re-add the `require_sha256_hex_any_case` definition                | `test_any_case_name_no_longer_resolves` RED; the live-scan end-state assertion (definition absent) RED                                                                                 | GREEN                       |
| M2 | make `_scan_live_call_counts` return `({}, False)` unconditionally | `test_live_scan_aggregates_a_known_caller_end_to_end` RED (its synthetic tree has a definition and a caller) while the end-state test stays GREEN, proving the control is load-bearing | GREEN                       |
| M3 | change the frozen audit header's `rows:` by one                    | `test_caller_audit_is_committed_and_covers_the_live_tree` header-consistency arm RED                                                                                                   | GREEN                       |
| M4 | drop `import string` from `d1_primitives.py`                       | `require_commit` NameError: `tests/test_d1_primitives.py` (or any commit-identity test) RED                                                                                            | GREEN                       |
