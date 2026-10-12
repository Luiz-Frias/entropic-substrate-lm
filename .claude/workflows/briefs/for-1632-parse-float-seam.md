# for-1632-parse-float-seam: amend the strict-JSON door with a `parse_float` seam and adopt it in the RunPod billing parser

Authoritative brief for one `td-phase-lifecycle.cascade` run, written 2026-10-08 by the conductor from operator ruling DN-1632-SEAM (2026-10-08: "parse_float seam > holdout"). Plan: `docs/plans/p14-consolidation.plan.md` DECISION-NEEDED register, row DN-1632-SEAM. Linear: FOR-1632 (the ruling is recorded on the ticket). Premises were measured at the merged p14 trunk `da4d35c` and hold at this branch's parent `fix/for-1633-commit-msg-exclude` @ `ee9df9a` (PR #333), whose two commits touch only the pre-commit config, one test and two docs files.

## Branch

| field                | value                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| branch               | `mythwave/for-1632-parse-float-seam`                                                                                     |
| parent               | `fix/for-1633-commit-msg-exclude` (PR #333, head `ee9df9a`); the operator merges the whole stack in one sweep at the end |
| cascade `phase.mode` | `arch`, always explicit (a door amendment)                                                                               |
| wave                 | serial, main checkout, no lanes                                                                                          |
| `gates.liveRequired` | `false` (no MLflow adapter in the touch set)                                                                             |
| gates                | `{"offline":"just check","exit":"just pc-ci","liveRequired":false}`                                                      |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                                   |
| publish              | the cascade's publish stage: `gt submit --draft --no-interactive` once, stacked on the parent                            |

The branch EXISTS at launch: the conductor created it off the parent with this brief (and the FOR-1626 brief) as its first commit. Do not recreate it; implement on top.

## Contract surface owned

| contract                                                                                                                          | kind                                                                                         | frozen_after           |
| --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------- |
| `kernel.codec.strict_json.strict_loads` gains an ADDITIVE keyword-only seam `parse_float: Callable[[str], object] \| None = None` | door amendment (the door was frozen after the p14-r1 rail; this amendment is operator-ruled) | true, after this phase |

Semantics of the seam, ruled here:

- `parse_float is None`: byte-for-byte today's behaviour (`float(token)`, refused through `fail` when not finite).
- `parse_float` given: the float token goes to `parse_float(token)`; a `Decimal` result is checked with `Decimal.is_finite()`, a `float` result with `math.isfinite`, and a non-finite result of either kind is refused through `fail` with the existing `non-finite float literal` message; any other result type is returned unchanged. The three non-finite CONSTANTS (`NaN`, `Infinity`, `-Infinity`) are still refused through `fail` regardless of `parse_float` (they never reach it: `parse_constant` handles them).
- No double decode, no float-to-Decimal reconstruction anywhere (ruling W4-21 as amended by DN-1632-SEAM).

## Contracts consumed

- `kernel.codec.strict_json` as landed by the p14-r1 rail and adopted by every `ladder/run` decoder except the billing parser (`tests/test_strict_json_adoption_run.py`, `_PENDING`).
- `ladder/run/closure/d1_runpod_billing_parser.py` `_fail(detail, error=None) -> NoReturn` is already assignable to `Callable[[str], NoReturn]`; the parser's own exception type `RunpodBillingSchemaError` stays the only type it raises.

## Sacred invariants

| invariant               | verbatim                                                                                                           | tripwire                                                                                                                                                               | guard-teeth recipe                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| House rule 5            | "TDD is non-negotiable, and guard-teeth prove both arms."                                                          | every behaviour-pinning test is committed RED before the change that turns it GREEN                                                                                    | the mutations below, both arms quoted in the PR body |
| House rule 3            | "Never fix flakiness by raising a timeout or a retry budget."                                                      | the diff adds no timeout, deadline, retry count or backoff constant                                                                                                    | reviewer greps the diff                              |
| Evidence-path exactness | the pinned billed amount `0.0017170833889394999` (19 significant digits) must survive decode as an exact `Decimal` | `tests/test_d1_runpod_billing_parser.py` round-trips it byte-for-byte                                                                                                  | mutation M1                                          |
| Exception surface       | each decoder raises its OWN exception type through `fail`                                                          | `tests/test_strict_json_adoption_run.py` section 2 stays green; the billing parser raises only `RunpodBillingSchemaError`                                              | mutation M3                                          |
| Reader identity (R-PIN) | the 51 reader-tuple files are never edited outside the identity rail                                               | `git diff --name-only ee9df9a...HEAD` has an empty intersection with the 51 files listed in `.claude/workflows/briefs/p14-t-verdict-vocab.md` under *Identity members* | proven at p0                                         |
| Layer contract          | `kernel < core < ladder < historical < composition`                                                                | `lint-imports` green at the tip                                                                                                                                        | proven at p0 and p13                                 |

## Identity members

None. `kernel/codec/strict_json.py`, `ladder/run/closure/d1_runpod_billing_parser.py` and the test files are not reader-tuple members (measured against the 51-file list). Editing any member is a HALT (DECISION), never a hand-off.

## Scope

Two commits, in this order.

1. `feat(codec): add a parse_float seam to strict_loads (FOR-1632)`: RED-first tests in `tests/test_strict_json.py`, then the seam in `strict_json.py` per the semantics above. Existing door tests stay green unmodified.
2. `refactor(closure): adopt strict_loads in the billing parser (FOR-1632)`: `parse_runpodctl_billing_observation` decodes with `strict_loads(raw, fail=_fail, parse_float=Decimal)`; delete `_pairs` and `_nonfinite`; `tests/test_strict_json_adoption_run.py`: `_PENDING` becomes `frozenset()` and `test_the_pending_exemption_is_exactly_the_billing_parser_and_still_needed` is restated to assert the exemption set is empty (its own comment says it goes RED and demands exactly that); the W4-21 comment above `_PENDING` and the header note (line 8) are restated as "adopted by FOR-1632". The FOR-1577 acceptance count for `ladder/run` returns from "falls by 16" to "falls by 17"; quote the forbidden-hook file count before and after (`grep -rlE 'def (_pairs|_strict_pairs|_nonfinite|_reject_constant)\b' src --include='*.py' | wc -l`, measured 3 at `da4d35c`).

Duplicate-key and non-finite messages at the billing parser become the door's wording (`duplicate key: 'x'`, `non-finite constant: 'NaN'`); no test pins the old wording (measured: `tests/test_d1_runpod_billing_parser.py` pins only `calendar`), and the exception type is unchanged, so this is inside the ticket's remedy text.

## ACCEPTANCE

- RED-first in `tests/test_strict_json.py`: (a) `strict_loads('{"a": 0.0017170833889394999}', fail=..., parse_float=Decimal)` returns `{"a": Decimal("0.0017170833889394999")}` exactly (`==` and `str()` equality); (b) with `parse_float=Decimal`, `NaN`, `Infinity` and `-Infinity` constants are still refused through `fail` with the existing messages; (c) with `parse_float=None` the overflow token `1e999` is still refused (existing test stays green); (d) with `parse_float=Decimal` the token `1e999` decodes to a finite `Decimal` (`Decimal("1E+999").is_finite()`), matching what `json.loads(parse_float=Decimal)` admitted at the billing site before; (e) a `parse_float` returning a non-finite `float` is refused through `fail`.
- `tests/test_d1_runpod_billing_parser.py` green with its existing assertions unmodified (the 19-digit amount survives); additive rows are admissible when they pin the exception-surface invariant (as landed: `897291f` added three rows pinning that out-of-range decimal tokens raise `RunpodBillingSchemaError`, closing a pre-existing bare `decimal.InvalidOperation` escape); `tests/test_strict_json_adoption_run.py` green with `_PENDING == frozenset()`.
- `grep -rn '_pairs\|_nonfinite' src/entropic_substrate/ladder/run/closure/d1_runpod_billing_parser.py` is empty.
- Guard teeth by mutation, both arms quoted (below).
- remap_check at the tip over the mapping RE-DERIVED per `docs/plans/measurements/p13-sweep-remap.txt`: inversions <= 1, package cycles <= 4.
- Mechanical counts at or under their values at `da4d35c`: forbidden-hook files 3 (must fall to 2), `def _mapping(` 23, `separators=(",", ":")` 64, lowercase-SHA files 29, `type: ignore[arg-type]` 9, files > 1000 lines 2.
- `git diff --name-only ee9df9a...HEAD` minus the conductor's brief commit is a subset of *Allowed files*.
- `just check` green at the tip under the host lock; exit gate `just pc-ci` certified by `uv run python scripts/run_exit_gate.py --timeout-s 1800 --cmd "just pc-ci"` (JSON ok:true at HEAD, produced in the reporting seat).
- Draft PR stacked on `fix/for-1633-commit-msg-exclude` with a filled body (Linear ref FOR-1632, the ruling, both mutation arms, the counts, remap stdout).

## Non-goals

- Does NOT touch any other `strict_loads` adopter, `canonical_json`, `d1_cli._checked_payload`, or the `_text` trim semantics.
- Does NOT change the billing amount arithmetic, `canonical_decimal`, or any provider field.
- Does NOT touch the three retained SHA sites (FOR-1626 owns them, next phase in this stack).
- Does NOT launch pods, spend, build containers or start an MLflow server.

## Allowed files

- `src/entropic_substrate/kernel/codec/strict_json.py`
- `src/entropic_substrate/ladder/run/closure/d1_runpod_billing_parser.py`
- `tests/test_strict_json.py`
- `tests/test_strict_json_adoption_run.py`
- `tests/test_d1_runpod_billing_parser.py` (existing assertions restated only when the change itself invalidates them; additive rows pinning the parser's own exception type are admissible; say so in the report. Corrected 2026-10-08 after PR #334: the original wording was narrower than the landed `897291f`, recorded in that PR's body)
- `.secrets.baseline` (mechanical companion, digest literals only, stated in the PR body)

Never edit: `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/regime_rows.py`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `tools/mapping.json`, the manifest, the plan, briefs, `pyproject.toml`, `uv.lock`.

## Already ruled

- DN-1632-SEAM (operator, 2026-10-08): the seam is added; the holdout is not retained.
- W4-21 (2026-10-05): no workaround at the billing parser (no double decode, no float-to-Decimal reconstruction); superseded only in that the seam now exists.
- Door messages (W1-17) carry over: `fail` receives the door's reason strings.
- Ruling 14: abstraction admissible only for 2+ concrete sites or a public surface; the seam is a public-surface change on an existing door, admissibility = the one adopter plus the door's own tests.
- The commit-msg hook now lints every commit from the repo root (#333); the per-stage `gitlint --msg-filename` pre-check stays as belt and braces.

## Guard teeth by mutation

| id | mutation                                                                         | expected RED                                                                            | expected GREEN after revert |
| -- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------- |
| M1 | in `strict_loads`, ignore `parse_float` and always call `float(token)`           | ACCEPTANCE (a) RED (the 19-digit amount rounds), the billing-parser round trip RED      | both GREEN                  |
| M2 | in the `parse_float` branch, skip the finiteness check                           | ACCEPTANCE (e) RED                                                                      | GREEN                       |
| M3 | in the billing parser, pass a `fail` that raises `ValueError` instead of `_fail` | the adoption test's codec-boundary row for the billing parser (add one in commit 2) RED | GREEN                       |
| M4 | restore a local `_pairs` definition in the billing parser                        | `test_run_and_d1_cli_define_no_local_strict_json_hooks` RED                             | GREEN                       |
