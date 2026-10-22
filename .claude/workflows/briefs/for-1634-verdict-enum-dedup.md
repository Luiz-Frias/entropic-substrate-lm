# for-1634-verdict-enum-dedup: the two ladder StrEnums alias the kernel verdict vocabularies

DRAFT 2026-10-19 by the conductor (Bug Squashing wave `bugsquash-2026-10-19`; issue FOR-1634, filed from
the p14-t-verdict-vocab L2/L1 findings on PR #330). Anchors measured at main `04ad7790` with `grep -n`;
the implementer re-verifies them in recon. The Linear ticket is authoritative.

## Branch

| field                | value                                                                                                                 |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1634-verdict-enum-dedup`                                                                                     |
| parent               | the wave base branch (main `04ad7790` + the briefs commit)                                                            |
| cascade `phase.mode` | `normal` (ladder -> kernel import, allowed layer direction; the StrEnum door stays frozen; persisted bytes identical) |
| publish              | none in the lane (`publishMode: defer`, CERTIFIED); the zip publishes                                                 |

## Why (measured at `04ad7790`)

- `src/entropic_substrate/kernel/verdict/d1_types.py:46` `ClauseStatus(StrEnum)` = PASS/FAIL/INCONCLUSIVE; `:89` `ReviewFindingStatus(StrEnum)` = OPEN/RESOLVED.
- `src/entropic_substrate/ladder/analyze/d1_clause_policy.py:80` `D1ConditionStatus(StrEnum)` = the same three values; imported by `src/entropic_substrate/ladder/review/d1_machine_analysis_binding.py` and `tests/test_d1_clause_policy.py`.
- `src/entropic_substrate/ladder/review/d1_review_adjudication.py:204` `D1MajorFindingStatus(StrEnum)` = the same two values; used by `D1MajorFindingDisposition.status` (`:212`) and `tests/test_d1_review_adjudication.py`.
- Neither ladder module imports `kernel.verdict` today (`d1_clause_policy.py:12-13`, `d1_review_adjudication.py:14,20` import other kernel modules). StrEnum `==` is str equality so the pairs compare equal but are not `is`-identical; no parity test pins them.
- Tripwire: `tests/test_d1_verdict_v2.py` byte-identical round-trip tests (`:70-137`) must stay green; `.value` serialization unchanged.

## Ruling (ticket remedy; one of two forms, implementer picks and records)

Form A: delete the two ladder enums and import the kernel ones under their old names
(`from entropic_substrate.kernel.verdict.d1_types import ClauseStatus as D1ConditionStatus`, same for
`ReviewFindingStatus as D1MajorFindingStatus`), keeping every importer source-compatible. Form B: keep
the names as module-level aliases `D1ConditionStatus = ClauseStatus`. Either way the ladder modules must
not redefine the member set. Add a parity test that fails if either kernel enum's value set drifts from
what the ladder consumers accept. Out of scope (recorded in the ticket as latent debt, not a defect): the
raw-literal comparisons in `d1_decision_validation.py` / `d1_validator.py` and the evidence-class wire
literals in the six producer modules.

## Scope

1. **C1, tests RED at the parent**: `tests/test_d1_clause_policy.py` and `tests/test_d1_review_adjudication.py` gain a pin that `D1ConditionStatus is ClauseStatus` and `D1MajorFindingStatus is ReviewFindingStatus` (identity, not equality) and that each member `is` the kernel member; RED at the parent (distinct classes). A parity test over the value sets (`{m.value for m in ...}` equal) that stays meaningful under M2.
2. **C2, implementation**: the alias/import in the two ladder modules; delete the duplicate class bodies; no `.value` change; the `# nosec B105` comment moves with the kernel definition (already there).
3. **C3**: `tests/test_d1_verdict_v2.py` and `tests/test_d1_machine_analysis_binding.py` (if present) stay green unchanged.

## Guard teeth by mutation

| id | mutation                                                           | expected RED                |
| -- | ------------------------------------------------------------------ | --------------------------- |
| M1 | restore a local `class D1ConditionStatus(StrEnum)` duplicate       | the identity pin RED        |
| M2 | add a member to the kernel `ReviewFindingStatus`                   | the parity test RED         |
| M3 | alias `D1MajorFindingStatus` to `ClauseStatus` (wrong kernel enum) | the member identity pin RED |

## Allowed files

`src/entropic_substrate/ladder/analyze/d1_clause_policy.py`, `src/entropic_substrate/ladder/review/d1_review_adjudication.py`,
`tests/test_d1_clause_policy.py`, `tests/test_d1_review_adjudication.py`. Never: `kernel/verdict/d1_types.py`
(the frozen door; M2 is a scratch mutation, restored), `d1_machine_analysis_binding.py` (an importer that must
stay source-compatible), `docs/*` (post-zip sweep).

## Acceptance

C1 RED at the parent, GREEN at the tip; M1-M3 RED one at a time and restored; `just check` exit 0 and the
exit-gate runner green under the suite lock; `tests/test_d1_verdict_v2.py` unchanged and green; diff
confined to the Allowed files.
