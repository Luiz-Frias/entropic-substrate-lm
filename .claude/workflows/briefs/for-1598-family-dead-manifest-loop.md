# for-1598-family-dead-manifest-loop: delete the dead manifest-diff loop in D1ExperimentFamily

DRAFT 2026-10-19 by the conductor (Bug Squashing wave `bugsquash-2026-10-19`, lane of a mass-patch-fanout
over the hexagonal-refactor project's Todo tickets; issue FOR-1598, thermo-nuclear-review finding at epoch
`0e55d9d`, adversarially verified). Anchors measured by the conductor at main `04ad7790` (v0.3.2) with
`sed -n`; the implementer re-verifies them in recon. The Linear ticket is authoritative.

## Branch

| field                | value                                                                 |
| -------------------- | --------------------------------------------------------------------- |
| branch               | `fix/for-1598-family-dead-manifest-loop`                              |
| parent               | the wave base branch (main `04ad7790` + the briefs commit)            |
| cascade `phase.mode` | `normal` (one module, no door, no reader-tuple member)                |
| publish              | none in the lane (`publishMode: defer`, CERTIFIED); the zip publishes |

## Why (measured at `04ad7790`)

`src/entropic_substrate/ladder/register/d1_family.py:247-261`: `baseline = self.drafts[0]`, the
`shared_fields` tuple (codec, encoder_pair, boundary_state, state_construction, target, future_law,
geometry, execution_protocol) and the loop that raises "D1 family arms must be capacity-matched outside
stabilization strategies" when any shared field differs. `:263-279`: `allowed_to_vary` = {arm, collapse,
collapse_strategy, ema, target_update}, then a second loop that builds every draft's
`construction_manifest`, diffs it against the baseline's and raises "D1 family cells may differ only on
registered stabilization strategy axes". The drafts are frozen dataclasses with value equality and every
`.manifest` is a pure function of the same fields, so an input the first loop accepts can never make the
second loop fire: the second loop is dead defensive code that costs two full recursive manifest walks.

## Scope

1. **C1, test RED at the parent**: `tests/test_d1_family.py` gains a pin that a family whose drafts differ
   ONLY on the five stabilization axes is accepted and whose drafts differ on any shared field is refused
   with the first loop's message; plus a pin that `construction_manifest` is NOT evaluated during
   `__post_init__` for an accepted family (e.g. a draft subclass / monkeypatched property that raises if
   read; the parent reads it, so RED at the parent). Existing tests in `tests/test_d1_family.py`,
   `tests/test_d1_stabilization.py`, `tests/test_d1_runtime_construction.py` stay green.
2. **C2, implementation**: delete `allowed_to_vary` and the second loop (`:263-279`); keep the first loop
   byte-identical (same message). Record the measurement in the module comment (house rule 4).

## Guard teeth by mutation

| id | mutation                                  | expected RED                         |
| -- | ----------------------------------------- | ------------------------------------ |
| M1 | drop the first loop too                   | the shared-field refusal pin RED     |
| M2 | re-add a manifest read in `__post_init__` | the "manifest not evaluated" pin RED |

## Allowed files

`src/entropic_substrate/ladder/register/d1_family.py`, `tests/test_d1_family.py`. Nothing else; no
`docs/architecture.md` edit in the lane (docs ride the conductor's post-zip sweep).

## Acceptance

C1 RED at the parent, GREEN at the tip; M1/M2 RED one at a time and restored; `just check` exit 0 and
the exit-gate runner green under the suite lock; diff confined to the Allowed files; no caller-visible
exception type or message changes beyond the deleted second message.
