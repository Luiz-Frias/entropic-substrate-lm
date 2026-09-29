# p14 pre-phase rung briefs: index

Written 2026-09-28 by the rung-brief lane. Four rungs, each parented on `refactor/p14-consolidation` and landed into it serially in this order before `p14-r1-strict-json` launches (plan, *Prerequisites and pre-phase rungs*; rulings 7 and 12). Premises measured on the live tree whose trunk ref read `4e0ae6a` at authoring; the phase briefs cite `eb93d5d`.

| order | brief             | branch                                        | issues   | identity members (reader-tuple files touched)                                                                                                      | conductor re-pin                   | touch set                                      | live gate required                          |
| ----- | ----------------- | --------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------- | ------------------------------------------- |
| 1     | `pre-for-1567.md` | `fix/for-1567-checkpoint-publication-history` | FOR-1567 | 5 expected: `object_worker`, `worker_snapshot`, `worker_transport`, `runner`, `seals`; up to 7 with `worker_execution`, `worker_command` if proven | yes, `h-checkpoint-reader-v2` only | 19 (18 existing, 1 new)                        | no                                          |
| 2     | `pre-for-1617.md` | `fix/for-1617-graceful-operator-stop`         | FOR-1617 | 3: `runner`, `worker_execution`, `object_worker`                                                                                                   | yes, `h-checkpoint-reader-v2` only | 15 (13 existing, 2 new)                        | no, unless `mlflow.py` is needed (DECISION) |
| 3     | `pre-for-1614.md` | `chore/for-1614-text-trim-semantics`          | FOR-1614 | 0                                                                                                                                                  | no                                 | 3 (2 new, 1 conditional on the gated commit 2) | no                                          |
| 4     | `pre-for-1615.md` | `chore/for-1615-lowercase-sha-boundary`       | FOR-1615 | 0                                                                                                                                                  | no                                 | 38 (35 source, 1 docstring-only test, 2 new)   | no                                          |

All members are `ladder/run/*.py` files in the checkpoint tuple (`ladder/run/checkpoint.py:65-117`); none is in the source tuple (`ladder/capture/capture.py:29-37`), so `h-source-reader-v3` never re-measures in these rungs. Touch sets count `src`, tests, the justfile and new evidence files, and exclude the conductor-only re-pin files and `.secrets.baseline`.

Every rung still publishes through the conductor's `just ship-it`, which runs `just container-test`; "live gate required" means `gates.liveRequired`.

## Open for the conductor before launch

- Rung 2: the per-step stop poll adds one R2 GET per step, cost UNMEASURED, and a new failure exposure under `max_attempts` 0; recommendation in the brief is to route it through rung 1's read retry, which extends a retry ruled for verification reads only.
- Rung 3: the conductor's instruction says no semantic change now; the ratified plan row lands commit 2 when the audit count is zero. The brief gates commit 2 on a zero count AND an explicit conductor go.
- Rungs 1 and 2: the registry says `tests/test_identity_epoch.py` is touched once for the whole sprint; ruling 7 gives each of these rungs its own re-pin. The briefs follow ruling 7.
- Rungs 1 and 2: each re-pin invalidates every checkpoint on R2 under the previous reader digest (`checkpoint.py:304-305`); the H image is rebuilt after rung 2 before any green-lit run.
