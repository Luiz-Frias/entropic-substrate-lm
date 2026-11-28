# Platform refactor debt ledger

Horizon M0 of the platform refactor commits the older debt plan's phases as a tracked ledger
(plan register row D25, `docs/plans/series-plugin-topology.plan.md`). This file is that ledger.
It records where each phase stands and which horizon absorbs it. It authorizes no work: each
horizon's own gate does that.

## Where the debt plan came from

The plan of record cites "the debt plan's phases 0 to 2" three times and never carried the plan
itself. No tracked file held it. The only copy found is the "Phased remediation" table on the
system page of the 2026-10-30 architecture survey, measured at commit `b9c570e0`. That page is
generated; its source table sits in the generator, outside this repository.

| fact | value | evidence |
| --- | --- | --- |
| source | "Phased remediation", six phases numbered 0 to 5, with the "Systemic seams" table above it | `[measured]` read from the survey page generator on 2026-11-28 |
| survey commit | `b9c570e0` | `[measured]` plan, "Context" |
| debt items behind the phases | 129, each with a file citation that survived a verifier | `[relayed]` plan, "Context" |
| operator approval | this ledger, with this source named in it, was approved as written on 2026-11-28 | the operator's words in the M0 session: "All four drafts approved!" |

The phase rows below quote the source's goal and items. The state columns are re-measured at
`9d7917c81af71c9faf23591469db374f1457c9e0`, whose `src` and `tests` trees are identical to the
epoch commit `5f6b6dd9`.

## Ledger

Evidence classes: `[measured]` by the command named, in the M0 worktree on 2026-11-28;
`[relayed]` from the source review `docs/reviews/2026-11-27-platform-refactor-source-review.md`,
not re-read here.

| phase | goal, from the source | state at the epoch commit | absorbed by |
| --- | --- | --- | --- |
| 0 Correctness | fix the three items where behaviour is wrong today | landed, see the item table below | closed before the sprint |
| 1 Digest seam | one SHA-256 validator, one `_tensor_digest`, one canonical JSON writer | not landed | M4 |
| 2 Codec seam | a bound strict-codec helper factory in the kernel | not landed | M4 |
| 3 Ownership rulings | decide each twice-declared name and each unwired path | partly ruled: unwired paths are deleted unless a production caller is named (D22); the twice-declared names are not ruled | M1 for the deletions, M6 for the names |
| 4 Cycle teeth | subpackage contracts in `.importlinter` | not landed; the cycles grew from four to five | M9 |
| 5 Docs sweep | retract stale measured claims in their own commits | not landed as a sweep | M2 for test paths in member comments; the rest rides each horizon's re-pin commit |

### Phase 0, item by item

| item, from the source | state | evidence |
| --- | --- | --- |
| the H tiny tracer finishes the sink FINISHED on a raised run | fixed | `[measured]` `git log origin/main`: `dd466fed`, "fix(h-tracer): finish the tiny trace run FAILED on a raised exit", pull request 412 |
| `read_json_artifact` accepts NaN and Infinity on an evidence path | fixed | `[measured]` `e1618a2d`, "fix(kernel): refuse non-finite JSON in read_json_artifact"; `kernel/artifacts.py:371` passes `parse_constant` to a refusing function |
| H price admission runs after pod create, so a refused price is already billing | reported landed | `[relayed]` source review section 1.1, "the correctness leg did"; not re-read here |

### Phases 1 and 2, sized

These counts size the seams M4 adopts. They are definitions and call sites found by `grep` under
`src/entropic_substrate`, all `[measured]`.

| seam | what was counted | count |
| --- | --- | --- |
| digest | `hashlib.sha256(` call sites | 147, in 83 files |
| digest | functions named `_canonical...` | 31, in 31 files |
| digest | functions named `_digest...` | 11 |
| digest | functions named `_tensor_digest` | 2, in 2 files |
| codec | module-level `_fail` helpers | 45, in 45 files |
| codec | module-level `_text` helpers | 30, in 30 files |
| codec | module-level `_mapping` helpers | 23, in 23 files |
| codec | module-level `_exact` helpers | 15, in 15 files |
| codec | module-level `_integer` helpers | 15, in 15 files |

The source review relays 235 clone definitions and 73 canonical writer sites for the same seams,
counted by a different instrument. The two sets of numbers are not reconciled here; M4's gate
re-measures its own touch-set before any file moves.

One constraint binds the digest seam and is not in the source: ruling R18 and source review risk
7 require two named canonical schemes that are pinned and never merged, with golden vectors per
scheme. "One canonical JSON writer" in the source therefore becomes two named ones.

### Phase 3, the items

The source names eight twice-declared names and seven unwired paths without one list for each.
From its "Systemic seams" table:

| kind | named in the source |
| --- | --- |
| twice-declared | `PreregistrationIdentity`, `D1CampaignCheckpointRepositoryPort`, `D1CampaignAuthorityLoaderPort`, `Verdict`, `_tensor_digest`, the R2 mailbox worker, the journal-event factories, the transient provider code set |
| unwired | the Build Cloud adapter, the strict MLflow chain, the alias RNG, the Docker analysis runner's real path, the review and filing stages past capture, `codec_equivariance`, two facades |

Register row D22 rules the unwired paths: delete unless a constructing production caller is
named, with the list shown to the operator at the M1 gate. Register row D19 already rules one of
them, the strict D1 chain that only tests construct. These names are from `b9c570e0`; M1 and M6
re-measure which still exist.

## Second-pass findings

Ruling R23 sends review findings that do not block to a second pass after M12. That bucket is
the Linear milestone "Bug squashing (second pass)" in project `P-FOR-45`, one issue per finding,
each naming its horizon, its pull request and thread, and the commit it was observed at. This
ledger does not copy those issues. It records the debt plan's phases only.

## Keeping this ledger

A horizon that lands a phase, or part of one, edits that phase's row in its own stack and names
the commit. A row is never deleted: a phase that turns out not to be needed is marked retired
with the ruling that retired it.
