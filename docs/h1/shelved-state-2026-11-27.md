# H1 shelved state, 2026-11-27

The operator ruled on 2026-11-27 that the platform refactor
(`docs/plans/series-plugin-topology.plan.md`, rung seven) runs before H1 resumes, and that the
in-flight H1 work is shelved and re-implemented on the new platform rather than rebased. This
note is what a later session needs to resume H1 without the chat that led here. It is a state
record, not a verdict, and it authorizes nothing.

Evidence labels: `[measured]` was read or run by the conductor on 2026-11-27 at `origin/main`
`5f6b6dd92b9da007624ce255492f55b779fd4bfd`; `[relayed]` comes from the in-flight lens of the
source review and its verifier
(`docs/reviews/2026-11-27-platform-refactor-source-review.md`), not re-read here; `[unknown]`
could not be determined.

## Where H1 stands

| item | state | evidence |
| --- | --- | --- |
| H0 | filed `tracer_complete` | `[measured]` `docs/h0/verdict-final-probe-2026-10-26.md` |
| H1 registration | nothing sealed; the skeleton reads "DRAFT / NOT SEALED" and its identity rows are blank | `[relayed]` `docs/h1/preregistration-skeleton.md:3`, `:316-318` |
| H1 captures | none tracked | `[relayed]` `git ls-files results` lists no H1 capture |
| W1 | merged | `[relayed]` |
| W2 slices L4a, L4b, L4c | merged 2026-11-04 | `[relayed]` `443056c1`, `cd7429c1`, `3760679d` are ancestors of main |
| W2 slice L4d | unmerged, shelved (below) | `[measured]` |
| W2 slice L4e | golden pins only, shelved (below) | `[measured]` |
| W3 (L7, L6, L5), W4 (L8 registration, L9 image re-pin), L10 pilot | not started | `[relayed]` sprint manifest |
| detail-preservation instruments (entities, numbers, negation, relations, order) | in no H1 phase; the skeleton marks the estimand blank | `[relayed]` `docs/h1/preregistration-skeleton.md:93`, `:325` |

Because nothing H1 is sealed, reader-identity re-pins during the refactor invalidate no
evidence. H0 captures already replay only at their epoch commit: 21 of the 54 reader member
files changed since the H0 tip `[relayed]`.

## The three shelved refs

| work | ref that holds it | head | own change | reader members touched |
| --- | --- | --- | --- | --- |
| L4d evaluation instruments (closed PR 435) | remote `refs/pull/435/head`; local branch `shelf/h1/l4d-evaluation-instruments` (not pushed); local `refs/gt-fetch-head/h1-l4d-evaluation-instruments` | `35b0b687` | 23 commits, 65 files, +5219/-605; src 16 files +1385/-200 `[relayed]` | 9 of the 54 checkpoint members, 0 source members `[relayed]` |
| L4e segmenter alignment | `origin/h1-l4e-segmenter-alignment` | `73dc7968` | 2 own commits, tests only: a golden of per-span byte boundaries for 91 documents and its guard cases `[relayed]` | none |
| FOR-1713 codec hand-off residual pins | `origin/feat/for-1713-codec-handoff-residual-pins` | `9c3729e1` | 1 own commit, tests only: pins on hand-off defect text, the seal walk and progress writer structure `[relayed]` | none |

Trap for whoever diffs these: `git diff origin/main...<ref>` on the second and third refs also
shows 11 stale L4d commits. Their own change is `git diff 391693b8..<ref>` `[relayed]`.

### L4d: what it did and what is left

- **Done on the ref.** Prior and mixture proposals beside the q proposal with their own
  effective-sample-size keys; the `h-predictive-evaluation-v2` record with v1 read-old; an
  additive `run_start.profile` widening; a latent-usage estimator chosen through a closed
  enumeration with one built member and one refused by name (FOR-1721); regime rows recomputed
  over its merge with main at `c161a032` `[relayed]`.
- **Not known.** Why PR 435 was closed, and whether CI was green at `35b0b687`. Review reached
  round eight against a budget of five `[relayed]`. No green claim exists for this head.
- **Cost of shelving, stated so the ruling is informed.** Ten of its sixteen source files are
  files the refactor splits across members, so resuming means re-implementing about 445 inserted
  lines by hand in the split files; the remaining 940 lines sit in files that move whole
  `[relayed]`. The source review recommends landing it first for that reason. The operator kept
  the shelve ruling on 2026-11-27 with that cost in view, register row D12 in the plan, and
  the engine design gates read this ref as design input.
- **What the refactor should take from it either way.** The evaluation record is validated
  three times in three modules and a bound was missed in one of them on three review rounds
  `[relayed]`: one schema module owned by the series and imported by the writer and both readers
  removes that class. The estimator is threaded through five parameters from three composition
  sites `[relayed]`: the series plugin binds it once and the engine calls it at one named point.
  The runner imports the H metric extractor directly `[relayed]`: after the move that is a
  platform-to-series import, so it becomes a port call.

### L4e: what it did and what is left

- **Done on the ref.** The golden spans, recorded at `391693b8`, and two guard cases. The golden
  is still valid at main: segmentation, corpus and records are unchanged since its base and
  pysbd is still 0.3.4 `[relayed]`.
- **Left, unwritten.** The DH-35 fix itself: sequential alignment of pysbd sentence texts from
  the cursor in `segment_document` (`src/entropic_substrate/core/sentence/segmentation.py`);
  failing-first fixtures for `. .`, `...`, `???` and the four enwik8 pages that fail today
  (indices 261, 569, 766 and 1280 in the first 10 MB); the before and after segmenter counts;
  the mutation teeth in the brief `.claude/workflows/briefs/h1-l4e-segmenter-alignment.md`
  `[relayed]`.
- **Doors at resume.** Segmentation is a member of both reader identities, so the fix is a
  conductor re-pin of every identity row plus a read-old-table row, and its gate is the live
  gate.
- **Use during the refactor, ruled 2026-11-27 as register row D13.** The golden is ported into
  the refactor's first rails horizon as the equivalence proof for moving segmentation into the H
  series member. It imports a test path that no longer exists, so it is relocated as it lands.

### FOR-1713: what it did and what is left

- **Done on the ref.** Mutation-proven pins on `ladder/run/codec_handoff.py`; no source change.
  Its anchors are current: that module moved by one comment since the ref's base `[relayed]`.
- **Left.** Relocation to the tiered test layout, which the same ruling does in the first rails
  horizon. Its structure scan reads one package root, so after the workspace split it would miss
  a second writer in another member; it needs a workspace-wide root.

## Things a resume must not assume

| item | state |
| --- | --- |
| the sprint manifest on main | wrong: `.claude/workflows/sprint-manifest.json` marks L4a, L4b and L4c `queued`; only the L4d ref marks them done `[relayed]`. Correcting it is a tracked change the refactor's first horizon owns |
| the W3 base pointer `results/h1-dev-quote-v4.499aca.dvc` | not found: no file of that name under the repository root in either checkout `[measured]`, and no commit carries the path `[relayed]`. The DVC remote was not queried. Treat it as lost until someone locates it |
| the DH-34 confirming quote pods | not run; spend and an operator GO, and frozen while reader members move |
| every path in the H1 briefs | stale after the refactor; re-measure each anchor before use |
| unreachable objects | the object store holds dropped stash entries dated 2026-11-22 among 1485 unreachable commits `[relayed]`; a `git gc --prune` deletes them. Nothing has been pruned |

## Freeze while the refactor runs

While any reader-identity member is in motion, no H1 registration is sealed, no H capture is
admitted, no image is re-pinned and no H pod is created. H1 resumes from the refactor's closing
tip with its briefs re-anchored. This is ruling R12 in the plan. The operator ruled on
2026-11-27 that the freeze is full and covers the DH-34 quote pods, and that the refactor closes
with an H pod run compared end to end against a run that already has results, ruling R25.

## Resume order, once the freeze lifts

1. Re-implement the L4d instruments as the H series' evaluation adapter on the one engine.
2. Run the DH-34 quote pods against them, under a fresh GO with the balance said aloud.
3. Write the DH-35 segmenter fix with the L4e golden as its rail.
4. W3, then L8 seals the registration, then L9 and the pilot.
5. File the detail-preservation instruments as their own H1 ticket before L8, or revise the
   planned exit, as the repo Horizon requires.
