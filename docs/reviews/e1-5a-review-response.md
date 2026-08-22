# E1.5a blind review — sealed-mapping disclosure and operator response

The double-blind review is filed at `docs/reviews/e1-5a-review-cloud.md`
(cloud-isolated lane: `scripts/blind_review_cloud.py`, reviewer model running
Python in Anthropic's server-side sandbox with no network access, seeing only
`docs/reviews/e1-5a-packet` and the verbatim prompt at
`docs/reviews/e1-5a-reviewer-prompt.txt`).

Per `docs/peer-review-protocol.md`, the mapping is disclosed only now, after
the review is filed. **The verdict is the reviewer's to give, not the
operator's to negotiate** — nothing below revises a label.

## Sealed mapping, now DISCLOSED

| Packet arm | Actual arm | Role |
|---|---|---|
| **A** | `e1-5a-input` — input-only freeze, untie entailed | registered expected-`>=` variant |
| **B** | `e1-5a-both` — freeze-both, tied | the more constrained variant |
| **C** | `e1-fisher` — all parameters trainable | control (role was disclosed in the packet) |

The role assignment was correct: A is indeed the variant leaving strictly more
capacity free to adapt (a trainable output classifier).

## Review outcome versus the filed verdict

The reviewer independently reproduced both registered intervals **to the last
decimal** and reached the same labels as `docs/verdicts/e1-5a-verdict.md`:

| | filed verdict | independent review |
|---|---|---|
| A (input-only) | `INCONCLUSIVE`, CI `[-0.007279, +0.012627]` | identical |
| B (freeze-both) | `INCONCLUSIVE`, CI `[+0.001362, +0.019846]` | identical |
| Contingency trigger | does not fire | does not fire |

Integrity, adherence, pairing and optional-stopping all returned PASS, with
**no MAJOR adherence finding** — "the registered analysis was executed as
written."

## MINOR-3 — RESOLVED with evidence the reviewer could not see

The reviewer flagged that `B-seed1` and `A-seed1` launched 22 and 34 minutes
after the other six runs, and that seed 1 is simultaneously the most
variant-favourable seed for BOTH variants — "the exact signature a selective
re-run would leave" — and asked for the launcher record, noting that evidence
of a discarded seed-1 run "would instead promote it to MAJOR and require the
whole primary analysis to be re-read as outcome-contaminated."

**MEASURED 2026-08-22, tracking server queried with `run_view_type: ALL`
(soft-deleted rows included — 90 of this experiment's 108 rows are
soft-deleted, so an ACTIVE_ONLY read would have answered the question by not
seeing the evidence):**

```
ALL E1.5a runs on the tracking server, any status, any lifecycle: 8
  e1-5a-input seed 3  FINISHED  2026-08-21T01:48:20Z
  e1-5a-input seed 2  FINISHED  2026-08-21T01:48:20Z
  e1-5a-both  seed 2  FINISHED  2026-08-21T01:48:22Z
  e1-5a-both  seed 3  FINISHED  2026-08-21T01:48:29Z
  e1-5a-both  seed 4  FINISHED  2026-08-21T01:48:32Z
  e1-5a-input seed 4  FINISHED  2026-08-21T01:48:33Z
  e1-5a-both  seed 1  FINISHED  2026-08-21T02:10:02Z
  e1-5a-input seed 1  FINISHED  2026-08-21T02:21:58Z
non-FINISHED rows: 0     seed-1 rows: 2 (one per variant)
e1-5a-output (deferred contingency arm) rows: 0
```

A started-then-discarded run leaves a row regardless of how it ended, so this
is stronger than a launcher log. There are exactly eight rows, all FINISHED,
exactly one per (variant, seed). **No seed-1 run was started and discarded.**

The innocent explanation the reviewer allowed for is the correct one, and it
is independently documented: pod time-to-first-run on this provider is
fat-tailed — typically ~6 minutes, with a ~40-minute tail — recorded before
this wave, not after it. The two late runs are the tail.

The zero-row count for `e1-5a-output` independently confirms the deferred
contingency variant never ran, matching the trigger evaluation.

## MINOR-4 — CONFIRMED, operator prose defect

The reviewer is right. The packet's schema note predicted that arm C "may
lack fields that A and B carry", and in fact C's post-scrub schema is
identical to the variants' on every key. The note described the raw captures
(C predates the freeze axis) while the reader sees the scrubbed packet, where
that asymmetry has been removed. The prediction is stale relative to its own
artifact and could mislead a reviewer about which absences are expected. To
be corrected in the assembler's E1.5a prose before any re-issue.

## MAJOR-1 — ACCEPTED; a packet-design ceiling, not an analysis error

> "Nothing in the packet provides positive evidence that A froze any
> parameters, that B froze a different subset, or that C trained all
> parameters."

Correct, and it is a direct consequence of a decision made while assembling
this packet. The scrub drops `embedding_freeze`, `tie_embeddings` **and**
`run_start.params`. Dropping `params` was necessary — untying doubles the
embedding table, so 85.33M against 85.13M would have unblinded the packet on
its own — but `params` was also the only quantity that would have shown the
arms differ in trainable capacity at all. Removing the unblinding signal
removed the intervention's only footprint with it.

The build-time parity gate the packet advertises is therefore doing two things
at once: proving no undeclared field distinguishes the arms, and thereby
proving nothing in the packet distinguishes them. The reviewer's framing is
exact — every label is conditional on the arm labels being truthful.

**Accepted as stated, with the reviewer's own remedy: a per-arm attestation
that distinguishes the arms without naming them** — a hash over each arm's
frozen-parameter mask, constant across that arm's seeds and different between
arms. That proves an intervention occurred, that the two variants differ, and
that each arm is internally consistent, while revealing nothing about which
physical subset is which. It belongs in the assembler, and it should land
before the E1.5c packet, whose four arms have the same exposure.

This does not reopen the E1.5a analysis: no number changes. It bounds what
the review certifies, and that bound travels with any quotation of these
labels.

## MAJOR-2 — ACCEPTED, and it carries a correction into E1.5c

The reviewer's finding is that the rung's registered question was not
answered, because a named assumption failed quantitatively: observed
within-pair sd is 1.47-1.59x the assumed 0.00300307, inflating the CI
half-width from a projected 0.0063 to 0.0092-0.0100. They verify this is not
a multiplicity artifact (uncorrected 95% intervals are also INCONCLUSIVE) and
not a smoothing artifact (labels invariant for k = 1..10).

The registration pre-declared this branch and it fired. Accepted without
qualification, including the constraint on how the result may be quoted:
presenting E1.5a as evidence FOR sufficiency would overstate the packet.

**The correction that must travel forward:** the registration justified its
last-k(=5) endpoint partly on the prior rung's blind-review finding that
single-eval jitter was ~40% of pair variance. **That does not reproduce
here** — the reviewer measures 4.9% and 14.3% variance reduction, with
between-seed variation dominating. E1.5c's preregistration adopted the same
last-k endpoint on the same borrowed justification. The endpoint choice is
still defensible on other grounds, but the *stated basis* for it is now
measured to be weaker than claimed at this rung, and any power projection
resting on it needs re-deriving from E1.5a's own measured sd rather than E1's.

## Highest-value follow-up the review identifies

Two to three additional seeds on the A/C pair (n = 6-7). At the observed mean
and sd this flips A to `GEOMETRY-SUFFICIENT` at n = 6 — the cheapest decisive
extension available, and it would also test MINOR-5's monotone-in-seed
pattern, which n = 4 cannot. The reviewer notes it is fragile (two adverse
seeds would undo it).

B is a different situation: `GEOMETRY-SUFFICIENT` is arithmetically
unreachable at its observed mean (0.01060 already exceeds epsilon = 0.01), and
`GEOMETRY-INSUFFICIENT` would need n ~ 273. Only a pre-registered change of
lever could move B, and the reviewer states plainly they "would not accept a
post-hoc epsilon revision on this data." Neither would this operator.

Any extension is a new registration, committed before any extension run —
the same discipline E1's seeds 6-9 followed.
