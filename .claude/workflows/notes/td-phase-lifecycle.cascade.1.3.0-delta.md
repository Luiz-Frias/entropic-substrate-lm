# 0.1.0-snapshot -> 1.3.0 delta reading (2026-09-17)

Re-based per coordinator instruction: `/reload-plugins` installed helmsman
1.3.0 mid-session, so the top-level `helmsman/0.1.0/.claude/workflows/`
directory (which never held anything but `worktrees/` on this machine anyway)
is no longer the live install; `helmsman/1.3.0/.claude/workflows/td-phase-lifecycle.cascade.js`
(1654 lines) is now authoritative. Verified line counts before anything else
(Lens axiom — check the instrument before the observation):

```text
$ wc -l 0.1.0-snapshot 1.3.0
    1285 .../0.1.0/.claude/worktrees/wf_1d8e000a-2db-6/.claude/workflows/td-phase-lifecycle.cascade.js
    1654 .../1.3.0/.claude/workflows/td-phase-lifecycle.cascade.js
$ diff 0.1.0-snapshot 1.3.0 | grep -c '^[<>]'
455
```

455 changed lines confirmed, matching the coordinator's figure exactly. The
five `run-exit-gate.ts` sites and two `bun install` sites the coordinator
named also match exactly:

```text
$ grep -n -i 'bun ' 1.3.0/.claude/workflows/td-phase-lifecycle.cascade.js
760:  // exitGate ... (bun src/gate/adapters/run-exit-gate.ts
825:    're-run the deterministic runner (`bun src/gate/adapters/run-exit-gate.ts --cmd ...
1303:      '5. EXIT GATE ... `bun src/gate/adapters/run-exit-gate.ts --cmd ...
1372:  ' ... then bun install (and any additional per-package installs ...
1378:    '- RE-RUN ... via `bun src/gate/adapters/run-exit-gate.ts --cmd ...
1459:      'ROUND C ... run bun install there if the gates need it; ...
1570:    '2. Re-run ... via `bun src/gate/adapters/run-exit-gate.ts --cmd ...
```

## Hunk list (diff -u 0.1.0-snapshot 1.3.0, headers only)

```text
@@ -61,7 +61,7 @@      meta.whenToUse: +args.phase.startTip mention
@@ -71,6 +71,7 @@      meta.phases: +Teardown entry
@@ -171,19 +172,50 @@    resolveGatePrecedence: explicit-gates completeness now enforced (fail loud)
@@ -276,19 +308,31 @@    sanitizeSlug/advisorLockPath: 'repo' silent-default -> throw (fail closed)
@@ -299,16 +343,31 @@    G.liveRequired comment rewritten (CodeRabbit follow-up, no behavior change)
@@ -450,6 +509,174 @@    NEW: packet git-ceremony + ancestry-chain hardening block (FOR-1302)
                         (packetGitCeremony, tddBundlingNote, composePacketWithCeremony,
                          implementPacketAncestryInstruction, composeImplementPacket,
                          postPacketAncestryHaltReason, seedEngineTrackedTip — all new
                          GUARD-PIN blocks)
@@ -473,7 +700,7 @@      haiku fixer packet now wrapped: composePacketWithCeremony(buildFixPacket(...))
@@ -488,7 +715,15 @@      sonnet fixer prompt now appends PACKET_GIT_CEREMONY_BLOCK
@@ -513,6 +748,7 @@      shared BRIEF: +TDD_BUNDLING_NOTE line
@@ -611,6 +847,25 @@      IMPL schema: +optional ancestryCheck {cmd, startTip, exitCode, output}
@@ -721,6 +976,23 @@      NEW: TEARDOWN schema (removed/kept/branchesDeleted/remaining/notes)
@@ -902,6 +1174,26 @@      NEW: engineTrackedTip seeded from seedEngineTrackedTip(PH) before packet loop
@@ -910,8 +1202,29 @@      haiku packet call wrapped in composeImplementPacket; ancestry check + halt
                         after every commit-bearing report (not just GREEN)
@@ -921,16 +1234,44 @@      sonnet takeover packet: same composeImplementPacket wrap + ancestry check
                         + engineTrackedTip advance guarded on typeof s.commit === 'string'
@@ -1267,6 +1608,33 @@    NEW: Phase 7 "Teardown" inserted between Validate and the terminal return
                         (TEARDOWN schema, haiku teardown agent, teardownReport)
@@ -1280,6 +1648,7 @@      terminal return: +teardown: teardownReport field
```

## What the new lines add

**1. Two CodeRabbit-driven fail-closed hardenings (gate resolution + advisor
lock), no new capability, just closing silent-fallback holes:**
`resolveGatePrecedence` used to accept an incomplete `explicitGates` object
(e.g. `{liveRequired: true}` alone) and silently fall through to the ratified
profile; now presence of ANY recognized key (`offline`/`exit`/`live`/
`liveRequired`) commits the caller to speaking explicitly, and an incomplete
object throws naming the missing keys. `sanitizeSlug`/`advisorLockPath` used
to bucket an unresolvable repo identity into the literal string `'repo'`,
which meant two different repos with no usable identity collided on the same
advisor lock file — now it throws instead. Neither changes cascade.js's
observable contract with a well-formed caller; both close a silent-fallback
class the axioms (`Teeth`, `Legibility`) flag as a defect, not a feature.

**2. Packet git-ceremony + ancestry-chain hardening (FOR-1302), a NORMAL-mode
implement-path-only capability add.** Every haiku/sonnet fix packet and every
haiku/sonnet implement packet now gets an ENGINE-APPENDED (never
agent-authored) block forbidding `git commit --amend`/`rebase`/`reset` and
requiring a STOP-and-DECISION on any git anomaly — closing a measured
incident (p8 first NORMAL-mode run: a haiku packet executor amended over a
prior packet's commit after a gitlint rejection). Implement packets
additionally self-report a post-commit `git merge-base --is-ancestor
<startTip> HEAD` check; `postPacketAncestryHaltReason` judges it
deterministically against `engineTrackedTip`, the engine's own record of the
last certified commit (not the report's own claim — closes a CodeRabbit
finding that a fabricated-but-well-formed startTip could otherwise pass).
`seedEngineTrackedTip` lets the conductor seed packet #1's anchor via a new
optional `args.phase.startTip` (a machine-read `git rev-parse` of the parent
tip). **This is the direct dependency mass-patch-fanout.js exercises**: its
`Provision` phase reads the parent tip machine-to-machine exactly once and
`laneArgs` forwards it as each lane's `phase.startTip` precisely to seed this
new chain-of-custody anchor — a lane composed without a valid 40-hex
`startTip` throws in `laneArgs` before it ever reaches the cascade engine.

**3. New Phase 7 "Teardown"** (operator ruling 2026-09-09): on the SUBMITTED
path only, a haiku agent releases `.claude/worktrees/wf_*` isolation
worktrees the harness creates one-per-reviewer-call and never cleans up
(measured: 30 leftovers from four runs). Scoped strictly to worktree paths
matching `/.claude/worktrees/wf_` **directly under `ARGS.repoToplevel`**;
clean trees are removed without `--force`, dirty/unremovable ones are kept
and reported, and a dead or failing teardown never changes the phase verdict
(Terminal axiom: per-resource release at ITS terminal, never blocking the
verdict on cleanup).

## checkout / lane / nested grep — cascade.js vs mass-patch-fanout.js

```text
$ grep -n -i 'checkout\|lane\|nested\|A\.checkout\|args\.checkout' td-phase-lifecycle.cascade.js | wc -l
20
$ grep -n -i 'checkout\|lane\|nested\|A\.checkout\|args\.checkout' mass-patch-fanout.js | wc -l
180
```

cascade.js itself carries no lane/fan-out vocabulary at all (no `lane`, no
`A.checkout`/`args.checkout` literal) — its only relevant surface is generic:
`ARGS.repoToplevel` (existing, pre-1.3.0 too) and the new Teardown prompt's
explicit exclusion clause: *"Anything else (the checkout itself, phase
worktrees, operator worktrees, **nested lane worktrees without the wf_
prefix**) is out of scope: never list it, never touch it."* That one phrase
is the only place cascade.js acknowledges nested-lane topology exists, and it
exists purely as a defensive exclusion — cascade.js has no idea it might be
running as a fan-out lane; it just refuses to touch anything that isn't its
own `wf_`-prefixed scratch worktree.

mass-patch-fanout.js is where the real machinery lives: it is an
**EXPERIMENTAL overlay that composes td-phase-lifecycle.cascade.js as nested
child workflows**, one lane per triage-validated ticket (K-wide, default 3,
hard cap 4). Its `Provision` phase creates one nested git worktree per lane —
`checkout/.claude/worktrees/wave-<ticket-slug>` — directly under the
conductor's own launch checkout (`A.checkout`), all rooted at one
machine-read parent tip captured exactly once (never re-read, to avoid lanes
anchoring to different physical tips). `laneArgs` (GUARD-PIN'd, pure) then
composes each lane's cascade-engine `args`:

```js
return {
  repoToplevel: common.checkout,   // the SHARED conductor checkout, NOT the lane's own worktree
  phase: { ..., startTip: startTip, ... },
  ...
}
```

**Key finding, worth surfacing rather than silently noting:** `repoToplevel`
for every lane is the **shared** top-level checkout, not that lane's own
nested worktree (`lane.worktree`) — deliberately, per the comment, so the
advisor-lock semaphore serializes per-repo across all parallel lanes rather
than one lock per lane. Lane WORK happens in the nested worktree (agents `cd`
there per an explicit isolation instruction in every lane prompt), but
`ARGS.repoToplevel` — and therefore the new Teardown phase's `Checkout:`
target — resolves to the SAME shared path for every lane in the wave. Since
`git worktree list --porcelain` is repo-global regardless of which nested
worktree you run it from, K lanes each reaching Teardown around the same time
would all list-and-filter the SAME `wf_*` worktree set from the SAME shared
checkout, concurrently. The Teardown prompt's own filter (dirty-check before
remove, remove without `--force`) makes an actual double-removal race
unlikely — a second `git worktree remove` on an already-removed path just
fails and gets reported as `kept` with the refusal text — but this is exactly
the kind of same-axis concurrent-write pattern axiom 4 (Ownership) asks to be
named aloud rather than assumed benign: nothing in either file explicitly
serializes concurrent Teardown calls across sibling lanes the way
`advisorLockPath` explicitly does for the advisor consult. Flagged here as a
delta-reading observation, not fixed — out of scope for this port (the task
is the toolchain/publishMode port of cascade.js alone, not a fanout.js
change), but worth a K0 entry if the operator wants to file it.

The `startTip` chain-of-custody wiring is the one place fanout.js reaches
into a *new* 1.3.0 capability rather than just tolerating it: `laneArgs`
throws before composing a lane's args at all if `startTip` isn't a full
40-hex SHA, specifically because (per its own comment) that value seeds
`seedEngineTrackedTip` inside the nested cascade run. Nothing else in
fanout.js reads or reacts to the new Teardown phase, `ancestryCheck` field,
or `TEARDOWN` schema directly — those are internal to each nested cascade
run's own lifecycle and only surface to fanout.js as part of that lane's
final `laneResult` (an opaque object fanout.js stores and replays on resume,
per `validateResumeIntake`'s comment, without inspecting its shape beyond
`{ verdict }`).
