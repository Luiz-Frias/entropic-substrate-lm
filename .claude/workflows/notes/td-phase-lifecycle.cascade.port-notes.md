# td-phase-lifecycle.cascade.js port — notes (2026-09-17, rebased, +policy.teardown)

**Rebase notice:** this port was first done against a stale 0.1.0 snapshot
(1285 lines) before the coordinator flagged that `/reload-plugins` had
installed helmsman 1.3.0 mid-session. This document supersedes that earlier
version entirely and is written against the authoritative SOURCE:
`/Users/luizfrias/.claude/plugins/cache/helmsman-marketplace/helmsman/1.3.0/.claude/workflows/td-phase-lifecycle.cascade.js`
(1654 lines, verified). The 0.1.0-to-1.3.0 delta reading lives in the sibling
file `cascade-1.3.0-delta.md` in this same directory.

Port written to (read-only on the repo; nothing written outside this tree):
`.claude/workflows/.claude/workflows/td-phase-lifecycle.cascade.js` where
`S=(staging dir)`.

Reference pattern followed: `/Users/luizfrias/CursorAI/entropic-substrate-lm/.claude/workflows/td-phase-lifecycle.js`
(project's own non-cascade port) for the `P.exitRunner` / `RUNNER = P.exitRunner`
policy-and-binding pattern. Its `installCmd` was **not** policy-driven there —
it inline-hardcoded `uv sync --all-groups`; this port makes `installCmd` a
first-class policy (`P.installCmd`) per the task's explicit instruction, a
deliberate divergence from that reference file's own choice. Its
`ISOLATION_NOTE` also hardcodes `.claude/worktrees/observability-sprint` — the
task warned against repeating that, and the cascade source's own
`ISOLATION_NOTE` was already generic ("the checkout this phase was launched
in"), so it was left untouched.

## IMPORTANT METHODOLOGY CORRECTION: `node --check` is not a reliable gate for this file

While re-verifying the final `Object.assign(...)` rewrite in the terminal
return, a direct visual read of the file's tail caught a real defect I had
introduced: the rewrite opened `return Object.assign(` but the trailing
`}` — left over from the pre-rewrite `return { ... }` object literal — was
never swapped for the `)` that call needed. That is an unambiguous,
`Unexpected token '}'` syntax error, confirmed in isolation:

```text
$ node --check /tmp/isolated_repro.js
SyntaxError: Unexpected token '}'
```

**But `node --check` on the FULL file, with that exact defect present,
reported exit 0 — no error, silently.** I do not have a confirmed root cause
(a plausible one: this file relies on a bare top-level `return`, which is
only legal because Node's `--check` / CJS-wrapSafe path is lenient in a way
that stops short of a complete grammar validation once it sees that
construct — but I did not verify that hypothesis further, so it is not
asserted as fact). What I DID verify empirically, with an actual
discriminating test, is that `node --check` cannot be trusted alone here:

```text
$ node /tmp/lex_check.js <broken-file>      # hand-rolled bracket-depth lexer
NEGATIVE BRACE at line 1753                  # correctly catches it

$ node --check <broken-file>
(exit 0 — silently wrong)

$ node /tmp/proper_check3.js <broken-file>   # code wrapped in
                                              # `(async function(...){ <code> })`
                                              # via vm.Script, export stripped,
                                              # so the wrapper matches how the
                                              # harness actually loads this file
                                              # (per its own header comment)
PARSE ERROR: Unexpected token '}'  — line 1754
                                              # correctly catches it
```

Both the hand-rolled lexer (tracks `(){}[]` depth while correctly skipping
over string/template/comment content) and the function-wrapped``vm.Script`parse catch the defect that bare`node --check`misses. I fixed the defect
(added the missing`)`), then re-ran **all three** checks against the
corrected file and against the pristine, untouched 1.3.0 source as a control
(the control must come back clean on all three, or the checks themselves are
suspect):

```text
$ node --check td-phase-lifecycle.cascade.js                    # port
exit 0
$ node --check <pristine 1.3.0 source>                          # control
exit 0

$ node /tmp/lex_check.js td-phase-lifecycle.cascade.js           # port
EOF state: code / depthParen: 0 / depthBrace: 0 / depthBracket: 0
$ node /tmp/lex_check.js <pristine 1.3.0 source>                 # control
EOF state: code / depthParen: 0 / depthBrace: 0 / depthBracket: 0

$ node /tmp/proper_check3.js td-phase-lifecycle.cascade.js       # port
PARSE OK
$ node /tmp/proper_check3.js <pristine 1.3.0 source>             # control
PARSE OK
```

All three agree now, on both the port and the control. **The syntax claim
below rests on the wrapped-`vm.Script` parse and the bracket-depth lexer, not
on `node --check` alone** — `node --check`'s result is reported for
completeness (the task asked for it explicitly) but is not, on its own,
sufficient evidence for this file, and I am flagging that rather than quietly
relying on a check I now know can false-green.

## 1. Toolchain port — every changed line, old -> new (line numbers are 1.3.0-source -> port)

| Source:line                                                           | Port:line                  | Old                                                                                         | New                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 63-68 (`meta.description`)                                            | 63-68                      | ends `"...stay with the operator."`                                                         | +1 sentence: `policy.publishMode "submit" (default) runs the normal gt-submit/remote-CR Publish and Validate stages; "defer" skips gt submit and PR-state validation, certifying the tip and exit gate locally instead and returning CERTIFIED in place of SUBMITTED.` |
| 65 (`meta.whenToUse`)                                                 | 65                         | ends `"...owns all merges + Linear writes."`                                                | +1 sentence: `Set policy.publishMode "defer" when there is no PR to submit against (e.g. a local-only or pre-remote phase); leave it "submit" (default) for the normal gt-submit flow.`                                                                                |
| 408 (`P` defaults, one line)                                          | 413-435                    | `{ maxImplLoops: 3, ..., maxPackets: 6 },` (9 keys)                                         | same 9 keys, multi-line, **+3 new keys**: `exitRunner: 'uv run python scripts/run_exit_gate.py', installCmd: 'uv sync --all-groups', publishMode: 'submit'`                                                                                                            |
| (new, after 408)                                                      | 433-434                    | —                                                                                           | `const RUNNER = P.exitRunner` / `const INSTALL = P.installCmd`                                                                                                                                                                                                         |
| 760 (comment)                                                         | 784                        | `// exitGate is the runner's JSON relayed VERBATIM (bun src/gate/adapters/run-exit-gate.ts` | `// exitGate is the runner's JSON relayed VERBATIM (the P.exitRunner script`                                                                                                                                                                                           |
| 825 (`implGateRetryFeedback`, inside GUARD-PIN — see exception below) | 849                        | ``'re-run the deterministic runner (`bun src/gate/adapters/run-exit-gate.ts --cmd ' + ...`` | ``'re-run the deterministic runner (`' + RUNNER + ' --cmd ' + ...``                                                                                                                                                                                                    |
| 1303 (impl step 5)                                                    | 1327                       | ``'5. EXIT GATE ... `bun src/gate/adapters/run-exit-gate.ts --cmd ' + ...``                 | ``'5. EXIT GATE ... `' + RUNNER + ' --cmd ' + ...``                                                                                                                                                                                                                    |
| 1372 (`ISOLATION_NOTE`)                                               | 1412                       | `...then bun install (and any additional...`                                                | ``...then `' + INSTALL + '` (and any additional...``                                                                                                                                                                                                                   |
| 1378 (L2 gate-checker prompt)                                         | 1418                       | ``'- RE-RUN ... via `bun src/gate/adapters/run-exit-gate.ts --cmd ' + ...``                 | ``'- RE-RUN ... via `' + RUNNER + ' --cmd ' + ...``                                                                                                                                                                                                                    |
| 1459 (Review Round C)                                                 | 1499                       | `run bun install there if the gates need it;`                                               | ``run `' + INSTALL + '` there if the gates need it;``                                                                                                                                                                                                                  |
| 1570 (Validate step 2, submit-mode branch)                            | folded into the §2 rewrite | `` `bun src/gate/adapters/run-exit-gate.ts --cmd ' + ... ``                                 | `` `' + RUNNER + ' --cmd ' + ... ``                                                                                                                                                                                                                                    |

All 7 sites match the coordinator's cited line numbers on the 1.3.0 source
exactly (verified by direct grep before editing).

**GUARD-PIN exception (unchanged from the prior port, re-verified against
1.3.0):** one substitution (source line 825) falls inside `GUARD-PIN BEGIN
implGateRetryFeedback` / `END`. Instruction #3 (invariants) says GUARD-PIN
blocks stay byte-identical; instruction #1 (toolchain port) separately and
explicitly names this line as a required substitution site. Resolution: apply
only the minimal, single-token substitution instruction #1 names, nothing
else inside that block. Verified below that this is the *only* diff inside
any of the file's 16 PARITY-PIN/GUARD-PIN blocks (1.3.0 added 7 new GUARD-PIN
blocks over the old snapshot — `packetGitCeremony`, `tddBundlingNote`,
`composePacketWithCeremony`, `implementPacketAncestryInstruction`,
`composeImplementPacket`, `postPacketAncestryHaltReason`,
`seedEngineTrackedTip` — none of which reference the toolchain and all of
which diff clean at zero bytes).

`grep -n bun` result (task asked to confirm this is 0):

```text
$ grep -n bun td-phase-lifecycle.cascade.js
495:    'W — WORK PACKAGE (your report IS the evidence bundle; ...)'
558:// block (ceremony + ancestry check + TDD-bundling note) syncs to it per the
592:// packet-executor's ONLY channel to the TDD-bundling rule (it never receives
```

**Not 0 — reported honestly.** Line 495 is the pre-existing substring `bun`
inside **"bundle"** (`evidence bundle`), inside the byte-pinned `workPackets`
GUARD-PIN block. Lines 558 and 592 are new in 1.3.0 — the substring `bun`
inside **"TDD-bundling"** — both inside byte-pinned blocks
(`seedEngineTrackedTip`'s preceding comment block and
`implementPacketAncestryInstruction`'s GUARD-PIN respectively; specifically
part of the un-pinned prose directly above/around those blocks describing
`TDD_BUNDLING_NOTE`, which itself is a pinned block name — none of these are
touchable without violating the byte-identical invariant). **Zero real `bun
<command>` toolchain invocations remain** — verified separately:

```text
$ grep -n -E '\bbun (install|src/)' td-phase-lifecycle.cascade.js | wc -l
0
```

## 2. New policy `publishMode: 'submit' | 'defer'`

Default `'submit'` reproduces the original behaviour exactly — the
`'submit'`-branch prompt text for Publish and Validate is unchanged, just
moved one level deeper inside a ternary.

Under `'defer'`:

| Stage                                                                                 | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Publish                                                                               | Schema switches `PUB` -> new `PUB_RECORD` (`{tip, exitGate, branch, parent, notes?}`, required `[tip, exitGate, branch, parent]`). Prompt drops `gt submit`, PR-body-fill, `@coderabbitai` trigger, and CR-poll steps entirely; replaces them with (1) verify branch + clean tree via `git rev-parse HEAD` + `git status --porcelain`, (2) run the exit gate via `RUNNER` and relay its JSON verbatim. Hard rules add `NEVER gt submit, never push`.         |
| Validate                                                                              | Same `VAL` schema (unchanged — it never referenced `pr`). Step 1 (tip resolution) drops the `gh pr view --json headRefOid` cross-check and instead cross-checks `git rev-parse HEAD` against the certified record's `pub.tip`. Step 5 ("PR state") becomes `N/A — policy.publishMode is "defer" ... record SKIP`. Steps 2/3/4/6/7/8/10 are untouched. Step 9 (ruling artifact duties) keeps the doc/ticket check but drops "PR body" from its parenthetical. |
| exit-gate block                                                                       | `exitGateBlockReason(val.exitGate, val.verifiedTip, G.exit)` runs **unconditionally**, same call, same location, right before Teardown/the terminal return — mode-independent, per the task's requirement.                                                                                                                                                                                                                                                   |
| **Teardown (new in 1.3.0, not part of the original 3-change scope — see note below)** | Runs on **either** success terminal now (SUBMITTED or CERTIFIED), not "SUBMITTED ONLY" as upstream 1.3.0 has it. Its own prompt text says `is ' + (P.publishMode === 'defer' ? 'CERTIFIED' : 'SUBMITTED') + ' and validated'` instead of a hardcoded `SUBMITTED`. Mechanics (wf_-prefix-only scoping, no `--force`, never touching the verdict) are byte-identical to upstream.                                                                              |
| Terminal result                                                                       | `verdict: P.publishMode === 'defer' ? 'CERTIFIED' : 'SUBMITTED'`. Same `Object.assign` attaches `record: pub` only when `publishMode === 'defer'`. `teardown: teardownReport` is retained unconditionally (already present in 1.3.0, orthogonal to publishMode).                                                                                                                                                                                             |

**Judgment call worth naming aloud (Legibility axiom):** 1.3.0 added a
Teardown phase that did not exist in the file version the original 3-change
instruction was written against. Its own comment says it runs "on the
SUBMITTED path ONLY", because SUBMITTED was the only success terminal that
existed when it was written. Teardown's actual job — releasing the engine's
own `wf_*` reviewer-isolation scratch worktrees — has nothing to do with
whether a PR exists; gating it on `verdict === 'SUBMITTED'` literally would
mean a `defer`-mode phase leaks its wf_* worktrees forever (the exact
30-leftover-worktree problem Teardown itself exists to close, reopened for
every deferred phase). I extended it to run on **either** success terminal
rather than leave that leak, and documented the change with a `PORT NOTE`
comment in the file itself. This is the one place this port goes beyond a
literal, mechanical reproduction of "apply exactly the same three changes" —
flagged here rather than left silent, per the collaboration protocol's
Legibility axiom (non-obvious move states its WHY before executing). If the
operator wants Teardown gated strictly on `SUBMITTED` only (matching upstream
literally, accepting the worktree leak on `defer` phases as out of scope),
that's a one-line revert of the `P.publishMode === 'defer' ? 'CERTIFIED' :
'SUBMITTED'` conditional in the Teardown prompt's status line.

## 3. Fourth and final change: `policy.teardown: true | false`

Coordinator ruling on the ownership hazard named above (§2): the fix is not
to revert the CERTIFIED-runs-Teardown extension (kept — Teardown is correctly
orthogonal to publish mode) but to give the conductor a knob to skip Phase 7
entirely when this phase is composed as a mass-patch-fanout lane, where the
hazard actually lives.

Default `true` reproduces the behaviour in §2 exactly (Teardown runs on
either success terminal, unconditionally). `false` skips Phase 7 outright:

- `phase('Teardown')` still fires (for the phase-list/telemetry contract),
  but the `if (P.teardown === false)` branch never calls `agent(...)` —
  no Teardown agent is dispatched at all.
- It logs exactly the string the coordinator specified: `'phase ' + PH.id +
  " teardown skipped: policy.teardown=false (fan-out lane; the wave zip
  releases wf_* worktrees once every lane is terminal)"`.
- `teardownReport` is set to `{ removed: [], kept: [], branchesDeleted: [],
  remaining: -1, notes: 'skipped by policy' }` — the exact same TEARDOWN
  schema shape a live run produces, so the terminal result's `teardown` field
  is present and shaped identically whether Teardown ran or was skipped; no
  downstream consumer needs a branch on whether the policy fired.
- The terminal return's `teardown: teardownReport` line is otherwise
  untouched — it reads whichever branch set `teardownReport`.

`meta.whenToUse` gained one sentence: `Set policy.teardown false to skip
Phase 7 entirely when this phase is composed as a mass-patch-fanout lane (its
wf_* worktrees live under the wave's shared checkout, so the wave's Zip stage
should release them once, not each lane concurrently); leave it true
(default) otherwise.` The `P` defaults object gained `teardown: true` (10th
key) with a comment explaining the hazard and why the wave's Zip stage — not
per-lane Teardown — is the correct single owner of that release.

This closes the exact race named in the delta reading (`cascade-1.3.0-delta.md`):
`laneArgs` sets every mass-patch-fanout lane's `repoToplevel` to the wave's
SHARED conductor checkout (not the lane's own nested worktree), so K parallel
lanes each running Teardown would list-and-filter the SAME `wf_*` worktree
set from the SAME shared checkout concurrently — nothing in either file
previously serialized that. `policy.teardown: false` lets the conductor keep
that release as a single, correctly-owned step in the wave's Zip stage
instead.

## 5. Invariant checks (all against the 1.3.0 source, not the stale 0.1.0 snapshot; re-run after adding policy.teardown)

| #  | Invariant                                                                                                                                                                                                                                                                                                                                                                                                                                  | Command                                                                                                                      | Result                                                                                                                                                                                                                     |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | `export const meta` is the literal first top-level statement                                                                                                                                                                                                                                                                                                                                                                               | `grep -n '^export const meta'`                                                                                               | Line 59, first non-comment line — unchanged position                                                                                                                                                                       |
| 2  | No import statements anywhere                                                                                                                                                                                                                                                                                                                                                                                                              | `grep -n "^import \|from '"`                                                                                                 | 0 hits                                                                                                                                                                                                                     |
| 3  | No fs/process/require/Date.now/Math.random                                                                                                                                                                                                                                                                                                                                                                                                 | `grep -n "require(\|process\\.\|Date\\.now(\|Math\\.random("`                                                                | 0 hits                                                                                                                                                                                                                     |
| 4  | All 16 PARITY-PIN/GUARD-PIN blocks (`GATE_RESOLUTION_MIRROR_TEXT`, `resolveGatesOrThrow`, `advisorLockPath`, `liveGateConfigReason`, `deriveCascadeMode`, `fixerTierForRound`, `workPackets`, `packetGitCeremony`, `tddBundlingNote`, `composePacketWithCeremony`, `implementPacketAncestryInstruction`, `composeImplementPacket`, `postPacketAncestryHaltReason`, `seedEngineTrackedTip`, `exitGateBlockReason`, `implGateRetryFeedback`) | extract by BEGIN/END marker, `diff` against 1.3.0 source, one per marker                                                     | 15 of 16 `IDENTICAL`; `implGateRetryFeedback` differs by exactly the one `RUNNER` substitution (task-mandated exception, §1) — unaffected by the policy.teardown addition, which lives entirely outside every pinned block |
| 5  | args boundary parse unchanged                                                                                                                                                                                                                                                                                                                                                                                                              | extract `let ARGS = args` ... closing `}`, `diff`                                                                            | `ARGS BOUNDARY IDENTICAL`                                                                                                                                                                                                  |
| 6  | `ISOLATION_NOTE` stays generic, no hardcoded sprint path                                                                                                                                                                                                                                                                                                                                                                                   | `grep -n "the checkout this phase was launched in"` (must hit) / `grep -n "observability-sprint\\\|sprint path"` (must miss) | 1 hit / 0 hits                                                                                                                                                                                                             |
| 7  | Real `bun <cmd>` toolchain invocations = 0                                                                                                                                                                                                                                                                                                                                                                                                 | `grep -n -E '\\bbun (install\|src/)' \| wc -l`                                                                               | `0`                                                                                                                                                                                                                        |
| 8  | Literal `grep -n bun` (task's check as literally written)                                                                                                                                                                                                                                                                                                                                                                                  | `grep -n bun`                                                                                                                | **3 hits, not 0** — all inside "bundle"/"TDD-bundling", all inside byte-pinned blocks (§1)                                                                                                                                 |
| 9  | Gate-resolution door untouched                                                                                                                                                                                                                                                                                                                                                                                                             | zero-byte diff on `GATE_RESOLUTION_MIRROR_TEXT` + `resolveGatesOrThrow` (row 4)                                              | confirmed; no hardcoded gate defaults added anywhere in this port                                                                                                                                                          |
| 10 | Terminal result shape unconditional on policy.teardown                                                                                                                                                                                                                                                                                                                                                                                     | inspect both branches of `if (P.teardown === false) { ... } else { ... }`                                                    | both set `teardownReport` to a TEARDOWN-schema-shaped object before the shared `teardown: teardownReport` line in the terminal return; no downstream field is added/removed by the policy                                  |

## 6. Diff stat vs the 1.3.0 source (final, includes policy.teardown)

```text
$ diff <1.3.0-source> <port> | grep -c '^<'
81
$ diff <1.3.0-source> <port> | grep -c '^>'
209
$ diff -u <1.3.0-source> <port> | grep -c '^@@'
10
```

Line counts: 1.3.0 source 1654 lines -> port 1782 lines (+128 net, up from
+99 before this change — the policy.teardown addition contributed the other
+29: the `P` defaults comment + `teardown: true` key, the `meta.whenToUse`
sentence, and the `if (P.teardown === false) {...} else {...}` wrapper around
Phase 7 with its skip-path log line and TEARDOWN-shaped stand-in result).
10 unified-diff hunks (down from 11 in the pre-policy.teardown revision, not
because a change was dropped but because the Publish/Validate rewrite hunk
and the Teardown/terminal-return rewrite hunk are contiguous enough in the
file that `diff -u`'s default 3-line context window merged them into one
hunk once the Teardown section grew) — every touched region is still exactly
one of the twelve intended edit sites: `meta` (1: now carries both the
publishMode and teardown sentences), `P` defaults + `RUNNER`/`INSTALL`
bindings + `teardown: true` (1), `XGATE` comment (1), `implGateRetryFeedback`
(1), `PUB_RECORD` schema addition (1), impl-step-5 `RUNNER` sub (1),
`ISOLATION_NOTE` `INSTALL` sub (1), L2 gate-checker `RUNNER` sub (1), Review
Round C `INSTALL` sub (1), and the merged Publish/Validate/Teardown/
terminal-return rewrite (1 hunk covering all of: the publishMode ternaries,
the publishMode-aware Teardown status line, the new `policy.teardown`
if/else wrapper, and the `Object.assign`-based terminal return). No hunk
touches anything outside these named sites.

## 7. Syntax check

```text
$ node --version
v26.7.0
$ node --check td-phase-lifecycle.cascade.js
exit 0
```

Node was already on PATH (Homebrew) and `mise exec -- node --version`
resolves to the same `v26.7.0`. **As established in the methodology-correction
section above, this result alone is not treated as sufficient** — it is
reported because the task asked for it explicitly, but the syntax claim for
this file rests on the two additional, discriminating checks, **re-run just
now against the file as it stands after adding `policy.teardown`** (both the
port and an untouched-1.3.0-source control, per the coordinator's explicit
instruction to re-run the vm.Script wrapped-parse check, never `node --check`
alone):

```text
$ node --check td-phase-lifecycle.cascade.js                # not sufficient
exit 0                                                       # alone; reported
                                                              # for completeness

$ node /tmp/proper_check3.js td-phase-lifecycle.cascade.js   # export stripped,
                                                              # wrapped in an
                                                              # async IIFE
                                                              # matching the
                                                              # harness's own
                                                              # documented
                                                              # execution model
PARSE OK

$ node /tmp/lex_check.js td-phase-lifecycle.cascade.js       # hand-rolled
                                                              # bracket-depth
                                                              # lexer, comment/
                                                              # string-aware
EOF state: code / depthParen: 0 / depthBrace: 0 / depthBracket: 0
                                                              # (1782 total
                                                              # lines, up from
                                                              # 1754 before
                                                              # policy.teardown)
```

Both scripts are throwaway, written for this verification and not part of
the deliverable; their logic is reproduced in full in the methodology section
above. All three checks agree, and the two discriminating ones — the only
ones actually trusted per the methodology correction — both come back clean
on the file as it stands with all four changes applied.

## What was deliberately NOT touched

- The gate-resolution door (`resolveGatePrecedence`, `resolveGatesOrThrow`,
  `KNOWN_TOP_LEVEL_KEYS`, `profileMalformedReason`, the 1.3.0 fail-closed
  hardening around explicit-gates completeness and `advisorLockPath`'s
  no-more-silent-'repo'-default): zero-byte diff, confirmed above. No
  hardcoded gate defaults were added anywhere.
- The packet git-ceremony + ancestry-chain hardening (FOR-1302: engine-appended
  ceremony block, ancestry check, `engineTrackedTip`, `seedEngineTrackedTip`,
  `args.phase.startTip`): all new in 1.3.0, all untouched by this port.
- `RECON`, `IMPL` (its new `ancestryCheck` field included), `L2A`/`L2B`
  schemas, the fixer ladder, `deriveCascadeMode`, `workPackets`, the
  review-round prompts (A/B, and Round C except its one `INSTALL`
  substitution), `LocalCR` stage: untouched.
- `TEARDOWN` schema and Teardown's mechanical steps (wf_-prefix scoping, no
  `--force`, dirty-tree handling): untouched apart from the one
  publishMode-aware status-line change documented in §2.
- Repo profile reading / `git` toplevel resolution: still entirely the
  invoking session's job via `args.profile` / `args.repoToplevel` — this
  script never gained fs/child_process access.
