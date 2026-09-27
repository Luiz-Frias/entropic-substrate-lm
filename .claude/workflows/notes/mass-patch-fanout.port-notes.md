# mass-patch-fanout.js — port notes (deliverable 3, 2026-09-17)

Source: `/Users/luizfrias/.claude/plugins/cache/helmsman-marketplace/helmsman/1.3.0/.claude/workflows/mass-patch-fanout.js` (717 lines)
Port: `.claude/workflows/.claude/workflows/mass-patch-fanout.js` (755 lines, after the §7 follow-up below — 746 lines at initial delivery) — never written into the repo; repo was read-only throughout.
Sibling port referenced by name in change 1: `.claude/workflows/.claude/workflows/td-phase-lifecycle.cascade.js` (already present in this deliverables tree — confirmed it implements `lanePolicy`'s four keys: `publishMode`, `exitRunner`, `installCmd`, `implPackets`, and that its `CERTIFIED` verdict carries `record: {tip, exitGate, branch, parent}` exactly as assumed below).

## 1. Every changed line (old → new)

GUARD-PIN blocks `normalizeSeatPins`, `kBound`, `zipOrderRationale`, `validateZipPlan`, `validateProvisionWorktrees`, `validateZipResult`, `validateResumeIntake` are **byte-identical** to source (verified below) — only `laneArgs` (the one authorized exception) and plain (non-guarded) orchestration code changed.

| #  | Old line(s)                         | Old text                                                                                                                                                                               | New line(s) | New text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Change                                                                                                                                                                                                                           |
| -- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | 5                                   | `...seatPins, resume ({plan,...}). ... conductor reviews the wave report, owns the zip force-push authorization and all merges.`                                                       | 5           | `...seatPins, lanePolicy (optional object merged into every lane's cascade policy ... this repo launches with lanePolicy: {publishMode:'defer', exitRunner:'uv run python scripts/run_exit_gate.py', installCmd:'uv sync --all-groups', implPackets:false} ... The conductor reviews the wave report and owns the zip's single \`just ship-it\` publish and all merges.`                                                                                                                                                                                                                                                         | meta.whenToUse: documents new `args.lanePolicy`, this repo's launch value, and replaces stale force-push-authorization language now that the zip force-pushes nothing                                                            |
| 2  | 10                                  | `{ title: 'Zip', detail: 'least-to-most-risk restack, per-tip exit gate, stack submitted as drafts' }`                                                                                 | 10          | `{ title: 'Zip', detail: 'least-to-most-risk restack, per-tip exit gate, one ship-it publish from the stack top' }`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | meta.phases Zip detail updated per instruction 3                                                                                                                                                                                 |
| 3  | 215 (inside `laneArgs` GUARD-PIN)   | `policy: { submitMode: 'draft', recon: true, implPackets: true, maxPackets: 2 },`                                                                                                      | 215–220     | 5 comment lines + `policy: Object.assign({ submitMode: 'draft', recon: true, implPackets: true, maxPackets: 2 }, common.lanePolicy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                  |
| 4  | 612 (Provision prompt)              | `... \`git worktree add ...\` (worktree path NESTED under the checkout), then run the repo's worktree ceremony ...`                                                                    | 617         | `... \`git worktree add ...\` (worktree path NESTED under the checkout); then copy the gitignored \`.env\` and \`justfile.local\` ... then run \`uv sync --all-groups\` in the lane; then run the repo's worktree ceremony ...`                                                                                                                                                                                                                                                                                                                                                                                                  | Change 4: Provision extras, inserted between the worktree-add and the ceremony clause                                                                                                                                            |
| 5  | 635–636 (comment, not guard-pinned) | `// SKILL.md "Resume" section (must mirror ... returns — {verdict, publish: {pr}, ...} — so laneSummary below composes it identically ...)`                                            | 640–643     | documents both the `SUBMITTED`/`publish.pr` shape and the new `CERTIFIED`/`record` shape                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Change 2 (doc only)                                                                                                                                                                                                              |
| 6  | (new, after 657)                    | —                                                                                                                                                                                      | 665         | `lanePolicy: A.lanePolicy,`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Change 1: threads `args.lanePolicy` into the `common` object at the Lanes call site                                                                                                                                              |
| 7  | 666                                 | `return { issue: lane.issue, branch: lane.branch, verdict: r ? r.verdict \|\| 'UNKNOWN' : 'DIED', pr: r && r.publish ? r.publish.pr : null }`                                          | 672–685     | expanded to compute `verdict` once, add `tip: verdict === 'CERTIFIED' && r && r.record ? r.record.tip : null`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Change 2: laneSummary carries `tip` from `record` on CERTIFIED                                                                                                                                                                   |
| 8  | 669                                 | `const unclean = laneSummary.filter((s) => s.verdict !== 'SUBMITTED')`                                                                                                                 | 688         | `const unclean = laneSummary.filter((s) => s.verdict !== 'SUBMITTED' && s.verdict !== 'CERTIFIED')`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Change 2: CERTIFIED is a clean lane                                                                                                                                                                                              |
| 9  | 677                                 | `decision: 'lane(s) not SUBMITTED: ' + ...`                                                                                                                                            | 696         | `decision: 'lane(s) not SUBMITTED/CERTIFIED: ' + ...`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Change 2: DECISION-NEEDED wording kept accurate                                                                                                                                                                                  |
| 10 | (new, before 694)                   | —                                                                                                                                                                                      | 713–721     | comment + `const ZIP_EXIT_RUNNER = (A.lanePolicy && typeof A.lanePolicy.exitRunner === 'string' && A.lanePolicy.exitRunner) \|\| 'bun src/gate/adapters/run-exit-gate.ts'`                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Change 3: exit-gate runner resolved from `args.lanePolicy.exitRunner`, falling back to the source's literal runner string                                                                                                        |
| 11 | 695                                 | Zip agent prompt: `gt track`/`gt move --onto` reparent, `bun .../run-exit-gate.ts`, `gt submit --stack --draft --no-interactive --force` (force-push framed as standing authorization) | 723         | Zip agent prompt: `git status --porcelain` clean-check + `git worktree remove` (unchanged step a); `gt restack` / `git rebase --onto` per `plan.zipOrder`, abort + DECISION on ANY conflict, **never hand-resolved** (step b); offline gate + `ZIP_EXIT_RUNNER` per restacked tip, 40-hex tip recorded from `git rev-parse` (step c); **one** `just ship-it` from the stack-top branch, never `gt submit`, never `--force`, never a bare push, DECISION quoting the failing gate on failure (step d); per-branch `gh pr view <branch> --json headRefOid -q .headRefOid` vs `git rev-parse <branch>`, both values quoted (step e) | Change 3: full prompt rewrite. Schema `ZIP`, `validateZipResult`, the `!zip \|\| zip.status !== 'ZIPPED'` DECISION handling, and the final `WAVE-SUBMITTED` return shape are byte-identical to source — untouched, as instructed |
| 12 | 696–698 (comment)                   | `// Fable at low effort: ... (conflict => fail loud, per-tip re-gates) — strong model ...`                                                                                             | 725–727     | same + `, one ship-it publish`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | cosmetic, keeps the model-choice rationale accurate                                                                                                                                                                              |

## 2. A documented deviation from the literal instruction text (flagging for your review)

Instruction 3, step (b), read literally, says: *"...hand-resolve conflicts or STOP with DECISION (a conflict falsifies edge-disjointness)."* Taken at face value this offers hand-resolving a rebase conflict as an option.

I did **not** implement that option. The source file (line 695, this port's un-rewritten predecessor), the pattern card (`mass-patch-fanout.md`: *"the zip agent never resolves a conflict by hand"*), and the SKILL.md ("a rebase conflict during the zip falsifies the disjointness claim and fails the wave; the zip agent never resolves a conflict by hand") are unanimous and explicit on this point, and it collides directly with two of your NON-NEGOTIABLE master rules: *"No quick fixes / workarounds / patches: solve for root cause"* and *"STOP and ask > silent fallback on unexpected conditions."* A rebase conflict here is exactly a falsified precondition (the planner's edge-disjointness claim) — hand-patching it in an unattended agent step is the textbook silent-workaround this rule exists to block, and it would silently mask a Plan-phase defect rather than surfacing it.

I read the instruction's phrasing as a drafting artifact (a dropped "never") rather than a deliberate policy reversal, and ported the **fail-loud, never-hand-resolve** behavior forward unchanged in substance (only the reparent mechanism changed, from `gt track`+`gt move --onto` to `gt restack`/`git rebase --onto`, per the instruction's explicit ask). If a hand-resolve path was actually intended, that is a one-line change to step (3) of the new Zip prompt (line 723) — please confirm before I'd make it, since it reverses a fail-loud invariant the rest of the file (and both upstream docs) treat as load-bearing.

## 3. Diff stat

```text
$ diff -u <source> <port> | diffstat
 mass-patch-fanout.js | 53 +++++++++++++++++++++++++++++++++++++++------------
 1 file changed, 41 insertions(+), 12 deletions(-)
```

717 → 746 lines (+29 net). 11 distinct edit sites (table above), all inside: `meta` (2 sites), `laneArgs` (1, the authorized GUARD-PIN exception), the Provision prompt (1), the Lanes phase's plain orchestration code (5: 1 comment + `lanePolicy` threading + laneSummary + unclean filter + decision string), and the Zip phase (2: new `ZIP_EXIT_RUNNER` const + prompt rewrite).

## 4. Invariant checks (commands + output)

```text
$ node --check .claude/workflows/.claude/workflows/mass-patch-fanout.js
(no output — exit 0)

$ head -c 40 .claude/workflows/.claude/workflows/mass-patch-fanout.js
export const meta = {
→ `export const meta = {` is line 1, first statement — unchanged.

$ grep -nE "require\(|^import |Date\.now|Math\.random|process\.|fs\.|from 'fs'|from \"fs\"" .claude/workflows/.claude/workflows/mass-patch-fanout.js
none found

$ grep -n "workflow(" .claude/workflows/.claude/workflows/mass-patch-fanout.js
652:      : workflow(
→ exactly one workflow() call site (the Lanes phase), same as source — nesting stays one level.
```

GUARD-PIN byte-identity check (extracted each `GUARD-PIN BEGIN X` … `GUARD-PIN END X` span from both files and diffed):

```text
normalizeSeatPins:        IDENTICAL
kBound:                   IDENTICAL
zipOrderRationale:        IDENTICAL
validateZipPlan:          IDENTICAL
validateProvisionWorktrees: IDENTICAL
validateZipResult:        IDENTICAL
validateResumeIntake:     IDENTICAL
laneArgs:                 DIFFERS (expected — the one authorized change, isolated to the `policy:` line + a 5-line comment)
```

`laneArgs` diff (the only GUARD-PIN delta):

```text
47c47,52
<     policy: { submitMode: 'draft', recon: true, implPackets: true, maxPackets: 2 },
---
>     // PORT (entropic-substrate-lm): lanePolicy passthrough — args.lanePolicy
>     // (threaded into `common` at the Lanes call site below) merges OVER these
>     // wave defaults so the repo's cascade port (publishMode 'defer', the
>     // Python exitRunner/installCmd, implPackets false) reaches every lane
>     // without hardcoding a repo-specific policy into this shared composer.
>     policy: Object.assign({ submitMode: 'draft', recon: true, implPackets: true, maxPackets: 2 }, common.lanePolicy || {}),
```

No changes to: the `ZIP` schema, `validateZipResult`, the `!zip || zip.status !== 'ZIPPED'` / `zipResultHalt` DECISION handling, or the final `WAVE-SUBMITTED` return object — confirmed by the unified diff (hunk `@@ -691,11 +710,21 @@` touches only the new `ZIP_EXIT_RUNNER` const, the agent prompt string, and one comment line; nothing below `if (!zip || zip.status !== 'ZIPPED')` differs).

House-rule cross-check (CLAUDE.md, this repo):

- Rule 1 (`just ship-it`, never bare `gt submit`) — the new Zip prompt step (5) runs exactly one `just ship-it` from the stack top and explicitly forbids `gt submit`, `--force`, and any other push; step (5) also STOPs with DECISION (quoting the failing gate) rather than retrying or relaxing the gate on failure, matching `just ship-it`'s own "no `ship-it-fast` bypass" posture.
- Rule 2 (never concurrent podman containers) — `just ship-it` invokes `container-test` itself (justfile: `ship-it: check container-test`), and the new Zip prompt runs it exactly once, sequentially, at the very end of the stack — no per-lane or per-tip container invocation was introduced, so this port cannot trigger the measured concurrent-MLflow-container hang. (Lanes run genuinely in parallel per the overlay's design, but under `lanePolicy.publishMode:'defer'` no lane touches `container-test`/MLflow at all — that only happens once, in Zip, serially.)

## 5. Launch args shape for this repo

```js
{
  tickets: [
    { id, title, issue, scope, acceptance, mode: 'arch' /* or 'normal' — planner still re-derives this */ },
    // ...
  ],
  parent: '<sprint-trunk-or-hardening-branch>',
  checkout: '<absolute path to the conductor's own worktree>',
  cascadeScriptPath: '<repo>/.claude/workflows/td-phase-lifecycle.cascade.js',
  gates: { offline: 'just check', exit: 'just pc-ci' },
  K: 3,
  lanePolicy: {
    publishMode: 'defer',
    exitRunner: 'uv run python scripts/run_exit_gate.py',
    installCmd: 'uv sync --all-groups',
    implPackets: false,
    // teardown: false is the OVERLAY'S OWN default (set inside laneArgs,
    // §7) and does not need to be repeated here — every lane launched by
    // this overlay gets teardown:false whether or not lanePolicy names it,
    // because K parallel lanes sharing one checkout make each lane's own
    // wf_* worktree sweep an ownership hazard (see §7). Listed here only
    // for visibility; omitting this key from a real launch is equivalent.
  },
  conductorRuling: '<standing per-wave context, optional>',
  seatPins: undefined,          // optional; defaults are plan=fable::low, provision=opus::medium, zip=opus::xhigh
  resume: undefined,            // optional; {plan, startTip, worktrees} to re-enter at Lanes/Zip
}
```

Notes on this shape:

- `tickets[].mode` is accepted on intake but the planner (Plan phase, unchanged) re-derives `mode` from the live tree regardless — curated `mode` is a hint, not a bypass, matching the source's existing contract.
- `gates.exit: 'just pc-ci'` is this repo's own gate vocabulary (`justfile` recipe, prek commit+push stages) — the overlay adds no gate vocabulary of its own (per SKILL.md) and passes `gates` through verbatim to both the cascade lanes and the Zip agent's per-tip re-certification.
- `lanePolicy.exitRunner` is consumed twice in this port: once per-lane, passed through to `td-phase-lifecycle.cascade.js` via the `policy` object (Change 1), and once at Zip, via the new `ZIP_EXIT_RUNNER` const (Change 3) — same string, same key, one source of truth in the launch args.
- `lanePolicy.implPackets: false` disables the cascade's normal-mode ManagerWorker packet ladder for lanes launched from this overlay; this is a repo choice recorded here, not something this port's changes require.
- Every lane certifies `CERTIFIED` (not `SUBMITTED`) under `publishMode: 'defer'` — the wave's *only* PR-equivalent artifact is the single `just ship-it` stack published once at Zip.
- `lanePolicy.teardown` defaults to `false` under this overlay (§7) regardless of what a wave's launch args pass — `laneArgs` bakes it into the default policy object itself, so every lane's own cascade-internal `wf_*` sweep is disabled and the ONE safe sweep happens once, serially, in the Zip prompt. A wave could still force `teardown: true` back on via `lanePolicy: { teardown: true }`, but doing so re-opens the ownership race whenever K > 1 — there is no known-safe reason to override it.

## 6. Open item carried from source (not touched, not asked to be)

FOR-1328's "other half" (a machine-produced per-tip exit-gate *certificate*, not just a bare SHA, in `zip.tips`) is still DECISION-GATED, NOT IMPLEMENTED in the source's `validateZipResult` comment (line 363–371, byte-identical in this port) — unchanged here since instruction 3 scoped the rewrite to the Zip *prompt* only, explicitly preserving `validateZipResult`. Flagging only because the new Zip prompt step (c) now names a *repo-specific* runner (`ZIP_EXIT_RUNNER`) whose JSON output would be the natural payload for that still-unimplemented certificate shape, if the operator ratifies it later.

## 7. Follow-up fix: `wf_*` worktree teardown ownership hazard (post-initial-delivery)

The cascade porter found a race: `laneArgs` sets every lane's `repoToplevel` to the SHARED conductor checkout (`common.checkout`), not the lane's own worktree (see the comment already on that field — "advisor consults serialize per-repo across parallel lanes"). The cascade child's own Phase 7 Teardown lists and removes every clean `wf_*` scratch worktree it finds under `repoToplevel`. With K lanes running truly in parallel (Phase 3: Lanes, `await parallel(...)`), each lane's Teardown independently sweeps the SAME shared checkout — lane A can remove lane B's still-active clean `wf_*` worktree mid-flight. This is exactly the "several nested worktrees coexisting" hazard class the file already documents for lane placement (laneArgs' isolation-invariant comment), one layer down, inside the cascade's own internal bookkeeping rather than the lanes themselves.

Fix, two parts, both inside the already-touched `laneArgs` GUARD-PIN block and the already-rewritten Zip prompt (no new edit sites, no new GUARD-PIN blocks):

| #  | Old line(s)                                                   | Old text                                                                                                                    | New line(s) | New text                                                                                                                                                                                                                               | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 13 | 220 (this port's own line, from §1 row 3 — inside `laneArgs`) | `policy: Object.assign({ submitMode: 'draft', recon: true, implPackets: true, maxPackets: 2 }, common.lanePolicy \|\| {}),` | 215–229     | + 9-line comment naming the hazard, `teardown: false` added to the default policy object: `policy: Object.assign({ submitMode: 'draft', recon: true, implPackets: true, maxPackets: 2, teardown: false }, common.lanePolicy \|\| {}),` | Disables each lane's own cascade-internal Phase 7 Teardown sweep by default (the cascade port honours `policy.teardown`); `args.lanePolicy.teardown` can still override per wave                                                                                                                                                                                                                                                                                                                |
| 14 | 731 (this port's own line, from §1 row 11 — Zip prompt)       | Steps numbered (1) lane-worktree-remove … (2) restack … (7) final report                                                    | 731–738     | New step (2) TEARDOWN SWEEP inserted between (1) and the old (2); everything after renumbers (2)→(3) … (7)→(8)                                                                                                                         | Zip runs ONE serial sweep, after every lane has finished: `git worktree list --porcelain` in `A.checkout`; candidates = paths containing `/.claude/worktrees/wf_` directly under the checkout; each CLEAN candidate removed with `git worktree remove` (no `--force`); dirty ones kept and reported by path; `git worktree prune`; removed/kept counts (+ kept paths) recorded in `evidence`. Placed BEFORE the restack step (3) and well before ship-it (6), per the coordinator's instruction |

Why the sweep belongs in Zip, not back in each lane: Zip is the one point in the wave's control flow that runs strictly after `await parallel(...)` has resolved for every lane (the `unclean` check at line 697 has already gated on it) — no lane is still executing, so listing and removing `wf_*` worktrees under the shared checkout is no longer a race. This mirrors the same "the conductor owns the zip" authority boundary the pattern card already assigns to this phase.

No schema change: the sweep's counts/paths ride in the existing free-text `evidence` field on the unchanged `ZIP` schema — `validateZipResult` and the `WAVE-SUBMITTED` return shape are untouched by this follow-up too.

### Updated diff stat (source → port, cumulative)

```text
$ diff -u <source> <port> | diffstat
 mass-patch-fanout.js | 62 +++++++++++++++++++++++++++++++++++++++++----------
 1 file changed, 50 insertions(+), 12 deletions(-)
```

717 → 755 lines (+38 net, up from +29 at initial delivery). `laneArgs` remains the only GUARD-PIN block that differs from source; all 7 others (`normalizeSeatPins`, `kBound`, `zipOrderRationale`, `validateZipPlan`, `validateProvisionWorktrees`, `validateZipResult`, `validateResumeIntake`) re-verified byte-identical after this follow-up.

### Verification: `node --check` is a false-green here — replaced with a real parse

`node --check` wraps every file in node's own CommonJS module-function wrapper (`(function(exports, require, module, __filename, __dirname){...})`) before parsing, which legalizes top-level `return` for ANY script — including one the Workflow VM itself would reject. Demonstrated directly:

```text
$ printf 'return 1;\n' > /tmp/bare_return.js
$ node --check /tmp/bare_return.js
(exit 0 — "passes" despite being a bare top-level return with no function around it anywhere in the source)
```

Replaced with a genuine `vm.Script` parse that mirrors how this file is actually consumed: `export const meta = {...}` is the literal first statement (checked structurally, then parsed on its own with `export` stripped, since a bare object-literal statement is valid outside any module/function wrapper); everything after it is parsed as the body of a plain `async () => { ... }` IIFE via `vm.Script` — no CommonJS wrapper, no ESM module context, so neither `export` nor a masked bad `return` can hide:

```text
$ node /tmp/verify_parse2.js .claude/workflows/.claude/workflows/mass-patch-fanout.js
PARSE OK (meta literal): .../mass-patch-fanout.js
PARSE OK (script body, async-IIFE-wrapped): .../mass-patch-fanout.js
ALL PARSE CHECKS PASSED: .../mass-patch-fanout.js
```

(Control check: running this same async-IIFE-wrapped `vm.Script` approach against a file that starts with `export` but is NOT followed by anything — or against a body containing a real syntax error — throws immediately, unlike `node --check`, which would still pass via its module wrapper for the `return`-only case above.)
