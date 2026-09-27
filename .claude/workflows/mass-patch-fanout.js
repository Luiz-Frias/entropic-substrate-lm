export const meta = {
  name: 'mass-patch-fanout',
  description: 'K parallel phase lifecycles over edge-disjoint tickets, zipped into one gate-certified stack',
  whenToUse:
    'EXPERIMENTAL overlay — batch of triage-validated, edge-disjoint small fixes needing full per-ticket lifecycle rigor. Pass {tickets[], parent, checkout, cascadeScriptPath, gates{offline,exit}} via args; optional: K (wave width, default 3, hard cap K_MAX=4 — out-of-range throws), conductorRuling, invariantsCommon[], seatPins, lanePolicy (optional object merged into every lane\'s cascade policy, e.g. {publishMode, exitRunner, installCmd, implPackets, ...} — the ported cascade child (td-phase-lifecycle.cascade.js, same directory) understands these keys; this repo launches with lanePolicy: {publishMode:\'defer\', exitRunner:\'uv run python scripts/run_exit_gate.py\', installCmd:\'uv sync --all-groups\', implPackets:false} — every lane certifies locally instead of opening a PR, and the wave publishes once at Zip via `just ship-it`), resume ({plan, startTip, worktrees} — re-enters the wave directly at Lanes/Zip, skipping Plan and Provision entirely; completed lanes pass through via lane.laneResult). Composes td-phase-lifecycle.cascade.js as nested child workflows (one nesting level). Card: .claude/rules-lib/pattern-cards/topology/mass-patch-fanout.md. The conductor reviews the wave report and owns the zip\'s single `just ship-it` publish and all merges.',
  phases: [
    { title: 'Plan', detail: 'batch planner (seat-pinned, default fable::low): edge-disjointness, wave partition, zip order, per-lane mode' },
    { title: 'Provision', detail: 'nested lane worktrees + ceremony; machine-read parent tip' },
    { title: 'Lanes', detail: 'K nested cascade lifecycles in parallel, one ticket each' },
    { title: 'Zip', detail: 'least-to-most-risk restack, per-tip exit gate, one ship-it publish from the stack top' },
  ],
}

// ═══════════════════════════════════════════════════════════════════════════
// mass-patch-fanout — the executable counterpart of the mass-patch-fanout
// topology card. Measured provenance: the 2026-08-27 three-lane pilot
// (FOR-1303/1304/1314 → PRs #17/#18/#19 zipped atop PR #16). Runs in the
// Workflow VM: NO Node builtins, NO filesystem — every IO action is performed
// by an agent (caller-supplies-IO), every deterministic decision is a pure
// function between GUARD-PIN markers so the guard suite can pin it without
// executing this file.
// ═══════════════════════════════════════════════════════════════════════════

const A = args
if (
  !A ||
  (!A.resume && (!Array.isArray(A.tickets) || A.tickets.length === 0)) ||
  typeof A.parent !== 'string' ||
  typeof A.checkout !== 'string' ||
  typeof A.cascadeScriptPath !== 'string' ||
  !A.gates ||
  typeof A.gates.offline !== 'string' ||
  typeof A.gates.exit !== 'string'
) {
  throw new Error('args.{tickets[], parent, checkout, cascadeScriptPath, gates{offline,exit}} is required — tickets are {id,title,scope,acceptance} entries from a triage-validated collection; checkout is the conductor session\'s own worktree (lanes nest under it); tickets is NOT required when args.resume is supplied (the resume path skips Plan entirely and never reads A.tickets)')
}
const K = resolveK(A.K)
const RULING = typeof A.conductorRuling === 'string' ? A.conductorRuling : ''
const COMMON_INVARIANTS = Array.isArray(A.invariantsCommon) ? A.invariantsCommon : []
const PINS = normalizeSeatPins(A.seatPins)

// GUARD-PIN BEGIN normalizeSeatPins
// Per-wave override for the overlay's three seat pins (optional
// args.seatPins) — the knob behind the operator's seat-tier experiments
// (e.g. the plan/zip swap: does judgment belong at composition or at
// execution?). Defaults are the shipped assignments; overrides merge per
// seat. Fail loud on unknown seats, unknown INNER keys (model/effort only —
// FOR-1329: a typo'd inner key like "modle" used to fall through `raw[seat]
// || {}` untouched and silently ran the seat's DEFAULT, i.e. the control
// condition, contradicting this function's own fail-loud contract), a
// non-object seat value, or an unrecognized model/effort — a typo'd
// experiment config must never silently run the control condition.
//
// FOR-1329 haiku note: the three plan/provision/zip agent() call sites below
// have a FIXED shape — `schema: X, model: PINS.<seat>.model, effort:
// PINS.<seat>.effort` — effort is always composed, never conditional on
// model. haiku-4-5 rejects an effort param (capability matrix cited in
// td-phase-lifecycle.cascade.js, 2026-07-13 — documented, not re-measured);
// that engine's own haiku agent() calls honor this by omitting the `effort`
// key entirely (the "no-effort-haiku" convention). Mirroring that here would
// mean restructuring these three call sites to branch on model — out of
// scope this phase. CHOSEN instead: throw. A seat pinned to haiku is refused
// here, naming the reason, rather than silently composing a rejected effort
// param two lines later at the agent() call. Revisit (switch to the
// effortless-pin alternative) only alongside a call-site refactor.
function normalizeSeatPins(raw) {
  // Defaults ratified by the operator 2026-08-28 after the wave-2 vs wave-4
  // seat-pin experiment: fable::low planned a K=4 wave flawlessly at ~equal
  // per-lane cost while opus::xhigh delivered the autonomous zip both times
  // it held that seat (evidence: docs/evidence/mass-patch-fanout-pilot.md).
  // The pre-ratification pins (plan=opus::xhigh, zip=fable::low) remain
  // reachable as an explicit seatPins override.
  const DEFAULTS = {
    plan: { model: 'fable', effort: 'low' },
    provision: { model: 'opus', effort: 'medium' },
    zip: { model: 'opus', effort: 'xhigh' },
  }
  // remote CodeRabbit (this PR, run 96dfbc39): `raw === null` used to share
  // this early-return with `undefined`, silently treating an EXPLICIT null
  // the same as an omitted arg — while every other non-object value throws
  // via the check below. `typeof null === 'object'` (the classic JS
  // footgun), so null must be excluded from that check explicitly rather
  // than relying on it to catch null on its own.
  if (raw === undefined) return DEFAULTS
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('args.seatPins must be an object of {plan|provision|zip: {model?, effort?}}')
  }
  // haiku excluded deliberately: see the FOR-1329 note above the function —
  // these three call sites always compose an effort param, which haiku-4-5
  // rejects.
  const MODELS = ['sonnet', 'opus', 'fable']
  const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max']
  const INNER_KEYS = ['model', 'effort']
  const out = { plan: Object.assign({}, DEFAULTS.plan), provision: Object.assign({}, DEFAULTS.provision), zip: Object.assign({}, DEFAULTS.zip) }
  for (const seat of Object.keys(raw)) {
    // own-property check, not `in`: `in` follows the prototype chain, so a
    // seat key inherited from Object.prototype (e.g. "constructor",
    // "toString") would pass this check, then `out[seat].model = o.model`
    // below would mutate the INHERITED value instead of throwing — silently
    // running the default seat, the exact failure this guard exists to
    // refuse (code-review, local CodeRabbit, s5 phase).
    if (!Object.prototype.hasOwnProperty.call(DEFAULTS, seat)) {
      throw new Error('args.seatPins: unknown seat ' + JSON.stringify(seat) + ' (valid: plan, provision, zip)')
    }
    const o = raw[seat]
    if (o === null || typeof o !== 'object' || Array.isArray(o)) {
      throw new Error('args.seatPins.' + seat + ' must be an object of {model?, effort?}; got ' + JSON.stringify(o))
    }
    for (const innerKey of Object.keys(o)) {
      if (!INNER_KEYS.includes(innerKey)) {
        throw new Error('args.seatPins.' + seat + ': unknown key ' + JSON.stringify(innerKey) + ' (valid: model, effort) — a typo here must never silently run the control condition')
      }
    }
    if (o.model !== undefined) {
      if (o.model === 'haiku') {
        throw new Error('args.seatPins.' + seat + '.model: haiku is not supported for the plan/provision/zip seats — these call sites always compose an effort param and haiku-4-5 rejects it (capability matrix, 2026-07-13); pick sonnet, opus, or fable')
      }
      if (!MODELS.includes(o.model)) throw new Error('args.seatPins.' + seat + '.model must be one of ' + MODELS.join('|') + '; got ' + JSON.stringify(o.model))
      out[seat].model = o.model
    }
    if (o.effort !== undefined) {
      if (!EFFORTS.includes(o.effort)) throw new Error('args.seatPins.' + seat + '.effort must be one of ' + EFFORTS.join('|') + '; got ' + JSON.stringify(o.effort))
      out[seat].effort = o.effort
    }
  }
  return out
}
// GUARD-PIN END normalizeSeatPins

// GUARD-PIN BEGIN kBound
// FOR-1324: args.K (wave width) previously accepted Infinity or any large
// finite number, defeating the run's ~16-agent concurrency budget (a cascade
// lane peaks 3-4 concurrent agents). K must now be a finite integer in
// [1, K_MAX]; anything else throws naming the valid range. The cap value is
// DECISION-adjacent (a machine/concurrency-cap fact, not a pure software
// constant) — it lives behind this named constant, default 4 (the measured
// dev-machine bound also documented in the SKILL.md K note and the
// mass-patch-fanout topology card's review trigger), and is surfaced in the
// PR body for operator ratification rather than silently blocking future
// widening: bump K_MAX here if a stronger machine or a lower per-lane agent
// count measurably raises the safe wave width. K_MAX is declared INSIDE the
// function (not as a sibling top-level const) deliberately: resolveK is
// called at module top level before its own textual position (function
// declarations hoist; a sibling top-level `const K_MAX` would not — it sits
// in the temporal dead zone until its own declaration line runs, so an
// out-of-function K_MAX would throw ReferenceError on every in-range K).
function resolveK(rawK) {
  const K_MAX = 4
  if (rawK === undefined) return 3
  if (typeof rawK !== 'number' || !Number.isInteger(rawK) || rawK < 1 || rawK > K_MAX) {
    throw new Error('args.K must be a finite integer in [1, ' + K_MAX + ']; got ' + String(rawK))
  }
  return rawK
}
// GUARD-PIN END kBound

// GUARD-PIN BEGIN zipOrderRationale
// The zip's ordering metric, stated once and pinned: lanes fan in from least
// to most merge-conflict risk, where a lane's risk = its touch-set overlap
// with the other lanes' touch-sets UNION the base branch's hot files (README,
// shared configs — files many changes gravitate to). Cheapest reparents land
// first; the likeliest conflict surfaces exactly once, at the top of the
// stack. Measured 2026-08-27: new-files-only → single-file → hot-README
// zipped with zero conflicts.
const zipOrderRationale =
  'Zip lanes LEAST to MOST merge-conflict risk. Risk metric: touch-set overlap with the other lanes plus overlap with base-branch hot files (README, shared configs). New-files-only lanes zip first; the lane touching the hottest shared file zips last, so the likeliest conflict surfaces exactly once, at the stack top.'
// GUARD-PIN END zipOrderRationale

// GUARD-PIN BEGIN laneArgs
// Pure composer for one lane's child-workflow args. The startTip parameter is
// the CONDUCTOR-read parent tip (machine output of `git rev-parse`, relayed by
// the Provision agent) — it anchors every lane's packet-#1 ancestry check via
// the cascade engine's seedEngineTrackedTip intake. Malformed startTip throws:
// a lane must never launch with a broken chain-of-custody anchor (fail loud,
// never fall open). The first invariant encodes the pilot's measured harness
// lessons: lanes are NESTED under the conductor's pin, reached by plain cd
// (subagent EnterWorktree reports success but Bash refuses siblings), and
// workers verify pwd + branch before acting because harness-carved isolation
// worktrees have landed under the wrong lane when several coexist.
function laneArgs(lane, common, startTip) {
  if (typeof startTip !== 'string' || !/^[0-9a-f]{40}$/.test(startTip)) {
    throw new Error('laneArgs: startTip must be the conductor-read full lowercase 40-hex parent tip; got ' + JSON.stringify(startTip))
  }
  if (typeof common.checkout !== 'string' || !common.checkout.startsWith('/')) {
    throw new Error('laneArgs: common.checkout must be the absolute launch-checkout path — the cascade engine\'s fail-closed advisorLockPath (FOR-1301) requires args.repoToplevel, and a lane composed without it dies at parse (measured: wave 4 first launch, all 4 lanes DIED); got ' + JSON.stringify(common.checkout))
  }
  const isolation =
    'WORKTREE ISOLATION: ALL work happens in ' +
    lane.worktree +
    ' (a NESTED worktree under the launch checkout - plain `cd` there works; do NOT call EnterWorktree). cd there FIRST in every stage AND verify with pwd + `git branch --show-current` = ' +
    lane.branch +
    ' before acting; on any mismatch STOP and return DECISION with the evidence - never work around a misplaced cwd. ' +
    (common.laneTotal > 1
      ? 'Sibling lanes run in parallel in ' + common.siblingWorktrees.join(', ') + '; their files are disjoint from yours by design - never reach into their worktrees. '
      : '') +
    'Editing files in any other checkout is a FAIL.'
  return {
    // The SHARED checkout, not the lane worktree: advisor consults serialize
    // per-repo across parallel lanes (one lock for the wave), matching the
    // pilot waves' shared-lock behavior.
    repoToplevel: common.checkout,
    phase: {
      id: lane.id,
      title: lane.title,
      issue: lane.issue,
      branch: lane.branch,
      parent: common.parent,
      startTip: startTip,
      mode: lane.mode === 'arch' ? 'arch' : 'normal',
      scope: 'MASS-PATCH FAN-OUT lane ' + (common.laneIndex + 1) + ' of ' + common.laneTotal + ' (one lifecycle per ticket, isolated nested worktrees, sibling branches off one parent tip).\n\n' + lane.scope,
      acceptance: lane.acceptance,
    },
    invariants: [isolation].concat(common.commonInvariants || []),
    gates: { offline: common.gates.offline, exit: common.gates.exit, liveRequired: false },
    // PORT (entropic-substrate-lm): lanePolicy passthrough — args.lanePolicy
    // (threaded into `common` at the Lanes call site below) merges OVER these
    // wave defaults so the repo's cascade port (publishMode 'defer', the
    // Python exitRunner/installCmd, implPackets false) reaches every lane
    // without hardcoding a repo-specific policy into this shared composer.
    // teardown:false is an ownership-hazard fix (cascade-porter finding): K
    // lanes share repoToplevel=common.checkout (see the comment above), so K
    // lanes hitting the cascade's own Phase 7 Teardown CONCURRENTLY each
    // list-and-remove every clean `wf_*` scratch worktree under that SHARED
    // checkout — lane A can remove lane B's still-active clean worktree. The
    // cascade port honours policy.teardown; false here defers that sweep to
    // the Zip stage below, which runs once, serially, after every lane has
    // finished (see the new Zip-prompt teardown step). args.lanePolicy can
    // still override this per wave.
    policy: Object.assign({ submitMode: 'draft', recon: true, implPackets: true, maxPackets: 2, teardown: false }, common.lanePolicy || {}),
    conductorRuling: common.conductorRuling,
    // PORT (entropic-substrate-lm, W2-COMMIT-LOCK 2026-09-23): optional args.implRuling
    // passthrough — the cascade appends it to the IMPLEMENTER prompt only, so a
    // ruling on an implement-stage DECISION reaches a resumed lane without
    // invalidating its cached recon/design-gate results.
    implRuling: common.implRuling,
  }
}
// GUARD-PIN END laneArgs

// GUARD-PIN BEGIN validateZipPlan
// FOR-1325: plan.zipOrder is a bare string array handed VERBATIM to the Zip
// agent's track/reparent/force-submit instructions — the sharpest blast
// radius in the overlay, since a schema-valid but semantically wrong plan
// can silently name an unrelated existing branch or drop a SUBMITTED lane
// from the stack. This pure validator runs once, post-plan, before any lane
// launches: the lane set itself must be well-formed (unique ids, unique
// branches, within the wave-width cap K) and zipOrder must be an EXACT
// permutation of the admitted lanes' branches — same multiset, no more, no
// fewer, never the parent branch. Returns null when the plan is well-formed,
// else a halt-reason STRING (never throws — a bad planner output is an
// expected, bounded failure the caller turns into DECISION-NEEDED, never a
// workflow crash).
function validateZipPlan(lanes, zipOrder, parent, K) {
  if (!Array.isArray(lanes) || !Array.isArray(zipOrder)) {
    return 'plan.lanes and plan.zipOrder must both be arrays'
  }
  if (lanes.length > K) {
    return 'plan.lanes.length (' + lanes.length + ') exceeds wave width K=' + K
  }
  const ids = lanes.map((l) => l.id)
  const dupIds = ids.filter((id, i) => ids.indexOf(id) !== i)
  if (dupIds.length) {
    return 'plan.lanes has duplicate lane id(s): ' + Array.from(new Set(dupIds)).join(', ')
  }
  const branches = lanes.map((l) => l.branch)
  const dupBranches = branches.filter((b, i) => branches.indexOf(b) !== i)
  if (dupBranches.length) {
    return 'plan.lanes has duplicate branch(es): ' + Array.from(new Set(dupBranches)).join(', ')
  }
  // FOR-1342: validate each worktreeName as a single safe path segment —
  // non-empty, no embedded slash, no traversal ('..'), no leading dot.
  for (const lane of lanes) {
    const wn = lane.worktreeName
    if (!wn) {
      return 'plan.lanes has a lane with an empty worktreeName (id=' + lane.id + ')'
    }
    if (wn.includes('/')) {
      return 'plan.lanes has a lane with an unsafe worktreeName (embedded slash, id=' + lane.id + '): ' + JSON.stringify(wn)
    }
    if (wn === '..') {
      return 'plan.lanes has a lane with an unsafe worktreeName (traversal, id=' + lane.id + '): ' + JSON.stringify(wn)
    }
    if (wn.startsWith('.')) {
      return 'plan.lanes has a lane with an unsafe worktreeName (leading dot, id=' + lane.id + '): ' + JSON.stringify(wn)
    }
  }
  const branchSet = new Set(branches)
  const seen = new Set()
  for (const entry of zipOrder) {
    if (entry === parent) {
      return 'zipOrder must not name the parent branch: ' + entry
    }
    if (!branchSet.has(entry)) {
      return 'zipOrder names a branch not among the admitted lanes: ' + entry
    }
    if (seen.has(entry)) {
      return 'zipOrder has a duplicate entry: ' + entry
    }
    seen.add(entry)
  }
  if (zipOrder.length !== branches.length) {
    return 'zipOrder is not an exact permutation of the admitted lane branches (has ' + zipOrder.length + ' entries, lanes admit ' + branches.length + ')'
  }
  return null
}
// GUARD-PIN END validateZipPlan

// GUARD-PIN BEGIN validateProvisionWorktrees
// FOR-1327: provision.worktrees is POSITIONALLY INDEXED against plan.lanes in
// laneArgs (provision.worktrees[i] pairs with plan.lanes[i]) — a short,
// reordered, duplicated, or relative-path entry silently bakes `undefined`
// or a sibling's path into a lane's isolation invariant text, defeating the
// very guarantee that invariant claims. This pure validator runs once, wired
// BEFORE the Lanes parallel() launch: worktrees must be an ordered,
// absolute-path match, one entry per lane, each path EXACTLY
// `checkout/.claude/worktrees/<worktreeName>` — nested under the launch
// checkout, never a sibling.
// FOR-1341: an endsWith(worktreeName) check alone let an absolute SIBLING
// path (e.g. /tmp/wave-for-1324, or any path outside
// checkout/.claude/worktrees/) through as long as its last path segment
// matched — defeating the nested-worktree isolation law the whole overlay
// depends on (EnterWorktree/Bash placement, see the laneArgs block comment).
// Exact string equality against the one legal nested path closes that hole.
// Symlink realpath resolution is explicitly out of scope here (FOR-1337).
// FOR-1337: a caller-supplied checkout with a trailing slash (shell
// completion, an env var, a copy-pasted config value) composed a
// DOUBLE-SLASH expected path ('checkout//.claude/worktrees/<name>') that
// never string-equals the single-slash path the real worktree tooling
// creates, false-halting every provision even for a correctly-nested
// worktree. Trailing slashes are stripped before composing `expected` —
// normalization only; the exact-match comparison itself, and everything
// FOR-1341 closed, are unchanged.
// Returns null when well-formed, else a halt-reason STRING (never throws —
// a bad provision-agent report is an expected, bounded failure the caller
// turns into DECISION-NEEDED, never a workflow crash).
function validateProvisionWorktrees(lanes, worktrees, checkout) {
  if (!Array.isArray(lanes) || !Array.isArray(worktrees)) {
    return 'plan.lanes and provision.worktrees must both be arrays'
  }
  if (typeof checkout !== 'string' || !checkout.startsWith('/')) {
    return 'checkout must be an absolute path'
  }
  if (worktrees.length !== lanes.length) {
    return 'provision.worktrees has ' + worktrees.length + ' entry(ies) but the plan admits ' + lanes.length + ' lane(s) — short or long list'
  }
  const normalizedCheckout = checkout.length > 1 ? checkout.replace(/\/+$/, '') : checkout
  const seen = new Set()
  for (let i = 0; i < worktrees.length; i++) {
    const wt = worktrees[i]
    if (typeof wt !== 'string' || !wt.startsWith('/')) {
      return 'provision.worktrees[' + i + '] is not an absolute path: ' + JSON.stringify(wt)
    }
    if (seen.has(wt)) {
      return 'provision.worktrees has a duplicate entry: ' + wt
    }
    seen.add(wt)
    const expected = normalizedCheckout + '/.claude/worktrees/' + lanes[i].worktreeName
    if (wt !== expected) {
      return 'provision.worktrees[' + i + '] must be exactly checkout/.claude/worktrees/' + lanes[i].worktreeName + ' (nested under the launch checkout) — a sibling, escaped, or reordered worktree list is refused: expected ' + expected + ', got ' + wt
    }
  }
  return null
}
// GUARD-PIN END validateProvisionWorktrees

// GUARD-PIN BEGIN validateZipResult
// FOR-1328 (shape-hardening half): the ZIP schema accepts an empty
// stack/tips pair with free-text evidence, and the only prior acceptance
// check was `zip.status !== 'ZIPPED'` — a schema-valid but empty or
// malformed zip report could still reach WAVE-SUBMITTED while overstating
// what was actually certified. This pure validator runs once, post-zip,
// before WAVE-SUBMITTED: stack and tips must both be non-empty, the same
// length (one tip per stack entry, positionally paired the same way
// provision.worktrees pairs with plan.lanes), every tip a full lowercase
// 40-hex SHA, and stack must contain every branch the plan's zipOrder named
// (a zip that silently dropped a lane from the final stack is exactly the
// failure this guards against). Returns null when well-formed, else a
// halt-reason STRING (never throws — a bad zip-agent report is an expected,
// bounded failure the caller turns into DECISION-NEEDED, never a workflow
// crash — same discipline as validateZipPlan/validateProvisionWorktrees).
//
// DECISION-GATED, NOT IMPLEMENTED (FOR-1328, the other half): requiring a
// machine-produced per-tip exit-gate CERTIFICATE (not just a bare SHA) in
// the zip report is a wire-format change to the ZIP schema and to what the
// zip agent must produce — proposed shape for operator ratification (also
// surfaced in this phase's report):
//   certificates: [{ branch: string, tip: <40-hex>, cmd: string,
//                     exitCode: 0, tail: string }]
// one entry per stack branch, sourced the same way run-exit-gate.ts already
// emits {cmd, testedTip, exitCode, tail} today — out of scope this phase.
function validateZipResult(stack, tips, zipOrder) {
  if (!Array.isArray(stack) || stack.length === 0) {
    return 'zip.stack must be a non-empty array'
  }
  if (!Array.isArray(tips) || tips.length === 0) {
    return 'zip.tips must be a non-empty array'
  }
  if (tips.length !== stack.length) {
    return 'zip.tips.length (' + tips.length + ') does not match zip.stack.length (' + stack.length + ')'
  }
  for (let i = 0; i < tips.length; i++) {
    if (typeof tips[i] !== 'string' || !/^[0-9a-f]{40}$/.test(tips[i])) {
      return 'zip.tips[' + i + '] is not a full lowercase 40-hex SHA: ' + JSON.stringify(tips[i])
    }
  }
  const stackSet = new Set(stack)
  const missing = (zipOrder || []).filter((b) => !stackSet.has(b))
  if (missing.length) {
    return 'zip.stack is missing zipOrder branch(es): ' + missing.join(', ')
  }
  return null
}
// GUARD-PIN END validateZipResult

// GUARD-PIN BEGIN validateResumeIntake
// FOR-1330: args.resume = {plan, startTip, worktrees} lets the conductor
// re-enter a wave directly at Lanes/Zip after a prior run's DECISION-NEEDED
// — Plan and Provision are skipped ENTIRELY on resume (see the RESUME
// branches around the Plan and Provision phases below). Caller-supplied
// args are validated the way every other args.* input on this file is:
// throw naming the defect, never a soft DECISION — a malformed resume
// payload is an operator/conductor authoring mistake, not an agent judgment
// call this engine adjudicates. Structural well-formedness of
// plan.lanes/zipOrder and of worktrees is delegated to validateZipPlan and
// validateProvisionWorktrees (FOR-1325/FOR-1327) rather than re-derived here
// — the same permutation/absolute-path/exact-nesting rules apply verbatim to
// a resumed plan as to a freshly-planned one; forking a parallel copy of
// that logic would let the two paths drift silently. Each lane object is
// also checked for the string fields laneArgs composes into a child brief
// (id/title/issue/branch/worktreeName/scope/acceptance) — a lane missing one
// of these would otherwise sail straight through both reused validators
// (neither one inspects those fields) and compose `undefined` into a lane's
// phase.scope/acceptance, exactly the silent-corruption failure this
// validator exists to refuse. `mode` is checked separately as the enum
// ['normal','arch'] the fresh-path PLAN schema already requires (line
// ~458): laneArgs composes `mode: lane.mode === 'arch' ? 'arch' : 'normal'`
// (a ternary, not a validated field), so an omitted or misspelled mode on a
// resumed lane would silently float mode to 'normal' — exactly the "a typo
// runs the control condition" failure FOR-1329 forbids for seatPins, here
// silently stripping an arch lane's design gate instead. Every legitimate
// resume payload originates from a prior Plan output and therefore already
// carries a valid mode, so this rejects only payloads already malformed by
// the fresh path's own contract.
function validateResumeIntake(rawResume, K, parent, checkout) {
  if (rawResume === undefined) return null
  if (rawResume === null || typeof rawResume !== 'object' || Array.isArray(rawResume)) {
    throw new Error('args.resume must be an object of {plan, startTip, worktrees}; got ' + JSON.stringify(rawResume))
  }
  const REQUIRED_KEYS = ['plan', 'startTip', 'worktrees']
  for (const key of Object.keys(rawResume)) {
    if (!REQUIRED_KEYS.includes(key)) {
      throw new Error('args.resume: unknown key ' + JSON.stringify(key) + ' (valid: plan, startTip, worktrees)')
    }
  }
  for (const key of REQUIRED_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(rawResume, key)) {
      throw new Error('args.resume is missing required key ' + JSON.stringify(key))
    }
  }
  const plan = rawResume.plan
  const startTip = rawResume.startTip
  const worktrees = rawResume.worktrees
  if (
    plan === null ||
    typeof plan !== 'object' ||
    Array.isArray(plan) ||
    !Array.isArray(plan.lanes) ||
    !Array.isArray(plan.zipOrder)
  ) {
    throw new Error('args.resume.plan must be an object with lanes[] and zipOrder[] (the same shape the fresh-path Plan agent emits)')
  }
  const LANE_STRING_FIELDS = ['id', 'title', 'issue', 'branch', 'worktreeName', 'scope', 'acceptance']
  const LANE_MODES = ['normal', 'arch']
  for (let i = 0; i < plan.lanes.length; i++) {
    const lane = plan.lanes[i]
    if (lane === null || typeof lane !== 'object' || Array.isArray(lane)) {
      throw new Error('args.resume.plan.lanes[' + i + '] must be an object')
    }
    for (const field of LANE_STRING_FIELDS) {
      if (typeof lane[field] !== 'string' || lane[field].length === 0) {
        throw new Error('args.resume.plan.lanes[' + i + '].' + field + ' must be a non-empty string — a lane missing it would compose undefined into the child lifecycle brief')
      }
    }
    if (!LANE_MODES.includes(lane.mode)) {
      throw new Error('args.resume.plan.lanes[' + i + '].mode must be one of ' + LANE_MODES.join('|') + '; got ' + JSON.stringify(lane.mode) + ' — laneArgs\' ternary would otherwise silently downgrade an arch lane to normal')
    }
    // CodeRabbit (this PR, run ea9d24ba): laneResult is truthy-tested by
    // Phase 3 (`lane.laneResult ? Promise.resolve(lane.laneResult) :
    // workflow(...)`) to decide whether a lane is already complete and
    // should NOT relaunch. {} is truthy, so an empty or malformed laneResult
    // used to silently skip relaunching the lane instead of throwing — the
    // lane would then report verdict 'UNKNOWN' rather than actually running
    // its ticket. laneResult must mirror what a nested workflow() call
    // itself returns (SKILL.md: "at minimum { verdict: 'SUBMITTED',
    // publish: { pr } }"), which requires at least a non-empty string
    // verdict; the verdict itself need NOT be 'SUBMITTED' — a poisoned or
    // blocked lane's real prior verdict (e.g. 'DECISION-NEEDED') is
    // legitimately replayed here so the conductor can adjudicate it without
    // relaunching, per the Lanes-phase 'unclean' check below.
    if (lane.laneResult !== undefined) {
      if (lane.laneResult === null || typeof lane.laneResult !== 'object' || Array.isArray(lane.laneResult)) {
        throw new Error('args.resume.plan.lanes[' + i + '].laneResult must be an object mirroring a nested workflow() result (at minimum { verdict }); got ' + JSON.stringify(lane.laneResult))
      }
      if (typeof lane.laneResult.verdict !== 'string' || lane.laneResult.verdict.length === 0) {
        throw new Error('args.resume.plan.lanes[' + i + '].laneResult.verdict must be a non-empty string — an empty or missing verdict would silently skip relaunching the lane instead of throwing')
      }
    }
  }
  if (typeof startTip !== 'string' || !/^[0-9a-f]{40}$/.test(startTip)) {
    throw new Error('args.resume.startTip must be the full lowercase 40-hex parent tip; got ' + JSON.stringify(startTip))
  }
  if (!Array.isArray(worktrees)) {
    throw new Error('args.resume.worktrees must be an array of absolute lane worktree paths')
  }
  const zipHalt = validateZipPlan(plan.lanes, plan.zipOrder, parent, K)
  if (zipHalt) {
    throw new Error('args.resume.plan is malformed: ' + zipHalt)
  }
  const provisionHalt = validateProvisionWorktrees(plan.lanes, worktrees, checkout)
  if (provisionHalt) {
    throw new Error('args.resume.worktrees is malformed: ' + provisionHalt)
  }
  return { plan: plan, startTip: startTip, worktrees: worktrees }
}
// GUARD-PIN END validateResumeIntake
const RESUME = validateResumeIntake(A.resume, K, A.parent, A.checkout)

// ─── Phase 1: Plan — the expensive tier judges the BATCH once, then detaches ─
phase('Plan')
const PLAN = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { enum: ['PLAN', 'DECISION'] },
    decision: { type: 'string' },
    lanes: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          issue: { type: 'string' },
          branch: { type: 'string' },
          worktreeName: { type: 'string' },
          scope: { type: 'string' },
          acceptance: { type: 'string' },
          mode: { enum: ['normal', 'arch'] },
          touchSet: { type: 'array', items: { type: 'string' } },
        },
        required: ['id', 'title', 'issue', 'branch', 'worktreeName', 'scope', 'acceptance', 'mode', 'touchSet'],
      },
    },
    zipOrder: { type: 'array', items: { type: 'string' } },
    rejected: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: { issue: { type: 'string' }, reason: { type: 'string' } },
        required: ['issue', 'reason'],
      },
    },
    evidence: { type: 'string', minLength: 1 },
  },
  required: ['verdict', 'lanes', 'zipOrder', 'rejected', 'evidence'],
}
let plan
if (RESUME) {
  // FOR-1330: args.resume supplies an already-judged plan (a prior run's
  // Plan-phase output, replayed verbatim) — re-invoking the Opus batch
  // planner here would re-judge disjointness against a tree the conductor
  // has already partially executed against, and could silently emit a
  // DIFFERENT plan than the one the existing lane worktrees/branches were
  // actually provisioned from. The planner agent() call is skipped entirely.
  plan = RESUME.plan
  log('resume: Plan skipped — reusing the supplied wave plan (' + plan.lanes.length + ' lane(s) [' + plan.lanes.map((l) => l.issue).join(' · ') + '])')
} else {
  plan = await agent(
    'You are the BATCH PLANNER for a mass-patch fan-out wave (read-only: zero edits, zero commits; your plan is the wave contract). Repo checkout: ' + A.checkout + ', parent branch ' + A.parent + '.\n\nTICKET COLLECTION (' + A.tickets.length + ' candidates):\n' + JSON.stringify(A.tickets, null, 2) + '\n\nDuties:\n1. EDGE-DISJOINTNESS: derive each ticket\'s touch-set by reading the actual tree (grep/read the files its scope names). Two tickets sharing ANY file are NOT both admissible in this wave — keep the higher-priority one, reject the other with reason "edge overlap with <issue>".\n2. WAVE WIDTH: admit at most ' + K + ' lanes (K bound: lanes share the run\'s ~16-agent concurrency cap and the review queue). Reject the overflow with reason "wave width cap" — rejected tickets are DEFERRED, not dropped; list every one.\n3. PER-LANE MODE: flag mode "arch" for any ticket whose scope touches contract surfaces (ports/domain boundaries, workflow schemas, wire formats); tickets that would EDIT the lifecycle engine files themselves are rejected outright with reason "engine edit target - serial phase required" (an engine cannot review its own edit in flight).\n4. LANE BRIEFS: for each admitted lane emit id (w<index>), title, issue, branch (feat/<issue-slug>), worktreeName (wave-<issue-slug>), a self-contained scope (the fix, verification constraints, TDD note), and acceptance (including gates ' + JSON.stringify(A.gates) + ' and an append-only clause).\n5. ZIP ORDER: ' + zipOrderRationale + ' Emit zipOrder as the admitted branch names, least risk first.\nVerdict DECISION (with the question) if the collection is un-batchable as framed; otherwise PLAN. Evidence: the touch-set table you derived.',
    // Default opus::xhigh: the plan IS the wave contract — disjointness
    // misjudgment costs every lane. Overridable per wave via args.seatPins.
    { label: 'plan:wave', phase: 'Plan', schema: PLAN, model: PINS.plan.model, effort: PINS.plan.effort },
  )
  if (!plan) return { verdict: 'BLOCKED', at: 'plan', reason: 'batch planner died (fail loud, never skip the disjointness judgment)' }
  if (plan.verdict === 'DECISION') return { verdict: 'DECISION-NEEDED', at: 'plan', decision: plan.decision || '(planner returned DECISION without text)', rejected: plan.rejected }
  if (!plan.lanes.length) return { verdict: 'DECISION-NEEDED', at: 'plan', decision: 'planner admitted zero lanes', rejected: plan.rejected }
  log('wave plan: ' + plan.lanes.length + ' lane(s) [' + plan.lanes.map((l) => l.issue).join(' · ') + '], ' + plan.rejected.length + ' deferred, zip order: ' + plan.zipOrder.join(' → '))
  if (plan.rejected.length) log('DEFERRED (no silent caps): ' + plan.rejected.map((r) => r.issue + ' (' + r.reason + ')').join('; '))
}
const zipPlanHalt = validateZipPlan(plan.lanes, plan.zipOrder, A.parent, K)
if (zipPlanHalt) return { verdict: 'DECISION-NEEDED', at: 'plan', decision: zipPlanHalt, plan: plan }

// ─── Phase 2: Provision — worktrees are IO, so an agent does it ─────────────
phase('Provision')
const PROVISION = {
  type: 'object',
  additionalProperties: false,
  properties: {
    status: { enum: ['READY', 'DECISION'] },
    decision: { type: 'string' },
    startTip: { type: 'string', pattern: '^[0-9a-f]{40}$' },
    worktrees: { type: 'array', items: { type: 'string' } },
    evidence: { type: 'string', minLength: 1 },
  },
  required: ['status', 'startTip', 'worktrees', 'evidence'],
}
let provision
if (RESUME) {
  // FOR-1330: worktrees and startTip were supplied by the conductor from a
  // prior run's Provision output — NEVER re-provisioned. No `git worktree
  // add`, no `git rev-parse` runs on resume: existing lane branches and
  // worktrees are reused exactly as reported, never recreated or truncated.
  provision = {
    status: 'READY',
    startTip: RESUME.startTip,
    worktrees: RESUME.worktrees,
    evidence: 'resumed via args.resume — worktrees and startTip supplied by the conductor, not re-provisioned',
  }
  log('resume: Provision skipped — reusing ' + provision.worktrees.length + ' existing lane worktree(s) off ' + A.parent + ' @ ' + provision.startTip.slice(0, 7))
} else {
  provision = await agent(
    // FOR-1326: startTip custody race — rev-parse the parent tip FIRST
    // (step 1), THEN create every lane worktree FROM that literal SHA (step
    // 2), never from the mutable A.parent ref by name. The prior ordering
    // (worktree-add per lane off A.parent, THEN one rev-parse at the end)
    // let a commit landing on A.parent mid-loop put lanes on different
    // physical tips while the reported startTip named a commit some (or
    // none) of them actually root at.
    'You are the PROVISION agent for a mass-patch fan-out wave. Work from ' + A.checkout + ' (the conductor\'s own checkout - lanes NEST under it; this placement is a measured law, siblings are unreachable to lane workers).\nSTEP 1, BEFORE creating any worktree: record the parent tip MACHINE-TO-MACHINE — run `git rev-parse ' + A.parent + '` and paste the full 40-hex SHA into startTip verbatim from command output (a hand-typed SHA is a FAIL). Capture it EXACTLY ONCE, first: ' + A.parent + ' is a mutable ref, and a commit landing on it between an early worktree add and a later rev-parse would put lane worktrees on different physical tips while startTip named a commit some (or none) of them actually root at.\nSTEP 2, for each lane below, using THAT SAME SHA as the literal start-point (never `' + A.parent + '` by name): `git worktree add .claude/worktrees/<worktreeName> -b <branch> <the 40-hex SHA from step 1>` (worktree path NESTED under the checkout); then copy the gitignored `.env` and `justfile.local` from the checkout root into the lane worktree root if they exist (the WorktreeCreate hook does not fire for git-worktree-add), then run `uv sync --all-groups` in the lane; then run the repo\'s worktree ceremony if a recipe exists (`just worktree-ceremony path=...`; skip with a note if absent).\nLANES: ' + JSON.stringify(plan.lanes.map((l) => ({ worktreeName: l.worktreeName, branch: l.branch })), null, 2) + '\nNEVER: commit, push, edit files, touch branches beyond the worktree adds, or rev-parse ' + A.parent + ' again after step 1 — every worktree add and your final report reuse the SAME step-1 SHA. Report absolute worktree paths in creation order and that same SHA as startTip. Any git anomaly → status DECISION with the command + output.',
    // Opus at medium: provisioning is mostly mechanical git ceremony, but it
    // mints the wave's chain-of-custody anchor (startTip) and its anomaly-STOP
    // judgment protects every lane — worth the stronger tier at modest effort.
    { label: 'provision:wave', phase: 'Provision', schema: PROVISION, model: PINS.provision.model, effort: PINS.provision.effort },
  )
  if (!provision || provision.status !== 'READY') {
    return { verdict: 'DECISION-NEEDED', at: 'provision', decision: (provision && provision.decision) || 'provision agent died or returned DECISION', plan: plan }
  }
  log('provisioned ' + provision.worktrees.length + ' lane worktree(s) off ' + A.parent + ' @ ' + provision.startTip.slice(0, 7))
}
const provisionHalt = validateProvisionWorktrees(plan.lanes, provision.worktrees, A.checkout)
if (provisionHalt) {
  return { verdict: 'DECISION-NEEDED', at: 'provision', decision: provisionHalt, plan: plan, startTip: provision.startTip }
}

// ─── Phase 3: Lanes — K nested lifecycles, one ticket each, truly parallel ──
phase('Lanes')
if (RESUME) {
  // FOR-1330: a lane carrying lane.laneResult (see the ternary below) is a
  // COMPLETED lane from the prior run — its worktree/branch is untouched and
  // its result is passed through unchanged; only the remaining lanes launch
  // a nested cascade lifecycle. Documented shape for lane.laneResult:
  // SKILL.md "Resume" section (must mirror what a nested workflow() call
  // returns — {verdict: 'SUBMITTED', publish: {pr}, ...} for a published lane,
  // or {verdict: 'CERTIFIED', publish, record: {tip, exitGate, branch, parent}}
  // for a lane certified locally under lanePolicy.publishMode 'defer' — so
  // laneSummary below composes it identically to a freshly-launched lane).
  const completedCount = plan.lanes.filter((l) => l.laneResult).length
  log('resume: ' + completedCount + ' lane(s) already complete (passed through unchanged), launching ' + (plan.lanes.length - completedCount) + ' remaining lane(s)')
}
const laneResults = await parallel(
  plan.lanes.map((lane, i) => () =>
    lane.laneResult
      ? Promise.resolve(lane.laneResult)
      : workflow(
          { scriptPath: A.cascadeScriptPath },
          laneArgs(
            Object.assign({}, lane, { worktree: provision.worktrees[i] }),
            {
              parent: A.parent,
              gates: A.gates,
              conductorRuling: RULING,
              laneIndex: i,
              laneTotal: plan.lanes.length,
              siblingWorktrees: provision.worktrees.filter((_, j) => j !== i),
              commonInvariants: COMMON_INVARIANTS,
              checkout: A.checkout,
              lanePolicy: A.lanePolicy,
              implRuling: typeof A.implRuling === 'string' && A.implRuling.trim() ? A.implRuling : undefined,
            },
            provision.startTip,
          ),
        ),
  ),
)
const laneSummary = plan.lanes.map((lane, i) => {
  const r = laneResults[i]
  const verdict = r ? r.verdict || 'UNKNOWN' : 'DIED'
  // CERTIFIED (lanePolicy.publishMode 'defer') is a clean lane too: the lane's
  // cascade certified a tip + exit gate locally instead of opening a PR, so
  // `pr` stays null and `tip` carries the certified record's tip for the zip
  // and the conductor to act on without re-deriving it from `publish`.
  return {
    issue: lane.issue,
    branch: lane.branch,
    verdict: verdict,
    pr: r && r.publish ? r.publish.pr : null,
    tip: verdict === 'CERTIFIED' && r && r.record ? r.record.tip : null,
  }
})
log('lanes: ' + laneSummary.map((s) => s.issue + '=' + s.verdict).join(' · '))
const unclean = laneSummary.filter((s) => s.verdict !== 'SUBMITTED' && s.verdict !== 'CERTIFIED')
if (unclean.length) {
  // A poisoned or blocked lane is adjudicated individually by the conductor
  // (verify-only resume when its work landed) — the zip NEVER folds it in
  // silently, and the clean lanes' results are preserved for the retry wave.
  // FOR-1330: worktrees is included so this report is directly replayable
  // as args.resume = {plan, startTip, worktrees} without the conductor
  // having to reconstruct the worktree list by hand.
  return { verdict: 'DECISION-NEEDED', at: 'lanes', decision: 'lane(s) not SUBMITTED/CERTIFIED: ' + unclean.map((s) => s.issue + '=' + s.verdict).join(', ') + ' - adjudicate individually (verify-only resume if work landed), then re-run the zip', lanes: laneSummary, plan: plan, startTip: provision.startTip, worktrees: provision.worktrees }
}

// ─── Phase 4: Zip — fan-in as a git-topology operation ──────────────────────
phase('Zip')
const ZIP = {
  type: 'object',
  additionalProperties: false,
  properties: {
    status: { enum: ['ZIPPED', 'DECISION'] },
    decision: { type: 'string' },
    stack: { type: 'array', items: { type: 'string' } },
    tips: { type: 'array', items: { type: 'string', pattern: '^[0-9a-f]{40}$' } },
    evidence: { type: 'string', minLength: 1 },
  },
  required: ['status', 'stack', 'tips', 'evidence'],
}
// PORT (entropic-substrate-lm): the zip publishes ONCE, through the repo's own
// gate, per CLAUDE.md house rule 1 — `just ship-it` (check + container-test +
// `gt submit --stack --no-edit --no-interactive --always`), never a bare
// `gt submit` and never `--force`/a force-push over any remote. The upstream
// per-tip exit-gate runner string is kept as the fallback for a wave launched
// without lanePolicy; args.lanePolicy.exitRunner (same key the cascade port
// takes) overrides it so the zip re-certifies restacked tips with the
// repo's own Python runner instead of the upstream bun script.
// PORT (entropic-substrate-lm, FOR-1547): optional per-wave text inserted at the head of zip
// step (5) — a command the zip agent runs at EVERY restacked tip before re-certifying it
// (this repo: regenerate shims + __legacy forwarders and commit them through the hooks).
const ZIP_RESTACK_HOOK = typeof A.zipRestackHook === 'string' && A.zipRestackHook.trim() ? ' FIRST, before any gate: ' + A.zipRestackHook.trim() + ' Then:' : ''
const ZIP_EXIT_RUNNER = (A.lanePolicy && typeof A.lanePolicy.exitRunner === 'string' && A.lanePolicy.exitRunner) || 'bun src/gate/adapters/run-exit-gate.ts'
const zip = await agent(
  'You are the ZIP (fan-in) agent. Work from ' + A.checkout + '. ' + zipOrderRationale + '\nZIP ORDER (planner-computed, least risk first): ' + plan.zipOrder.join(' → ') + '. Lane worktrees to remove first (branches cannot rebase while checked out elsewhere; verify each is CLEAN with `git status --porcelain` before removing): ' + provision.worktrees.join(', ') + '.\nSteps: (1) `git worktree remove` each clean lane worktree. (2) TEARDOWN SWEEP (ownership-hazard fix): every lane launched with policy.teardown:false, so the cascade\'s own Phase 7 Teardown never pruned its `wf_*` scratch worktrees while sibling lanes were still running under the SAME shared checkout (a lane\'s sweep could otherwise race-remove a still-active sibling\'s clean worktree) - the zip runs strictly after every lane has finished, so this is the one safe place to do it, once, serially: run `git worktree list --porcelain` in ' + A.checkout + '; candidates are every listed worktree path containing `/.claude/worktrees/wf_` directly under the checkout. For each candidate, check `git status --porcelain` inside it - if CLEAN, `git worktree remove` it (NEVER `--force`); if dirty, KEEP it untouched and record its path. Then run `git worktree prune`. Report the removed count and the kept (dirty) count with each kept path in your evidence. (3) In zip order, restack each lane branch onto its predecessor in that order (the first branch restacks onto ' + A.parent + '): prefer `gt restack` when the branch is gt-tracked, else `git rebase --onto <predecessor> <branch\'s current base> <branch>`. (4) A REBASE CONFLICT at any step falsifies the planner\'s edge-disjointness claim: `git rebase --abort` (or the gt equivalent) immediately and return status DECISION naming both branches and the conflicting file(s) - NEVER resolve a conflict by hand, under any circumstance; the wave fails loudly, no exceptions, no partial zip. (5) At each RESTACKED tip (any branch whose SHA changed):' + ZIP_RESTACK_HOOK + ' run the offline gate (' + A.gates.offline + ') and the exit gate via the deterministic runner `' + ZIP_EXIT_RUNNER + ' --cmd ' + JSON.stringify(A.gates.exit) + ' --expect-tip <sha>` - a restacked tree is a NEW tree, prior certificates do not transfer; record the branch\'s 40-hex tip pasted verbatim from `git rev-parse <branch>` output. (6) Once every stack branch is restacked and re-gated, checkout the TOP branch of the stack in ' + A.checkout + ' and run ONE `just ship-it` - this is the ONLY publish for the whole wave. NEVER run `gt submit` directly, NEVER pass `--force` to anything, NEVER force-push or push directly to a remote outside `just ship-it` itself. If `just ship-it` fails, STOP: return status DECISION quoting the failing gate\'s command and its output verbatim - do not retry with a relaxed gate, a skipped step, or a hand-applied fix. (7) After `just ship-it` succeeds, for EVERY branch in the stack verify the remote matches the restacked local tip: run `gh pr view <branch> --json headRefOid -q .headRefOid` and `git rev-parse <branch>`, and quote BOTH values for each branch in your evidence - a mismatch is a FAIL, return status DECISION naming the branch and both SHAs. (8) Report the final stack bottom-up with each branch\'s verified 40-hex tip, and end on the conductor\'s original branch.\nNEVER: merge anything, touch ' + A.parent + ' itself or any branch outside the zip order, amend/reset, force-push, or run `git config user.*`.',
  // Fable at low effort: the zip is a scripted git-topology sequence whose
  // hard cases are all pre-ruled (conflict => fail loud, per-tip re-gates,
  // one ship-it publish) — strong model for faithful execution, minimal
  // deliberation budget.
  { label: 'zip:wave', phase: 'Zip', schema: ZIP, model: PINS.zip.model, effort: PINS.zip.effort },
)
if (!zip || zip.status !== 'ZIPPED') {
  return { verdict: 'DECISION-NEEDED', at: 'zip', decision: (zip && zip.decision) || 'zip agent died', lanes: laneSummary, plan: plan }
}
const zipResultHalt = validateZipResult(zip.stack, zip.tips, plan.zipOrder)
if (zipResultHalt) {
  return { verdict: 'DECISION-NEEDED', at: 'zip', decision: zipResultHalt, lanes: laneSummary, plan: plan }
}
log('zipped: ' + zip.stack.join(' → '))

return {
  verdict: 'WAVE-SUBMITTED',
  lanes: laneSummary,
  deferred: plan.rejected,
  zip: { stack: zip.stack, tips: zip.tips },
  startTip: provision.startTip,
  evidence: 'plan + provision + per-lane lifecycle reports + zip evidence in the journal; conductor reviews lane PRs and owns merges',
}
