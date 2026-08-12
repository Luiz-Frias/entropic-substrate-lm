// LAUNCH RULE: invoke via scriptPath (byte-verified persisted copy), NOT by name —
// the name registry serves a session-start snapshot, so mid-session edits never reach
// a by-name launch. args may arrive as a JSON string (parsed at the boundary below).
// Invariants/gates/policy come VERBATIM from td-phase-lifecycle.registry.json (same dir).
export const meta = {
  name: 'td-phase-lifecycle',
  description:
    'One stacked-sprint phase lifecycle: Sonnet TDD impl (loop-until-green) → DISTINCT L2 gate-conformance + cross-branch integration checks → 3 ROTATED fresh-reviewer rounds (sonnet from mid-r2) (contract / integration / adversarial guard-teeth, budget-scaled) → local CodeRabbit loop → DISTINCT publish agent. Stops at PR-submitted; merges + DECISION-NEEDED stay with the operator.',
  whenToUse:
    'Invoke ONCE per phase/branch of a stacked sprint, STRICTLY SEQUENTIALLY (single shared worktree — never two runs in parallel). Pass {phase, invariants, gates, policy} via args. The conductor reviews each report between invocations, resolves DECISION-NEEDED, owns all merges + Linear writes.',
  phases: [
    { title: 'Implement', detail: 'L3 Sonnet TDD + self-/verify, loop-until-green', model: 'sonnet' },
    { title: 'Gate-check', detail: 'L2 gate-conformance ∥ cross-branch integration (DISTINCT agents)', model: 'sonnet' },
    { title: 'Review', detail: 'rotated fresh-reviewer rounds: A contract · B integration · C adversarial (worktree guard-teeth)', model: 'sonnet' },
    { title: 'LocalCR', detail: 'local CodeRabbit loop — quick-wins applied, rest deferred', model: 'sonnet' },
    { title: 'Publish', detail: 'distinct publish agent: gt submit → remote CR to a REAL finish (tri-state)', model: 'sonnet' },
    { title: 'Validate', detail: 'L1 validator: gates + runtime surface + PR state + advisor consult — evidence attached', model: 'opus' },
  ],
}

// ─── boundary: args may arrive as a JSON string (harness passes it verbatim) ───
let ARGS = args
if (typeof ARGS === 'string') {
  try {
    ARGS = JSON.parse(ARGS)
  } catch (e) {
    throw new Error('td-phase-lifecycle: args arrived as a non-JSON string: ' + e.message)
  }
}

// ─── parameter artifacts (the 3 sprint-tunable inputs + policy) ────────────────
const PH = ARGS && ARGS.phase
if (!PH || !PH.id || !PH.branch || !PH.parent || !PH.scope || !PH.acceptance) {
  throw new Error('args.phase{id,title,issue,branch,parent,scope,acceptance} is required (phase-manifest entry)')
}
const INV = (ARGS && ARGS.invariants) || [] // invariant-registry: strings, each with a tripwire + guard-teeth recipe
const G = Object.assign(
  {
    offline:
      'RUN_LIVE_LLM_TESTS= bun run typecheck && RUN_LIVE_LLM_TESTS= bun run test && RUN_LIVE_LLM_TESTS= bun run lint',
    exit: 'RUN_LIVE_LLM_TESTS= prek run --all-files',
    live: 'doppler run -p sales-collateral-agx -c dev -- bun run test:live',
    liveRequired: false, // gate-config: flip on for phases that touch an LLM-facing seam
  },
  (ARGS && ARGS.gates) || {},
)
const P = Object.assign(
  // maxL2Loops default raised 2→3 (biome-sprint P4: one fixer attempt on a precisely-
  // specified finding is often one too few; the extra round is cache-cheap on resume).
  { maxImplLoops: 3, maxL2Loops: 3, maxReviewLoops: 3, maxCrLoops: 5, submitMode: 'draft' },
  (ARGS && ARGS.policy) || {},
)

// ─── shared context block (every agent gets the same ground truth) ─────────────
const INVARIANTS = INV.length
  ? 'SACRED INVARIANTS (must hold; a break is a FAIL, never a "note"):\n' + INV.map((v) => '- ' + v).join('\n')
  : 'SACRED INVARIANTS: none registered for this phase.'
const BRIEF = [
  'PHASE ' + PH.id + ' — ' + PH.title,
  'Linear issue: ' + (PH.issue || 'n/a') + ' (read it if Linear is reachable; the FILESYSTEM is ground truth on conflict — report divergences, follow reality)',
  'Branch: ' + PH.branch + '  (parent: ' + PH.parent + ')',
  'SCOPE:\n' + PH.scope,
  'ACCEPTANCE (machine-checkable):\n' + PH.acceptance,
  INVARIANTS,
  'GATES:\n  offline: ' + G.offline + '\n  exit: ' + G.exit + (G.liveRequired ? '\n  LIVE (required): ' + G.live : ''),
  'HARD CONSTRAINTS: no merge · no gt fold · no .pre-commit-config.yaml edits · single shared worktree (verify pwd + clean tree before edits; unexpectedly dirty/foreign → STOP, report DECISION) · prefix offline gates, the EXIT gate, AND git commit with RUN_LIVE_LLM_TESTS= · models come from env, never hardcoded · VERIFY-BEFORE-ESCALATE: a DECISION at any stage is malformed unless it attaches the verification you RAN to test the blocking premise (command + observed output, in the evidence/decision text) — verify the claim empirically first; many decisions dissolve under verification.',
].join('\n\n')
// FOR-759 #1: ROUND schema requires evidence (below) — this shared clause is
// appended to every ROUND-producing prompt (L2 gate-check ×2, Review lenses) so
// agents know what satisfies the requirement before they ever hit the schema wall.
const EVIDENCE_REQUIREMENT =
  'EVIDENCE REQUIREMENT (FOR-759, required by schema): your evidence field must carry VERBATIM proof for every check behind your verdict — the exact command + observed output/exit code for anything you ran (gate tails), or exact excerpts (diff hunks, grep hits, file:line) for anything you inspected. A verdict backed only by a summary, or left empty, does not satisfy the schema and is a FAIL on your OWN report.'

// ─── schemas ────────────────────────────────────────────────────────────────────
// FOR-754: exitGate is the runner's JSON relayed VERBATIM (bun scripts/run-exit-gate.ts
// executes the gate itself — the agent is a courier, never a witness). The workflow
// gates on it deterministically below; SUBMITTED is unreachable without it.
const XGATE = {
  type: 'object',
  additionalProperties: false,
  properties: {
    ok: { type: 'boolean' },
    cert: {
      type: 'object',
      additionalProperties: false,
      properties: { cmd: { type: 'string' }, certifiedTip: { type: 'string' }, tailDigest: { type: 'string' } },
      required: ['cmd', 'certifiedTip', 'tailDigest'],
    },
    error: { type: 'object' },
    evidence: {
      type: 'object',
      additionalProperties: false,
      properties: {
        cmd: { type: 'string' },
        exitCode: { type: 'number' },
        testedTip: { type: 'string' },
        tail: { type: 'string' },
        dirty: { type: 'boolean' },
      },
      required: ['cmd', 'exitCode', 'testedTip', 'tail', 'dirty'],
    },
  },
  required: ['ok'],
}
// Deterministic exit-gate verdict — the single predicate every stage reuses.
// Returns null when the evidence certifies `tip`; otherwise the blocking reason.
// EXACT matching only (r3-2, CR hardening): tips must be full 40-hex SHAs (the
// schemas pattern-enforce this at agent-emit time), cert must accompany ok:true,
// and both cert.cmd + evidence.cmd must equal the configured exit gate verbatim.
// PARITY-PIN BEGIN exitGateBlockReason — scripts/exit-gate.test.ts extracts this
// block and asserts behavioral identity with scripts/exit-gate.ts
// runnerReportBlockReason (same verdicts + same strings). Change them together.
const exitGateBlockReason = (xg, tip, exitCmd) => {
  const FULL_SHA = /^[0-9a-f]{40}$/
  if (!xg || xg.ok !== true) return 'exit gate did not certify (missing or ok!==true): ' + JSON.stringify((xg && xg.error) || 'no exitGate relayed')
  if (!xg.cert || !xg.evidence) return 'ok:true without cert+evidence — non-canonical runner output'
  if (xg.evidence.cmd !== exitCmd || xg.cert.cmd !== exitCmd) return 'exit-gate command mismatch: expected ' + JSON.stringify(exitCmd)
  if (xg.evidence.exitCode !== 0) return 'exit-gate evidence nonzero exit code'
  if (!xg.evidence.tail || !String(xg.evidence.tail).trim()) return 'exit-gate evidence tail is empty (silence is not evidence)'
  if (xg.evidence.dirty) return 'exit-gate ran on a dirty tree (unfolded auto-fixes)'
  if (!FULL_SHA.test(xg.evidence.testedTip || '') || xg.cert.certifiedTip !== xg.evidence.testedTip) return 'exit-gate tip not a full canonical SHA or cert/evidence tip mismatch'
  if (!FULL_SHA.test(tip || '')) return 'phase tip is not a full 40-hex SHA: ' + (tip || 'missing')
  if (xg.evidence.testedTip !== tip) return 'exit-gate tip mismatch: certified ' + xg.evidence.testedTip + ' but phase tip is ' + tip
  return null
}
// PARITY-PIN END
// GUARD-PIN BEGIN implGateRetryFeedback — scripts/td-phase-lifecycle-gate-mechanics.test.ts
// extracts this block and unit-tests it directly (FOR-759 #2). A re-run of ONLY the
// exit-gate runner is not enough when the block reason is a dirty post-run tree
// (prek auto-fixed files): the offline gates the implementer ran in step 3 certified
// the PRE-fix tree, not the one about to be amended in. Force a re-run of BOTH gate
// sets on the amended tree before the next GREEN can be trusted. Pure + parameterized.
const DIRTY_TREE_BLOCK_REASON = 'exit-gate ran on a dirty tree (unfolded auto-fixes)'
function implGateRetryFeedback(attemptNum, implGateBlock, offlineCmd, exitCmd) {
  const header =
    '\nPREVIOUS ATTEMPT #' + attemptNum + ' reported GREEN but its exit-gate relay did not certify: ' + implGateBlock +
    '. The WORK likely stands — do NOT reimplement. Verify branch/tree/commit state first, '
  const rerunExit =
    're-run the deterministic runner (`bun scripts/run-exit-gate.ts --cmd ' + JSON.stringify(exitCmd) + '`), and re-emit your FULL report with the runner JSON VERBATIM in the exitGate field and the full 40-hex commit SHA in the commit field.\n'
  if (implGateBlock === DIRTY_TREE_BLOCK_REASON) {
    return (
      header +
      'fold prek\'s auto-fixes into an amended commit, then RE-RUN THE OFFLINE GATES VERBATIM (`' + offlineCmd + '`) ON THE AMENDED TREE — the offline gates from step 3 certified the pre-fix tree, not this one, so they no longer speak for what you are about to submit. Only once offline gates are green again do you ' +
      rerunExit
    )
  }
  return header + rerunExit
}
// GUARD-PIN END implGateRetryFeedback
const IMPL = {
  type: 'object',
  additionalProperties: false,
  properties: {
    status: { enum: ['GREEN', 'RED', 'DECISION'] },
    // full 40-hex only (r3-2): the exit-gate predicate uses EXACT tip equality, so a
    // short SHA is rejected at emit time (schema retry) instead of BLOCKing the phase.
    commit: { type: 'string', pattern: '^[0-9a-f]{40}$' },
    gates: { type: 'string' },
    exitGate: XGATE,
    divergences: { type: 'array', items: { type: 'string' } },
    decision: { type: 'string' },
    evidence: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['status', 'gates'],
}
const ROUND = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { enum: ['PASS', 'FAIL', 'DECISION'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          severity: { enum: ['blocker', 'major', 'minor'] },
          summary: { type: 'string' },
          file: { type: 'string' },
        },
        required: ['severity', 'summary'],
      },
    },
    decision: { type: 'string' },
    evidence: { type: 'string', minLength: 1 },
    notes: { type: 'string' },
  },
  // evidence REQUIRED + non-empty (FOR-759 #1): a gate-check/review verdict with no
  // attached evidence is structurally rejected before any JS control-flow runs on
  // it — mirrors the VAL schema's exitGate-required precedent (FOR-754). minLength
  // closes the "evidence: ''" loophole — an empty string is still "no evidence".
  required: ['verdict', 'findings', 'evidence'],
}
const CR = {
  type: 'object',
  additionalProperties: false,
  properties: {
    applied: { type: 'number' },
    rejected: { type: 'array', items: { type: 'string' } },
    deferred: { type: 'array', items: { type: 'string' } },
    commit: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['applied', 'rejected', 'deferred'],
}
const PUB = {
  type: 'object',
  additionalProperties: false,
  properties: {
    pr: { type: 'string' },
    // full 40-hex only (r3-2): Validate pins the exit gate to this exact tip.
    tip: { type: 'string', pattern: '^[0-9a-f]{40}$' },
    crStatus: { enum: ['CLEAN', 'FINDINGS_APPLIED', 'PENDING', 'FAILED'] },
    actionable: { type: 'number' },
    notes: { type: 'string' },
  },
  required: ['pr', 'tip', 'crStatus'],
}
const VAL = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { enum: ['VALIDATED', 'FAIL'] },
    evidence: { type: 'string' },
    exitGate: XGATE,
    advisor: { type: 'string' },
    findings: { type: 'array', items: { type: 'string' } },
    reconciledDivergences: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: { text: { type: 'string' }, status: { type: 'string' } },
        required: ['text', 'status'],
      },
    },
    notes: { type: 'string' },
  },
  // exitGate REQUIRED (FOR-754): a validation without machine-produced exit-gate
  // evidence is structurally rejected at the schema layer, before any JS gate runs.
  required: ['verdict', 'evidence', 'exitGate'],
}

// ─── Phase 1: Implement — L3 Sonnet TDD, loop-until-green (adaptive, not fixed-N) ─
phase('Implement')
let impl = null
let feedback = ''
let implAttempts = 0
for (let i = 1; i <= P.maxImplLoops; i++) {
  implAttempts = i
  impl = await agent(
    BRIEF +
      '\n\nYou are the L3 IMPLEMENTER (TDD). Work in the repo checkout you are launched in.\n' +
      '1. Verify pwd + `git status` clean; create/switch to the branch (`gt create ' + PH.branch + '` off ' + PH.parent + ', or switch if it exists).\n' +
      '2. TDD: write the failing test FIRST for each acceptance item, then implement to green. Pin 0/empty/blank/error cases at authoring time.\n' +
      '2b. SIBLING-COPY CHECK (W1, hardening sprint): before fixing a helper/pattern, grep for byte-identical or near-identical sibling copies; if siblings exist that the scope does NOT name, do not half-migrate silently — fix only the named copy and REPORT the siblings in divergences (the conductor rules widen-vs-defer).\n' +
      '3. Run the offline gates verbatim. Self-review your own diff (a /verify-quality pass: re-read every hunk against scope + invariants).\n' +
      '4. Commit (conventional message, ≤72-char subject' + (PH.trailer ? ', trailer: ' + PH.trailer : '') + '); report the FULL 40-hex commit SHA in your commit field (`git rev-parse HEAD`).\n' +
      '5. EXIT GATE (deterministic runner — do NOT run the gate by hand or transcribe results): `bun scripts/run-exit-gate.ts --cmd ' + JSON.stringify(G.exit) + '`. It executes the gate, reads the tip, and checks tree cleanliness itself. If prek auto-fixed files it reports DIRTY_TREE: fold the fixes into your commit (amend), then RE-RUN BOTH gate sets ON THE AMENDED TREE — the offline gates (' + G.offline + ') you ran in step 3 certified the PRE-fix tree, not this one, so they no longer speak for what you are about to submit; only once they are green again do you re-run the exit-gate runner so the certificate matches the final committed tip. A commit is never accepted GREEN on a re-run of the runner alone. Relay its ENTIRE JSON output verbatim in your exitGate field — GREEN without a certifying exitGate will be BLOCKED by the harness, not reviewed.\n' +
      'Spawn Haiku scout sub-agents for recon if available. If you hit a MATERIAL contract question the scope does not answer → status DECISION with the question. A dirty or foreign working-tree state at ANY point is ALWAYS status DECISION (SET-AT-CUT: STOP has no third option) — never GREEN with a divergence note. Return status GREEN only when every gate passes.\n' +
      feedback,
    { label: 'impl:' + PH.id + '#' + i, phase: 'Implement', schema: IMPL, model: 'sonnet', effort: 'high' },
  )
  if (!impl) {
    // FB3: agent death (API transient, e.g. 529) consumes a loop attempt instead of
    // terminally blocking — the retry gets a tree-state caution, not a blank slate.
    if (i === P.maxImplLoops) return { verdict: 'BLOCKED', at: 'implement', reason: 'implementer died on final attempt' }
    log('impl attempt #' + i + ' died (API transient?) — retrying with tree-state caution')
    feedback =
      '\nPREVIOUS ATTEMPT #' + i + ' DIED mid-run (likely an API transient), it did NOT fail on the merits. ' +
      'The checkout may hold its partial state: inspect `git status` + branch state FIRST; the branch may already exist (switch, do not re-create); ' +
      'continue from clean committed state and treat uncommitted debris as discardable only if clearly incomplete mechanical work.\n'
    continue
  }
  if (impl.status === 'DECISION') return { verdict: 'DECISION-NEEDED', at: 'implement', decision: impl.decision, partial: impl }
  if (impl.status === 'GREEN') {
    // FOR-754: deterministic exit-gate check — GREEN self-report alone cannot pass
    // this point; the runner's machine-produced evidence must certify the phase
    // commit. r3-3 (hex-refactor sprint): a GREEN report whose exitGate relay is
    // missing or non-certifying now retries IN-LOOP with targeted feedback instead
    // of BLOCKing the traversal — the bare field omission killed 3 of 8 t1s that
    // sprint as pure engine waste (the WORK was conductor-verified fine every
    // time; only the relay was missing). Exhausting the loop still BLOCKs honestly.
    const implGateBlock = exitGateBlockReason(impl.exitGate, impl.commit, G.exit)
    if (implGateBlock === null) break
    if (i === P.maxImplLoops) {
      return {
        verdict: 'BLOCKED',
        at: 'implement:exit-gate',
        reason: implGateBlock,
        evidence: (impl.exitGate && impl.exitGate.evidence) || null,
        last: impl,
      }
    }
    feedback = implGateRetryFeedback(i, implGateBlock, G.offline, G.exit)
    continue
  }
  feedback = '\nPREVIOUS ATTEMPT #' + i + ' was RED. Fix these first (verbatim gate output):\n' + impl.gates
}
if (impl.status !== 'GREEN') return { verdict: 'BLOCKED', at: 'implement', reason: 'gates never converged', last: impl }
log('impl GREEN @ ' + (impl.commit || 'uncommitted?') + ' after ' + implAttempts + ' attempt(s) — exit gate certified')

// ─── Phase 2: Gate-check — L2 as TWO DISTINCT agents (never folded) ─────────────
phase('Gate-check')
// FREEZE-BREAK #r3-8 (2026-07-29, conductor-ruled): args.conductorRuling carries a
// settled answer to a PRIOR DECISION-NEEDED so the phase can be resumed past it. It is
// injected ONLY into the L2 + review prompts, NEVER the implementer's — that asymmetry
// is the whole point: on `resumeFromRunId`, the impl agent's (prompt, opts) stay
// byte-identical so its expensive work replays from cache, while the checkers that
// raised the decision re-run informed by the ruling. Putting the ruling in the shared
// scope/acceptance text instead would invalidate every cache entry including impl's.
const RULING =
  ARGS && ARGS.conductorRuling
    ? '\n\nCONDUCTOR RULING (BINDING — resolves a prior DECISION-NEEDED raised on this phase; treat it as settled contract and do NOT re-litigate it. Verify the code CONFORMS to it; a finding that merely re-raises the ruled question is invalid):\n' +
      ARGS.conductorRuling
    : ''
// FREEZE-BREAK #r3-9 (2026-08-04): reviewer isolation is HARNESS-ENFORCED, ported
// identically from freeze #opus-3 (spawn probe-verified; freeze-break #1's failure
// was a since-fixed WorktreeCreate contract). Isolated agents start on a NON-phase
// checkout, so this note redirects them onto the phase tip, detached.
const ISOLATION_NOTE =
  'ISOLATION (freeze #r3-9 — OVERRIDES any cd-into-worktree instruction in the scope): you run in your OWN isolated git worktree (your starting cwd). FIRST ACTS: git checkout --detach ' +
  PH.branch +
  ' (branch refs are repo-global; detach avoids the two-worktrees-one-branch refusal — if the branch does not exist yet, STOP and return DECISION), then bun install, then (cd web && bun install). Run ALL commands, probes, flip-tests and gate runs HERE. The shared launch worktree (.claude/worktrees/m6-refinement) is READ-ONLY to you: inspect it via git show <ref>:<path> only; NEVER write there, never check anything out there, never run mutating git commands there.\n\n'

const l2Prompts = {
  gate:
    BRIEF +
    '\n\nYou are the L2 GATE-CHECKER (fresh context; you did NOT write this code). Answer ONE question: does branch ' + PH.branch + ' pass its OWN gates + ceremony?\n' +
    '- RE-RUN the offline gates verbatim AND the exit gate via `bun scripts/run-exit-gate.ts --cmd ' + JSON.stringify(G.exit) + '` (do not trust the implementer’s transcript; a non-certifying runner output — ok!==true — is a FAIL finding, quote its error JSON).\n' +
    '- POST-RUN DIRTY-TREE CHECK (FOR-759 #3, independent of the runner’s own dirty flag): immediately AFTER your gate re-runs finish, run `git status --porcelain` yourself — do not rely solely on the runner’s self-reported ok/dirty flag. ANY output (including hook-autofixed-but-uncommitted files the runner might not have caught) is an UNCONDITIONAL FAIL finding; quote it verbatim in evidence. A prek run that mutates the working tree must never silently pass.\n' +
    '- Check /tdd-first-stacked-pr conformance: failing-test-first evidence (tests pin acceptance + edge cases), commit hygiene, scope containment vs the issue.\n' +
    '- Check /gt-stacked-pr-sprint ceremony: branch parentage, one-unit-per-commit, no foreign files.\n' +
    'FAIL on any gate red or ceremony breach. List findings.\n\n' +
    EVIDENCE_REQUIREMENT +
    RULING,
  integration:
    BRIEF +
    '\n\nYou are the L2 INTEGRATION/CONSISTENCY REVIEWER (fresh context; DISTINCT from the gate-checker — do not re-litigate its gates). Answer ONE question: does this branch stay CONSISTENT with its siblings + the trunk?\n' +
    '- Diff ' + PH.parent + '..' + PH.branch + ': trace every changed seam’s consumers (grep callers, check sibling idioms — same error-model, naming, config patterns as the rest of the stack).\n' +
    '- Cross-branch glue: does anything this branch changes contradict a decision already landed below it in the stack?\n' +
    '- Inter-phase considerations: does it leave the NEXT phase a clean base (no half-migrated seams)?\n' +
    'FAIL on drift or contradiction. List findings.\n\n' +
    EVIDENCE_REQUIREMENT +
    RULING,
}
for (let l2Round = 1; l2Round <= P.maxL2Loops; l2Round++) {
  const [gate, integ] = await parallel([
    () => agent(ISOLATION_NOTE + l2Prompts.gate, { label: 'L2:gate#' + l2Round, phase: 'Gate-check', schema: ROUND, model: 'sonnet', effort: 'xhigh', isolation: 'worktree' }),
    () => agent(ISOLATION_NOTE + l2Prompts.integration, { label: 'L2:integration#' + l2Round, phase: 'Gate-check', schema: ROUND, model: 'sonnet', effort: 'xhigh', isolation: 'worktree' }),
  ])
  const both = [
    { key: 'gate', r: gate },
    { key: 'integration', r: integ },
  ]
  const dec = both.find((x) => x.r && x.r.verdict === 'DECISION')
  if (dec) return { verdict: 'DECISION-NEEDED', at: 'gate-check:' + dec.key, decision: dec.r.decision, findings: dec.r.findings }
  const failed = both.filter((x) => !x.r || x.r.verdict === 'FAIL')
  if (!failed.length) break
  if (l2Round === P.maxL2Loops) return { verdict: 'BLOCKED', at: 'gate-check', unresolved: failed.map((x) => x.key), findings: failed.flatMap((x) => (x.r && x.r.findings) || []) }
  const findings = failed.flatMap((x) => (x.r && x.r.findings) || [])
  log('L2 FAIL (' + failed.map((x) => x.key).join('+') + ') — dispatching fixer, ' + findings.length + ' finding(s)')
  await agent(
    BRIEF + '\n\nYou are the L3 FIXER (Sonnet). Apply MINIMAL fixes for these verified L2 findings, keep tests pinned, re-run offline gates to green, and land the fix as a FOLLOW-UP COMMIT — FIXERS-NEVER-AMEND (registry invariant, 2026-08-03): git commit --amend and --no-verify are FORBIDDEN in fixer sessions regardless of branch publish state or apparent authorship; the phase commit belongs to the implementer or a prior stage, and a fixer that believes only an amend can work STOPS and reports instead. A predecessor attempt may have left commits or uncommitted state: inspect git status and the branch reflog before acting; commits from prior stages belong to this workflow — never rewrite them. Scope-owned docs are PART OF THE DELIVERABLE — sync them in the same follow-up commit (a code fix with stale docs is an incomplete fix; three review failures in the biome sprint were docs-drift).\n' + JSON.stringify(findings, null, 2) + RULING,
    { label: 'fix:L2#' + l2Round, phase: 'Gate-check', schema: IMPL, model: 'sonnet', effort: 'high' },
  )
}

// ─── Phase 3: Review — ROTATED fresh-reviewer rounds (sonnet from mid-r2; operator freeze-break 2026-07-07) (independence via distinct lenses),
//     budget-scaled extra adversaries, loop-until-converged ──────────────────────
phase('Review')
// FREEZE-BREAK #r3-7 (2026-07-24, operator-ratified): shared mutation-isolation
// clause for the parallel flip-performers that are NOT Round C (C carries its own
// stronger DO-NOT-trust variant below). Overnight r3-sweep evidence: shared-checkout
// flip-test collisions (phase-1 models.ts, phase-6 thermo-nuclear-review.js) came
// from rounds A/B and the local-CR loop's flip-proofs racing C's reads — C itself
// was already isolated.
const MUTATION_ISOLATION =
  ' MUTATION ISOLATION (W4): reads/analysis happen in the launch checkout, but ANY tree mutation you perform that is not meant to be committed — flip-tests, probe edits, temporary reverts — MUST run in a self-created scratch worktree, NEVER in the launch checkout (parallel reviewers share it; a Write there races their evidence): TIP=$(git rev-parse HEAD); WT="$(mktemp -d)/wt"; git worktree add --detach "$WT" "$TIP"; mutate + prove inside $WT only; when done `git worktree remove --force "$WT"` and verify `git worktree list` no longer shows it (a leaked scratch worktree is a FAIL in your own report). Zero mutations performed = clause satisfied trivially.'
const LENSES = [
  {
    key: 'A-contract',
    brief:
      'ROUND A — GATE/CONTRACT CONFORMANCE. Fresh context, implementation-blind. Verify each acceptance item against the ACTUAL diff (read the code, not the report). Verify owned contracts changed exactly as scoped and no unscoped contract drifted. Re-run typecheck yourself.' +
      MUTATION_ISOLATION,
  },
  {
    key: 'B-integration',
    brief:
      'ROUND B — CROSS-BRANCH INTEGRATION. Fresh context. Trace every changed seam to its consumers ACROSS the stack; verify sibling-idiom consistency and that downstream phases inherit a coherent base. Different failure modes than Round A — do not repeat its checks. Evidence integrity: if the shared checkout shows foreign churn mid-review, wait for quiescence (git status clean + stable tip), then re-collect the affected evidence; report the churn as a finding. Scope-owned doc/comment drift left contradictory by this phase is a FAIL finding (the fixer will apply it), not a minor note.' +
      MUTATION_ISOLATION,
  },
  {
    key: 'C-adversarial',
    brief:
      'ROUND C — ADVERSARIAL INVARIANT-BREAK (guard-teeth). DO NOT trust any pre-provided isolation. FIRST self-isolate: TIP=$(git rev-parse HEAD); WT="$(mktemp -d)/wt"; git worktree add --detach "$WT" "$TIP"; cd "$WT"; run bun install there if the gates need it; then VERIFY $(git rev-parse --show-toplevel) differs from the launch checkout toplevel — if identical, return FAIL immediately (isolation unavailable). Do ALL mutation inside $WT; NEVER mutate the launch checkout. For each sacred invariant + each new guard/test this phase adds: FLIP it (re-introduce the bug, silent fallback, or invariant breach) and PROVE the gate/test goes RED. A guard that stays green when flipped is a FAIL finding. When done: cd out, `git worktree remove --force "$WT"`, then VERIFY `git worktree list` no longer shows it (W3: a leaked scratch worktree is a FAIL in your own report).',
    worktree: true,
  },
]
const extraAdv = budget.total ? Math.min(2, Math.floor(budget.remaining() / 200000)) : 0
for (let n = 1; n <= extraAdv; n++) {
  LENSES.push({
    key: 'C-extra-' + n,
    brief:
      'ADDITIONAL ADVERSARIAL REFUTER #' + n + ' — attack a surface Round C likely skipped (concurrency/partial-failure paths, env/config edges, serialization boundaries). Same self-isolation rules as ROUND C: create your own throwaway git worktree with the toplevel guard; flip, prove RED; never mutate the launch checkout.',
    worktree: true,
  })
}
log('review lenses: ' + LENSES.map((l) => l.key).join(', ') + (extraAdv ? ' (+' + extraAdv + ' budget-scaled)' : ''))

let pendingLenses = LENSES
const roundsSummary = {}
for (let round = 1; round <= P.maxReviewLoops && pendingLenses.length; round++) {
  const results = await parallel(
    pendingLenses.map((l) => () =>
      agent(ISOLATION_NOTE + BRIEF + '\n\n' + l.brief + '\nReturn verdict PASS/FAIL (+ findings) or DECISION for a material contract question.\n\n' + EVIDENCE_REQUIREMENT + RULING, {
        label: 'round:' + l.key + '#' + round,
        phase: 'Review',
        schema: ROUND,
        model: 'sonnet',
        effort: 'xhigh',
        isolation: 'worktree', // FREEZE-BREAK #r3-9: restored — hook contract fixed, spawn probe-verified 2026-08-04
        // FREEZE-BREAK #1 (hardening sprint, 2026-07-04): isolation:'worktree' removed —
        // the WorktreeCreate hook can fail at spawn ("no worktree_path in hook input"),
        // killing the C lens before it runs (3× NULL→FAIL, S7-b1 wf_5ae98f2d). Round C's
        // brief SELF-ISOLATES (FB1) with a FAIL-loud toplevel guard; that layer is the
        // load-bearing isolation and travels inside the brief on any machine.
      }),
    ),
  )
  const paired = pendingLenses.map((l, i) => ({ lens: l, r: results[i] }))
  const dec = paired.find((x) => x.r && x.r.verdict === 'DECISION')
  if (dec) return { verdict: 'DECISION-NEEDED', at: 'review:' + dec.lens.key, decision: dec.r.decision, findings: dec.r.findings, rounds: roundsSummary }
  paired.forEach((x) => {
    roundsSummary[x.lens.key] = x.r ? x.r.verdict + ' (' + x.r.findings.length + ' findings)' : 'NULL→FAIL'
  })
  const failed = paired.filter((x) => !x.r || x.r.verdict === 'FAIL')
  if (!failed.length) {
    pendingLenses = []
    break
  }
  if (round === P.maxReviewLoops) return { verdict: 'BLOCKED', at: 'review', unresolved: failed.map((x) => x.lens.key), findings: failed.flatMap((x) => (x.r && x.r.findings) || []), rounds: roundsSummary }
  const findings = failed.flatMap((x) => (x.r && x.r.findings) || [])
  log('review round ' + round + ': ' + failed.length + ' lens(es) FAIL — fixer dispatched, re-running only failed lenses')
  await agent(
    BRIEF + '\n\nYou are the L3 FIXER (Sonnet). Apply MINIMAL fixes for these verified review findings, keep gates green, and land the fix as a FOLLOW-UP COMMIT — FIXERS-NEVER-AMEND (registry invariant, 2026-08-03): git commit --amend and --no-verify are FORBIDDEN in fixer sessions regardless of branch publish state or apparent authorship; if you believe only an amend can work, STOP and report instead. A predecessor attempt may have left commits or uncommitted state: inspect git status and the branch reflog before acting; commits from prior stages belong to this workflow — never rewrite them. Scope-owned docs are PART OF THE DELIVERABLE — sync them in the same follow-up commit (a code fix with stale docs is an incomplete fix).\n' + JSON.stringify(findings, null, 2) + RULING,
    { label: 'fix:review#' + round, phase: 'Review', schema: IMPL, model: 'sonnet', effort: 'high' },
  )
  pendingLenses = failed.map((x) => x.lens) // re-run ONLY the failed lenses, fresh context each time
}

// ─── Phase 4: Local CodeRabbit loop (bounded) ───────────────────────────────────
phase('LocalCR')
const cr = await agent(
  BRIEF +
    '\n\nYou are the LOCAL CODE-REVIEW agent (fresh context). Run the local CodeRabbit review (/coderabbit:code-review or `coderabbit review --plain`) on the phase diff, at most ' + P.maxCrLoops + ' loops:\n' +
    '- APPLY zero-behavior-risk quick-wins as FOLLOW-UP COMMITS on the phase branch (FIXERS-NEVER-AMEND: no git commit --amend, no --no-verify; keep gates green).\n' +
    '- REJECT false positives with empirical evidence (known FP: camelCase Zod-schema naming — AGX convention).\n' +
    '- DEFER heavy/behavior-risk items with a one-line ticket recommendation each.\n' +
    'Never expand scope. Report applied/rejected/deferred + the final commit SHA.' +
    MUTATION_ISOLATION +
    ' CARVE-OUT: fix-application edits you intend to land as follow-up commits on the phase branch are the exception — they belong in the launch checkout; only flip-proofs and probe/revert mutations go to the scratch worktree.',
  { label: 'local-cr', phase: 'LocalCR', schema: CR, model: 'sonnet', effort: 'xhigh' },
)
if (!cr) return { verdict: 'BLOCKED', at: 'local-cr', reason: 'CR agent died' }

// ─── Phase 5: Publish — DISTINCT fresh agent; stop at PR-submitted ──────────────
phase('Publish')
const pub = await agent(
  BRIEF +
    '\n\nYou are the PUBLISH agent (fresh context, distinct from all reviewers). Stateful steps, in order:\n' +
    '1. `gt submit --' + P.submitMode + ' --no-interactive` for ' + PH.branch + '.\n' +
    '2. gt --no-interactive leaves the PR body as the empty template → fill it via `gh api --method PATCH .../pulls/<N> -F body=@file` (summary, Linear ref ' + (PH.issue || '') + ', test plan).\n' +
    '3. Stacked PRs need a trigger: post `@coderabbitai review`.\n' +
    '4. Poll the CR summary to a REAL finish (check PR *reviews* as well as issue comments — the actionable line often lives in the review body): trust ONLY an "Actionable comments posted: N" / "No actionable comments" line. "Review finished" alone can be a Fair-Usage-throttle false-green; W2 RECOVERY: if the throttle names a reset window ("next review in N minutes") and N fits within your remaining cap, WAIT IT OUT interleaved with other verification, then re-trigger ONCE — a real finish after the window beats a PENDING. "Review failed" is a retriable infra transient (one `@coderabbitai full review` retry). Cap ~10 min total → crStatus PENDING (never false-green).\n' +
    '5. If actionable findings land: triage (apply quick-wins as FOLLOW-UP COMMITS + re-submit — the branch is on origin, amend and force are forbidden — defer the rest), at most ' + P.maxCrLoops + ' loops.\n' +
    'HARD RULES: NEVER merge, NEVER gt fold, never touch other branches. Report PR#, tip SHA, crStatus, actionable count.' +
    '\nLOCAL-CR DISPOSITIONS (BINDING PRECEDENT, freeze #r3-9): the local CR stage REJECTED or DEFERRED the items below with evidence; a remote CR finding matching one INHERITS that disposition (cite it in your triage) — never re-apply or re-litigate: REJECTED: ' +
    JSON.stringify(cr.rejected || []) +
    ' DEFERRED: ' +
    JSON.stringify(cr.deferred || []),
  { label: 'publish', phase: 'Publish', schema: PUB, model: 'sonnet', effort: 'high' },
)
if (!pub) return { verdict: 'BLOCKED', at: 'publish', reason: 'publish agent died' }
log('phase ' + PH.id + ' SUBMITTED — PR ' + pub.pr + ' @ ' + pub.tip + ' (CR: ' + pub.crStatus + ')')

// ─── Phase 6: Validate — L1 validator gates the SUBMITTED phase with evidence ───
phase('Validate')
const val = await agent(
  BRIEF +
    '\n\nYou are the L1 VALIDATOR (fresh context, distinct from every prior agent; you fix NOTHING — validation only). The phase is SUBMITTED: PR ' + pub.pr + ' @ tip ' + pub.tip + '. Validate it end-to-end; every check gets EVIDENCE (the command you ran + observed output) in your evidence field:\n' +
    '1. Re-run the offline gates verbatim; record each command, its exit code, and tail output for the tested tip. Then the exit gate via the deterministic runner: `bun scripts/run-exit-gate.ts --cmd ' + JSON.stringify(G.exit) + ' --expect-tip ' + pub.tip + '` — relay its ENTIRE JSON output verbatim in your exitGate field (required; the harness blocks SUBMITTED on missing, stale-tip, dirty-tree, empty-tail, or nonzero evidence).\n' +
    '2. RUNTIME surface verification: find the smallest path that EXECUTES the changed code at a real surface (boot the demo/dev server, drive the CLI, hit the route) and observe it. Tests/typecheck do NOT count as runtime verification. If the diff has NO runtime surface (config/docs-only), record SKIP with the reason.\n' +
    (G.liveRequired ? '3. LIVE gate (required for this phase): run `' + G.live + '`; record the tail.\n' : '3. Live gate not required for this phase — note and skip.\n') +
    '4. PR state: head SHA equals local branch tip; correct base branch; draft status as expected.\n' +
    '5. ADVISOR probe (conditional, freeze #r3-9): run ONE ToolSearch query for the advisor tool. If ABSENT — the expected state on a Mythos-tier session per the operator config — record advisor: N/A (Mythos-tier posture) and SKIP this step entirely: no semaphore, no retries, no waiting. Only if the tool exists: acquire the shared semaphore (W4): `until mkdir /tmp/agx-advisor-lock 2>/dev/null; do if [ -n "$(find /tmp/agx-advisor-lock -maxdepth 0 -mmin +3 2>/dev/null)" ]; then rmdir /tmp/agx-advisor-lock 2>/dev/null; else sleep 10; fi; done`; call it once; `rmdir /tmp/agx-advisor-lock` IMMEDIATELY after it returns; advice is COUNSEL, not command — weigh it against your own evidence and record the material points.\n' +
    '6. DIVERGENCE RECONCILIATION (freeze #r3-9): re-test EVERY Phase-1 impl divergence below against the DELIVERED tip by reading the code — never trust the report text. Return reconciledDivergences as [{text, status}] with status RESOLVED-BY-FIXER(<7-char sha>) or STILL-OPEN; only STILL-OPEN items become conductor rulings. Impl divergences to reconcile: ' + JSON.stringify(impl.divergences || []) + '\n' +
    '7. STOP-CLASS SCAN (freeze #r3-9): if any reconciled divergence or your own checks reveal a dirty tree, foreign or uncommitted files, or a wrong-checkout state, the verdict CANNOT be VALIDATED — return FAIL carrying it as the finding. GREEN and a STOP-class condition are mutually exclusive.\n' +
    'Verdict VALIDATED only if every non-skipped check passes with evidence; any failed check → FAIL with findings.',
  { label: 'validate:' + PH.id, phase: 'Validate', schema: VAL, model: 'opus', effort: 'xhigh' },
)
if (!val) return { verdict: 'BLOCKED', at: 'validate', reason: 'validator died', publish: pub }
if (val.verdict === 'FAIL') {
  return { verdict: 'BLOCKED', at: 'validate', findings: val.findings || [], evidence: val.evidence, publish: pub, phase: PH.id, branch: PH.branch }
}
// FOR-754: the SUBMITTED transition is gated on machine-produced exit-gate evidence
// certifying the PUBLISHED tip — a VALIDATED verdict with missing, stale-tip,
// changed-tip, dirty-tree, empty-tail, or nonzero evidence cannot reach the return
// below (AC: "cannot return SUBMITTED for missing/stale/changed/nonzero evidence").
const valGateBlock = exitGateBlockReason(val.exitGate, pub.tip, G.exit)
if (valGateBlock) {
  return {
    verdict: 'BLOCKED',
    at: 'validate:exit-gate',
    reason: valGateBlock,
    evidence: (val.exitGate && val.exitGate.evidence) || null,
    publish: pub,
    phase: PH.id,
    branch: PH.branch,
  }
}
log('phase ' + PH.id + ' VALIDATED — ' + (val.notes || 'all checks green') + ' — exit gate certified @ published tip')

return {
  verdict: 'SUBMITTED', // merge + Linear reconciliation stay with the operator/conductor
  phase: PH.id,
  branch: PH.branch,
  commit: (cr && cr.commit) || impl.commit,
  implAttempts: implAttempts,
  rounds: roundsSummary,
  localCr: { applied: cr.applied, rejected: cr.rejected, deferred: cr.deferred },
  publish: pub,
  validation: { verdict: val.verdict, evidence: val.evidence, advisor: val.advisor || 'n/a' },
  exitGate: val.exitGate,
  divergences: (val.reconciledDivergences || []).filter((d) => typeof d.status === 'string' && d.status.startsWith('STILL-OPEN')).map((d) => d.text),
  reconciledDivergences: val.reconciledDivergences || (impl.divergences || []).map((t) => ({ text: t, status: 'UNRECONCILED (relayed from impl; may be stale)' })),
}
