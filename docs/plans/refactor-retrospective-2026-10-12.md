# Refactor retrospective: planned versus actual, p0 to p14 and the H horizon

Written 2026-10-12 at `main` 65542d4 (tag v0.2.0). Conductor synthesis of two read-only worker
reports kept beside this file in the session archive (`retro/part-a-hex-refactor.md`,
`retro/part-b-p14-northstar.md`); every figure below is traceable to a named source there.

Evidence labels: **[M]** measured from a repo object, ledger row, PR body or Linear read on
2026-10-12; **[not found]** no record exists in any source; **[unverified]** not checked.
Numbers are never estimated. Where two sources disagree, both are named.

## 1. Scope and sources

| stage | planned in | executed | landed on `main` as |
| --- | --- | --- | --- |
| Hexagonal refactor p0 to p13 | plan-mode plan 2026-09-17 (`~/.claude/plans/dreamy-questing-scroll.md`), `docs/plans/src-hexagonal-refactor.plan.md` | 2026-09-18 to 2026-09-26, PRs #285 to #304 on `refactor/hexagonal-redesign` | promotion chain 9c595a7 to `dev` to `main` 0e55d9d, 2026-09-27 [M] |
| H0 exit-gate probe wave | `docs/hierarchical-latent-predictor.plan.md` H0 exit criterion | 2026-09-27 to 2026-09-28, RunPod pods | no run completed; evidence under `refactor-survey-2026-09-17/exit-gate/` [M] |
| p14 consolidation | `docs/plans/p14-consolidation.plan.md`, `.claude/workflows/sprint-manifest.json` | 2026-09-28 to 2026-10-10, PRs #310 to #336 on `refactor/p14-consolidation` | merge 3ca01bd, release v0.2.0 65542d4, 2026-10-12 [M] |

North star and Horizon in `CLAUDE.md` are unchanged since c1b559e on 2026-09-15 [M, `git log -L`].

## 2. TL;DR

| question | answer | evidence |
| --- | --- | --- |
| Did the structural refactor land as planned? | Yes in content. All 15 phases landed, plus five unplanned PRs (FOR-1523, FOR-1515, the p5b rail, p7b, w2-reconcile). | part A §B [M] |
| Did it run as planned? | No. Of five phases planned to run in parallel, three did; the planned publish path failed from p4 on; the measurement oracle stopped working mid-sprint; the per-wave triage step never ran. | part A gaps 2 to 5 [M] |
| Did the topology goal hold? | At f7b8a21 the committed baseline measures 1 layer inversion and 4 package cycles against 142 and 11 at the survey epoch. The plan's 4 / 5 target was measured with a mapping this repo cannot reproduce, so the two figures are not a like-for-like comparison. | `docs/plans/measurements/p13-sweep-remap.txt` [M]; part A gap 4 |
| Did p14 land as planned? | Yes in content: 15 phases, 6 pre-phase rungs, 2 unplanned branches, 4 follow-ups. The Lean tail, rung 7 FOR-1623 and FOR-1614 commit 2 never happened. | part B §A [M] |
| Did the certified history survive? | Trees yes, SHAs no. Every certified tree is byte-identical to what landed, but most certified tip SHAs are not ancestors of `main`; #333 to #336 landed as single-parent commits, not the ruled `--no-ff` sweep. | part B §A.4 [M, `git merge-base --is-ancestor`, tree compare] |
| Is the Horizon met? | No. The furthest remote run reached global step 2000 of 6000 on one H100 arm. No H capture, preregistration, analysis, review or verdict exists. | part B §D, §E [M] |
| Is `CLAUDE.md` current? | Four command and path lines are provably wrong; the status paragraph is under-informative; one line contradicts `docs/architecture.md`. | §6 below [M] |

## 3. Stage 1: hexagonal refactor, planned versus actual

| phase | planned | actual | gate actually used | deviation that forced it |
| --- | --- | --- | --- | --- |
| p0 | serial tracer, 0 moves | serial, 12 resumes, #285 | lifecycle publish, CI green | closure count 52 to 73, sentinel 15 to 18, CI regime rows added |
| p1 | serial | serial, #286 | lifecycle publish | closure 74 once `paths.py` joined |
| p2 | serial, 3 edits | serial, 13 edits, #287 | validator cured, publish | E-1 widened 2026-09-20 |
| FOR-1523 | not planned | inserted between p2 and p3, #288 | draft PR | operator request |
| p3 | serial, 330 shims | serial, 332 shims, 4 launches, #289 | pre-push suite | typed `.py` + `.pyi` shim pairs, facade copies emptied; CodeRabbit skipped at 663 files |
| p3b | serial, live gate | serial, 4 launches, 8 resumes, #290 | live gate run ids in PR body | fixer hook bypass produced an 84-char title; FOR-1536 deny hook followed |
| p4 | serial, 61 moves | serial, 62 moves, commit ladder, #291 | `just ship-it` gates green; `gt submit --stack` failed on FOR-1541 | implementer declined a single-shot 61-module phase |
| FOR-1515 | not planned | fix on `dev`, cherry-picked, #292 | draft PR to `dev` | forward-port ruling D-FWDPORT |
| p5 | serial, 14 + 2 splits | serial, #293 | exit and live gate green | budget-monotone arm landed then retracted |
| p5b | not planned | new serial rail, #294 | live 5/3/3, CI green | `ladder_campaign` artifact half had 14 reader-set importers |
| p6, p7 | wave w1, K=3 | ran serial, #295 #296 | live 11/11 | p7 imports the `observation` facade p6 moved (D-WAVES r3) |
| p7b | not planned | conductor rail, #297 | `just ship-it`, FOR-1541 fallback | reader label and `--verify` hardening |
| p8, p9, p11 | wave w2, K=3 | ran parallel, #298 #299 #300 | lanes CERTIFIED; zip's one `ship-it` failed on FOR-1541 | sibling-target imports: forwarders plus W2-COMMIT-LOCK after a host livelock |
| w2-reconcile | not planned | serial rung, #301, 104 lines in 42 files | lifecycle CERTIFIED, fallback publish | 34 forwarders had to be rewritten |
| p10 | serial | serial, 5 resumes, #302 | gates green, fallback publish | `gt submit` hung at push; FOR-1558 filed |
| p12 | serial, 57 moves | serial, 44 moves, #303 | container 5/3/3, fallback publish | `__legacy` kept 10 generator files, not empty |
| p13 | serial, 395 test files | three CERTIFIED chunks, 427 test files, #304 | container-test at db23213, fallback publish | implement stage returned DECISION twice; CodeRabbit skipped at 758 files |

Merge style: PRs #285 to #304 landed on the trunk as two-parent merge commits whose subject is the
PR title [M, `git cat-file`]; the ledger's "squash-merged" wording is wrong. Five landing commits
(p1, p2, p3b, p5, p8) carry a first-commit subject instead of a phase title, so `git log
--first-parent` misdescribes them [M].

### Rulings not honored or without record

| ruling | planned | what happened | source |
| --- | --- | --- | --- |
| O-1 triage half | `/pr-stack-comment-triage` after every wave publish | zero ledger rows; review came from local CodeRabbit loops, a remote CodeRabbit that skipped the two largest PRs, and the p14 thermo review | part A §C1 [M, grep over 361 rows] |
| O-1 publish half | lanes CERTIFIED, zip publishes once via `just ship-it` | the one `ship-it` ran and failed on FOR-1541; branches published one by one; later phases used `git push` + `gh pr create --draft` | part A gap 2 [M] |
| E-9 | record `run/construction` as a second composition site in `docs/architecture.md` | no such record at 9c595a7 or 65542d4 | [not found] |
| E-5 | image re-pin once at p13 | re-pinned twice the same day (bbfbceb, then 98233be in the exit-gate lane) and again today from 65542d4 | [M] |
| plan precondition line 808 | sprint starts only after H0's lineage is closed | the refactor ran first; H0 probes ran afterwards as its exit gate and did not close | part A gap 1 [M] |
| verification promises | cascade port proven on a docs-only phase; `remap_check` reproduces 142 / 11 as a control | no record of either run | [not found] |

Token cost data points measured at the time: p0 about 24M subagent tokens over 12 resumes; each
design-gate halt about 300K; the p14 thermo review 16.1M, roughly sixteen times its estimate [M,
memory and ledger rows named in part A §D].

## 4. Stage 2: p14 consolidation, planned versus actual

| planned | actual | source |
| --- | --- | --- |
| 4 pre-phase rungs (FOR-1567, 1617, 1614, 1615) | 6 landed (#310 to #315); rung 7 FOR-1623 never launched; FOR-1614 commit 2 (strict-trim flip) never landed because the audit found 0 untrimmed records in 461 files | part B §A.1 [M] |
| 15 phases: rail, w1 (3), w2 (3), serial capture-sink, w3 (3), w4 (2), two serial tails | all 15 landed (#316 to #331); w1 and w2 as waves, w3 as a wave of 3, w4 as K=2 | part B §A.2 [M] |
| merges between waves | ruled 2026-10-01: stack linearly, one `--no-ff` sweep; executed 2026-10-08 for #320 to #332 | GOAL memory, ledger row 158 [M] |
| no extra branches | two added: FOR-1541 hook wrapper #329 and the docs sweep #332 | [M] |
| Lean tail after z-identity, DN-LEAN-BOX timeboxed | never launched; DN-LEAN-BOX never set | [M, no launch row] |
| four follow-ups stacked on #333, one `--no-ff` sweep | #333 to #336 landed as single-parent commits fc31214, 008c179, 58e52a3, ecc6e88; the DN-8DB6295-MSG reword 5c521f7 is therefore absent from `main` | [M, `git cat-file`] |

Ratchet counts, epoch 4e0ae6a versus 65542d4 [M, same grep both sides]:

| count | epoch | now |
| --- | --- | --- |
| files defining `_pairs` and siblings | 36 | 2 |
| `type: ignore[arg-type]` | 19 | 9 |
| `separators=(",", ":")` | 68 | 64 |
| `def _mapping(` | 23 | 23 |
| `composition/entrypoints/cli.py` lines | 1144 | 1150 |
| layer inversions / package cycles | 1 / 4 | 1 / 4 |

DECISION-NEEDED register: eight operator rulings in the plan and eight recorded only in the ledger
or memory, all executed except DN-LEAN-BOX (never set) and the `--no-ff` landing of the
follow-up stack. Full table in part B §B.

Housekeeping drift found: the sprint manifest still reads `lifecycle: queued` with every phase
`queued`; the plan's post-merge paragraph still calls FOR-1632 and FOR-1626 Todo; the DN-1626 row
says rows were repointed when they were removed; CHANGELOG 0.2.0 lists FOR-1579 three times;
ledger rows 99 and 100 are not JSON [M].

## 5. The H horizon: what is proven and what is not

Horizon sentence one, clause by clause [M, part B §D.1]:

| clause | local evidence at 65542d4 | remote evidence | unproven |
| --- | --- | --- | --- |
| causal encoder, predicted latent, text decoder | `tests/test_sentence_model.py`, `tests/test_sentence_objective.py`, tracer `composition/wiring/h_tiny_tracer.py` | both carriers passed prediction step 1 at solver budget 64 on H100, 2026-09-28 | no run completed the path end to end |
| generates multiple continuations | `tests/test_sentence_rollout.py` | none | no captured generation from a trained model |
| without future target latents | the two tests above plus `tests/test_predictive_state_state.py` | none | no causality audit on a trained checkpoint |
| survives checkpoint/resume | `tests/test_sentence_checkpoint.py`, `tests/test_sentence_runner.py` | never exercised; the H100 retry was a fresh start | remote resume |
| capture, fresh analysis, review, filing, gate | capture custody of failure bytes verified | none | no H capture, analysis, review, verdict or gate; `docs/verdicts/` holds E-series files only |

What the probe waves measured: a3c74e on 2026-09-27 failed both arms at prediction step 1 with
`solver_nonconvergence` at budget 32 (cost 1.00 USD); the CPU diagnosis showed width 32 needs 43
to 45 conjugate-gradient iterations; preflight at budget 64 passed; the b64 wave on 2026-09-28
reached global step 2000 on the sphere arm and at least 1022 on the magnitude arm before the
operator deleted the pods, because each 250-step checkpoint publication stalled about 4.5
minutes and 6000 steps would not fit the 10800 s controller deadline [M, part B §E.1].

Fixes landed since, none yet measured on a GPU: FOR-1567 history-independent publication, FOR-1617
graceful stop, FOR-1621 final-leg publication, FOR-1622 async MLflow sink, FOR-1578 single
`create_run`, four reader-identity re-pins [M].

Preconditions for the next wave as of 2026-10-12 16:00Z:

| precondition | status |
| --- | --- |
| image built from the tip the worker must match | built from 65542d4 today: index `fd10b3fb...`, amd64 manifest `sha256:762ed810ebaef60a2981eac1e7a78ffa271f7e46b71feaf557fd4b535daa0a9a`; native validation passed (exact tree match, 491 documents, corpus identity `cf938a0b` unchanged); capture `results/h-image-publication-65542d4.dvc` pushed; re-pin commit cb2d852 on `fix/h-image-repin-v0.2.0` [M] |
| RunPod template bound to that digest | bwkvsm8tq0 still names the old image `f2469dc9`; a new template is required [M, get-template] |
| six RunPod secrets | present [M, list-secrets] |
| preflight PASS per profile at the new tip | not re-run; `SolverConfig.max_iterations` default is still 32, budget 64 lives only in the wave profiles, no H0 profile is committed [M] |
| FOR-1567 and FOR-1617 landed | yes, #310 and #311 [M] |
| balance said aloud, fresh GO | balance 13.57 USD, spend 0/h, zero pods at 15:45Z; no GO given [M, `runpodctl user`] |

The E-series status paragraph in `CLAUDE.md` ("local generative path is implemented; complete
flywheel integration and remote validation remain open") is true but under-informative, and
`docs/architecture.md` line 3 still says the sentence-first design "is not yet implemented"; one
of the two is stale [M].

## 6. `CLAUDE.md` reconciliation

| line | text | finding | disposition |
| --- | --- | --- | --- |
| 70 | `just check` = fmt-check + lint + typecheck + unit tests (= CI) | the recipe also runs `lint-imports`; CI runs basedpyright and lint-imports | factual fix applied in the docs commit |
| 73 | `tests/test_kappa.py::test_name` | that file never existed in any ref; kappa tests are `test_kappa_geometry.py`, `test_kappa_norm.py`, `test_kappa_autocast_island.py` | factual fix applied |
| 74 | `exp ∈ {..., tiny}` | `train --exp tiny` exits 2 "invalid choice"; `tiny` is the `--tiny` flag; the set has 18 names in `kernel/runtime/config.py` | factual fix applied |
| 157 | "commit 82e584e" | 82e584e is not an ancestor of `main`; the landed commit with the identical tree is 3648a00 | factual fix applied |
| 75 | `models/e3-latest.pt` | no such local file; `models/` holds `e3-tiny-final.pt` | left as an example path; noted |
| 95 | `CLAUDE.local.md` is never loaded | one worker observed its contents in context on 2026-10-12; the 2026-08-14 sentinel probe should be re-run with a control before editing | DECISION-NEEDED |
| 137-139 | status paragraph | true but under-informative; proposed replacement below | DECISION-NEEDED (status text is the operator's) |
| 5-36, 38-63 | North star, Horizon | unchanged since 2026-09-15; no clause falsified; the Horizon's remote half is unmet | no change proposed |

Proposed status paragraph (not applied):

> The current forward architecture is the codec-separated predictor above. Its local generative
> path is implemented and unit-tested. Remote validation is partial: on 2026-09-28 both carriers
> passed prediction step 1 at solver budget 64 on H100 and one arm reached global step 2000 of
> 6000 before teardown. No H run has completed, and no H capture, analysis, review or verdict
> exists. The deploy image is rebuilt from the v0.2.0 tip; a plumbing probe on it is the next step.

## 7. What remains (Linear, measured 2026-10-12)

Project "entropic-substrate-lm - hexagonal refactor": p0 to p13 milestones at 100 percent; p14
consolidation at 91 percent with 4 Backlog items; Bug Squashing at 48 percent with 27 Backlog
items; zero issues Todo or In Progress; FOR-1633, FOR-1632, FOR-1626, FOR-1636 Done today.

| theme | tickets |
| --- | --- |
| cascade engine defects | FOR-1629 (fixer GREEN while its commit still runs), FOR-1558 (CR stage re-runs on ruling change) |
| observability | FOR-1563, FOR-1552, FOR-1630, FOR-1631 |
| H worker and probes | FOR-1624, FOR-1625, FOR-1525, FOR-1616, FOR-1623 (status unverified, not in the Backlog listing) |
| tooling and gates | FOR-1514, FOR-1546, FOR-1518, FOR-1549, FOR-1551, FOR-1555, FOR-1565, FOR-1564, FOR-1619 |
| p14 leftovers and carried debt | FOR-1612, FOR-1609, FOR-1608 (ruled closed, still Backlog), FOR-1598, FOR-1634, FOR-1635, FOR-1627, FOR-1628, FOR-1560, FOR-1561, FOR-1540 |

Open work with no ticket: FOR-1614 commit 2; the Lean tail and DN-LEAN-BOX; the manifest status
fields; the `docs/architecture.md` line 3 contradiction.

## 8. Lessons, stated as principles

1. **Preconditions are gates, not prose.** The plan said the sprint waits for H0's lineage to close; nothing enforced it and the order inverted. A precondition that matters needs a machine check at launch.
2. **A publish path is unproven until it has run on a published chain.** `just ship-it` passed its gates every time and failed its last step from p4 to p13 on one hook. The fix took until 2026-10-05. Test the full path on a throwaway published branch before the first real phase.
3. **"Disjoint by construction" is a measurement, not a property.** Reader-identity membership and sibling-target imports each falsified it; both were measurable before launch with the tools that existed. Measure edge-disjointness per wave from the live tree, then plan K.
4. **The oracle must survive the tree it measures.** `remap_check` died at the quarantine commit and the topology target was never compared like for like. Keep the measurement instrument green as a per-phase postcondition.
5. **A dropped review step leaves no trace unless the ledger demands it.** O-1's triage half never ran and nothing flagged the absence. Required steps need a ledger row type whose absence fails the next launch.
6. **Cite trees, not tips, or merge in a way that keeps tips.** Rebase and squash landings preserved every certified tree and discarded every certified SHA, so PR bodies, the ledger and `CLAUDE.md` now point outside `main`. Either land with `--no-ff` or record tree hashes beside tips.
7. **Landing subjects are documentation.** Five trunk commits describe their first commit, not their phase. The merge subject deserves the same lint as the commit subject (FOR-1565).

## 9. Next steps

| order | step | owner |
| --- | --- | --- |
| 1 | merge the image re-pin PR so release-please cuts 0.2.1 with a deploy pin that matches its tree | operator |
| 2 | rule on the probe wave: GO with the balance stated, GPU, and 1K-step plumbing probe versus a 6K pair; create the RunPod template for the new digest; preflight at the new tip | operator GO, conductor executes |
| 3 | rule on the `CLAUDE.md` status paragraph and the `CLAUDE.local.md` line | operator |
| 4 | file tickets for FOR-1614 commit 2, the Lean tail timebox, manifest status fields, and the `docs/architecture.md` contradiction | conductor on confirmation |
| 5 | once the plumbing probe is green, hand off to a fresh session for the Lean proofs and the H-series continuation | operator asks for the hand-off prompt |
