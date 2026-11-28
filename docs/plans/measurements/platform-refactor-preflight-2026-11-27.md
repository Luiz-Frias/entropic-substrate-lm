# Platform refactor sprint preflight, 2026-11-27

Produced by `helmsman:sprint-preflight`, run `wf_3a9edf65-e85`, probed at
`a50ea2a852a71d67f71f698b19d14b66ff6a107d`. Sections 1 to 6 below are the assembler's brief as
written; its section 7, the draft manifest, is superseded by the tracked
`.claude/workflows/sprint-manifest.json`. The reviewers' gap list is the sibling file
`platform-refactor-preflight-lens-gaps-2026-11-27.md`. Nothing here is a ruling.

## Conductor notes, 2026-11-27

| brief item | what happened since the brief was written |
| --- | --- |
| placeholder ids `PRE-0`, `M-0` to `M-12`, `MF-0` (Q9) | replaced: Linear project P-FOR-45, issues FOR-1728 (prelaunch) and FOR-1729 to FOR-1742 (M0 to MF), one milestone per horizon |
| Q1 trunk and gt registration | done by the conductor under the operator's standing access grant of 2026-11-27: the trunk is pushed at `5f6b6dd9` and registered as a gt trunk |
| Q2 plan of record reaches the trunk | option (a): the `prelaunch` phase, one branch, two commits, one pull request into the trunk |
| Q3 stack shape | option (a), which is ruling R15 |
| Q10 H1 groundwork manifest | rotated in the prelaunch commit: `archived`; L4a to L4c corrected to `done`; the eight unstarted phases `retired` with a note |
| Q11 rotated launch queue | left local and untracked, matching prior rotations; a fresh local `launch-queue.jsonl` starts from section 6 |
| Q12 durable ref for the L4d head | the conductor pushed `shelf/h1/l4d-evaluation-instruments` |
| Q4 to Q8, Q13, Q14 and Q15 to Q41 | open; each is ruled by the operator at the horizon gate it belongs to. M0 carries Q4, Q8 and Q13 to Q18 |

# Platform refactor sprint preflight: executive brief

## TL;DR

- **Stamp check.** HEAD matches the stamp: `a50ea2a8` on `docs/plan/platform-refactor-rung-seven`. The trunk is a different commit:
  - `refactor/platform-topology` is `5f6b6dd9`, the same commit as `main` and the epoch-tag peel.
  - The trunk exists only locally and is not registered with gt.
  - HEAD is one commit above the trunk: 3 docs files, with zero `src` or `tests` bytes. Every src and tests premise therefore holds at both SHAs.
- **Shape.** 15 phases: one conductor-only `prelaunch`, then M0 to M12 and MF.
  - Every phase is serial, with one stack per horizon.
  - No parallel group is possible: every phase has conflict edges.
  - M9 shares measured files with M7 and M8, so it does not run beside them.
- **HALT.** One consumed-before-owned HALT was found and resolved structurally. Every horizon reads the rung-seven plan, and the plan is not on the trunk, so the new `prelaunch` phase owns it (Q2).
- **Baseline.** `just check` is green with 11126 passed. `just pc-ci` is RED on a missing DVC data directory in this worktree, which is not a code defect. No horizon exit is green-claimable until that gap is closed (Q8).
- **Verification.** The manifest validates against the schema, measured with `jsonschema` 4.26.0:
  - zero rails violations;
  - every door owner runs `arch`;
  - 49 hub files, each pair with a typed edge;
  - the floor regexes match nothing in the three `direct` phases.
- **Open questions.** 41 operator questions follow. None is pre-resolved; `ratified:false`.
- **Advisor.** Consulted once, under the shared lock. Its three blocking corrections are applied: an owner sweep, a hub sweep, and a single json fence.

Evidence classes: **[M]** measured by a command in this run or by a lens that re-measured it. **[R]** relayed from a scout or from the review and not re-measured. **[mem]** from project memory, not re-measured.

## 1. Verified facts

| fact | value | evidence |
| --- | --- | --- |
| HEAD | `a50ea2a8` on `docs/plan/platform-refactor-rung-seven` | `git rev-parse HEAD` [M] |
| trunk | `refactor/platform-topology` = `5f6b6dd9` = `main` tip | `git rev-parse` [M] |
| trunk on origin | absent; `git ls-remote origin refs/heads/refactor/platform-topology` is empty | [M] |
| gt registration | `gt trunk --all` prints only `main` | topology scout [R] |
| HEAD minus trunk | 1 commit: plan, review, shelved-state note; 965 insertions, 8 deletions, zero src or tests | `git diff --stat` [M] |
| epoch tag | annotated `0edde818`, peels to `5f6b6dd9`, on origin. The M0 tag step is DONE | `git ls-remote` [M] |
| L4d shelf `35b0b687` | held by local `refs/heads/shelf/h1/l4d-evaluation-instruments` and `refs/gt-fetch-head/...`; on origin only as `refs/pull/435/head` | `git for-each-ref --contains`, `ls-remote` [M] |
| reader identity | 54 unique entries at `ladder/run/checkpoint.py:78-133`. ladder 26; core 22 (19 `core.sentence`, plus `core.latent.dynamics`, `core.latent.metric`, `core.probe.score_gain`); composition 4; kernel 2. No `__init__` and no bare package | AST literal_eval [M] |
| string imports | only `checkpoint.py:6/:139` and `capture.py:4/:44` | plan:485 [M], freshness lens [M] |
| Protocols under src | 97 | M6 probe [R], freshness lens [M] |
| `just check` | exit 0; 11126 passed; lint-imports 3 kept 0 broken | baseline scout [R] |
| `just pc-ci` | exit 1. The post-commit rerun failed 9 tests, all with `results/h0-final-probe-1k-v4-92c9db is absent`. The record run said 2 failed; the two counts are unreconciled | baseline scout [R] |
| DVC state | two pointer variants are tracked, `results/h0-final-probe-1k-v4-92c9db.dvc` and `...-1k-v4.92c9db.dvc`. Only the HEpP7g data is present. 4 integration files read DVC data | `git ls-files results`, `grep -rln 'dvc pull' tests` [M] |
| freeze predicate today | would be green: `git ls-files 'results/h1-*'` is empty | [M] |
| CI triggers | push on `main`, `refactor/**`, `ci/**`; PR base on `main`, `refactor/**`, `mythwave/**`, `ci/**` | `.github/workflows/ci.yaml:12`, `:20` [M] |
| CodeRabbit | `.coderabbit.yaml` has no `base_branches` key (`auto_review` at :82, `drafts: false` at :92); memory says non-main bases are skipped | [M] [mem] |
| ship scope | `INFRA_PATTERNS` at `scripts/ship_scope.py:30-48`, matched with `fnmatchcase` at :81, so `*` crosses `/` | [M] |
| live gate image | `logger-smoke-image` mounts `./src` and `./scripts` read-only over the digest-pinned image | `justfile:308-309` [M] |
| launch-queue files | `.gitignore:98` ignores only `launch-queue.jsonl`. `launch-queue.h1-groundwork.jsonl` is untracked and not ignored. No queue file is tracked, and the main checkout holds a local `launch-queue.jsonl` | `git check-ignore -v`, `git ls-files` [M] |
| H1 manifest | `lifecycle: active`, `ledger: .claude/workflows/launch-queue.jsonl`; statuses 4 done, 1 holding, 10 queued | [M] |
| open PRs | 9, all based on `main`. Draft #444 (`fix/FOR-1724`) edits `.claude/hooks/worktree-post-create.sh` and `.claude/workflows/mass-patch-fanout.js`; the other 8 are dependabot | `gh pr view 444`, `gh pr list` [M] |
| unmerged remote refs | 11; none classified by patch equivalence | `git branch -r --no-merged main` [M] |
| cascade floor | `/\bports\//`, `/\bdomain\//`, `.claude/workflows/schema/`, `wire[- ]format`, `durable (schema\|surface)` | `td-phase-lifecycle.cascade.js:477-487` [M] |
| schema issue ids | pattern `^[A-Z]+-[0-9]+$`, `minItems: 1`; no tracker ids exist | `sprint-manifest.schema.json:91-99` [M] |
| kernel non-stdlib imports | `kernel/runtime/checkpoint.py:39` torch, `config.py:16` yaml, `logging.py:62` loguru, `paths.py:31` entropic_substrate | freshness lens [M] |
| conftest | imports pytest, torch and loguru at `tests/conftest.py:10-12` | [M] |
| test-edge ratchet | `EDGE_BUDGET = 387` at `tests/architecture/test_test_import_budget.py:31`, asserted equal at :137 | [M] |
| d_ladder worker literal | 8 tracked sites: `scripts/entrypoints.env`, `provider_launch.py`, and 6 test or support files | `grep -rln` [M] |
| L4e port edge | `73dc7968:tests/test_segmentation_golden.py:23` imports `tests.test_sentence_sampling`; `tests/latent_usage_support.py` is absent on main; the FOR-1713 relocated siblings exist under `tests/unit/ladder/run/` | `git show`, `ls`, `git ls-files` [M] |

## 2. Scope table

Numbering: the plan governs. Review section 4.2 "M10 Series members" is plan M11, and review "M11 Composition root" is plan M12. Plan M10 has no review section; it comes from R19 to R22.

| phase | premise | disposition | correction carried into the manifest |
| --- | --- | --- | --- |
| prelaunch | new | stack, conductor-only | resolves the HALT: the plan and the review exist only at `a50ea2a8` |
| M0 | PARTIAL | narrowed | tag ALREADY DONE. Still to do: freeze test, charter text (`CLAUDE.md:5`, `:38`), debt ledger, ref classification, replay recipe and replay corpus, all new. H1 ticket is a Linear write. Fix `CLAUDE.md:224` |
| M1 | PARTIAL | narrowed | `logger_smoke.py` and `dead_code_report.py` exist, so extend them. Export-origin probe is absent, so design it. D13 is a delta merge into relocated files, not a cherry-pick |
| M2 | HOLDS | stack | door per R18 (-v1), not review:350. The "22 non-member imports" count is unmeasured |
| M3 | PARTIAL | stack | no workspace table today; `platform.toml` and the strict loader are new; D26 measurement not run |
| M4 | PARTIAL | narrowed | kernel is not stdlib-only. `Sha256` and `GitCommit` do not exist. Clone set: 24 `_canonical` + 10 `_digest` definitions in 31 files, plus 147 `hashlib.sha256` call sites [R] |
| M5 | PARTIAL | narrowed | of the 5 ports only `MetricSink` exists (`metric_sink.py:40`). The "nine D1-named modules" are not reproduced; the gate enumerates them |
| M6 | PARTIAL | stack, needs-recon on the registry | 97 Protocols confirmed. Registry typestate not probed. D24 open |
| M7 | PARTIAL | stack | "five loops" not reproduced: 7 `optimizer.step` files, 4 loop headers. Review's `-v3` and read-old adapters struck by R18 |
| M8 | PARTIAL | narrowed | `ExhaustedPending` is new. One guard exists, at `h_probe_watchdog.py:128`. No fake Provider in src; private doubles in 5 files. Launch V5 today (`h_probe_launch.py:230`) |
| M9 | HOLDS | stack, serial after M8 | "thirteen D1 roles" UNVERIFIED (`D1ArtifactRole` has 5 members). Register->run is only `h1_pilot.py:43`; the probe's `:19` is a docstring |
| M10 | PARTIAL | stack | cell exists only after M3; `transformers`, `peft`, `safetensors` are new; X-LoRA result is unverifiable here |
| M11 | PARTIAL | stack | **19 `core.sentence` members, not 21.** Lanes are edge-disjoint only if M4 keeps `metric`, `dynamics` and `blocks` in core |
| M12 | PARTIAL | narrowed, needs-recon for `composition/` | the plan's `historical/` is `src/entropic_substrate/historical/` (34 files). No probe covered the move of `composition/` into the cli member |
| MF | HOLDS | stack, appended commit | root `__init__` is not a reader member at this SHA, so no re-pin. Re-check after M3 |

No phase is verify-and-close. Rows that say "does not exist yet" are marked not-started, not HOLDS (freshness lens).

## 3. Conflict matrix, stack proposal, merge order

**Corrections applied from the lenses.**

- **Stamp.** The trunk is carried as `5f6b6dd9`; `a50ea2a8` is the unpushed docs tip.
- **Stacks.**
  - One stack per horizon, replacing the assembly's seven multi-horizon stacks. Source: R15 at plan:513 and the operator's one-sub-session-per-horizon request.
  - Hub files are conductor commits, and the hub set now includes grouping defect 4's seven consumer hubs.
  - Lane counts follow the plan, three per fanout horizon (plan:590).
- **Placement.**
  - M9 runs serial after M8.
  - M10 runs serial after M9 and merges before M11 opens.
- **Counts and registers.**
  - `core.sentence` holds 19 reader members, not 21.
  - The shelf branch is kept, not retired. Source: D12 at plan:675.

**Measured edges that decide the order.** The manifest carries every edge with its evidence.

| pair | shared surface | class | resolution |
| --- | --- | --- | --- |
| prelaunch -> every horizon | rung-seven plan and review exist only at `a50ea2a8` | [M] | rails-before; HALT resolved |
| M2 -> M7 | `ladder/run/checkpoint.py`: M2 at `:6`, `:78-133`, `:138-139`; M7 at save `:287` and loader | [M] | M2 owns |
| M7 -> M9 | 12 of 18 `ladder/run/trainer` files import `ladder.register` | [M] | M9 serial-after |
| M8 -> M9 | `campaign/d1_operator_controller.py` (3 register imports), `d1_operator_values.py` (2), `worker_execution.py`, `worker_request.py` | [M] | M9 serial-after |
| M5 -> M9 | `d1_predecessor_loader.py`, `ladder_predecessor.py`, 5 transport files | [M] | M9 serial-after |
| M4 -> M11 | 10 H->D import lines into `core/latent/metric.py` or `dynamics.py`; 1 D->E at `core/latent/state.py:19` | [M] | M4 rules residency (Q23) |
| M1 -> M6 | `EDGE_BUDGET = 387` equality; the L4e port adds an edge | [M] | M1 conductor-edits the line; M6 owns the ban |
| M3, M10, M11, M12, MF | `pyproject.toml`, `uv.lock` | [M] | M3 owns; regenerate the lock, never hand-merge |
| M5, M8, M9, M11, M12 | `scripts/entrypoints.env:1-6` | [M] | M12 owns; earlier edits are conductor zip lines |
| M1, M3, M4, M5, M8, M11 | `scripts/ship_scope.py:30-48`, `scripts/logger_smoke.py:84-94` | [M] | M1 owns, adding self-tests before any move |
| M4, M5, M8, M9, M11 | sentinel map at `tests/architecture/test_module_identity.py:132-154` | [M] for `kernel.lifecycle`, others [R] | M9 owns |
| M3, M4, M5, M9, M11, M12 | `.importlinter:42-50` landed layers name `entropic_substrate.kernel` and others | [M] | M3 owns |
| M6 -> M9 | `scripts/assemble_review_packet.py:388-430` run_start parity across arms | [R compat] | M9 owns; scrub ruling (Q28) |

**Hub owners.**

| owner | hubs |
| --- | --- |
| M0 | `CLAUDE.md` |
| M1 | `tests/conftest.py`, `scripts/suite_stages.py`, `ship_scope.py`, `logger_smoke.py`, `tests/support/**`, `.pre-commit-config.yaml` (asks only) |
| M2 | `ladder/run/checkpoint.py`, `ladder/capture/capture.py`, `tools/regime_rows.py`, `scripts/h_probe_bootstrap.py` |
| M3 | `pyproject.toml`, `uv.lock`, `.importlinter`, `Containerfile`, `justfile`, `src/entropic_substrate/__init__.py`, `tests/support/repo_paths.py` |
| M4 | `kernel/**`, `tests/support/helpers.py` |
| M5 | `ladder/capture/{ports,reader,adapters,custody}/**`, `ladder/run/transport/**` |
| M6 | `composition/wiring/**`, `test_test_import_budget.py`, `tests/typing/**` |
| M7 | `ladder/run/trainer/**`, `historical/e_series/**`, `core/probe/predictive_probe.py`, `docs/deploy-h100.md` |
| M8 | `ladder/run/provider/**`, `campaign/d1_operator_*`, `worker_execution.py`, `worker_request.py` |
| M9 | `ladder/{register,analyze,review,file,gate}/**`, `test_module_identity.py`, the review-engine scripts |
| M11 | `core/{sentence,latent,substrate,geometry}/**`, the four DVC-backed integration files |
| M12 | `scripts/entrypoints.env`, `scripts/run_ladder.sh`, `tools/mapping.json`, `docs/architecture.md`, `historical/**` |
| MF | the release-please files |

**Merge order.** It is fully serial. Each horizon PR merges into the trunk, and the trunk merges into `main` at each horizon exit:

> prelaunch -> M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 -> M8 -> M9 -> M10 -> M11 -> M12 -> MF

The rails are ordered by descending fan-out. M9 precedes M10 because M9 owns a door consumed by M11 and M12, while M10 owns none. M12 is the disruptive branch and lands last; MF is one appended commit after the closing pin.

## 4. Compat audit verdicts

`web/` is absent. Fixtures carry only `h-captured-solve-fixture-v1`, which no change touches. Nothing in `src` reads `ENTRYPOINT_*`. [R compat]

| horizon | verdict | consumers that break outside co-located tests | material |
| --- | --- | --- | --- |
| M0 | COMPATIBLE | none; the freeze predicate must not key on label text in `preregistration-skeleton.md:316-317` | charter text (D16) |
| M1 | BREAKS | `scripts/suite_stages.py:75-82` silently drops a moved folder until the floor lands | `.pre-commit-config.yaml` ask |
| M2 | BREAKS | `tools/regime_rows.py:63-75` labels, updated in the door commit; docs `read-old-table.md` and the skeleton | yes, one-way door |
| M3 | BREAKS | `h_probe_bootstrap.py:21-32` with its parity test `:283-316`; `Containerfile:63-65`; `justfile:240/:256/:309`; `.pre-commit-config.yaml:148`; `ship_scope.py:36-48`; `test_release_please_config.py:89-101` | packaging, namespace, hooks |
| M4 | BREAKS | `assemble_review_packet.py:330`, `logger_smoke.py:133/:166`, `h_solver_preflight.py:63`, `prepare_enwik8_source.py:41`, `regime_rows.py:42-45` | public `ModelConfig` and `ContentHash` paths |
| M5 | BREAKS | `ship_scope.py:41` goes inert on `sinks.py`; `ship_scope.py:46-48`; `logger_smoke.py:84-94` | typed `MetricSink` envelope |
| M6 | BREAKS, latent | `assemble_review_packet.py:388-430` whole-record run_start parity | run_start shape |
| M7 | BREAKS | the `train` CLI at `justfile:250-251`, `run_ladder.sh:65`, `Containerfile:90`, `docs/deploy-h100.md:318-329` | family, cadence, RNG custody |
| M8 | BREAKS | `justfile:76`, `logger_smoke.py:84-86`, `provider_launch.py:36-37`; `docs/deploy-h100.md:185` is already stale at v4 | launch family, stall field, new terminal state |
| M9 | BREAKS | `assemble_review_packet.py:222-223` prefix selection goes empty; `entrypoints.env:5`; review-engine paths | filing mint, D2 to D5 retirement |
| M10 | BREAKS, not line-measured | dependency and lock tests; the image would change, but no re-pin is allowed under the freeze | new dependencies |
| M11 | BREAKS | `h0_final_probe_checks.py:40-56`, `prepare_enwik8_source.py:29-50`, `e0_bytes_seen_plot.py:29`, `e1_gap_pareto_plot.py:24`, `h_solver_preflight.py:56-64`, `h_solver_trace.py:46-48`, `regime_rows.py:42-45` | public import paths |
| M12 | BREAKS | deleting `entrypoints.env` breaks `justfile:15/:17-19`, `run_ladder.sh:33`, `test_cli_surface.py:102`, `test_no_old_prefix.py:71/:398-402`, `test_h_probe_bootstrap.py:283-316`; deleting `historical/` breaks `tools/mapping.json` (41 references) and `.importlinter:47/:85` | R25 spend, entrypoint-name mapping |
| MF | BREAKS by design | `test_release_please_config.py:53/:64/:78-86/:89-115/:118-131`; replace the teeth, never delete them | public tag format |

## 5. Decision surface

Every item is the operator's to rule. The default is a recommendation, never applied.

**HALT report.** R-rails consumed-before-owned:

- **What was found.** Every horizon consumes `plan-of-record-rung-seven`, which is absent from the trunk.
- **How it is resolved.** Structurally, by the `prelaunch` phase, which owns the plan of record.
- **Check result.** After the fix, `check_manifest.py` finds no other consumed-before-owned edge.

### Before prelaunch

**Q1. How do the trunk and gt registration come into being?**

- (a) The operator pushes `refactor/platform-topology` at `5f6b6dd9` and runs `gt trunk --add`, before publishing prelaunch.
- (b) Base the horizon stacks on `main`.
- (c) Push the trunk with prelaunch already merged in.

Default: (a).

Evidence: `ls-remote` is empty; `gt trunk --all` prints only `main`; R15 sets one long-lived trunk.

**Q2. How does the plan of record reach the trunk (the HALT)?**

- (a) Keep the `prelaunch` phase: the existing docs commit plus one rotation commit, carrying this manifest. Publish it as one PR into the trunk and merge before M0 opens.
- (b) Stack M0 on the docs branch.
- (c) The conductor fast-forwards the trunk without a PR.

Default: (a).

Evidence: the HEAD-to-trunk diff is the plan, the review and the shelved-state note only.

**Q3. What is the stack shape?**

- (a) One stack per horizon, each based on the trunk tip after the previous horizon merged.
- (b) The assembly's seven multi-horizon stacks.

Default: (a).

Evidence: R15 at plan:513; under (b), one fix to M4 restacks M5's zipped lanes and voids their re-pins.

**Q4. Which merge method do horizon PRs into the trunk, and the trunk into `main`, use?**

- (a) `--no-ff` throughout.
- (b) Squash merges for horizon PRs and `--no-ff` for the trunk.

Default: (a).

Evidence: the operator's merge default is `--no-ff`, never squash; memory records squash-restack duplication [mem].

**Q5. Which horizon branch prefix?**

- (a) `refactor/<scope>/<slug>`, as drafted. Push events and upstack PR bases then match CI. Cost: commit subjects must be typed `refactor`.
- (b) Type-matched prefixes such as `docs/` or `test/`, plus a `ci.yaml` trigger edit, which is an operator ask.
- (c) `mythwave/`, which matches PR base only, not push.

Default: (a).

Evidence: `ci.yaml:12` and `:20`; the branch-naming ruling of 2026-11-21 [mem].

**Q6. What is the remote review channel for trunk-based PRs?**

- (a) `@codex review` on every horizon PR, the ruled fallback, with the local CodeRabbit CLI as pre-flight.
- (b) Add `base_branches` for `refactor/.*` to `.coderabbit.yaml`, which is a config ask.
- (c) Review only the trunk-to-main PRs.

Default: (a).

Evidence: no `base_branches` key in `.coderabbit.yaml` (`auto_review` at :82, `drafts: false` at :92); non-main bases are skipped [mem].

**Q7. How does the live gate run under the image freeze?**

`logger-smoke-image` mounts the new tree over a pinned image (`justfile:308-309`), and R12/R25 forbid a re-pin. After M3 the image may import its own installed packages instead of the mounted tree. That would be false-green, but it is [inferred], not measured.

- (a) Measure `logger-smoke-image` at the M3 tip. If it cannot import the tree, rule `liveDeferred` for that recipe on M3 onward, with M12's fresh image as the re-certification owner, keep `container-test`, and run the host logger smoke as the named substitute.
- (b) Keep the full gate and accept a stop at each horizon.
- (c) Defer the whole live gate from M3 onward.

Default: (a).

Evidence: `ship_scope.py:30-48` forces `ship-it-full` on M1, M3 to M5, M8 and M10 to MF.

**Q8. How is the baseline DVC gap closed, and which pointer is authoritative?**

- (a) Before M0, the conductor runs `dvc pull results/h0-final-probe-1k-v4-92c9db.dvc` in each sprint worktree and captures one run that names the failing set, reconciling 2 against 9. The worktree-hook extension becomes an M0 ask instead of an M3 ask.
- (b) Symlink `results/` from the main checkout, which has the data.
- (c) Deselect the failing tests. Rejected: a deselected test reads as green.

Default: (a), plus a ruling on whether `-92c9db.dvc` or `.92c9db.dvc` is authoritative.

Evidence: 4 DVC-reading integration files; both pointer variants are tracked.

**Q9. How are tracker ids supplied?**

- (a) The conductor files one Linear issue per horizon, plus prelaunch, and replaces the placeholders before ratification.
- (b) Ratify with the placeholders.
- (c) Amend the schema pattern.

Default: (a).

Evidence: `schema:91-99`. The placeholders `PRE-0`, `M-0` to `M-12` and `MF-0` are not tracker ids.

**Q10. What happens to the H1 groundwork manifest?**

- (a) Prelaunch flips `lifecycle` to `archived` and sets the 10 queued and 1 holding phases to `retired`, with the note "shelved under R12, re-implemented on the new platform". It repoints `ledger` to `launch-queue.h1-groundwork.jsonl`. This satisfies M0's "corrected H1 statuses" early.
- (b) Leave `active` and correct the statuses in M0.
- (c) Use `holding` instead of `retired`.

Default: (a).

Evidence: the measured H1 manifest fields; the rotation protocol at `schema:293`.

**Q11. What happens to the rotated launch-queue files?**

- (a) Leave `launch-queue.h1-groundwork.jsonl` untracked and local, matching prior rotations, none of which is tracked. Copy the main checkout's `launch-queue.jsonl` beside it, then start a fresh `launch-queue.jsonl` from section 6.
- (b) Commit the rotated file.
- (c) Widen `.gitignore:98` to `launch-queue*.jsonl`, which is a config edit.

Default: (a).

Evidence: `.gitignore:98`; `git ls-files` shows no queue file.

**Q12. Which refs need durability before any prune (risk 11)?**

- (a) The operator pushes `shelf/h1/l4d-evaluation-instruments`, and deletes nothing until M0's `git cherry` table is filed. The FOR-1713 ref (`9c3729e1`) and the L4e ref (`73dc7968`) are already on origin.
- (b) Rely on `refs/pull/435/head`.

Default: (a).

Evidence: `for-each-ref --contains 35b0b687` shows no `refs/remotes` holder; `ls-remote` confirms the other two refs. D12 says "kept shelved", so the topology scout's "retire" is not adopted.

**Q13. When does PR #444 land?**

#444 edits the M3 hook file and the fanout engine that M1, M5 and M11 run on.

- (a) Land it on `main` before M1, then merge `main` into the trunk `--no-ff` at the M0 exit, so the epoch tag stays the branch point.
- (b) Defer it past the sprint and freeze both files.
- (c) Close it and fold it into the M3 hook ask.

Default: (a).

Evidence: `gh pr view 444`.

**Q14. What happens to the 8 dependabot PRs?**

They edit `pyproject.toml` and `uv.lock`, the M3, M10, M12 and MF hub.

- (a) Hold all of them until MF, then rebase.
- (b) Land the safe ones on `main` before pushing the trunk.
- (c) Merge `main` into the trunk at each exit.

Default: (a).

Evidence: anchor at `pyproject.toml:108`, which only moves forward; lock regeneration is never hand-merged.

### M0 to M2 content

**Q15. Where is the debt plan source that M0 commits as a ledger?**

D25 rules that it is committed but does not locate it.

- (a) The operator names its path or pastes it.
- (b) The conductor reconstructs it from plan:86, :117, :398 and review:76, :535, as a draft for approval.

Default: (a).

Evidence: no tracked file carries it.

**Q16. Who files the H1 detail-preservation ticket that D25 rules is filed in M0?**

- (a) The conductor, as a Linear write the operator approves.
- (b) The operator.

Default: (a).

Evidence: plan:688.

**Q17. Who owns the replay recipe and the replay corpus enumeration?**

The corpus is one H0 capture plus every retained D1 fixture. M4 golden vectors, M9 replay equality and M12 fresh-clone replay all read the corpus.

- (a) M0, as docs only, before M2 renames the first family.
- (b) M2, inside the door commit.

Default: (a).

Evidence: R18 at plan:516; Risks amended row 2 at plan:654. No file carries a recipe today.

**Q18. Where does the platform member's charter go while the member does not exist?**

D7 and D16 say the conductor drafts it in M0.

- (a) M0 drafts `docs/platform/charter-draft.md`, and M3 moves it into the member `CLAUDE.md` after approval.
- (b) Draft it at M3.

Default: (a).

Evidence: D16 at plan:679; member roots exist only after M3.

**Q19. Who owns the R18 generated catalog and define-once lint?**

- (a) M2 introduces them, ratcheted to the families renamed so far, because a full lint is red on day one: 348 literals with duplicates [R].
- (b) M1, as tooling.

Default: (a).

Evidence: R18 at plan:516.

**Q20. Do every door name and every sub-session brief follow R18 rather than the review's door text?**

- (a) Yes. Each door is a new family at `-v1`, and the names are chosen at each design gate.
- (b) Use the review's labels: `h-checkpoint-reader-v3`, `platform-training-checkpoint-v3`, launch v6.

Default: (a).

Evidence: R18 governs. review:350, :407 and :418 conflict with it.

**Q21. Which `engine_mode` do door owners whose plan vehicle is a fanout or one commit get, M11 and MF?**

- (a) Keep the door: `arch` is schema-forced (if/then). The horizon gets one design gate, and its lanes run under it.
- (b) Reclassify M11's keys as values under M2's frozen identity door and MF's release lanes as a contract, so both become `direct`.

Default: (a), fail-closed.

Evidence: `schema:254-284`; the fanout vehicle at plan:620.

### M3 to M6 content

**Q22. How does the pytest plugin fit a stdlib-only kernel?**

- (a) `entropic_kernel` gets a `testing` extra whose dependency, pytest, is declared only on the extra. Torch determinism and loguru registration live in members that declare them, and the kernel gate runs without the extra.
- (b) A separate `entropic-kernel-testing` distribution.
- (c) A reasoned D21 deviation.

Default: (a).

Evidence: `conftest.py:10-12`; plan:534 says "a testing extra"; plan:576 and D21 at plan:684 require stdlib-only.

**Q23. Do `core/latent/metric.py`, `core/latent/dynamics.py` and `core/substrate/blocks.py` stay in core?**

- (a) Yes, ruled at M4. M11's lanes are then edge-disjoint, and the plan's "H lane only" re-pin holds.
- (b) Send them to series-d. Then H imports D, which is a stop.

Default: (a).

Evidence: 10 measured H->D lines and 1 D->E line; review:255 and :259.

**Q24. Who owns `kernel/verdict/**`, and what is the "nine D1-named shared modules" set?**

- (a) M11's D lane moves `kernel/verdict/**`. M5's gate enumerates the nine by file.
- (b) M4 moves `kernel/verdict/**`.

Default: (a).

Evidence: review:268 sends it to series-d; only 6 `d1_` files were found under `kernel/`.

**Q25. Who owns the `ModelFactory` port shape?**

- (a) M6, as drafted. M6 also shapes `Spine` and `AdapterMethod`; M7 builds the engine over them; M10 implements the adapters.
- (b) M7, as the grouping lens proposes, because the topology table puts `ModelFactory` in the engine cell at plan:543.

Default: (a). Why not (b): M6's oracle needs "a toy series runs through every port" (plan:578), and R19 composes `Spine` and `AdapterMethod` into `ModelFactory`. Shaping the parts without the composite leaves M6's toy series unable to build a model.

Also rule the `Spine` port name before M6. It collides with `core/sentence/spine.py:19` `SpineSpec` and `:109` `SentenceSpine`.

**Q26. Is `Tracker` a separate mirror interface, or does Trackio sit behind `MetricSink`?**

- (a) M5 shapes `Tracker` as an adapter-side mirror beside `MetricSink`, and M12 adds Trackio behind `Tracker`.
- (b) Trackio goes behind `MetricSink` directly.

Default: (a).

Evidence: plan:551 and :584. The tree has only `MetricSink` at `metric_sink.py:40`.

**Q27. Does D24 allow series-internal strategy Protocols?**

It is open and gates M6.

- (a) Yes, as plugin-internal. Tooth 4 refuses `ports` modules and cross-member Protocol imports.
- (b) No.

Default: (a).

Evidence: plan:687; review:534.

**Q28. Does `run_start` get an arm-blind scrub before M6 adds digest fields?**

- (a) Add an explicit exemption list to `assemble_review_packet.py` in M9, before any blinded packet is built from post-M6 captures. This is latent under the freeze.
- (b) Have M6 add the scrub itself.

Default: (a).

Evidence: `assemble_review_packet.py:388-430` [R compat]; `telemetry.py:166-179` tolerates extra keys.

### M7 to MF content

**Q29. Who owns the E field pre-training adapter and its parity check (D25)?**

- (a) M7, which deletes the loops, including `field_pretrain.py:356`.
- (b) M11's E lane, with M12's deletion gated on it.

Default: (a).

Evidence: plan:688; grouping defect 9b.

**Q30. Where does the D19 courier fix land?**

- (a) M8, since D19 is confirmed at the M5 and M8 gates and the M8 oracle needs it.
- (b) M5.

Default: (a).

Evidence: plan:580, :682.

**Q31. Where do M9 and M10 sit?**

- (a) M9 serial after M8, then M10 serial after M9, with M10 merged before M11 opens.
- (b) The plan's "beside" option, with a host-wide `just stack` lock and a re-measured touch set.

Default: (a).

Evidence: the M9 overlap was measured. `just stack` shares the clipboard [mem]. M11 regenerates `uv.lock`.

**Q32. Are the `engine_mode` verdicts accepted?**

| phase | mode | reason |
| --- | --- | --- |
| prelaunch | `direct` | conductor-only |
| M0 | `direct` | no door; floor matches nothing |
| M1 | `direct` | no door; floor matches nothing. Re-run it when the D22 list exists: a deletion under `ladder/capture/ports/` would match |
| M4 | `arch`, conductor-forced | D21 is confirmed at the M4 gate (plan:684), and the digest seam is unnamed |
| M5 | `arch`, floor-forced | matches `/\bports\//`; D19 is confirmed at its gate |
| M10 | `arch`, conductor-forced | R21's pre-registered margin, the adapter-delta slot, new dependencies, the X-LoRA ruling |
| M2, M3, M6, M7, M8, M9, M11, M12, MF | `arch`, schema-forced | each owns a door |

- (a) Accept.
- (b) Name changes.

Default: (a).

**Q33. How are the reader members that M11 does not carry assigned?**

- (a) `kernel.artifacts`, `kernel.lifecycle` and `core.probe.score_gain` re-pin at M4 (review:374). The 25 `ladder.run` members and `ladder.capture.capture` re-pin with the M5, M7 and M8 touches. The 4 `composition` members re-pin at M12. M11 carries 19 `core.sentence` members plus the H run members.
- (b) Defer every assignment to the zip.

Default: (a).

Evidence: AST counts.

**Q34. What are M1's lane shape and D13 port mechanics?**

- (a) Lane 1 runs alone first: the floor, the stage table and the scope self-tests. Lanes 2 and 3 then fan out, K=2: plugin plus rewriter, then ports plus the D22 list. Port the deltas into the existing relocated files, with one authorized move of shared fixtures into `tests/support`, and the conductor adjusts `EDGE_BUDGET` at the zip. M5 keeps three lanes, with GitFacts and ProcessRunner inside infra.
- (b) The assembly's 7 and 4 lane seeds.

Default: (a).

Evidence: plan:590; the L4e import at `:23`; `tests/latent_usage_support.py` is absent.

**Q35. Who retires the `d_ladder_worker_command` literal?**

- (a) M9, with all eight sites in its allowed delta. M1's rewriter dry run must prove it finds all eight.
- (b) M12.

Default: (a).

Evidence: 8 sites measured; R3 retirement via D25.

**Q36. Who moves `composition/` into the cli member?**

This is about 28 files and 6.5k lines, with 4 reader members [R review:249-251].

- (a) M12, the composition-root horizon, after a touch-set probe at its gate. The `cli.py` CLI surface keeps an old-to-new mapping.
- (b) Split it: registry and command service at M6, process mains at M12.

Default: (a).

Evidence: no probe covered it; `justfile:250-251`, `run_ladder.sh:65`.

**Q37. Are the `liveRequired` values accepted?**

They are computed from today's `INFRA_PATTERNS`, which M1, M3 and M5 rewrite, so every value after M5 is provisional. M6 and M7 are elected `true` beyond the forced set, because `run_start` and the engine become every series' emission path (house rule 1).

- (a) Accept, and recompute at each gate.
- (b) Use only the forced set.

Default: (a).

**Q38. Can two horizon sub-sessions be active at once on this host?**

- (a) No. Only one horizon sub-session commits at a time; the conductor commits only between its windows; gates run under `/tmp/esl-suite.lock`.
- (b) Yes, with a `just stack` host lock first.

Default: (a).

Evidence: the `just stack` clipboard race [mem]; the 4 vCPU podman VM.

**Q39. Where does the R23 second-pass bucket live?**

- (a) One Linear label, plus one line per finding in the M0 debt ledger.
- (b) A tracked file per horizon.

Default: (a).

Evidence: R23 at plan:521.

**Q40. Is the stale `.importlinter` header fixed at M3?**

Its DN-P1 header allows edits only at p0, p3b and p13.

- (a) Yes, as part of the M3 rewrite, naming the conflict.
- (b) Leave it.

Default: (a).

Evidence: `.importlinter:1-35`.

**Q41. When does M12's R25 comparison spend money?**

- (a) Only under a fresh GO, with the balance stated aloud. The pod is torn down when its own run ends.
- (b) Defer it past the sprint.

Default: (a).

Evidence: R25 at plan:523. No other horizon may need a pod under R12 and R25.

## 6. Draft launch queue

These are ready to append to `launch-queue.jsonl` after the interview. `prelaunch` is omitted because it is conductor-only and never launched. `verifiedAt` is the probed HEAD.

```text
{"phaseId":"M0","branch":"refactor/charter/m0-flush-freeze-charter","parent":"refactor/platform-topology","issues":["M-0"],"gates":{"liveRequired":false},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M1","branch":"refactor/tests/m1-rails-before-motion","parent":"refactor/platform-topology","issues":["M-1"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M2","branch":"refactor/identity/m2-identity-door","parent":"refactor/platform-topology","issues":["M-2"],"gates":{"liveRequired":false},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M3","branch":"refactor/workspace/m3-workspace-spec-door","parent":"refactor/platform-topology","issues":["M-3"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M4","branch":"refactor/kernel/m4-kernel-core-members","parent":"refactor/platform-topology","issues":["M-4"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M5","branch":"refactor/platform/m5-planes-pod-kernel","parent":"refactor/platform-topology","issues":["M-5"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M6","branch":"refactor/platform/m6-ports-registry-kit","parent":"refactor/platform-topology","issues":["M-6"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M7","branch":"refactor/engine/m7-custody-engine","parent":"refactor/platform-topology","issues":["M-7"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M8","branch":"refactor/pod/m8-pod-machinery","parent":"refactor/platform-topology","issues":["M-8"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M9","branch":"refactor/evidence/m9-protocol-split","parent":"refactor/platform-topology","issues":["M-9"],"gates":{"liveRequired":false},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M10","branch":"refactor/tuning/m10-finetune-cell","parent":"refactor/platform-topology","issues":["M-10"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M11","branch":"refactor/series/m11-series-members","parent":"refactor/platform-topology","issues":["M-11"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"M12","branch":"refactor/close/m12-root-teeth-lift","parent":"refactor/platform-topology","issues":["M-12"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
{"phaseId":"MF","branch":"refactor/release/mf-per-member-versions","parent":"refactor/platform-topology","issues":["MF-0"],"gates":{"liveRequired":true},"state":"queued","verifiedAt":"a50ea2a852a71d67f71f698b19d14b66ff6a107d"}
```
