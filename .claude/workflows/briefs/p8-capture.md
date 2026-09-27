# p8-capture — the DVC-captured evidence stage lands in `ladder/capture`

Authoritative brief for one `td-phase-lifecycle.cascade` LANE of wave `w2`. v2, written 2026-09-23,
superseding the 2026-09-17 v1 under the conductor's ruling D-WAVES r3 (2026-09-23), and amended by
fix rounds 1-3 the same day (round 3 applies rulings **W2-SCC** and **BUDGET-PER-COMMIT** after the
w2 design gate HALTED on a measured defect), and by round 4 the same day (the SECOND w2 design
gate halted p8 on three measured contradictions, ruled **P8-PACKET-SCRIPT**, **P8-SHIM-LAYER-RETIRE**
and **P8-OBSERVERS-RENAME-ONLY**; the p9 gate's **DATA-BUILD-RUNG** landed on the parent). Plan:
`docs/plans/src-hexagonal-refactor.plan.md` (§mechanism, §P-1, R-PIN row line 839, D-WAVES r2 line
841, D-WAVES r3 line 843, the gate-1 and gate-2 ruling rows that follow D-WAVES r3). Every count below is MEASURED
at `79d8228` (the p7 branch at measurement time, an ancestor of the parent tip `c2f64a0`) with the
command recorded beside it, or labelled `[predicted]` with its arithmetic. **Epoch moved 2026-09-23
(first move):** the parent became `mythwave/p7b-freeze-and-verify` @ `b7af1aa`, and the one count
that could move was RE-MEASURED there — landed→`__legacy` edges are **45**, unchanged, equal to the
highest-ordered fragment's committed budget (`p7-register-gate.json`, no p7b fragment exists), so the
lens check held at that epoch. **Epoch moved again 2026-09-23 (second move):** the parent tip is now
`c2f64a0` = `e01867a` (docs: the first w2 design-gate rulings) + the DATA-BUILD-RUNG commit. MEASURED
`git diff --stat 79d8228 c2f64a0 -- src/entropic_substrate/__legacy`: ONE file,
`__legacy/data/build.py` (its five first-party import lines); `git diff --stat b7af1aa c2f64a0` adds
only one new test (`tests/test_legacy_data_build_cold_import.py`), three briefs, the manifest and the
plan. So every `79d8228` count stays valid EXCEPT build.py's own-import rows, RE-MEASURED at
`c2f64a0` under *Rewrite-on-arrival* (RETRACTION 8), and the 45 holds: `tests/test_legacy_coupling.py`
11 passed at `c2f64a0` (equality arm against the committed 45; the checkout carried only uncommitted
manifest and plan edits). The second design gate measured at `e01867a`; `c2f64a0` changes none of its
subjects except build.py. Every CONTROL COMMAND names `mythwave/p7b-freeze-and-verify` (tip
`c2f64a0`), never p7 (RETRACTION 7). The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is
where that section lives.

## Branch

| field                                        | value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p8-capture`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| parent                                       | `mythwave/p7b-freeze-and-verify` @ `c2f64a0` (MEASURED: p7 tip `9b0cb59` + the FOR-1550/FOR-1548 hardening commits `40e72ac`, `d7c5807`, `b7af1aa`, then `e01867a` (docs: first w2 design-gate rulings) and `c2f64a0` (ruling DATA-BUILD-RUNG: `__legacy/data/build.py`'s five import lines + `tests/test_legacy_data_build_cold_import.py`); the counts below were taken at `79d8228`, an ancestor of this tip, and stay valid except build.py's own-import rows, RE-MEASURED at `c2f64a0` (RETRACTION 8); the 45 landed→`__legacy` edges were RE-MEASURED at `b7af1aa` and re-checked green at `c2f64a0` by `tests/test_legacy_coupling.py`). Was `b7af1aa` until the second epoch move of 2026-09-23. Was `mythwave/p7-register-gate` @ `79d8228` until 2026-09-23 (operator asked for the hardening PR between p7 and w2). CORRECTED 2026-09-23 from v1's `mythwave/p6-core`: w1 ran SERIAL (p6, then p7), so the integration tip w2 forks off is p7's. v1's parenthetical about "the w1 zip order is p7 (20) then p6 (33)" is struck — w1 had no zip. STARTTIP: the lane's `startTip` (40-hex in your args, read by the planner at provision) is this branch's tip at launch, one docs commit AFTER `c2f64a0` (the commit that lands these amendments); every control and footprint diff compares against `<startTip>`, never a typed SHA. |
| mode (sprint manifest, parallel eligibility) | `ratchet` (the schema enum reads "ratchet = known set (parallel-eligible)"; p8 is a wave member, unlike p6/p7 which are `tracer`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| engine_mode                                  | `arch` (manifest field, machine-read by the launch composer; ruled 2026-09-18)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| wave                                         | `w2`, PARALLEL: p8 ∥ p9 ∥ p11 (D-WAVES r3)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `gates.liveRequired`                         | `false`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| live gate                                    | DEFERRED to the zip. `gates.liveDeferred` names the wave zip as re-certification owner: ONE `just ship-it` on the integrated tip, never K concurrent podman stacks (house rule 2, 4 vCPU / 4 GB VM).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| gates                                        | lifecycle arg `{"offline":"just check","exit":"just pc-ci"}`; the lane adds no gate vocabulary. The sprint-manifest `gates` object keeps only `{"liveRequired": false, "liveDeferred": …}` — the schema sets `additionalProperties:false` and admits nothing else.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `policy.publishMode`                         | `'defer'` (ruling O-1). The lane ends CERTIFIED and publishes NOTHING.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| launch                                       | WAVE MEMBER: one ticket `{id,title,scope,acceptance}` inside the single `mass-patch-fanout` invocation for `w2`. `cascadeScriptPath` is the repo's ported `.claude/workflows/td-phase-lifecycle.cascade.js`. No per-lane session, no hand-written wave script.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| lane worktree                                | NESTED at `<checkout>/.claude/worktrees/wave-p8-capture`, never a sibling of the conductor's worktree (measured law, 2026-08-27 pilot: a sibling lane is unreachable to workers; a nested path is reachable by plain `cd`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| lane isolation                               | `cd` to the lane worktree FIRST in every stage, never `EnterWorktree`. Verify `pwd` and branch before acting and FREEZE on mismatch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| zip order                                    | `p8 → p9 → p11` (D-WAVES r3, by inbound dependency mass: p8 receives **41** from p9 + **50** from p11 = 91 of the wave's **96** cross-lane edges (50 + 41 + 4 + 1); MEASURED, pack `depends_in`). The planner's `zipOrder` is validated as an exact permutation before any lane launches.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| trunk                                        | `refactor/hexagonal-redesign`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

## Wave w2 mechanics

The lane is not a phase running alone. Six mechanics bind it, all ruled (D-WAVES r3, FOR-1547,
P8-FRAG):

1. **Worktree and parent.** Nested worktree off `mythwave/p7b-freeze-and-verify` (see the parent row), branch
   `mythwave/p8-capture`. The lane never rebases itself; the zip restacks it.
2. **The lane never edits the sprint manifest.** The manifest amendment for w2 (parent →
   `mythwave/p7b-freeze-and-verify`, now @ `c2f64a0`, touch sets REGENERATED from the slice JSON rather than hand-listed,
   phantom package-key paths dropped, notes) is a CONDUCTOR precondition and **the conductor amended
   it in the same commit that lands this brief**; the round-4 amendment adds
   `scripts/assemble_review_packet.py` (P8-PACKET-SCRIPT) and `tests/test_shim_layer.py`
   (P8-SHIM-LAYER-RETIRE) to p8's touch set. A manifest defect the lane notices is a `DECISION`
   to report, never an edit.
3. **`publishMode: 'defer'`.** Terminal state is CERTIFIED (tip SHA + exit-gate JSON). No
   `gt create`, no `gt submit`, no `just ship-it` inside the lane. The zip publishes once.
4. **Forwarders (FOR-1547, landed `e9e5897`).** `tools/gen_shims.py` emits, for every resolved row
   with `landed: true` whose vacated `entropic_substrate.__legacy.<x>` address is STILL IMPORTED
   under `src/` or `tests/`, a FORWARDER at that exact vacated path — the same exact-form
   static-import shim as a leaf shim, plus its AST-derived `.pyi`. Emission is DERIVED from a static
   scan (`collect_legacy_importers`), never declared in the fragment. **New tooling this session**
   (`tools/gen_shims.py`, the `ast.ImportFrom` branch of `collect_legacy_importers`, comment
   "MEASURED 2026-09-23, p9 review B3"): the scan now also records `<package>.<name>` for
   PACKAGE-FORM `from` imports whose module resolves to a `__legacy` package DIRECTORY under a scan
   root. So `tests/test_d1_training_step_split.py:38`'s
   `from entropic_substrate.__legacy import … d1_training_step_records` is forwarder-served after p8
   moves the record half; before that change only `node.module` was recorded and no leaf address ever
   matched.
5. **Rung-owned file.** `src/entropic_substrate/ladder/gate/ladder_predecessor.py` (landed by p7)
   carries explicit `__legacy` edges to BOTH p8 targets (`:14` `evidence.d1_capture_artifacts`,
   `:20` `evidence.d1_capture_bundle`) and p11 targets (`:8` `analysis.ladder_analysis`, `:12`
   `analysis.ladder_filing`) — MEASURED with `grep -n "__legacy" <that file>`. **p8 does NOT touch
   it.** It is declared `--rung-owned` in the slice command, leaves every slice, and the lane's own
   `gen_shims` run serves it with forwarders inside this worktree. The post-zip rung rewrites it
   once. NOT in Allowed files; a diff touching it is a `DECISION`.
6. **The serial reconciliation rung.** After the zip, ONE conductor rung rewrites the remaining
   sibling-target imports, deletes the forwarders whose last importer is gone, regenerates shims,
   re-pins regime rows if a reader-set member changed, re-certifies and publishes. The lane must NOT
   anticipate it.

**What the lane must NOT do:** publish · rebase or restack · touch a sibling lane's slice · touch
`ladder/gate/ladder_predecessor.py` · run `just ship-it` or `just container-test` · start a podman
stack · edit the sprint manifest, `tools/mapping.json`, `.importlinter`, `scripts/entrypoints.env`,
the plan, another phase's fragment (the ONE exception is the P8-FRAG migration out of
`p5-kernel.json`), any EXISTING test outside the TWO Q1 carve-outs (the four split-test lines, and
the one `tests/test_shim_layer.py` RED-case retirement, ruling P8-SHIM-LAYER-RETIRE), or any script
other than the ADDITIVE `scripts/assemble_review_packet.py` edit (ruling P8-PACKET-SCRIPT).

**Exact slice command** (copied verbatim from `lane_slices_cmd.sh`; the p8 target list is the
authoritative 52):

```bash
python3 tools/lane_slices.py src tools/mapping.json \
  p8-capture=entropic_substrate.capture_results,entropic_substrate.d1_calibration_records,entropic_substrate.d1_content_addressed_custody,entropic_substrate.d1_journal_snapshot,entropic_substrate.d1_numerical_defect,entropic_substrate.d1_observability,entropic_substrate.d1_tracking,entropic_substrate.d1_training_step_records,entropic_substrate.data.build,entropic_substrate.data.enwik8,entropic_substrate.data.fineweb,entropic_substrate.data.synthetic,entropic_substrate.evidence.d1_abort_evidence,entropic_substrate.evidence.d1_capture_aggregate,entropic_substrate.evidence.d1_capture_aggregate_codec,entropic_substrate.evidence.d1_capture_aggregate_validation,entropic_substrate.evidence.d1_capture_artifacts,entropic_substrate.evidence.d1_capture_bundle,entropic_substrate.evidence.d1_capture_declaration_codec,entropic_substrate.evidence.d1_capture_filesystem,entropic_substrate.evidence.d1_capture_ports,entropic_substrate.evidence.d1_capture_preparation_custody,entropic_substrate.evidence.d1_capture_preparation_custody_repository,entropic_substrate.evidence.d1_capture_roster,entropic_substrate.evidence.d1_capture_service,entropic_substrate.evidence.d1_capture_subprocess,entropic_substrate.evidence.d1_capture_validation,entropic_substrate.evidence.d1_journal,entropic_substrate.evidence.d1_journal_values,entropic_substrate.evidence.d1_lifecycle_repository,entropic_substrate.evidence.d1_lifecycle_values,entropic_substrate.evidence.d1_process_manager,entropic_substrate.evidence.d1_process_replay,entropic_substrate.evidence.d1_telemetry,entropic_substrate.evidence.d1_telemetry_measurements,entropic_substrate.evidence.d1_telemetry_payloads,entropic_substrate.evidence.d1_telemetry_query,entropic_substrate.evidence.d1_telemetry_values,entropic_substrate.evidence.d1_terminal_defect_capture,entropic_substrate.evidence.d1_terminal_defect_query,entropic_substrate.evidence.resource_billing,entropic_substrate.evidence.resource_billing_reconciliation,entropic_substrate.evidence.resource_ledger,entropic_substrate.evidence.resource_ledger_codec,entropic_substrate.evidence.resource_meter_models,entropic_substrate.evidence.resource_meter_runtime,entropic_substrate.evidence.resource_meter_strategy,entropic_substrate.evidence.resource_work,entropic_substrate.evidence.resource_work_codec,entropic_substrate.mlflow_aggregate,entropic_substrate.sinks,entropic_substrate.telemetry \
  --regenerable tests/test_identity_epoch.py,tools/regime_rows.py,tests/test_regime_rows.py,docs/identity/read-old-table.md,.secrets.baseline \
  --rung-owned entropic_substrate/ladder/gate/ladder_predecessor.py --allow-dependencies --json
```

(The conductor runs it with all three lanes in one invocation; the p9 and p11 specs are in
`lane_slices_cmd.sh`. `--rung-owned` has TEETH: a declared file that fewer than two slices carry is a
malformed declaration, exit 2, so the flag cannot decay into an ignore list.)

## Contract surface owned

| contract                                                                                                                                                                                                   | kind     | frozen_after |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| the 52 `ladder.capture.*` addresses enumerated under *Scope*                                                                                                                                               | contract | p13          |
| destination package `__init__.py` shape for `ladder/capture/{custody,records,reader,resource,ledger}`: DOCSTRING-ONLY, no re-exports (p4/p6/p7 precedent)                                                  | contract | p13          |
| `ladder/capture/reader/telemetry.py` remains the ONLY admissible read side for `results/*.jsonl`; inadmissible telemetry RAISES, never returns empty                                                       | door     | p8           |
| the D1 observer adapters under `ladder/capture/adapters/`, MOVED rename-only with `D1MlflowMetricClient` still PUBLIC (ruling P8-OBSERVERS-RENAME-ONLY; the p3b-R1 re-basing is FOR-1552's, RETRACTION 10) | contract | p13          |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item
below is LANDED at the parent tip.

- kernel addresses (p4, p5, p5b): MEASURED, the 52 legacy sources' landed-address imports resolve to
  `kernel.identity.d1_primitives`, `kernel.identity.factory_attestation`,
  `kernel.runtime.{config,checkpoint,logging,paths}`, `kernel.lifecycle`, `kernel.artifacts`,
  `kernel.verdict.{d1_types,d1_evaluation_values}`,
  `kernel.evidence_class.{resource_provenance,resource_provenance_codec}` — the *Landed* bucket below
  counts them: 71 lines at `c2f64a0` (73 at `79d8228`, RETRACTION 8), enumerated per line in the pack
- core addresses (p6): `core.corpus.batching`, `core.sentence.records`
- ladder.register addresses (p7): `ladder.register.{d1_plan,d1_config,d1_provenance,d1_family,d1_execution}`
- **p3b's capture rail:** `ladder/capture/ports/metric_sink.py` (`MetricSink` protocol: `log`,
  `log_artifact`, `finish`), `ladder/capture/ports/run_descriptor.py` (`TrackingRunDescriptor` `:53`,
  `ExtractedMetrics` `:86`, `MetricExtractor` `:97`, `LegacyMetricExtractor` `:103`),
  `ladder/capture/adapters/{jsonl,mlflow,composite}.py`
- `tools/gen_shims.py` INCLUDING the FOR-1547 forwarder path and its package-form importer scan, the
  landed-set fragment convention, `tests/test_legacy_coupling.py` (EQUALITY arm),
  `tests/test_no_old_prefix.py`, `tests/test_module_identity.py` (sentinel reverse-map)
- `tools/regime_rows.py` + `tests/test_regime_rows.py` + `tests/test_identity_epoch.py` (identity
  mirrors, ruling Q1) — WRITTEN here, see R-PIN
- `tools/lane_slices.py`, `tools/import_graph.py` (measurement oracles). **`tools/remap_check.py`
  is NOT a consumed oracle here — see RETRACTION 5 (P9-REMAP, FOR-1549).**
- the `d1_training_step` / `d1_calibration` SPLIT CONTRACT (p5), both halves still under `__legacy/`

## Sacred invariants

The five below are quoted verbatim from `CLAUDE.md` (North star and *Non-negotiable numerics*). They
bind every phase without exception. `SI-*` ids are local to this brief set; the text is the contract.
Each guard-teeth recipe names the MEASURED instrument that goes RED, or says no instrument exists.

| id   | verbatim invariant                                                                                                          | tripwire                                                                                                                                                                  | guard-teeth recipe (house rule 5, both arms)                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | This lane moves NO numerics module: `touch_set ∩ core/** = ∅`. p6 landed `core/geometry`; its FP32 islands and `tests/test_kappa_autocast_island.py` stay byte-identical. | The load-bearing arm is p6's autocast-STATE spy `tests/test_kappa_autocast_island.py`, never a dtype arm (`_geometry_dtype()` floors to fp32 regardless of autocast state — p0 round-C, FOR-1517). This lane records the arm UNTOUCHED: `git diff mythwave/p7b-freeze-and-verify -- tests/test_kappa_autocast_island.py src/entropic_substrate/core/geometry/` EMPTY (the PARENT tip, not p7 — RETRACTION 7).                                                                                                              |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | `touch_set ∩ core/geometry = ∅`.                                                                                                                                          | Widen the κ clamp bound by one ulp and feed a boundary input; `tests/test_kappa_geometry.py`'s κ-domain arm goes RED. Revert in-branch.                                                                                                                                                                                                                                                                                                                                                                                    |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | `touch_set ∩ core/substrate = ∅`.                                                                                                                                         | MEASURED instruments, both in `tests/test_blocks.py`: finiteness is `:169 test_kappa_moe_model_reports_stats_and_aux_loss` (`:174 assert out.aux_loss is not None and torch.isfinite(out.aux_loss)` — the load-balance term IS `aux_loss`; the literal token `load_balance` appears nowhere under `tests/`), and κ-reachability is `:248 test_kappa_attention_gradient_reaches_kappa_at_exactly_zero`. Mutation: return a non-finite `aux_loss`, or detach κ from it. Revert in-branch.                                    |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | `touch_set ∩ core/sentence = ∅`. `data/fineweb.py:16` IMPORTS `core.sentence.records`; it must not change one carrier line.                                               | MEASURED instrument: `tests/test_sentence_numerics.py:17 test_magnitude_interface_preserves_the_actual_carrier` (with `:35 test_direction_and_magnitude_are_separate_controls`). Mutation: append a `UnitNorm` to the carrier output → RED. Revert in-branch.                                                                                                                                                                                                                                                              |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | `touch_set ∩ core/latent = ∅`. MEASURED: no p8 target imports `core.latent.*`, so the latent law is unreachable from this slice.                                          | **NO INSTRUMENT: EXHIBIT OR RETRACT.** RETRACTION 2026-09-24 (house rule 4): `grep -rln "future_law\|log_magnitude" tests/` names 23 files (e.g. `tests/test_predictive_state_future_law.py`), so the earlier sentence "names no test" was wrong; none of those tests is reachable from this slice, and the row's compliance instrument stays `touch_set ∩ core/latent = ∅` (independently confirmed via the forbidden-path diff). The lane either exhibits a named failing test or retracts the row with the measurement. |

House rules from `CLAUDE.md`, verbatim headline, each binding in this lane:

| id   | verbatim house rule                                                            | tripwire                                                                                                  | guard-teeth recipe                                                                                                                                                                                                                                                                      |
| ---- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HR-1 | "Publish with `just ship-it`, never a bare `gt submit`."                       | A lane under `publishMode:'defer'` that publishes at all. `just ship-it` runs ONCE, at the zip.           | Inspect the lane transcript for any `gt create`, `gt submit` or `just ship-it`; presence is a BLOCKED verdict, not a note.                                                                                                                                                              |
| HR-2 | "Probe sequentially. Never draw a conclusion from concurrent containers."      | Three lanes on one 4 vCPU / 4 GB podman VM; two MLflow stacks at once make even a control appear hung.    | This lane starts NO container. Any container conclusion in the wave must name the single container it came from.                                                                                                                                                                        |
| HR-3 | "Never fix flakiness by raising a timeout or a retry budget."                  | A gate that goes green only after a timeout bump. `sinks.py` is a p8 target and carries that measurement. | A diff that raises any timeout or retry constant is a BLOCKED verdict unless it carries the measurement showing the cause was found.                                                                                                                                                    |
| HR-4 | "Comments record measurements, with the measurement."                          | A comment in this diff stating a fact with no measurement attached.                                       | A wrong measured claim is retracted explicitly, in the commit that fixes it. This lane carries ELEVEN such retractions (see *Scope*, *Rewrite-on-arrival*, *ACCEPTANCE*, *Non-goals*, *Observability conformance* and the epoch note in the header; SEVEN until the round-4 amendment). |
| HR-5 | "TDD is non-negotiable, and guard-teeth prove both arms."                      | A guard landed without its RED arm demonstrated.                                                          | Every guard carries the mutation listed under *Guard teeth by mutation*, performed and reverted in-branch, evidence in the lane report.                                                                                                                                                 |
| HR-6 | "Secrets never enter git."                                                     | gitleaks runs at pre-commit and pre-push. `sinks.py` and `mlflow_aggregate.py` read MLflow configuration. | No credential, token or `.env` value in any diff of this lane.                                                                                                                                                                                                                          |
| HR-7 | "Upgrading an existing venv across the mlflow → mlflow-skinny swap breaks it." | `import mlflow` raising `ModuleNotFoundError` in a fresh lane worktree.                                   | Fix with `uv sync --all-groups --reinstall-package mlflow-skinny`, never by pinning around it. A fresh worktree needs `uv sync --all-groups` before any gate.                                                                                                                           |

## Scope

The DVC-CAPTURED evidence stage: **52 leaf modules**, MEASURED as the `tools/mapping.json` rows whose
`dest` starts with `entropic_substrate.ladder.capture`, whose legacy source is a leaf `.py` under
`src/entropic_substrate/__legacy/`, and which no landed-set fragment already marks `landed: true`.
Zero behaviour change except the ONE permitted observability class (see *Observability conformance*).

MEASURED sub-package partition, by `tools/mapping.json` note group (command: read each target's
`dest` + `note` out of `tools/mapping.json` and count by group):

| sub-package | count  | mapping note that defines the group                                                                                                                                   |
| ----------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| custody     | 19     | "local->DVC/Git custody hexagon: ports, filesystem/subprocess adapters, service"                                                                                      |
| records     | 12     | "authoritative journal + telemetry records and their strict parsers" (10) + the 2 explicit `dest: ladder.capture.records` split rows                                  |
| resource    | 9      | "resource/billing evidence with per-reading EvidenceClass tagging"                                                                                                    |
| reader      | 8      | "the only admissible read side for results/*.jsonl, plus its sinks" — **6 land in `reader/`, 2 (`d1_observability`, `d1_tracking`) in `adapters/` per ruling p3b-R1** |
| ledger      | 4      | "durable process-lifecycle ledger beneath the pure typestate"                                                                                                         |
| **total**   | **52** |                                                                                                                                                                       |

RETRACTION 1 (house rule 4): v1's *Scope* said "the 53 remaining `ladder.capture` modules" and both
v1 and the sub-packages ruling said "custody 20, records 10, resource 9, reader 9, ledger 4". BOTH
are wrong and are retracted, not overwritten. The measured partition is 19 / 12 / 9 / 8 / 4 = 52. The
five-sub-package DECOMPOSITION stands (ruled 2026-09-17); only the per-group counts move.

RETRACTION 2 (house rule 4): the w2 audit at `331dc1e` recorded "p8: **zero** `action: split` leaf
rows … the record halves have **no `tools/mapping.json` row** … `tools/lane_slices.py` is blind to
them". All three clauses are WRONG and are retracted, not overwritten. MEASURED at `79d8228`: both
record halves have mapping rows — `entropic_substrate.d1_training_step_records` and
`entropic_substrate.d1_calibration_records`, each with `dest: entropic_substrate.ladder.capture.records`
and **`action: split`** (so p8 owns TWO `action: split` leaf rows, not zero; p9 owns the two torch
halves, also `action: split`). They are already shimmed at
`src/entropic_substrate/d1_{training_step,calibration}_records.py`, carried in `p5-kernel.json` as
`landed: false`, and named by `tools/gen_shims.py`'s own `pending:` documentation. The slice tool sees
them and the pack's 52 includes them.

**Package keys are NEVER fragment rows.** `entropic_substrate.evidence` and `entropic_substrate.data`
are package keys: `tools/gen_shims.py::merge_fragments` raises `ManifestError` on a fragment naming
one, and `tools/lane_slices.py::parse_lane` exits 2. Facade disposition, MEASURED by reading the files:

| package    | moved facade `__legacy/<pkg>/__init__.py`                                                                                                                                                                                          | live old-path facade `src/entropic_substrate/<pkg>/__init__.py`                       |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `data`     | ALREADY the p3 "Retracted facade body" one-liner. UNTOUCHED.                                                                                                                                                                       | stays VERBATIM, untouched                                                             |
| `evidence` | 305 bytes, docstring + `from __future__ import annotations`, zero re-exports — docstring-only but NOT yet the retracted-facade form. **p8 does NOT touch it** (ruling FACADE-EMPTY; the package is not empty after p8, see below). | stays VERBATIM, untouched (MEASURED byte-identical text to the `__legacy` copy today) |

**RULING FACADE-EMPTY (conductor 2026-09-23, wave-wide).** A moved package facade
`__legacy/<pkg>/__init__.py` is retracted to the p3 one-liner ONLY in the commit after which the
package DIRECTORY holds no leaf module — measured with
`ls src/entropic_substrate/__legacy/<pkg>/*.py` returning only `__init__.py`. Package facades are in
NO lane slice, so the retraction never belongs to a lane that leaves leaves behind.

MEASURED at `79d8228` (`ls src/entropic_substrate/__legacy/evidence/*.py` minus `__init__.py`, each
remaining name resolved through `tools/mapping.json` against the three w2 target sets): the directory
holds **48** leaves. p8 moves **37**, p11 moves **10**, and **ONE** leaf survives the whole wave:

| leaf remaining after p8 (37) + p11 (10) | mapping `dest`                        | owning phase                                                                                                                                                                 |
| --------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evidence/ladder_campaign.py`           | `entropic_substrate.kernel.artifacts` | the phase that moves its still-quarantined GATE half (p5b landed the artifact half only; `landed: false` in `p5b-ladder-campaign.json`) — **p12**, in neither w2 lane nor p9 |

The p11 leaves, MEASURED (ten, not nine): `d1_decision_validation`, `d1_filing`, `d1_filing_assembly`,
`d1_filing_edges`, `d1_filing_graph`, `d1_filing_receipts`, `d1_gate`, `d1_process_filing`,
`d1_review_adjudication`, `d1_validator`.

So **p8 does NOT empty `__legacy/evidence/` and neither does p11**: the facade is retired by **p12**,
the phase after which the directory is finally empty. `__legacy/evidence/__init__.py` is therefore OUT
of this lane's Allowed files and listed under *Deliberately EXCLUDED*.

RETRACTION 6 (house rule 4): this brief's own earlier v2 draft read "p8 applies the retracted-facade
form when its last `evidence` leaf moves out", put the file in Allowed files and gave it a step in
commit 5. That is retracted, not overwritten: MEASURED, 11 `evidence` leaves survive p8 (10 p11 plus
`ladder_campaign`), so the package is not empty and the precondition never fires here. Every
"last leaf moves" phrasing is struck from this brief.

### The 52 targets

| legacy source (moves)                                            | destination address                                                |
| ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| `__legacy/d1_content_addressed_custody.py`                       | `ladder.capture.custody.d1_content_addressed_custody`              |
| `__legacy/data/{build,enwik8,fineweb,synthetic}.py`              | `ladder.capture.custody.{build,enwik8,fineweb,synthetic}`          |
| `__legacy/evidence/d1_capture_aggregate.py`                      | `ladder.capture.custody.d1_capture_aggregate`                      |
| `__legacy/evidence/d1_capture_aggregate_codec.py`                | `ladder.capture.custody.d1_capture_aggregate_codec`                |
| `__legacy/evidence/d1_capture_aggregate_validation.py`           | `ladder.capture.custody.d1_capture_aggregate_validation`           |
| `__legacy/evidence/d1_capture_artifacts.py`                      | `ladder.capture.custody.d1_capture_artifacts`                      |
| `__legacy/evidence/d1_capture_bundle.py`                         | `ladder.capture.custody.d1_capture_bundle`                         |
| `__legacy/evidence/d1_capture_declaration_codec.py`              | `ladder.capture.custody.d1_capture_declaration_codec`              |
| `__legacy/evidence/d1_capture_filesystem.py`                     | `ladder.capture.custody.d1_capture_filesystem`                     |
| `__legacy/evidence/d1_capture_ports.py`                          | `ladder.capture.custody.d1_capture_ports`                          |
| `__legacy/evidence/d1_capture_preparation_custody.py`            | `ladder.capture.custody.d1_capture_preparation_custody`            |
| `__legacy/evidence/d1_capture_preparation_custody_repository.py` | `ladder.capture.custody.d1_capture_preparation_custody_repository` |
| `__legacy/evidence/d1_capture_roster.py`                         | `ladder.capture.custody.d1_capture_roster`                         |
| `__legacy/evidence/d1_capture_service.py`                        | `ladder.capture.custody.d1_capture_service`                        |
| `__legacy/evidence/d1_capture_subprocess.py`                     | `ladder.capture.custody.d1_capture_subprocess`                     |
| `__legacy/evidence/d1_capture_validation.py`                     | `ladder.capture.custody.d1_capture_validation`                     |
| `__legacy/d1_training_step_records.py`                           | `ladder.capture.records.d1_training_step_records`                  |
| `__legacy/d1_calibration_records.py`                             | `ladder.capture.records.d1_calibration_records`                    |
| `__legacy/evidence/d1_abort_evidence.py`                         | `ladder.capture.records.d1_abort_evidence`                         |
| `__legacy/evidence/d1_journal.py`                                | `ladder.capture.records.d1_journal`                                |
| `__legacy/evidence/d1_journal_values.py`                         | `ladder.capture.records.d1_journal_values`                         |
| `__legacy/evidence/d1_telemetry.py`                              | `ladder.capture.records.d1_telemetry`                              |
| `__legacy/evidence/d1_telemetry_measurements.py`                 | `ladder.capture.records.d1_telemetry_measurements`                 |
| `__legacy/evidence/d1_telemetry_payloads.py`                     | `ladder.capture.records.d1_telemetry_payloads`                     |
| `__legacy/evidence/d1_telemetry_query.py`                        | `ladder.capture.records.d1_telemetry_query`                        |
| `__legacy/evidence/d1_telemetry_values.py`                       | `ladder.capture.records.d1_telemetry_values`                       |
| `__legacy/evidence/d1_terminal_defect_capture.py`                | `ladder.capture.records.d1_terminal_defect_capture`                |
| `__legacy/evidence/d1_terminal_defect_query.py`                  | `ladder.capture.records.d1_terminal_defect_query`                  |
| `__legacy/evidence/resource_billing.py`                          | `ladder.capture.resource.resource_billing`                         |
| `__legacy/evidence/resource_billing_reconciliation.py`           | `ladder.capture.resource.resource_billing_reconciliation`          |
| `__legacy/evidence/resource_ledger.py`                           | `ladder.capture.resource.resource_ledger`                          |
| `__legacy/evidence/resource_ledger_codec.py`                     | `ladder.capture.resource.resource_ledger_codec`                    |
| `__legacy/evidence/resource_meter_models.py`                     | `ladder.capture.resource.resource_meter_models`                    |
| `__legacy/evidence/resource_meter_runtime.py`                    | `ladder.capture.resource.resource_meter_runtime`                   |
| `__legacy/evidence/resource_meter_strategy.py`                   | `ladder.capture.resource.resource_meter_strategy`                  |
| `__legacy/evidence/resource_work.py`                             | `ladder.capture.resource.resource_work`                            |
| `__legacy/evidence/resource_work_codec.py`                       | `ladder.capture.resource.resource_work_codec`                      |
| `__legacy/capture_results.py`                                    | `ladder.capture.reader.capture_results`                            |
| `__legacy/d1_journal_snapshot.py`                                | `ladder.capture.reader.d1_journal_snapshot`                        |
| `__legacy/d1_numerical_defect.py`                                | `ladder.capture.reader.d1_numerical_defect`                        |
| `__legacy/d1_observability.py`                                   | `ladder.capture.adapters.d1_observability` (ruling p3b-R1)         |
| `__legacy/d1_tracking.py`                                        | `ladder.capture.adapters.d1_tracking` (ruling p3b-R1)              |
| `__legacy/mlflow_aggregate.py`                                   | `ladder.capture.reader.mlflow_aggregate`                           |
| `__legacy/sinks.py`                                              | `ladder.capture.reader.sinks`                                      |
| `__legacy/telemetry.py`                                          | `ladder.capture.reader.telemetry`                                  |
| `__legacy/evidence/d1_lifecycle_repository.py`                   | `ladder.capture.ledger.d1_lifecycle_repository`                    |
| `__legacy/evidence/d1_lifecycle_values.py`                       | `ladder.capture.ledger.d1_lifecycle_values`                        |
| `__legacy/evidence/d1_process_manager.py`                        | `ladder.capture.ledger.d1_process_manager`                         |
| `__legacy/evidence/d1_process_replay.py`                         | `ladder.capture.ledger.d1_process_replay`                          |

Each moved leaf gets a typed shim pair at its old path (`src/entropic_substrate/<old path>.py` +
`.pyi`), regenerated by `tools/gen_shims.py`, never hand-edited. Shim form, quoted from the
generator's own output:

```python
# Generated by gen_shims.py from .claude/workflows/landed-set -- do not hand-edit.
# Alias shim for entropic_substrate.<old> -> entropic_substrate.ladder.capture.<new> (docs/plans/src-hexagonal-refactor.plan.md 'Shim form (required)').
import sys

import entropic_substrate.ladder.capture.<new> as _target

sys.modules[__name__] = _target
```

The `_target` is the NEW `ladder.capture.*` address, never `__legacy`. RETRACTION 3 (house rule 4):
v1's *Already ruled* "shim form" row read `import entropic_substrate.__legacy.<x> as _target` — the
pre-p3 form, WRONG for a phase that lands a destination. Retracted, not overwritten. For the ONE
entrypoint row (`entropic_substrate.telemetry`) the generator emits the ENTRYPOINT variant instead,
which `--verify` accepts as an exact alternative: MEASURED, the live shim at
`src/entropic_substrate/telemetry.py` is **15 lines / 5 statements** (`:1-15`), whose
`if __name__ == "__main__"` block calls
`runpy.run_module("entropic_substrate.__legacy.telemetry", run_name="__main__", alter_sys=True)`
before the ordinary 3-statement rebind.

**MEASURED sub-package assignment caveat.** `tools/mapping.json`'s `dest` is PACKAGE-LEVEL only
(`entropic_substrate.ladder.capture`, except the two explicit `.records` rows), so the leaf-to-
sub-package assignment above is DERIVED from each row's `note` group, not read off `dest`. The
fragment restates each `target` verbatim, as the generator's schema requires. This is compatible with
`tests/test_module_identity.py`'s cross-check, which accepts `normalized == dest` OR
`normalized.startswith(dest + ".")` (`:126-133`) — a sub-package target passes. A row whose note
group the lane reads differently is a `DECISION`, not a quiet re-assignment.

## Landed importers (rewrite-on-arrival)

MEASURED with `tools/lane_slices.py` (slice JSON `slices-1d97033.json`) and confirmed by
`grep -n "__legacy" <file>`: **6** landed modules import a p8 target by its explicit
`entropic_substrate.__legacy.<x>` address. p8 rewrites every line ONCE, on arrival.

| landed importer                                               | line(s)                                                                        | old import                                          | becomes                                                   | `h-checkpoint-reader-v2` member? |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------- | --------------------------------------------------------- | -------------------------------- |
| `src/entropic_substrate/composition/source_cli.py`            | `10`, `99`                                                                     | `__legacy.data.fineweb`                             | `ladder.capture.custody.fineweb`                          | **YES**                          |
| `src/entropic_substrate/ladder/run/execution.py`              | `11`, `12`                                                                     | `__legacy.evidence.resource_meter_{models,runtime}` | `ladder.capture.resource.resource_meter_{models,runtime}` | **YES**                          |
| `src/entropic_substrate/ladder/run/generation.py`             | `14`                                                                           | `__legacy.evidence.resource_meter_models`           | `ladder.capture.resource.resource_meter_models`           | **YES**                          |
| `src/entropic_substrate/ladder/run/span_diagnostics.py`       | `11`                                                                           | `__legacy.evidence.resource_meter_models`           | `ladder.capture.resource.resource_meter_models`           | **YES**                          |
| `src/entropic_substrate/ladder/capture/adapters/composite.py` | `123` (function-local) + docstring cites at `5,20,22,30,46,114,120`            | `__legacy.sinks`                                    | `ladder.capture.reader.sinks`                             | no                               |
| `src/entropic_substrate/ladder/capture/adapters/mlflow.py`    | `144`, `262` (function-local) + docstring cites at `4,10,24,25,28,140,259,484` | `__legacy.sinks`                                    | `ladder.capture.reader.sinks`                             | no                               |

The two adapters' function-local imports are the deliberate p3b cycle-break; p8 rewrites the address
and keeps the import LOCAL. Their docstrings name the old address and are rewritten in the SAME commit
(house rule 4: a docstring is a measured claim about where the residue lives).

**R-PIN OWED.** Four of the six are members of `h-checkpoint-reader-v2`'s 51-module `_READER_MODULES`
tuple (`src/entropic_substrate/ladder/run/checkpoint.py:57-109`; the audit cites `:60,89,90,97`). So
this lane re-pins `_CHECKPOINT_REGIME_TABLE` with `tools/regime_rows.py` in the SAME commit as the
rewrite — label and module SET unchanged, only the composed digests move — and re-measures
`h-source-reader-v3`'s table (`ladder/capture/capture.py:29-37`, 7 members) UNCHANGED as the CONTROL
(the `p6-core.json` form). MEASURED: **no p8 destination address** and no p8 target appears in the
source-reader tuple, and neither `checkpoint.py` nor `capture.py` is in any lane slice.
`docs/identity/read-old-table.md` gains a dated p8 row naming 4 touched checkpoint members of 51 and
0 touched source members of 7.

RETRACTION 7 (house rule 4): every earlier round of this brief wrote the identity CONTROL as
`git diff mythwave/p7-register-gate -- tests/test_identity_epoch.py` (and the SI-1 arm the same way).
That is retracted, not overwritten: the parent is now `mythwave/p7b-freeze-and-verify` (tip
`c2f64a0` since the second 2026-09-23 epoch move, `b7af1aa` before it),
and **p7b itself re-pinned the checkpoint regime rows and froze the reader label and module set**
(FOR-1550, commits `40e72ac` / `b7af1aa`), so a diff against p7 is NON-EMPTY before the lane touches
anything and would read as this lane's change. The control is
`git diff mythwave/p7b-freeze-and-verify -- tests/test_identity_epoch.py` — the PARENT tip — and the
same substitution applies to every control command in this brief. Tool command:
`uv run --no-sync python tools/regime_rows.py --python <measured> --torch <measured> --pysbd <measured>`
— regime strings are MEASURED inputs, never guessed.

## Rewrite-on-arrival for the 52 moved modules' own imports

MEASURED (pack `pack-p8-capture.json`, every line enumerated `file:line`): the 52 legacy sources carry
**193** `entropic_substrate.*` import lines, `missing_source: 0`, in four buckets.

| bucket                              | lines                                               | becomes                                                                                                                              |
| ----------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| intra-set (sibling p8 target)       | **111**                                             | the sibling's NEW `ladder.capture.<sub>.<leaf>` address (at `c2f64a0`, `data/build.py:15-17` spell theirs `__legacy.data.<x>`)       |
| landed address                      | **71** at `c2f64a0` (73 at `79d8228`, RETRACTION 8) | the `kernel.*` / `core.*` / `ladder.{register,capture}.*` address the pack's `landed` list names per line                            |
| sibling lane target (p9)            | **4**                                               | p8's OWN record-half addresses — see below; this drives p8→p9 to **zero**                                                            |
| already-landed address (no rewrite) | **7** at `c2f64a0` (5 at `79d8228`, RETRACTION 8)   | UNCHANGED: `sinks.py:65-69` (p3b's `ladder.capture.{ports,adapters}.*`, not quarantined) and, since `c2f64a0`, `data/build.py:18,19` |

MEASURED DIFFERENCE from the audit: the audit's table at `331dc1e` says p8's landed bucket is **72**;
the pack at `79d8228` says **73**; the build.py re-measure at `c2f64a0` gives **71** (RETRACTION 8).
The lane MEASURES its own number; no figure is a target.

**RETRACTION 8 (house rule 4): `__legacy/data/build.py`'s rows, ruling DATA-BUILD-RUNG.** The buckets
were measured at `79d8228`, where build.py's five first-party lines read their OLD names and the pack
counted 3 intra-set (`:17-19` `data.{enwik8,fineweb,synthetic}`) + 2 landed (`:15` `config` →
`kernel.runtime.config`, `:16` `data.batching` → `core.corpus.batching`). Those rows, and the totals
**landed 73** and **already-landed 5**, are retracted, not overwritten: the conductor rung `c2f64a0`
rewrote all five lines on the shared parent. RE-MEASURED at `c2f64a0` with
`grep -n "^from entropic_substrate" src/entropic_substrate/__legacy/data/build.py`:

| line  | import at `c2f64a0`       | bucket at `c2f64a0` | p8 on arrival (SCC commit)                   |
| ----- | ------------------------- | ------------------- | -------------------------------------------- |
| `:15` | `__legacy.data.enwik8`    | intra-set           | rewrite → `ladder.capture.custody.enwik8`    |
| `:16` | `__legacy.data.fineweb`   | intra-set           | rewrite → `ladder.capture.custody.fineweb`   |
| `:17` | `__legacy.data.synthetic` | intra-set           | rewrite → `ladder.capture.custody.synthetic` |
| `:18` | `core.corpus.batching`    | already-landed      | UNCHANGED, already the destination address   |
| `:19` | `kernel.runtime.config`   | already-landed      | UNCHANGED, already the destination address   |

Buckets at `c2f64a0`: intra-set 111 → **111** (count unchanged; three lines now use the explicit
`__legacy` spelling), landed 73 → **71**, p9 sibling 4 → **4**, already-landed 5 → **7**; total 193 →
**193**. Only build.py was re-measured; the other 51 sources are byte-identical between `79d8228` and
`c2f64a0` (header `git diff --stat`). **Rewrite-once is AMENDED for this ONE quarantined file**
(ruling DATA-BUILD-RUNG): the rung touched all five lines once, and p8 touches them once more on
arrival. The ruling says p8 "rewrites these five lines once more"; MEASURED, only `:15-17` change
bytes on arrival, and `:18-19` already read their destination. NO forwarder is owed for `:15-17`: all
four `data` leaves land together in the SCC commit, so their only static importer of
`__legacy.data.{enwik8,synthetic}` (build.py) moves in the same commit, while `__legacy.data.fineweb`
is forwarder-served until commit 6 because `composition/source_cli.py:10,99` imports it.

**RULED (D-4, conductor 2026-09-23; surfaced by this amendment).** The rung's own test `tests/test_legacy_data_build_cold_import.py` (unit-marked) originally pinned the literal address `entropic_substrate.__legacy.data.build` and its `__name__`, which the SCC commit would have broken (no static importer, so no forwarder; and the old-name shim would resolve to the landed address). The conductor changed the test on the parent in the same docs commit that lands these amendments (the lane's startTip carries it): it now derives the module's CURRENT address live through `gen_shims.load_fragments -> merge_fragments -> resolve_all_modules` (the landed target once `p8-capture.json` marks `entropic_substrate.data.build` landed, its `__legacy` address while quarantined) and asserts the entered-first cold import against that address; the old-name control asserts the same resolved name. It therefore FOLLOWS p8's move with no edit and no carve-out; the lane runs it at every landing commit. The earlier `[predicted]` RED for `:48`/`:50`/`:57` is retracted as no longer applicable (the lines it cited no longer exist).

**Every rewritten import is an ABSOLUTE static import** of the landed address
(`from entropic_substrate.<landed> import …`), never a relative import and never an `importlib` or
`runpy` string (second design gate's constraint): `build_reader`, `tools/import_graph.py`,
`collect_legacy_importers` and the `lint-imports` layers contract see absolute static imports only,
and a relative import passes `build_reader`'s substring check while shipping a broken standalone
reader.

**The p8→p9 back-edge, driven to zero.** The four sibling lines read RECORD symbols off the torch
halves' OLD names. VERIFIED by reading both the import lines and the record halves' definitions:

| line                                                              | symbols imported                                                                                       | defined in                                                     | rewrites to                                       |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------- |
| `__legacy/evidence/d1_telemetry_measurements.py:9`                | `D1ComponentGradientEvidence`, `D1GradientEvidenceKind`, `D1GradientReachability`, `D1TrainStepResult` | `__legacy/d1_training_step_records.py:48,39,76,109` — ALL FOUR | `ladder.capture.records.d1_training_step_records` |
| `__legacy/evidence/d1_telemetry_query.py:10`                      | `D1TrainStepResult`                                                                                    | `__legacy/d1_training_step_records.py:109`                     | `ladder.capture.records.d1_training_step_records` |
| `__legacy/evidence/d1_telemetry_values.py:19`                     | `D1TrainStepResult`                                                                                    | `__legacy/d1_training_step_records.py:109`                     | `ladder.capture.records.d1_training_step_records` |
| `__legacy/evidence/d1_telemetry_payloads.py:178` (function-local) | `D1BaselineCalibrationAttestation`                                                                     | `__legacy/d1_calibration_records.py:21`                        | `ladder.capture.records.d1_calibration_records`   |

Every imported symbol IS owned by a record half that p8 itself moves, so none of the four needs the
`__legacy` address and none needs a FOR-1547 forwarder. `depends_out.p9-run-lower` goes 4 → **0**. Had
any symbol lived only in a torch half, that line would stay on
`entropic_substrate.__legacy.d1_{training_step,calibration}` and the brief would say so; MEASURED,
none does.

**No moved module keeps an explicit `__legacy` import.** MEASURED: 111 + 71 + 4 + 7 = 193 = every
line at `c2f64a0` (111 + 73 + 4 + 5 at `79d8228`, RETRACTION 8), and after the rewrites all 193
resolve to a LANDED address. `ladder/capture/**` therefore adds
ZERO landed→`__legacy` edges.

## `legacy_edge_budget` prediction `[predicted]`

At the first epoch move, parent tip `mythwave/p7b-freeze-and-verify` @ `b7af1aa` MEASURED **45** landed→`__legacy` edges
(`import_graph.build_graph(src/entropic_substrate, "entropic_substrate")` plus the dotted-prefix
classifier of `tests/test_legacy_coupling.py::count_landed_to_legacy_edges`), exactly equal to the
highest-ordered committed budget (`p7-register-gate.json` = 45; no `p7b-*.json` fragment exists) —
lens check passed at the NEW epoch, and identical to the count measured at the measurement epoch `79d8228`, so
p7b's re-pin moved no import edge. RE-CHECKED at the current tip `c2f64a0`:
`tests/test_legacy_coupling.py` 11 passed (equality against the committed 45). build.py is
quarantined, so the DATA-BUILD-RUNG rewrite adds no landed→`__legacy` edge.

p8 REMOVES 7 distinct `(landed module, __legacy module)` pairs, all from the six rewritten importers:

```text
composition.source_cli            -> __legacy.data.fineweb                        (1)
ladder.capture.adapters.composite -> __legacy.sinks                               (1)
ladder.capture.adapters.mlflow    -> __legacy.sinks                               (1)
ladder.run.execution              -> __legacy.evidence.resource_meter_models      (1)
ladder.run.execution              -> __legacy.evidence.resource_meter_runtime     (1)
ladder.run.generation             -> __legacy.evidence.resource_meter_models      (1)
ladder.run.span_diagnostics       -> __legacy.evidence.resource_meter_models      (1)
```

p8 ADDS **0**: the 193 own-import lines all resolve to landed addresses (previous section), and the
two `ladder.gate.ladder_predecessor → __legacy.evidence.d1_capture_{artifacts,bundle}` edges are
NEITHER removed NOR added — that file is rung-owned and its `__legacy` addresses keep resolving
through FOR-1547 forwarders, so the pair survives unchanged.

**Predicted: 45 − 7 + 0 = 38.** Labelled `[predicted]`; the lane MEASURES its own number and
reproduces nothing from this paragraph. DIRECTION for this lane: DECREASING. That is an outcome, not
a requirement — see *Already ruled*, BUDGET-MONOTONE retracted.

**BUDGET-PER-COMMIT (ruled 2026-09-23, wave-wide).** `tests/test_legacy_coupling.py::load_committed_budget`
does a bare `int(top["legacy_edge_budget"])` on the highest-ordered fragment present, so the moment
`p8-capture.json` exists with rows (the first landing commit, FRAGMENT-PER-COMMIT) that lookup selects IT. The key
must therefore exist from the FIRST LANDING COMMIT (commit 2, the first commit that writes rows), and **every landing commit writes its own MEASURED value**,
re-measured as the tip moves (`tools/import_graph.py` plus the test's dotted-prefix classifier) —
"written once at the seal" is RETRACTED. MEASURED consequence for the new partition: the four landing
commits (2-5) rewrite NO importer, and a moved leaf's old `__legacy` address keeps its edge through the
FOR-1547 forwarder, so the measured value stays **45** through commit 5; the six importer rewrites in
commit 6 take it to **38 [predicted]**, and the seal carries that final number. p7 is the precedent:
35 at `2ee5e87`, 45 at `f94b808`, both measured, in the same fragment.

MEASURED CORRECTION to the equality arm's subject:
`tests/test_legacy_coupling.py::load_committed_budget` reads the **highest-ordered fragment present**
per the sprint-manifest `sequence`, not the SUM of fragments (`gen_shims.highest_ordered_fragment`,
the same lookup `--print-budget --sequence-file` performs). In this lane's worktree p8 is the highest
present, so equality binds to p8's own measured 38. At the ZIP the sequence is
`… p8-capture, p9-run-lower, p11-analyze-review-file`, so the highest-ordered fragment present is
**p11's**, measured in a worktree that contains neither p8's nor p9's edits. That is CLOSED by ruling
**ZIP-REMEASURE** (plan rulings table, conductor 2026-09-23): the zip's `zipRestackHook` re-runs the
equality arm at EVERY restacked tip and re-pins the highest-ordered fragment's number there with a
dated `note`, then re-certifies. Nothing about it is open, and nothing about it is the lane's: the
lane measures its own tip and never tunes its number to a wave-level total.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- The 52 moved modules exist at the destination addresses enumerated under *Scope*; every shim pair
  regenerated by `tools/gen_shims.py` in one generate-then-`--verify` step, and every shim's
  `_target` is the `ladder.capture.*` address, never `__legacy`. Instruments:
  `tools/gen_shims.py … --verify` (non-zero exit on drift) and `tests/test_gen_shims.py`.
- FOR-1547 forwarders exist at every vacated `__legacy` path whose address is still imported, and
  `--verify` names ZERO stale forwarders. Instrument: `--verify` output plus the `forwarders` key of
  the emitted manifest, quoted in the lane report. MEASURED expectation: forwarders are owed at
  `__legacy/d1_{training_step,calibration}_records.py` (importers: `__legacy/d1_training_step.py:21`,
  `__legacy/d1_calibration.py:17`, and `tests/test_d1_training_step_split.py:38,40,43` — `:38` is
  served by the NEW package-form scan) and at `__legacy/evidence/d1_capture_{artifacts,bundle}.py`
  (importer: the rung-owned `ladder/gate/ladder_predecessor.py:14,20`). Those four are PERMANENT
  through the lane tip. TRANSIENT forwarders (second design gate, MEASURED at `e01867a`) exist from
  the commit that moves each module until commit 6: `__legacy/evidence/resource_meter_{models,runtime}.py`
  from commit 3, `__legacy/sinks.py` and `__legacy/data/fineweb.py` from commit 4 (`ast.walk` sees the
  function-local imports at `composite.py:123` and `mlflow.py:144,262`; `source_cli.py:10,99` imports
  `__legacy.data.fineweb`); `delete_stale_forwarders` retires them at commit 6 only by reading the
  PREVIOUS `.emitted.json`. Therefore: run the canonical generate then `--verify` at the UNTOUCHED
  parent tip first and assert `git status --porcelain` is empty (`.emitted.json` is gitignored), then
  generate + `--verify` at EVERY landing commit, in order, in the same worktree; never delete
  `.emitted.json` and never `git clean -x` mid-lane; quote the manifest's `forwarders` key at each
  landing commit. No forwarder is owed for `__legacy/data/build.py:15-17` (RETRACTION 8).
- **P8-FRAG migration clean.** The two record-half rows LEAVE
  `.claude/workflows/landed-set/p5-kernel.json` and ENTER
  `.claude/workflows/landed-set/p8-capture.json` as `landed: true`, in the commit that moves them: a
  module may appear in exactly ONE fragment (`gen_shims.merge_fragments` disjointness is fail-loud),
  so leaving the `landed: false` rows behind is a merge error. Instrument:
  `tools/gen_shims.py … --preflight` CLEAN after the migration (and `--verify` after generate), with
  its output in the lane report.
- `git diff -M --stat` shows every moved module as a pure rename plus import-line rewrites, the two
  D1 observers INCLUDED (ruling P8-OBSERVERS-RENAME-ONLY; RETRACTION 10 strikes this bullet's earlier
  clause "and, for the two D1 observers, the R1 re-basing"). For every vacated path a forwarder
  REOCCUPIES in the same commit, the lane MEASURES the list and shows the move with a rename-capable
  form (`git diff -B -M` plus an explicit per-path list), never "pure rename" from `-M --stat` alone
  (second design gate's constraint, `[NOT MEASURED by the gate]`: git reads a path that still exists
  at the tip as a modification, so the moved file may read as an ADD). No admissibility predicate, no
  digest literal, no schema constant, no ledger arithmetic changes bytes. Instrument: the
  rename-detected diff in the lane report.
- `ladder/capture/{custody,records,reader,resource,ledger}/__init__.py` are docstring-only, no
  re-exports, and are exactly five NEW files (second design gate, MEASURED at `e01867a`: none exists,
  and `custody/` itself is absent); no EXISTING `__init__.py` is modified anywhere (`ladder/capture/`,
  `adapters/`, `ports/`, `__legacy/**`, the old-path facades). Instrument: **EXHIBIT OR RETRACT** — there is no mechanical arm for "a facade grew a
  re-export" (`lint-imports` only fires if the re-export crosses a layer, and MEASURED every
  candidate re-export here would be intra-`ladder`). The lane either exhibits a RED arm it wrote, or
  records the row as a diff-review obligation with that measurement, per house rule 4.
- **`test_module_identity.py` stays green at EVERY commit.** `tests/test_module_identity.py::_canonical_sentinel_key`
  reverse-maps a landed module's new name through the merged fragments' `target` field (`:101-133`).
  MEASURED: TWO p8 targets carry frozen sentinels — `entropic_substrate.evidence.d1_capture_aggregate`
  (`_AGGREGATE_TOKEN`) and `entropic_substrate.evidence.d1_capture_bundle` (`_BUNDLE_TOKEN`), both in
  `_FROZEN_SENTINEL_MAP` (`:196-215`), both landing in commit 4 (the SCC commit {custody + records + reader}). So the commit that moves them MUST
  carry their fragment rows, or the reverse-lookup fails to resolve and the 18-sentinel arm
  (`:222 assert len(found) == _FROZEN_SENTINEL_MODULE_COUNT`) goes RED. See *FRAGMENT TIMING*.
- **`telemetry.py` still RAISES rather than returning an empty value on inadmissible telemetry.**
  Instrument: `tests/test_telemetry.py` (`test_missing_file_is_a_gap`, `test_no_eval_record_is_a_gap`,
  `test_kappa_absent_from_the_final_eval_is_a_gap`, `test_truncated_jsonl_line_is_a_gap` — all
  `pytest.raises(TelemetryGap, …)`), green unchanged through the shim.
- **`scripts/run_ladder.sh` keeps parsing.** RETRACTION 4 (house rule 4): v1's bullet said "two
  `float()`-consuming `-m` calls" with no test named. MEASURED at `79d8228`: `run_ladder.sh` has ONE
  `float()` site (`:125`, a `"$PY" -c` one-liner) and ONE `-m "$ENTRYPOINT_TELEMETRY" final-eval`
  site (`:72`, invoked once per arm) whose stdout that `float()` consumes, plus a `kappa-report` call
  at `:170`. The entrypoint name is parameterized from ONE source, `scripts/entrypoints.env`
  (`ENTRYPOINT_TELEMETRY=entropic_substrate.telemetry`), read by `run_ladder.sh:33` and the
  `justfile:15,18`. Instruments:
  `tests/test_cli_surface.py::TestMachineConsumedStdoutSitesArePinnedNotRouted::test_telemetry_final_eval_subcommand_exists_and_is_raw_print`
  (resolves the module by `importlib` precisely "to survive p8's shim rebind") and
  `test_path_site_contains_its_frozen_literals_verbatim[scripts/entrypoints.env]`.
- **`scripts/entrypoints.env` is NOT edited, and E-4 has an EMPTY SUBJECT here.** MEASURED: the
  generated entrypoint shim is the 15-line / 5-statement form that `runpy`-executes the target under
  `__main__` (`src/entropic_substrate/telemetry.py:1-15`), so `python -m entropic_substrate.telemetry`
  keeps working after the move with the shim's runpy target regenerated to the landed address. The
  frozen literal `entropic_substrate.telemetry` inside `scripts/entrypoints.env` therefore stays and
  `tests/test_cli_surface.py` stays green unedited.
- **Review-packet reader byte-identical (ruling P8-PACKET-SCRIPT).** The SCC commit's rewrite of the
  moved `telemetry.py` imports defeats `build_reader`'s strip filter: its prefix tuple
  (`scripts/assemble_review_packet.py:634-637`) keys on `entropic_substrate.d1_` and
  `entropic_substrate.evidence.d1_`, and its logging `replace` (`:734-737`) matches only the
  `entropic_substrate.logging` spelling. MEASURED by the second design gate at `e01867a` (in memory,
  no tree write): the unamended filter leaks **4** import lines (`telemetry.py:44,45,57,60` at their
  new addresses) into the blinded reader, and `tests/test_review_packet.py` (unit-marked) goes RED at
  pre-commit. The SCC commit therefore edits the script ADDITIVELY: both p0-frozen literals stay
  (pinned by `tests/test_cli_surface.py:91-95`), the tuple APPENDS
  `entropic_substrate.ladder.capture.reader.d1_` and `entropic_substrate.ladder.capture.records.d1_`,
  and the logging substitution is EXTENDED to the `entropic_substrate.kernel.runtime.logging`
  spelling while the old `replace` stays (`tests/test_review_packet.py:252-264` still feeds the OLD
  spelling at `:255`). Instruments: the reader emitted from the moved `telemetry.py` is BYTE-IDENTICAL
  to the parent tip's (RETRACTION 2026-09-24, house rule 4: the earlier literal `11069 bytes` was a CHARACTER count from an in-memory measurement; independently rebuilt at `289d9f2` and `5b6f43c` the raw `build_reader` output is 11093 bytes, sha256 `3fe1b788…`, identical on both sides, and the CLI `main()` pipeline emits 11530 bytes, sha256 `46ae424a…`, identical; byte-identity is the load-bearing arm and holds. `c2f64a0` touches neither
  `telemetry.py` nor the script), compared and quoted in the lane report; and
  `tests/test_review_packet.py` + `tests/test_cli_surface.py` green UNEDITED.
- **The data-facade RED case is retired, not re-homed (ruling P8-SHIM-LAYER-RETIRE).**
  `tests/test_shim_layer.py::test_restoring_one_facade_original_bytes_breaks_cold_import_RED_case`
  (`:228-270`, hard-coded `facade = "data"`, `child = "batching"`) plants a cycle that is reachable
  only while the `data/{build,enwik8,fineweb,synthetic}.py` shims import quarantined leaves. MEASURED
  by the second design gate at `e01867a` on an isolated overlay: control arm rc=1 with "partially
  initialized module 'entropic_substrate.data.build'"; post-move emulation rc=0 with
  `COLD_IMPORT_IDENTICAL`. So the hazard is unreachable once `__legacy/data` holds no leaves, and the
  test goes RED at the SCC commit. RE-MEASURED at `c2f64a0` by this amendment: the RED case still
  passes, so the DATA-BUILD-RUNG rewrite left its RED arm reachable before the move. The SCC commit
  deletes that ONE test function and leaves a dated house-rule-4 retraction comment at its site citing
  both arms. It is NOT re-homed onto `predictive_state`, the only other `FACADE_CHILDREN` facade with
  quarantined leaves (8, all mapped to `ladder.run`): it would re-break when that lane moves them.
  Nothing else in the file changes; the fixture `committed_legacy_facades_untouched` (`:149`), whose
  only consumer is this test (MEASURED at `c2f64a0`, `:229`), is reported as orphaned for the p13
  sweep, not deleted. Instrument: `git diff -U0 -- tests/test_shim_layer.py` shows only that
  retirement, and the rest of the file is green.
- **The D1 observers move RENAME-ONLY (ruling P8-OBSERVERS-RENAME-ONLY, FOR-1552).** Commit 5 moves
  `__legacy/d1_{observability,tracking}.py` to `ladder/capture/adapters/` as renames plus import-line
  rewrites only. `D1MlflowObserver`'s constructor `(*, client, run_id, extractor=None)`, its
  `isinstance(client, D1MlflowMetricClient)` guard (`__legacy/d1_observability.py:81-82`), its
  `log_metric` projection (`:97`) and its raise-through are byte-unchanged, and `D1MlflowMetricClient`
  stays PUBLIC. Instruments: `tests/test_d1_observability.py` green UNEDITED through the regenerated
  shims, and the rename-detected diff of the two files shows import lines only.
- **Cold import at the NEW address, fresh interpreter per moved leaf**
  (`tests/test_capture_cold_import.py`, NEW, RED-first; the subprocess-plus-stdout-marker instrument
  of `tests/test_shim_layer.py` and p6's `tests/test_core_cold_import.py`, never exit code alone):
  each of the 52 destination addresses imports cleanly with NOTHING warming the chain first. The lane
  MEASURES the pre-move failure count for the 52 legacy addresses and records it honestly, including
  a measured zero (p7 precedent: a zero is reported, never dressed as a closed hazard). Subjects
  derive from `p8-capture.json`'s own landed rows, or from the five p8 sub-package prefixes plus the
  two `adapters.d1_*` targets, NEVER from a `ladder.capture.*` prefix (second design gate, MEASURED:
  `p3b-observability-rail.json` and `p4-h-tracer.json` already land one `ladder.capture.*` target
  each). Once `p8-capture.json` exists an EMPTY collected set FAILS rather than skips, and the
  collected count equals that fragment's landed-row count: **52** at the tip, shown in the lane
  report (precedent: the non-empty guard at `tests/test_register_gate_cold_import.py:120-124`).
- `tests/test_no_old_prefix.py` green: ZERO landed→old-prefix edges, i.e. no moved module keeps an
  `entropic_substrate.<pkg>.<leaf>` import.
- `legacy_edge_budget` in `.claude/workflows/landed-set/p8-capture.json` equals the MEASURED
  landed→`__legacy` edge count at the lane tip, measured with `tools/import_graph.py` and never
  typed. Instrument:
  `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget`
  (EQUALITY arm). `[predicted] 38`; direction DECREASING is an observation, not a gate.
- **R-PIN discharged:** `_CHECKPOINT_REGIME_TABLE` re-pinned with `tools/regime_rows.py` in the SAME
  commit as the four reader-member rewrites; `_SOURCE_REGIME_TABLE` re-measured UNCHANGED as the
  control; `docs/identity/read-old-table.md` gains its dated p8 row. Instruments:
  `tests/test_regime_rows.py`, `tests/test_identity_epoch.py`, tool output in the lane report.
- **Q1 carve-out 1 of 2 is discharged exactly** (carve-out 2 of 2, the `tests/test_shim_layer.py`
  RED-case retirement, is its own bullet above): `tests/test_d1_training_step_split.py` updated at lines
  `52`, `53`, `127`, `128` ONLY (the two literal `entropic_substrate.__legacy.d1_*_records` module
  pins and the two `src/entropic_substrate/__legacy/d1_*_records.py` path pins). `:38` (the
  package-form record-half import — a **p8** line, NOT a torch-half line) and `:40,43` (record-half
  dotted `__legacy` imports) keep resolving through the forwarders and stay UNTOUCHED; `:36-37` (the
  torch halves `d1_calibration`, `d1_training_step`) are p9's and stay. Instrument:
  `git diff -U0 -- tests/test_d1_training_step_split.py` shows exactly those four lines, and the file
  is green. Note for the design gate (not an edit): `:92-96`'s test NAME and docstring
  (`test_both_record_halves_are_importable_from_their_own_module_before_any_split_move`, "Both halves
  stay under `__legacy/` this phase") become historical prose once p8 lands; the carve-out is
  unchanged and the rename rides p13's sweep.
- **The p3b residue-relocation set is discharged (`composite.py:30-50`, architect constraint
  2026-09-21).** `build_sink_for_run`'s and `adapters/mlflow.py`'s function-scope imports reach into
  `__legacy.sinks` for SIX names: `_DEFAULT_EXPERIMENT`, `MLFLOW_TRACKING_ENV`,
  `MLFLOW_EXPERIMENT_ENV`, `mlflow_enabled` (in `composite.py`), `MLFLOW_SYSTEM_METRICS_ENV`
  (`mlflow.py:144`) and `MlflowStartupError` (`mlflow.py:262`, raised at `:292`, `:328`). Re-pointing
  them is a BODY edit, not a door violation: the run-descriptor door freezes SIGNATURES and
  raise/abort semantics only. **`MlflowStartupError` must be RE-EXPORTED as the SAME class object**,
  never redefined — out-of-envelope callers catch it by the old prefix (`tests/test_sinks.py`,
  `tests/test_caddy_auth.py`, `__legacy/cli.py`). Instruments: those two tests green unedited, plus an
  `is`-identity assertion in the lane report.
- **`lint-imports` forbidden contracts stay green, MEASURED premise.** `.importlinter` has exactly two
  forbidden contracts. `core-forbids-external-io` (`:42`) sources only `entropic_substrate.core`, and
  MEASURED the only p8 source touching a forbidden library is `__legacy/data/fineweb.py` (lazy
  `from datasets import load_dataset` at `:46`, `:65`), which lands in `ladder.capture.custody`, not
  `core` — a near-miss, not a trip. `mlflow-only-from-adapter` (`:70`) sources all five landed roots
  and forbids `mlflow`, with ONE `ignore_imports` line for
  `ladder.capture.adapters.mlflow -> mlflow`; MEASURED (`grep -n "import mlflow\|from mlflow"` over
  the 52 sources) ZERO p8 source imports `mlflow` directly — `sinks.py:67` imports `MlflowSink` from
  the adapter, and the indirect chain terminates on the ignored edge. `.importlinter` is NOT edited
  (p0/p13 only) and no contract is invented.
- **Conformance arm 3 needs no fragment member, MEASURED.**
  `grep -n "JsonlSink(\|MlflowSink(\|CompositeSink("` over the 52 sources returns ZERO hits, so the
  lane creates no `MetricSink` construction site and `sink_construction_modules` stays p3b's single
  member (`entropic_substrate.ladder.capture.adapters.composite`). `sinks.py:243` returns a
  `MetricSink` by delegation; it does not construct one.
- `lint-imports` green: ZERO inversions among the landed layers
  `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13,
  so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them.
  MEASURED premise: every rewritten import resolves to `kernel.*`, `core.*` or `ladder.*`, so no
  inversion is possible by construction.
- `tests/test_observability_conformance.py` green, all three arms. Its print-ban arm already covers
  `ladder.capture.*` by construction: `observability_conformance_config.LANDED_LAYER_PREFIXES`
  contains `entropic_substrate.ladder` and the test `rglob`s that root, so an arriving module is in
  scope the moment it lands. MEASURED consequence: the one raw `print(` the lane carries
  (`__legacy/telemetry.py:506`) needs p8's fragment to **ADD** the landed dotted name to
  `print_allowlist` in the same commit that moves the file, or arm 1 goes RED.
- **NO `remap_check` bullet. RETRACTION 5 (house rule 4), ruling P9-REMAP (FOR-1549, wave-wide):**
  v1's "remap_check at capture sub-package granularity reports no NEW cycle versus target" is
  retracted, not overwritten — it named no baseline and no target number, AND the tool cannot run on
  the post-quarantine tree at all. MEASURED at `79d8228`:
  `python3 tools/import_graph.py src/entropic_substrate <dir> --package entropic_substrate` then
  `python3 tools/remap_check.py <dir>/edges.json tools/mapping.json` exits **1** with
  `FATAL: 328 UNMAPPED module(s) referenced by edges.json but absent from the mapping` — every
  `entropic_substrate.__legacy.*` address is unmapped, so the tool aborts BEFORE any cycle count. p9
  retracted the identical bullet; p8 retracts it the same way. No cycle claim is made by this lane.
- `just check` and `just pc-ci` (exit gate via `uv run python scripts/run_exit_gate.py`) green on the
  lane tip. `uv sync --all-groups` runs FIRST in a fresh worktree (HR-7).
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the
  lane report.
- The actual diff's file set is a SUBSET of *Allowed files*, and disjoint from p9's and p11's slices.
  Undeclared overlap is a HALT at the zip, not a note.
- Verdict CERTIFIED (tip SHA + exit-gate JSON). The lane does NOT publish. The exit gate re-runs at
  the restacked tip at the zip: a restacked tree is a new tree, and this lane's pre-zip certificate
  speaks for a tree that no longer exists.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N−1 and raises
  DECISION-NEEDED naming the lane.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for
correctness is a `DECISION`, not an edit.

- Does NOT re-own `ladder/capture/ports/**` or `ladder/capture/adapters/{jsonl,composite,mlflow}.py`
  beyond the measured import-line, docstring-cite and residue-name rewrites.
- Does NOT re-own `ladder/capture/capture.py`; p4 landed it. RETRACTION 11 (house rule 4): this row
  read "Does NOT re-own `ladder/capture/custody/h_source.py` or `ladder/capture/capture.py`; p4 landed
  them". The `h_source.py` half is retracted, not overwritten: MEASURED at `e01867a` by the second
  design gate and re-measured at `c2f64a0`, `grep -rln h_source src tools` is empty and
  `ladder/capture/custody/` does not exist (`ls src/entropic_substrate/ladder/capture/` lists only
  `__init__.py`, `adapters/`, `capture.py`, `ports/` and a `__pycache__/`). p8 CREATES `custody/`.
- Does NOT re-base the D1 observers on `MetricSink`, does NOT retire `D1MlflowMetricClient`, and does
  NOT add a strict raising sink: that unification and its test carve-out are FOR-1552's (ruling
  P8-OBSERVERS-RENAME-ONLY).
- Does NOT add a second sink implementation. The three adapters are complete.
- Does NOT perform either symbol split. p5 did that; p8 MOVES the record halves only.
- Does NOT touch the torch halves `__legacy/d1_{training_step,calibration}.py`; p9 moves them.
- Does NOT touch `ladder/gate/ladder_predecessor.py` (rung-owned) or `ladder/gate/d1_verdict.py`
  (p11's).
- Does NOT relax `telemetry.py`'s raise-on-inadmissible behaviour, and does NOT route the pinned raw
  `print(` at `__legacy/telemetry.py:506` through the logger.
- Does NOT remove p3b's `entropic_substrate.telemetry` `print_allowlist` key (see M2 note under
  *Observability conformance*), and does NOT re-key `conforming_entrypoints` before p13.
- Does NOT edit the sprint manifest, `tools/mapping.json`, `.importlinter`, `scripts/entrypoints.env`,
  briefs, plan docs, engine scripts or CI. Does NOT edit any `scripts/` file except the ADDITIVE
  `scripts/assemble_review_packet.py` edit (P8-PACKET-SCRIPT). Does NOT edit any EXISTING test except
  the TWO Q1 carve-outs: the four split-test lines, and the one `tests/test_shim_layer.py` RED-case
  retirement (P8-SHIM-LAYER-RETIRE, RETRACTION 9). Does NOT edit another phase's fragment except the
  P8-FRAG migration.
- Does NOT publish, rebase, restack, or run the live gate.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this
envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the
verification you RAN attached (command + observed output). RETRACTED 2026-09-18 (Q1 ruled (b)): this
section previously granted "the capture-side tests, for import-path rewrite only" as a prose test
grant. Struck — a prose grant cannot discharge the subset-of-Allowed-files check, and shims keep
existing tests green under rewrite-on-arrival.

- the 52 legacy sources enumerated under *Scope* (moved with `git mv`, import lines rewritten once on
  arrival per the *Rewrite-on-arrival* tables)
- `src/entropic_substrate/ladder/capture/custody/**`, `.../records/**`, `.../reader/**`,
  `.../resource/**`, `.../ledger/**` (the moved modules plus a docstring-only `__init__.py` each)
- `src/entropic_substrate/ladder/capture/adapters/d1_observability.py` and
  `.../adapters/d1_tracking.py` (the rename-only destinations, ruling P8-OBSERVERS-RENAME-ONLY; they
  were listed as "the R1 re-basing destinations" until RETRACTION 10)
- `src/entropic_substrate/ladder/capture/adapters/composite.py` and `.../adapters/mlflow.py` —
  IMPORT-LINE, DOCSTRING-CITE and six-residue-name rewrites only
- the four other landed importers, one or two import lines each plus any docstring naming the old
  address, in the same commit (R-PIN): `src/entropic_substrate/composition/source_cli.py`,
  `src/entropic_substrate/ladder/run/execution.py`,
  `src/entropic_substrate/ladder/run/generation.py`,
  `src/entropic_substrate/ladder/run/span_diagnostics.py`
- the 52 typed shim pairs at the old paths (`.py` + `.pyi`) and the FOR-1547 forwarder pairs at the
  vacated `__legacy` paths — all regenerated by `tools/gen_shims.py`, never hand-edited
- `tests/test_capture_cold_import.py` (new, RED-first: fresh interpreter per moved leaf at its NEW
  address)
- `tests/test_d1_training_step_split.py` — **Q1 carve-out 1 of 2**, lines `52`, `53`, `127`, `128`
  ONLY. MEASURED at `79d8228`: `:52-53` are compared against `__module__` (`:69`, `:85`, `:97-101`)
  and `:127-128` are read from disk for the AST no-torch check, so both go RED on the move and neither
  is repairable by a shim or a forwarder.
- `tests/test_shim_layer.py` — **Q1 carve-out 2 of 2, RULING P8-SHIM-LAYER-RETIRE (conductor
  2026-09-23):** in the SCC commit (commit 4), retire
  `test_restoring_one_facade_original_bytes_breaks_cold_import_RED_case` (facade `data`) with its
  dated house-rule-4 retraction comment citing the measurement (control rc=1 "partially initialized
  module 'entropic_substrate.data.build'"; post-move rc=0 `COLD_IMPORT_IDENTICAL`: the hazard is
  unreachable once `__legacy/data` holds no leaves). NOT re-homed onto `predictive_state`. Nothing
  else in the file. RETRACTION 9 (house rule 4): this brief listed `tests/test_shim_layer.py` under
  *Deliberately EXCLUDED* ("A diff touching any of them is a `DECISION`") and its *Non-goals* allowed
  test edits to "the four lines of the Q1 carve-out" only, i.e. it claimed the file stays untouched.
  MEASURED false for the SCC commit (the RED case goes RED on the move, see ACCEPTANCE), so both are
  retracted, not overwritten.
- `scripts/assemble_review_packet.py` — **RULING P8-PACKET-SCRIPT (conductor 2026-09-23), ADDITIVE
  only, riding the SCC commit (commit 4):** keep both p0-frozen literals `entropic_substrate.d1_` and
  `entropic_substrate.evidence.d1_` (pinned by `tests/test_cli_surface.py:91-95`); APPEND
  `entropic_substrate.ladder.capture.reader.d1_` and `entropic_substrate.ladder.capture.records.d1_`
  to `_PACKET_READER_EXCLUDED_IMPORT_PREFIXES` (`:634-637`); EXTEND the logging substitution
  (`:734-737`) to the `entropic_substrate.kernel.runtime.logging` spelling, keeping the old one. No
  other line.
- `.claude/workflows/landed-set/p8-capture.json` — THIS lane's disjoint fragment: the 52 rows
  `landed: true` with their verbatim `ladder.capture.*` targets (added PER LANDING COMMIT, see
  *FRAGMENT TIMING*), the MEASURED `legacy_edge_budget`, the ADDED `print_allowlist` entry, and a
  `note` recording the ELEVEN numbered retractions.
- `.claude/workflows/landed-set/p5-kernel.json` — **RULING P8-FRAG (conductor 2026-09-23), that ONE
  edit only:** REMOVE the two record-half rows (`entropic_substrate.d1_training_step_records`,
  `entropic_substrate.d1_calibration_records`, both `landed: false` today) in the commit that moves
  them, because a module may appear in exactly one fragment. The generator's "serial phases only"
  rider exists because two PARALLEL lanes editing one fragment would collide; p8 is the only w2 lane
  editing `p5-kernel.json`, so file-disjointness holds. No other row, no budget change
  (`legacy_edge_budget` stays 56). Applied as a TEXTUAL edit, never a `json.dump` re-serialization:
  `git diff -U0` shows only the two row removals plus the preceding row's trailing-comma line (second
  design gate's constraint).
- ZIP-OWNED REGENERABLE (in the envelope, marked so; the zip regenerates them on the integrated tip in
  ONE rung and never hand-merges them): `tests/test_identity_epoch.py`, `tools/regime_rows.py`,
  `tests/test_regime_rows.py`, `docs/identity/read-old-table.md`, `.secrets.baseline`. They are
  excluded from `lane_slices.py`'s overlap check by `--regenerable` precisely because p8 and p9 both
  re-pin the same checkpoint rows.
- generator outputs (D-B precedent, p5b): any `.pyi` stub whose target this lane edits, listed in the
  lane report from `git status --porcelain` after generate + `--verify`. An import-line rewrite does
  not change a bound name, so each stub changes ONLY in the module path of its
  `from … import X as X` lines.
- `.claude/workflows/landed-set/.emitted.json` is UNTRACKED derived state (FOR-1539); never `git add`
  it, never treat it as a companion of an allowed file.
- `.secrets.baseline` (STANDING RULE, conductor 2026-09-19: a mechanical companion of allowed
  content. detect-secrets regenerates it whenever an allowed file's digest literals move, so its diff
  may contain ONLY line renumbering, removal of orphaned entries, the `generated_at` stamp, and new
  `Hex High Entropy String` entries whose flagged lines are sha256 / git-SHA literals inside this
  lane's own Allowed files; the lane verifies that in its report. New digest literals in TEST files
  carry an inline `# pragma: allowlist secret` instead of a baseline entry. Never a credential.)

**Deliberately EXCLUDED:** `.claude/workflows/sprint-manifest.json` (conductor's, amended in the same
commit as this brief), `src/entropic_substrate/ladder/gate/ladder_predecessor.py` (RUNG-OWNED),
`src/entropic_substrate/ladder/gate/d1_verdict.py` (p11's),
`__legacy/d1_{training_step,calibration}.py` (p9's),
`src/entropic_substrate/{evidence,data}/__init__.py` (live old-path facades, stay verbatim),
`src/entropic_substrate/__legacy/data/__init__.py` (already retracted),
`src/entropic_substrate/__legacy/evidence/__init__.py` (ruling FACADE-EMPTY: 11 leaves survive p8, so
the retraction is p12's), `tools/mapping.json`,
`.importlinter`, `scripts/entrypoints.env`, `tests/observability_conformance_config.py` (p3b's; it
unions every fragment present, so the lane changes its OWN fragment instead),
`tests/test_cli_surface.py`, `tests/test_telemetry.py`, `tests/test_d1_observability.py`
(unedited under P8-OBSERVERS-RENAME-ONLY), `tests/test_review_packet.py`,
`tests/test_module_identity.py`, `tests/test_legacy_data_build_cold_import.py` (the DATA-BUILD-RUNG
test; resolves its address live, so it follows the move unedited: see the RULED D-4 note under *Rewrite-on-arrival*). `tests/test_shim_layer.py` LEFT this list on
2026-09-23 (RETRACTION 9, ruling P8-SHIM-LAYER-RETIRE). A diff touching any of them is a
`DECISION`, not an edit.

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                        | status                                                                                           | ruling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-WAVES r3                | OPERATOR 2026-09-23 (plan line 843)                                                              | w2 is PARALLEL, p8 ∥ p9 ∥ p11, launched as NESTED worktrees off `mythwave/p7b-freeze-and-verify` @ `c2f64a0` (was `b7af1aa` until the second epoch move; parent = that branch, NOT p6; the ruling as first written named p7, superseded by the epoch p7 → p7b row). `publishMode:'defer'`, lane ends CERTIFIED, `liveRequired:false`, the zip publishes ONCE. Zip order `p8 → p9 → p11`. Afterwards ONE serial conductor rung rewrites the sibling-target imports and retires the forwarders.                                                                                                                                                                                                                                                                                                                                                                                                            |
| manifest amendment (B1)   | CONDUCTOR 2026-09-23                                                                             | The w2 sprint-manifest amendment (parent → `mythwave/p7b-freeze-and-verify`, now @ `c2f64a0`, was `b7af1aa`; touch sets REGENERATED from the slice JSON, phantom package-key paths dropped, notes; round 4 adds `scripts/assemble_review_packet.py` and `tests/test_shim_layer.py`, each scoped to its ruling) is a CONDUCTOR PRECONDITION and lands in the SAME COMMIT as this brief. **The lane never edits the manifest**; a manifest defect it notices is a `DECISION` to report.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ZIP-REMEASURE             | CONDUCTOR 2026-09-23 (plan rulings table)                                                        | The budget equality arm binds to the HIGHEST-ORDERED fragment present. The zip's `zipRestackHook` re-runs that arm at EVERY restacked tip and re-pins the highest-ordered fragment's number there with a dated `note`, then re-certifies. This closes the former D-P8-BUDGET question; no lane tunes its number to a wave-level total.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| W2-SCC                    | CONDUCTOR 2026-09-23 (w2 design gate HALT)                                                       | The six-landing-commit partition is INFEASIBLE: MEASURED 14 forward-reference intra-set lines and a CYCLIC sub-package graph (`records ↔ custody` 2+1, `records ↔ reader` 5+4). Landing commits partition by STRONGLY-CONNECTED GROUP in topological order: `{ledger}` and `{resource}` (zero outgoing, either order), then `{custody + records + reader}` as ONE 37-leaf commit, then `{adapters}`. A topological check runs BEFORE commit 1 with its command and output in the lane report. Transient `__legacy` addresses inside the phase (option B) are NOT permitted.                                                                                                                                                                                                                                                                                                                              |
| BUDGET-PER-COMMIT         | CONDUCTOR 2026-09-23, WAVE-WIDE                                                                  | `load_committed_budget` does a bare `int(top["legacy_edge_budget"])` on the highest-ordered fragment, and `p8-capture.json` becomes that fragment at commit 2 (the first landing commit; commit 1 lands only the RED-first test), so the key must EXIST from commit 2 and every landing commit writes its own MEASURED value, re-measured as the tip moves. "Written once at the seal" is RETRACTED. The seal carries the final 38 `[predicted]`. Precedent: p7 wrote 35 at `2ee5e87` and 45 at `f94b808`.                                                                                                                                                                                                                                                                                                                                                                                               |
| epoch p7 → p7b (FOR-1550) | CONDUCTOR 2026-09-23                                                                             | The parent tip was `mythwave/p7b-freeze-and-verify` @ `b7af1aa`; since the second epoch move it is `c2f64a0` (`e01867a` + DATA-BUILD-RUNG), where every `79d8228` measurement holds except build.py's own-import rows (RETRACTION 8) and `tests/test_legacy_coupling.py` re-checked the 45 green. p7b re-pinned the checkpoint regime rows and froze the reader label and module set (FOR-1550, `40e72ac`), so every CONTROL COMMAND naming `mythwave/p7-register-gate` is retracted (RETRACTION 7) and names the parent tip instead. RE-MEASURED at the new epoch: landed→`__legacy` edges are 45, unchanged from `79d8228`.                                                                                                                                                                                                                                                                            |
| FACADE-EMPTY              | CONDUCTOR 2026-09-23, WAVE-WIDE                                                                  | A moved package facade `__legacy/<pkg>/__init__.py` is retracted to the p3 one-liner ONLY in the commit after which the package DIRECTORY holds no leaf module (`ls src/entropic_substrate/__legacy/<pkg>/*.py` → only `__init__.py`). MEASURED: `evidence/` holds 48 leaves; p8 moves 37, p11 moves 10, and `evidence/ladder_campaign.py` survives the wave, so the facade is **p12's** to retire. p8 does not touch it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| P9-REMAP (FOR-1549)       | CONDUCTOR 2026-09-23, WAVE-WIDE                                                                  | `tools/remap_check.py` cannot run on the post-quarantine tree: MEASURED exit 1, `FATAL: 328 UNMAPPED module(s)`, before any cycle count. Every lane's remap bullet is RETRACTED (p9 first, p8 here). No lane claims a cycle count; re-enabling the oracle is FOR-1549's own work, not a lane's.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| DATA-BUILD-RUNG           | CONDUCTOR 2026-09-23 (p9 gate, second w2 launch; plan line 856), LANDED `c2f64a0`                | `__legacy/data/build.py` imported `data.batching` by its OLD name, so a landed consumer entering `data.build` at its explicit `__legacy` address hit a partially-initialised-module ImportError (MEASURED by the p9 gate: 41 of 101 cold probes in a simulated p9 move). One conductor rung on the shared parent rewrote build.py's five first-party import lines to explicit addresses (`kernel.runtime.config`, `core.corpus.batching`, `__legacy.data.{enwik8,fineweb,synthetic}`), RED-first (`tests/test_legacy_data_build_cold_import.py`). It AMENDS rewrite-once for this ONE quarantined file: p8 rewrites these lines once more on arrival (MEASURED: `:15-17` change, `:18-19` already read their destination). Bucket consequence: RETRACTION 8. The rung's test resolves its address live (D-4 RULED, see *Rewrite-on-arrival*), follows the move unedited, and stays out of Allowed files. |
| P8-PACKET-SCRIPT          | CONDUCTOR 2026-09-23 (p8 gate-2 D-1; plan line 857)                                              | `scripts/assemble_review_packet.py`'s blinded-reader strip filter keys on the p0-frozen prefixes, so p8's rewrite of the moved `telemetry.py` imports leaks four import lines and `tests/test_review_packet.py` goes RED. p8 edits the script ADDITIVELY in the SCC commit: both frozen literals stay (pinned by `tests/test_cli_surface.py:91-95`), the `ladder.capture.reader.d1_` and `ladder.capture.records.d1_` prefixes are appended, and the logging substitution is extended to the `kernel.runtime.logging` spelling. Acceptance: the emitted reader is byte-identical to the parent tip's (size literal retracted 2026-09-24, see ACCEPTANCE: 11093 B raw / 11530 B via CLI, identical on both sides); `tests/test_review_packet.py` and `tests/test_cli_surface.py` green unedited. Manifest touch set amended.                                                                              |
| P8-SHIM-LAYER-RETIRE      | CONDUCTOR 2026-09-23 (p8 gate-2 D-2; plan line 858)                                              | `tests/test_shim_layer.py::test_restoring_one_facade_original_bytes_breaks_cold_import_RED_case` (facade `data`) asserts a planted cycle reachable only while the `data/*` shims import quarantined leaves. MEASURED: control rc=1 "partially initialized module 'entropic_substrate.data.build'"; post-move rc=0 `COLD_IMPORT_IDENTICAL`. Narrow Q1 carve-out: p8 retires that RED case in the SCC commit with an explicit house-rule-4 retraction citing the measurement; NOT re-homed onto `predictive_state` (it would re-break when p9 moves those leaves). RETRACTION 9.                                                                                                                                                                                                                                                                                                                           |
| P8-OBSERVERS-RENAME-ONLY  | CONDUCTOR 2026-09-23 (p8 gate-2 D-3; plan line 859), FOR-1552                                    | The p3b-R1 re-basing of `D1MlflowObserver` on `MetricSink` contradicts Q1 (`tests/test_d1_observability.py` constructs it with a non-`MetricSink` fake at `:202`, `:227`, `:290`) and fail-loud (the only mlflow sink is best-effort). p8 moves both observers to `ladder/capture/adapters/` as renames plus import rewrites only; constructor, `isinstance(client, D1MlflowMetricClient)`, `log_metric` projection and raise-through unchanged; `D1MlflowMetricClient` stays public; `tests/test_d1_observability.py` untouched and green. The unification, a strict raising sink and the test carve-out are FOR-1552's, which gates p13's "D1 observer protocols gone" postcondition. RETRACTION 10.                                                                                                                                                                                                   |
| P8-FRAG                   | CONDUCTOR 2026-09-23                                                                             | The two record-half rows sit in `p5-kernel.json` as `landed: false`. A module may appear in exactly ONE fragment (`merge_fragments` disjointness, fail-loud), so in the commit that moves them the rows LEAVE `p5-kernel.json` and ENTER `p8-capture.json` as `landed: true`. p8 is the only w2 lane editing `p5-kernel.json`, so the generator's "serial phases only" rider is satisfied. `--preflight` clean afterwards.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| fragment-per-commit       | WAVE RULE 2026-09-23 (from the p11 review)                                                       | `tests/test_module_identity.py` reverse-maps a moved module's new name through the fragments' `target` field, so EVERY landing commit carries the fragment rows for the modules IT lands. The fragment GROWS per commit; every landing commit writes its own MEASURED `legacy_edge_budget` (BUDGET-PER-COMMIT), and the last commit's value is the final one. MEASURED: two p8 targets carry frozen sentinels (`evidence.d1_capture_aggregate`, `evidence.d1_capture_bundle`), both in commit 4 (the SCC commit).                                                                                                                                                                                                                                                                                                                                                                                        |
| FOR-1547 forwarders       | RULED 2026-09-23, LANDED `e9e5897`                                                               | `gen_shims` emits an exact-form forwarder plus `.pyi` at every landed row's vacated `__legacy` address while that address is still imported under `src/` or `tests/`. Emission is DERIVED, never declared. Its importer scan now records `<package>.<name>` for package-form `from` imports of a `__legacy` package DIRECTORY (MEASURED 2026-09-23, p9 review B3), which is what serves the split test's `:38`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| sibling-target imports    | CONDUCTOR 2026-09-23, p8 EXCEPTION                                                               | A moved module importing a SIBLING lane's target is rewritten to the explicit `entropic_substrate.__legacy.<x>` address. EXCEPTION for p8, MEASURED: the four p8→p9 lines read record symbols that p8's OWN record halves define, so they rewrite to `ladder.capture.records.*` and `depends_out.p9-run-lower` becomes 0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| rung-owned importer       | CONDUCTOR 2026-09-23 (D-WAVES r3 consequence)                                                    | `ladder/gate/ladder_predecessor.py` carries `__legacy` edges to BOTH p8 (`:14,:20`) and p11 (`:8,:12`) targets, so NEITHER lane rewrites it. Declared `--rung-owned`, served by forwarders inside each lane worktree, rewritten once by the reconciliation rung. NOT in Allowed files.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| BUDGET-MONOTONE retracted | CONDUCTOR 2026-09-23 (landed `795f8e6`)                                                          | p5's strict-decrease arm is RETRACTED; p7's fragment note records "Direction is INCREASING, as predicted". v1's ACCEPTANCE "legacy_edge_budget < p7's" and its guard-teeth "STRICTLY LESS than the previous phase's" are STRUCK. The binding arm is the EQUALITY arm against a MEASURED number, whose subject is the highest-ordered fragment present.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| R-PIN (OWED here)         | CONDUCTOR 2026-09-22 (p5, p5b, p6 precedent)                                                     | Rewriting a `_READER_MODULES` member re-pins the affected reader's regime rows in the SAME commit; label and module SET unchanged. Here 4 of 51 `h-checkpoint-reader-v2` members are rewritten → `_CHECKPOINT_REGIME_TABLE` re-pins; `h-source-reader-v3`'s 7-member table re-measures UNCHANGED as the CONTROL.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| package keys              | tools (`gen_shims`, `lane_slices`) + MEASURED                                                    | `entropic_substrate.evidence` and `entropic_substrate.data` are PACKAGE KEYS, never fragment rows and never lane targets. `__legacy/data/__init__.py` is already the p3 retracted-facade one-liner; `__legacy/evidence/__init__.py` (305 bytes, docstring-only, zero re-exports) is not, and p8 does NOT touch it (ruling FACADE-EMPTY).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Q1 (existing tests)       | RULED 2026-09-18, ONE CARVE-OUT 2026-09-23, SECOND 2026-09-23                                    | No phase before p13 edits an EXISTING test. Carve-out: `tests/test_d1_training_step_split.py` lines `52,53,127,128`, because they pin LITERAL module names and file paths that p8's `git mv` invalidates and no shim or forwarder can repair. Second carve-out (P8-SHIM-LAYER-RETIRE): retire `tests/test_shim_layer.py`'s data-facade RED case in the SCC commit, with a house-rule-4 retraction.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| splits                    | RULED 2026-09-17                                                                                 | `d1_training_step` / `d1_calibration` were SPLIT at p5, serially, under `__legacy/`. p8 moves only the record halves, p9 only the torch halves. Both record halves are `action: split` mapping rows.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| sub-package decomposition | RULED 2026-09-17, COUNTS AMENDED 2026-09-23                                                      | `ladder/capture` decomposes into five sub-packages beside p3b's `ports/` and `adapters/`; do not leave it as one bucket. The counts are MEASURED 19 / 12 / 9 / 8 / 4 = 52, not the ruling's 20 / 10 / 9 / 9 / 4 (retraction 1).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| p3b-R1 D1 observers       | CONDUCTOR 2026-09-21 (wf_12821961-8cd); SUPERSEDED for p8 2026-09-23 by P8-OBSERVERS-RENAME-ONLY | As first ruled: "The D1 observer re-basing deferred from p3b is p8's: see *Observability conformance* for the concrete target modules and which protocol classes retire." RETRACTION 10: the re-basing is no longer p8's. p8 keeps only R1's DESTINATION (`ladder/capture/adapters/`) and moves both observers rename-only; the one-metric-port unification is FOR-1552's.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| O-1                       | RULED 2026-09-17                                                                                 | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| P-1                       | RULED 2026-09-17 (plan PRIORITY AMENDMENT)                                                       | Observability is wired through DURING the refactor on the E-series pattern: rail phase p3b plus the extended rewrite-on-arrival rule. This is the ONE permitted behaviour-change class inside a structural phase.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| E-4                       | RULED 2026-09-17, EMPTY SUBJECT HERE                                                             | Edit the scripts, no forwarders: the dotted entrypoint names live in ONE parameterized source (`scripts/entrypoints.env`) consumed by `run_ladder.sh` and the `justfile`. MEASURED for p8: the generated ENTRYPOINT shim runpy-executes the target under `__main__`, so `entropic_substrate.telemetry` keeps working and NO script edit is owed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| FOR-1539                  | 2026-09-22                                                                                       | `gen_shims`' `.emitted.json` manifest is untracked derived state; no phase commits it and it is never a companion of an allowed file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| rewrite-on-arrival        | RULED 2026-09-17, AMENDED 2026-09-23                                                             | A file's import lines are rewritten exactly once, by the phase that moves its target. This lane never rewrites an importer outside its own slice. AMENDED by DATA-BUILD-RUNG for ONE quarantined file: `__legacy/data/build.py` was rewritten by the conductor rung `c2f64a0` and is touched once more by p8 on arrival (RETRACTION 8).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| shim form                 | RULED 2026-09-17, CORRECTED 2026-09-23                                                           | `import entropic_substrate.<new landed address> as _target` then `sys.modules[__name__] = _target`, as the generator emits it; the 15-line / 5-statement entrypoint variant for `entrypoint: true` rows. v1's `__legacy` `_target` is retracted (retraction 3). A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| skeleton                  | RULED 2026-09-17                                                                                 | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical < composition`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

The second design gate's constraint 1 (the implementer never resolves D-1, D-2 or D-3 on its own
reading and does not attempt commits 4 and 5 until they are ruled) is DISCHARGED by
P8-PACKET-SCRIPT, P8-SHIM-LAYER-RETIRE and P8-OBSERVERS-RENAME-ONLY: commits 4 and 5 are unblocked
exactly within those rulings and no wider. Its other binding constraints (absolute static imports;
cold-import subjects; generate-then-`--verify` from the untouched parent; rename evidence for
reoccupied paths; function-local sinks imports; five new `__init__.py`; the textual P8-FRAG edit; the
lane's own topological check) are applied under *Rewrite-on-arrival*, *ACCEPTANCE*, *Allowed files*,
*Guard teeth by mutation* and *Planned commit branches*.

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes
RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached
it". Each mutation is performed and reverted in-branch; both arms go in the lane report. Every row
names an EXISTING test or tool, or carries the label EXHIBIT OR RETRACT.

| mutation (RED arm)                                                                                                                                                               | must go RED (named instrument)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | GREEN arm                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| make the moved `telemetry.py` RETURN an empty value instead of raising `TelemetryGap` on a missing file                                                                          | `tests/test_telemetry.py::test_missing_file_is_a_gap` (and `test_no_eval_record_is_a_gap`). The only admissibility predicate for `results/*.jsonl`, shared by the CLI preflight and `scripts/run_ladder.sh`.                                                                                                                                                                                                                                                                                                             | both green at the lane tip, unedited, through the regenerated shim                                                                                                                                                                                                                                                                                                                                      |
| swallow an mlflow error inside the moved (rename-only) `D1MlflowObserver` (wrap the client call in `try/except Exception: pass`)                                                 | `tests/test_d1_observability.py::test_mlflow_observer_failure_propagates_after_event_is_durable` (`:223`) — the fail-LOUD measurement-path arm                                                                                                                                                                                                                                                                                                                                                                           | the same test green after the rename-only move, unedited                                                                                                                                                                                                                                                                                                                                                |
| drop ONE appended prefix from `_PACKET_READER_EXCLUDED_IMPORT_PREFIXES`, or the `kernel.runtime.logging` substitution, in `scripts/assemble_review_packet.py` (P8-PACKET-SCRIPT) | `tests/test_review_packet.py::test_default_telemetry_source_resolves_through_the_module` (`:556`, `assert "entropic_substrate" not in reader`, on the reader built from the real module). Leak size per mutation [predicted from `telemetry.py:44,45,57,60`]: `reader.d1_` 1 line, `records.d1_` 2 lines, logging spelling 1 line; the second design gate MEASURED **4** leaked lines with the whole amendment absent                                                                                                    | RETRACTED 2026-09-24 (conductor ruling, house rule 4): MEASURED GREEN under all three mutations because the script's ruff pass (F401) strips the leaked, unused import lines before the test reads the reader; the tooth is unreachable by that mutation class. Load-bearing arm = byte-identity vs the parent tip (11093 B raw / 11530 B CLI, identical). Instrument re-point filed as FOR-1556 (p13). |
| hoist the rewritten `ladder.capture.reader.sinks` import in `adapters/mlflow.py` to module top, above `class MlflowSink` (second design gate's constraint)                       | `tests/test_sinks.py::test_cold_import_of_mlflow_adapter_adapter_first_succeeds` (`:895`) through its returncode arm. Its `:921` assertion (`'entropic_substrate.__legacy.sinks' not in sys.modules`) loses its subject after commit 6, so the returncode arm is the only remaining tooth; that is a p13-sweep note, no edit (Q1)                                                                                                                                                                                        | revert; the imports at `composite.py:123` and `mlflow.py:144,262` stay function-local, test green                                                                                                                                                                                                                                                                                                       |
| REDEFINE `MlflowStartupError` at its new home instead of re-exporting the SAME class object                                                                                      | `tests/test_sinks.py` and `tests/test_caddy_auth.py` catch it by the OLD prefix; a redefinition makes `except MlflowStartupError` silently stop catching (`composite.py:41-50`: a bad `MLFLOW_TRACKING_URI` then burns GPU hours)                                                                                                                                                                                                                                                                                        | both tests green, plus an `is`-identity assertion between the old-prefix and new names                                                                                                                                                                                                                                                                                                                  |
| leave ONE moved leaf importing a sibling by its OLD dotted name (`entropic_substrate.evidence.d1_capture_roster`)                                                                | `tests/test_no_old_prefix.py` (landed→old-prefix edge count ≠ 0)                                                                                                                                                                                                                                                                                                                                                                                                                                                         | zero old-prefix edges at the tip                                                                                                                                                                                                                                                                                                                                                                        |
| inflate the committed `legacy_edge_budget` in `p8-capture.json` by one                                                                                                           | `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget` (EQUALITY, so an inflated budget and an uncounted edge are the same failure)                                                                                                                                                                                                                                                                                                                                               | measured == committed at the tip                                                                                                                                                                                                                                                                                                                                                                        |
| land the custody commit WITHOUT its fragment rows (or leave the record-half rows in `p5-kernel.json`)                                                                            | `tests/test_module_identity.py` (the sentinel reverse-map cannot resolve `evidence.d1_capture_{aggregate,bundle}`; `:222`'s 18-module arm fails) and `gen_shims --preflight` / `merge_fragments` disjointness                                                                                                                                                                                                                                                                                                            | every landing commit carries its rows; `--preflight` clean                                                                                                                                                                                                                                                                                                                                              |
| DELETE one FOR-1547 forwarder while its importer remains (e.g. `__legacy/d1_training_step_records.py` while `__legacy/d1_training_step.py:21` imports it)                        | `tools/gen_shims.py --verify` names the MISSING/STALE emitted file with its path, and `tests/test_d1_training_step_split.py` fails at collection                                                                                                                                                                                                                                                                                                                                                                         | forwarder present, `--verify` reports zero stale                                                                                                                                                                                                                                                                                                                                                        |
| route the pinned raw `print(` at the moved `telemetry.py` through `logger`, or omit the `print_allowlist` ADD from the fragment                                                  | `tests/test_observability_conformance.py` arm 1 (AST print-ban over `LANDED_LAYER_PREFIXES`, which contains `entropic_substrate.ladder`) — with its own subject-set SIZE assertion, so a zero-subject pass cannot hide it                                                                                                                                                                                                                                                                                                | arm 1 green with the added allowlist entry                                                                                                                                                                                                                                                                                                                                                              |
| hand-edit a `_CHECKPOINT_REGIME_TABLE` digest instead of re-deriving it with `tools/regime_rows.py`                                                                              | `tests/test_identity_epoch.py::TestComposedDigestsPerMeasuredRegime` and `tests/test_regime_rows.py`                                                                                                                                                                                                                                                                                                                                                                                                                     | both green after the tool-produced re-pin; the source table byte-identical to the parent tip                                                                                                                                                                                                                                                                                                            |
| point ONE regenerated shim's `_target` at the `__legacy` address instead of the `ladder.capture.*` address                                                                       | `tools/gen_shims.py --verify`. **MEASURED UPGRADE at the parent tip:** round 1 dropped this tooth because `--verify` checked AST SHAPE only; commit `d7c5807` (FOR-1548, on `mythwave/p7b-freeze-and-verify`) added `parse_shim_target` + `find_shim_target_mismatches`, which read the rebind's dotted name out of the ON-DISK AST and compare it against the resolved target, with both arms measured on the fixture tree in that commit. The tooth is therefore mechanically instrumented again and is restored here. | generate-then-`--verify` clean, zero target mismatches, zero stale forwarders                                                                                                                                                                                                                                                                                                                           |
| add a re-export to any of the five destination `__init__.py`                                                                                                                     | **EXHIBIT OR RETRACT.** MEASURED: `lint-imports` fires only if the re-export crosses a layer, and every candidate here is intra-`ladder`, so no mechanical arm exists today. The lane exhibits an arm it wrote, or retracts this row.                                                                                                                                                                                                                                                                                    | five docstring-only `__init__.py`                                                                                                                                                                                                                                                                                                                                                                       |
| touch `ladder/gate/ladder_predecessor.py` (rung-owned) or any p9/p11 slice file                                                                                                  | `tools/lane_slices.py` overlap check (exit 1) and the zip's predicted-vs-actual touch-set diff, which HALTS the wave                                                                                                                                                                                                                                                                                                                                                                                                     | the lane diff is a subset of Allowed files and disjoint from both sibling slices                                                                                                                                                                                                                                                                                                                        |

No `.importlinter` contract is invented for this lane: that file is p0/p13-only and its two forbidden
contracts (`:42`, `:70`) are frozen. The lane VERIFIES they stay green after the move.

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class
permitted inside a structural phase, scoped to log routing, metric routing and entry-point
configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the
identity fixtures and the numerics tests are the tripwire. Capture is where the metric path LIVES, so
unlike p7 this lane has a NON-EMPTY subject.

**Measured subject** (command:
`grep -n "print(\|argparse\|def main(\|__main__\|configure_logging\|logger\|MetricSink"` over the 52
legacy sources):

- Bare `print(` sites: exactly ONE, `__legacy/telemetry.py:506`
  (`print(final_eval_metric(args.path, args.metric))`), the p0-pinned machine-consumed stdout site for
  the `final-eval` subcommand. It stays a RAW `print` forever; a loguru prefix is a hard break.
  `d1_journal_snapshot.py:74` is a false positive of the grep (`_snapshot_fingerprint`).
- **`print_allowlist` is an ADD, never a re-key (M2).** MEASURED: the sets are UNIONED across every
  fragment present (`tests/observability_conformance_config.py::_load_merged_fragment_sets`, which
  `update`s per fragment), and p3b's fragment already carries
  `print_allowlist: ["entropic_substrate.telemetry", "entropic_substrate.cli"]`. p3b's fragment is OUT
  of this lane's envelope, so p8 CANNOT remove that key: it ADDS the landed dotted name
  (`entropic_substrate.ladder.capture.reader.telemetry`) to its OWN fragment, and the stale p3b key
  SURVIVES to p13. Honest consequence: no instrument flags a superfluous allowlist entry — arm 1 only
  checks that every found `print(` IS allowlisted — so the stale key is invisible until p13's sweep
  and is recorded here rather than silently carried.
- Already-routed logging: `capture_results.py:125,241`, `mlflow_aggregate.py:124,132,140,179`,
  `telemetry.py:499,508` all go through `entropic_substrate.logging` (rewritten to
  `kernel.runtime.logging`). Nothing to convert.
- Entry points: ONE, `__legacy/telemetry.py` — `argparse` at `:36,475-476`, `def main(` at `:492`,
  `configure_logging()` as the FIRST statement at `:494`, `if __name__ == "__main__"` at `:513`. It is
  ALREADY one of the two CONFORMING entry points. Per the KEYING ruling (conductor 2026-09-21),
  `conforming_entrypoints` entries are keyed by the `tools/mapping.json` dotted name (the PRE-MOVE
  name, since mapping.json is read-only until p13) and are NEVER re-keyed before p13. Fragment keys,
  MEASURED from `p3b-observability-rail.json`: `print_allowlist`, `conforming_entrypoints`,
  `sink_construction_modules`.
- Sink construction: `sinks.py` imports `MetricSink` from `ladder.capture.ports.metric_sink` (`:68`)
  and returns one from a factory at `:243`; the three adapters remain the only implementations and
  `ladder.capture.adapters.composite` remains the single sanctioned construction seam (arm 3). The
  lane adds NO construction site, so `sink_construction_modules` gains no member.

**D1 observers: rename-only in p8 (ruling P8-OBSERVERS-RENAME-ONLY, FOR-1552).** This heading read
"D1 observer re-basing (ruling p3b-R1), concrete targets" until RETRACTION 10.

RETRACTION 10 (house rule 4): every earlier sentence of this brief saying p8 re-bases the D1 observers
on the `MetricSink` port is retracted, not overwritten: the contract-surface row ("the re-based D1
observer adapters"), the rename bullet in ACCEPTANCE ("the R1 re-basing"), the Allowed-files row
("the R1 re-basing destinations"), the guard row ("the re-based `D1MlflowObserver`"), this heading,
the table row below ("becomes a STRICT adapter that CONSUMES an INJECTED `MetricSink` … in place of
today's `D1MlflowMetricClient` slice"), the protocol disposition "`D1MlflowMetricClient` RETIRES",
commit 5's "re-based on p3b's `MetricSink` port", and the p3b-R1 carry in *Already ruled*. MEASURED by
the second design gate at `e01867a`: `tests/test_d1_observability.py` constructs
`D1MlflowObserver(client=_RecordingMlflowClient(), run_id=…)` at `:202`, `:227` and `:290`, and that
fake is NOT a `MetricSink` (isinstance False; no `log` / `log_artifact` / `finish`), so the re-basing
contradicts Q1; the only mlflow `MetricSink`, `MlflowSink`, is best-effort (it swallows tracking-write
failures, `adapters/mlflow.py:171-172`, `:325`), so a `MetricSink`-consuming observer cannot stay
fail-loud (the `:223` guard); and zero production constructors of `D1MlflowObserver` exist in `src`.
The one-metric-port unification, a strict raising sink and the test carve-out are FOR-1552's, which
gates p13's "D1 observer protocols gone" postcondition.

Concrete targets, MEASURED by reading
`__legacy/d1_observability.py`, `__legacy/d1_tracking.py`,
`ladder/capture/ports/{metric_sink,run_descriptor}.py` and `ladder/capture/adapters/`:

| source                         | surface today                                                                                                                                                                                                               | destination and disposition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__legacy/d1_observability.py` | `AuthoritativeJournalOnlyObserver`, `D1JournalObserver` (Protocol), `D1MlflowMetricClient` (Protocol), `D1MlflowObserver`, `ObservedD1Journal`; `:20` imports `MetricExtractor` from `tracking_contracts`                   | `ladder/capture/adapters/d1_observability.py`, RENAME plus import-line rewrites only (`:20`'s `MetricExtractor` import moves from the `tracking_contracts` shim to `ladder.capture.ports.run_descriptor`). Constructor, `isinstance(client, D1MlflowMetricClient)` guard (`:81-82`), `log_metric` projection (`:97`) and raise-through unchanged: fail-LOUD, never best-effort. `D1MlflowMetricClient` stays public. The former disposition here ("becomes a STRICT adapter that CONSUMES an INJECTED `MetricSink`") is retracted (RETRACTION 10). |
| `__legacy/d1_tracking.py`      | `__all__ = ["D1MetricExtractor", "d1_tracking_descriptor"]` (`:25`), `d1_tracking_descriptor` at `:109`, `D1MetricExtractor` at `:223`; `:23` imports `ExtractedMetrics`, `TrackingRunDescriptor` from `tracking_contracts` | `ladder/capture/adapters/d1_tracking.py`, RENAME plus import-line rewrites only. `d1_tracking_descriptor` consumes the FROZEN run-descriptor door (`TrackingRunDescriptor`, `ExtractedMetrics` from `ladder.capture.ports.run_descriptor`), never a compat export: `:23` moves from the `tracking_contracts` shim (MEASURED: a generated alias of `ladder.capture.ports.run_descriptor`) to that landed address, with no body change.                                                                                                              |

Protocol dispositions, as RULED (P8-OBSERVERS-RENAME-ONLY):

- **`D1MlflowMetricClient` stays PUBLIC** at the new address, unchanged. RETRACTION 10: this bullet
  read "`D1MlflowMetricClient` RETIRES as a public protocol at the new address … `MetricSink` is now
  the one boundary"; retracted, not overwritten. Its retirement, if any, is FOR-1552's.
- **`MetricExtractor` does NOT retire and is NOT redefined.** It is p3b's, at
  `ladder/capture/ports/run_descriptor.py:97`, alongside `LegacyMetricExtractor` (`:103`) and
  `ExtractedMetrics` (`:86`). `D1MetricExtractor` becomes one more implementation of it.
- **`D1JournalObserver` does NOT retire.** `observe(event: D1JournalEvent)` is a journal-EVENT
  protocol, not a metric protocol; it stays outside the one-METRIC-port rule, and
  `AuthoritativeJournalOnlyObserver` stays its explicit null projection. Folding it into `MetricSink`
  would widen a frozen port — a `DECISION`, not an edit.

`tests/test_d1_observability.py` is NOT edited (Q1): it imports by the old dotted names and the
regenerated shims keep it green. Its `:223` failure-propagation test is the guard tooth for the moved
observer's raise-through. The second design gate recorded this sentence as false only IN CONJUNCTION
with the retracted re-basing (both could not hold); under P8-OBSERVERS-RENAME-ONLY it holds and is
AFFIRMED.

## Planned commit branches

**RULING W2-SCC (conductor 2026-09-23, after the w2 design gate HALTED).** The earlier
six-landing-commit partition (custody · records · resource · ledger · reader · adapters) is
INFEASIBLE: it makes **14** intra-set import lines FORWARD REFERENCES, and the sub-package graph is
CYCLIC, so no ordering of those six groups exists. MEASURED over the pack's 111 intra-set lines, each
line's source and target leaf mapped to its sub-package:

| edge (source sub-package → target sub-package) | intra-set lines |
| ---------------------------------------------- | --------------- |
| `records` → `reader`                           | 5               |
| `reader` → `records`                           | 4               |
| `records` → `resource`                         | 4               |
| `custody` → `reader`                           | 4               |
| `adapters` → `records`                         | 4               |
| `records` → `custody`                          | 2               |
| `adapters` → `resource`                        | 2               |
| `custody` → `records`                          | 1               |
| **total cross-sub-package**                    | **26**          |

Two 2-cycles make `{custody, records, reader}` one strongly-connected component:
`records ↔ custody` (2 + 1) and `records ↔ reader` (5 + 4). Under the old order (custody 1, records 2,
resource 3, ledger 4, reader 5, adapters 6) the forward references are
`custody→reader` 4 + `custody→records` 1 + `records→reader` 5 + `records→resource` 4 = **14**.
`ledger` has ZERO intra-set edges in either direction; `resource` has zero OUTGOING.

**Landing commits now partition by strongly-connected group, in topological order.** Transient
`__legacy` addresses inside the phase (the design gate's option B) are **NOT permitted**.

| landing group                            | leaves | outgoing intra-set edges  | position                               |
| ---------------------------------------- | ------ | ------------------------- | -------------------------------------- |
| `{ledger}`                               | 4      | 0                         | first (either order with `{resource}`) |
| `{resource}`                             | 9      | 0                         | first (either order with `{ledger}`)   |
| `{custody + records + reader}` — the SCC | 37     | internal only             | after both zero-outgoing groups        |
| `{adapters}` (the two D1 observers)      | 2      | 4 → records, 2 → resource | last                                   |

**Topological check, BEFORE the first commit.** The lane proves, read-only at the parent tip, that no
planned commit lands a module whose intra-set imports name a p8 leaf not yet landed: for each of the
four groups in order, resolve every `entropic_substrate.*` import line of its legacy sources to a p8
target and assert the target is in this group or an earlier one. Command and full output go in the
lane report; a single violation is a `DECISION`, not a re-ordering the lane invents. The check is an
AST pass over the 52 sources INCLUDING function-local imports, resolved through `gen_shims`'
`load_mapping`, `merge_fragments` and `resolve_all_modules`, and it must map the `__legacy.data.<x>`
spelling at `data/build.py:15-17` back to its p8 target (RETRACTION 8), or those lines fall into
"resolves to neither" and trip a spurious `DECISION`. Expected at `c2f64a0` [predicted from the
build.py re-measure; the second design gate MEASURED 111 / 73 / 5 / 4 at `e01867a`]: 193 lines = 111
intra-set + 71 landed-via-fragment + 7 already-landed + 4 p9-torch-half-named; zero violations for
the order ledger, resource, SCC, adapters; zero imports resolving to neither. The lane REPRODUCES it
at its own epoch with its own command and output and never cites the gate's; any drift is a
`DECISION`.

The ids are the manifest's own `phases[p8-capture].issues` = FOR-1464 … FOR-1470 (seven), mapped 1:1
to the seven commits. **ORDERING RULED 2026-09-23:** the cold-import test is commit 1, RED-first, and
the landing groups follow — the earlier "landing commits first" instruction governed ID MAPPING, never
the placement of a test relative to the code it guards, and house rule 5 wins. None of the seven ids
is left over, so no id carries an "absorbed into commit N" annotation.

| # | Linear   | commit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | fragment rows added in THIS commit                         | budget in THIS commit                 |
| - | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------- |
| 1 | FOR-1464 | `tests/test_capture_cold_import.py` — RED-first, BOTH arms in the commit ladder (house rule 5 wins over id-mapping order): the 52 destination addresses are derived LIVE from the fragment, never typed (the p7 form) and never from a `ladder.capture.*` prefix (an empty set FAILS once the fragment exists; collected count == landed-row count, 52 at the tip; second design gate), so at commit 1 (no p8 fragment rows yet) it parametrizes over zero landed rows and its RED arm is exhibited by seeding ONE fragment row for a not-yet-moved module (cold import fails at the new address), the seed removed before commit; from commit 2 on it collects the landed rows and is RED for any unmoved one; the RED arm is re-exhibited by mutation after the landing commits and both arms go in the lane report.                                                                                                                                                                                                                                          | none                                                       | none (no fragment yet at this commit) |
| 2 | FOR-1465 | Land `{ledger}` — 4 leaves into `ladder/capture/ledger/`. Zero intra-set outgoing edges, so nothing it imports is unlanded. FIRST fragment write (commit 1 lands no module): the `legacy_edge_budget` key must EXIST here (BUDGET-PER-COMMIT).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | the 4 ledger rows                                          | MEASURED (expect 45)                  |
| 3 | FOR-1466 | Land `{resource}` — 9 leaves into `ladder/capture/resource/`. Zero intra-set outgoing edges. Order with the `{ledger}` commit is interchangeable.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | the 9 resource rows                                        | MEASURED (expect 45)                  |
| 4 | FOR-1467 | Land the SCC `{custody + records + reader}` — **37 leaves** in ONE commit (19 + 12 + 6), because the group is strongly connected and split-free landing is the only feasible partition. Carries: both p5 record halves and the **P8-FRAG** migration of their two rows out of `p5-kernel.json`; the Q1 carve-out lines 52/53/127/128 of `tests/test_d1_training_step_split.py`; the four ex-p9 sibling lines rewritten onto the record halves' new addresses; the `print_allowlist` **ADD** (same commit that moves `telemetry.py`, or conformance arm 1 goes RED); the two frozen-sentinel modules `evidence.d1_capture_{aggregate,bundle}`, so its rows are mandatory. The ADDITIVE `scripts/assemble_review_packet.py` edit (P8-PACKET-SCRIPT; reader byte-identical, 11069 bytes); the retirement of `tests/test_shim_layer.py`'s data-facade RED case with its retraction (P8-SHIM-LAYER-RETIRE); `data/build.py:15-17` rewritten once more to `ladder.capture.custody.*`, `:18-19` unchanged (DATA-BUILD-RUNG). **No facade step** (ruling FACADE-EMPTY). | the 37 custody + records + reader rows (2 via P8-FRAG)     | MEASURED (expect 45)                  |
| 5 | FOR-1468 | Land `{adapters}` — the two D1 observers MOVED as renames plus import rewrites only (`adapters/d1_observability.py`, `adapters/d1_tracking.py`; destination per p3b-R1, content per P8-OBSERVERS-RENAME-ONLY: constructor, `isinstance` guard, `log_metric` projection and raise-through unchanged, `D1MlflowMetricClient` public; the `MetricSink` unification is FOR-1552's, RETRACTION 10). Last LANDING commit, because its 6 intra-set edges point back into `records` and `resource`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | the 2 observer rows (fragment now complete: 4+9+37+2 = 52) | MEASURED (expect 45)                  |
| 6 | FOR-1469 | Rewrite the SIX landed importers (`composition/source_cli.py`, `ladder/run/{execution,generation,span_diagnostics}.py`, `ladder/capture/adapters/{composite,mlflow}.py` — addresses, docstring cites and the six residue names, `MlflowStartupError` re-exported by IDENTITY) **WITH** the `_CHECKPOINT_REGIME_TABLE` re-pin in the SAME commit (R-PIN; four of the six are reader-set members). This is the ONLY commit that moves the edge count.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | none                                                       | MEASURED (expect 38)                  |
| 7 | FOR-1470 | SEAL the fragment: the final MEASURED `legacy_edge_budget` (**38 [predicted]**), the `read-old-table.md` dated p8 row, and a `note` recording the ELEVEN numbered retractions (counts · audit split rows · shim form · run_ladder · remap_check unrunnable FOR-1549 · facade wording · p7→p7b epoch and control commands · build.py bucket rows, DATA-BUILD-RUNG · test_shim_layer untouched, P8-SHIM-LAYER-RETIRE · observer re-basing, P8-OBSERVERS-RENAME-ONLY / FOR-1552 · `custody/h_source.py` non-goal). Commit the `lane_slices.py` output and the topological-check output in the lane report. NO `remap_check` output: the tool cannot run (RETRACTION 5).                                                                                                                                                                                                                                                                                                                                                                                            | none                                                       | final MEASURED                        |

**FRAGMENT TIMING (binding, not a style note).** The fragment is NOT written once at the end. THREE
mechanisms force per-commit rows and a per-commit budget: a shim points at the new address only when a
fragment row says `landed: true` (so without them the first landing commit leaves shims pointing at
deleted `__legacy` sources and prek's pre-commit `pytest -m unit` fails `tests/test_shim_layer.py`);
`tests/test_module_identity.py` reverse-maps moved modules through the fragments' `target` field (so
the SCC commit's two frozen-sentinel modules would be unresolvable); and
`tests/test_legacy_coupling.py::load_committed_budget` selects the HIGHEST-ORDERED fragment present,
which is `p8-capture.json` from the first landing commit (commit 2) onward, so its `legacy_edge_budget` key must exist and be
MEASURED at every landing commit (BUDGET-PER-COMMIT). p7 is the precedent: 35 at `2ee5e87`, 45 at
`f94b808`. The seal carries the final number, never the first write.
