# p11-analyze-review-file — `ladder/{analyze,review,file}`

Authoritative brief for one `td-phase-lifecycle.cascade` lane inside wave **w2**. v2, written
2026-09-23, superseding the 2026-09-17 v1 under the conductor's ruling D-WAVES r3 (operator,
2026-09-23). Plan: `docs/plans/src-hexagonal-refactor.plan.md` (§p11, PRIORITY AMENDMENT,
rulings table rows `BUDGET-MONOTONE retracted` line 842 and `D-WAVES r3 ruled` line 843).

**EPOCH STAMP.** The lane measurement pack was produced at `1d97033`; every count in this brief was
re-measured at **`79d8228`** (the `mythwave/p7-register-gate` tip) with the command recorded beside
it, or is explicitly labelled `[predicted]` with its arithmetic. The only file differing between the
two commits is `tests/test_lane_slices.py`, which is in no lane slice, so the pack's src-derived
numbers carry forward unchanged. No table mixes the two classes unlabelled. The lifecycle's `acceptance` arg reads "Per the brief's ACCEPTANCE
section"; this file is where that section lives.

Like p7, **p11 owes NO R-PIN** (measured: zero reader-set members in its slice). The control is an
assertion, not an omission — see *R-PIN: not owed*.

## Branch

| field                                        | value                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p11-analyze-review-file`                                                                                                                                                                                                                                                                                                                  |
| parent                                       | **`mythwave/p7-register-gate`** (D-WAVES r3). CORRECTED from v1's `mythwave/p6-core`: w1 ran SERIAL (p6 → p7), so the w1 integration tip is p7's branch, not p6's. v1 line 10's reasoning is struck.                                                                                                                                                |
| mode (sprint manifest, parallel eligibility) | `ratchet` (known set, parallel-eligible; `parallel_group: w2`)                                                                                                                                                                                                                                                                                      |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate.                                                                                                                                                                                                                          |
| engine_mode                                  | `arch` (manifest field, machine-read by the launch composer; ruled 2026-09-18)                                                                                                                                                                                                                                                                      |
| wave                                         | `w2` = p8 ∥ p9 ∥ p11, ONE `mass-patch-fanout` invocation, K=3                                                                                                                                                                                                                                                                                       |
| `gates.liveRequired`                         | `false`                                                                                                                                                                                                                                                                                                                                             |
| live gate                                    | DEFERRED. `gates.liveDeferred` names the wave zip as re-certification owner: one `just ship-it` on the integrated tip, never K concurrent podman stacks (house rule 2, 4 vCPU / 4 GB VM).                                                                                                                                                           |
| gates                                        | lifecycle arg `{"offline":"just check","exit":"just pc-ci"}`. The sprint-manifest `gates` object admits only `{"liveRequired": false, "liveDeferred": ...}` (`additionalProperties: false`).                                                                                                                                                        |
| `policy.publishMode`                         | `'defer'` (ruling O-1). The lane ends **CERTIFIED** and publishes nothing.                                                                                                                                                                                                                                                                          |
| launch                                       | WAVE MEMBER: one ticket `{id,title,scope,acceptance}` inside the single w2 `mass-patch-fanout` invocation. `cascadeScriptPath` = `.claude/workflows/td-phase-lifecycle.cascade.js`.                                                                                                                                                                 |
| lane worktree                                | NESTED at `<checkout>/.claude/worktrees/p11-analyze-review-file`, never a sibling of the conductor's worktree (measured law, 2026-08-27 pilot: a sibling lane is unreachable to workers).                                                                                                                                                           |
| lane isolation                               | `cd` to the lane worktree FIRST in every stage, never `EnterWorktree`. Verify `pwd` and branch before acting; FREEZE on mismatch.                                                                                                                                                                                                                   |
| zip order                                    | **p8 → p9 → p11**, p11 LAST. MEASURED inbound dependency mass: p8 receives **91** (50 from p11 + 41 from p9), p9 receives 5 (4 from p8 + 1 from p11), **p11 receives 0**; 96 cross-lane edges in all (50 + 41 + 4 + 1) (`depends_in: {}` in the lane pack). The planner's `zipOrder` is validated as an exact permutation before any lane launches. |
| trunk                                        | `refactor/hexagonal-redesign`                                                                                                                                                                                                                                                                                                                       |

RETRACTION (house rule 4): plan lines 375-376 still read "w2 zips p11 (49), p9 (52), p8 (53), tip
`mythwave/p8-capture`" and "p8, p9 and p11 all parent on `mythwave/p6-core`". Both are superseded by
the r3 row at plan line 843 and are retracted here, not overwritten. The plan text is left as the
historical record; this brief is not authorised to edit it.

## Wave w2 mechanics

What a parallel lane may and may not do, all of it ruled — see *Already ruled* rows `D-WAVES r3`,
`sibling-target imports` and `rung-owned`.

- **Parent / worktree.** Fork `mythwave/p7-register-gate`. The lane never rebases itself; the zip
  restacks it.
- **`zipRestackHook`** (landed `14c587a`, read from `.claude/workflows/mass-patch-fanout.js:733` and
  `launch-queue.jsonl` decision `D-WAVES-2026-09-22`): at EVERY restacked tip, FIRST, before any
  gate — `tools/gen_shims.py` generate + `--verify`, then commit the forwarders. Only then do the
  gates run.
- **Forwarders (FOR-1547, `e9e5897`).** `gen_shims` writes a FORWARDER at a vacated `__legacy` path
  whenever a `landed: true` row is still imported at that address by a LIVE, non-generated consumer
  under `src/` or `tests/` (the predicate is `collect_legacy_importers(importer_scan_roots(out_dir))`,
  `tools/gen_shims.py:933` `importer_scan_roots` (roots = `out_dir` plus its sibling `tests/`) and
  `:941` `collect_legacy_importers`, which SKIPS exact-form shims and forwarders so no forwarder is
  its own justification). MEASURED by running that predicate at `79d8228`: **6** of the 47 targets'
  `__legacy` addresses have a live importer today — `analysis.ladder_analysis`,
  `analysis.ladder_filing` (from `ladder/gate/ladder_predecessor.py:8,12`, RUNG-OWNED) and
  `evidence.{d1_filing, d1_filing_graph, d1_gate, d1_validator}` (from
  `ladder/gate/d1_verdict.py:11,18,22,23`). This lane REWRITES the four `d1_verdict` lines onto
  `ladder.file.*`, so after its own rewrite the predicate leaves **2** forwarders in this worktree:
  `src/entropic_substrate/__legacy/analysis/ladder_analysis.py` and
  `.../__legacy/analysis/ladder_filing.py` (+ their `.pyi` siblings). They are generator output,
  never hand-edited, they are IN Allowed files, and the actual set is self-declaring through
  `git status --porcelain` after generate + `--verify` (D-B).
- **The lane rewrites its sibling-target imports to the explicit `entropic_substrate.__legacy.<x>`
  address**, which is a real file in this worktree. After the zip, the sibling's own `gen_shims` run
  leaves a forwarder there, so the address stays valid. The post-zip serial reconciliation rung
  (conductor) rewrites all of them to the final `ladder.capture.*` / `ladder.run.*` addresses and
  deletes the forwarders.
- **The lane must NOT:** publish (`publishMode:'defer'`; a `gt submit` in the lane is a defect) ·
  run `just container-test` or any podman stack (house rule 2) · touch
  `src/entropic_substrate/ladder/gate/ladder_predecessor.py` (RUNG-OWNED, below) · edit
  `tools/mapping.json`, `.importlinter`, `tests/observability_conformance_config.py`, any EXISTING
  test, another phase's fragment, the plan, the manifest, or CI · rebase itself onto a sibling ·
  assume a sibling's `ladder.*` address exists.
- **RUNG-OWNED file.** `src/entropic_substrate/ladder/gate/ladder_predecessor.py` (p7-landed) holds
  explicit `__legacy` edges to TWO lanes' targets — p11's at `:8`
  (`__legacy.analysis.ladder_analysis`) and `:12` (`__legacy.analysis.ladder_filing`), p8's at `:14`
  (`__legacy.evidence.d1_capture_artifacts`) and `:20` (`__legacy.evidence.d1_capture_bundle`).
  Under rewrite-on-arrival both lanes would rewrite one file → same-file conflict at the zip.
  Ruled: **neither lane touches it.** It is declared `--rung-owned` in the slice command, leaves
  every slice, is reported under `RUNG-OWNED`, is NOT in Allowed files, and the reconciliation rung
  rewrites it once. Teeth on the declaration itself: `tools/lane_slices.py` exits 2 if fewer than
  two slices carry a declared file, so the flag cannot decay into an ignore list.
- **CONTRAST — `src/entropic_substrate/ladder/gate/d1_verdict.py` IS p11's importer** and IS
  rewritten by p11. It carries only p11 targets (`:11,18,22,23` → `__legacy.evidence.d1_filing`,
  `d1_filing_graph`, `d1_gate`, `d1_validator`, all four dest `ladder.file`). It is IN Allowed
  files. The old manifest touch_set omitted it; that omission is a defect, not a rule.

### The exact slice command (conductor's measurement oracle, verbatim)

```bash
python3 tools/lane_slices.py src tools/mapping.json \
  p8-capture=entropic_substrate.capture_results,entropic_substrate.d1_calibration_records,entropic_substrate.d1_content_addressed_custody,entropic_substrate.d1_journal_snapshot,entropic_substrate.d1_numerical_defect,entropic_substrate.d1_observability,entropic_substrate.d1_tracking,entropic_substrate.d1_training_step_records,entropic_substrate.data.build,entropic_substrate.data.enwik8,entropic_substrate.data.fineweb,entropic_substrate.data.synthetic,entropic_substrate.evidence.d1_abort_evidence,entropic_substrate.evidence.d1_capture_aggregate,entropic_substrate.evidence.d1_capture_aggregate_codec,entropic_substrate.evidence.d1_capture_aggregate_validation,entropic_substrate.evidence.d1_capture_artifacts,entropic_substrate.evidence.d1_capture_bundle,entropic_substrate.evidence.d1_capture_declaration_codec,entropic_substrate.evidence.d1_capture_filesystem,entropic_substrate.evidence.d1_capture_ports,entropic_substrate.evidence.d1_capture_preparation_custody,entropic_substrate.evidence.d1_capture_preparation_custody_repository,entropic_substrate.evidence.d1_capture_roster,entropic_substrate.evidence.d1_capture_service,entropic_substrate.evidence.d1_capture_subprocess,entropic_substrate.evidence.d1_capture_validation,entropic_substrate.evidence.d1_journal,entropic_substrate.evidence.d1_journal_values,entropic_substrate.evidence.d1_lifecycle_repository,entropic_substrate.evidence.d1_lifecycle_values,entropic_substrate.evidence.d1_process_manager,entropic_substrate.evidence.d1_process_replay,entropic_substrate.evidence.d1_telemetry,entropic_substrate.evidence.d1_telemetry_measurements,entropic_substrate.evidence.d1_telemetry_payloads,entropic_substrate.evidence.d1_telemetry_query,entropic_substrate.evidence.d1_telemetry_values,entropic_substrate.evidence.d1_terminal_defect_capture,entropic_substrate.evidence.d1_terminal_defect_query,entropic_substrate.evidence.resource_billing,entropic_substrate.evidence.resource_billing_reconciliation,entropic_substrate.evidence.resource_ledger,entropic_substrate.evidence.resource_ledger_codec,entropic_substrate.evidence.resource_meter_models,entropic_substrate.evidence.resource_meter_runtime,entropic_substrate.evidence.resource_meter_strategy,entropic_substrate.evidence.resource_work,entropic_substrate.evidence.resource_work_codec,entropic_substrate.mlflow_aggregate,entropic_substrate.sinks,entropic_substrate.telemetry \
  p9-run-lower=entropic_substrate.d1_build_cloud_adapter,entropic_substrate.d1_calibration,entropic_substrate.d1_command_runner,entropic_substrate.d1_dual_failure,entropic_substrate.d1_dvc_remote_config,entropic_substrate.d1_evaluation,entropic_substrate.d1_evaluation_codec,entropic_substrate.d1_fresh_trainer,entropic_substrate.d1_git_provenance_adapter,entropic_substrate.d1_operator_ports,entropic_substrate.d1_optimizer,entropic_substrate.d1_r2_runtime_composition,entropic_substrate.d1_resource_activation,entropic_substrate.d1_resume_runtime,entropic_substrate.d1_resume_trainer,entropic_substrate.d1_runpod_adapter,entropic_substrate.d1_segment_engine,entropic_substrate.d1_system_runtime_probe,entropic_substrate.d1_terminal_failure,entropic_substrate.d1_trainer_evidence_adapter,entropic_substrate.d1_trainer_handoff,entropic_substrate.d1_trainer_handoff_codec,entropic_substrate.d1_trainer_mailbox_contracts,entropic_substrate.d1_trainer_mailbox_worker,entropic_substrate.d1_trainer_object_transport,entropic_substrate.d1_trainer_output_publication,entropic_substrate.d1_trainer_process_contracts,entropic_substrate.d1_trainer_request_codec,entropic_substrate.d1_trainer_runtime,entropic_substrate.d1_trainer_s3_object_adapter,entropic_substrate.d1_trainer_stage,entropic_substrate.d1_training_step,entropic_substrate.experiments.ladder_matching,entropic_substrate.predictive_state.checkpoint_envelope,entropic_substrate.predictive_state.checkpoint_repository,entropic_substrate.predictive_state.checkpoint_validation,entropic_substrate.predictive_state.checkpoint_values,entropic_substrate.predictive_state.ladder_runtime,entropic_substrate.predictive_state.runtime,entropic_substrate.predictive_state.runtime_components,entropic_substrate.predictive_state.runtime_construction,entropic_substrate.provider_launch \
  p11-analyze-review-file=entropic_substrate.analysis.charts,entropic_substrate.analysis.codec_equivariance,entropic_substrate.analysis.d1_analysis_api,entropic_substrate.analysis.d1_analysis_receipt_codec,entropic_substrate.analysis.d1_blind_aliases,entropic_substrate.analysis.d1_capture,entropic_substrate.analysis.d1_clause_policy,entropic_substrate.analysis.d1_diagnostics,entropic_substrate.analysis.d1_execution_receipt,entropic_substrate.analysis.d1_external_analysis_adapter,entropic_substrate.analysis.d1_external_analysis_codec,entropic_substrate.analysis.d1_external_analysis_control,entropic_substrate.analysis.d1_external_review_adapter,entropic_substrate.analysis.d1_external_worker,entropic_substrate.analysis.d1_factorial,entropic_substrate.analysis.d1_machine_analysis,entropic_substrate.analysis.d1_machine_analysis_binding,entropic_substrate.analysis.d1_machine_analysis_codec,entropic_substrate.analysis.d1_machine_analysis_verification,entropic_substrate.analysis.d1_output_access,entropic_substrate.analysis.d1_real_arm_verdict,entropic_substrate.analysis.d1_review_execution,entropic_substrate.analysis.d1_review_packet,entropic_substrate.analysis.embedding_geometry,entropic_substrate.analysis.field_head_audit,entropic_substrate.analysis.ladder_analysis,entropic_substrate.analysis.ladder_filing,entropic_substrate.analysis.ladder_registration,entropic_substrate.analysis.mlflow_attach,entropic_substrate.analysis.palette,entropic_substrate.analysis.seed_sizing,entropic_substrate.analysis.series,entropic_substrate.d1_filing_application,entropic_substrate.d1_inspect_codec,entropic_substrate.d1_inspect_projection,entropic_substrate.d1_inspect_request,entropic_substrate.d1_inspect_values,entropic_substrate.evidence.d1_decision_validation,entropic_substrate.evidence.d1_filing,entropic_substrate.evidence.d1_filing_assembly,entropic_substrate.evidence.d1_filing_edges,entropic_substrate.evidence.d1_filing_graph,entropic_substrate.evidence.d1_filing_receipts,entropic_substrate.evidence.d1_gate,entropic_substrate.evidence.d1_process_filing,entropic_substrate.evidence.d1_review_adjudication,entropic_substrate.evidence.d1_validator \
  --regenerable tests/test_identity_epoch.py,tools/regime_rows.py,tests/test_regime_rows.py,docs/identity/read-old-table.md,.secrets.baseline \
  --rung-owned entropic_substrate/ladder/gate/ladder_predecessor.py --allow-dependencies --json
```

MEASURED output for this lane: `n_targets` 47, `slice_file_count` **142** = 47 legacy sources + 94
shim-pair files (47 × `.py` + `.pyi`) + 1 landed importer (`d1_verdict.py`). `depends_in: {}`.

## Contract surface owned

| contract                                                                                                                                                             | kind     | frozen_after |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| the 47 leaf addresses under *Scope*: `ladder.analyze` **22**, `ladder.review` **13**, `ladder.file` **12**                                                           | contract | p13          |
| destination package `__init__.py` shape for `ladder/{analyze,review,file}`: DOCSTRING-ONLY, no re-exports (p4/p6/p7 precedent, `core/sentence/__init__.py` 85 bytes) | contract | p13          |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT.

- kernel (p4, p5) and `kernel.artifacts` as extended by p5b; core (p6); `ladder.register.*` and
  `ladder.gate.*` (p7). MEASURED: **118** of the 47 sources' import lines resolve to a LANDED
  address (table below).
- the RETIRED D1 reader source closure (p2, ruling E-1, `8cf5aae`) — consumed as an ABSENCE:
  `analysis.d1_reader_source_closure` has a `tools/mapping.json` row (`:156`) but **no file on
  disk**. It is not a target, and the old manifest's touch_set rows for it are phantoms.
- `tools/gen_shims.py` (`--verify`, `--preflight`, forwarders), the landed-set fragment convention,
  `tests/test_legacy_coupling.py` (EQUALITY arm), `tests/test_no_old_prefix.py` (p3b),
  `tests/test_observability_conformance.py` + `tests/observability_conformance_config.py` (p3b,
  read-only here — this lane promotes through its OWN fragment keys).
- `kernel/runtime/logging.py` (p3b) — specifically `configure_logging` (consumed by the entry-point
  promotion) and the module `logger` the two rewritten `print(` sites route through. This is the ONE
  behaviour-change class permitted here (P-1).
- `tools/lane_slices.py`, `tools/import_graph.py` (conductor measurement oracles).

RETRACTED 2026-09-18 (Q2 ruled (b)): this list previously carried `ladder/capture/reader` (p8), a
same-wave contract edge R-parallel forbids. Under r3 the sibling relationship is handled by the
forwarder mechanism, not by a contract edge; the struck row stays struck.

## Sacred invariants

The five below are quoted verbatim from `CLAUDE.md` (North star and *Non-negotiable numerics*).
`SI-*` ids are local to this brief set; the text is the contract.

| id   | verbatim invariant                                                                                                          | tripwire                                                                                                                                                  | guard-teeth recipe (house rule 5, both arms)                                                                                                                                                                                                                                                                                                                          |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | MEASURED: `touch_set ∩ core/** = ∅`. Two targets IMPORT core geometry (`charts.py:40`, `embedding_geometry.py:53`) and must not change one numerics line. | The load-bearing arm is p6's autocast-STATE spy (`tests/test_kappa_autocast_island.py`), never a dtype arm (`_geometry_dtype()` floors to fp32 regardless of autocast state — p0 round-C, FOR-1517). This lane records the arm UNTOUCHED: `git diff mythwave/p7-register-gate -- tests/test_kappa_autocast_island.py src/entropic_substrate/core/geometry/` is EMPTY. |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | `touch_set ∩ core/geometry = ∅`.                                                                                                                          | Widen the κ clamp bound by one ulp and feed a boundary input; `tests/test_kappa_geometry.py`'s κ-domain arm must go RED. Revert in-branch.                                                                                                                                                                                                                            |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | `touch_set ∩ core/substrate = ∅`. `embedding_geometry.py:54,56` and `field_head_audit.py:55` import it read-only.                                         | Detach κ from the load-balance term; the gradient-reaches-κ unit test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                  |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | `touch_set ∩ core/sentence = ∅`.                                                                                                                          | Append a `UnitNorm` to the magnitude-preserving carrier's output; the carrier magnitude test must go RED. Revert in-branch.                                                                                                                                                                                                                                           |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | `touch_set ∩ core/latent = ∅`.                                                                                                                            | **EXHIBIT OR RETRACT.** Substitute a `sqrt(p + eps)` for the latent law in the main path; p8's lane MEASURED that no named latent-path law test exists, so the arm cannot be claimed from this wording. The lane either EXHIBITS a captured RED naming the instrument, or RETRACTS this row in the PR body with the measurement (house rule 4). Revert in-branch.     |

House rules from `CLAUDE.md`, verbatim headline, each binding in this phase:

| id   | verbatim house rule                                                            | tripwire                                                                                                   | guard-teeth recipe                                                                                                                        |
| ---- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| HR-1 | "Publish with `just ship-it`, never a bare `gt submit`."                       | Lane under `publishMode:'defer'` ends CERTIFIED and NEVER publishes; `just ship-it` runs once, at the zip. | Inspect the lane transcript for any `gt submit`; presence is a BLOCKED verdict, not a note.                                               |
| HR-2 | "Probe sequentially. Never draw a conclusion from concurrent containers."      | Three lanes live on a 4 vCPU / 4 GB podman VM; two MLflow stacks make even a control appear hung.          | The lane runs NO container. Any container conclusion in the lane transcript invalidates itself.                                           |
| HR-3 | "Never fix flakiness by raising a timeout or a retry budget."                  | A gate that goes green only after a timeout bump.                                                          | A diff that raises any timeout or retry constant is a BLOCKED verdict unless it carries the measurement showing the cause was found.      |
| HR-4 | "Comments record measurements, with the measurement."                          | A comment in this lane's diff stating a fact with no measurement attached.                                 | A wrong measured claim is retracted explicitly, in the commit that fixes it. This lane carries NINE such retractions (see *Retractions*). |
| HR-5 | "TDD is non-negotiable, and guard-teeth prove both arms."                      | A guard landed without its RED arm demonstrated.                                                           | Every guard carries the mutation listed under *Guard teeth by mutation*, performed and reverted in-branch, evidence in the PR body.       |
| HR-6 | "Secrets never enter git."                                                     | gitleaks runs at pre-commit and pre-push.                                                                  | No credential, token or `.env` value in any diff of this phase.                                                                           |
| HR-7 | "Upgrading an existing venv across the mlflow → mlflow-skinny swap breaks it." | `import mlflow` raising `ModuleNotFoundError` in a fresh lane worktree.                                    | Fix with `uv sync --all-groups --reinstall-package mlflow-skinny`, never by pinning around it.                                            |

## Scope

The ANALYZED, BLIND-REVIEWED and FILED stages: **47 leaf modules**, MEASURED as the
`tools/mapping.json` rows whose `dest` is `entropic_substrate.ladder.{analyze,review,file}`, whose
legacy source is a leaf `.py` under `src/entropic_substrate/__legacy/`, and which no landed-set
fragment already marks `landed: true`. Zero behaviour change except the observability class (P-1).

Command: `python3 tools/lane_slices.py … --json` (above) → `n_targets: 47`; dest split measured by
`collections.Counter(dest_by_target.values())` → analyze 22, review 13, file 12.

**RETRACTION (house rule 4).** v1's Scope, Allowed files and ACCEPTANCE all said **49**, and its
*Contract surface* said "analyze (24), review (13), file (12)". Both are WRONG and are retracted,
not overwritten. 49 = 47 leaves **+ 2 non-targets, both dest `ladder.analyze`**, which is exactly
why the analyze count was inflated by two:

1. `entropic_substrate.analysis` — a **PACKAGE KEY**. Never a movable target and never a fragment
   row: `tools/gen_shims.py::merge_fragments` raises `ManifestError` on a fragment naming one, and
   `tools/lane_slices.py::parse_lane` exits 2 on one.
2. `entropic_substrate.analysis.d1_reader_source_closure` — **RETIRED at p2** (ruling E-1 extended
   2026-09-20, landed `8cf5aae`). Its mapping row survives; the file does not
   (`ls src/entropic_substrate/__legacy/analysis/ | grep reader` → no match).

The 47 legacy sources, by destination package:

- **`ladder/analyze` (22):** `analysis/{charts, codec_equivariance, d1_analysis_receipt_codec,
  d1_capture, d1_clause_policy, d1_diagnostics, d1_factorial, d1_machine_analysis,
  d1_machine_analysis_codec, d1_output_access, embedding_geometry, field_head_audit,
  ladder_analysis, ladder_registration, mlflow_attach, palette, seed_sizing, series}.py` and
  `__legacy/{d1_inspect_codec, d1_inspect_projection, d1_inspect_request, d1_inspect_values}.py`
- **`ladder/review` (13):** `analysis/{d1_analysis_api, d1_blind_aliases, d1_execution_receipt,
  d1_external_analysis_adapter, d1_external_analysis_codec, d1_external_analysis_control,
  d1_external_review_adapter, d1_external_worker, d1_machine_analysis_binding,
  d1_machine_analysis_verification, d1_review_execution, d1_review_packet}.py` and
  `evidence/d1_review_adjudication.py`
- **`ladder/file` (12):** `analysis/{d1_real_arm_verdict, ladder_filing}.py`,
  `__legacy/d1_filing_application.py`, and `evidence/{d1_decision_validation, d1_filing,
  d1_filing_assembly, d1_filing_edges, d1_filing_graph, d1_filing_receipts, d1_gate,
  d1_process_filing, d1_validator}.py`

MEASURED: `tools/mapping.json` carries **9** rows with `action: split`
(`python3 -c "…[k for k,v in modules.items() if v.get('action')=='split']"`), of which **exactly one
is a p11 target**: `entropic_substrate.evidence.d1_filing_receipts` (dest `ladder.file`). The other
eight belong to p8, p9, p10 and p12.

**Shim form** (ruled 2026-09-17, corrected 2026-09-23 at p7; quoted from the generator's own
output): `import entropic_substrate.<new landed address> as _target` then
`sys.modules[__name__] = _target`. The `_target` is the NEW `ladder.*` address, never `__legacy`. A
string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously
green.

**Package `__init__.py` disposition.** `entropic_substrate.analysis` is a package key, so it is
never a fragment row. Three files, three dispositions, MEASURED with
`wc -l` + `cat` at `79d8228`:

| file                                                     | measured now                                                                                  | this lane does                                                                                                                                                                                |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__legacy/analysis/__init__.py`                          | 21 lines, module docstring + `from __future__ import annotations` + `__all__: list[str] = []` | **UNTOUCHED.** The facade retraction is DEFERRED to p12 (ruling P11-FACADE, below). It stays a real `__init__.py`: two non-p11 leaves and this lane's two forwarders live inside the package. |
| `src/entropic_substrate/analysis/__init__.py` (old path) | 21 lines, byte-identical to the above                                                         | **UNTOUCHED, verbatim.**                                                                                                                                                                      |
| `ladder/{analyze,review,file}/__init__.py`               | do not exist                                                                                  | NEW, DOCSTRING-ONLY, zero re-exports (p4/p6/p7 precedent).                                                                                                                                    |

**Why the facade stays (ruling P11-FACADE, conductor 2026-09-23).** MEASURED at `79d8228` with
`ls src/entropic_substrate/__legacy/analysis/*.py | grep -v __init__ | wc -l`: the package holds **34**
leaf modules, of which **32** are p11 targets. The remaining **2** — `d1_probe.py` and
`ladder_probe.py` — carry mapping dest `entropic_substrate.historical.d_ladder` and belong to **p12**
(plan §p12: "`analysis/d1_probe` and `analysis/ladder_probe` to `historical/d_ladder`"). p11
therefore does NOT empty the package, the p3 emptied-facade precondition is not met, and the
retraction rides p12, whose postcondition is an EMPTY `__legacy`. The reference form, for p12's use,
is `src/entropic_substrate/__legacy/data/__init__.py` (1 line, a single docstring beginning
`"""Retracted facade body (p3-quarantine, data): …`).

**Which phase retires `__legacy/evidence/__init__.py`? MEASURED: not p11, and not p8 either.** p11
moves **10** `__legacy/evidence/*` leaves (the 9 dest-`ladder.file` ones plus
`d1_review_adjudication`, dest `ladder.review`), so ruling FACADE-EMPTY has to be evaluated for that
package as well. Counted at `79d8228` by intersecting the on-disk leaves with the three w2 lane target
lists in `lane_slices_cmd.sh`:

| bucket                                                            | count  |
| ----------------------------------------------------------------- | ------ |
| `__legacy/evidence/*.py` leaves on disk (excluding `__init__.py`) | **48** |
| owned by p8-capture                                               | 37     |
| owned by p11-analyze-review-file                                  | 10     |
| owned by p9-run-lower                                             | 0      |
| **remaining after all three lanes**                               | **1**  |

The survivor is `__legacy/evidence/ladder_campaign.py` — the gate half p5b left quarantined
(`landed: false` in `p5b-ladder-campaign.json`), mapping dest `entropic_substrate.kernel.artifacts`,
and a target of no w2 lane. Therefore the `evidence` package is NON-EMPTY after commit 3 and after
the whole w2 zip, `__legacy/evidence/__init__.py` stays UNTOUCHED by p11 (and by p8), and its
retraction falls to the phase that lands that gate half — at the latest **p12**, whose postcondition
is an EMPTY `__legacy`. This is also why both `ladder_analysis.py:29` and `ladder_filing.py:22` keep
an explicit `__legacy.evidence.ladder_campaign` edge after their splits.

### Rewrite-on-arrival: the 47 sources' own imports

MEASURED total **280** `entropic_substrate.*` import lines (`own_import_counts` in the lane pack).
The buckets, as this brief classifies them — the pack's own bucketing differs by ONE line, see the
reclassification retraction below: intra-set **104** · landed **118** (= 117 pack-measured + 1
reclassified) · sibling→p8 **53** · sibling→p9 **1** · other-quarantined **4**. Sum:
104 + 118 + 53 + 1 + 4 = **280**. Each bucket has
its own rule.

**Intra-set (104 lines).** `entropic_substrate.analysis.<leaf>` → `entropic_substrate.ladder.<analyze|review|file>.<leaf>`,
`entropic_substrate.evidence.<leaf>` → `.ladder.<…>.<leaf>`, `entropic_substrate.d1_inspect_*` /
`d1_filing_application` → their `ladder.*` addresses, **for the 47 targets only**, each to the dest
this brief's *Scope* lists for it. Note the cross-package cases inside the set: e.g.
`d1_analysis_api.py:9` (review → analyze), `d1_real_arm_verdict.py:22,23,27,28,42` (file → file),
`d1_filing_edges.py:13,16` (file → analyze), `d1_external_analysis_control.py:27` (review → file).
Full `file:line` list: `own_imports.intra` in the lane pack, reproduced by the slice command.

**Landed (118 lines = 117 pack-measured + 1 reclassified).** One row per distinct destination address; `file:line` cites are complete.

| becomes                                                                         | lines                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kernel.identity.d1_primitives` (from `experiments.d1_primitives`)              | `d1_analysis_api.py:42` · `d1_blind_aliases.py:15` · `d1_capture.py:40` · `d1_clause_policy.py:24` · `d1_diagnostics.py:15` · `d1_execution_receipt.py:37` · `d1_external_analysis_adapter.py:32` · `d1_external_analysis_codec.py:22` · `d1_external_analysis_control.py:32` · `d1_external_review_adapter.py:26` · `d1_factorial.py:28` · `d1_machine_analysis.py:21` · `d1_machine_analysis_binding.py:38` · `d1_machine_analysis_verification.py:29` · `d1_real_arm_verdict.py:44` · `d1_review_execution.py:27` · `d1_review_packet.py:34` · `ladder_analysis.py:45` · `ladder_registration.py:29,382,438` · `d1_inspect_codec.py:38` · `d1_inspect_projection.py:37` · `d1_inspect_request.py:10` · `d1_inspect_values.py:22` · `evidence/d1_filing_edges.py:43` · `evidence/d1_filing_graph.py:32` |
| `kernel.verdict.d1_contract` (from `evidence.d1_contract`)                      | `d1_analysis_api.py:41` · `d1_capture.py:34` · `d1_clause_policy.py:17` · `d1_diagnostics.py:14` · `d1_external_analysis_control.py:26` · `d1_factorial.py:21` · `d1_machine_analysis.py:19` · `d1_machine_analysis_binding.py:27` · `d1_real_arm_verdict.py:21` · `d1_review_packet.py:27` · `evidence/d1_decision_validation.py:9` · `evidence/d1_filing.py:12` · `evidence/d1_filing_graph.py:19` · `evidence/d1_validator.py:10`                                                                                                                                                                                                                                                                                                                                                                      |
| `kernel.verdict.d1_types` (from `evidence.d1_types`)                            | `d1_blind_aliases.py:14` · `d1_clause_policy.py:23` · `d1_external_analysis_control.py:31` · `d1_factorial.py:27` · `d1_machine_analysis_binding.py:37` · `d1_machine_analysis_codec.py:13` · `d1_real_arm_verdict.py:38` · `d1_review_packet.py:33` · `d1_filing_application.py:59` · `evidence/d1_decision_validation.py:16` · `evidence/d1_filing_edges.py:36` · `evidence/d1_gate.py:6` · `evidence/d1_process_filing.py:12` · `evidence/d1_validator.py:30`                                                                                                                                                                                                                                                                                                                                          |
| `kernel.verdict.d1_statistical_gaps`                                            | `d1_clause_policy.py:18` · `d1_factorial.py:26` · `d1_machine_analysis.py:20` · `d1_machine_analysis_binding.py:32` · `d1_machine_analysis_codec.py:12` · `d1_real_arm_verdict.py:33` · `evidence/d1_validator.py:29`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `kernel.verdict.d1_evaluation_values` (from `d1_evaluation_values`)             | `d1_capture.py:20` · `d1_factorial.py:20` · `d1_inspect_values.py:11`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `kernel.verdict.d1_parsing`                                                     | `evidence/d1_decision_validation.py:10` · `evidence/d1_validator.py:15`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `kernel.lifecycle` (from `evidence.lifecycle`)                                  | `d1_real_arm_verdict.py:43` · `ladder_analysis.py:38` · `ladder_filing.py:34` · `d1_filing_application.py:64` · `evidence/d1_decision_validation.py:28` · `evidence/d1_filing.py:14` · `evidence/d1_filing_edges.py:42` · `evidence/d1_filing_graph.py:31` · `evidence/d1_gate.py:12` · `evidence/d1_process_filing.py:17` · `evidence/d1_validator.py:54`                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `kernel.identity.factory_attestation`                                           | `d1_real_arm_verdict.py:49` · `d1_review_execution.py:28` · `d1_review_packet.py:40` · `evidence/d1_review_adjudication.py:14`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `kernel.artifacts` (from `d1_artifact_snapshot`)                                | `ladder_analysis.py:23` · `ladder_filing.py:21` · `d1_filing_application.py:28`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `kernel.evidence_class.resource_provenance{,_codec}`                            | `d1_inspect_values.py:19` (`resource_provenance`) · `d1_inspect_codec.py:28` (`…_codec`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `kernel.runtime.checkpoint` (2)                                                 | `embedding_geometry.py:49` · `field_head_audit.py:54`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `kernel.runtime.config` (1)                                                     | `embedding_geometry.py:50`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `kernel.runtime.logging` (2)                                                    | `embedding_geometry.py:55` · `mlflow_attach.py:19`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `core.geometry.{kappa,gram}`                                                    | `charts.py:40` (kappa) · `embedding_geometry.py:53` (gram)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `core.substrate.{fisher,backbone,field_encoder}`                                | `embedding_geometry.py:54,56` · `field_head_audit.py:55`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `core.corpus.{batching,ladder_source_order}`                                    | `embedding_geometry.py:51` · `ladder_analysis.py:24,25` · `ladder_registration.py:27`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ladder.register.d1_registered_design` (p7)                                     | `d1_analysis_api.py:43` · `d1_capture.py:41` · `d1_diagnostics.py:16` · `d1_external_analysis_codec.py:23` · `d1_external_analysis_control.py:33` · `d1_factorial.py:29` · `d1_machine_analysis.py:22` · `d1_machine_analysis_binding.py:39` · `d1_review_packet.py:39`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `ladder.register.{d1_config,d1_family,d1_registration_codec,d1_execution}` (p7) | `d1_external_worker.py:30,31,32` · `d1_external_analysis_codec.py:24` · `d1_external_analysis_control.py:37` · `ladder_registration.py:381` · `d1_inspect_values.py:21`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **`core.observation.contracts`** (`CodecIdentity`)                              | `codec_equivariance.py:21` — **RECLASSIFIED, see the retraction below**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

**RETRACTION (house rule 4), pack classification.** The lane pack lists
`codec_equivariance.py:21 entropic_substrate.observation` under `other_quarantined`, i.e. as a line
that stays on an explicit `__legacy` address. That is WRONG, and it is the same misclassification p7
retracted for `d1_family.py:43`: the pack buckets by mapping row, and the `observation` PACKAGE key
is never marked landed, but the NAME it imports is. MEASURED at `79d8228`: the line imports
`CodecIdentity`, which lives at `src/entropic_substrate/core/observation/contracts.py:27` (p6
landed it, and it is in that module's `__all__` at `:18`), while
`src/entropic_substrate/__legacy/observation/__init__.py` is the p3 retracted one-liner with **zero
re-exports** (`cat` output). Writing `from entropic_substrate.__legacy.observation import
CodecIdentity` would fail at import. The line rewrites to
`entropic_substrate.core.observation.contracts`. Consequence, carried consistently through this
brief: landed **118** (117 pack-measured + 1 reclassified) and other-quarantined **4**, never
117 / 5; the 280 total is unchanged.

**Sibling-target imports (54 lines) — rewrite to the explicit `__legacy` address, forwarder-served.**
53 lines reach a p8 target, 1 reaches a p9 target. Every one becomes
`entropic_substrate.__legacy.<same suffix>`, a real file in this worktree; the rung rewrites them to
the final `ladder.capture.*` / `ladder.run.*` addresses after the zip.

*→ p9 (1 line):* `analysis/d1_external_review_adapter.py:21` `entropic_substrate.d1_command_runner`
→ `entropic_substrate.__legacy.d1_command_runner` (mapping dest `ladder.run`).

*→ p8 (53 lines), grouped by the address each becomes (all prefixed `entropic_substrate.__legacy.`):*

| becomes `__legacy.…`                            | lines                                                                                                                                                                                                                      |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `telemetry`                                     | `charts.py:41` · `d1_capture.py:11` · `ladder_analysis.py:94` · `ladder_registration.py:383,439` · `series.py:20`                                                                                                          |
| `evidence.d1_capture_aggregate`                 | `d1_analysis_api.py:40` · `d1_capture.py:21` · `d1_external_analysis_adapter.py:31` · `d1_machine_analysis_verification.py:26` · `d1_review_packet.py:16` · `evidence/d1_filing_graph.py:10`                               |
| `evidence.d1_capture_artifacts`                 | `d1_capture.py:25` · `d1_execution_receipt.py:36` · `d1_external_analysis_control.py:21` · `d1_review_packet.py:17` · `ladder_analysis.py:26` · `ladder_registration.py:437`                                               |
| `evidence.d1_capture_bundle`                    | `d1_capture.py:219` · `ladder_analysis.py:27` · `evidence/d1_filing_graph.py:14`                                                                                                                                           |
| `evidence.d1_capture_roster`                    | `d1_capture.py:30,260` · `d1_review_packet.py:18` · `evidence/d1_filing_graph.py:15`                                                                                                                                       |
| `evidence.d1_telemetry_values`                  | `d1_capture.py:35` · `d1_clause_policy.py:19` · `d1_diagnostics.py:155,171` · `d1_review_packet.py:32` · `ladder_analysis.py:28` · `ladder_registration.py:28` · `d1_inspect_projection.py:30` · `d1_inspect_values.py:14` |
| `evidence.d1_capture_aggregate_codec`           | `d1_external_analysis_control.py:17` · `d1_external_worker.py:27`                                                                                                                                                          |
| `evidence.d1_capture_ports`                     | `d1_external_analysis_control.py:25`                                                                                                                                                                                       |
| `evidence.d1_capture_validation`                | `d1_review_packet.py:22`                                                                                                                                                                                                   |
| `evidence.resource_work{,_codec}`               | `d1_inspect_values.py:20` (`resource_work`) · `ladder_analysis.py:44` · `d1_inspect_codec.py:33` (`…_codec`)                                                                                                               |
| `evidence.d1_journal_values`                    | `d1_inspect_codec.py:27` · `d1_inspect_projection.py:29` · `d1_inspect_values.py:13`                                                                                                                                       |
| `evidence.d1_process_manager`                   | `d1_filing_application.py:55` · `evidence/d1_process_filing.py:10`                                                                                                                                                         |
| `evidence.d1_{lifecycle_values,process_replay}` | `evidence/d1_process_filing.py:9,11`                                                                                                                                                                                       |
| `d1_numerical_defect`                           | `d1_inspect_codec.py:26` · `d1_inspect_values.py:12`                                                                                                                                                                       |
| `d1_journal_snapshot`                           | `d1_inspect_projection.py:28`                                                                                                                                                                                              |
| `mlflow_aggregate`                              | `mlflow_attach.py:20`                                                                                                                                                                                                      |
| `data.build`                                    | `embedding_geometry.py:52`                                                                                                                                                                                                 |

**Other-quarantined (4 lines after the retraction above) — ordinary quarantine edges, explicit
`__legacy` addresses, NOT forwarder-dependent. Those rows move in p10/p12; this lane does not
touch them.**

| line                         | old import                                               | becomes                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ladder_registration.py:33`  | `entropic_substrate.d1_excluded_capture`                 | `entropic_substrate.__legacy.d1_excluded_capture` (mapping dest `ladder.run`, p10's closure 6)                                                                                                                                                                                                                                                                                                         |
| `ladder_registration.py:380` | `entropic_substrate.d1_excluded_capture`                 | same address (function-local import of the same module — one EDGE, two lines)                                                                                                                                                                                                                                                                                                                          |
| `ladder_analysis.py:29`      | `entropic_substrate.evidence.ladder_campaign` (7 names)  | **SPLITS**: `artifact_from_mapping`, `artifact_mapping`, `object_fields`, `read_bound_artifact`, `read_json_artifact` → `entropic_substrate.kernel.artifacts` (MEASURED: all five in its `__all__`, `kernel/artifacts.py:15-28`); `actor_from_mapping`, `read_campaign_design` → `entropic_substrate.__legacy.evidence.ladder_campaign` (the gate half, `landed: false` in `p5b-ladder-campaign.json`) |
| `ladder_filing.py:22`        | `entropic_substrate.evidence.ladder_campaign` (10 names) | **SPLITS**: the same five → `kernel.artifacts`; `_TOKEN`, `FiledLadderGate`, `actor_from_mapping`, `read_campaign_design`, `validate_ladder_phase_transition` → `__legacy.evidence.ladder_campaign` (MEASURED: none of the five is in `kernel/artifacts.py`'s `__all__`)                                                                                                                               |

## `legacy_edge_budget` — mechanism, baseline, prediction

**Mechanism, MEASURED, and NOT what the wave audit says.** `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget`
asserts `measured == load_committed_budget(...)`, and `load_committed_budget` (same file, `:72-81`)
calls `gen_shims.highest_ordered_fragment(fragments, sequence)` against
`.claude/workflows/sprint-manifest.json`'s `sequence` — **the highest-ordered fragment's own number,
never a sum.** RETRACTION (house rule 4): the w2 audit's "measured == sum(committed)" and the
launch-brief wording "equality-asserted against the SUM of fragments" are both WRONG. The lens
check: the seven present fragments sum to **237** while the tree measures **45**, which is exactly
`p7-register-gate.json`'s committed 45.

```text
python3 -c "import sys;sys.path.insert(0,'tools');from pathlib import Path;import import_graph;…"
# MEASURED landed->__legacy edges at 79d8228: 45
```

**BUDGET-MONOTONE strict-decrease is RETRACTED** (`795f8e6`; p7's fragment note reads "Direction is
INCREASING, as predicted"). v1's ACCEPTANCE "legacy_edge_budget < p7's" and its guard-teeth row
"STRICTLY LESS than the previous phase's" are struck, not overwritten. The binding arm is EQUALITY
against a MEASURED number. Direction for p11 is INCREASING, steeply.

**`[predicted]` arithmetic (conductor).** Edges are distinct `(landed module, __legacy module)`
pairs, which is what `tools/import_graph.py` counts.

| term                                                                              | value  | basis                                                                                                                                                       |
| --------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| baseline at `79d8228`                                                             | 45     | MEASURED with `tools/import_graph.py` (= p7's committed 45, lens check passed)                                                                              |
| − removed: `ladder/gate/d1_verdict.py:11,18,22,23` rewritten onto `ladder.file.*` | −4     | MEASURED: 4 of the tip's 45 edges originate in `ladder.gate.d1_verdict`                                                                                     |
| + added: sibling → p8                                                             | +50    | MEASURED distinct pairs (`depends_out.p8-capture`); 53 LINES collapse to 50 pairs                                                                           |
| + added: sibling → p9                                                             | +1     | MEASURED (`analysis.d1_external_review_adapter → d1_command_runner`)                                                                                        |
| + added: other-quarantined                                                        | +3     | MEASURED: `ladder_analysis→ladder_campaign`, `ladder_filing→ladder_campaign`, `ladder_registration→d1_excluded_capture` (`:33`/`:380` collapse to one pair) |
| `ladder/gate/ladder_predecessor.py`'s 2 p11 edges                                 | 0      | RUNG-OWNED, untouched, forwarder-served, STILL COUNTED at both ends                                                                                         |
| **`[predicted]` p11 fragment budget in the lane worktree**                        | **95** | 45 − 4 + 50 + 1 + 3                                                                                                                                         |

The lane MEASURES its own number with `tools/import_graph.py` and reproduces nothing from this
paragraph.

**ZIP CONSEQUENCE, stated so nobody "fixes" a red gate by editing a fragment.** The manifest
`sequence` is `… p7-register-gate, p8-capture, p9-run-lower, p11-analyze-review-file, p10-run-upper …`,
so **p11 is the highest-ordered w2 fragment**. In the lane worktree the arm reads p11's number
against a tree carrying p11's changes alone (predicted 95). At the zip the SAME arm reads p11's
number against the INTEGRATED tree (p8 + p9 + p11) and will be RED unless the tip is re-measured.
The gap is wave-level, not p11's: **only p8's restack matches by construction** (tree p7+p8,
highest-ordered present p8); at the **p9 and p11 restack tips** the highest-ordered fragment was
measured against a smaller tree. The `zipRestackHook` as landed (`14c587a`: generate + `--verify` +
commit forwarders) does NOT re-pin a budget. Re-pinning `legacy_edge_budget` on each restacked tip is
the **zip / reconciliation rung's** obligation, not a lane DECISION and not a lane edit.

## R-PIN: not owed

- MEASURED: this lane rewrites exactly ONE landed importer, `ladder/gate/d1_verdict.py`, and it is
  **not** a reader-set member. `h-checkpoint-reader-v2`'s 51-module tuple
  (`src/entropic_substrate/ladder/run/checkpoint.py:57-109`) and `h-source-reader-v3`'s 7-module
  tuple (`src/entropic_substrate/ladder/capture/capture.py:29-37`) contain **no `ladder.gate.*`
  address and no p11 target under its old name** — MEASURED, both halves, with
  `grep -nE "analysis\.|d1_inspect|d1_filing|evidence\.d1_(gate|validator|decision|process_filing|review_adj)" src/entropic_substrate/ladder/run/checkpoint.py src/entropic_substrate/ladder/capture/capture.py`
  → ZERO matches. Neither `checkpoint.py` nor `capture.py` is in this lane's slice.
- **The CONTROL is an assertion, not an omission** (p7 form): at every commit,
  `git diff mythwave/p7-register-gate -- tests/test_identity_epoch.py tools/regime_rows.py tests/test_regime_rows.py docs/identity/read-old-table.md`
  is EMPTY — both `_CHECKPOINT_REGIME_TABLE` and `_SOURCE_REGIME_TABLE` byte-identical to the parent
  tip — and `tests/test_identity_epoch.py` + `tests/test_regime_rows.py` are green unchanged.
  Command output in the PR body.
- Those four identity files therefore **LEAVE Allowed files** (p7 form). A diff touching any of them
  is a `DECISION`, not an edit. They remain `zip_owned_regenerable` for the wave's OTHER lanes
  (p8 and p9 both re-pin); that is their business, not this lane's.

## The "isolation ring in review/" — what it actually is, measured

The plan's §p11 row reads "The container / ACL / hermetic isolation ring lands in `review/isolation`,
not `analyze` (proposal C's named deviation)". MEASURED at `79d8228`, what that resolves to:

- **No `ladder.review.isolation` destination exists.** `tools/mapping.json`'s dests under these
  three packages are exactly `entropic_substrate.ladder.{analyze,review,file}`, flat — command:
  `python3 -c "import json; …sorted dest set…"`. `tools/mapping.json` is read-only until p13, so a
  `review/isolation/` subpackage cannot be created by this lane without amending a frozen contract.
- **The ring's parts, and they are already all dest `ladder.review`:**
  `D1IsolationRequest` is DEFINED in `analysis/d1_execution_receipt.py` and referenced by
  `analysis/{d1_analysis_api, d1_external_analysis_adapter, d1_external_worker}.py`
  (`grep -rn "D1IsolationRequest" src/entropic_substrate/__legacy/` → those four files only);
  the container-boundary surface is `analysis/d1_external_analysis_adapter.py` ("Host-side mount
  identities; container destinations and controls stay fixed." `:91`, tmpfs mount option literal at
  `:219`, the one-way-door comment at `:188-190`); the bounded one-attempt subprocess is
  `analysis/d1_external_review_adapter.py` ("Hardened one-attempt subprocess adapter for D1 blind
  review." `:1`) which reaches `entropic_substrate.d1_command_runner` — a **p9** target — at `:21`.
- **Ruling for this lane: the ring is a p13+ concern.** p11 lands these files FLAT in
  `ladder/review/` at the addresses *Scope* enumerates, and records the measured edge list above as
  the ring's current shape. It creates no `isolation/` subpackage and asserts no isolation contract.
  Consequence: **v1's planned commit branch 3, "Land ladder/review/isolation as the named
  deviation", is UNEXECUTABLE and is dropped**, not silently reinterpreted.

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1): the ONE behaviour-change class permitted inside a
structural phase, scoped to log routing, metric routing and entry-point configuration. Measured
scientific behaviour stays frozen; the identity fixtures and the numerics tests are the tripwire.
The instrument is `tests/test_observability_conformance.py` (p3b, frozen until p13), whose
`LANDED_LAYER_PREFIXES` includes `entropic_substrate.ladder` — MEASURED, so all three arms cover
`ladder.{analyze,review,file}` the moment these files land. The per-phase sets are promoted through
THIS lane's own fragment keys; `tests/observability_conformance_config.py` is p3b's file and is NOT
in this envelope.

- **Bare `print(` in the 47 sources: exactly 2.** MEASURED with a per-file `count("print(")` over
  the 47 legacy sources (total 7 occurrences) followed by `grep -n "print("` on the 5 files that
  matched: five of the seven are `_fingerprint(`/`_reveal_fingerprint(` DEFINITIONS, not calls. The
  two real sites are both in `analysis/d1_external_worker.py`, inside `main()`'s error path:
  `:363` `print(str(error), file=sys.stderr)` and `:366`
  `print("D1 external worker refused [worker_boundary]", file=sys.stderr)`.
  RETRACTION (house rule 4): v1 asserted "ZERO bare `print(` may survive in the 49 moved modules"
  and "Neither of the two pinned raw-`print` sites lives in this phase" with no command. The first
  is an obligation, not a measurement, and the second is beside the point — these are two NEW sites.
  Retracted, not overwritten.
- **Route both to `logger`** (rule: `print(` → logger except pinned sites). MEASURED premise for
  choosing routing over an allowlist entry: nothing pins the stderr text —
  `grep -rn "worker_boundary\|D1 external worker refused" tests/` returns ZERO matches — so routing
  breaks no existing test (Q1 ruled (b) forbids editing one). And p4's D-A allowlist amendment does
  NOT reach here: it is scoped to "machine-consumed stdout sites (JSON result envelopes with a named
  parsing consumer)", while these are stderr diagnostics. If the lane finds a consumer that parses
  this stderr, that is a `DECISION` carrying the evidence, and the fallback is a `print_allowlist`
  entry keyed to the LANDED dotted name `entropic_substrate.ladder.review.d1_external_worker`.
- **Entry-point promotion.** `entropic_substrate.analysis.d1_external_worker` is one of the seven
  `entrypoint: true` mapping rows (MEASURED). This lane promotes it through its own fragment key
  `conforming_entrypoints`, keyed by the **PRE-MOVE** dotted name (KEYING ruling, conductor
  2026-09-21; `print_allowlist` entries are the only ones re-keyed to the landed name). Arm 2
  requires the entry point's `main()` to call `configure_logging` as its FIRST statement. MEASURED:
  `analysis/d1_external_worker.py` has `def main(` at `:343`, `argparse` at `:5`, `if __name__ ==`
  at `:370`, and **no `configure_logging` anywhere** — so the promotion REQUIRES adding that call.
  The form is p4's: `from entropic_substrate.kernel.runtime.logging import configure_logging`, called
  as the first statement (see `src/entropic_substrate/ladder/run/worker_command.py:24,252`).
- **No other entry point and no sink.** MEASURED across all 47 sources: the only other `argparse`
  reference is `analysis/d1_external_analysis_adapter.py` (no `main()`, no `__main__` guard), and
  ZERO files reference `MetricSink` or `logging.getLogger`. No `sink_construction_modules` entry is
  claimed; arm 3's subject set is empty here.
- **Charts keep writing files, not stdout.** The Okabe-Ito chart lane's colour contract is pinned by
  `tests/test_analysis_palette.py` and `tests/test_analysis_charts.py` (named, not alluded to — see
  *Guard teeth*).

## ACCEPTANCE

Machine-checkable, one line each, every one naming its instrument. The lane's `acceptance` arg
points here.

- The 47 modules exist at the destination addresses enumerated under *Scope* (analyze 22, review 13,
  file 12), each shim pair regenerated by `tools/gen_shims.py` with `--verify` byte-comparing a
  fresh re-derivation; every shim's `_target` is the `ladder.{analyze,review,file}.*` address, never
  `__legacy`. Instrument: `tools/gen_shims.py --verify` + `tests/test_gen_shims.py`.
- **`tests/test_module_identity.py` green at EVERY commit, not only at the tip** (FRAGMENT-PER-COMMIT,
  see *Planned commit branches*): its `canonicalized == _FROZEN_SENTINEL_MAP` assertion (`:228-229`)
  reverse-maps six p11 targets' factory sentinels through the fragments' `target` field, so a landing
  commit without its fragment rows goes RED. Instrument:
  `PYTHONPATH=src uv run pytest -q tests/test_module_identity.py -m unit`, run at each commit.
- `git diff -M --stat` shows every moved module as a pure rename plus import-line rewrites only —
  **with two declared exceptions:** `__legacy/analysis/ladder_analysis.py` and
  `__legacy/analysis/ladder_filing.py`, whose old paths are REWRITTEN INTO FORWARDERS by
  `gen_shims` rather than deleted, so rename detection does not apply to them. No verdict
  arithmetic, no alias/commitment semantics, no digest literal and no schema constant changes bytes.
- `ladder/{analyze,review,file}/__init__.py` are docstring-only with ZERO re-exports;
  and `git diff mythwave/p7-register-gate -- src/entropic_substrate/analysis/__init__.py src/entropic_substrate/__legacy/analysis/__init__.py`
  is EMPTY (both facades untouched; the retraction is p12's, ruling P11-FACADE).
- **Cold import at the NEW address, fresh interpreter per moved leaf**:
  `tests/test_analyze_review_file_cold_import.py` (NEW, RED-first; the subprocess-plus-stdout-marker
  instrument of `tests/test_shim_layer.py`, p6's `tests/test_core_cold_import.py` and p7's
  `tests/test_register_gate_cold_import.py` — never exit code alone). Each of the 47 destination
  addresses imports cleanly with NOTHING warming the chain first. This arm is also the load-bearing
  guard for the `codec_equivariance.py:21` reclassification: the `__legacy.observation` address
  exports nothing, so a literal reading of the lane pack fails here.
- The conformance instrument is green with a NON-EMPTY subject set:
  `tests/test_observability_conformance.py` arm 1 (zero bare `print(` under a landed package outside
  `PRINT_ALLOWLIST` — the two `d1_external_worker` sites routed to `logger`) and arm 2
  (`configure_logging` first in the promoted entry point's `main()`, with
  `entropic_substrate.analysis.d1_external_worker` in this fragment's `conforming_entrypoints`).
  MEASURED premise for the PRE-MOVE keying being safe: arm 2 resolves the entry point's real source
  through `importlib.import_module(dotted).__file__`, never an assumed
  `src/entropic_substrate/<name>.py` path (`tests/test_observability_conformance.py:179-200`, with the
  architect constraint spelled out in its docstring), so the shim at the old path forwards to the
  landed file and the arm keeps working across the move with zero edit. Arm 2 also asserts the
  partition covers all seven `entrypoint: true` rows with ≥ 2 conforming (`:132-145`), so this
  promotion needs no companion edit: MEASURED,
  `NOT_YET_CONFORMED_ENTRYPOINTS = _all_entrypoints - _conforming_from_fragments`
  (`tests/observability_conformance_config.py:155`) is DERIVED, so adding the fragment key moves the
  row across the partition by itself.
- `legacy_edge_budget` in `.claude/workflows/landed-set/p11-analyze-review-file.json` equals the
  MEASURED landed→`__legacy` edge count at the lane tip, measured with `tools/import_graph.py` and
  never typed. Instrument:
  `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget`
  (EQUALITY arm; the strict-decrease arm is RETRACTED). DIRECTION: INCREASING, `[predicted]` 95.
- `tests/test_no_old_prefix.py` green: ZERO landed→old-prefix edges, i.e. no moved module keeps an
  `entropic_substrate.<pkg>.<leaf>` import and no sibling-target line was left on its old name.
- **Regime-table control (no R-PIN owed):**
  `git diff mythwave/p7-register-gate -- tests/test_identity_epoch.py tools/regime_rows.py tests/test_regime_rows.py docs/identity/read-old-table.md`
  is EMPTY at every commit, and `tests/test_identity_epoch.py`, `tests/test_regime_rows.py` are
  green unchanged. Zero of the 51 checkpoint-reader members and zero of the 7 source-reader members
  are touched. Command output in the PR body.
- `lint-imports` green: ZERO inversions among `kernel < core < ladder < historical < composition`
  (contract `[importlinter:contract:landed-layers]`, `.importlinter:32`).
  `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not
  linted; `tests/test_legacy_coupling.py` bounds them instead. MEASURED premise: every
  non-`__legacy` import in the 47 targets resolves to `kernel.*`, `core.*` or `ladder.register.*` —
  at or below `ladder` — so no inversion is possible by construction.
- The colour and blind-review contracts pass UNCHANGED, named:
  `tests/test_analysis_palette.py`, `tests/test_analysis_charts.py`,
  `tests/test_d1_blind_alias_custody.py`, `tests/test_d1_blind_review_execution.py`. Baseline for the
  first and third: **"7 passed"** from MY OWN targeted run at `79d8228` —
  `PYTHONPATH=src uv run pytest -q tests/test_analysis_palette.py tests/test_d1_blind_alias_custody.py -m unit -p no:cacheprovider`.
- `just check` and `just pc-ci` (exit gate via the runner) green on the lane tip. NO live gate, NO
  container, NO publish.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the
  PR body.
- The actual diff's file set is a subset of *Allowed files*, and the diff touches
  `ladder/gate/ladder_predecessor.py` ZERO times. Undeclared overlap is a HALT at the zip, not a note.
- Verdict **CERTIFIED** (tip SHA + exit-gate JSON). The lane does NOT publish.
- The lane branch restacks onto its zip parent with ZERO conflicts, the `zipRestackHook` runs FIRST
  at the restacked tip (`gen_shims` generate + `--verify` + commit forwarders), and the exit gate
  re-runs there: a restacked tree is a new tree, and the pre-zip certificate speaks for a tree that
  no longer exists.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N-1 and raises
  DECISION-NEEDED naming the lane.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for
correctness is a `DECISION`, not an edit.

- Does NOT create `ladder/review/isolation/` or assert any isolation contract (see *the isolation
  ring*). Does NOT re-introduce a reader source closure (p2 retired it, ruling E-1).
- Does NOT retract ANY `__legacy` package facade — neither `__legacy/analysis/__init__.py` (ruling
  P11-FACADE: `d1_probe.py` / `ladder_probe.py` survive) nor `__legacy/evidence/__init__.py` (ruling
  FACADE-EMPTY: `ladder_campaign.py` survives all three w2 lanes). Both fall to p12 at the latest.
- Does NOT change alias commitment / reveal semantics, verdict arithmetic, clause policy thresholds,
  or any digest.
- Does NOT touch `core/**`, `kernel/**` (including `kernel/artifacts.py` — the two `ladder_campaign`
  splits CONSUME p5b's five names), `ladder/register/**`, or `ladder/capture/**`.
- Does NOT touch `src/entropic_substrate/ladder/gate/ladder_predecessor.py` (RUNG-OWNED).
- Does NOT move any p8, p9, p10 or p12 module (including `__legacy/analysis/{d1_probe,ladder_probe}.py`
  and `__legacy/evidence/ladder_campaign.py`), and does NOT rewrite a sibling lane's importer.
- Does NOT re-pin any regime row, bump an identity LABEL or change a module SET.
- Does NOT edit `tools/mapping.json`, `.importlinter`, `tests/observability_conformance_config.py`,
  briefs, plan docs, the manifest, engine scripts or CI. Does NOT edit any EXISTING test.
- Does NOT publish, run a container, or run `just ship-it`.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this
envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the
verification you RAN attached (command + observed output). RETRACTED 2026-09-18 (Q1 ruled (b)): this
section previously granted "the analyze/review/file tests, for import-path rewrite only" as a prose
test grant. Struck — no phase before p13 edits EXISTING tests; shims keep them green under
rewrite-on-arrival, and the prose form cannot discharge the subset-of-Allowed-files check anyway.

- the **47** legacy sources enumerated under *Scope* (moved with `git mv`, import lines rewritten
  once on arrival per the *Rewrite-on-arrival* tables)
- `src/entropic_substrate/ladder/analyze/**` (22 moved modules + a docstring-only `__init__.py`)
- `src/entropic_substrate/ladder/review/**` (13 moved modules + a docstring-only `__init__.py`)
- `src/entropic_substrate/ladder/file/**` (12 moved modules + a docstring-only `__init__.py`)
- the **47 typed shim pairs** at the old paths (`.py` + `.pyi`), regenerated by
  `tools/gen_shims.py`, never hand-edited
- the **2 forwarders** `gen_shims` emits inside this worktree at
  `src/entropic_substrate/__legacy/analysis/{ladder_analysis,ladder_filing}.py` (+ `.pyi`) —
  generator output, never hand-edited
- **`src/entropic_substrate/ladder/gate/d1_verdict.py`** — this lane's ONE landed importer; its
  four `__legacy.evidence.*` lines (`:11,18,22,23`) rewrite onto `ladder.file.*`. ADDED here: the
  old manifest touch_set omitted it.
- `tests/test_analyze_review_file_cold_import.py` (NEW, RED-first: fresh interpreter per moved leaf
  at its NEW address)
- `.claude/workflows/landed-set/p11-analyze-review-file.json` — THIS phase's disjoint fragment (the
  47 rows `landed: true` with their `ladder.*` targets, the MEASURED `legacy_edge_budget`,
  `conforming_entrypoints: ["entropic_substrate.analysis.d1_external_worker"]`, and a `note`
  recording the retractions). No other phase's fragment. `.importlinter` is NOT in this envelope: it
  is edited only at p0 and p13.
- generator outputs (D-B precedent, p5b): any `.pyi` stub whose target this phase edits, listed in
  the PR body from `git status --porcelain` after generate + `--verify`. An import-line rewrite does
  not change a bound name, so each stub changes ONLY in the module path of its
  `from … import X as X` lines.
- `.claude/workflows/landed-set/.emitted.json` is UNTRACKED derived state (FOR-1539); never
  `git add` it, never treat it as a companion of an allowed file
- `.secrets.baseline` (STANDING RULE, conductor 2026-09-19): a mechanical companion of allowed
  content. Its diff may contain ONLY line renumbering, removal of orphaned entries, the
  `generated_at` stamp, and new `Hex High Entropy String` entries whose flagged lines are sha256 /
  git-SHA literals inside this phase's own Allowed files; the lane verifies that in the PR body.
  New digest literals in TEST files carry an inline `# pragma: allowlist secret`. Never a credential.

**Deliberately EXCLUDED** — a diff touching any of these is a `DECISION`, not an edit:
`src/entropic_substrate/ladder/gate/ladder_predecessor.py` (RUNG-OWNED); the four identity files
`tests/test_identity_epoch.py`, `tools/regime_rows.py`, `tests/test_regime_rows.py`,
`docs/identity/read-old-table.md` (no R-PIN owed, p7 form); `tests/observability_conformance_config.py`
(p3b's; this lane promotes through its own fragment); **both `analysis` facades** —
`src/entropic_substrate/analysis/__init__.py` and
`src/entropic_substrate/__legacy/evidence/__init__.py` (ruling FACADE-EMPTY: one leaf,
`ladder_campaign.py`, survives all three w2 lanes);
`src/entropic_substrate/__legacy/analysis/__init__.py` (ruling P11-FACADE: the retraction is p12's,
because `d1_probe.py` and `ladder_probe.py` stay); and `tests/test_shim_layer.py` — MEASURED at `79d8228`,
`grep -n "analysis" tests/test_shim_layer.py` returns ZERO matches, so p6's Q-1 collision (hard-coded
facade children) does not recur here and the p3-era rule "no phase before p13 edits existing tests"
holds unqualified.

**Manifest phantoms to DROP** (paths that do not exist; `gen_shims` never emits them and
`lane_slices.py` exits 2 on them): `src/entropic_substrate/__legacy/analysis.py`,
`src/entropic_substrate/analysis.{py,pyi}` (package-key shim paths), and
`analysis/d1_reader_source_closure.{py,pyi}` + `__legacy/analysis/d1_reader_source_closure.py`
(retired at p2).

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                           | status                                                                                                                                                                                                                        | ruling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-WAVES r3                   | OPERATOR 2026-09-23                                                                                                                                                                                                           | w2 stays PARALLEL: p8 ∥ p9 ∥ p11 as NESTED worktrees off `mythwave/p7-register-gate` (parent = that branch, NOT p6). `publishMode:'defer'`, terminal state CERTIFIED, `liveRequired:false`. Zip order p8 → p9 → p11 (p11 last: it receives zero inbound dependencies). Then ONE serial reconciliation rung (conductor) rewrites the sibling-target imports to their final addresses and retires the forwarders. Plan line 843 is the record.                                                                                                                                                                                                                                                                                                         |
| sibling-target imports       | OPERATOR 2026-09-23 (r3 option A)                                                                                                                                                                                             | Each of p11's 53 lines importing a p8 target and its 1 line importing a p9 target is rewritten to the explicit `entropic_substrate.__legacy.<x>` address, a real file in this worktree; `tools/gen_shims.py` emits forwarders there after the zip (FOR-1547, `e9e5897`); the rung rewrites them to `ladder.capture.*` / `ladder.run.*`. All 54 enumerated with `file:line` above. The other-quarantined lines stay on explicit `__legacy` addresses; those rows move in p10/p12.                                                                                                                                                                                                                                                                     |
| rung-owned                   | CONDUCTOR 2026-09-23                                                                                                                                                                                                          | `src/entropic_substrate/ladder/gate/ladder_predecessor.py` imports p11 targets at `:8,12` AND p8 targets at `:14,20`. Neither lane rewrites it. Declared `--rung-owned entropic_substrate/ladder/gate/ladder_predecessor.py`; NOT in Allowed files; the lane's own `gen_shims` run emits a forwarder for those two rows inside this worktree; the rung rewrites it once. Its 2 p11 edges stay in the budget.                                                                                                                                                                                                                                                                                                                                         |
| `d1_verdict.py` is p11's     | CONDUCTOR 2026-09-23                                                                                                                                                                                                          | `ladder/gate/d1_verdict.py` carries ONLY p11 targets, so p11 rewrites it and it enters Allowed files. The old manifest touch_set's omission of it is a defect, not a rule.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| FRAGMENT-PER-COMMIT          | CONDUCTOR 2026-09-23 (wave rule)                                                                                                                                                                                              | Every landing commit carries the fragment rows for exactly the modules it lands, because `tests/test_module_identity.py`'s `_canonical_sentinel_key` reverse-maps new dotted names through the fragments' `target` field and SIX frozen sentinels are p11 targets. The `legacy_edge_budget` is written once, at the last commit. The fragment is never a tail-only commit.                                                                                                                                                                                                                                                                                                                                                                           |
| commit plan (M5)             | CONDUCTOR 2026-09-23                                                                                                                                                                                                          | Exactly FIVE commit branches, one per Linear issue **FOR-1478 … FOR-1482**, mapped 1:1 in *Planned commit branches*. Never a silent sixth.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| target set                   | mapping.json (ratified) + MEASURED 2026-09-23                                                                                                                                                                                 | 47 leaves = `dest ∈ {ladder.analyze, ladder.review, ladder.file}` ∧ leaf `.py` under `__legacy/` ∧ not already landed; split analyze 22 / review 13 / file 12. `entropic_substrate.analysis` is a PACKAGE KEY (never a fragment row) and `analysis.d1_reader_source_closure` was RETIRED at p2 (`8cf5aae`, no file on disk). v1's "49" and "analyze (24)" are retracted, not overwritten.                                                                                                                                                                                                                                                                                                                                                            |
| BUDGET-MONOTONE retracted    | CONDUCTOR 2026-09-23 (landed `795f8e6`)                                                                                                                                                                                       | The strict-decrease arm is RETRACTED. The binding arm is EQUALITY against a MEASURED count, read from the HIGHEST-ORDERED fragment per the manifest `sequence` (`test_legacy_coupling.py:72-81`), never a sum. Direction for p11 is INCREASING; re-pinning on the integrated tip is the zip's obligation.                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| R-PIN (NOT OWED HERE)        | CONDUCTOR 2026-09-22 (p5/p5b/p6/p7 precedent)                                                                                                                                                                                 | MEASURED: zero reader-set members among p11's targets and its one importer. **No re-pin is owed.** The four identity files stay OUT of Allowed files; the control is a byte-identity assertion against `mythwave/p7-register-gate`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ZIP-REMEASURE                | CONDUCTOR 2026-09-23: the budget equality arm binds to the highest-ordered fragment; the zip's `zipRestackHook` re-measures at every restacked tip and re-pins that fragment's number with a dated note (plan rulings table). |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| FACADE-EMPTY                 | CONDUCTOR 2026-09-23 (wave-wide)                                                                                                                                                                                              | A `__legacy/<pkg>/__init__.py` facade is retracted to the p3 "Retracted facade body" one-liner ONLY in the commit after which that package holds NO leaf module. P11-FACADE is this rule's `analysis` instance. Its `evidence` instance binds p11 too: p11 moves TEN `__legacy/evidence/*` leaves, so the rule must be evaluated there as well — MEASURED result under *Which phase retires `__legacy/evidence/__init__.py`*, and the answer is NOT p11.                                                                                                                                                                                                                                                                                             |
| P11-FACADE                   | CONDUCTOR 2026-09-23                                                                                                                                                                                                          | The `__legacy/analysis/__init__.py` facade retraction is **DEFERRED to p12** and `__legacy/analysis/__init__.py` LEAVES Allowed files. MEASURED: the package holds 34 leaves, 32 of them p11 targets; `d1_probe.py` and `ladder_probe.py` (dest `historical.d_ladder`) remain after p11, so p11 does not empty the package and the p3 emptied-facade precondition is not met. Both the quarantined facade and the old-path `src/entropic_substrate/analysis/__init__.py` stay UNTOUCHED, byte-identical to the parent tip. v2's commit branch 5 ("Retract `__legacy/analysis/__init__.py`") is DROPPED.                                                                                                                                              |
| isolation ring               | CONDUCTOR 2026-09-23 (amends the 2026-09-17 wording)                                                                                                                                                                          | No `ladder.review.isolation` dest exists and `tools/mapping.json` is read-only until p13. p11 lands the ring's modules FLAT in `ladder/review/` and RECORDS the measured edges; the subpackage and any isolation contract are p13+. v1's commit branch 3 is dropped.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| shim form                    | RULED 2026-09-17, CORRECTED 2026-09-23 (p7)                                                                                                                                                                                   | `import entropic_substrate.<new landed address> as _target` then `sys.modules[__name__] = _target`. v1's `entropic_substrate.__legacy.<x>` `_target` is the pre-p3 form and is WRONG for a phase that lands a destination; retracted, not overwritten. A string-only `importlib.import_module` shim is forbidden.                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| FOR-1548                     | PUBLISH AGENT 2026-09-23 (p7, retracted row)                                                                                                                                                                                  | A mispointed shim `_target` does **NOT** go RED under `gen_shims --verify` (AST shape only). This brief claims no such tooth.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `.importlinter` tooth struck | CONDUCTOR 2026-09-23                                                                                                                                                                                                          | MEASURED with `grep -n "^\[importlinter:contract" .importlinter`: THREE contracts — `:32` `landed-layers` (`type = layers`), `:42` `core-forbids-external-io` (`type = forbidden` at `:44`) and `:70` `mlflow-only-from-adapter` (`type = forbidden` at `:72`). Both `forbidden` contracts are about mlflow / `ladder.capture.adapters`; there is NO `ladder.analyze` contract, and the file is edited only at p0 and p13. v1's "give `ladder/analyze` a live-service import and the forbidden contract goes RED" is UNEXECUTABLE and is struck, not reinterpreted. (The arm-blind review cited `:41`/`:69` for the two `forbidden` headers; MEASURED they are `:42`/`:70`, an off-by-one, while the `type` lines it cited, `:44`/`:72`, are exact.) |
| FOR-1539                     | 2026-09-22                                                                                                                                                                                                                    | `gen_shims`' `.emitted.json` is untracked derived state; no phase commits it and it is never a companion of an allowed file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| O-1                          | RULED 2026-09-17                                                                                                                                                                                                              | Wave lane: ends CERTIFIED, never publishes, `liveRequired:false`; the zip publishes once through `just ship-it` and `/pr-stack-comment-triage` files into Linear after the wave publish.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| P-1                          | RULED 2026-09-17                                                                                                                                                                                                              | Observability is wired through DURING the refactor. Rail p3b plus the extended rewrite-on-arrival rule. The one permitted behaviour-change class inside a structural phase.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| p3b-promotion + KEYING       | 2026-09-21 / 2026-09-22 (D-A)                                                                                                                                                                                                 | The conformance config DERIVES its sets from fragment keys; each phase promotes through its OWN fragment. `conforming_entrypoints` keyed by the PRE-MOVE mapping name; only `print_allowlist` is re-keyed to the LANDED name. D-A's stdout allowance covers machine-consumed JSON envelopes with a named parsing consumer — it does not reach stderr diagnostics.                                                                                                                                                                                                                                                                                                                                                                                    |
| E-1                          | RULED 2026-09-17, EXTENDED 2026-09-20                                                                                                                                                                                         | The D1 reader source closure, the `reader_source_manifest_base64` envelope field, the receipt `readers` block and the filing graph's `analysis_reader` artifact are RETIRED at p2. Nothing here re-introduces them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| E-4                          | RULED 2026-09-17                                                                                                                                                                                                              | Edit the scripts, no forwarders, for dotted ENTRYPOINT names: they live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists. (Distinct from FOR-1547's import forwarders.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| rewrite-on-arrival           | RULED 2026-09-17                                                                                                                                                                                                              | A file's import lines are rewritten exactly once, by the phase that moves its target. This lane rewrites exactly one importer outside its own moved set: `ladder/gate/d1_verdict.py`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| skeleton                     | RULED 2026-09-17                                                                                                                                                                                                              | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical < composition`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes
RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached
it". Each mutation is performed and reverted in-branch; BOTH arms go in the PR body. Every row names
a test that EXISTS at `79d8228`, or is explicitly marked **EXHIBIT OR RETRACT**.

| mutation (RED arm)                                                                                                                   | must go RED (named instrument)                                                                                                                                                                                                                                                                                                                                                                                                                      | GREEN arm                                           |
| ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| leave one moved leaf importing a sibling target by its OLD dotted name (e.g. `entropic_substrate.analysis.d1_capture`)               | `tests/test_no_old_prefix.py` (landed→old-prefix edge count ≠ 0)                                                                                                                                                                                                                                                                                                                                                                                    | the same test green at the tip                      |
| drop one rewritten `__legacy` import, or point it at a name the quarantined module does not export                                   | `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget` (measured ≠ committed) **and** `tests/test_analyze_review_file_cold_import.py`. NOTE: the delta need not be one — several lines to the same module COLLAPSE into a single edge, so dropping one line may move the count by 0 or 1; the RED arm is the INEQUALITY, not a magnitude                                                                     | both green at the tip                               |
| land one leaf WITHOUT its fragment row (the FRAGMENT-PER-COMMIT hazard)                                                              | `tests/test_module_identity.py` (`canonicalized == _FROZEN_SENTINEL_MAP`, `:228-229`) for any of the six p11 sentinels at `:197`, `:198`, `:199`, `:200`, `:201`, `:210`                                                                                                                                                                                                                                                                            | green at every commit                               |
| write `codec_equivariance.py:21` as `from entropic_substrate.__legacy.observation import CodecIdentity` (the pack's literal reading) | `tests/test_analyze_review_file_cold_import.py` — MEASURED: `__legacy/observation/__init__.py` is the retracted one-liner with zero re-exports, so the import raises                                                                                                                                                                                                                                                                                | the `core.observation.contracts` form imports clean |
| reintroduce a bare `print(` in any landed p11 module without a `print_allowlist` entry                                               | `tests/test_observability_conformance.py` arm 1 (AST-based, `LANDED_LAYER_PREFIXES` includes `entropic_substrate.ladder` — MEASURED)                                                                                                                                                                                                                                                                                                                | arm 1 green with a non-empty subject set            |
| remove the `configure_logging` call from the promoted entry point's `main()` (or promote it in the fragment without adding the call) | `tests/test_observability_conformance.py` arm 2 (every CONFORMING entry point's `main()` calls `configure_logging` as its FIRST statement)                                                                                                                                                                                                                                                                                                          | arm 2 green with `d1_external_worker` promoted      |
| change one Okabe-Ito hex in `palette.py` (e.g. `#D55E00` → `#FF0000`)                                                                | `tests/test_analysis_palette.py::test_okabe_ito_matches_the_published_cud_hex_codes_exactly` (`:56`) and `::test_okabe_ito_has_no_red_or_green_pure_hues` (`:62`) — the ONLY load-bearing arms. `tests/test_analysis_charts.py:94` is DROPPED from this tooth: it IMPORTS `OKABE_ITO` from the mutated module, so it FOLLOWS the mutation instead of catching it.                                                                                   | **"7 passed"**, my own targeted run at `79d8228`    |
| perturb the alias COMMITMENT (weaken the salted commitment or the durability precondition)                                           | `tests/test_d1_blind_alias_custody.py::test_preparation_randomizes_all_four_aliases_and_commits_salted_reveal` (`:37`) and `::test_commitment_is_durable_before_private_packet_excluded_reveal` (`:64`)                                                                                                                                                                                                                                             | both green unchanged                                |
| perturb the REVEAL round trip (release a reveal without a durable review receipt, or substitute the payload)                         | `tests/test_d1_blind_review_execution.py::test_alias_reveal_requires_durable_review_receipt_and_opens_commitment` (`:213`) and `::test_alias_reveal_rejects_receipt_to_request_packet_substitution` (`:292`)                                                                                                                                                                                                                                        | both green unchanged                                |
| add a re-export to `ladder/{analyze,review,file}/__init__.py`                                                                        | **EXHIBIT OR RETRACT.** MEASURED: no mechanical test asserts destination-facade emptiness at `79d8228`, and `lint-imports` fires only if the re-export crosses a layer, which a same-layer re-export does not. The lane EXHIBITS a captured RED (named test or linter output) or RETRACTS this row in the PR body with the measurement (house rule 4, p7's SCC-row precedent). It must not report the mutation as proven from this brief's wording. | the three `__init__.py` docstring-only              |
| touch `ladder/gate/ladder_predecessor.py` or either `analysis/__init__.py` facade                                                    | **EXHIBIT OR RETRACT.** `tools/lane_slices.py` is a CONDUCTOR oracle run OUTSIDE the lane, not a diff test, and the Allowed-files subset check is a review arm, not a mechanical one. The lane either exhibits a mechanical RED it can run in-branch or records this explicitly as a review-arm-only guard.                                                                                                                                         | zero diff on those three paths                      |

NOT claimed here, and why: the mispointed-`_target` tooth (FOR-1548 measured it non-firing) and the
`.importlinter` live-service tooth (no such contract; p0/p13-only file).

## Planned commit branches

**EXACTLY FIVE**, one per Linear issue **FOR-1478 … FOR-1482** (ruled 2026-09-23, M5), inside this
phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR).
There is no sixth: the facade-retraction branch is dropped under P11-FACADE, and the fragment is NOT
a separate commit — it GROWS with every landing commit under the rule below.

**FRAGMENT-PER-COMMIT (wave rule, conductor 2026-09-23) — why the fragment cannot be a tail commit.**
`tests/test_module_identity.py` freezes 18 factory sentinels and canonicalizes discovered module
names through `_canonical_sentinel_key` (`:60-90`), whose step 2 reverse-maps a LANDED module's new
dotted name to its `tools/mapping.json` source key **via the landed-set fragments' `target` field**
(the `dest`-based form was retracted 2026-09-19 as unsound: `dest` is a destination PACKAGE shared by
up to 100 modules). MEASURED: **six** of the 18 frozen keys are p11 targets —
`analysis.d1_external_analysis_codec` (`:197`), `analysis.d1_external_analysis_control` (`:198`),
`analysis.d1_factorial` (`:199`), `analysis.d1_machine_analysis_binding` (`:200`),
`analysis.d1_machine_analysis_verification` (`:201`) and `evidence.d1_filing_graph` (`:210`).
Consequence: a commit that moves one of those files WITHOUT its fragment row leaves the sentinel
discoverable only under its new name, `_canonical_sentinel_key` cannot map it back, and the
`canonicalized == _FROZEN_SENTINEL_MAP` assertion (`:228-229`) goes RED at that commit — a red
pre-commit hook mid-stack. So: **every landing commit carries the fragment rows for exactly the
modules it lands**; the final MEASURED `legacy_edge_budget` is written in **the last commit that
changes an edge** — commit 4 as planned — and commit 5 AMENDS it if it moves one. It is never written
early and never left stale.

| # | Linear   | commit branch                                                                                                                                                                                                                                                                     | fragment written in this commit                                                                                                   |
| - | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1 | FOR-1478 | Land `ladder/analyze` (22 leaves) with `tests/test_analyze_review_file_cold_import.py` RED-first; `charts`/`palette`/`series`/`embedding_geometry`/`field_head_audit` rewrite onto `core.*`, `codec_equivariance.py:21` onto `core.observation.contracts` (the reclassified line) | the 22 analyze rows, `landed: true` with their `ladder.analyze.*` targets (carries sentinel `d1_factorial`)                       |
| 2 | FOR-1479 | Land `ladder/review` (13 leaves) FLAT — the isolation-ring modules included, no `isolation/` subpackage; the measured ring edges recorded in the commit message                                                                                                                   | the 13 review rows (carries sentinels `d1_external_analysis_codec`, `…_control`, `d1_machine_analysis_binding`, `…_verification`) |
| 3 | FOR-1480 | Land `ladder/file` (12 leaves), including the one `action: split` row `evidence.d1_filing_receipts`                                                                                                                                                                               | the 12 file rows (carries sentinel `evidence.d1_filing_graph`)                                                                    |
| 4 | FOR-1481 | Rewrite `ladder/gate/d1_verdict.py:11,18,22,23` onto `ladder.file.*` — the only commit that REMOVES landed→`__legacy` edges — and regenerate shims + the 2 forwarders with `gen_shims` + `--verify`                                                                               | the **final MEASURED `legacy_edge_budget`** plus the `note` carrying the retractions                                              |
| 5 | FOR-1482 | Observability: route `d1_external_worker.py:363,366` to `logger`, add `configure_logging` as the first statement of `main()`, promote `entropic_substrate.analysis.d1_external_worker`                                                                                            | the `conforming_entrypoints` key                                                                                                  |

If commit 5 changes an import or an edge count, commit 4's budget is re-measured and amended IN
commit 5 — never left stale, never split into a sixth commit.

## Retractions carried by this lane (house rule 4)

Each is retracted explicitly, in the commit that fixes it, never silently overwritten.

1. **49 → 47** targets, and **analyze 24 → 22**: the difference is the `analysis` package key plus
   `d1_reader_source_closure`, retired at p2.
2. **`codec_equivariance.py:21` is a LANDED rewrite**, not an `__legacy` edge; landed is 118
   (117 pack-measured + 1 reclassified) and other-quarantined is 4, not 117 / 5.
3. **`legacy_edge_budget` is the highest-ordered fragment's number, not a sum** (237 ≠ 45); the
   strict-decrease arm is retracted; direction here is INCREASING.
4. **Parent is `mythwave/p7-register-gate`**, not `mythwave/p6-core`; plan lines 375-376 are stale.
5. **The `.importlinter` "forbidden contract" tooth is unexecutable** and is struck; so is the
   mispointed-`_target` tooth (FOR-1548).
6. **Bare `print(` in the 47 sources is 2, not 0 and not 7**, both in `d1_external_worker.py`
   (`:363,366`), both stderr; v1's "zero may survive" was an obligation stated as a measurement.
7. **`review/isolation` is unexecutable** (no such dest; `tools/mapping.json` read-only until p13);
   v1's commit branch 3 is dropped, and the ring is recorded as measured edges instead.
8. **The `__legacy/analysis/__init__.py` facade retraction is NOT p11's** (ruling P11-FACADE):
   `d1_probe.py` and `ladder_probe.py` remain in the package, so 32 of 34 leaves move and the p3
   emptied-facade precondition fails. The facade-retraction commit branch is dropped and both
   `analysis/__init__.py` files leave Allowed files.
9. **The `.importlinter` contract-header line numbers** the arm-blind review cited (`:41`/`:69`) are
   off by one; MEASURED they are `:42`/`:70`. Its `type`-line cites (`:44`/`:72`) are exact.
