# w2-reconcile: the wave-w2 reconciliation rung rewrites sibling-target imports and retires their forwarders

Authoritative brief for one `td-phase-lifecycle.cascade` lifecycle. v1, drafted 2026-09-24 under
D-WAVES r3 and RUNG-OWNED (plan `docs/plans/src-hexagonal-refactor.plan.md`, conductor-rulings
table). The lifecycle's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; that section is
below.

**Epoch.** Every MEASURED count below was taken at `55b60aa7b97f071d3b63ab510b813d36021db01d`,
the tip of `mythwave/p11-analyze-review-file`, the w2 zip tip. That branch sits on
`mythwave/p9-run-lower` at `9d474be`, which sits on `mythwave/p8-capture` at `5b6f43c`. The drafting
session measured a read-only export of that tree. It never touched the main checkout, where a
publish's pre-push hook was running. If the parent tip moves, every MEASURED number here is void.
The lane restates it at the new SHA and never reinterprets it.

**Evidence classes.** Each number carries one of three labels:

- `[MEASURED]`: command output at `55b60aa`.
- `[predicted]`: conductor arithmetic on measured inputs, never typed into a reproduction path.
- `[read from code]`: behaviour derived by reading `tools/gen_shims.py` at `55b60aa`, not executed.

The lane MEASURES its own numbers and reproduces nothing from a `[predicted]` or `[read from code]` row.

## Branch

| field                   | value                                                                                                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| phase id                | `w2-reconcile` (the ONE serial reconciliation rung D-WAVES r3 names)                                                                                                                                            |
| branch                  | `mythwave/w2-reconcile`                                                                                                                                                                                         |
| parent                  | `mythwave/p11-analyze-review-file` @ `55b60aa`; pass it as `args.phase.startTip`                                                                                                                                |
| cascade `phase.mode`    | `arch`, explicit (the Opus design gate runs)                                                                                                                                                                    |
| sprint-manifest entry   | NONE required for the budget mechanism, see *Already ruled* NO-RUNG-FRAGMENT. Whether the conductor records the rung in `sprint-manifest.json` is conductor-owned; the manifest is outside this lane's envelope |
| wave                    | `serial`, post-zip. Not a wave lane: no lane worktree, no zip, no `publishMode:'defer'`, no CERTIFIED terminal state                                                                                            |
| `gates` (lifecycle arg) | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}`                                                                                                                 |
| live gate               | `just ship-it` runs INSIDE the lifecycle (house rule 1). FOR-1541 branch-only fallback as ruled for p5..p7                                                                                                      |
| `repoToplevel`          | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the main checkout. Launch only when that checkout is quiescent: no publish or pre-push hook running, `git status --porcelain` empty                          |
| trunk                   | `refactor/hexagonal-redesign`                                                                                                                                                                                   |

## Contract surface owned

None new. The rung moves no module, adds no address and edits no door. It DISCHARGES two
obligations the plan hands it:

| obligation                                                                 | source                                                                                        | discharge                                         |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| rewrite every sibling-target `__legacy` import to its final landed address | D-WAVES r3 ("ONE serial reconciliation rung (conductor) rewrites the sibling-target imports") | 99 src lines in 41 files, *Scope*                 |
| rewrite the RUNG-OWNED importer once                                       | RUNG-OWNED (`ladder/gate/ladder_predecessor.py` lines 8/12/14/20)                             | commit 1                                          |
| retire every forwarder whose last importer is rewritten                    | FOR-1547 derived lifecycle (`tools/gen_shims.py` RULING 1)                                    | 30 of 34 forwarder pairs deleted BY THE GENERATOR |

## Contracts consumed

All landed before launch.

- The 34 landed destination addresses the forwarders point at: p8 `ladder.capture.*` (29),
  p9 `ladder.run.trainer.*` and `ladder.run.provider.d1_command_runner` (3), p11
  `ladder.analyze.ladder_analysis`, `ladder.file.ladder_filing` (2).
- `tools/gen_shims.py`: generate, `--verify`, `--preflight`, the derived forwarder predicate
  (`_forwards` = `_forwardable_row` ∧ address present in `collect_legacy_importers(src, tests)`),
  `delete_stale_forwarders`, `find_stale_forwarders`.
- `tools/import_graph.py` plus `tests/test_legacy_coupling.py`'s prefix classifier. This is the
  budget oracle, and the EQUALITY arm is the binding one.
- `tests/test_no_old_prefix.py`, `tests/test_module_identity.py`, `tests/test_gen_shims.py`.
- The five cold-import tests `tests/test_{capture,run_lower,analyze_review_file,register_gate,legacy_data_build}_cold_import.py`.
  All are consumed unchanged.
- Identity mirrors, READ-ONLY: `tests/test_identity_epoch.py`, `tools/regime_rows.py`,
  `tests/test_regime_rows.py`, `docs/identity/read-old-table.md`.

## Sacred invariants

Quoted verbatim from `CLAUDE.md`. The rung edits IMPORT LINES ONLY, in `ladder/**`, and deletes
generated files under `__legacy/**`. MEASURED tripwire for all five: the touch set intersected
with `core/**` is empty, because every Allowed file is under `src/entropic_substrate/ladder/`,
`src/entropic_substrate/__legacy/`, or is the p11 fragment.

| id   | verbatim invariant                                                                                                          | tripwire / guard                                                                                                     |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | `git diff mythwave/p11-analyze-review-file -- src/entropic_substrate/core tests/test_kappa_autocast_island.py` EMPTY |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | same empty diff over `core/geometry`                                                                                 |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | same empty diff over `core/substrate`                                                                                |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | same empty diff over `core/sentence`                                                                                 |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | same empty diff over `core/latent`                                                                                   |

The rung touches seven trainer files, among them `d1_training_step.py` and `d1_segment_engine.py`.
The numerics tripwire there is the IMPORT-ONLY instrument in ACCEPTANCE. Every non-import
statement of every rewritten file must be AST-identical to the parent.

House rules HR-1..HR-7 bind exactly as tabled in `.claude/workflows/briefs/p7-register-gate.md`.
HR-3 is live in a new way here: the cold-import sweep runs 44 fresh interpreters per commit, and a
slow probe under host contention is investigated, never met with a raised timeout.

## Scope

### The worklist, and why it is complete

Input `rung-worklist.json` lists 34 forwarders, 42 importer files, 104 import lines and zero
reader importers. Checks run against the tree `[MEASURED]`:

- All 104 `(file, line, address)` triples resolve to an `ast.Import` / `ast.ImportFrom` node at
  that line naming that address: 103 `ImportFrom`, 1 plain `import … as`.
- The worklist is COMPLETE. Every src import of a `landed: true` row's `__legacy` address was
  checked, and the set not on the worklist is `[]`.
- It splits as 99 src lines in 41 files plus 5 test lines in 1 file, `tests/test_d1_training_step_split.py`.
- The 99 src lines are 96 DISTINCT `(importer, __legacy module)` edges, and all 96 are present in
  `tools/import_graph.py`'s edge set. The three same-edge duplicates are `d1_capture.py:21/260`,
  `d1_diagnostics.py:155/171` and `ladder_registration.py:381/440`.
- ZERO importer lives under `ladder/capture/**`. The four-way partition in the rung's commission
  therefore has an empty member, and the rung is THREE commits, see *Planned commit branches*.
- The plan's D-WAVES r3 row said "measured ~105 edges at 3e9ffa2", at a different epoch. It is
  restated here at `55b60aa` as 104 lines, 96 distinct src edges. It is a restatement, not a
  retraction: the epoch moved.

Lane reproduction, read-only, run at the parent tip before the first edit. It prints each src
line importing a landed row's vacated `__legacy` address. It printed `99` against the `55b60aa`
export `[MEASURED]`, with paths made absolute and the logic unchanged.

```bash
PYTHONPATH=src uv run python - <<'PY'
import ast, sys; from pathlib import Path
sys.path.insert(0, "tools"); import gen_shims as g
m = g.load_mapping(Path("tools/mapping.json"))
r = g.resolve_all_modules(g.merge_fragments(g.load_fragments(Path(".claude/workflows/landed-set")), m), m)
landed = {g.legacy_address(o): e["target"] for o, e in r.items() if g._forwardable_row(e)}
n = 0
for p in sorted(Path("src").rglob("*.py")):
    t = p.read_text()
    if g.LEGACY_PREFIX not in t or g.verify_shim_source(t)[0]: continue
    for x in ast.walk(ast.parse(t)):
        mods = [a.name for a in x.names] if isinstance(x, ast.Import) else [x.module] if isinstance(x, ast.ImportFrom) and x.module and not x.level else []
        for mod in mods:
            if mod in landed: n += 1; print(f"{p}:{x.lineno}  {mod} -> {landed[mod]}")
print(n)
PY
```

### Rewrite table, per commit, per file:line

Each row names the address the line imports today and the address it imports after the rewrite,
with `entropic_substrate.` omitted. The rewrite is an address substitution only. Imported names,
`as` aliases and statement scope are unchanged. Line numbers are `[MEASURED]` at `55b60aa`.
They will SHIFT once ruff isort re-sorts a rewritten block, as p11 commit 3 recorded. So ACCEPTANCE
never pins a post-rewrite line number.

`d1_capture.py:11` is the one plain-`import` form:
`import entropic_substrate.__legacy.telemetry as telemetry_api` becomes
`import entropic_substrate.ladder.capture.reader.telemetry as telemetry_api`.

Eight lines are FUNCTION-LOCAL `[MEASURED]`: `d1_capture.py:219,260`, `d1_diagnostics.py:155,171`,
`ladder_analysis.py:96`, `ladder_registration.py:381,436,440`. `ast.walk` counts them, so they are
edges and they are rewritten. But a cold import never executes them, which is why ACCEPTANCE carries
a separate binding probe for them.

#### Commit 1, ladder/gate (RUNG-OWNED): 1 file, 4 lines, 4 distinct edges

| file under `src/entropic_substrate/` | line | old address                              | new address                                   |
| ------------------------------------ | ---- | ---------------------------------------- | --------------------------------------------- |
| `ladder/gate/ladder_predecessor.py`  | 8    | `__legacy.analysis.ladder_analysis`      | `ladder.analyze.ladder_analysis`              |
| 〃                                   | 12   | `__legacy.analysis.ladder_filing`        | `ladder.file.ladder_filing`                   |
| 〃                                   | 14   | `__legacy.evidence.d1_capture_artifacts` | `ladder.capture.custody.d1_capture_artifacts` |
| 〃                                   | 20   | `__legacy.evidence.d1_capture_bundle`    | `ladder.capture.custody.d1_capture_bundle`    |

#### Commit 2, ladder/run: 17 files, 41 lines, 41 distinct edges

| file under `src/entropic_substrate/`                   | line | old address                                 | new address                                       |
| ------------------------------------------------------ | ---- | ------------------------------------------- | ------------------------------------------------- |
| `ladder/run/construction/d1_r2_runtime_composition.py` | 19   | `__legacy.evidence.d1_capture_ports`        | `ladder.capture.custody.d1_capture_ports`         |
| 〃                                                     | 24   | `__legacy.evidence.d1_capture_subprocess`   | `ladder.capture.custody.d1_capture_subprocess`    |
| `ladder/run/provider/d1_runpod_adapter.py`             | 21   | `__legacy.evidence.resource_billing`        | `ladder.capture.resource.resource_billing`        |
| `ladder/run/resource/d1_resource_activation.py`        | 10   | `__legacy.evidence.resource_ledger`         | `ladder.capture.resource.resource_ledger`         |
| 〃                                                     | 16   | `__legacy.evidence.resource_meter_models`   | `ladder.capture.resource.resource_meter_models`   |
| 〃                                                     | 20   | `__legacy.evidence.resource_meter_strategy` | `ladder.capture.resource.resource_meter_strategy` |
| `ladder/run/trainer/checkpoint_validation.py`          | 9    | `__legacy.evidence.d1_journal_values`       | `ladder.capture.records.d1_journal_values`        |
| 〃                                                     | 13   | `__legacy.evidence.resource_ledger_codec`   | `ladder.capture.resource.resource_ledger_codec`   |
| `ladder/run/trainer/checkpoint_values.py`              | 14   | `__legacy.evidence.d1_journal_values`       | `ladder.capture.records.d1_journal_values`        |
| 〃                                                     | 15   | `__legacy.evidence.resource_ledger`         | `ladder.capture.resource.resource_ledger`         |
| 〃                                                     | 16   | `__legacy.evidence.resource_ledger_codec`   | `ladder.capture.resource.resource_ledger_codec`   |
| `ladder/run/trainer/d1_calibration.py`                 | 17   | `__legacy.d1_calibration_records`           | `ladder.capture.records.d1_calibration_records`   |
| `ladder/run/trainer/d1_evaluation.py`                  | 11   | `__legacy.d1_numerical_defect`              | `ladder.capture.reader.d1_numerical_defect`       |
| `ladder/run/trainer/d1_fresh_trainer.py`               | 11   | `__legacy.evidence.d1_journal`              | `ladder.capture.records.d1_journal`               |
| 〃                                                     | 12   | `__legacy.evidence.d1_journal_values`       | `ladder.capture.records.d1_journal_values`        |
| `ladder/run/trainer/d1_resume_runtime.py`              | 14   | `__legacy.d1_observability`                 | `ladder.capture.adapters.d1_observability`        |
| 〃                                                     | 18   | `__legacy.evidence.d1_journal_values`       | `ladder.capture.records.d1_journal_values`        |
| 〃                                                     | 23   | `__legacy.evidence.resource_ledger`         | `ladder.capture.resource.resource_ledger`         |
| 〃                                                     | 27   | `__legacy.evidence.resource_ledger_codec`   | `ladder.capture.resource.resource_ledger_codec`   |
| `ladder/run/trainer/d1_resume_trainer.py`              | 10   | `__legacy.d1_observability`                 | `ladder.capture.adapters.d1_observability`        |
| 〃                                                     | 11   | `__legacy.evidence.d1_journal`              | `ladder.capture.records.d1_journal`               |
| 〃                                                     | 12   | `__legacy.evidence.d1_journal_values`       | `ladder.capture.records.d1_journal_values`        |
| 〃                                                     | 18   | `__legacy.evidence.resource_ledger`         | `ladder.capture.resource.resource_ledger`         |
| 〃                                                     | 21   | `__legacy.evidence.resource_work`           | `ladder.capture.resource.resource_work`           |
| `ladder/run/trainer/d1_segment_engine.py`              | 11   | `__legacy.evidence.d1_journal_values`       | `ladder.capture.records.d1_journal_values`        |
| 〃                                                     | 17   | `__legacy.evidence.resource_meter_models`   | `ladder.capture.resource.resource_meter_models`   |
| 〃                                                     | 18   | `__legacy.evidence.resource_work`           | `ladder.capture.resource.resource_work`           |
| 〃                                                     | 19   | `__legacy.evidence.resource_work_codec`     | `ladder.capture.resource.resource_work_codec`     |
| `ladder/run/trainer/d1_terminal_failure.py`            | 5    | `__legacy.d1_numerical_defect`              | `ladder.capture.reader.d1_numerical_defect`       |
| 〃                                                     | 10   | `__legacy.evidence.resource_ledger`         | `ladder.capture.resource.resource_ledger`         |
| `ladder/run/trainer/d1_trainer_runtime.py`             | 14   | `__legacy.data.build`                       | `ladder.capture.custody.build`                    |
| 〃                                                     | 15   | `__legacy.evidence.resource_ledger`         | `ladder.capture.resource.resource_ledger`         |
| `ladder/run/trainer/d1_training_step.py`               | 21   | `__legacy.d1_training_step_records`         | `ladder.capture.records.d1_training_step_records` |
| `ladder/run/transport/d1_trainer_handoff.py`           | 9    | `__legacy.evidence.d1_journal_values`       | `ladder.capture.records.d1_journal_values`        |
| 〃                                                     | 10   | `__legacy.evidence.resource_ledger`         | `ladder.capture.resource.resource_ledger`         |
| 〃                                                     | 11   | `__legacy.evidence.resource_work`           | `ladder.capture.resource.resource_work`           |
| `ladder/run/transport/d1_trainer_handoff_codec.py`     | 12   | `__legacy.evidence.d1_journal`              | `ladder.capture.records.d1_journal`               |
| 〃                                                     | 13   | `__legacy.evidence.d1_journal_values`       | `ladder.capture.records.d1_journal_values`        |
| 〃                                                     | 18   | `__legacy.evidence.resource_ledger_codec`   | `ladder.capture.resource.resource_ledger_codec`   |
| 〃                                                     | 23   | `__legacy.evidence.resource_work_codec`     | `ladder.capture.resource.resource_work_codec`     |
| `ladder/run/values/d1_operator_ports.py`               | 24   | `__legacy.evidence.resource_billing`        | `ladder.capture.resource.resource_billing`        |

#### Commit 3, ladder/analyze + ladder/review + ladder/file: 23 files, 54 lines, 51 distinct edges

| file under `src/entropic_substrate/`                | line | old address                                    | new address                                         |
| --------------------------------------------------- | ---- | ---------------------------------------------- | --------------------------------------------------- |
| `ladder/analyze/charts.py`                          | 30   | `__legacy.telemetry`                           | `ladder.capture.reader.telemetry`                   |
| `ladder/analyze/d1_capture.py`                      | 11   | `__legacy.telemetry`                           | `ladder.capture.reader.telemetry`                   |
| 〃                                                  | 12   | `__legacy.evidence.d1_capture_aggregate`       | `ladder.capture.custody.d1_capture_aggregate`       |
| 〃                                                  | 16   | `__legacy.evidence.d1_capture_artifacts`       | `ladder.capture.custody.d1_capture_artifacts`       |
| 〃                                                  | 21   | `__legacy.evidence.d1_capture_roster`          | `ladder.capture.custody.d1_capture_roster`          |
| 〃                                                  | 25   | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |
| 〃                                                  | 219  | `__legacy.evidence.d1_capture_bundle`          | `ladder.capture.custody.d1_capture_bundle`          |
| 〃                                                  | 260  | `__legacy.evidence.d1_capture_roster`          | `ladder.capture.custody.d1_capture_roster`          |
| `ladder/analyze/d1_clause_policy.py`                | 12   | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |
| `ladder/analyze/d1_diagnostics.py`                  | 155  | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |
| 〃                                                  | 171  | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |
| `ladder/analyze/d1_inspect_codec.py`                | 8    | `__legacy.d1_numerical_defect`                 | `ladder.capture.reader.d1_numerical_defect`         |
| 〃                                                  | 9    | `__legacy.evidence.d1_journal_values`          | `ladder.capture.records.d1_journal_values`          |
| 〃                                                  | 10   | `__legacy.evidence.resource_work_codec`        | `ladder.capture.resource.resource_work_codec`       |
| `ladder/analyze/d1_inspect_projection.py`           | 9    | `__legacy.d1_journal_snapshot`                 | `ladder.capture.reader.d1_journal_snapshot`         |
| 〃                                                  | 10   | `__legacy.evidence.d1_journal_values`          | `ladder.capture.records.d1_journal_values`          |
| 〃                                                  | 11   | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |
| `ladder/analyze/d1_inspect_values.py`               | 11   | `__legacy.d1_numerical_defect`                 | `ladder.capture.reader.d1_numerical_defect`         |
| 〃                                                  | 12   | `__legacy.evidence.d1_journal_values`          | `ladder.capture.records.d1_journal_values`          |
| 〃                                                  | 13   | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |
| 〃                                                  | 18   | `__legacy.evidence.resource_work`              | `ladder.capture.resource.resource_work`             |
| `ladder/analyze/embedding_geometry.py`              | 49   | `__legacy.data.build`                          | `ladder.capture.custody.build`                      |
| `ladder/analyze/ladder_analysis.py`                 | 17   | `__legacy.evidence.d1_capture_artifacts`       | `ladder.capture.custody.d1_capture_artifacts`       |
| 〃                                                  | 18   | `__legacy.evidence.d1_capture_bundle`          | `ladder.capture.custody.d1_capture_bundle`          |
| 〃                                                  | 19   | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |
| 〃                                                  | 24   | `__legacy.evidence.resource_work_codec`        | `ladder.capture.resource.resource_work_codec`       |
| 〃                                                  | 96   | `__legacy.telemetry`                           | `ladder.capture.reader.telemetry`                   |
| `ladder/analyze/ladder_registration.py`             | 17   | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |
| 〃                                                  | 381  | `__legacy.telemetry`                           | `ladder.capture.reader.telemetry`                   |
| 〃                                                  | 436  | `__legacy.evidence.d1_capture_artifacts`       | `ladder.capture.custody.d1_capture_artifacts`       |
| 〃                                                  | 440  | `__legacy.telemetry`                           | `ladder.capture.reader.telemetry`                   |
| `ladder/analyze/mlflow_attach.py`                   | 19   | `__legacy.mlflow_aggregate`                    | `ladder.capture.reader.mlflow_aggregate`            |
| `ladder/analyze/series.py`                          | 20   | `__legacy.telemetry`                           | `ladder.capture.reader.telemetry`                   |
| `ladder/file/d1_filing_application.py`              | 12   | `__legacy.evidence.d1_process_manager`         | `ladder.capture.ledger.d1_process_manager`          |
| `ladder/file/d1_filing_graph.py`                    | 10   | `__legacy.evidence.d1_capture_aggregate`       | `ladder.capture.custody.d1_capture_aggregate`       |
| 〃                                                  | 14   | `__legacy.evidence.d1_capture_bundle`          | `ladder.capture.custody.d1_capture_bundle`          |
| 〃                                                  | 15   | `__legacy.evidence.d1_capture_roster`          | `ladder.capture.custody.d1_capture_roster`          |
| `ladder/file/d1_process_filing.py`                  | 7    | `__legacy.evidence.d1_lifecycle_values`        | `ladder.capture.ledger.d1_lifecycle_values`         |
| 〃                                                  | 11   | `__legacy.evidence.d1_process_manager`         | `ladder.capture.ledger.d1_process_manager`          |
| 〃                                                  | 12   | `__legacy.evidence.d1_process_replay`          | `ladder.capture.ledger.d1_process_replay`           |
| `ladder/review/d1_analysis_api.py`                  | 9    | `__legacy.evidence.d1_capture_aggregate`       | `ladder.capture.custody.d1_capture_aggregate`       |
| `ladder/review/d1_execution_receipt.py`             | 15   | `__legacy.evidence.d1_capture_artifacts`       | `ladder.capture.custody.d1_capture_artifacts`       |
| `ladder/review/d1_external_analysis_adapter.py`     | 13   | `__legacy.evidence.d1_capture_aggregate`       | `ladder.capture.custody.d1_capture_aggregate`       |
| `ladder/review/d1_external_analysis_control.py`     | 16   | `__legacy.evidence.d1_capture_aggregate_codec` | `ladder.capture.custody.d1_capture_aggregate_codec` |
| 〃                                                  | 20   | `__legacy.evidence.d1_capture_artifacts`       | `ladder.capture.custody.d1_capture_artifacts`       |
| 〃                                                  | 24   | `__legacy.evidence.d1_capture_ports`           | `ladder.capture.custody.d1_capture_ports`           |
| `ladder/review/d1_external_review_adapter.py`       | 17   | `__legacy.d1_command_runner`                   | `ladder.run.provider.d1_command_runner`             |
| `ladder/review/d1_external_worker.py`               | 17   | `__legacy.evidence.d1_capture_aggregate_codec` | `ladder.capture.custody.d1_capture_aggregate_codec` |
| `ladder/review/d1_machine_analysis_verification.py` | 10   | `__legacy.evidence.d1_capture_aggregate`       | `ladder.capture.custody.d1_capture_aggregate`       |
| `ladder/review/d1_review_packet.py`                 | 13   | `__legacy.evidence.d1_capture_aggregate`       | `ladder.capture.custody.d1_capture_aggregate`       |
| 〃                                                  | 16   | `__legacy.evidence.d1_capture_artifacts`       | `ladder.capture.custody.d1_capture_artifacts`       |
| 〃                                                  | 17   | `__legacy.evidence.d1_capture_roster`          | `ladder.capture.custody.d1_capture_roster`          |
| 〃                                                  | 21   | `__legacy.evidence.d1_capture_validation`      | `ladder.capture.custody.d1_capture_validation`      |
| 〃                                                  | 26   | `__legacy.evidence.d1_telemetry_values`        | `ladder.capture.records.d1_telemetry_values`        |

### Tests are NOT rewritten; four forwarders survive, test-served

The ruling says existing tests stay untouched before p13. The 5 worklist lines in
`tests/test_d1_training_step_split.py` therefore stay at their `__legacy` addresses. Two of them use
the PKG-FORM package-form `from entropic_substrate.__legacy import (…)` at `:35`, which the
importer scan expands. The forwarders those lines keep alive SURVIVE by the derived predicate. The
list is `[MEASURED]`: the forwarder predicate was recomputed with `gen_shims`' own scan roots
`src` + `tests`, minus exactly the 99 src worklist lines, and asserted equal to
`collect_legacy_importers` on the unmodified tree first.

| surviving forwarder (`src/entropic_substrate/__legacy/…`) | points at                                         | kept alive by                                                      |
| --------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| `d1_calibration.py` + `.pyi`                              | `ladder.run.trainer.d1_calibration`               | `tests/test_d1_training_step_split.py:35` (package form)           |
| `d1_training_step.py` + `.pyi`                            | `ladder.run.trainer.d1_training_step`             | `tests/test_d1_training_step_split.py:35` (package form)           |
| `d1_training_step_records.py` + `.pyi`                    | `ladder.capture.records.d1_training_step_records` | `tests/test_d1_training_step_split.py:35` (package form) and `:43` |
| `d1_calibration_records.py` + `.pyi`                      | `ladder.capture.records.d1_calibration_records`   | `tests/test_d1_training_step_split.py:40`                          |

After the rung the counts are `[predicted]` 4 forwarders expected, 30 deleted, and 8 generated files
left under `__legacy`. At p13, when that test is rewritten, the next `generate()` retires all four.
p12's `__legacy`-empty postcondition is the backstop.

String references to `__legacy` in tests are invisible to the static importer scan, so each one
was READ for a runtime dependency on a deleted forwarder `[MEASURED]`. The result is none:

- `test_legacy_data_build_cold_import.py` derives the landed address live through `_current_address`.
- `test_review_packet.py:566` and `scripts/assemble_review_packet.py:651` are literal shim text in a
  tmp fixture and a docstring.
- `test_legacy_coupling.py:221` is a synthetic edge tuple.
- Outside `src/` and `tests/`, `__legacy` appears only in comments: `justfile:117`,
  `pyproject.toml:85`, `.github/workflows/ci.yaml:77`.
- No test builds a probe subject from `gen_shims.legacy_address(...)` without a literal on disk
  `[MEASURED]`. `grep -ln legacy_address tests/` hits only two files:
  `test_gen_shims.py:2250`, a `tmp_path` fixture test name, and
  `test_legacy_data_build_cold_import.py:51`, which returns the landed `target` when the row is
  landed.

### Forwarder retirement per commit, in the recommended order

`[predicted]`, from the derived predicate recomputed after each commit's lines are removed:

| after commit          | forwarders expected | generated files under `__legacy` | pairs deleted by `generate()` in this commit                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------- | ------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| parent `55b60aa`      | 34 `[MEASURED]`     | 68 `[MEASURED]`                  | none                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 1 gate                | 32                  | 64                               | `analysis/ladder_analysis`, `analysis/ladder_filing`                                                                                                                                                                                                                                                                                                                                                                         |
| 2 run                 | 24                  | 48                               | `d1_observability`, `evidence/{d1_capture_subprocess, d1_journal, resource_billing, resource_ledger, resource_ledger_codec, resource_meter_models, resource_meter_strategy}`                                                                                                                                                                                                                                                 |
| 3 analyze+review+file | 4                   | 8                                | `d1_command_runner`, `d1_journal_snapshot`, `d1_numerical_defect`, `mlflow_aggregate`, `telemetry`, `data/build`, `evidence/{d1_capture_aggregate, d1_capture_aggregate_codec, d1_capture_artifacts, d1_capture_bundle, d1_capture_ports, d1_capture_roster, d1_capture_validation, d1_journal_values, d1_lifecycle_values, d1_process_manager, d1_process_replay, d1_telemetry_values, resource_work, resource_work_codec}` |

Each forwarder is deleted as a `.py` + `.pyi` pair, so the rung deletes 30 × 2 = 60 files in total.
`evidence/d1_capture_artifacts` and `evidence/d1_capture_bundle` retire in commit 3, not commit 1,
because commit 3's files import them too.

### The generator step is manifest-keyed; step 0 is mandatory

This is the one mechanism in the rung that can go silently wrong. `[read from code]`, not executed:

- `generate()` deletes a stale forwarder ONLY IF the PREVIOUS `.emitted.json` lists it
  (`delete_stale_forwarders(prev_doc, …)`, `tools/gen_shims.py` around L1189).
- `--verify` finds stale forwarders ONLY among manifest-listed ones (`find_stale_forwarders`).
- `.emitted.json` is UNTRACKED (FOR-1539). The main checkout's copy reflects whichever branch last
  ran generate; during drafting that was a different branch.
- Without a manifest that lists the 34, a post-rewrite `generate()` deletes NOTHING. It writes a
  fresh manifest without the dead entries, and `--verify` then passes CLEAN while 60 dead tracked
  files remain on disk. That is a false green.

Therefore:

1. **Step 0, lens control, before the first edit.** On `mythwave/w2-reconcile` at `55b60aa`, run the
   canonical generate. Expect `git status --porcelain` EMPTY: generate at the zip tip reproduces the
   committed files, and `55b60aa` is the commit that ran generate + `--verify` on this tree.
   `.emitted.json`'s `forwarders` must have length 34. Then run `--verify` and expect green. Record
   all three outputs. A non-empty status or a length other than 34 is a FREEZE; do not rewrite on
   an unverified lens.
2. **An independent file-count instrument** closes the manifest blind spot at every commit. It
   has two forms, and both are required:

   ```bash
   # working tree, BEFORE each commit, right after generate:
   grep -rl '^# Generated by gen_shims' src/entropic_substrate/__legacy | wc -l
   # committed tree, AFTER each commit:
   git grep -l '^# Generated by gen_shims' HEAD -- src/entropic_substrate/__legacy | wc -l
   ```

   The committed form measured 68 at `55b60aa` `[MEASURED]`. Both forms must read 64, then 48,
   then 8 `[predicted]`. `--verify` alone is not accepted as evidence of deletion.

### Budget: the highest-ordered fragment, re-measured per commit

- `tests/test_legacy_coupling.py`'s equality arm reads `legacy_edge_budget` from the
  highest-ordered fragment present in the manifest `sequence`. That is `p11-analyze-review-file.json`:
  p10 has no fragment yet, and p10 is sequenced after p11.
- MEASURED at `55b60aa`: 128 landed→`__legacy` edges by `tools/import_graph.build_graph` +
  `test_legacy_coupling.count_landed_to_legacy_edges`. That equals the committed budget of 128, so
  the lens check passes.
- The landed layers carry 132 `__legacy` import statements `[MEASURED]`. After the rewrite the
  expected count is 132 − 99 = 33 `[predicted]`, covering 32 distinct edges.

Every rewritten line removes an edge, because each `(importer, __legacy module)` pair loses its
only line. Checked `[MEASURED]`: no residual edge from a rewritten importer targets a rewritten
module. The 11 residual edges out of rewritten importers all go to `landed: false` modules, such as
`d1_operator_values`, `evidence.ladder_campaign` and `d1_predecessor_loader`. The new edges land in
`ladder.*` and are not counted.

`[predicted]` arithmetic, in the recommended order:

| commit                | removes | budget written into `p11-analyze-review-file.json` |
| --------------------- | ------- | -------------------------------------------------- |
| parent                |         | 128 `[MEASURED]`                                   |
| 1 gate                | 4       | 128 − 4 = **124**                                  |
| 2 run                 | 41      | 124 − 41 = **83**                                  |
| 3 analyze+review+file | 51      | 83 − 51 = **32**                                   |

The 32 residual edges `[predicted]` group as `ladder.run` 27, `ladder.gate` 2, `ladder.analyze` 2
and `ladder.file` 1. All go to still-quarantined modules owned by p10/p12:
`d1_operator_values`, `d1_operator_evidence`, `d1_predecessor_loader`, `evidence.ladder_campaign`,
`d1_excluded_capture`, `d1_checkpoint_dvc_{recovery,sync}`, `d1_runpod_billing_parser`.

BUDGET-PER-COMMIT sets the per-commit write. Each commit sets `legacy_edge_budget` to the number it
MEASURES, never the number above. It APPENDS one dated clause to the fragment's single `note`
string, in the ZIP-REMEASURE form. The literal form is:
`re-measured at w2-reconcile commit <k>/3 (<scope>) on <YYYY-MM-DD>: was <N>, now <M>`.
No other fragment field changes, and `modules` stays byte-identical.

The lane measures with:

```bash
PYTHONPATH=src uv run python -c "import sys; sys.path[:0]=['tools','tests']; from pathlib import Path; import import_graph, test_legacy_coupling as t; _,e=import_graph.build_graph(Path('src/entropic_substrate'),'entropic_substrate'); print(t.count_landed_to_legacy_edges(e))"
```

### R-PIN: none owed, MEASURED

`[MEASURED]` at `55b60aa`, by `ast.literal_eval` of the two `_READER_MODULES` tuples:

- `h-checkpoint-reader-v2` in `ladder/run/checkpoint.py:65` has 51 members, and 0 of them are
  among the 41 importers.
- `h-source-reader-v3` in `ladder/capture/capture.py:29` has 7 members, and 0 of them are among the
  importers.
- 0 of the 34 forwarder addresses are members of either set.

No member's bytes change, so no regime row moves. A grep of 408 hashing files (`sha256`,
`hashlib`, `read_bytes`) for importer filenames found no source-byte pin on any importer. The one
real hit, `tests/test_d1_external_worker.py`, relocates the worker and control sources by
`__file__` and hashes only artifacts.

The CONTROL is an assertion:
`git diff --exit-code mythwave/p11-analyze-review-file -- tests/test_identity_epoch.py tools/regime_rows.py tests/test_regime_rows.py docs/identity/read-old-table.md`
must be EMPTY, and `tests/test_identity_epoch.py` and `tests/test_regime_rows.py` must be green
unchanged.

### FACADE-EMPTY: triggered for `__legacy/data/`, already satisfied, no edit

After commit 3, `__legacy/data/` holds only `__init__.py`. Its sole leaf was the `build.py`
forwarder pair `[predicted]` from the table above, and the listing is `[MEASURED]` at `55b60aa`.
That `__init__.py` is ALREADY the p3 retracted docstring-only facade `[MEASURED]`: 648 bytes, one
docstring statement, zero re-exports. FACADE-EMPTY is therefore met with NO edit. Removing the now
leaf-empty package directory belongs to p12's `__legacy`-empty postcondition, not to this rung.

`__legacy/evidence/` keeps real `ladder_campaign.py`, p5b's gate half, and `__legacy/analysis/`
keeps real `d1_probe.py` and `ladder_probe.py` per P11-FACADE. Neither empties.

## ACCEPTANCE

Machine-checkable, one line each. The lifecycle's `acceptance` arg points here.

- **Step 0 recorded.** At `55b60aa` on the new branch: canonical generate leaves
  `git status --porcelain` empty, `.emitted.json` lists 34 forwarders, and `--verify` is green.
- **Rewrite complete.** At the tip, the *Scope* reproduction snippet prints `0`: no src line
  imports a landed row's `__legacy` address.
- **Import-only diff.** For each of the 41 files, compare the parent and the tip with the address
  map applied: the non-import statements' `ast.dump` sequences are identical, and the multiset of
  import statements is identical. Every imported name, alias and scope is preserved. The instrument
  is a scratch AST-compare probe quoted in the PR body. A ruff isort reorder inside an import block
  is admissible; any other change is a DECISION.
- **Generator clean at every commit.** The canonical generate is followed by `--verify`
  ("verified: …") and `--preflight`. At the tip, `--preflight` prints
  `4 __legacy forwarder(s) expected, 0 not yet on disk`, against
  `34 __legacy forwarder(s) expected, 0 not yet on disk` `[MEASURED]` at the parent.
- **Deletion evidenced independently of the manifest.** The `git grep -l '^# Generated by gen_shims'`
  count under `__legacy` reads 64, 48, 8 at commits 1, 2, 3. The four survivors are exactly the
  pairs tabled under *Scope*, named by path in the PR body.
- **Budget equality at every commit.** `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget`
  is green at each commit, with the committed value MEASURED. The direction is DECREASING, with
  predictions 124, 83, 32. The equality arm binds; there is no monotone arm (BUDGET-MONOTONE
  retracted).
- `tests/test_no_old_prefix.py::test_measured_landed_to_old_prefix_edges_is_zero` is green.
- `tests/test_module_identity.py` and `tests/test_gen_shims.py` are green unchanged. The rung
  changes no fragment `target` and no generator code.
- **Cold import, fresh interpreter per subject, subject entered FIRST, stdout marker not exit
  code.** Run at every commit over the 44 subjects: the 41 rewritten importers plus the three pod
  entrypoints `entropic_substrate.d1_trainer_mailbox_command`, `entropic_substrate.d_ladder_worker_command`
  and `entropic_substrate.sentence_prediction.worker_command`. The sweep is the p9-brief shape,
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -c "import $m as _m; print('COLD_OK', _m.__name__)"`.
  Baseline at the parent is **ok=44 fail=0** `[MEASURED]`, run on the `55b60aa` export while a
  `pytest -n auto` shared the host. The required result is ok=44 at each commit. It is a scratch
  probe quoted in the certification record, not a committed test: `tests/**` is outside the
  envelope. The 41 importers are the files in the three commit tables; the 42nd worklist file is
  the test, which is not a subject.
- **Function-local binding probe.** The 8 function-local lines are unreachable by cold import.
  Each post-rewrite statement runs verbatim, first, in a fresh interpreter, with the same marker
  and the addresses shown:
  - `d1_capture.py:219`: `from entropic_substrate.ladder.capture.custody.d1_capture_bundle import D1ValidatedRunCapture`
  - `d1_capture.py:260`: `from entropic_substrate.ladder.capture.custody.d1_capture_roster import D1ExpectedRun`
  - `d1_diagnostics.py:155` and `:171`: `from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunEvidence`
  - `ladder_analysis.py:96`, `ladder_registration.py:381` and `:440`: `from entropic_substrate.ladder.capture.reader.telemetry import read_d1_run_evidence_bytes`
  - `ladder_registration.py:436`: `from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import D1ArtifactRole, verify_d1_artifact`
- The existing cold-import tests `test_{capture,run_lower,analyze_review_file,register_gate,legacy_data_build}_cold_import.py`
  are green unchanged.
- **R-PIN control.** The four identity files show an EMPTY diff against the parent, and the
  identity tests are green unchanged. Command output goes in the PR body.
- **Numerics control.** `git diff mythwave/p11-analyze-review-file -- src/entropic_substrate/core tests/test_kappa_autocast_island.py`
  is EMPTY.
- **`lint-imports` green.** Every rewrite is `ladder.*` → `ladder.*`, a same-layer edge, so the
  layers contract cannot fire. `mlflow-only-from-adapter` reports indirect chains. Replacing
  `A → forwarder → T` with `A → T` preserves the set of modules reachable from every source, so its
  verdict is invariant `[read from code]`. The green run is the evidence.
- `.secrets.baseline` has no entry for any of the 41 importers, the 68 forwarder files or the p11
  fragment `[MEASURED]`, so it is predicted untouched. If detect-secrets moves it, the standing rule
  applies.
- `just check`, `just pc-ci` through the exit runner, and `just ship-it` are green on the tip, and
  `gh pr view --json headRefOid` equals `git rev-parse HEAD`.
- **Envelope.** The actual diff against the parent is a subset of *Allowed files*. The predicted
  total is 41 modified + 60 deleted + 1 fragment = 102 paths `[predicted]`. No `.pyi` of a leaf
  shim at an OLD path changes: the rewrite leaves every target's bound names unchanged
  `[read from code: derive_stub_body keys on the target's own bound names]`. Any extra path from
  `git status --porcelain` after generate is a DECISION.
- Every mutation under *Guard teeth by mutation* is performed and reverted in-branch, both arms in
  the PR body.

## Non-goals

Thoroughness beyond ACCEPTANCE is NOT welcome. Anything below that looks necessary is a
`DECISION`, not an edit.

- No module moves. No `git mv`, no new address, no fragment `modules` row, no `landed` flip.
- No facade retraction or facade edit (FACADE-EMPTY): `__legacy/{data,evidence,analysis}/__init__.py`
  and every old-path `src/entropic_substrate/<pkg>/__init__.py` stay byte-identical. The leaf-empty
  `__legacy/data/` directory stays for p12.
- No test edits. `tests/test_d1_training_step_split.py` keeps its 5 `__legacy` lines, so its four
  forwarders survive.
- No rewrite of the 33 residual `__legacy` statements. They name `landed: false` modules owned by
  p10/p12. Rewriting them would violate rewrite-on-arrival.
- No hand deletion or hand edit of any forwarder or stub. The generator deletes; the lane verifies.
- No edit to `tools/gen_shims.py`, even though step 0 exists only because of its manifest-keyed
  blind spot (see *Open for the conductor*).
- No regime re-pin, no identity-label bump, no module-set change.
- No edit to `tools/mapping.json`, `.importlinter`, `sprint-manifest.json`, briefs, plan docs,
  engine scripts or CI.

## Allowed files

The escalation door: if correctness would require touching a file outside this envelope, DO NOT
EDIT. Return `DECISION` with the verification you RAN attached as command plus observed output.

- The 41 importer files in the three commit tables, as IMPORT-LINE rewrites only. A ruff
  isort/format fixed point on those same import blocks is included.
- The 30 retired forwarder PAIRS under `src/entropic_substrate/__legacy/`, 60 files, listed in the
  *Forwarder retirement* table. They are DELETED by `tools/gen_shims.py` generate only.
- `.claude/workflows/landed-set/p11-analyze-review-file.json`: `legacy_edge_budget` plus ONE
  appended clause in `note` per commit. Nothing else.
- `.secrets.baseline`, under the STANDING RULE of conductor 2026-09-19 as quoted in the p7 brief.
  It is MEASURED not to reference any file above, so it is predicted untouched.
- `.claude/workflows/landed-set/.emitted.json` is UNTRACKED derived state (FOR-1539). Regenerate it
  (step 0 requires that); never `git add` it.

**Deliberately EXCLUDED:**

- `tests/**`, including `tests/test_d1_training_step_split.py`.
- `tools/**`, including `tools/mapping.json` and `tools/gen_shims.py`.
- `.importlinter`, `.claude/workflows/sprint-manifest.json`, every fragment other than p11's, and
  briefs and plan docs.
- The four identity files, as the control: `tests/test_identity_epoch.py`, `tools/regime_rows.py`,
  `tests/test_regime_rows.py`, `docs/identity/read-old-table.md`.
- The four surviving forwarder pairs, every leaf shim and stub at an old path, every package
  `__init__.py`, and `scripts/**`.

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                        | status                           | ruling as it binds this rung                                                                                                                                                                                                                                                                                                       |
| ------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-WAVES r3                | OPERATOR 2026-09-23              | One serial reconciliation rung after the zip rewrites the sibling-target imports to their final addresses, deletes those forwarders, regenerates shims, re-pins only if a reader-set member changed, re-certifies and publishes. A moved module's old `__legacy` address is a forwarder until the rung rewrites its last importer. |
| RUNG-OWNED                | CONDUCTOR 2026-09-23             | `ladder/gate/ladder_predecessor.py` lines 8/12/14/20 are rewritten by this rung, once.                                                                                                                                                                                                                                             |
| Q1 / tests untouched      | CONDUCTOR 2026-09-18             | No phase before p13 edits an existing test, so the test-served forwarders survive by the derived predicate.                                                                                                                                                                                                                        |
| PKG-FORM                  | CONDUCTOR 2026-09-23 (`44c3dc5`) | Package-form `from entropic_substrate.__legacy import x` counts as importing `__legacy.x`, which is why three survivors are kept by `:35`.                                                                                                                                                                                         |
| ZIP-REMEASURE             | CONDUCTOR 2026-09-23             | The equality arm binds to the highest-ordered fragment; the budget is re-measured with a dated `note` clause; no other fragment field changes. The rung applies the same form at each commit.                                                                                                                                      |
| BUDGET-PER-COMMIT         | CONDUCTOR 2026-09-23             | Every commit writes its own MEASURED budget; the key exists at every commit.                                                                                                                                                                                                                                                       |
| FRAGMENT-PER-COMMIT       | CONDUCTOR 2026-09-23             | Binds landing commits. The rung lands nothing, so `modules` is untouched and `test_module_identity.py` is unaffected.                                                                                                                                                                                                              |
| BUDGET-MONOTONE retracted | CONDUCTOR 2026-09-23 (`795f8e6`) | Equality only. The rung's direction is DECREASING, and that is predicted, not asserted.                                                                                                                                                                                                                                            |
| FACADE-EMPTY              | CONDUCTOR 2026-09-23             | Triggered for `__legacy/data/` and already satisfied by p3's retracted body. No edit; directory removal is p12's.                                                                                                                                                                                                                  |
| P11-FACADE                | CONDUCTOR 2026-09-23             | `__legacy/analysis/` keeps `d1_probe.py` and `ladder_probe.py`; its facade retraction is p12's.                                                                                                                                                                                                                                    |
| DATA-BUILD-RUNG           | CONDUCTOR 2026-09-23 (`c2f64a0`) | p8 moved `data/build.py` to `ladder.capture.custody.build`. The rung rewrites its two remaining src importers, `d1_trainer_runtime.py:14` and `embedding_geometry.py:49`, and the forwarder retires. `test_legacy_data_build_cold_import.py` follows the landed address live, so it stays green unchanged.                         |
| R-PIN                     | CONDUCTOR 2026-09-22             | A re-pin is owed only when a `_READER_MODULES` member is rewritten. MEASURED 0/51 and 0/7, so none is owed; the control is the empty identity-file diff.                                                                                                                                                                           |
| FOR-1539                  | 2026-09-22                       | `.emitted.json` is untracked derived state.                                                                                                                                                                                                                                                                                        |
| FOR-1541                  | CONDUCTOR 2026-09-22             | Branch-only `gt submit` after a green `just ship-it` whose sole failure is the commitizen empty range.                                                                                                                                                                                                                             |
| W2-SCC (not applicable)   | CONDUCTOR 2026-09-23             | W2-SCC partitions LANDING commits by strongly-connected group. The rung moves nothing, so there is no intra-set graph to cycle. The package partition is legal in ANY order: every intermediate tree is valid under the derived forwarder predicate, because an un-rewritten importer keeps its forwarder.                         |
| NO-RUNG-FRAGMENT          | THIS BRIEF, derived from code    | No `w2-reconcile.json` fragment. `highest_ordered_fragment` raises `ManifestError` on any fragment absent from the manifest `sequence`, and the manifest is conductor-owned. The budget therefore lives in p11's fragment, the highest-ordered one present.                                                                        |

## Guard teeth by mutation

House rule 5: break the guarded thing and show RED, then revert in-branch. Both arms go in the PR
body. Expectations marked `[read from code]` were derived from `tools/gen_shims.py` and are what
the lane must EXHIBIT, not report as proven.

| #  | mutation, applied at the tip, reverted after                                                                                                                                                                                                                                                           | must go RED                                                                                                                                                                                                                                                                                                                                                                                                               | notes                                                                                                                                                                                                                                                                                                                              |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T1 | Re-introduce ONE `__legacy` import of a landed target: `ladder/analyze/series.py`'s telemetry import back to `entropic_substrate.__legacy.telemetry`, WITHOUT regenerating                                                                                                                             | (a) `--verify` exit 1 with the MISSING drift message for the forwarder path `entropic_substrate/__legacy/telemetry.py` and its stub `[read from code: second find_manifest_drift over forwarders + stubs]`; (b) `test_legacy_coupling` equality, measured 33 vs 32 `[predicted]`; (c) cold-import sweep FAIL for `entropic_substrate.ladder.analyze.series`, `ModuleNotFoundError` on the deleted forwarder `[predicted]` | MEASURE WHICH, as the commission requires. Then run generate: it RE-EMITS the forwarder pair (`test_gen_shims.py::test_adding_one_legacy_importer_makes_the_forwarder_appear` pins that on a fixture), `--verify` goes green, and (b) STAYS RED at 33. That shows the budget, not the generator, is the tooth against re-coupling. |
| T2 | Commit 1's rewrite applied, generate SKIPPED, then `--verify`                                                                                                                                                                                                                                          | `--verify` exit 1: "2 STALE forwarder(s)", naming `__legacy/analysis/ladder_analysis.py` and `ladder_filing.py` `[read from code: find_stale_forwarders]`                                                                                                                                                                                                                                                                 | GREEN control: run generate and verify again, and the two pairs are gone.                                                                                                                                                                                                                                                          |
| T3 | Blind-spot arm, at the tip: restore two retired forwarder pairs as untracked working-tree files with `git show 55b60aa:<path> > <path>` for `__legacy/analysis/ladder_{analysis,filing}.{py,pyi}`, run `rm .claude/workflows/landed-set/.emitted.json`, then run the canonical generate and `--verify` | `--verify` passes CLEAN, which is the blind spot, while the working-tree `grep -rl '^# Generated by gen_shims' src/entropic_substrate/__legacy \| wc -l` reads 12, not 8 `[read from code]`                                                                                                                                                                                                                               | This proves the file-count instrument is LOAD-BEARING and step 0 is required, not decorative. Revert: delete the four restored files, re-run generate, and the manifest is back to 4 forwarders and the count to 8.                                                                                                                |
| T4 | One OLD-PREFIX import: `series.py` to `from entropic_substrate.telemetry import …`, the root-level compat shim                                                                                                                                                                                         | `tests/test_no_old_prefix.py::test_measured_landed_to_old_prefix_edges_is_zero`                                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                    |
| T5 | Budget mismatch in `p11-analyze-review-file.json`, both directions: measured + 1, then measured − 1                                                                                                                                                                                                    | `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget`, both times                                                                                                                                                                                                                                                                                                                 | Equality catches an inflated and a deflated budget alike.                                                                                                                                                                                                                                                                          |
| T6 | Hand-delete one SURVIVOR, `__legacy/d1_calibration_records.py` + `.pyi`                                                                                                                                                                                                                                | `--verify` exit 1 for a manifest-listed forwarder missing (fixture-pinned by `test_gen_shims.py::test_verify_fails_loud_when_a_manifest_listed_forwarder_is_missing`) and `tests/test_d1_training_step_split.py` collection ERROR `ModuleNotFoundError`                                                                                                                                                                   | Proves the four survivors are load-bearing, so the "4 remain" claim is not slack.                                                                                                                                                                                                                                                  |

## Observability conformance

Empty subject, enforced by the import-only ACCEPTANCE AST instrument rather than measured here.
The diff is import-line-only, so there is no new `print(`, logging call, `configure_logging`, entrypoint or
`MetricSink` site. P-1's rewrite-on-arrival print obligation and the entry-point obligation have an
EMPTY SUBJECT. No `conforming_entrypoints` or `print_allowlist` key changes; the fragment change is
budget plus note only. The three pod entrypoints appear only as cold-import SUBJECTS.

## Planned commit branches

Three commits, not four. ZERO importers live under `ladder/capture/**` `[MEASURED]`, which corrects
the commission's four-way partition. There is one Linear issue per commit, inside the refactor
project: team Form_fxn, key FOR. The ids are PLACEHOLDERS, and the conductor assigns them.

**Order recommendation: gate, then run, then analyze+review+file.** The RUNG-OWNED file is the
reason the rung exists. Landing it first discharges RUNG-OWNED in a one-file commit whose
retirement is exactly its two exclusive forwarders, `analysis/ladder_{analysis,filing}`. That makes
the delete-on-last-importer mechanism, and teeth T2 and T3, visible on the smallest possible diff
before the two large commits. The order changes no final number. The alternative, run then
analyze+review+file then gate, gives budgets 87, 36, 32 and forwarders 26, 8, 4; the final 32 and 4
are the same either way.

| # | Linear    | scope                                                                                                           | files  | lines  | distinct edges removed | budget written `[predicted]` | forwarders after | generated files after |
| - | --------- | --------------------------------------------------------------------------------------------------------------- | ------ | ------ | ---------------------- | ---------------------------- | ---------------- | --------------------- |
| 1 | FOR-TBD-1 | `ladder/gate` (RUNG-OWNED `ladder_predecessor.py`), plus step 0 recorded                                        | 1      | 4      | 4                      | 124                          | 32               | 64                    |
| 2 | FOR-TBD-2 | `ladder/run` (construction, provider, resource, trainer, transport, values)                                     | 17     | 41     | 41                     | 83                           | 24               | 48                    |
| 3 | FOR-TBD-3 | `ladder/analyze` + `ladder/review` + `ladder/file`, the certification evidence, and teeth T1..T6 in the PR body | 23     | 54     | 51                     | 32                           | 4                | 8                     |
|   |           | **total**                                                                                                       | **41** | **99** | **96**                 | **128 → 32**                 | **34 → 4**       | **68 → 8**            |

Per-commit procedure, identical for all three:

1. Rewrite the commit's lines by address map. Dry-run the substitution and preview the diff before
   applying it in place.
2. Run ruff fix and format on the touched files.
3. Run the canonical generate, then `--verify`, then `--preflight`.
4. Read the working-tree `grep -rl` count now, and the `git grep -l … HEAD` count after step 7.
5. MEASURE the budget and write it with the dated note clause.
6. Run the 44-subject cold sweep, plus the binding probe in commit 3.
7. Commit through the hooks, never `--no-verify`. The pre-commit `pytest -m unit` runs the equality
   arm at every commit.

## Open for the conductor (not ruled here)

1. **p10's parent.** Plan line 377 parents p10 on `mythwave/p11-analyze-review-file`. With the rung
   stacked on that tip, p10 either parents on `mythwave/w2-reconcile` or keeps the p11 parent and
   inherits the 34 forwarders. That is a stack-shape ruling, conductor-owned.
2. **A `--verify` blind spot in `tools/gen_shims.py`** `[read from code]`. Stale-forwarder detection
   and deletion both key off the untracked `.emitted.json`. A tree whose manifest is missing or
   stale verifies clean with dead forwarders on disk. This rung mitigates it with step 0 and the
   file-count instrument. The structural fix is a tooling issue outside this envelope, for example
   `--verify` enumerating exact-form shims under `__legacy` independently of the manifest. File it
   before p12, whose `__legacy`-empty postcondition leans on the same mechanism.
3. **Linear ids** for FOR-TBD-1..3, and whether the rung gets a `sprint-manifest.json` entry
   (NO-RUNG-FRAGMENT holds either way).

## Not measured by the drafting session

- Every post-rewrite number: the budgets 124/83/32, the forwarder counts 32/24/4, the file counts
  64/48/8, the 33 residual statements, the per-commit deletion lists, and the T1..T6 RED arms. All
  are `[predicted]` or `[read from code]`, from in-memory recomputation over the unmodified export.
  The drafting session was held to write nothing into the tree.
- Step 0's `git status --porcelain` emptiness after generate at `55b60aa`. It is predicted from
  `55b60aa` being the zip commit that itself ran generate + `--verify` clean.
- `lint-imports`, `just check`, `just pc-ci`, `just ship-it`: none was run.
- `tools/lane_slices.py --allow-dependencies`, which the plan names as the worklist oracle, was NOT
  run. It is a pre-move lane-slice tool. After the zip, the authoritative oracle is `gen_shims`' own
  derived predicate, which the *Scope* snippet uses.
