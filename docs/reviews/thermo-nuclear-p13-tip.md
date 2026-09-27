# Thermo-nuclear review: whole-tree synthesis at `0e55d9d`

## Verdict: REJECT, read as the consolidation backlog, not as a revert signal

The hexagonal refactor already merged at `0e55d9d`. This verdict says the tree **does not yet pass the approval bar**. Several presumptive blockers still stand. They are exactly the input the extract / abstract / consolidate phase exists to take. Behaviour is broadly correct. The bar is structural, and structurally the codebase carries one large duplicated concept, one package cycle, and two correctness defects on evidence-bearing paths.

| # | Presumptive blocker | Bar clause | Consensus | Behaviour risk |
|---|---|---|---|---|
| B1 | Strict canonical-JSON boundary codec reimplemented tree-wide | canonical-helper duplication; code-judo left on table | 6 verified findings + scout + synthesizer grep | extraction is safe; two semantic drifts need explicit rulings |
| B2 | `gate <-> run` package cycle | logic in the wrong layer | scout + synthesizer | none for the move |
| B3 | `D1RunReader` silently drops gap metrics | structural regression on a fail-LOUD path | 1 lens, verified | fix necessarily changes behaviour; that is the point |
| B4 | MLflow cold-start retry orphans runs | structural regression on the live gate | 1 lens, verified | preferred remedy is behaviour-preserving |
| B5 | Fresh trainer duplicates the resume trainer's extracted failure helper | canonical-helper duplication | 2 lenses | none |
| B6 | `scripts/assemble_review_packet.py` pushed past 1000 lines | file-size | scout + synthesizer history check | none |

File-size evidence, measured by `git show <sha>:<path> | wc -l`:

| File | Pre-refactor `8c7a7f7` | Now `0e55d9d` | Reading |
|---|---|---|---|
| `composition/entrypoints/cli.py` | 1133 as `src/entropic_substrate/cli.py` | 1144 | carried debt, not pushed past by this change; remedy is finding 3.3 |
| `scripts/assemble_review_packet.py` | 948 | 1038 | **pushed past 1000 by the refactor**; stands as a blocker until justified or split |

Cross-cutting counts at `0e55d9d`, all measured by grep over `src/`:

| Pattern | Count |
|---|---|
| files defining `_nonfinite` / `_pairs` / `_strict_pairs` | 31 |
| `def _mapping(` definitions | 23 |
| hand-rolled `separators=(",", ":")` canonical dumps | 67 occurrences, 19 of them in `ladder/run`, 10 in `ladder/review` |
| files carrying a local "lowercase SHA-256" check | 27 |
| `# type: ignore[arg-type]` | 19 occurrences in 12 files |

---

## 1. Structural regressions

### 1.1 `D1RunReader` drops gap-valued metrics silently. Fail-LOUD violation on the evidence path
`src/entropic_substrate/ladder/analyze/ladder_analysis.py:121`. Lens: type-boundary. Verified.

- **Problem.** `read_run` pre-filters with `type(getattr(item.quality, name)) in (int, float)` before `scalar_metric(name)`. That makes `D1QualityEvidence.scalar_metric`'s `D1TelemetryGap` raise unreachable. A `D1CosineGap` metric such as `mean_pairwise_cosine_distance` simply vanishes from `StatisticalEvaluation.metrics`. Siblings disagree: `ladder_registration._d1_runs` raises; `d1_capture._cell` channels gaps into `diagnostic_gaps`. Three readers, three outcomes. No test pins the gap case for this reader.
- **Preferred remedy (the verifier's stronger move).** Delete the pre-filter. Call `item.quality.scalar_metric(name)` unconditionally. That is one line removed and matches `_d1_runs`. Do not build a shared gap-policy helper now. Whether analysis readers should ever channel gaps is a separate design question.
- **Behaviour risk: yes, intentionally.** A run with a cosine gap will now raise instead of silently losing data. Pin it TDD-first with a red-then-green gap fixture.

### 1.2 `MlflowSink.__init__` retry loop re-invokes `create_run`, orphaning RUNNING runs
`src/entropic_substrate/ladder/capture/adapters/mlflow.py` startup loop, roughly lines 308-336. Lens: orchestration. Verified.

- **Problem.** `create_run` and the `self._run_id` binding sit inside the same retried `try` as `set_tag` and every `log_param`. A cold-start timeout on the Nth call re-enters from the top, creates a second run, and rebinds `_run_id`. The first run stays RUNNING forever. This is the scenario the retry budget exists for, on the only path that runs the live MLflow stack.
- **Rejected remedy.** Moving `set_tag`/`log_param` under the best-effort guard would weaken the documented startup-abort contract.
- **Preferred remedy.** Split into two phases with identical failure policy. Phase 1 retries `get_experiment_by_name` / `create_experiment` / `create_run` exactly as today. Phase 2 retries only `set_tag` and the `log_param` loop under the same cold-start/deadline policy. It never re-creates the run and never rebinds `_run_id`. Startup still aborts with `MlflowStartupError` on every path it aborts on today.
- **Behaviour risk:** none beyond removing the orphan. Validate with `just container-test`, not unit tests alone. Unit tests skip this path.

### 1.3 `gate <-> run` package cycle *(synthesizer-added; both import edges verified by grep at `0e55d9d`, remedy unverified)*
- **Evidence.** `ladder/gate/ladder_predecessor.py:39` imports `ladder.run.worker.d1_predecessor_loader.verified_git_bundle`. `ladder/run/worker/d1_predecessor_loader.py:37` imports `ladder.gate.ladder_campaign`. It is the only `ladder.gate` import anywhere under `ladder/run`.
- **Diagnosis.** This is a misfiled module, not a bad import. `ladder_predecessor.py` also imports `analyze`, `capture.custody`, and `file.ladder_filing`. It is a post-filing predecessor-chaining orchestrator sitting in the register-gate layer. The architecture sketch's "gate imports from run" inversion is one symptom of that.
- **Preferred remedy.** Move `ladder_predecessor.py` above `file/`, into the layer or composition seam that legitimately depends on analyze + capture + file + run. `verified_git_bundle` is git-bundle proof machinery with only `kernel` and `run.provider` deps. It can stay in `run`. Deleting the misplacement dissolves the cycle without inventing a port.

### 1.4 `d1_output_access.py`: an ACL lease the worker never takes
`src/entropic_substrate/ladder/analyze/d1_output_access.py`. Lens: layering. Verified. Git history shows it was dead from its introducing commit.

- **Problem.** A `setfacl`/`getfacl` subprocess lease sits in the statistics package and has zero production callers. Its docstring promises isolation of `/d1/output`. `d1_external_worker.py` actually secures that mount with `os.fchmod(..., 0o600)`.
- **Preferred remedy.** Split it. The in-scope refactor is relocating the module and its test beside `review/d1_external_worker.py`. **DECISION-NEEDED** for the operator: retire it, or wire `grant`/`revoke` into the worker lifecycle. Wiring it in changes runtime behaviour.

---

## 2. Missed code-judo

### 2.1 One concept, one kernel module: the strict canonical-JSON boundary codec *(cross-cutting consolidation; synthesizer grouping of verified findings)*
These verified findings are facets of one missing primitive. Treat them as one backlog item, not eight:

| Facet | Verified finding site(s) |
|---|---|
| duplicate-key `object_pairs_hook` + non-finite `parse_constant` | `run/provider/d1_runpod_adapter.py:61-73` and 5 more in `ladder/run`; `review/d1_external_analysis_control.py:117-137` and 6 more in analyze/review |
| exact-key-set check | `kernel/verdict/d1_parsing.py:25-49`, `kernel/artifacts.py:331-334` `object_fields`, `d1_statistical_gaps.py`, `resource_provenance_codec.py`, `d1_evaluation_values.py`, plus 7 `_object` variants in analyze/file; `capture/records/d1_telemetry_payloads.py` `exact_fields` *(synthesizer-found)* |
| `_mapping` / `_exact` / `_text` | `register/d1_registration_codec.py:104-124`, `register/d1_launch_attempt_codec.py:41-61`, `register/d1_plan.py:59-78`, `gate/ladder_campaign.py:48` |
| canonical dumps reimplemented instead of `kernel.identity.d1_primitives.canonical_json` | `composition/wiring/d1_trainer_cli_services.py:69-81`, `composition/entrypoints/d1_trainer_worker_command.py:44-51`, `d1_trainer_mailbox_command.py:62-69`, `run/transport/d1_trainer_process_contracts.py:143-150` |
| lowercase-SHA-256 check | 27 files measured; see 2.2 for the naming collision |

- **Preferred remedy.** Create one kernel module, e.g. `kernel/codec/strict_json.py`, holding `strict_loads(raw, *, fail)`, `exact_keys(actual, required, optional=frozenset())` returning `(missing, unknown)`, `require_mapping(value, *, fail)`, and `require_lowercase_sha256(value, *, fail)`. It takes a single `fail: Callable[[str], NoReturn]` parameter, so every site keeps its own exception type and its message shape. Keep `d1_parsing.object_exact` as a thin D1 wrapper over `exact_keys`. Point the composition-layer canonical dumps at the existing `canonical_json`.
- **Do not fold `d1_cli.py:180` `_checked_payload` into `canonical_json`.** It is a strict admission gate. `thaw_for_digest` would silently widen it to accept `MappingProxyType`. If it is unified at all, name it `assert_json_native` and never thaw.
- **Behaviour risk: two drifts, file separately as explicit rulings, never folded into the extraction commit:**
  - `_text` trim semantics. `d1_launch_attempt_codec.py:58` accepts untrimmed non-blank strings. `d1_registration_codec.py:121` and `ladder_campaign.py:48` reject them.
  - SHA case semantics. `kernel.identity.d1_primitives.require_sha256` accepts mixed-case hex via `string.hexdigits`. Every untrusted-boundary site requires lowercase. Never point boundary sites at the looser kernel function.
- **Scope note.** Leave `d1_plan.py` out of the pairs / nonfinite / round-trip part. It validates already-decoded objects.

### 2.2 Two functions named `require_sha256` with different contracts *(synthesizer-added; both definitions verified by reading at `0e55d9d`)*
- `kernel/identity/d1_primitives.py:37`: `(value: str, *, label) -> None`, case-insensitive.
- `ladder/capture/records/d1_telemetry_payloads.py:68`: `(value: object, *, path) -> str`, lowercase-only, raises a telemetry gap.
- **Remedy.** This capture version is already the lowercase-strict contract that 2.1 needs. Promote its predicate into the new kernel module as `require_lowercase_sha256`, and rename or retire the collision. Today a reader cannot tell from the name which contract an import gives them.

### 2.3 Fresh trainer duplicates the resume trainer's extracted failure helper
`ladder/run/trainer/d1_fresh_trainer.py:234-315` vs `d1_resume_trainer.py:210-294`. Lenses: code-judo, spaghetti. Verified.

- **Preferred remedy (the stronger move).** Move `_trainer_failure` plus a single reconciled `_event` verbatim into `trainer/d1_trainer_runtime.py`. Type it against the existing `D1FreshJournal`-shaped structural protocol, with `.state` and `.append()`. Keep returning `(handoff, failure_cause)` so each caller keeps its one-line `raise ... from`. Do not parameterize by exception class.
- **Behaviour risk:** none.

### 2.4 E-series optimizer and autocast duplicated
`historical/e_series/field_pretrain.py:158-181, 227-243` vs `train.py:276-296`. Lenses: code-judo, layering. Verified.

- **Remedy.** Extract free functions `build_adamw_decay_groups` and `autocast_for` next to the already-shared `cosine_lr`. Both call sites delegate. Delete the defensive docstrings that justify the copy.
- **Behaviour risk:** none.

### 2.5 `D1ExperimentFamily` re-proves its own invariant
`ladder/register/d1_family.py:263-279`. Verified.

- **Remedy.** Delete the `allowed_to_vary` manifest-diff loop. The preceding field-equality loop is already sufficient.
- **Behaviour risk:** none.

### 2.6 Scout-sourced duplications *(scout mechanical findings, not adversarially verified)*
- `ladder/review/d1_machine_analysis_binding.py:50-70` vs `d1_machine_analysis_verification.py:40-60` have identical `_fail` / `_mapping` / `_rows`. These fold into 2.1.
- `ladder/capture/records/d1_telemetry_values.py:377-385` vs `ladder/run/trainer/d1_evaluation.py:199-207` have identical `state_mean` / `state_variance` / `state_covariance_eigenvalues` accessors. The run trainer should read the capture value type rather than restate it.

### 2.7 Smaller verified extractions
- `core/observation/byte_codec.py:65-87` vs `permuted_byte_codec.py:108-131`. Extract only `validate_observation_ids` into `observation/contracts.py` and leave the one-line encode guard inline.
- `core/corpus/batching.py`. Delete `_starts` and call `_predictive_starts(1)`, since its docstring already asserts equivalence. Add one `_to_device(tensor)` helper for the 5 device-dispatch sites.
- `core/probe/score_gain.py:102-122` vs `predictive_score_gain.py:97-114`. Share the gain-statistics reducer over a small protocol.
- `composition/{cli,capture_cli,span_cli,source_cli}.py`. Add `resolve_d1_relative_root(root, value, *, label)` to `kernel/artifacts.py`.

---

## 3. Spaghetti / branching

### 3.1 `D1SequentialOperatorController` fresh vs resume
`ladder/run/campaign/d1_operator_controller.py`, finalize bodies near 398-414 and 590-606. Verified, downgraded to Minor.

- **Rejected remedy.** The proposed one-flow merge is not behaviour-preserving. prepare and launch genuinely differ: `admit_launch` vs none, `start_running` vs `resume`, the extra `TRAINER_STARTED` transition, and the extra segment validation.
- **Preferred remedy.** Extract only the checkpoint-custody decision into `_persist_finalization_artifact(...)`, called from both finalize closures.

### 3.2 `kernel/runtime/config.py` `_build` special-cases `TrainConfig.betas`
Lenses: spaghetti, type-boundary. Verified.

- **Remedy.** Move the coercion into `TrainConfig.__post_init__` and delete the branch from the generic builder.
- **Behaviour risk:** none.

### 3.3 `cli.py` `main()` dispatches three ways
`composition/entrypoints/cli.py:1062-1144`. Lenses: spaghetti, abstraction. Verified.

- **Remedy.** Fold the six `if args.command ...` blocks into the existing `handlers` dict, the same way `field-pretrain` already is. This is also the decomposition lever for the 1144-line file.
- **Behaviour risk:** none.

### 3.4 `d_ladder` resources ternary
`historical/d_ladder/d_ladder_training.py:115, 174` and `d_ladder_execution.py:249-251`. Verified.

- **Rejected remedy.** A null object is unsafe. `resources is None` is load-bearing for the scientific-mode invariant at `d_ladder_execution.py:147`.
- **Preferred remedy.** Add a local `_resource_phase(resources, phase)` helper used only at the three `measure()` sites.

---

## 4. Boundary / type contract

### 4.1 `object`-typed seams papered with `type: ignore` *(synthesizer grouping; the 19-occurrence count is measured, the per-site remedies come from verified findings)*
One pattern shows up at five verified sites. The sweep list is `grep -rn "type: ignore\[arg-type\]" src`.

| Site | Preferred remedy |
|---|---|
| `run/closure/d1_billing_finalizer.py` | change the signature to `finalize_d1_billing(request: D1BillingFinalizationRequest)` and build the request at the 3 call sites; deletes 7 ignores |
| `composition/wiring/d1_campaign_controller_factory.py:54-55, 213, 227, 266, 286` | reuse the existing `D1RunPodPort` and `D1TrainerMailboxObjectPort`; add one thin `D1R2Runtime` protocol |
| `capture/resource/resource_meter_strategy.py:138-151` `restore(**ports)` | typed keyword-only params **including `max_power_gap_seconds`**; the original remedy omitted it and would `TypeError` on resume with a power registration |
| `run/resource/d1_resource_activation.py:233-255` | compute `gap` once and make one explicit keyword call; no dict splat |
| `composition/entrypoints/d1_trainer_mailbox_command.py` | annotate with the existing `D1TrainerStartReportPublisher` |

### 4.2 `MetricExtractor.extract(event: object)` merges two contracts
`capture/ports/run_descriptor.py:97-100`. Lenses: type-boundary, layering. Verified.

- **Remedy.** Split into a JSONL extractor and a journal extractor protocol. In the **same commit**, narrow each concrete `extract` signature. The split alone is a no-op under parameter contravariance. Type `D1MlflowObserver.__init__`'s `extractor`.

### 4.3 Verdict value objects carry closed vocabularies as bare `str`
`kernel/verdict/d1_types.py`. Verified.

- **Remedy.** Promote `CLAUSE_STATUS`, `EVIDENCE_CLASSES`, `CONTRAST_KINDS`, `CONTRAST_DECISIONS` and the review statuses to `StrEnum`. Add `__post_init__` validation matching `d1_contract.py`'s registered types.
- **Caution.** This file sits on the verdict path. Check identity-bearing hashes before touching it. Per the docs-sweep ruling, some kernel modules re-pin identity on any edit.

### 4.4 Minor
`core/latent/horizons.py`: add one typed `_head(horizon)` accessor to replace the two casts.

---

## 5. File size / decomposition

| File | Lines | Remedy |
|---|---|---|
| `scripts/assemble_review_packet.py` | 1038, up from 948 | presumptive blocker: split by packet section or justify |
| `composition/entrypoints/cli.py` | 1144, up from 1133 | carried debt; 3.3 plus moving handler families into their existing `*_cli.py` modules brings it under |

## 6. Modularity / abstraction: one dead-surface deletion commit
All verified behaviour-neutral:
- `ladder/gate/d1_family_adapter.py`: delete; promote `_family_identity` if ever needed
- `EncoderShape` in `kernel/identity/d1_primitives.py:195-215`
- `ByteObservationCodec.encode_bytes` / `decode_ids`
- `core/latent/targets.py` facade: repoint 5 test imports
- `d1_configs_dir()`: repoint 2 internal and 10 test imports
- `load_ladder_model` in `historical/d_ladder/d_ladder_artifacts.py`

Also, not dead but misplaced: move `GatedKappaNorm` and `KappaNormStats` from `core/geometry/kappa_norm.py` into `core/substrate/blocks.py` beside `build_norm`. Geometry keeps the stateless maths.

## 7. Legibility
- `capture/custody/d1_capture_ports.py`: validate via `_ = self.binding` instead of re-spelling the `D1DvcBinding(...)` construction three times.
- `kernel/artifacts.py` dir_fd descent loop, 3 copies. **Do not extract as proposed.** The helper would double-close or leak an fd on a mid-loop failure. Only extract with end-to-end error-path ownership inside the helper, scoped as a contract change.
- `historical/d_ladder/d_ladder_remote.py` partial materialization. Preferred fix: `shutil.rmtree(root, ignore_errors=True)` in `run_remote_worker`'s existing `except`. Do not buffer N checkpoints in memory.

---

## Relay integrity: remedies replaced by the verifier's stronger move
The original remedy was rejected as behaviour-changing or incomplete, and the report above carries the replacement, for: the registration codec extraction (2.1), the MLflow retry (1.2), `D1RunReader` (1.1), the d_ladder remote workspace (7), the canonical-JSON reuse in `d1_cli.py` (2.1), the d_ladder resources ternary (3.4), the operator controller (3.1), the dir_fd descent (7), the launch-attempt codec extraction (2.1), `restore(**ports)` (4.1), the external-analysis control SHA reuse (2.1), and `d1_output_access` (1.4). A reader who opens the raw findings must not implement the original remedy text for these items.

## What's strong
- **Layer discipline mostly holds.** The scout found exactly one inverted import edge in a 480-file tree, and it traces to a single misfiled module.
- **Typed refusal is already the norm.** `D1TelemetryGap`, `D1RegisteredContrast.__post_init__` and the `StrEnum`s in `resource_provenance.py` show the target posture. Most findings ask stragglers to match an existing in-repo pattern rather than invent one.
- **The canonical primitives already exist.** `canonical_json`, `object_fields` and `resource_meter_strategy`'s ports are all in place. Consolidation is mostly deletion toward established homes.
- **Symlink-race-safe artifact I/O.** The `dir_fd` / `O_NOFOLLOW` / atomic-publish machinery in `kernel/artifacts.py` is correct as written, which is why its dedup must be done carefully.

## Verdict against the bar
**REJECT.** Four presumptive-blocker classes stand without justification at `0e55d9d`:

1. **Canonical-helper duplication and code-judo left on the table.** B1 and B5, measured at tree scale.
2. **Logic in the wrong layer.** B2, the `gate <-> run` cycle.
3. **A file pushed past 1000 lines.** B6, `assemble_review_packet.py`, measured 948 to 1038.
4. **Structural regressions on evidence-bearing paths.** B3 violates the repo's own fail-LOUD rule; B4 sits on the live gate.

Every other finding is Major or Minor and would on its own merit APPROVE_WITH_FOLLOWUPS. None of this argues for reverting the merge. It is the ordered input to the consolidation phase.

## Next steps, mapped to the extract / abstract / consolidate phases
1. **Fix-first, TDD, own branch:** 1.1 `D1RunReader`, with a red gap fixture first; 1.2 MLflow two-phase startup, gated by `just container-test`. These are correctness fixes, not consolidation, so they should not wait on it.
2. **Consolidate:** the strict-JSON kernel module (2.1 + 2.2 + 2.6 first bullet). It has the largest site count and the clearest trigger with 2 or more consolidation sites. Land the extraction behaviour-preserving, then file the `_text` trim and SHA case drifts as two DECISION-NEEDED rulings.
3. **Restructure:** move `ladder_predecessor.py` out of `gate/` to break the cycle (1.3).
4. **Type-seam sweep:** 4.1 + 4.2 in one pass over the 12 files carrying `type: ignore[arg-type]`.
5. **Extract:** 2.3 trainer failure helper, 2.4, 2.7, 3.1 to 3.4.
6. **Delete:** the section 6 dead-surface commit. Split or justify `assemble_review_packet.py`.
7. **Operator ruling pending:** retire or wire the `d1_output_access` ACL lease (1.4).
