# `src/entropic_substrate` Hexagonal Refactor Plan

Written 2026-09-17 on `refactor/hexagonal-redesign`. Approved plan, not a proposal: the operator
ruled the skeleton, the scope, the identity dispositions and the execution shape on 2026-09-17.
This document is the forward contract for 15 phases; the per-phase briefs under
`.claude/workflows/briefs/` are authoritative for each lane and this plan is their index.

**Evidence labelling.** Every number below carries `[M, <source>]` measured or `[E]` estimated.
Measured and estimated values never share a row without a per-row label. No number here is
inferred by analogy or approximated from folklore.

---

## Context

`src/entropic_substrate` holds three eras in one tree: the E byte substrate, the D predictive-state
ladder, and the H sentence predictor. `docs/architecture.md` already declares a modular hexagonal
monolith; the code half-follows it.

A 50-agent survey ran 2026-09-17 (39 haiku scouts, 6 sonnet cross-checks, 1 historian, 3 blind opus
proposals, 1 sealed-rubric arbiter). Proposal C (ladder-cadence axis) won 81/100 over B (one root
hexagon) at 64/100 `[M, survey arbiter report §4, retained outside the repo]`; A was inadmissible on an unmarked
identity Door change. Two Plan agents produced the phase decomposition and the execution overlay
this document carries.

The historian's finding frames the value: D1 never reached preregistration or capture, and H0 is
live but local-test gated only. **The refactor exists to give H1+ a cadence-shaped home, not to
rescue D.**

---

## Blast Radius Summary

| quantity | value | evidence |
|---|---|---|
| epoch (pinned) | `bbe7976` on `refactor/hexagonal-redesign` (== `dev`) | `[M, git rev-parse]` |
| files in the package | 332 | `[M, graph-bbe7976/summary.json "lines"]` |
| edge-bearing modules | 326 | `[M, graph-bbe7976/edges.json]` |
| import edges | 1783 | `[M, graph-bbe7976/edges.json]` |
| layer inversions | 142 | `[M, tools/remap_check.py at bbe7976]` |
| package-level cycles | 11 | `[M, graph-bbe7976/summary.json "package_cycles"]` |
| module-level SCCs | 2, both size 2 (`experiments.d1_family + d1_runs`; `d_ladder_artifacts + d_ladder_execution`) | `[M, Tarjan over edges.json]` |
| flat root modules, no package boundary | 118, 38% of lines | `[M, survey epoch 7ba2788]` |
| survey epoch, for provenance | 330 modules / 1768 edges, same 142 / 11 | `[M, graph-7ba2788/]` |
| delta 7ba2788 → bbe7976 | +2 modules, +15 edges, all outbound from `h_probe_command` and `h_probe_controller`; 142 / 11 UNCHANGED | `[M, node-set diff]` |
| highest eager fan-out | `kernel.identity`, 123 importers | `[M, tools/blast_radius.py]` |
| second highest | `kernel.lifecycle`, 79 external importers | `[M, tools/blast_radius.py]` |
| H lane external importers | 5 (`cli`, `d1_git_provenance_adapter`, `d1_predecessor_loader`, `data.fineweb`, `experiments.h_preservation`) | `[M, blast_radius.py]` |
| test files to rewrite at p13 | 395 | `[M, find tests -name "*.py"]` |

**The decisive structural fact.** The module graph is a DAG: 2 SCCs, both of size 2 `[M]`. Every
one of the 11 package cycles is therefore an *aggregation artifact* of the current package
boundaries, not a real import cycle. **This is a topological cut problem, measurable on paper
before any file moves**, which is what makes the phase decomposition executable rather than
exploratory.

**Measured after-state** for the winning mapping: 4 inversions / 5 package cycles
`[M, remap_check.py on mapping.json × graph-7ba2788, reproducing proposal C's submitted report]`.
Refining the two large buckets (`ladder/run` 100 modules, `ladder/capture` 54) into the declared
sub-packages raises the cycle count to 16, with 11 hidden inside the buckets `[M, same oracle,
refined mapping]`. That is the arbiter's dissent made numeric, and it is why p8 and p9 decompose
the buckets rather than shipping them whole. Those 11 are cut by seven module promotions
(DAG-depth evidence) and five declared symbol splits; the honest residue is class-level and named
below, not promised away.

---

## Public Facade (Contract-Pin)

The skill's "public facade" is Rust vocabulary. In this Python tree the facade is the set of
**import-linter contracts, identity fixtures and doors** that consumers bind to. Contract-pin-first
means all of them are frozen and test-verified at p0, before a single module moves.

| contract surface | kind | owned by | frozen_after | pinned in test? |
|---|---|---|---|---|
| `.importlinter` layers over the LANDED layers `kernel < core < ladder < historical < composition`, `__legacy` OUTSIDE the contract until p13 | door | p0 | p13 | yes, `lint-imports` in `just check` |
| `.importlinter` forbidden: `core.*` ↛ mlflow, boto3, botocore, dvc, runpod, huggingface_hub, datasets | door | p0 | p0 | yes |
| `.importlinter` forbidden: `mlflow` importable from `ladder.capture.adapters.mlflow` only | door | p3b | p13 | yes |
| `h-source-reader-v2` → `v3` | door | p4 | p4 | yes, `tests/test_identity_epoch.py` |
| `h-checkpoint-reader-v1` → `v2` | door | p4 | p4 | yes |
| `docs/identity/read-old-table.md` | door | p4 | p4 | yes, digests asserted |
| D1 reader source closure, 74-entry manifest (52 retracted at p0; 73 measured at p0; 74 measured at p1 when `paths.py` joined the closure, PR #286) | door | p2 (RETIRED) | p2 | yes, until retracted |
| D1 external-analysis control envelope, `reader_source_manifest_base64` | door | p2 (REMOVED) | p2 | yes |
| `CHECKPOINT_FORMAT_VERSION`, `_TAU_ERAS`, both codec construction digests, every `*-v1`/`*-v2` literal | door | p0 | p0 | yes |
| sentinel-singleton identity over every dotted path, both spellings | contract | p0 | p3 | yes, `tests/test_module_identity.py` |
| CLI `--help` surface + the frozen dotted-name literal set | contract | p0 | p13 | yes, `tests/test_cli_surface.py` |
| packaging discovery + basedpyright / pre-commit include regexes | contract | p0 | p13 | yes, `tests/test_packaging_identity.py` |
| `tools/gen_shims.py` shim form | door | p3 | p3 | yes, `tests/test_shim_layer.py` |
| landed-set FRAGMENT convention, `.claude/workflows/landed-set/<id>.json` (modules landed, targets, `legacy_edge_budget`) | contract | p3 | p13 | yes, schema assertion |
| `tests/test_legacy_coupling.py` (landed ↔ `__legacy` edges ≤ merged budget; every shim static-import form) | contract | p3 | p13 | yes |
| `tools/mapping.json` (the machine input; every node mapped) | contract | p0 | p5 | yes, p0 coverage gate |
| `MetricSink` port, `ladder/capture/ports/metric_sink.py` | door | p3b | p3b | yes, `tests/test_observability_conformance.py` |
| run-descriptor door: `TrackingRunDescriptor` + `build_sink_for_run` | door | p3b | p3b | yes |
| parameterized entrypoint source (`[tool.entropic_substrate.entrypoints]`) | door | p10 | p12 | yes, p0 CLI pin |
| `__legacy/` empty postcondition | door | p12 | p12 | yes, `tests/test_legacy_empty.py` |
| the old dotted namespace, RETIRED | door | p13 | p13 | yes |

**Ambiguous surfaces** go to `linear-empirical-triage` before they become tickets. The ones still
open appear in the DECISION-NEEDED register below, not in this table.

---

## Target topology (ruled)

```
src/entropic_substrate/
  kernel/        L0  identity · lifecycle typestate · artifacts · evidence_class · verdict · runtime
                     (no I/O; imports nothing above)
  core/          L1  geometry · substrate · observation · latent · sentence · corpus · probe
                     (the science; no mlflow/dvc/boto/runpod)
  ladder/        L2  seven stage hexagons, import PEERS communicating only via kernel artifacts
                     and typestate:
                     register{rungs/,arms/,codec/}
                     run{values,transport,provider,resource,training,construction,campaign,
                         closure,ckptsync,worker,journal,spanprobe}
                     capture{custody,records,resource,reader,ledger,ports,adapters}
                     analyze · review · file · gate
  historical/    L3  governed READ-ONLY: e_series (train.py, field_pretrain.py, E model stack,
                     E charts) · d_ladder (11 modules + 3 probe modules)
  __legacy/      L3  quarantine; EMPTY after p12
  composition/   L4  cli.py · wiring/ (the only place a concrete adapter is chosen)
                     · entrypoints/ (python -m workers)
```

Layer contract `kernel < core < ladder < historical|__legacy < composition`, enforced by
import-linter from p0 onward.

---

## PRIORITY AMENDMENT (ruled 2026-09-17): observability wired through DURING the refactor

**Measured gap.** The E-series pattern exists and works. `logging.py` is the loguru substrate
(`configure_logging` at CLI entry: human stderr, report stdout, structured JSONL; loguru is the
only logging import in `src/`, zero stdlib-logging users) `[M, grep over src/]`. `sinks.py` is the
`MetricSink` fan-out: `JsonlSink` authoritative → `MlflowSink` best-effort via `CompositeSink`,
built once at `cli.py:445`; `mlflow` is imported lazily only in `sinks.py` and `cli.py`
`[M, source read]`.

Only the E entry points use either: `cli`, `train`, `field_pretrain`, `mlflow_aggregate`,
`capture_results`, `checkpoint`, `telemetry`, and 2 analysis modules `[M]`. The D lane emits
nothing through them (0 imports; it carries its own `D1JournalObserver` / `D1MlflowObserver`
protocols at `d1_observability.py:39,64` and `d1_tracking_descriptor` at `d1_tracking.py:109`)
`[M, source read]`. The H lane has 8 bare `print(` sites, its own JSONL journal
(`sentence_prediction/journal.py`, `worker_command.py`) and no MLflow path at all `[M]`.
**The live gate has therefore never exercised an H run.**

**The rule.** Same E-series wiring, everywhere, as each package lands.

1. **p3b lands the seam FIRST**, before the H tracer. Substrate → `kernel/runtime/logging.py`;
   the `MetricSink` Protocol (`sinks.py:209`) → `ladder/capture/ports/metric_sink.py`; the three
   sinks (`JsonlSink` `sinks.py:225`, `MlflowSink` `sinks.py:333`, `CompositeSink` `sinks.py:632`)
   → `ladder/capture/adapters/{jsonl,composite,mlflow}.py`, with `adapters/mlflow.py` the ONLY
   module in the tree allowed to import `mlflow`; `TrackingRunDescriptor`
   (`tracking_contracts.py:32`) + `build_sink_for_run` (`sinks.py:729`) become the run-descriptor
   door; D1's observer protocols are RE-BASED as adapters of the one port, so no second port
   survives. E entry points keep working through shims. The p3b `ship-it` live gate proves the
   moved MLflow path end to end. All five module destinations are `[M, tools/mapping.json]`.
2. **Rewrite-on-arrival is extended.** When a phase moves a module it also (a) routes its
   diagnostic `print(` calls through `logger`; (b) routes its metric emission through `MetricSink`
   with the run descriptor; (c) makes every `python -m` entry point and CLI subcommand it moves
   call `configure_logging` at entry, never at import.
   Two machine-consumed stdout sites stay raw `print` and are pinned by p0's CLI snapshot:
   `telemetry final-eval` and `cli eval --json`.
3. **p4 is the first proof.** `core/sentence` training and evaluation emit through a
   `CompositeSink` (JSONL authoritative, MLflow best-effort) under a descriptor; the worker
   configures the substrate; p4 keeps `liveRequired:true` so `just ship-it` runs the MLflow
   container gate on the H lane for the first time.
4. **Named exception to "zero behaviour change".** Observability conformance is the ONE
   behaviour-change class permitted inside a structural phase. Scope: log routing, metric routing,
   entry-point configuration. Measured scientific behaviour (losses, metric values, identities)
   stays frozen. The identity fixtures and the numerics tests are the tripwire.

**Teeth, proven by mutation.** A forbidden contract pinning `mlflow` to one importer (mutation:
`import mlflow` in `core/sentence` → RED). `tests/test_observability_conformance.py` with three
arms: zero bare `print(` in landed packages outside the two pinned sites (mutation: add one →
RED); every entry point's first stderr line carries the substrate's human prefix (mutation: drop
`configure_logging` from `worker_command` → RED); every landed emitter resolves its sink through
the port, so `MetricSink` construction sites are the composition root only (mutation: construct
one elsewhere → RED). p13's postconditions add: single `mlflow` importer, zero unconfigured entry
points, zero `print` outside the allowlist, D1 observer protocols gone.

**Consequence for the archived Observability sprint (ruling O-2).** Its `p1-loguru` and
`p2-mlflow` are marked `retired`, note "absorbed into refactor p3b + rewrite-on-arrival rule".
Retired, not abandoned.

---

## The mechanism rule that makes phases disjoint

**Rewrite-on-arrival.** A file's import lines are rewritten *exactly once*, by the phase that moves
that file. A phase never rewrites importers outside its own touch set; they keep the old dotted
name and resolve through the re-pointed shim until their own phase moves them.

Four consequences, all load-bearing:

1. `touch_set` per phase = modules moved + their shim files. Disjoint by construction, which is
   what makes the waves legal at all.
2. Per-module physical shims beat a single redirect table. One table file would be a hub every
   phase edits, serialising the sprint.
3. Destination packages ship an **empty `__init__.py`** during structural phases, and so does the MOVED copy of each of the seven re-export facades under `__legacy/` (RULED 2026-09-20, p3 design gate: a moved facade body re-enters its sibling leaf shim mid-import, load-order dependent, measured with a control). Justified: the
   re-export facades have zero measured inbound edges `[M, arbiter-reproduced]`. RETRACTION
   (house rule 4, conductor, 2026-09-20, p3 design gate): this item previously said "the only two
   re-export facades in the plan (`data/__init__.py`, `predictive_state/__init__.py`)"; MEASURED
   at de61455, SEVEN package `__init__.py` files re-export by absolute old-prefix imports (data 5,
   geometry 1, init 1, layers 1, models 2, observation 3, predictive_state 7 statements). Without this rule
   `ladder/run/worker/__init__.py` would be co-owned by p4 and p10.
4. **There is NO import-linter ignore list** (DN-P1, ruled). `.importlinter` is edited exactly
   twice: p0 creates it, p13 adds `__legacy` and turns on full enforcement. From p3b to p12 it
   declares the layers contract over the LANDED layers `kernel < core < ladder < historical <
   composition`, with `__legacy` deliberately OUTSIDE the contract, so edges to and from the
   quarantine are simply not linted while landed-layer inversions are always ZERO and strictly
   enforced. Quarantine coupling is bounded instead by `legacy_edge_budget` in each phase's own
   disjoint fragment `.claude/workflows/landed-set/<id>.json` (modules landed, targets, budget),
   which `tools/gen_shims.py` merges to emit and re-point shims and which
   `tests/test_legacy_coupling.py` checks against the AST graph. Two lanes of one wave therefore
   never share a file, and no phase but p0, p3b and p13 touches `.importlinter` (p3b adds exactly one forbidden contract, the mlflow-only-from-adapter door, once its package exists; ruled 2026-09-19, p3b is a serial rail so this is not a hub).

**Shim form (required).** AMENDED 2026-09-20 (operator ruling, option (b) of the p3 design-gate DECISION): every leaf shim is a PAIR, the `.py` alias plus a `.pyi` stub of derived explicit `from <target> import <name> as <name>` re-exports (not a star import: it would honour the target's `__all__` and trip ruff F403, measured 2026-09-20), because basedpyright cannot see through the `sys.modules` rebind (measured: 1741 old-prefix from-imports in 296 quarantined files, 80 names in the 7 restored facades); the stub is type-level only, never imported by CPython, invisible to grimp and to `tools/import_graph.py`, and is deleted with its shim at p13.

```python
import entropic_substrate.__legacy.<x> as _target
sys.modules[__name__] = _target
```

A real static import so import-linter's `grimp` backend sees the chain, then the rebind. A
string-only `importlib.import_module` shim is an opaque literal to a static analyser: the layers
contract goes **vacuously green for every edge routed through a shim**. p0 demonstrates that
blindness on a deliberately illegal edge, and p3 repeats the demonstration on a real shim. This is
the single most dangerous silent failure in the mechanism, and it is proven by mutation, not
argued.

**Generator contract, redesigned 2026-09-18 (four defects closed, DN-P1 gen_shims v2).** The
generator is being redesigned in parallel with this plan:

1. **Fragments only ADD.** A module absent from every phase's landed-set fragment defaults to
   `__legacy` — no module ever migrates between fragments as it lands, so p3's own fragment is
   empty (or carries only the convention and its budget) and no later phase edits it. This closes
   the ten-writer hub on p3's fragment that a migrate-on-land contract would otherwise create.
2. **The generator never rebinds or writes a package `__init__.py`.** Only LEAF modules get the
   3-line `sys.modules` alias shim. p3 restores each original package `__init__.py` at its old
   (now `__legacy`) location, unmodified, after the wholesale `git mv`; its ABSOLUTE old-prefix re-export
   imports resolve through the sibling leaf shims exactly as they did before the move (RETRACTION,
   house rule 4, conductor, 2026-09-20: this sentence previously said "relative imports"; MEASURED at
   de61455, `grep "^from \.\|^import \." src/entropic_substrate/*/__init__.py` matches nothing --
   no package `__init__.py` in the tree uses a relative import). A multi-destination package
   key (7 of 12 measured from `tools/mapping.json`) therefore never hits the single-target rebind
   hazard, because no package key is ever rebound at all.
3. **A fragment naming a package key or the root key fails loud.** `mapping.json` carries the
   package root itself (`entropic_substrate` → `entropic_substrate.composition`); the generator
   refuses rather than silently rebinding the tree root or a package's `__path__`.
4. **`--verify` checks an emitted-files manifest only**, never an `rglob` over `out_dir` — a real,
   restored package `__init__.py` is not a shim and must not be asked to look like one.

Canonical invocation, pinned so `out_dir` is never left to prose:

```
uv run python tools/gen_shims.py .claude/workflows/landed-set tools/mapping.json src \
  --manifest-out .claude/workflows/landed-set/.emitted.json
```

The manifest is derived, untracked state (gitignored since 2026-09-22, FOR-1539): every lane
regenerates it in the step that verifies its shims, and no phase commits it, so the wave lanes
share no file through the generator.

**Honest caveat on wave legality.** Under eager rewriting, p6 and p7 would share 8 importer files
`[M, core.latent × ladder.register]` and wave 1 would be illegal. The waves are a consequence of
rewrite-on-arrival, not of the packages' independence.

---

<a id="p0"></a>

## p0-contract-pin

*tracer · serial · `liveRequired:true` · 0 modules move · branch `mythwave/p0-contract-pin`*

Contract-pin-first. You cannot refactor behind a seam you have not drawn. Everything a later phase
consumes is created and test-verified here, before anything moves.

**(a) Graph re-derivation gate.** Re-derive at a clean HEAD; restate 332 files / 326 edge-bearing
modules / 1783 edges `[M, graph-bbe7976/]`. Map every unmapped module or the phase FAILS. Four
modules were unmapped against the survey's mapping and are now carried by the repo's
`tools/mapping.json`: `git_environment` → `kernel/runtime`
(12 lines, zero outbound, two D1 importers, so it is NOT H-lane despite the ruling's wording,
ruling E-12) `[M]`; `h_probe_allocation` → `ladder/run/worker` (99 lines, zero importers, seven
outbound, so free in any phase) `[M]`; `h_probe_command` and `h_probe_controller` →
`ladder/run/worker`, both added since the survey, +15 edges all outbound `[M, node-set diff
7ba2788 → bbe7976]`.

**(b) Identity fixtures at the old epoch.** One test that computes and compares as literals:
`h-source-reader-v2`, `h-checkpoint-reader-v1`, the 73-entry D1 closure manifest (path + sha256; the survey's 52 was retracted at p0, measured 2026-09-18),
`CHECKPOINT_FORMAT_VERSION`, the `_TAU_ERAS` tuple, both observation-codec `construction_digest`s,
every `*-v1`/`*-v2` schema literal. Read-only fixtures: p4 bumps two, p2 deletes one, and each
change is an explicit fixture edit in the same commit.

**(c) Sentinel-singleton test.** Import every dotted path under both spellings; assert
`sys.modules[a] is sys.modules[b]` and that every module-level sentinel found by an **AST
PREDICATE** — `Name = object()` at module scope, not a name pattern — plus the `weakref` registry in
`analysis/d1_execution_receipt.py`, is a single object under both spellings.
**RETRACTED 2026-09-18 (Q5 ruled (a)):** this paragraph previously pinned "15 `_TOKEN` modules / 27
files `[M, arbiter §0 re-count, superseding B's 18/25]`". That figure does not reproduce under any
principled method: a literal `_TOKEN = object()` grep finds 8; a column-0 `^_[A-Z_]*TOKEN[A-Z_]*=`
grep minus one regex-like false positive finds 15 by two cancelling errors, one of which silently
drops `_REQUEST_V2_TOKEN` (`[A-Z_]*` cannot match a digit); the AST predicate finds 18 module-level
sentinels in 18 modules, 16 `_TOKEN`-named plus `_MATRIX_ATTESTATION`
(`experiments/d1_confirmatory_matrix.py:26`) and `_DESIGN_ATTESTATION`
(`experiments/d1_registered_design.py:28`), which are the same double-load hazard class and escape
every name-pattern method. The AST-derived count at `144a9b5` is **`18 sentinels in 18 modules (AST predicate `Name = object()` at module level, measured at 144a9b5; includes `_MATRIX_ATTESTATION` and `_DESIGN_ATTESTATION` beyond the 16 TOKEN-named)`**
(conductor-substituted; p0's coverage rule is the predicate, not this number). **This is p3's exit
gate.** RETRACTED, same date, DN-P1 gen_shims v2: package `__init__.py` files are never shimmed
under the redesigned generator (leaf modules only; the 12 original package `__init__.py` files are
restored at their old locations after the `git mv` — see p3-quarantine below and
`.claude/workflows/briefs/p3-quarantine.md`), so the prior rationale here ("12 shims are package
`__init__.py` files whose `__path__` can double-load a child") no longer describes the mechanism.
Verify by test, not by reasoning.

**(d) CLI and packaging pins.** `--help` snapshots for `entropic_substrate.cli` and every
subcommand, `-m entropic_substrate.telemetry`, `-m entropic_substrate.sentence_prediction.worker_command`.
The frozen dotted-name literal set: `provider_launch.py:33-34`, `d1_runpod_adapter.py:345`,
`h_probe_launch.py:164`, `justfile:42`, `d1_external_analysis_adapter.py:224`, and
`assemble_review_packet.py:635-636`'s prefix selectors. Packaging: ASSERT, do not edit.
`where = ["src"]` already auto-discovers nested packages `[M, pyproject.toml:92]`, so DN-17
collapses to a pin.

**(e) Import-linter, and the machine input.** `import-linter` into the dev group, `lint-imports`
into `just check`. One `layers` contract and the `core.*` forbidden contract, declared over
the LANDED layers only, with `__legacy` deliberately outside the contract until p13, so the
contract is green from the first phase without a single exception entry.
`.pre-commit-config.yaml` stays untouched (operator ask-first).

Checklist:

- [ ] Graph re-derived at `bbe7976`, all four new modules mapped, diff vs `7ba2788` printed
- [ ] `tests/test_identity_epoch.py`, `tests/test_module_identity.py`, `tests/test_cli_surface.py`, `tests/test_packaging_identity.py` written
- [ ] `.importlinter` added over the landed layers, `__legacy` outside the contract
- [ ] `tools/mapping.json` committed, every node at this epoch covered
- [ ] Four mutation proofs recorded in the PR body
- [ ] Ambiguous contract surfaces opened as Linear issues via `linear-empirical-triage`

---

## Phase 1 — Strangler-fig decomposition (the 15 phases)

Each phase is one stacked branch `mythwave/<id>`, independently landable and green. Move counts
are `[M, pre-amendment phase decomposition]`: p3b's 5 modules are pre-claimed from p5 and p8, and
the exact re-count is owed to p0's mapping re-derivation.

| id | title | mode | moves | live gate | wave | parent |
|---|---|---|---|---|---|---|
| [p0-contract-pin](#p0) | epoch, identity fixtures, sentinel test, CLI/packaging pins, import-linter baselined | tracer | 0 | ship-it | serial | `refactor/hexagonal-redesign` |
| [p1-anchors](#p1) | layout-invariant `paths.py` resolver at five depth-derived sites, RED-first guard tests | tracer | 5 edits | ship-it | serial | p0 |
| [p2-d1-closure](#p2) | retire D1 reader source provenance end to end: closure, envelope arm, receipt readers block, filing artifact + binding (E-1 extended 2026-09-20) | tracer | 13 src edits, 1 deletion | ship-it | serial | p1 |
| [p3-quarantine](#p3) | `git mv` whole tree into `__legacy/`, generate the shim layer, own the landed-set fragment convention and `tests/test_legacy_coupling.py` | tracer | 332 | ship-it | serial | p2 |
| [**p3b-observability-rail**](#p3b) | loguru substrate → `kernel/runtime/logging.py`; `MetricSink` port → `ladder/capture/ports/`; three sinks → `ladder/capture/adapters/`; run-descriptor door; D1 observers re-based | tracer | 5 | **ship-it (live)** | serial | p3 |
| [p4-h-tracer](#p4) | H lane out first (55 H modules + `h_probe_command` + `h_probe_controller` + 4 reader-closure rail modules + the `d_ladder_worker_runtime` live half); both H identities bumped | tracer | 62 (55 + 2 + 4 + the live half; 63 was an arithmetic slip, docs sweep 2026-09-22) | ship-it (live) | serial | p3b |
| [p5-kernel](#p5) | `kernel/{identity,evidence_class,verdict,runtime}` (`kernel.identity`: 123 eager importers); splits `d1_training_step` and `d1_calibration` RED-first under `__legacy/` | ratchet | 14 + 2 splits | ship-it | serial | p4 |
| [p6-core](#p6) | `core/{geometry,substrate,observation,corpus,probe,latent}`; `kappa_norm` filing retraction | ratchet | 33 | deferred → zip | **w1** | p5 |
| [p7-register-gate](#p7) | `ladder/register` + `ladder/gate`; `rungs/`, `arms/` scaffolded; `d1_family+d1_runs` SCC as one unit | ratchet | 20 | deferred → zip | **w1** | p5 |
| [p8-capture](#p8) | `ladder/capture` five sub-packages beside p3b's ports/adapters; moves p5's two record halves | ratchet | 53 | deferred → zip | **w2** | p6 |
| [p9-run-lower](#p9) | `ladder/run/{values,transport,provider,resource,training,construction}`; moves p5's two torch halves | ratchet | 52 | deferred → zip | **w2** | p6 |
| [p11-analyze-review-file](#p11) | `ladder/{analyze,review,file}`; isolation ring in `review/` | ratchet | 49 | deferred → zip | **w2** | p6 |
| [p10-run-upper](#p10) | `ladder/run/{campaign,closure,ckptsync}` + D1 worker; consumes p9's values/transport | ratchet | 19 | ship-it | serial | p8 |
| [p12-historical-composition](#p12) | `historical/{e_series,d_ladder}`, `composition/`; `__legacy` empty; justfile/run_ladder.sh/Containerfile | ratchet | 57 | ship-it | serial | p10 |
| [p13-sweep](#p13) | rewrite 395 test files, delete shims + fragments + generator, add `__legacy` to the contract, re-pin the image | ratchet | 0 src | ship-it | serial (last, R-train) | p12 |

**Sequence.**

```
p0  p1  p2  p3  p3b  p4  p5  [p6 ∥ p7]  [p8 ∥ p9 ∥ p11]  p10  p12  p13
```

Wave-member parents follow the ratified `sprint-manifest.json`, **CORRECTED 2026-09-18** under this
plan's own R-train rule (ascending footprint, §307-311 below): `sequence` LISTS p6 before p7 (the
serial-reading order) and p8/p9/p11 in that order, but the ZIP order inside each wave is the
opposite of listing order — w1 zips p7 (20) then p6 (33), tip `mythwave/p6-core`; w2 zips p8 (52), p9 (42),
p11 (47) in dependency order, tip `mythwave/p11-analyze-review-file` (SUPERSEDED 2026-09-23 by
D-WAVES r3: w1 ran serial p6 then p7, so p8, p9 and p11 all parent on `mythwave/p7b-freeze-and-verify`, the FOR-1550/FOR-1548 hardening PR stacked on p7), and p10 parents on `mythwave/p11-analyze-review-file` (w2's zip
tip under D-WAVES r3), which contains p9's `ladder/run/{values,transport}` that p10 consumes — a parent at the
zip's bottom would provably exclude it (DEFECT 5). Inside a wave the FANOUT PLANNER's `zipOrder` is
authoritative and is validated as an exact permutation of the admitted lanes before any lane
launches; a non-permutation halts the wave with DECISION-NEEDED. Each zip's ledger entry stamps the
actual tip.

**Why the rails are forced by measurement, not taste.**

- **p1 before p3**: both pod source-tree guards break at the quarantine while `just check` stays
  green. `worker_execution.py:38` requires `Path(__file__).resolve().parent == repo_root/"src/entropic_substrate/sentence_prediction"`;
  `d_ladder_worker_runtime.py:120` requires `.../src/entropic_substrate` `[M, source read]`. A grep
  for either raise-message across `tests/` and `src/` returns **only the two raise sites, zero test
  assertions** `[M]`. CI stays green; the pods fail. The equality holds only at the original path,
  so quarantining the file breaks it wherever it lands.
- **p2 before p3**: `d1_external_analysis_control.py:100` calls the closure at MODULE IMPORT, and
  both its importers land in `ladder/review`, the ring H1 needs `[M, source read]`. Without p2 the
  quarantine makes the review ring unimportable.
- **p3b before p4**: p4 must consume LANDED observability addresses, not old-prefix shims. The
  per-phase no-old-prefix test would otherwise fail on `core/sentence` at p4.
- **p5 alone**: `kernel.identity` carries 123 eager importers, the largest fan-out in the repo
  `[M, blast_radius.py]`. p5 also carries the two symbol splits, because as a p8 ↔ p9 hub edge they
  were forbidden inside a `parallel_group` by R-parallel.
- **p10 out of w2**: it consumes `ladder/run/values` and `ladder/run/transport`, which p9 creates.
  A contract edge, resolved rails-before.
- **p13 last**: maximal conflict footprint (395 test files plus every remaining shim), so R-train
  lands it against a settled base.

Full per-phase `contracts_owned` / `contracts_consumed` / `touch_set` / `conflict_edges` / `scope`
/ `acceptance` live in the briefs and in `scope-strings.json`; every scope and acceptance string is
measured under 400 characters.

---

<a id="p1"></a>

## p1-anchors

Rewrite five depth-derived path sites onto one resolver anchored on the package `__init__`:
`config.py:382` and `sinks.py:141` (`parents[2]` with a silent CWD fallback),
`d1_reader_source_closure.py:263`, and the two fail-closed pod guards. Recommended form: compare
`Path(entropic_substrate.__file__).resolve().parent` against `repo_root/"src/entropic_substrate"`.
The package `__init__.py` never moves under any of the three candidate trees, so the guard becomes
layout-invariant **without** relaxing fail-closed semantics. This is a behaviour change to a guard
(ruling recommendation E-3), which is exactly why it is its own phase with RED-first tests.

<a id="p2"></a>

## p2-d1-closure

Retire the D1 reader source closure and the `reader_source_manifest_base64` envelope field
outright (ruling E-1). Measured support: no `d1-verdict` exists, no D1 preregistration exists, no
D1 confirmatory capture exists `[M, survey progression report §2, arbiter-reproduced; retained outside the repo]`, so **zero stored
envelopes carry the field**. The closure has exactly two importers, both `analysis/d1_external_*`,
and zero outbound edges `[M]`. The identity fixture drops the 73-entry manifest row (p0-measured) and records the
retraction in the same commit (house rule 4).

<a id="p3"></a>

## p3-quarantine

`git mv` the whole package into `__legacy/` with zero byte edits, then generate one alias shim per
old dotted LEAF path in the required static-import form (see *Generator contract* above); the 12
original package `__init__.py` files are restored, unmodified, at their old locations right after
the move, never shimmed. `verify_source_tree` sees the shims in the inventory, which is fine: they
are committed and the comparison is against the same commit. The exit gate is p0's sentinel test
over every dotted path under both spellings.

<a id="p3b"></a>

## p3b-observability-rail

The PRIORITY AMENDMENT's seam, landed before the H tracer. Contents, measured destinations and
teeth are in the amendment section above and in `.claude/workflows/briefs/p3b-observability-rail.md`.
`liveRequired:true`: the `ship-it` gate is the only evidence that moving the port did not break the
best-effort MLflow sink.
The 2026-09-21 rulings that shaped the landed phase (D1/D2/D3, R1, entry-point arm, gate-check,
promotion) are in the conductor-rulings table under Rulings; the brief carries them verbatim.

<a id="p4"></a>

## p4-h-tracer

The 55-module H lane leaves `__legacy`, plus `h_probe_command` and `h_probe_controller` (added
since the survey, destination `ladder/run/worker`) `[M, node-set diff]`, plus the 4 reader-closure
members it digests (`d1_artifact_snapshot`, `evidence.lifecycle`, `predictive_state.metric`,
`predictive_state.dynamics`) so both H identities fire exactly once.

Two specifics that are not optional:

- **`checkpoint.py:45-64` RAISES, it does not merely change value.** `_reader_identity()` globs
  `sentence_prediction/*.py`; once `package.parent` is `ladder/run` the glob has no valid root
  `[M, source read]`. Rewrite it to an explicit dotted-name tuple enumerating the same file set at
  the new names, the idiom `capture.py` already uses. That is `h-checkpoint-reader-v2`;
  `_READER_MODULES` becomes `h-source-reader-v3`; `docs/identity/read-old-table.md` carries the
  p0-epoch digests.
- **`d_ladder_worker_runtime` splits HERE, not at the end.** `worker_execution.py:12` imports
  `measure_hardware, verify_source_tree` from it `[M]`. Live half → `ladder/run/worker`;
  `verify_worker_source` → `historical/d_ladder` at p12.

`kernel.identity` is deliberately NOT in the H rail: the H lane does not consume it at all, and it
carries 123 eager importers `[M]`.

<a id="p5"></a>

## p5-kernel

The remaining kernel: identity (2), evidence_class (2), verdict (5), runtime (4 minus p3b's
`logging.py`), plus `git_environment` at `kernel/runtime` (ruling recommendation E-12, diverging
from the ruling's wording explicitly rather than silently: zero H importers, two D1 importers
`[M]`). `kernel/artifacts` and `kernel/lifecycle` already landed in p4. This phase also adds
`tests/test_no_old_prefix.py` and the per-phase assertion that `legacy_edge_budget` strictly
decreases.

**p5 also owns the two symbol splits (R-parallel fix).** `d1_training_step` and `d1_calibration`
split RED-first into record halves (`d1_training_step_records.py`, `d1_calibration_records.py`) and
torch halves, **both still under `__legacy/`**. The splits were originally scheduled inside wave 2,
with the record halves in p8 and the torch halves in p9. That is an intra-wave hub edge between two
members of the same `parallel_group`, which R-parallel forbids. Splitting serially at p5 leaves p8
moving only the record halves and p9 only the torch halves, **sharing no file**, each consuming p5's
split contract rails-before. Wave 2 stays `{p8, p9, p11}`. `tools/mapping.json` already carries
both new record modules (`d1_training_step_records`, `d1_calibration_records` →
`ladder.capture.records`), with the torch halves keeping their existing dests, so p5 reads the
mapping rather than extending it.

<a id="p6"></a>

## p6-core

The science layer: geometry 5, substrate 6, observation 5, corpus 3, probe 2, latent 11
(`metric` + `dynamics` landed in p4) `[M, mapping.json]`. **Zero algorithm or parameter change**;
`core/` is the operator-ruled zone. The `layers/kappa_norm.py` "dead end" filing is retracted in
the moving commit: `UnitNorm` is live in `build_norm` `[M, source read]`. House rule 4 requires an
explicit retraction, not a quiet overwrite.

<a id="p7"></a>

## p7-register-gate

PREREGISTER and SUCCESSOR-AUTHORIZED. Carries the `experiments.d1_family` + `d1_runs` SCC as one
commit. Scaffolds `rungs/` and `arms/` empty so H1's spec has a home before H1 is preregistered:
a new rung then has exactly seven possible touch points and the usual rung touches two.

<a id="p8"></a>

## p8-capture

Five measured sub-packages: custody 20, records 10, resource 9, reader 9, ledger 4, with external
fan-in 74 / 35 / 46 / 27 / 20 `[M, edges.json boundary count]`. `capture` decomposes cleanly (1-8
crossing edges per group) where `run` does not. Lands beside p3b's `ports/` and `adapters/`, and
moves p5's two record halves (`d1_training_step_records.py`, `d1_calibration_records.py`).
`telemetry.py` remains the only admissibility predicate for `results/*.jsonl` and keeps raising
rather than returning empty. The record halves are MOVED here, not split here: p5 performed both
splits, so p8 and p9 share no file.

<a id="p9"></a>

## p9-run-lower

Decomposed by measured DAG depth, not by name. Within the 101-module `run` subgraph, **44 of 312
intra-run edges (14%) violate the depth order** `[M, longest-path depth per sub-package]`, and they
concentrate on seven modules:

| module | violating edges | current group | move |
|---|---|---|---|
| `d1_operator_values` | 8 | campaign | → `run/values` (fan-in 41; imports only `resource_provenance` + 2 `experiments`, a pure value) |
| `d1_operator_evidence` | 4 | campaign | → `run/values` |
| `d1_operator_ports` | 4 | provider | → `run/values` |
| `sentence_prediction.journal` | 5 | worker | → `run/journal` |
| `d1_resource_activation` | 4 | provider | → `run/resource` |
| `d1_r2_runtime_composition` | 3 | transport | → `run/construction` |
| `d1_runpod_billing_parser` | 1 | closure | → `run/provider` |

All `[M, DAG-depth analysis over graph edges]`.

The two torch halves (`d1_training_step`, `d1_calibration`) are MOVED here, not split here: p5
performed both splits serially under `__legacy/`, so p8 and p9 share no file and wave 2 is legal.

**The honest residue.** Promotions do NOT reach zero cycles. Four successive promotion sets each
land on 16, and the DAG-derived 14-promotion set lands on 15 `[M, remap_check]`. What improves, and
is the real signal, is the minority leg of each cycle collapsing to one or two edges `[M]`. Three
classes result: clean promotions (the seven above), declared symbol splits (`d1_training_step`,
`d1_calibration`, `d1_filing_receipts`, `ladder_campaign`, `d_ladder_worker_runtime`), and a
**genuine class-level tangle that is OUT OF SCOPE for every structural phase**:
`d1_trainer_handoff ↔ d1_evaluation`, and `sentence_prediction.journal ↔ execution`, the latter
bottoming out on the single 370-line class in `sentence_prediction/execution.py` that proposal B
independently flagged `[M, B §10]`. **This plan does not promise zero intra-run cycles at p13.** It
promises the sub-packages exist, are measured, and the residue is named and scheduled after p13
with its own operator ruling and its own `remap_check` gate.

Membership is validated with `remap_check` BEFORE the first `git mv`. The oracle reads only
`edges.json` and a `mapping.json`, is pure and writes nothing `[M, 23 lines, print-only]`, so every
decomposition step is a paper exercise at zero execution risk.

<a id="p10"></a>

## p10-run-upper

Campaign 15, closure 6, ckptsync 5, minus the modules promoted down in p9. Owns the parameterized
entrypoint source (ruling E-4): the dotted entrypoint names live in ONE place consumed by
`scripts/run_ladder.sh`, the 13 justfile lines and the worker allowlists. The fail-closed allowlists
(`provider_launch.known_workers`, `d1_runpod_adapter.py:345`) are edited in the SAME COMMIT as the
move, because a stale allowlist fails closed on a pod and not in CI.

<a id="p11"></a>

## p11-analyze-review-file

ANALYZED, BLIND REVIEWED and FILED. The container / ACL / hermetic isolation ring lands in
`review/isolation`, not `analyze` (proposal C's named deviation). The two `d1_external_*` modules
arrive already free of the retired reader closure (p2).

<a id="p12"></a>

## p12-historical-composition

E-series to a governed read-only `historical/`; the 11 `d_ladder_*` modules plus
`analysis/d1_probe` and `analysis/ladder_probe` to `historical/d_ladder`, runnable, same governance
(ruling recommendation E-6). Their only importers are inside the lane `[M, arbiter-reproduced]`, and
CLAUDE.md's "retained as mechanistic groundwork" argues against deletion. One composition root:
`cli.py`, per-lane `wiring/`, `-m` `entrypoints/`. `d_ladder_artifacts` + `d_ladder_execution` move
as one unit (the second module-level SCC). **Postcondition: `__legacy` is empty**, asserted by test.
The six E/D CLI subcommands still dispatch into `historical/` so E0-E1.5c replay stays possible.

<a id="p13"></a>

## p13-sweep

One mechanical pass rewriting 395 test files onto the new dotted paths, then delete every shim and
the generator, and every landed-set fragment. Add `__legacy` to the import-linter contract and
enforce it with zero exceptions. That is the second and last edit of `.importlinter`. Record `remap_check` at the new HEAD in-repo as the next refactor's baseline. **This is
the only step that breaks an external caller, so it is announced separately.** The deployment image
is rebuilt and re-pinned ONCE here (ruling recommendation E-5, spend-adjacent).

**Docs ownership (Q6, emitted default accepted).** p13 owns `CLAUDE.md`, `docs/architecture.md` and
`docs/deploy-h100.md`, updating all three to the landed layout — the emitted default from the
sprint-preflight interview, ratified here rather than left placed-by-default. Sealed artifacts
(`docs/verdicts/*`, `docs/e1-*preregistration*.md`) are NEVER edited by this phase or any other and
stay stale by design even though p13's terminal state leaves their dotted paths invalid — **ruling
pending** on the operator's one-line disposition of that staleness (Q6's second half), not yet
ruled as of 2026-09-18.

---

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test
goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never
reached it". Every mutation below is performed and reverted in-branch, both arms in the PR body.

| phase | mutation | must go RED |
|---|---|---|
| p0 | `import boto3` in a `core.*` module | forbidden contract |
| p0 | flip one byte in a closure member | identity fixture |
| p0 | load one module under a second name | sentinel-singleton test |
| p0 | emit a probe shim in the string-only form on an illegal edge | layers contract shown GREEN-BLIND; the static-import form must catch it |
| p1 | point the resolver at a wrong root | BOTH relocated-guard tests. Today **nothing** goes red, measured |
| p2 | reintroduce a `parents[2]` anchor in the review ring's call path | relocated-import test |
| p3 | emit one shim as a star-import | sentinel-singleton test |
| p3b | `import mlflow` in `core/sentence` | `mlflow` forbidden contract |
| p3b | add a bare `print(` outside the two pinned sites | observability conformance test |
| p3b | drop `configure_logging` from `worker_command` | entry-point stderr-prefix assertion |
| p3b | construct a `MetricSink` outside the composition root | sink-construction-site assertion |
| p4 | revert one of the four rail modules to `__legacy` | checkpoint identity test, with a RESOLUTION error, not a digest mismatch |
| p5-p12 | reintroduce one old-prefix import in a landed package | no-old-prefix test |
| p5-p12 | raise this phase's `legacy_edge_budget` to the previous phase's value | the budget-monotonicity assertion; separately, exceed the merged budget and `tests/test_legacy_coupling.py` goes RED |
| p6 | add a live-service import to `core` | forbidden contract |
| p7, p12 | split an SCC pair across commits | collection-time `ImportError` |
| p8 | make `telemetry.py` return empty instead of raising | admissibility test |
| p9 | move `d1_operator_values` back to `campaign` | committed `remap_check` output regenerates strictly worse |
| p10 | leave one allowlist literal stale | p0 CLI / entrypoint pin |
| p12 | leave one file under `__legacy/` | `tests/test_legacy_empty.py` |
| p13 | any old dotted import anywhere | layers contract AND no-old-prefix test |

**The blind spot, named.** `grimp` is static. A string-only shim body is an opaque literal, so the
linter stops at the shim and the layers contract is vacuously green for every edge routed through
one. Two fixes, both required: the static-import shim form (p3), and a per-phase test asserting zero
old-prefix imports in `src/**` for every package named in the merged landed set, plus the strictly
decreasing `legacy_edge_budget` that `tests/test_legacy_coupling.py` enforces against the AST
graph (p3 owns the test, p5 onward carry the budget). Without the second, a missed import rewrite
is undetectable by the suite by construction.

---

## Execution overlay (summary)

Serial phases launch `td-phase-lifecycle.cascade.js` directly. **Waves w1 and w2 each launch as ONE
`mass-patch-fanout` invocation.** Doctrine check: cross-phase parallelism is only a
`parallel_group` of zero-conflict-edge RATCHET phases. w1 and w2 are exactly that.

**The wave overlay is `mass-patch-fanout`, not a hand-written script.** helmsman 1.3.0 ships it
(`.claude/workflows/mass-patch-fanout.js`, card
`.claude/rules-lib/pattern-cards/topology/mass-patch-fanout.md`, skill
`.claude/skills/mass-patch-fanout/SKILL.md`). It IS this plan's wave overlay: a batch planner
validates edge-disjointness across the wave's tickets, K nested cascade lifecycles run as nested
`workflow()` children (one per ticket), and a zip stage restacks least → most conflict risk with a
per-tip exit gate. There is no bespoke `refactor-wave-overlay.js`, and no "one session per lane"
ceremony: the overlay provisions and drives the lanes itself.

**Lane args per wave.** `tickets[]` are the wave's phases as `{id, title, scope, acceptance}`, with
`scope` pointing at the brief and the plan anchor and `acceptance` reading "Per the brief's
ACCEPTANCE section". `parent` is the wave parent branch. `checkout` is the conductor session's own
worktree. `cascadeScriptPath` is the REPO's ported cascade, `.claude/workflows/td-phase-lifecycle.cascade.js`.
`gates` are `{offline: "just check", exit: "just pc-ci"}` byte-for-byte from the ratified repo
profile. Every lane carries `mode: 'arch'` explicitly. **K WORDING, corrected 2026-09-18:** `K` is
the fan-out's MAXIMUM LANE WIDTH, not a per-wave cardinality — ruling O-3 sets the cap at `3` here;
the engine's hard cap is `K_MAX = 4` and an out-of-range `K` throws rather than silently truncating.
w1 launches 2 lanes and w2 launches 3, both admissible under the same cap; neither wave's lane count
needs to equal `K`. Measured band 4.75M-6.57M tokens for 3-4 lanes; halt at >1.5× band.

**Gate contention, recorded not budgeted (Q8 ruled (b)).** Lane offline gates (`just check`) run
CONCURRENTLY on the one conductor host — `just check` measures 696s (11:36) single-lane for 4565
unit tests, and the wave runs up to 3 lanes at once in separate worktrees. Nothing bounds host CPU
or wall-clock for that concurrency; house rule 2 ("Probe sequentially. Never draw a conclusion from
concurrent containers.") has no stated analogue for concurrent host gates, and the port notes only
clear the container hazard (`publishMode:'defer'` keeps every lane off `container-test`; MLflow
runs once, serially, at the zip). The first wave's measured wall-clock under concurrency is a
FINDING to record after it runs, not a budget to gate on before it runs.

**Lane worktrees NEST under the conductor's checkout.** Exactly
`<checkout>/.claude/worktrees/<lane>`, and lane agents `cd` there first in every stage. This is a
measured law from the 2026-08-27 three-lane pilot, not style: lane worktrees placed as SIBLINGS of
the conductor's worktree are unreachable to workflow workers (subagent `EnterWorktree` reports
success and Bash then refuses to execute there), while nested paths are reachable by plain `cd`.
The engine validates the provisioned list before any lane launches: each path must be exactly
`checkout/.claude/worktrees/<worktreeName>`, and a sibling, relative or reordered path halts the
wave with DECISION-NEEDED rather than baking a broken isolation invariant into a lane.

**The repo carries a PORTED copy of `mass-patch-fanout.js`.** Three deltas from the upstream file,
each forced by a ruling or a measurement:

1. **Policy passthrough to lanes**, so `policy.publishMode:'defer'` reaches every nested cascade.
2. **A `just ship-it` zip**, replacing the upstream zip's per-lane draft-PR publish.
3. **Provisioning that runs the ceremony itself** (mise, uv sync, `.env`, and the `justfile.local`
   copy), because `git worktree add` does NOT fire the `WorktreeCreate` hook, so
   `worktree-post-create.sh` never runs for an engine-provisioned lane.

**Lanes never publish (ruling O-1).** Upstream, lanes publish their own draft PRs; here they do not.
Cascade Publish is a bare `gt submit` (house rule 1), and K × `just ship-it` is K concurrent podman
MLflow stacks on a 4 vCPU / 4 GB VM (house rule 2). `policy.publishMode:'defer'` makes a lane end
**CERTIFIED** (tip SHA + exit-gate JSON), Validate skips the PR steps, and the zip runs ONE
`just ship-it` on the integrated tip. No force-push ever; one live gate per wave.

**A zip conflict falsifies the plan, it does not get resolved.** The overlay's fitness function is
that every lane branch rebases onto its zip parent with ZERO conflicts. A conflict is a falsified
edge-disjointness claim and fails the WAVE, not just the lane. Gate re-certification after restack
is mandatory: a restacked tree is a new tree, and the lane's old certificate speaks for a tree that
no longer exists.

**Rollback (Q7 ruled (a)).** Axiom 8 (Terminal) requires three honest exits per loop; a failed
INTEGRATED gate at the zip is not a fourth path around that requirement. When the zip's `just
ship-it` (or the re-run exit gate at a restacked tip) fails, the zip UNWINDS to tip N-1 — the last
gate-certified tip before the failing restack — and raises `DECISION-NEEDED`, naming the specific
lane whose restack broke the gate. This is cheap because per-tip re-certification (re-running the
exit gate at EVERY restacked tip, above) already localises the failure to one restack: the unwind
never has to bisect across lanes, it just returns to the last tip that was actually proven green.
Nothing is auto-ruled; the conductor decides whether to re-fix the lane, re-order the zip, or pull
the lane from the wave entirely.

**Why these engines are ported, not invoked (§8 item 3).** `td-phase-lifecycle.cascade.js` and
`mass-patch-fanout.js` are repo-local copies of the helmsman engine, not calls to the installed
plugin commands, because the installed plugin cannot express four things this sprint needs: the
bun-based exit-gate runner and `bun install` are hardcoded at seven call sites upstream, with no
seam for this repo's `just`-based gates; the `gt submit` lane policy is hardcoded upstream, with no
seam for `publishMode:'defer'` (ruling O-1); a teardown race in the upstream zip tears down a lane's
worktree before its exit-gate log is flushed; and the Workflow tool refuses a `scriptPath` inside a
plugin's own cache directory when invoked from a host repo, which is exactly how the installed
command would have to be reached. Each is a real, reproduced blocker, not a preference, and each is
tracked upstream as Linear **FOR-1501** so the local copies can retire once fixed there.

**Mode is always explicit.** NAME COLLISION: the sprint manifest's `phases[].mode` is
`tracer|ratchet` (parallel eligibility); the cascade's `args.phase.mode` is `arch|normal` (model
tier). The overlay MAPS, never forwards. Feeding `ratchet` into `deriveCascadeMode` yields a silent
`normal` and **no Opus design gate on a contract-surface refactor**. Every lane passes the literal
`'arch'`.

**Args are brief-as-file, and the 961-character truncation now applies to the WHOLE fanout
invocation, not per lane.** One `mass-patch-fanout` call carries K tickets plus `parent`,
`checkout`, `cascadeScriptPath` and `gates`, so three tickets with long titles push past 900 on
their own. Preflight check: each wave's fanout `args` measured under 950 characters total, with a
ticket `title` falling back to the bare phase id if the wave is over. No profile rides in the args
and no brief text is inlined. Each ticket's `scope` points at
`.claude/workflows/briefs/<id>.md` and the plan anchor; `acceptance` reads "Per the brief's
ACCEPTANCE section". Gates are explicit, never profile-resolved. `phases[].gates.liveDeferred`
names the zip as re-certification owner for every wave member. Serial phases launch the cascade
directly with `liveRequired:true` and publish through `just ship-it` inside the lifecycle;
`repoToplevel` is the MAIN checkout path, never a worktree, because `advisorLockPath` derives from
its basename and K validators would otherwise race the lock.

**Zip (fan-in), conductor = single writer, serial, operator-gated.** The overlay's zip stage orders
lanes least → most conflict risk (R-train, ascending footprint) and reparents each onto the
previous with `gt track` + `gt move --onto`, re-running the exit gate at EVERY restacked tip. The
conductor then: diffs predicted vs actual touch sets, where undeclared overlap is a HALT, not a
note → re-derives the import graph at the new tip → runs ONE `just ship-it` (Bash ≥300s; verify
`gh pr view --json headRefOid` equals `git rev-parse` per PR; on "updated remotely outside
Graphite" prove fast-forward before `gt submit --force`) → appends the ledger and stamps the new
`verifiedAt` → tears down succeeded lanes' worktrees (`lsof +D` clear, `git worktree list`
verified). Failed and exhausted-pending lanes KEEP their worktrees and go to the DECISION BOARD;
nothing is auto-ruled. A halted wave is re-entered with `args.resume = {plan, startTip, worktrees}`
from the halted run's own report, which skips Plan and Provision entirely and never recreates a
lane worktree.

**After every wave publish (ruling O-1):** run `/pr-stack-comment-triage` over the contiguous PR
stack the zip just published, pinned to immutable commit refs. Findings that survive its empirical
validation are filed into the Linear project **"entropic-substrate-lm — hexagonal refactor"**
(team Form_fxn, key FOR): refactor-caused defects into the owning phase's milestone, everything
else into **Bug Squashing**, each labelled `severity::major|minor|trivial` and `P0|P1|P2`. Triage
output is the DECISION BOARD input for the next wave's mass-patch-fanout Plan stage; nothing in
it is auto-applied.

**Linear structure.** One milestone per phase, p0 through p13 (15 milestones including p3b), plus
a Bug Squashing milestone. Issues inside a milestone are the commit branches of that phase's stack,
listed in each brief's *Planned commit branches* section. `phases[].issues` in the manifest carries
the ids, pattern `^[A-Z]+-[0-9]+$`.

**Ledger.** `.claude/workflows/launch-queue.jsonl`, append-only, conductor is the only writer,
committed between waves and never mid-wave. Rows: `wave` / `lane` / `zip`. Resume anchors: the
wave's own `{plan, startTip, worktrees}` report triple, replayed as `args.resume`;
`subagents/workflows/<runId>/journal.jsonl` for liveness, because a RUNNING workflow has no state
file and "no state file" must never be read as "never launched"; and the last `zip.newVerifiedAt`
per wave.

**Preconditions before p0 and before wave 1.** Current branch merged `--no-ff` into `dev`;
`refactor/hexagonal-redesign` carrying the plan commit; `feature/<slug>` cut; working tree clean;
prunable worktrees pruned; the Observability manifest rotated and archived (ruling O-2); the
registry re-authored; the two `codex/deferred-d-*-20260914` branches landed, re-snapshotted or
abandoned BEFORE p3 (ruling E-11); the graph re-derived at `verifiedAt` with the diff vs the survey
epoch printed and every new module mapped. The sprint starts only when H0's lineage is closed per
`docs/hierarchical-latent-predictor.plan.md`'s H0 exit criterion.

---

## Rulings

**Ruled 2026-09-17 (operator). Binding. Do not re-open.**

| id | ruling |
|---|---|
| E-1 | Retire the D1 reader source closure AND the `reader_source_manifest_base64` envelope field outright in p2. Zero stored envelopes carry it. A closure over the new tree, if H1 wants reader-source pinning, is a later separately ruled step. EXTENDED 2026-09-20 (operator, "widen p2", after the p2 design gate measured 9 src + 14 test consumers of the arm's product): the D1 analysis-receipt `readers` block and the filing graph's `analysis_reader` artifact with its `reader_sha256` binding are retired in p2 as well, each with an explicit house-rule-4 retraction; zero persisted receipts or filings carry them. |
| O-1 | `publishMode:'defer'` in the ported cascade; lanes end CERTIFIED, the zip publishes once via `just ship-it`; every wave publish is followed by `/pr-stack-comment-triage` filing into the Linear project. |
| O-2 | Archive the Observability manifest as `sprint-manifest.observability.json` with p1-loguru / p2-mlflow marked `retired` (note: absorbed into p3b + rewrite-on-arrival), then a follow-up commit REMOVES the archived file from disk. Named deviation from the schema's keep-renamed rotation; git history is the archive. |
| P-1 | **Priority: observability wired through DURING the refactor, on the E-series pattern.** Implemented as rail phase p3b plus the extended rewrite-on-arrival rule and its teeth. The one permitted behaviour-change class inside structural phases. |
| E-4 | Edit the scripts, no forwarders, and PARAMETERIZE: the dotted entrypoint names live in one source consumed by `run_ladder.sh`, the justfile and the worker allowlists. |
| DN-P1 | **No import-linter ignore list exists.** `.importlinter` is edited ONLY at p0 (create) and p13 (add `__legacy`, full enforcement). From p3b to p12 it declares the layers contract over the LANDED layers `kernel < core < ladder < historical < composition`, with `__legacy` deliberately OUTSIDE the contract: quarantine edges are not linted, and landed-layer inversions are always ZERO and strictly enforced. Plus the forbidden contracts (`core` ↛ mlflow/boto3/botocore/dvc/runpod/huggingface_hub/datasets; `mlflow` only from `ladder.capture.adapters.mlflow`, active from p3b). Each phase from p3 on commits ONE disjoint fragment `.claude/workflows/landed-set/<id>.json` carrying modules landed, targets and `legacy_edge_budget`; `tools/gen_shims.py` merges the fragments to emit and re-point shims (since 2026-09-20, FOR-1522, it refuses to emit while any row has no module source on disk: `retired: true` marks a module deleted before its move, `pending: "<phase-id>"` one a later phase creates, and `--preflight` reports the walk without writing; `tools/mapping.json` is never pruned before p13); `tests/test_legacy_coupling.py` (owned by p3) asserts from the AST graph that landed ↔ `__legacy` edges are ≤ the merged budget and that every shim carries the static-import form. Ruled by the conductor as an implementation seam. |

Also ruled in the same session and carried as structural givens: whole-src scope; H lane first
tracer; merged skeleton (C tree + B grafts); `__legacy` empties with E replay in a governed
`historical/`; survey intermediates retained outside the repo (operator ruling); the execution shape (merge `--no-ff` into
`dev` → cut `refactor/` trunk → commit plan → cut `feature/` trunk → sprint-preflight → fan-out
waves → zip at fan-in → serial).

**Conductor rulings 2026-09-19..21 (the seam below the operator's; each recorded in the brief it binds and in the run's args, never only here).** Added 2026-09-21 after p3b's review round found this document lagging the briefs.

| id | ruling |
|---|---|
| p3-shim-pair | Operator, 2026-09-20 (option (b) of the p3 design-gate DECISION): every leaf shim is a typed PAIR (3-line `.py` alias + `.pyi` stub of DERIVED explicit re-exports, never `import *`); `tools/gen_shims.py` is the only writer, `--preflight` before any move, `--verify` at every commit touching `src/`; `tools/mapping.json` read-only until p13; a `pending: "<phase>"` fragment state sanctions one migration to the creating phase. |
| p3b-D1/D2/D3 | Design gate, 2026-09-21: `logging` and `tracking_contracts` land; `sinks`, `d1_observability`, `d1_tracking` stay quarantined (a shim has one target; their old-prefix names have out-of-envelope importers); four zero-byte package markers; the mlflow forbidden contract carries exactly one `ignore_imports` line naming the single sanctioned edge (not DN-P1's ignore list). |
| p3b-R1 | 2026-09-21: the D1 observer re-basing is DEFERRED to p8 (`D1MlflowObserver` becomes a STRICT `MetricSink` adapter; `D1JournalObserver` observes journal EVENTS and sits outside the one-METRIC-port rule; `d1_tracking_descriptor` consumes the frozen door). p3b leaves both `__legacy` files untouched; FOR-1432 re-parented to p8. |
| p3b-entry-arm | 2026-09-21: the conformance test's entry-point arm is defined over the seven `entrypoint: true` rows of `tools/mapping.json`, partitioned by constants in `tests/observability_conformance_config.py` (outside the frozen test); measured conforming today: `cli`, `telemetry`. |
| p3b-gate-check | 2026-09-21: `tests/test_legacy_coupling.py` takes ONE ruled change (the synthetic-edge control no longer compares against the real budget); the no-old-prefix arm lives in the new `tests/test_no_old_prefix.py`, created by p3b and CONSUMED by p4..p12. |
| p3b-promotion | 2026-09-21: the conformance config DERIVES its sets from optional landed-set fragment keys (`conforming_entrypoints`, `print_allowlist`, `sink_construction_modules`); each phase promotes through its own disjoint fragment, so the file never enters a wave lane's touch set (w2's p8 and p11 would otherwise share it). p13 inlines the merged sets when it retires the fragments. AMENDED 2026-09-22 (p4 design gate, D-A): `print_allowlist` may also GAIN entries for machine-consumed stdout sites a phase lands (JSON result envelopes with a named parsing consumer); p4 adds the four H CLI modules, so the pinned raw-print set is six modules, not two. |
| ownership-by-SHA | Engine, 2026-09-20: conductor-owned files (briefs, plan docs, manifest, engine scripts, CI workflow; `tools/regime_rows.py` and its test LEFT the list under ruling Q1 on 2026-09-22 because the tool mirrors the reader identities and the identity owner must re-pin it) change only in conductor commits whose SHAs ride in `args.conductorCommits`; the engine checks them by SHA and a lane returns DECISION instead of editing them. |
| R-PIN | 2026-09-22: a phase that rewrites a `_READER_MODULES` member re-pins both checkpoint regime rows in the same commit (tool-measured; label and module set unchanged). MEASURED at 0ac49ed: 44 explicit `__legacy` edges inside h-checkpoint-reader-v2 across 20 files, targets moving in p5..p10, so every remaining serial phase re-pins. The wave consequence (w1/w2 lanes share those files and the fixtures) is DECISION D-WAVES-2026-09-22 on the ledger, not ruled here. |
| FOR-1539 | 2026-09-22: gen_shims' emitted-files manifest is untracked derived state; no phase commits it. |
| D-WAVES r2 | Operator, 2026-09-22: KEEP the fan-out waves. MEASURED at 1c10969 with targets attributed by mapping dest: zero landed files carry edges to two lanes of one wave; the only cross-wave entanglement was `evidence.ladder_campaign`'s unowned artifact half (14 reader-set importers), which lands as the serial rail **p5b-ladder-campaign** between p5 and w1. Slice rule: lane slice = targets ∪ landed importers ∪ shim pairs, pairwise disjoint within a wave, verified by `tools/lane_slices.py` at plan-wave and at the zip; `zip_owned_regenerable` (the four regime-row fixtures + `.secrets.baseline`) is excluded from the check and regenerated by the zip in one rung. Revision 1 of this decision (serialize) was an attribution error and is retracted. |
| BUDGET-MONOTONE retracted | 2026-09-23: p5 landed an arm requiring each fragment's `legacy_edge_budget` to be STRICTLY BELOW its predecessor's. MEASURED at 3e9ffa2: a phase whose moved modules import still-quarantined modules legitimately RAISES the landed→`__legacy` count on arrival (p7: 0 removed, ~12 added), so the premise was wrong, not the guard. The arm is removed; the budget stays the MEASURED count committed per fragment and asserted by equality (`tests/test_legacy_coupling.py`); zero old-prefix imports (`tests/test_no_old_prefix.py`) remains the no-regression guard; the count reaches zero at p12 when `__legacy` empties. |
| D-WAVES r3 ruled | Operator, 2026-09-23: w1 ran serial (p6 → p7: p7's modules import the `observation` facade p6 moved). w2 STAYS PARALLEL (p8 ∥ p9 ∥ p11) under option A: `tools/gen_shims.py` emits a FORWARDER at every landed module's old `__legacy` address (FOR-1547; exact shim form, typed pair, in the emitted manifest, `--verify`-covered), so a lane's rewrite of a sibling-target import to the explicit `__legacy` address stays valid after the zip restack. Zip order by dependency: p8, p9, p11. After the zip, ONE serial reconciliation rung (conductor) rewrites the sibling-target imports to their final addresses (worklist = `tools/lane_slices.py --allow-dependencies`, measured ~105 edges at 3e9ffa2), deletes those forwarders, regenerates shims, re-pins regime rows if a reader-set member changed, re-certifies and publishes. The mechanism rule is amended accordingly: a moved module's old `__legacy` address is a forwarder until the rung that rewrites its last importer; p12's `__legacy`-empty postcondition covers leftovers. |
| RUNG-OWNED | Conductor, 2026-09-23 (consequence of D-WAVES r3): a landed importer carrying explicit `__legacy` edges to TWO lanes' targets (measured: p7's `ladder/gate/ladder_predecessor.py`, p8 lines 14/20, p11 lines 8/12) is rewritten by NEITHER lane; each lane's `gen_shims` run emits a forwarder for it inside that lane's worktree (derived predicate) and the reconciliation rung rewrites it once. Declared with `tools/lane_slices.py --rung-owned <file>` (leaves every slice; exit 2 unless two or more slices carry it, so it cannot decay into an ignore list). |
| ZIP-REMEASURE | Conductor, 2026-09-23: `tests/test_legacy_coupling.py`'s equality arm binds to the HIGHEST-ORDERED fragment present (manifest sequence), which each parallel lane measured in its own worktree without its siblings' edits. At every restacked tip the zip's `zipRestackHook` regenerates shims and forwarders, re-runs the test, and on failure sets that fragment's `legacy_edge_budget` to the measured count with a dated `note` line (`re-measured at zip restack of <branch>: was N, integrated M`), commits through the hooks, then re-certifies. Never any other fragment field. |
| FRAGMENT-PER-COMMIT | Conductor, 2026-09-23 (measured at p11 review): `tests/test_module_identity.py` reverse-maps a moved module's new dotted name through the landed-set fragments' `target` field, so every landing commit carries the fragment rows for the modules it lands; the fragment grows per commit; the budget half of this row is amended by BUDGET-PER-COMMIT below (every landing commit writes its own measured value). |
| P8-FRAG | Conductor, 2026-09-23: the two record-half rows `d1_training_step_records` / `d1_calibration_records` leave `landed-set/p5-kernel.json` (`landed: false`) and enter `p8-capture.json` (`landed: true`) in the commit that moves them; `gen_shims`' "serial phases only" rider on this migration guards two lanes editing one fragment, and p8 is the only w2 lane editing `p5-kernel.json`, so file-disjointness holds. |
| P9-REMAP | Conductor, 2026-09-23, FOR-1549: `tools/remap_check.py` cannot run on the post-quarantine tree (MEASURED at 79d8228: 328 unmapped modules, exit 1 before any count). The p9 guard tooth "move `d1_operator_values` back to campaign → committed remap output strictly worse" and every pre-move refined-mapping acceptance are retracted WAVE-WIDE (p8's commit-7 before/after pair likewise); no remap artifact lands in w2; an edges normalizer is p13-adjacent work. |
| P11-FACADE | Conductor, 2026-09-23: `__legacy/analysis/` keeps `d1_probe.py` and `ladder_probe.py` (dest `historical.d_ladder`, p12), so p11 does not empty the package; the retraction of `__legacy/analysis/__init__.py` to the p3 one-liner is deferred to p12. |
| FACADE-EMPTY | Conductor, 2026-09-23 (wave-wide): a moved package facade `__legacy/<pkg>/__init__.py` is retracted to the p3 one-liner only in the commit after which that package directory holds no leaf module (measured by listing it); p11 moves ten `evidence/*` leaves and one (`evidence/ladder_campaign.py`, p5b's gate half) survives the wave, so neither p8 nor p11 retires `__legacy/evidence/__init__.py`; each brief states which phase does. |
| PKG-FORM | Conductor, 2026-09-23 (44c3dc5): the forwarder importer scan also records `<package>.<name>` for package-form `from` imports of a `__legacy` package directory (measured: `tests/test_d1_training_step_split.py:35`), so leaves imported that way are forwarder-served after their move. |
| W2-SCC | Conductor, 2026-09-23 (w2 design gates, p8 and p9): the per-sub-package landing-commit partition is infeasible because the sub-package import graphs are cyclic (p8: records↔custody, records↔reader, 14 forward-reference lines; p9: construction↔trainer↔transport (sub-package renamed by P9-TRAINER)). Landing commits partition by strongly-connected groups over the MEASURED intra-set import graph in topological order (a cyclic group lands as one commit), proven by a topological check recorded before the first commit. Transient `__legacy` addresses inside a phase are NOT permitted (a second rewrite breaks rewrite-on-arrival). Surplus Linear ids are annotated "absorbed into commit N". |
| BUDGET-PER-COMMIT | Conductor, 2026-09-23 (p9 gate D-2, wave-wide; amends FRAGMENT-PER-COMMIT): `load_committed_budget` does a bare `int(top["legacy_edge_budget"])` on the highest-ordered fragment, so the key must exist from the fragment's first commit. Every landing commit writes its own MEASURED budget (`tools/import_graph.py` + the test's classifier), re-measured as the tip moves; "written once at the last commit" is retracted. p7 is the precedent (35 at 2ee5e87, 45 at f94b808). |
| P9-TRAINER | Conductor, 2026-09-23 (p9 gate D-1), DECISION-NEEDED for operator ratification of the name: `src/entropic_substrate/ladder/run/training.py` is a p4-landed 166-line module and member of the frozen h-checkpoint-reader-v2 set; a package directory `ladder/run/training/` would silently shadow it (proven in 3.12.13). p9's training sub-package is renamed `ladder/run/trainer/` (verified free); the 17 addresses (15 training rows + the two failure modules per the gate's C-5) become `ladder.run.trainer.*`; no frozen member is touched. Lanes assert all six destination names are free before the first `git mv`. **RATIFIED by operator 2026-09-24**: the name `trainer` stands (candidates weighed: trainer, training_loop; `training` off the table as p4's landed reader member). |
| P11-WORKER-PINNED | Conductor, 2026-09-23 (p11 gate), DECISION-NEEDED for operator ratification as a third pinned site under the P-1 amendment: `analysis.d1_external_worker.main()` runs in the D1 analysis container (`--network none --read-only`), its stderr is content-addressed into evidence (`stderr_sha256`), and `configure_logging()` would make that digest wall-clock dependent and try to mkdir a JSON sink on a read-only rootfs. p11 does NOT promote the worker: `:363,366` stay raw `print(..., file=sys.stderr)`, listed in p11's fragment `print_allowlist` with this measurement; the entrypoint stays not-yet-conformed. **RATIFIED by operator 2026-09-24** as the third pinned print site under P-1 (with `telemetry final-eval` and `cli eval --json`). |
| W2-ADMISSIBLE-DELTAS | Conductor, 2026-09-23 (P9-SOLO / P9-CONTINUE; anchored in the plan 2026-09-24 because the ledger's `G6` label had no versioned home): at the w2 tips the p9→p8 DEPENDS edge is absent; p9's slice loses exactly the 12 importers p8 landed; p8's slice gains the 17 p9 files that import p8 targets by explicit `__legacy` address (p8 slice 163, p9→p8 38 edges, was 41). These deltas are ADMISSIBLE under rewrite-on-arrival and the r3 sibling-target ruling; a zip or rung reader must not halt on the 17-file intersection. |
| DATA-BUILD-RUNG | Conductor, 2026-09-23 (p9 gate, second w2 launch; option i): `__legacy/data/build.py` imported `data.batching` by its OLD name, so the first landed consumer that rewrites its own `data.batching` import onto `core.corpus.batching` while keeping `data.build` at the explicit `__legacy` address enters the old-path facade from inside the package and hits a partially-initialised-module ImportError (measured: 41 of 101 cold probes in a simulated p9 move, incl. two reader members and all three pod entrypoints). One conductor rung on the shared parent rewrote build.py's five first-party imports to explicit addresses, RED-first (`tests/test_legacy_data_build_cold_import.py`); p8 rewrites them once more on arrival. Amends rewrite-once for one quarantined file, the p3 "Retracted facade body" hazard class. |
| P8-PACKET-SCRIPT | Conductor, 2026-09-23 (p8 gate D-1): `scripts/assemble_review_packet.py`'s blinded-reader strip filter keys on the p0-frozen prefixes, so p8's rewrite of the moved `telemetry.py` imports leaks four import lines and `tests/test_review_packet.py` goes RED. p8 may edit the script ADDITIVELY in the SCC commit: keep both frozen literals (pinned by `tests/test_cli_surface.py:91-95`), append the `ladder.capture.reader.d1_` and `ladder.capture.records.d1_` prefixes and the `kernel.runtime.logging` spelling of the logging substitution; acceptance = the emitted reader is byte-identical to the parent tip's (measured 11069 bytes). Manifest touch set amended. |
| P8-SHIM-LAYER-RETIRE | Conductor, 2026-09-23 (p8 gate D-2): `tests/test_shim_layer.py::test_restoring_one_facade_original_bytes_breaks_cold_import_RED_case` (facade `data`) asserts a planted cycle reachable only while the `data/*` shims import quarantined leaves; after p8 moves them it measures rc=0 and the RED case is unreachable. Narrow Q1 carve-out: p8 retires that RED case in the SCC commit with an explicit house-rule-4 retraction citing the measurement; not re-homed onto `predictive_state` (it would re-break when p9 moves those leaves). |
| P8-OBSERVERS-RENAME-ONLY | Conductor, 2026-09-23 (p8 gate D-3), FOR-1552: the p3b-R1 re-basing of `D1MlflowObserver` on the `MetricSink` port contradicts Q1 (`tests/test_d1_observability.py` constructs it with a non-MetricSink fake at :202/:227/:290) and fail-loud (the only mlflow sink is best-effort). p8 moves both observers to `ladder/capture/adapters/` as renames plus import rewrites only; constructor, isinstance guard, log_metric projection and raise-through unchanged; `D1MlflowMetricClient` stays public. The unification, a strict raising sink and the test carve-out are FOR-1552, which gates the p13 "D1 observer protocols gone" postcondition. |
| P11-WORKER-TEST-CARVEOUT | Conductor, 2026-09-23 (p11 gate): `tests/test_d1_external_worker.py:98-108` hard-codes the `__legacy/analysis/` paths of two p11 targets and copies them by path, so the review+file commit ERRORs at setup (measured). Precedent: p3's manifest grant for the same file and p6's Q-1 ("the rule yields to a test whose own addresses move with the phase"). p11's Allowed files admit the file for ONE edit riding commit 2: re-point `_CONTROL_SOURCE`/`_WORKER_SOURCE` by resolving the imported modules' `__file__`, and re-prove the documented RED arm at the new depth (`parents[3]`, measured). |
| P8-D4-RUNG-TEST-FOLLOWS | Conductor, 2026-09-23 (p8 gate D-4, surfaced while amending): the DATA-BUILD-RUNG test pinned the literal `__legacy.data.build` address and `__name__`, which p8's SCC commit would break (no static importer, no forwarder; the old-name shim resolves to the landed address). `tests/test_legacy_data_build_cold_import.py` now derives the module's current address live through the landed-set fragments and asserts the entered-first cold import against it, so it follows the move unedited; no carve-out. |

**Recommendations carried into the sprint-preflight interview (not yet ruled at authoring; RATIFIED as recommended at the interview, 2026-09-18, per the manifest's ratification; the p4 brief's E-2 row says so, this heading lagged it until the 2026-09-22 docs sweep).**

| id | question | recommendation |
|---|---|---|
| E-2 | h-checkpoint-reader-v2 membership | faithful restatement: same file set at new dotted names. Narrowing to domain modules is a scientific change and is not offered. |
| E-3 (was DN-8) | anchor rewrite classification | behaviour change to a guard; its own phase p1 with RED-first relocated-guard tests |
| E-5 (was DN-7) | image re-pin | once, at p13; spend-adjacent |
| E-6 (was DN-10) | d_ladder + probes | `historical/d_ladder/`, runnable, same governance as `e_series` |
| E-8 (was DN-16) | import-linter into dev group and `just check` | yes; `.pre-commit-config.yaml` untouched unless the operator adds it |
| E-9 (was DN-23) | `run/construction` as a second composition site | accept the named deviation, record it in `docs/architecture.md` at p9 |
| E-10 (was DN-20) | one-impl port, `*Fake` doubles | defer past p13 |
| E-12 | `git_environment.py` home | `kernel/runtime` in p5 (zero H importers), diverging from the ruling's wording explicitly |
| O-3 | K for wave 1 | 3 |

**Absorbed, not dropped.** E-7 (`d_ladder_worker_runtime` splits no later than p4) is absorbed into
the p4 specifics above and p4's brief. E-11 (the two `codex/deferred-d-*-20260914` branches land,
re-snapshot or abandon before p3) is absorbed into the execution overlay's preconditions and p3's
brief. Arbiter items DN-1, DN-2, DN-3, DN-9, DN-11, DN-12, DN-17, DN-19, DN-24 and DN-A1 are closed
by the rulings above.

---

## DECISION-NEEDED register

Never auto-ruled. Each names its door or scope question, its measured support and a recommendation.

| id | question | measured state | recommendation |
|---|---|---|---|
| DN-P2 | **p3b's five modules are pre-claimed from p5 and p8.** Move counts in the phase table are pre-amendment. | `logging.py` → `kernel.runtime`; `sinks.py`, `tracking_contracts.py`, `d1_observability.py`, `d1_tracking.py` → `ladder.capture` `[M, mapping.json]` | Re-count at p0's mapping re-derivation and stamp the corrected counts into the ratified manifest. p5 and p8 carry explicit hub-owner conflict edges against p3b in their briefs. |
| DN-P3 | **`sentence_prediction/execution.py`'s 370-line class.** The `journal ↔ execution` tangle bottoms out here, and proposal B independently proposed decomposing it. | class-level bidirectional type dependency, not a package artifact `[M]` | OUT OF SCOPE for every structural phase. Schedule after p13 as a behaviour-adjacent decomposition with its own operator ruling and its own `remap_check` gate. |
| DN-P4 | **Wave-member branch parents are predictions.** R-train orders the zip by ascending footprint, so w2's parent depends on w1's actual landing order. | footprint ranks from the decomposition `[M]` | Treat the parents in the phase table as predicted; the zip ledger's `newVerifiedAt` confirms or corrects them before the next wave launches. |
| DN-P5 | **Linear project is outward-facing.** Creating the project, 15 phase milestones, Bug Squashing and one issue per planned commit branch is a write outside this repo. | project "entropic-substrate-lm — hexagonal refactor", team Form_fxn, key FOR | Confirm with the operator before the write. The per-phase issue lists are already drafted in each brief's *Planned commit branches* section. |

---

## Verification (end to end)

- **Plan authoring.** This document renders; `sprint-manifest.json` validates against the schema;
  R-rails / R-hub / R-parallel are checked by the overlay's Plan stage, whose `zipOrder` is
  validated as an exact permutation of the admitted lanes BEFORE any lane launches;
  every lane args string measured under 900 characters.
- **Brief conformance.** All 15 briefs carry exactly the 12 mandated `##` sections in order under a
  `# <id> — <title>` H1, verified by
  a generated-then-asserted section check, not by reading. Every scope and acceptance string
  measured under 400 characters. Each wave's fanout `args` measured under 950 characters.
- **Cascade port.** Launch the ported script on a throwaway docs-only phase in a scratch worktree
  with `mode:'arch'`; confirm the Opus design gate fires, `exitRunner` resolves, and CERTIFIED
  returns without `gt`.
- **Measurement oracle.** `tools/remap_check.py` on the identity mapping reproduces 142 / 11 at the
  pinned epoch (the control), and on the refined mapping reports the agreed per-phase targets before
  any `git mv`.
- **Phase gates.** `just check` + `lint-imports` green per branch; mutation proofs recorded in the
  PR body; `legacy_edge_budget` strictly decreasing with `tests/test_legacy_coupling.py` green;
  serial phases `just ship-it` green; each wave zip
  runs one `ship-it` on the integrated tip with `headRefOid` verified per PR.
- **Postconditions.** p12 asserts zero files under `__legacy/`; p13 asserts `__legacy` inside the
  layers contract with zero exceptions,
  zero old-prefix imports, and `remap_check` at the new HEAD recorded in-repo as the next baseline;
  the six E/D CLI subcommands still dispatch into `historical/`.
- **Topology end-state (Q9), owned not left implicit.** The p13 terminal state is **4 inversions /
  5 package cycles**, ACCEPTED as ruled debt — not zero. This refactor is a topological cut of the
  measured 142/11 starting state, not a promise to drive topology to zero; §Blast Radius Summary and
  §p9's "honest residue" already carry the measurement, this line makes it a stated verification
  target instead of something a reader has to infer. Verified by `remap_check` run at the p13 tip
  and recorded in-repo as the next refactor's baseline, the same artifact p13's own postcondition
  above names.
- **Observability.** p3b and p4 `just ship-it` live gates green, with the H lane's first MLflow run
  id recorded in the PR body; `tests/test_observability_conformance.py` green with its arms shown
  RED under mutation; at p13 a single `mlflow` importer, every entry point configured, zero prints
  outside the allowlist, D1 observer protocols gone.

**Green is claimable only from machine evidence at a pinned SHA.** Every green claim in this sprint
names its SHA. A moved HEAD voids the claim and it is restated, never reinterpreted.

---

## Milestones

Deliberately UNDATED. False-precision dates are not honest; the team reviews the blast radius and
the issue list first.

- **M1** — Rails landed: p0, p1, p2, p3. Contracts pinned, anchors layout-invariant, closure
  retired, tree quarantined behind a proven shim layer.
- **M2** — Observability rail and tracer: p3b, p4. The MetricSink port is the only metric port, and
  the H lane has its first live MLflow run.
- **M3** — Kernel and wave 1: p5, then [p6 ∥ p7] zipped. `kernel.identity`'s 123 importers land on a
  stable prefix; the science layer and the register/gate stages follow.
- **M4** — Wave 2 and the upper run stack: [p8 ∥ p9 ∥ p11] zipped, then p10. The two large buckets
  are decomposed and measured.
- **M5** — Closure: p12, p13. `__legacy` empty, tests on new paths, shims and fragments gone,
  `__legacy` inside the contract,
  the new baseline recorded in-repo.
- **M6+** — Post-refactor, separately ruled: the class-level residue (`d1_trainer_handoff ↔
  d1_evaluation`, `sentence_prediction/execution.py`), `layers` tightened toward `independence`
  where measurement supports it, and the deferred port questions (E-10).

---

## Detail-reference key

The ratified scope strings quote the phase decomposition's own section labels. They resolve here:

| quoted label | this document |
|---|---|
| `plan §C` | [p0-contract-pin](#p0) |
| `plan §0.5` | the p2-before-p3 bullet under *Why the rails are forced by measurement* |
| `plan §0.6` | the p1-before-p3 bullet under *Why the rails are forced by measurement* |
| `plan §mechanism` | *The mechanism rule that makes phases disjoint* |
| `plan §P-1` | *PRIORITY AMENDMENT* |
| `plan §A-p4` … `plan §A-p13` | the matching `#pN` anchor |
| `plan §D` | [p8-capture](#p8) and [p9-run-lower](#p9) |
| `plan §E-1`, `§E-6` | *Rulings* |

Anchors are dual: `#p6` and `#p6-core` both resolve, so lane args may cite either.
