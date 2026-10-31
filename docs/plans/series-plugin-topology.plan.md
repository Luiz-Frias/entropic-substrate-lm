# Series Plugin Topology Plan: platform engines, series adapters, digest-bound specs

Written 2026-10-30 on the read-only survey worktree `entropic-substrate-arch-survey-842143` at
`b9c570e0535c228c6bd4ea975ad87749bd6d7030` (main). It records the operator rulings of
2026-10-30 and the target end state for a sprint that starts **only after H1 completes entirely**
(ruling T1). Nothing in this plan authorizes a lane, a pod, or a push. Before the first lane
launches, the repo `CLAUDE.md` gains a north star and horizon section for this initiative
(house rule: no development without both) and a sprint manifest is ratified.

**Evidence labelling.** Every number carries `[M, <source>]` measured or `[E]` estimated. Measured
and estimated values never share a row without a per-row label. The survey this plan reads from
is published as three pages (system, module index, data flows) and its inputs sit under the
project-artifact data directory on the operator's machine; the per-agent journal is
`wf_5c35190f-817`.

---

## Context

The 2026-10-30 architecture survey (12 Sonnet scouts, 12 Opus citation verifiers, 3 Opus flow
tracers, 3 Opus refuters; 30 agents, 702 tool calls `[M, workflow usage]`) measured the package at
`b9c570e0`:

| quantity | value | evidence |
| --- | --- | --- |
| modules under `src/entropic_substrate` | 345 | `[M, ast-grep import extraction, build_graph.py]` |
| module-level internal import edges | 1981 | `[M, same]` |
| cross-layer edges, all downward | 897 | `[M, same; layer_edges]` |
| subpackage two-cycles inside `ladder` | 4 (gate/run, analyze/gate, file/review, file/gate) | `[M, same; subpackage_edges]` |
| debt items with a file citation surviving verification | 129 | `[M, survey.json]` |
| file citations re-checked / bad | 683 / 21 | `[M, survey.json]` |

The five-layer contract in `.importlinter` (`kernel < core < ladder < historical|__legacy <
composition`) holds at the layer level. Below it the tree mixes two axes that the contract does
not see:

- **Series axis.** `ladder/run` top level is the H run engine; its subpackages are the D1
  execution lane; `ladder/analyze` is one flat directory of D, H and E analysis grouped by
  filename prefix `[M, module index, lanes ladder-run-core, ladder-run-training,
  ladder-analyze]`. The H worker imports D1-named modules for operator values, the command
  runner, mailbox contracts, object transport, the S3 adapter and the DVC remote reader, so part
  of the D1-named surface is a shared run kernel `[M, lane ladder-run-campaign]`.
- **Status axis.** `historical/` names lifecycle status in a directory. Its two `__init__`
  docstrings say "governed read-only", yet `composition` imports both lanes and
  `ladder/run/provider/provider_launch.py` allowlists the D-ladder worker module as a pod entry
  point `[M, lane historical-configs-scripts]`. Moving code between `historical/` and the active
  tree on each status change is the cost this plan removes.

A third fact drives the engine ruling below: the package carries **three training pipelines and
two pod machineries** that do the same job with different custody. Trainers:
`historical/e_series/train.py` (plain `torch.save` with `format_version`),
`ladder/run/trainer/d1_segment_engine.py` plus `d1_fresh_trainer.py` (a 15-key exact-resume
checkpoint mapping, journal-first), `ladder/run/runner.py` plus `execution.py` (a sealed
`h-training-checkpoint-v2` chain) `[M, data flows page, all three flows]`. Pod machineries:
D1 `ladder/run/campaign` plus `transport` (mailbox over an object store, sequential operator
controller) and H `ladder/run/h_probe_*` plus `worker_*` (hash-bound request, watchdog process,
once-only allocation) `[M, lanes ladder-run-campaign, ladder-run-core]`.

Six systemic seams explain most of the 129 debt items `[M, system page, Tech debt tab]`:
per-module strict-codec helper copies, nine hand-written SHA-256 validators, eight
twice-declared names, stale measured-claim comments from the hexagonal re-layout, seven
production paths only tests construct, and the four cycles above. The consolidation this plan
asks for is the structural fix for the first three; the debt plan's phases 0 to 2 (correctness,
digest seam, codec seam) are prerequisites and run before this sprint, on the current trunk.

### Where pins and hashes live today

| kind | location | count | evidence |
| --- | --- | --- | --- |
| tool version pins in code | `ladder/run/h_probe_launch.py:40` runpodctl 3.0.1; `ladder/run/provider/d1_runpod_adapter.py:51` runpodctl 2.11.0; `core/sentence/segmentation.py:21` pysbd 0.3.4 | 3, two disagreeing | `[M, grep]` |
| 64-hex digest literals in `src` | `ladder/run/h_probe_launch.py`, `ladder/capture/custody/enwik8.py` | 2 | `[M, grep]` |
| image digest | `docs/deploy-h100.md:12` only | 1 | `[M, grep]` |
| wire schema version strings (`<name>-vN`) | scattered literals; `ladder/run/trace_capture.py` alone holds 24 | 343 | `[M, grep]` |
| reader-identity member lists | 11 files; `ladder/run/checkpoint.py` holds the 52-string tuple | 11 | `[M, grep]` |
| manifest loaders | six YAML loaders across three manifest families (experiment config, D1 launch profile, D-ladder profile); no TOML | 6 | `[M, grep]` |
| operator entry points | `scripts/entrypoints.env`, six names, one pointing into `historical` | 6 | `[M, file]` |
| environment-bound settings | five names | 5 | `[M, grep]` |

The manifest habit exists. What is missing is one seam stating which values are environment
selection and which are code identity.

---

## Rulings (2026-10-30)

| # | ruling | consequence |
| --- | --- | --- |
| R1 | The ladder protocol (register, run, capture, analyze, review, file, gate) is **platform**. The D1 four-arm family, the D2 to D5 rung specs, the factorial analysis and the D1 filing roles are a **series**. | `platform` is series-neutral; `series/d` holds the D-specific implementations of the platform ports. |
| T1 | The sprint runs as **its own sprint after H1 completes entirely**. | No topology move lands on the H1 trunk. The debt plan's phases 0 to 2 may land before, as rails on the current trunk. |
| R2 | Series plug in through an **explicit registry in composition**, shaped by the spec files below; no entry-point discovery. | The registry loads `series/*/series.toml`, binds each plugin by an allowlisted module path, and type-checks it against the ports. |
| R3 | Retirement is a **registry flag plus an admission row**. | `series/e` stays where it is when retired; its spec says `status = "retained"` and the admission table refuses new confirmatory runs. Nothing moves between directories on a status change. |
| C1 | Infrastructure values such as hashes and pins are set from **spec files**, not code literals. | `platform.toml` and `series/<x>/series.toml`, strict-loaded, frozen, digest-bound (below). |
| R4 | **Engines are platform; series are adapters.** The training pipeline, the pod machinery and capture are owned once, by the platform, and call out through ports. A series supplies the scientific adapters (model construction, objective, telemetry schema, analysis, filing roles, admission rows) and nothing else: no step loop, no checkpoint I/O, no pod code. | The three trainers and two pod machineries above collapse to one each (phases P3b and P3c). The series packages shrink to what differs between E, D and H. |
| R5 | **Extraction-ready modularity.** The target is a uv workspace whose members mirror a Rust multi-crate repository: each member has its own `pyproject.toml`, declared dependencies, tests and gate, so a member can be copied into another repository with a light refactor. The platform generalizes the infrastructure; a series extends it by swapping adapters. | The directory tree below is the member list. The acceptance test for every member is its own gate run from a bare directory (the validate-from-the-target's-context rule). |
| R6 | **The platform is a library for training machinery, not for this repo's series.** Its motive is to be lifted into scratch repositories to try other schedules, curricula, backbones and skeletons, mixture-of-experts machinery, RL environments, fine-tuning and LoRA adapters. It owns what sits around the optimizer step (admission, launch, custody, telemetry, capture, evidence), not what PyTorch does (autograd, modules) or what Ray does (distributed scheduling). | The scientific ports are named by what a training harness needs (`ModelFactory`, `Objective`, `Schedule`, `Curriculum`, `Environment`, `TelemetrySchema`, `AnalysisStrategy`, `FilingRoles`, `Admission`), never by E, D or H. The member gets its own north star and horizon before P1 (house rule), because this ruling widens its bearing beyond the repo's. |
| R7 | **Each member carries its own `CLAUDE.md`.** Three parts: bearing (the platform member states its own north star and horizon under D7; series members inherit the repo's and name the horizon stage they serve), boundaries (the ports it exposes or implements, the members it may import, the spec files it reads, its gate command), and local rulings (numerics islands, custody rules, series-specific doors). Precedence is the existing scope-first rule: a member file is the most local scope for files under it and wins over the root on conflict, with the conflict named aloud. The root `CLAUDE.md` shrinks to the repo bearing, the member map, and the cross-member rules no single member can own (publish sequence, live gate, secrets). | A lifted member keeps its rules without the old root (R5). Nested loading is documented, not measured in this repo; P1 repeats the 2026-08-14 sentinel measurement before any member relies on it, with the documented `paths:` frontmatter rule under `.claude/rules/` as the fallback (path-scoped, but it does not travel with a copied member). |
| C2 | **Observability is a platform port family, and the evidence plane stays authoritative.** JSONL plus DVC capture is the only verdict input; every tracker (MLflow today, Trackio and a W&B-compatible adapter next) is a best-effort mirror behind a `Tracker` port, with the fail-loud strict adapter as the one D1 exception. | `MetricSink`, `Tracker`, `ArtifactStore` and `RunRegistry` are ports in `platform/observability`; adapters are `jsonl` (authoritative), `mlflow`, `trackio`, `wandb-compatible`. Line of sight into the platform is a port, not a dependency on one vendor. |

### Recommended picks awaiting the operator (DECISION-NEEDED register, end of file)

| # | question | recommendation | why |
| --- | --- | --- | --- |
| D1 | Spec format | TOML for `platform.toml` and `series.toml`; experiment hyperparameters stay YAML under each series | stdlib `tomllib`, no implicit typing, pins are typed scalars; YAML stays where the three existing manifest families already are |
| D2 | Reader-identity member lists | move into `series.toml`; the identity test pins the spec digest; the composed identity is the concatenation of member digests in registry order | the list is selection, the digest is the promise; a spec edit is a deliberate re-pin commit exactly as a tuple edit is today, and no single member has to own a hash that spans members |
| D3 | Wire schema version strings | stay in code beside their codecs; one enumeration module per member gives the 343 literals a single home | a schema version names a record shape the reader promises to accept; putting it in a spec lets a config edit change what evidence is admissible |
| D5 | Unified checkpoint custody | one platform checkpoint format with a sealed chain (the H shape) and a per-series payload slot; E and D payloads become slot contents with their own schema version | the H chain is the strictest of the three and already carries source and reader identity; the other two have no production confirmatory run open under T1 |
| D6 | Second tracker adapter | Trackio, behind the same `Tracker` port MLflow sits behind; its `wandb.init/log/finish` surface also covers a W&B-compatible adapter for free | local-first, SQLite-backed, no server to stand up in a scratch repo, optional sync to a Hugging Face Space; adopted by alphaXiv OpenResearch on 2026-10-23 `[M, @askalphaxiv post 2026-10-23; alphaXiv/OpenResearch issue 571; huggingface.co/blog/trackio]` |

---

## The seam: selection in specs, promises in code

A spec carries what the operator **selects**: tool and image pins, registry and bucket topology,
which series are active, which run classes they admit on which surface and stage, which module
implements each port, which modules form a reader identity. Code keeps what the package
**promises**: wire-record shapes and their version strings, numerical invariants, the lifecycle
state machine, strict-codec refusals.

Every spec is strict-loaded (duplicate keys and non-finite values refused, unknown keys refused,
exact key sets per table), frozen into a slotted value, and its canonical digest enters run
identity. Editing a pin therefore restates the epoch; it never drifts silently. This is the
Epoch axiom applied to configuration, and it is what makes "configurable" and "sealed"
compatible.

```toml
# platform.toml (one per deployment regime; digest enters every run identity)
[image]
reference = "ghcr.io/<org>/entropic-substrate-lm"
digest = "sha256:<64 hex>"          # today docs/deploy-h100.md:12
[tools]
runpodctl = "3.0.1-551d6a4"          # today two disagreeing code literals
pysbd = "0.3.4"
[topology]
dvc_remote = "r2-entropic-substrate-lm"   # today .dvc/config (stays there; this names it)
object_store_prefix = "s3://entropic-substrate-lm"
[numerics]
execution_precision = "fp32"         # today ladder/run/execution.py require_execution_precision
```

```toml
# packages/entropic-series-h/series.toml
id = "h"
status = "active"                    # active | retained | retired  (ruling R3)
[entrypoints]
worker = "entropic_series_h.worker"                        # today scripts/entrypoints.env
[plugin]
module = "entropic_series_h.plugin"                        # allowlisted by the registry
[telemetry]
lane_tag_prefix = "h-"               # today the MLflow schema tag partition in capture
[[admission]]                        # today ladder/run/admission.py (H only)
run_class = "probe"
surface = "pod"
stages = ["warmup", "prediction"]
[reader_identity]                    # D2: today the 52-string tuple in ladder/run/checkpoint.py
checkpoint = ["entropic_series_h.carrier", "entropic_series_h.law", "..."]
source = ["..."]
```

---

## Target topology (end state): a uv workspace

```text
pyproject.toml                 workspace root: [tool.uv.workspace] members, one lockfile
platform.toml                  pins and topology (ruling C1)
packages/
  entropic-kernel/             values, strict codecs, lifecycle FSM, spec loader and digest
                               depends on: stdlib, pyyaml (torch-light; checkpoint format guard only)
  entropic-core/               byte_substrate (today core/substrate + geometry), corpus, latent,
                               observation, sentence, probe; depends on: kernel, torch
  entropic-platform/           the reusable infrastructure; depends on: kernel, core
    engine/                    ONE training pipeline: phase schedule, checked optimizer step,
                               atomic checkpoint + seal chain + journal, evaluation admission,
                               resource meter (today three trainers)
    pod/                       ONE pod machinery: launch spec, provider port, watchdog, worker
                               command, mailbox and object transport (today two machineries)
    observability/             MetricSink, Tracker, ArtifactStore, RunRegistry ports (ruling C2);
                               adapters jsonl (authoritative), mlflow, trackio, wandb-compatible
                               (today ladder/capture/adapters + reader/sinks.py)
    capture/ analyze/ review/ file/ gate/   series-neutral evidence protocol (ruling R1)
    ports.py                   the Protocols below
    providers/                 infra adapters: runpod, r2/s3, docker, local (one each)
  entropic-series-e/           scientific adapters for E + configs + tests; depends on: platform
  entropic-series-d/           D1 family, D2-D5 rung specs, factorial, filing roles; depends on: platform
  entropic-series-h/           sentence carrier, law, probes, K-checks, h_preservation; depends on: platform
  entropic-cli/                composition: registry, entry points, wiring; depends on: everything
tests are per member (packages/<m>/tests), not one tree
every member carries CLAUDE.md (bearing, boundaries, local rulings; ruling R7)
```

Dependencies bind twice: each member's `pyproject.toml` declares what it may import, and the
import-linter contracts restate it at module level (`kernel < core < platform < series < cli`,
`series.*` mutually independent, `platform` forbidden from importing any series), because Python
does not refuse an undeclared import at runtime the way Cargo does. Per-member `pyright` and
per-member tests give each crate its own gate. `historical/` and `scripts/entrypoints.env`
retire in the last phase.

### Placement under R4: what is an engine, a port, an adapter

| owner | what it is | what collapses into it at `b9c570e0` `[M, module index]` |
| --- | --- | --- |
| platform engines (one each) | training pipeline; pod machinery; capture and readers | E `train.py`, D1 `d1_segment_engine.py` + `d1_fresh_trainer.py`, H `runner.py` + `execution.py`; D1 `campaign/` + `transport/`, H `h_probe_*` + `worker_*` |
| platform scientific ports | `ModelFactory` (spec to module plus parameter attestation); `Objective` (forward, loss, step metrics, evaluation); `Schedule` (learning-rate and phase schedule); `Curriculum` (data ordering and sampling, with its order digest); `Environment` (rollout source for RL-shaped objectives; identity for supervised ones); `TelemetrySchema` (run_start and run_end shapes, lane tag, admissibility); `AnalysisStrategy` (admit, aggregate, report, check suite); `FilingRoles`; `Admission` (run classes by surface by stage) | `ladder/run/construction`, `core/substrate.build_model`; the three trainers' step and eval bodies; the warmup/cosine schedule in `d1_training_step` and the H `RunSchedule`; `SequenceSampler` and the H sentence batch sampler; no RL source today; the capture lane partition; `d1_factorial`, `h_aggregate`, `series.py`; the D1 13-role bundle and `ladder_filing`; `ladder/run/admission.py` |
| platform observability ports (ruling C2) | `MetricSink` (the step-row door), `Tracker` (mirror of params, tags, metrics, artifacts to a dashboard), `ArtifactStore` (checkpoints and bundles by content hash), `RunRegistry` (run identity, status, lineage) | `ladder/capture/adapters/{jsonl,mlflow,mlflow_strict,composite}.py`, `reader/sinks.py` descriptor seam, the DVC custody in `capture/custody`, the MLflow run search in `h_wave_monitor` |
| platform infra ports | `Provider` (create, observe, delete), `ObjectStore`, `Isolation` (docker runner) | `d1_runpod_adapter`, `h_runpod_transport`, the S3 adapter, `DockerD1ExternalAnalysisAdapterV2` |
| series adapters | implementations of the scientific ports only | E byte substrate and nats-per-byte eval; D1 four-arm family and factorial; H sentence carrier, law, K1 to K7 |
| provider adapters | implementations of the infra ports | runpod (one, not two runpodctl pins), r2/s3, docker, local |
| tracker adapters | implementations of the observability ports | jsonl (authoritative, always on), mlflow (today), trackio (D6), wandb-compatible |

### Observability: line of sight into the platform

The evidence doctrine in `docs/architecture.md` "Evidence plane" does not change: JSONL plus DVC
capture is authoritative, the admissible readers are the only read side, and a tracker is a
best-effort operational mirror that never feeds a verdict. What changes is that the mirror is a
port. Today the mirror is one vendor wired in three places (the MLflow adapter, the strict D1
chain that only tests construct, and the REST wave monitor) `[M, lane ladder-capture; lane
composition]`. In the end state:

- `MetricSink` stays the step-row door every engine writes through; `CompositeSink` fans out to
  the authoritative JSONL adapter plus zero or more tracker adapters.
- `Tracker` carries params, tags, scalar metrics and artifact references to a dashboard. The
  MLflow adapter implements it today; Trackio is the second adapter (D6) because its
  `wandb`-shaped API is also the W&B surface, it needs no server in a scratch repository, and it
  can publish a dashboard to a Hugging Face Space when a run should be shared.
- `ArtifactStore` and `RunRegistry` name what the DVC custody and the run descriptor seam
  already do, so a lifted platform can bind them to a local directory, R2, or a tracker's own
  artifact store without touching the engine.
- The lane partition on the MLflow `schema` tag becomes a `TelemetrySchema` concern, so a
  tracker that has no tags (file-only TensorBoard, for instance) still partitions correctly.

Comparison for orientation, not a dependency: Thinking Machines' Tinker exposes a managed
fine-tuning service as a few primitives (forward-backward, optimizer step, sample, save state)
with LoRA adapters on top. The platform engine's port set is the open, self-hosted analogue of
that primitive set plus the evidence and custody machinery this repo already has; it is not a
training framework and does not compete with PyTorch or Ray.

Typestate survives unchanged: the lifecycle FSM stays in kernel, factory attestation stays the
only way to mint filing-eligible values, and adapters are passed into platform factories rather
than discovered. The registry's own typestate is `SeriesSpec` (parsed, digest known) to
`AdmittedSeries` (status and admission rows checked against `platform.toml`) to `BoundSeries`
(plugin module imported from the allowlist and type-checked against `ports.py`).

### What stands in the way of the copy test today

- The 52-member checkpoint reader identity spans kernel, core and ladder `[M, lane
  core-hseries-engine: 22 members in core/sentence alone]`, so no single member owns the hash.
  D2 resolves this: each member's spec lists its own reader members and the composed identity
  is the concatenation in registry order.
- `kernel/runtime/config.py` value types (`ModelConfig`, `DataConfig`) are consumed by
  `core/latent/state.py`, `ladder/run/construction` and `ladder/capture/custody/build.py`
  `[M, lane kernel, corrected claim]`, so the E-era config is a shared contract, not E-owned.
  The value types stay in kernel; the `EXPERIMENTS` registry and `E15_PREREGISTRATIONS` move
  to `entropic-series-e`.
- Two cross-lane private imports under pyright suppressions (`ladder/run/campaign/
  d1_operator_evidence.py` into a capture private; `ladder/file/ladder_filing.py` into
  `ladder_campaign._TOKEN`) `[M, lanes ladder-run-campaign, ladder-review-file]` are exactly
  the edges a member boundary refuses; each becomes a public port or disappears.
- `tests/` is one directory of 456 files `[M, find]`; splitting it per member is part of each
  move, not a sweep at the end.

### Three teeth that make it bind

1. `tests/test_identity_epoch.py` (or its per-member successor) pins the canonical digest of
   `platform.toml` and of every `series.toml` beside the module-set digests it pins today, so a
   pin edit is a deliberate re-pin commit.
2. `.importlinter` gains the `platform` never imports `series` contract and a mutual
   independence contract across `series.*`; both proven by mutation before they ship.
3. A prek hook strict-loads every spec at commit; a spec that fails to load blocks the commit
   and never degrades to a default.

---

## Phases

Sizes are file counts from the survey's series attribution `[M, survey.json series tables]`
unless marked `[E]`. The order is forced by three facts: the member layout must exist before any
file moves, so each file moves once, straight into its final member; forwarders at old import
paths ride for exactly one wave (the w2 ruling of 2026-09-23); and each series move that touches
reader-identity members ends in one deliberate re-pin commit the conductor owns.

| phase | content | size | rail |
| --- | --- | --- | --- |
| P0 | `CLAUDE.md` north star and horizon section; Linear project; sprint manifest ratified; the debt plan's phases 0 to 2 landed (three correctness fixes, digest seam, codec seam) | docs plus about 30 modules `[E]` | gate before any lane |
| P1 | workspace skeleton: root `pyproject.toml` with the members, each member an empty distribution with its own `pyproject.toml`, `tests/` and `CLAUDE.md` (R7), one lockfile proven by `uv sync`; the nested `CLAUDE.md` sentinel measurement; `kernel/spec` loader and digest; `platform.toml` with the three tool pins, the image digest, registry and topology names; reconcile the two runpodctl pins | about 10 modules plus 8 manifests plus 7 member rule files `[E]` | proves digest-in-identity, the member gate and nested rule loading end to end before anything moves |
| P2 | `platform/pod` carved from the D1-named shared kernel: operator values, command runner, mailbox contracts, object transport, S3 adapter, DVC remote reader; forwarders at old paths | 6 modules, 40+ importers `[M, lane ladder-run-campaign inbound]` | pure rename plus import rewrite |
| P3a | `platform/ports.py` and the registry; `series.toml` for e, d, h in place; bind the three existing trainers and analyses behind the scientific ports without moving them; `platform/observability` ports with the jsonl and mlflow adapters moved behind `MetricSink` and `Tracker`, and the Trackio adapter added (D6) with the same conformance test the MLflow adapter passes | 3 trainers, 3 analyses, construction; 4 sink adapters `[M, lane ladder-capture]` | arch-gated; typestate must survive review; the tracker conformance test is the proof that line of sight is a port |
| P3b | ONE training engine: unify the three pipelines under D5 (sealed chain with a per-series payload slot); series keep only `ModelFactory` and `Objective` | 3 trainers, 3 checkpoint custodies, the journal and seal modules `[M, lanes ladder-run-core, ladder-run-training]` | the phase with scientific risk: identity-bearing custody; its own design gate; bitwise-equal tiny-fixture proof per series before P4 |
| P3c | ONE pod machinery: the H shape (hash-bound request, watchdog, once-only allocation) absorbs the D1 campaign controller's attempt flow and billing closure behind the `Provider` and `ObjectStore` ports | D1 `campaign/` + `closure/` + `transport/` (about 45 files), H `h_probe_*` + `worker_*` (about 20) | arch-gated; the D1 sequential queue and checkpoint CAS become engine features, not D1 features |
| P4 | `entropic-series-e` | 45 files | no reader-identity members; may run beside P2 |
| P5 | `entropic-series-d` and the protocol split under R1: register, review, file, gate go to platform; D1 roles, family, rungs, factorial stay with D | about 190 files | two D1 capture manifests embed hashed module strings `[M, lane ladder-capture debt]`: one re-pin commit |
| P6 | `entropic-series-h` | 65 files, 22 in the checkpoint reader tuple `[M, lane core-hseries-engine]` | one reader re-pin commit; last for that reason |
| P7 | import-linter contracts, delete forwarders, retire `historical/` and `entrypoints.env`, per-member bare-directory gate, shrink the root `CLAUDE.md` to the repo bearing, member map and cross-member rules (R7), docs sweep of the stale measured-claim paths | contracts, two rule files, comments | starts red unless P2 to P6 are clean |

Waves: P1 then P2 serial (P2 needs the member layout and the spec loader); P3a, P3b, P3c serial
and arch-gated, in that order; P4 may run beside P2 or P3a (file-disjoint); P5 and P6 serial,
each ending in the conductor's re-pin; P7 serial. The 456 test files are the hidden cost: every
move rewrites test imports and relocates the test, which is why the forwarders exist.

### What each phase must prove

- P1: a changed pin in `platform.toml` turns the identity test red; a malformed spec blocks a
  commit through the prek hook; the runpodctl version used by the H launch and the D1 adapter is
  one value; `uv sync` resolves the workspace and each empty member's gate runs from a bare
  directory; a sentinel token placed only in `packages/entropic-platform/CLAUDE.md` is answered
  PRESENT by a fresh `claude --print` asked from the repo root while a platform file is being
  read, with a control question answered from the root file (the 2026-08-14 method); an ABSENT
  result routes R7's local rulings through `.claude/rules/` with `paths:` frontmatter instead.
- P2: the H worker and the D1 operator both import the platform modules; the old paths forward;
  `just check` green at the tip.
- P3a: each of the three trainers runs its existing tiny smoke through the scientific ports with
  no behaviour change (bitwise-equal checkpoint payload on the tiny fixture); the registry
  refuses a plugin module outside the allowlist and a plugin missing a port.
- P3b: one engine runs the E, D1 and H tiny fixtures; each produces the same telemetry records
  as the retired trainer (compared through the admissible reader, never by eye); the re-pin
  commit carries the measured before and after reader digests; resume from a sealed chain works
  for all three payload slots.
- P3c: a D1 tiny campaign and an H tiny probe both launch through the one machinery against the
  local provider adapter; the watchdog tears down a pod whose controller dies (the measured
  idle-burn failure of 2026-08-21 must be the test).
- P4 to P6: the per-series tiny tracer or smoke runs after the move produce the same telemetry
  records as before; the re-pin commit carries the measured before and after digests; the
  member's own gate is green from a bare directory.
- P7: both new import-linter contracts go red under a one-line mutation and green on revert,
  recorded in the PR body; every member passes the copy test (clone the member directory alone
  with its `pyproject.toml`, `CLAUDE.md`, spec and tests, `uv sync`, run its gate, and ask the
  sentinel question from inside it with no root file present).

---

## Risks

| risk | likelihood / impact | mitigation |
| --- | --- | --- |
| Reader-identity re-pins land while an H confirmatory lineage is open | low under T1, high if T1 slips | T1: nothing moves until H1 completes; P6 is last |
| Unified checkpoint custody (D5) silently changes what a resumed run restores | medium / high | P3b's bitwise-equal proof per series and resume test per payload slot; the old readers stay as read-old adapters for one tested rollback window (compat policy for persisted artifacts) |
| Specs become casual config and pins drift | medium / high | teeth 1 and 3; the identity test is the same instrument that guards module sets today |
| The protocol split under R1 leaves D1 roles half in platform | medium / medium | P3a's design gate lists every `d1_*` module with its destination member before P5 starts; the module index is the inventory |
| Forwarders outlive their wave and become a second `__legacy` | medium / medium | one wave, then P7 deletes them; a test counts forwarders and fails above zero after P7 |
| Workspace members drift into importing each other through the shared lockfile | medium / medium | per-member pyright and import-linter contracts; the copy test in P7 catches it structurally |
| Test import rewrites exceed the lane budget | high / low | mechanical rewrite by tool with a dry run first, per the bulk-edit default |

---

## DECISION-NEEDED register

| # | decision | recommendation | status |
| --- | --- | --- | --- |
| D1 | spec format | TOML for platform and series specs; YAML stays for experiment hyperparameters | open |
| D2 | reader-identity member lists in `series.toml` | yes, with the spec digest pinned by the identity test; composed identity by concatenation in registry order | open |
| D3 | wire schema version strings | stay in code, one enumeration module per member | open |
| D4 | start trigger | H1 filed complete (its exit in `CLAUDE.md` Horizon) | ruled T1 |
| D5 | unified checkpoint custody | the H sealed chain with a per-series payload slot | open |
| D6 | second tracker adapter | Trackio behind the `Tracker` port; W&B-compatible surface comes with it | open |
| D7 | platform member north star and horizon | written before P1, in the member's own `CLAUDE.md`, stating the library bearing of R6 and the reach (which harness shapes it must host before it is sufficient); separate from the repo's north star and horizon, which stay in the root file | ruled 2026-10-30: separate documents |

Rulings are recorded here, in this file, on the commit that records them; chat is transport.
