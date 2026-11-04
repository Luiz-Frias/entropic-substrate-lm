# Series Plugin Topology Plan: platform engines, series adapters, digest-bound specs

Written 2026-10-30 on the read-only survey worktree `entropic-substrate-arch-survey-842143` at
`b9c570e0535c228c6bd4ea975ad87749bd6d7030` (main). It records the operator rulings of
2026-10-30 and the target end state for a sprint that starts **only after H1 completes entirely**
(ruling T1). Nothing in this plan authorizes a lane, a pod, or a push. Before the first lane
launches, the repo `CLAUDE.md` gains a north star and horizon section for this initiative
(house rule: no development without both) and a sprint manifest is ratified.

**Amended 2026-11-01 (rung five)** from the corpus review
`docs/reviews/2026-11-01-series-plugin-topology-corpus-review.md`, which read every rules card,
doctrine body, pattern card, playbook and the TypeScript, Rust and Python packs against this
plan. Its amendments are folded below in place, each marked "Amended 2026-11-01"; rulings R8 to
R10 record the operator's answers to D8 to D10; D11 was opened and ruled the same day as R11.

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
| R2 | Series plug in through an **explicit registry in composition**, shaped by the spec files below; no entry-point discovery. | The registry loads `series/*/series.toml`, binds each plugin by an allowlisted module path, and type-checks it against the ports. Amended 2026-11-01: the set of series is a promise, so `SeriesId` is a `StrEnum` in `entropic-cli` and a spec may only name a member of it; the registry refuses two specs with one `id`, checks each spec's `platform_api` against the version `ports.py` exports, binds in order sorted by `id`, wraps a plugin import failure as `PluginLoadError(series_id, cause)` and never a bare `ImportError`, hands a plugin only the ports it implements, and prints the resolved registry with every spec digest at startup without secrets; that record enters the run identity. |
| R3 | Retirement is a **registry flag plus an admission row**. | `series/e` stays where it is when retired; its spec says `status = "retained"` and the admission table refuses new confirmatory runs. Nothing moves between directories on a status change. |
| C1 | Infrastructure values such as hashes and pins are set from **spec files**, not code literals. | `platform.toml` and `series/<x>/series.toml`, strict-loaded, frozen, digest-bound (below). Amended 2026-11-01: every spec carries `spec_version`; the digest is over the canonical JSON of the parsed frozen value, never file bytes; numerics are a code promise and leave the spec (R10). |
| R4 | **Engines are platform; series are adapters.** The training pipeline, the pod machinery and capture are owned once, by the platform, and call out through ports. A series supplies the scientific adapters (model construction, objective, telemetry schema, analysis, filing roles, admission rows) and nothing else: no step loop, no checkpoint I/O, no pod code. | The three trainers and two pod machineries above collapse to one each (phases P3b and P3c). The series packages shrink to what differs between E, D and H. |
| R5 | **Extraction-ready modularity.** The target is a uv workspace whose members mirror a Rust multi-crate repository: each member has its own `pyproject.toml`, declared dependencies, tests and gate, so a member can be copied into another repository with a light refactor. The platform generalizes the infrastructure; a series extends it by swapping adapters. | The directory tree below is the member list. The acceptance test for every member is its own gate run from a bare directory (the validate-from-the-target's-context rule). Amended 2026-11-01: series members are packaging and gate boundaries, not hexagons, and own no `ports` module (tooth 4); each member gate runs `deptry` against its own `pyproject.toml`, the Python half of Cargo's undeclared-dependency refusal; each member carries a `.python-version` at the regime patch version so a lifted member keeps its identity regime; optional adapters are extras, the Python form of a cargo feature. |
| R6 | **The platform is a library for training machinery, not for this repo's series.** Its motive is to be lifted into scratch repositories to try other schedules, curricula, backbones and skeletons, mixture-of-experts machinery, RL environments, fine-tuning and LoRA adapters. It owns what sits around the optimizer step (admission, launch, custody, telemetry, capture, evidence), not what PyTorch does (autograd, modules) or what Ray does (distributed scheduling). | The scientific ports are named by what a training harness needs (`ModelFactory`, `Objective`, `Schedule`, `Source`, `TelemetrySchema`, `AnalysisStrategy`, `FilingRoles`, `Admission`; `Source` replaced `Curriculum` and `Environment` under R9), never by E, D or H. The member gets its own north star and horizon before P1 (house rule), because this ruling widens its bearing beyond the repo's. Amended 2026-11-01: the first scratch consumer is the second consumer under the paved-road rule and names the seam with a review; extraction to a repository with its own owner waits for the third. Consumption modes are R8. |
| R7 | **Each member carries its own `CLAUDE.md`.** Three parts: bearing (the platform member states its own north star and horizon under D7; series members inherit the repo's and name the horizon stage they serve), boundaries (the ports it exposes or implements, the members it may import, the spec files it reads, its gate command), and local rulings (numerics islands, custody rules, series-specific doors). Precedence is the existing scope-first rule: a member file is the most local scope for files under it and wins over the root on conflict, with the conflict named aloud. The root `CLAUDE.md` shrinks to the repo bearing, the member map, and the cross-member rules no single member can own (publish sequence, live gate, secrets). | A lifted member keeps its rules without the old root (R5). Nested loading is documented, not measured in this repo; P1 repeats the 2026-08-14 sentinel measurement before any member relies on it, with the documented `paths:` frontmatter rule under `.claude/rules/` as the fallback (path-scoped, but it does not travel with a copied member). Amended 2026-11-01: the member file names the member's system class and the operator STOP list that class carries (`entropic-kernel` class 1 for numerics and class 5 for the spec loader; `entropic-core` class 1; `entropic-platform/engine` and `/pod` class 2, where contracts, checkpoint wire shape and pipeline promises are the operator's; `entropic-platform/capture` through `/gate` class 4; series members class 1 plus 4, where the algorithm and its parameters are the operator's; `entropic-cli` class 3 and 5), its gate command, and its public surface, the `__all__` of the member root. |
| C2 | **Observability is a platform port family, and the evidence plane stays authoritative.** JSONL plus DVC capture is the only verdict input; every tracker (MLflow today, Trackio and a W&B-compatible adapter next) is a best-effort mirror behind a `Tracker` port, with the fail-loud strict adapter as the one D1 exception. | `MetricSink`, `Tracker`, `ArtifactStore` and `RunRegistry` are ports in `platform/observability`; adapters are `jsonl` (authoritative), `mlflow`, `trackio`, `wandb-compatible`. Line of sight into the platform is a port, not a dependency on one vendor. Amended 2026-11-01: a tracker outcome is a value that never escapes the composite sink as an exception, and the authoritative run record carries `mirrors = [{tracker, status, last_step}]` with status in `ok`, `degraded`, `refused`, so a lagging mirror is a recorded gap and never silence; the D1 strict adapter is the one raise-through member and says so in its name; `RunRegistry` is a projection over the capture ledger and a tracker's own registry mirrors that projection; tracker SDKs sit behind extras and the `mlflow-only-from-adapter` contract generalizes to one contract per tracker SDK. |
| R8 | **Two first-class consumption modes for the platform** (ruled 2026-11-01, closes D8). Default: a scratch repository consumes `entropic-platform` as a pinned dependency (`[tool.uv.sources]` git tag or path) and sends its refactors upstream. Alternate, chosen by the operator per repository: a **lift**, a clone or fork that keeps history, refactored to the new repository's own goals as a new lineage with its own north star and horizon. "Copy and paste" names the shape of the lift recipe, not its mechanism. | The lift recipe is implemented and tested, never optional: P7's copy test proves the member lifts clean, and `docs/platform/lift.md` records the recipe (clone or fork keeping history, re-pin `.python-version` and `platform.toml`, run the member gate from the bare directory, write the lineage's own `CLAUDE.md` bearing). The pinned mode keeps wrap-sibling-never-patch: a consumer-side patch is a defect signal filed upstream. A lifted lineage is a class 6 spike until its own north star and horizon are written; cross-pollination back is by cherry-pick or port, never shared history. |
| R9 | **One `Source` port, two implementation families** (ruled 2026-11-01, closes D9). `Curriculum` is the supervised family: data ordering and sampling with an order digest; its planned pipeline is the bounded FineWeb-Edu capture named in `docs/hierarchical-latent-predictor.plan.md` plus enwik8 for regression and enwik9 as the operator named it on 2026-11-01. `Environment` is the RL family: it stays modular and extensible as a factory over strategies, so an external RL library or a bespoke environment plugs in behind one adapter port. The seam is implemented now; the RL arm is a typed placeholder that refuses by name, never a stub that writes. | `Source` replaces `Curriculum` and `Environment` in the port list (eight ports). `platform/sources/` holds `curriculum/` (the two measured implementations today) and `environment/` with the `EnvironmentAdapter` port, an `EnvironmentFactory` keyed by library, and one placeholder adapter that raises `NotImplementedError` naming the decision-complete ticket filed at P3a and carries an expiry test that goes red when the first RL objective lands without it (the pull-forward rule's admissible deferral). The external environment API the adapter speaks is R11. |
| R10 | **Numerics are a promise, not a selection** (ruled 2026-11-01, closes D10). The FP32 computation islands are repo doctrine and `require_execution_precision` stays a code gate. | `[numerics]` leaves `platform.toml`. The only selectable numerics value is the ambient autocast dtype, in the run profile, branched on compute capability per the hardware playbook; the islands stay FP32 regardless. `TORCH_ALLOW_TF32` and `NVIDIA_TF32_OVERRIDE` are refused by a code gate when set, never read as configuration. |
| R11 | **The RL adapter speaks the Gymnasium contract, with TorchRL as the first adapter** (ruled 2026-11-01, closes D11). The `EnvironmentAdapter` port is shaped on Gymnasium's `reset` and `step` contract with `terminated` and `truncated` kept distinct, the standard the Farama Foundation maintains. TorchRL `EnvBase` is the first concrete adapter: it wraps Gymnasium environments and ships `torchrl.envs.llm.ChatEnv`, `LLMCollector`, `GRPOLoss` and `SFTLoss` under the PyTorch organization, so classic and LLM environments plug in through one adapter and a bespoke environment implements the port directly. Hugging Face TRL waits for the first LLM-RL objective and is consumed for its reward and rollout pieces, never its trainer. | Pins and dates `[M, web 2026-11-01]`: Gymnasium v1.3.0 (2026-04-22) clears the `exclude-newer` anchor of 2026-08-13; v1.4.0 (2026-10-05) needs the anchor advanced forward on its schedule, never backdated; TorchRL's LLM API arrived in 0.9.0 (2025-07-10) and the current release is v0.14.1 (release date not read). Both land as extras of the platform member (`entropic-platform[gymnasium]`, `entropic-platform[torchrl]`) behind the `EnvironmentFactory`, under the official-SDK-adapter form: the client is constructed only in the adapter, vendor types never cross the port, one live probe per adapter. The placeholder adapter under R9 is replaced by the TorchRL adapter when the first RL objective lands; its expiry test is what forces that. |

### Recommended picks awaiting the operator (DECISION-NEEDED register, end of file)

| # | question | recommendation | why |
| --- | --- | --- | --- |
| D1 | Spec format | TOML for `platform.toml` and `series.toml`; experiment hyperparameters stay YAML under each series. Amended 2026-11-01: every spec carries `spec_version`, and the digest is SHA-256 over the canonical JSON of the parsed frozen value (sorted keys, typed scalars), never over file bytes | stdlib `tomllib`, no implicit typing, pins are typed scalars; YAML stays where the three existing manifest families already are; TOML has no canonical serialization, so a byte digest would re-pin on whitespace |
| D2 | Reader-identity member lists | move into `series.toml`; the identity test pins the spec digest; the composed identity is the concatenation of member digests in registry order. Amended 2026-11-01: registry order is sorted by `id`; the composed identity is generated by the identity tool inside the gate and diffed, never hand-edited | the list is selection, the digest is the promise; a spec edit is a deliberate re-pin commit exactly as a tuple edit is today, and no single member has to own a hash that spans members |
| D3 | Wire schema version strings | stay in code beside their codecs; one enumeration module per member gives the 343 literals a single home | a schema version names a record shape the reader promises to accept; putting it in a spec lets a config edit change what evidence is admissible |
| D5 | Unified checkpoint custody | one platform checkpoint format with a sealed chain (the H shape) and a per-series payload slot; E and D payloads become slot contents with their own schema version. Amended 2026-11-01: the slot is a typed port, `CheckpointPayload` with `encode`, `decode` and a per-series `payload_schema` version kept in code beside its codec (D3), because an opaque series-specific mapping is the under-determined-port tell; the unified format is `platform-training-checkpoint-v3`, written only by the engine, with `h-training-checkpoint-v2`, the D1 15-key mapping and the E `format_version` file admitted read-old for one tested rollback window and deleted by a removal PR whose test proves no retained artifact needs them; sealed H0 and H1 checkpoints never migrate, replay at their epoch commit, and P3b and P6 add their rows to `docs/identity/read-old-table.md` | the H chain is the strictest of the three and already carries source and reader identity; the other two have no production confirmatory run open under T1 |
| D6 | Second tracker adapter | Trackio, behind the same `Tracker` port MLflow sits behind; its `wandb.init/log/finish` surface also covers a W&B-compatible adapter for free. Prerequisite added 2026-11-01: the pinned Trackio release must resolve under the `exclude-newer` anchor (`2026-08-13` in `pyproject.toml`), or the anchor advances forward on its schedule, never backdated | local-first, SQLite-backed, no server to stand up in a scratch repo, optional sync to a Hugging Face Space; adopted by alphaXiv OpenResearch on 2026-10-23 `[M, @askalphaxiv post 2026-10-23; alphaXiv/OpenResearch issue 571; huggingface.co/blog/trackio]` |

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

**The digest, defined (amended 2026-11-01).** TOML has no canonical serialization, so the
digest is SHA-256 over the canonical JSON of the parsed, frozen value: sorted keys, typed
scalars, no floats where an integer or string is declared. File bytes are not the identity.
Every spec carries `spec_version = N`; the loader enumerates the versions it admits in code and
refuses others by name, which keeps the additive-extension question answerable the day the
platform is lifted.

**Environment names, classified once (amended 2026-11-01).** The five names read today
`[M, grep at 07129a58]` go three ways. Selection moves into a spec and is digested:
`MLFLOW_EXPERIMENT_NAME` to `platform.toml [observability]`, `ENTROPIC_SUBSTRATE_DEVICE` to the
run profile. Runtime identity injected by the provider (`RUNPOD_POD_ID`) is recorded, never
configured. Numerics promises (`TORCH_ALLOW_TF32`, `NVIDIA_TF32_OVERRIDE`) are refused by a code
gate when set (R10). Secrets stay out of band. Nothing layers silently over a spec: the CLI
prints the effective configuration with its digest at startup.

```toml
# platform.toml (one per deployment regime; digest enters every run identity)
spec_version = 1
[image]
reference = "ghcr.io/<org>/entropic-substrate-lm"
digest = "sha256:<64 hex>"          # today docs/deploy-h100.md:12
[tools]
runpodctl = "3.0.1-551d6a4"          # today two disagreeing code literals
pysbd = "0.3.4"
[topology]
dvc_remote = "r2-entropic-substrate-lm"   # today .dvc/config (stays there; this names it)
object_store_prefix = "s3://entropic-substrate-lm"
[observability]
experiment = "entropic-substrate-lm"  # today the MLFLOW_EXPERIMENT_NAME environment read
# no [numerics] table: the FP32 islands are a code promise (R10, 2026-11-01)
```

```toml
# packages/entropic-series-h/series.toml
spec_version = 1
platform_api = 1                     # checked against the version ports.py exports (R2)
id = "h"                             # must be a member of entropic_cli.SeriesId (R2)
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
| platform scientific ports | `ModelFactory` (spec to module plus parameter attestation); `Objective` (forward, loss, step metrics, evaluation; called once per step, never per op, so the hot loop keeps its static shape); `Schedule` (learning-rate and phase schedule); `Source` (next batch for a step with its order digest; `Curriculum` is the supervised family and `Environment` the RL family behind `EnvironmentAdapter`, R9); `TelemetrySchema` (run_start and run_end shapes, lane tag, admissibility); `AnalysisStrategy` (admit, aggregate, report, check suite); `FilingRoles`; `Admission` (run classes by surface by stage) | `ladder/run/construction`, `core/substrate.build_model`; the three trainers' step and eval bodies; the warmup/cosine schedule in `d1_training_step` and the H `RunSchedule`; `SequenceSampler` and the H sentence batch sampler, plus the refusing RL placeholder under R9; the capture lane partition; `d1_factorial`, `h_aggregate`, `series.py`; the D1 13-role bundle and `ladder_filing`; `ladder/run/admission.py` and the D1 confirmatory matrix in `ladder/register/d1_confirmatory_matrix.py` (corrected 2026-11-01: the port has two implementations, not "H only") |
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
- Amended 2026-11-01: a tracker outcome is a value. The composite sink records
  `mirrors = [{tracker, status, last_step}]` in the authoritative run record, so a mirror that
  fell behind is a recorded gap rather than silence (the Artifact axiom). `RunRegistry` is a
  projection over the capture ledger, never a second source of truth. Tracker SDKs are extras
  of the platform member, and the jsonl adapter is never optional.

Comparison for orientation, not a dependency: Thinking Machines' Tinker exposes a managed
fine-tuning service as a few primitives (forward-backward, optimizer step, sample, save state)
with LoRA adapters on top. The platform engine's port set is the open, self-hosted analogue of
that primitive set plus the evidence and custody machinery this repo already has; it is not a
training framework and does not compete with PyTorch or Ray.

Typestate survives unchanged: the lifecycle FSM stays in kernel, factory attestation stays the
only way to mint filing-eligible values, and adapters are passed into platform factories rather
than discovered. The registry's own typestate is `SeriesSpec` (parsed, digest known) to
`AdmittedSeries` (status and admission rows checked against `platform.toml`) to `BoundSeries`
(plugin module imported from the allowlist and type-checked against `ports.py`). Amended
2026-11-01, the Python form named: the three are frozen, slotted, `init=False` classes under
`entropic_cli.registry`, minted only by the registry's own factories, the form this tree already
uses in 14 places `[M, grep at 07129a58]`; each transition is a method that returns the next
class; the runtime guard stays because Python cannot make the static guarantee. The compile-fail
lane is a `tests/typing/` fixture tree run by basedpyright with
`reportUnnecessaryTypeIgnoreComment` on, so an expected-error comment goes red when the guard
weakens; the kernel lifecycle FSM becomes a total transition function with `assert_never`.

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

### Four teeth that make it bind (three until 2026-11-01)

1. `tests/test_identity_epoch.py` (or its per-member successor) pins the canonical digest of
   `platform.toml` and of every `series.toml` beside the module-set digests it pins today, so a
   pin edit is a deliberate re-pin commit. Amended 2026-11-01: a `hypothesis` round trip proves
   the digest is stable under key reordering and whitespace and changes on any value edit.
2. `.importlinter` gains the `platform` never imports `series` contract and a mutual
   independence contract across `series.*`; both proven by mutation before they ship. Amended
   2026-11-01: the mutation includes the string-import form, because this repo measured that the
   layers contract stays green on `importlib.import_module` shims (the `.importlinter` header);
   a prek ast-grep rule therefore refuses `importlib.import_module` and `__import__` outside
   `entropic_cli`, itself proven by mutation; and a forbidden contract keeps series out of
   platform submodules, so a series may import `entropic_platform` and `entropic_platform.ports`
   only, the Python approximation of `pub(crate)`.
3. A prek hook strict-loads every spec at commit; a spec that fails to load blocks the commit
   and never degrades to a default. Note added 2026-11-01: this and the ast-grep rule edit
   `.pre-commit-config.yaml`, which is an operator ask under the non-negotiable rule; P1 lists
   both as decisions, not lane actions.
4. Added 2026-11-01: a test fails if any `entropic-series-*` member contains a `ports` module or
   a Protocol class whose name ends in `Port`; only the platform owns ports.

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

**Vehicles, hub owners and the open-door rule (amended 2026-11-01).** Vehicles follow the
measured bands in `pick-vehicle-by-size`: P1, P3a, P3b and P3c are one cascade-arch lane each
(a door or a design judgment); P2 and P4 are mechanical and run as one lane or hand-authored
under the measured band; P5 and P6 are one lane each plus the conductor's re-pin; P7 is one
lane. Every hub file has one owning stack per wave: P1 owns `pyproject.toml`, `uv.lock`,
`.importlinter`, `platform.toml`, `tests/test_identity_epoch.py` and the
`.pre-commit-config.yaml` ask, and declares all seven members up front so the root manifest and
the lockfile are touched once; later phases add to their own member only. P1 also updates the
worktree-create hook to `uv sync --all-packages`, because a fresh worktree with one package root
installed fails the member gates with errors that look like base defects. P1, P3b, P5 and P6
each move a door (configuration, checkpoint custody, reader identity twice) and bind
`pull-forward-at-open-door`: every producer and consumer of the door is truthful at the lane's
tip, with no stub reader and no placeholder status.

### What each phase must prove

- P1: a changed pin in `platform.toml` turns the identity test red; a malformed spec blocks a
  commit through the prek hook; the runpodctl version used by the H launch and the D1 adapter is
  one value; `uv sync` resolves the workspace and each empty member's gate runs from a bare
  directory; a sentinel token placed only in `packages/entropic-platform/CLAUDE.md` is answered
  PRESENT by a fresh `claude --print` asked from the repo root while a platform file is being
  read, with a control question answered from the root file (the 2026-08-14 method); an ABSENT
  result routes R7's local rulings through `.claude/rules/` with `paths:` frontmatter instead.
  Added 2026-11-01: each empty member's gate runs `deptry` and basedpyright over its
  `tests/typing/` fixtures; the file-ceiling lint is set to the tightest threshold the tree fits
  (largest file today 1007 lines `[M]`) and only ratchets down; `uv sync --all-packages` from a
  fresh worktree installs every member root.
- P2: the H worker and the D1 operator both import the platform modules; the old paths forward;
  `just check` green at the tip.
- P3a: each of the three trainers runs its existing tiny smoke through the scientific ports with
  no behaviour change (bitwise-equal checkpoint payload on the tiny fixture); the registry
  refuses a plugin module outside the allowlist and a plugin missing a port.
- P3b: one engine runs the E, D1 and H tiny fixtures; each produces the same telemetry records
  as the retired trainer (compared through the admissible reader, never by eye); the re-pin
  commit carries the measured before and after reader digests; resume from a sealed chain works
  for all three payload slots. Added 2026-11-01: the engine's run loop is one function readable
  top to bottom with no policy branching of its own, every series difference entering through a
  port call at a named point (thin means linear and total, not scattered); the engine member
  passes the file ceiling.
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
| A file-bytes digest re-pins identity on whitespace, or a parsed-value digest hides a type change (added 2026-11-01) | medium / medium | the digest is over canonical JSON of the typed frozen value (D1); tooth 1's round trip pins both directions |
| An environment variable layers silently over a spec after the move (added 2026-11-01) | medium / high | the five names are classified once (seam section); the CLI prints the effective configuration and its digest at startup; numerics names are refused, not read |

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
| D7 | platform member north star and horizon | written before P1, in the member's own `CLAUDE.md`, stating the library bearing of R6 and the reach (which harness shapes it must host before it is sufficient); separate from the repo's north star and horizon, which stay in the root file. Amended 2026-11-01: the horizon names its oracle, the three series tiny fixtures plus one foreign fixture (a LoRA fine-tune of a tiny public checkpoint, the smallest shape R6 names) run through the one engine with telemetry compared through the admissible reader; the Tinker sentence becomes a four-row parity matrix (forward-backward, optimizer step, sample, save state) or is removed; the horizon closes with the plan-versus-actual audit | ruled 2026-10-30: separate documents |
| D8 | how a scratch repository consumes the platform | both modes first class: pinned dependency by default, lift (clone or fork keeping history, refactored to the new repository's goals) as the operator's alternate; the lift recipe is implemented and tested, never optional | ruled 2026-11-01: R8 |
| D9 | fold `Curriculum` and `Environment` into one `Source` port | yes; `Curriculum` is the supervised family with the FineWeb-Edu, enwik8 and enwik9 pipeline planned; `Environment` is the RL family behind an `EnvironmentAdapter` port and factory, implemented now as a typed refusing placeholder with an expiry test | ruled 2026-11-01: R9 |
| D10 | remove `[numerics]` from `platform.toml` | yes; the FP32 islands are a code promise, `require_execution_precision` stays a gate, the ambient dtype is the only selectable numerics value and lives in the run profile | ruled 2026-11-01: R10 |
| D11 | which external environment API the `EnvironmentAdapter` speaks (opened 2026-11-01 under R9) | shape the port on Gymnasium's `reset`/`step` contract with `terminated` and `truncated` kept distinct, the de facto standard maintained by the Farama Foundation; make TorchRL `EnvBase` the first concrete adapter, because it wraps Gymnasium environments and ships `torchrl.envs.llm.ChatEnv`, `LLMCollector`, `GRPOLoss` and `SFTLoss` for LLM post-training under the PyTorch organization, so classic and LLM environments plug in through one adapter and a bespoke environment implements the port directly; defer Hugging Face TRL to the first LLM-RL objective, consuming its reward and rollout pieces rather than its trainer. Measured 2026-11-01 `[M, web]`: Gymnasium v1.4.0 released 2026-10-05 and v1.3.0 on 2026-04-22, so v1.3.0 clears the `exclude-newer` anchor and v1.4.0 needs it advanced; TorchRL's LLM API arrived in 0.9.0 on 2025-07-10 and the current release is v0.14.1 (release date not read) | ruled 2026-11-01: R11, the recommendation as written |

Rulings are recorded here, in this file, on the commit that records them; chat is transport.
