# Series Plugin Topology Plan: platform ports, series plugins, digest-bound specs

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
| R1 | The ladder protocol (register, run, capture, analyze, review, file, gate) is **platform**. The D1 four-arm family, the D2 to D5 rung specs, the factorial analysis and the D1 filing roles are a **series**. | `platform/` is series-neutral; `series/d` holds the D-specific implementations of the platform ports. |
| T1 | The sprint runs as **its own sprint after H1 completes entirely**. | No topology move lands on the H1 trunk. The debt plan's phases 0 to 2 may land before, as rails on the current trunk. |
| R2 | Series plug in through an **explicit registry in `composition`**, shaped by the spec files below; no entry-point discovery. | The registry loads `series/*/series.toml`, binds each plugin by an allowlisted module path, and type-checks it against the SPI. |
| R3 | Retirement is a **registry flag plus an admission row**. | `series/e` stays where it is when retired; its spec says `status = "retained"` and the admission table refuses new confirmatory runs. Nothing moves between directories on a status change. |
| C1 | Infrastructure values such as hashes and pins are set from **spec files**, not code literals. | `platform.toml` and `series/<x>/series.toml`, strict-loaded, frozen, digest-bound (below). |

### Recommended picks awaiting the operator (DECISION-NEEDED register, end of file)

| # | question | recommendation | why |
| --- | --- | --- | --- |
| D1 | Spec format | TOML for `platform.toml` and `series.toml`; experiment hyperparameters stay YAML under each series | stdlib `tomllib`, no implicit typing, pins are typed scalars; YAML stays where the three existing manifest families already are |
| D2 | Reader-identity member lists | move into `series.toml`; the identity test pins the spec digest | the list is selection, the digest is the promise; a spec edit is then a deliberate re-pin commit exactly as a tuple edit is today |
| D3 | Wire schema version strings | stay in code beside their codecs; one enumeration module per series gives the 343 literals a single home | a schema version names a record shape the reader promises to accept; putting it in a spec lets a config edit change what evidence is admissible |

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
# series/h/series.toml
id = "h"
status = "active"                    # active | retained | retired  (ruling R3)
[entrypoints]
worker = "entropic_substrate.series.h.worker_command"     # today scripts/entrypoints.env
[plugin]
module = "entropic_substrate.series.h.plugin"             # allowlisted by the registry
[telemetry]
lane_tag_prefix = "h-"               # today the MLflow schema tag partition in capture
[[admission]]                        # today ladder/run/admission.py (H only)
run_class = "probe"
surface = "pod"
stages = ["warmup", "prediction"]
[reader_identity]                    # D2: today the 52-string tuple in ladder/run/checkpoint.py
checkpoint = ["entropic_substrate.kernel.artifacts", "entropic_substrate.kernel.lifecycle", "..."]
source = ["..."]
```

---

## Target topology (end state)

```text
src/entropic_substrate/
  kernel/                      unchanged, plus kernel/spec/: strict TOML loader, SpecDigest
  core/                        capability names, not series names
    byte_substrate/            today core/substrate + core/geometry (E engine; core/latent/state.py
                               also imports build_block from it, so it is not E-only)
    corpus/ latent/ observation/ sentence/ probe/   unchanged
  platform/                    series-neutral ports and default adapters (today: ladder/* minus
                               the d1_* and h_* specifics)
    spi.py                     the port Protocols a series implements (table below)
    run/                       Admission, Launcher, Worker, Transport, Mailbox, CheckpointSync ports;
                               adapters: runpod, r2/s3, local, docker (one each)
    capture/                   MetricSink port, jsonl/mlflow adapters, readers, custody, ledger
    analyze/                   rung-neutral registration, seed sizing, paired contrasts, palette
    review/                    blind-alias custody, packet, cloud review, adjudication
    file/ gate/                filing graph, one-way gate, successor authorization
  series/                      the plugin seam; one package per series, same SPI
    e/  series.toml  specs, byte trainer, E charts, configs/*.yaml
    d/  series.toml  D1 family, D2-D5 rung specs, factorial analysis, D1 filing roles, ladder engine
    h/  series.toml  sentence run engine (today ladder/run top level), probes, h_analysis, h_preservation
  composition/
    registry.py                loads series/*/series.toml; SeriesSpec -> AdmittedSeries -> BoundSeries
    entrypoints/ wiring/       the CLI root dispatches by series id
platform.toml
```

The import-linter contract becomes `kernel < core < platform < series < composition`, with
`series.*` packages mutually independent and `platform` forbidden from importing `series`.
`historical/` and `scripts/entrypoints.env` retire in the last phase.

### The SPI a series implements

| port | what a series supplies | who consumes it today `[M, module index]` |
| --- | --- | --- |
| `Construction` | spec value types, `build_system(spec)`, parameter attestation | `ladder/run/construction`, `historical/d_ladder`, `core/substrate.build_model` |
| `TrainerStrategy` | step, evaluate, checkpoint payload builder, resume validator | three trainers: `historical/e_series/train.py`, `ladder/run/trainer/d1_segment_engine.py`, `ladder/run/execution.py` |
| `TelemetrySchema` | run_start and run_end record shapes, lane tag, admissibility predicate | `ladder/capture/reader` lane partition keyed on the MLflow `schema` tag |
| `AnalysisStrategy` | admit captured runs, aggregate, report, check suite | `ladder/analyze/d1_factorial.py`, `h_aggregate.py`, `series.py` |
| `FilingRoles` | the artifact roles a verdict needs | D1 13-role bundle; `ladder/file/ladder_filing.py` for D2 to D5 |
| `Admission` | run classes by surface by stage | `ladder/run/admission.py` (H only today) |

Typestate survives unchanged: the lifecycle FSM stays in `kernel/lifecycle.py`, factory
attestation stays the only way to mint filing-eligible values, and strategies are passed into
platform factories rather than discovered. The registry's own typestate is `SeriesSpec` (parsed,
digest known) to `AdmittedSeries` (status and admission rows checked against `platform.toml`)
to `BoundSeries` (plugin module imported from the allowlist and type-checked against `spi.py`).

### Three teeth that make it bind

1. `tests/test_identity_epoch.py` pins the canonical digest of `platform.toml` and of every
   `series.toml` beside the module-set digests it pins today, so a pin edit is a deliberate
   re-pin commit.
2. `.importlinter` gains the `platform` never imports `series` contract and a mutual
   independence contract across `series.*`; both proven by mutation before they ship.
3. A prek hook strict-loads every spec at commit; a spec that fails to load blocks the commit
   and never degrades to a default.

---

## Phases

Sizes are file counts from the survey's series attribution `[M, survey.json series tables]`
unless marked `[E]`. The order is forced by two facts: forwarders at old import paths ride for
exactly one wave (the w2 ruling of 2026-09-23), and each series move that touches reader-identity
members ends in one deliberate re-pin commit the conductor owns.

| phase | content | size | rail |
| --- | --- | --- | --- |
| P0 | `CLAUDE.md` north star and horizon section; Linear project; sprint manifest ratified; the debt plan's phases 0 to 2 landed (three correctness fixes, digest seam, codec seam) | docs plus about 30 modules `[E]` | gate before any lane |
| P1 | `kernel/spec` loader and digest; `platform.toml`; move the three tool pins, the image digest, registry and topology names into it; reconcile the two runpodctl pins | about 8 modules `[E]` | smallest phase; proves digest-in-identity end to end |
| P2 | `platform/run` carved from the D1-named shared kernel: operator values, command runner, mailbox contracts, object transport, S3 adapter, DVC remote reader; forwarders at old paths | 6 modules, 40+ importers `[M, lane ladder-run-campaign inbound]` | pure rename plus import rewrite |
| P3 | `platform/spi.py` and `composition/registry.py`; bind the three existing trainers and analyses behind the ports without moving them; `series.toml` for e, d, h in place | 3 trainers, 3 analyses, construction | the design-gate phase (arch mode); typestate must survive review |
| P4 | `series/e` | 45 files | no reader-identity members; can run beside P2 |
| P5 | `series/d` and the protocol split under R1: register, review, file, gate go neutral; D1 roles, family, rungs, factorial stay with D | about 190 files | two D1 capture manifests embed hashed module strings `[M, lane ladder-capture debt]`: one re-pin commit |
| P6 | `series/h` | 65 files, 22 in the checkpoint reader tuple `[M, lane core-hseries-engine]` | one reader re-pin commit; last for that reason |
| P7 | import-linter contracts, delete forwarders, retire `historical/` and `entrypoints.env`, docs sweep of the stale measured-claim paths | contracts and comments | starts red unless P2 to P6 are clean |

Waves: P1 then P2 serial (P2 needs the spec loader for the entry-point names); P3 serial and
arch-gated; P4 may run beside P2 or P3 (file-disjoint); P5 and P6 serial, each ending in the
conductor's re-pin; P7 serial. The 456 test files `[M, find tests]` are the hidden cost: every
move rewrites test imports, which is why the forwarders exist.

### What each phase must prove

- P1: a changed pin in `platform.toml` turns the identity test red; a malformed spec blocks a
  commit through the prek hook; the runpodctl version used by the H launch and the D1 adapter is
  one value.
- P2: the H worker and the D1 operator both import the platform modules; the old paths forward;
  `just check` green at the tip.
- P3: each of the three trainers runs its existing tiny smoke through the `TrainerStrategy`
  port with no behaviour change (bitwise-equal checkpoint payload on the tiny fixture); the
  registry refuses a plugin module outside the allowlist and a plugin missing a port.
- P4 to P6: the per-series tiny tracer or smoke runs after the move produce the same telemetry
  records as before (compared through the admissible reader, never by eye); the re-pin commit
  carries the measured before and after digests.
- P7: both new import-linter contracts go red under a one-line mutation and green on revert,
  recorded in the PR body.

---

## Risks

| risk | likelihood / impact | mitigation |
| --- | --- | --- |
| Reader-identity re-pins land while an H confirmatory lineage is open | low under T1, high if T1 slips | T1: nothing moves until H1 completes; P6 is last |
| Specs become casual config and pins drift | medium / high | teeth 1 and 3; the identity test is the same instrument that guards module sets today |
| The protocol split under R1 leaves D1 roles half in platform | medium / medium | P3's design gate lists every `d1_*` module with its destination before P5 starts; the module index is the inventory |
| Forwarders outlive their wave and become a second `__legacy` | medium / medium | one wave, then P7 deletes them; a test counts forwarders and fails above zero after P7 |
| Test import rewrites exceed the lane budget | high / low | mechanical rewrite by tool with a dry run first, per the bulk-edit default |

---

## DECISION-NEEDED register

| # | decision | recommendation | status |
| --- | --- | --- | --- |
| D1 | spec format | TOML for platform and series specs; YAML stays for experiment hyperparameters | open |
| D2 | reader-identity member lists in `series.toml` | yes, with the spec digest pinned by the identity test | open |
| D3 | wire schema version strings | stay in code, one enumeration module per series | open |
| D4 | start trigger | H1 filed complete (its exit in `CLAUDE.md` Horizon) | ruled T1 |

Rulings are recorded here, in this file, on the commit that records them; chat is transport.
