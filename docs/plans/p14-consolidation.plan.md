# p14 Consolidation Plan: extract, abstract, consolidate

Written 2026-09-28 on `refactor/p14-consolidation` by the p14 planning lane. The operator rulings
of 2026-09-28 are binding and are restated in the rulings table below. The sprint manifest
`.claude/workflows/sprint-manifest.json` carries `ratified: false`: the operator ratifies it, and
no lane launches before that. The per-phase briefs under `.claude/workflows/briefs/p14-*.md` are
authoritative for each lane; this plan is their index.

**Evidence labelling.** Every number carries `[M, <source>]` measured or `[E]` estimated. Measured
and estimated values never share a row without a per-row label.

---

## Context

The whole-tree thermo-nuclear review ran at the p13 tip and is filed at
`docs/reviews/thermo-nuclear-p13-tip.md`. Its verdict is REJECT, read as the consolidation backlog
and never as a revert signal: behaviour is broadly correct, and the bar it fails is structural.
It names six presumptive blockers.

| # | blocker | disposition in p14 |
| --- | --- | --- |
| B1 | strict canonical-JSON boundary codec reimplemented tree-wide | rail `p14-r1-strict-json` plus four adoption phases |
| B2 | `gate <-> run` package cycle, FOR-1612 | held out, ruling 3 |
| B3 | `D1RunReader` silently drops gap metrics, FOR-1579 | fail-loud in `p14-j-analyze-review-file`, ruling 11 |
| B4 | MLflow cold-start retry orphans runs, FOR-1578 | `p14-t-capture-sink`, live gate |
| B5 | fresh trainer duplicates the resume trainer's failure helper, FOR-1568 | `p14-t-trainer` |
| B6 | `scripts/assemble_review_packet.py` past 1000 lines, FOR-1609 | out, ruling 13 |

The triage filed 48 issues into the Linear milestone `p14 consolidation`, FOR-1568 to FOR-1615
`[M, Linear]`. After the rulings, 41 of them are in the 15 phases below, two become pre-phase
rungs, and five are out or closed `[M, phase generator]`. Two more tickets from Bug Squashing, FOR-1567 and
FOR-1617, become pre-phase rungs too.

| quantity | value | evidence |
| --- | --- | --- |
| epoch | `4e0ae6a59117d054cc9b63d705d5004e06b982cb` on `main` | `[M, git rev-parse]` |
| trunk | `refactor/p14-consolidation` at `eb93d5d`, 651af30 plus one docs-only commit | `[M, git diff --stat 651af30 eb93d5d]` |
| modules / import edges | 378 / 1805 | `[M, tools/import_graph.py at eb93d5d, src identical to 651af30]` |
| layer inversions / package cycles | 1 / 4, equal to the p13 baseline | `[M, tools/remap_check.py over a mapping re-derived by the p13 procedure]` |
| committed `p13-sweep-mapping.json` against the epoch graph | FATAL: `composition.wiring.h_tiny_tracer` unmapped | `[M, remap_check.py]` |
| reader-tuple members | 51 checkpoint members, 7 source members, union 51 files | `[M, AST read of both _READER_MODULES tuples]` |
| files defining `_pairs`/`_strict_pairs`/`_nonfinite`/`_reject_constant` | 36 | `[M, grep -rlE 'def (_pairs\|_strict_pairs\|_nonfinite\|_reject_constant)\b' src]` |
| `def _pairs` / `def _strict_pairs` / `def _nonfinite` or `_reject_constant` | 28 / 2 / 22 | `[M, grep]` |
| `def _mapping(` | 23 | `[M, grep]` |
| `separators=(",", ":")` occurrences | 68 | `[M, grep]` |
| files naming a lowercase SHA-256 check | 29 | `[M, grep -rliE 'lowercase SHA-?256\|lowercase sha256' src]` |
| `# type: ignore[arg-type]` | 19 | `[M, grep]` |
| files over 1000 lines, `src` and `scripts` | 2: `composition/entrypoints/cli.py` 1144, `scripts/assemble_review_packet.py` 1038 | `[M, wc -l]` |

The review's own counts of 31, 67 and 27 come from different greps at `0e55d9d`. They are not
comparable to the rows above and are quoted here only for provenance.

---

## Admissibility rule

Ruling 14, binding for every abstraction a p14 lane introduces:

- An abstraction is admissible only when it consolidates two or more concrete sites, or when it
  stabilizes a public surface.
- A one-implementation port collapses unless it sits at an external boundary: RunPod, MLflow,
  DVC, R2, containers.
- Carry tiers in order: compile-time structure first, meaning typestate, sealed unions, newtypes,
  `StrEnum`; a builder only at a public surface; strategy, factory or registry last and only on a
  measured axis.
- Evidence per abstraction, in the PR body: the consolidated sites named with `path:line`, and the
  AST branch count at each call site holding flat or falling.

The two abstractions this sprint introduces are both consolidations: `kernel/codec/strict_json.py`
consolidates 36 primitive-defining files `[M]`, and the verdict `StrEnum` types consolidate five
bare-`str` vocabularies into compile-time structure. Everything else is extraction or deletion.

---

## Prerequisites and pre-phase rungs

**Prerequisites, in order.**

1. Tag `v0.1.0` exists on `main`. The conductor then rebases the trunk onto that tip, adds it as a
   Graphite trunk, and pushes it. Ruling 1.
2. A full `just check` runs at that epoch with no other lane active; the conductor pins its SHA.
   Ruling 2. The manifest's `verified_at` is re-stamped to that SHA at the prelaunch ceremony, and
   every premise in the briefs is re-probed there. The `src` diff from 651af30 is expected to be
   empty; a non-empty diff re-opens the touch sets.
3. **Any h-ladder flywheel experimental run needs a fresh operator green light.** No phase and no
   rung launches pods, spends GPU, or starts an experiment. Ruling 8.

**Pre-phase rungs.** Seven `fix/`, `chore/` and `feat/` branches (rung 5 added 2026-09-29 on ruling DN-1567-C;
rungs 6 and 7 on the operator's 2026-09-29 ask to move persistence off the training hot path), rungs 1, 2 and 5 a
stack (trunk → 1567 → 1617 → 1567c), rungs 3, 4 and 6 parented on the trunk, rung 7 gated on the first measured
run; landed serially in the listed order before `p14-r1-strict-json` launches. They are not manifest phases; each carries a brief under
`.claude/workflows/briefs/pre-for-*.md` (installed 2026-09-28/29), their scope lives there and in their
Linear tickets. Rulings 7 and 12.

| order | branch | ticket | scope | reader re-pin |
| --- | --- | --- | --- | --- |
| 1 | `fix/for-1567-checkpoint-publication-history` | FOR-1567 | make per-checkpoint work independent of run history: publish only new or changed artifacts plus a manifest that references already-verified keys; verify only the new seal against its verified parent; add a bounded retry on R2 verification READS only, never on publishes, with the retry count recorded in the receipt. Instrument first: per-leg timers so L1 seal re-verification, L2 the second full walk, and L3 the R2 republish are measured separately | yes, conductor: the path crosses `object_worker`, `worker_snapshot`, `worker_transport`, `seals`, `runner`, `journal`, all tuple members |
| 2 | `fix/for-1617-graceful-operator-stop` | FOR-1617 | a stop marker the worker honours at the next step boundary: seal, publish every sealed artifact, write a `stopped_by_operator` receipt, exit 0; a host verb that writes the marker and waits for the receipt; evaluation reports publish at seal time | yes, conductor: `runner`, `worker_command` are tuple members |
| 3 | `chore/for-1614-text-trim-semantics` | FOR-1614 | commit 1 audits every persisted launch-attempt ledger for untrimmed-but-nonblank strings and pins each codec's current `_text` boundary RED-first. Commit 2, only when the audit count is zero: `d1_launch_attempt_codec._text` becomes strict-trimmed, the pin flips, both arms shown. A non-zero count halts as DECISION-NEEDED for a read-old window | no: the three `_text` sites are not tuple members |
| 4 | `chore/for-1615-lowercase-sha-boundary` | FOR-1615 | audit every caller of the loose kernel `require_sha256` and classify each as boundary or in-process; rename it to `require_sha256_hex_any_case` and repoint the 34 `src` importers mechanically, zero behaviour change; commit the audit table as `docs/plans/measurements/p14-sha-callers.txt` | no: none of the 34 importers is a tuple member `[M, comm against the tuple files]` |
| 5 | `fix/for-1567-final-leg-publication` | FOR-1621 | option C of DN-1567-C (operator 2026-09-29): the `h-worker-event-v1` `complete` event gains the `publication` block (write-new, read-old for one rollback window in both readers) and the final snapshot leg reuses the run's publication ledger and the bounded verification-read retry; started and failure snapshots untouched; stacked on rung 2, brief `.claude/workflows/briefs/pre-for-1567c.md` | yes: `object_worker.py`, `worker_completion.py` |
| 6 | `feat/for-1622-mlflow-async-sink` | FOR-1622 | operator ask 2026-09-29 (persistence off the hot path, part 1): `MlflowSink.log` enqueues into a bounded queue drained by one background worker into `log_batch`; `finish()` drains with a declared bound; best-effort accounting unchanged (failed batches counted, dead worker reported, full queue blocks, never drops); the `MetricSink` port is not widened. Adapter-local, no reader member, live-required. Worked in a parallel worktree off the trunk tip, rebased and lifecycle-checked at the next gap; brief `.claude/workflows/briefs/pre-for-1622.md` | no |
| 7 | `feat/for-1623-publication-actor` | FOR-1623 | part 2, GATED on the first measured `publication` block from a green-lit run after rung 1: the runner seals locally as today and hands sealed keys to a background publication actor (ledger, retry, checkpoint events, stop-marker poll) with backpressure at N unpublished checkpoints, synchronous drain at the final/stopped boundary, and fail-loud health checks at every boundary; N sized from the measurement | yes: `runner.py`, `object_worker.py`, `worker_execution.py` (+ `worker_snapshot.py`, `worker_transport.py`, `h_probe_events.py`) |

**Why FOR-1615 splits this way.** The ruling asks for lowercase-strict SHA at every boundary and a
rename of the loose kernel function, landed before the consolidation phases. The lowercase-strict
predicate's ruled home is `kernel/codec/strict_json.py`, which the rail creates. Creating it in the
chore would pre-own the rail's frozen contract; defining a second lowercase function in
`kernel.identity` would recreate the very name collision FOR-1615 exists to remove. So the chore
lands the rename and the audit, the rail lands `require_lowercase_sha256`, and each adoption phase
repoints the audited boundary callers in its own package; `p14-t-historical` takes the two
`historical/d_ladder` importers. Every one of the 34 importers `[M, AST import scan at eb93d5d]` is placed in
exactly one adoption phase's touch set now, so the enforcement is total and the touch sets are
fixed before launch. This is a planning-lane sequencing decision, recorded as P14-SHA-SEQ below for
the operator to ratify.

---

## Phase table

| id | title | mode | issues | wave | live gate | identity members | touch set |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [p14-r1-strict-json](#p14-r1-strict-json) | strict canonical-JSON codec module | tracer | FOR-1613 | rail | ship-it | 0 | 3 |
| [p14-t-local](#p14-t-local) | TrainConfig betas, D1DvcBinding | ratchet | FOR-1585, FOR-1602 | w1 | zip | 0 | 4 |
| [p14-t-composition](#p14-t-composition) | one dispatch table, typed ports, canonical_json reuse | ratchet | FOR-1571, FOR-1583, FOR-1582, FOR-1584, FOR-1607 | w1 | zip | 0 | 10 |
| [p14-j-register](#p14-j-register) | register and gate codecs adopt strict_json | ratchet | FOR-1575, FOR-1599, FOR-1577, FOR-1613 | w1 | zip | 0 | 14 |
| [p14-t-trainer](#p14-t-trainer) | one failure helper, shared accessors | ratchet | FOR-1568, FOR-1611 | w2 | zip | 0 | 10 |
| [p14-t-campaign](#p14-t-campaign) | typed billing request, custody decision, meter restore | ratchet | FOR-1576, FOR-1587, FOR-1603, FOR-1601 | w2 | zip | 0 | 12 |
| [p14-t-historical](#p14-t-historical) | shared optimizer, resource-phase helper, workspace cleanup | ratchet | FOR-1570, FOR-1586, FOR-1606, FOR-1580 | w2 | zip | 0 | 14 |
| [p14-t-capture-sink](#p14-t-capture-sink) | MLflow startup split, extractor contracts | ratchet | FOR-1578, FOR-1569 | serial | **ship-it (live)** | 0 | 12 |
| [p14-t-core](#p14-t-core) | decode validator, offset sampler, horizon heads, GatedKappaNorm | ratchet | FOR-1593, FOR-1591, FOR-1592, FOR-1595, FOR-1597 | w3 | zip | 0 | 14 |
| [p14-t-dead](#p14-t-dead) | dead-code sweep | ratchet | FOR-1574, FOR-1590, FOR-1594, FOR-1600, FOR-1605 | w3 | zip | 0 | 21 |
| [p14-j-kernel-capture](#p14-j-kernel-capture) | kernel and capture codecs adopt strict_json | ratchet | FOR-1572, FOR-1577, FOR-1613 | w3 | zip | 0 | 23 |
| [p14-j-analyze-review-file](#p14-j-analyze-review-file) | analyze/review/file adopt strict_json; D1RunReader fails loud | ratchet | FOR-1604, FOR-1610, FOR-1572, FOR-1577, FOR-1579, FOR-1613 | w4 | zip | 0 | 32 |
| [p14-j-run](#p14-j-run) | ladder/run and d1_cli adopt strict_json | ratchet | FOR-1577, FOR-1582, FOR-1613 | w4 | zip | 0 | 41 |
| [p14-t-verdict-vocab](#p14-t-verdict-vocab) | verdict vocabularies as StrEnum, read-old window | tracer | FOR-1573 | serial | ship-it | 0 | 8 |
| [p14-z-identity](#p14-z-identity) | identity rail, one conductor re-pin | tracer | FOR-1589, FOR-1588, FOR-1596 | last | ship-it | 6 | 11 |

Touch-set sizes count `src` plus test files `[M, phase generator over the live tree]`. Every path
was verified to exist at eb93d5d except the rail's three new files. `gates.liveRequired` is `true`
only for the phase that touches `ladder/capture/adapters/mlflow.py`; every wave still publishes
through one `just ship-it` at its zip, which runs `just container-test`.

**Identity members, measured per phase.** The only phase whose touch set intersects the 51 reader
files is `p14-z-identity`, with six members: `composition/{cli,capture_cli,span_cli,source_cli}.py`,
`core/probe/score_gain.py`, `kernel/artifacts.py` `[M, comm of each touch set against the tuple
files]`. That makes it the identity rail. The ruled identity list names 13 files plus
`core/sentence/*`; the tuples carry 20 more members than it names, among them all six rail members
`[M]`. The briefs cite the tuples, never the ruled list.

---

## The mechanism rules that make phases disjoint

1. **Touch sets are derived, never recalled.** Each touch set is the grep- or AST-derived set of
   sites the phase's issues name, plus every consumer a signature change or deletion forces, plus
   the dedicated test modules. The generator that emitted the manifest checks existence, duplicate
   paths, pairwise overlap and identity membership.
2. **One owner per file inside a wave.** Zero files are shared inside any `parallel_group`
   `[M, overlap matrix]`. Two files cross waves and resolve serial-after by wave order:
   `kernel/verdict/d1_evaluation_values.py`, owned by `p14-t-trainer` and edited later by
   `p14-j-kernel-capture`; `tests/test_d_ladder_remote.py`, owned by `p14-t-historical` and edited
   later by `p14-t-dead`. A file shared by two phases of one wave is a HALT, never a note.
3. **Adoption is partitioned by package.** Each strict_json adoption phase owns one package set:
   `p14-j-register` owns `ladder/{register,gate}`, `p14-j-kernel-capture` owns `kernel` and
   `ladder/capture`, `p14-j-analyze-review-file` owns `ladder/{analyze,review,file}`, `p14-j-run`
   owns `ladder/run` and `composition/wiring/d1_cli.py`. The 36 primitive-defining files split
   2 / 8 / 7 / 17, with the identity-exempt `ladder/capture/capture.py` and `ladder/run/journal.py`
   in no phase `[M]`.
4. **Identity-bearing files belong to the rail only.** No lane but `p14-z-identity` may change the
   bytes of a tuple member, and that lane never edits `tests/test_identity_epoch.py`: the conductor
   re-pins once with `tools/regime_rows.py`. Identity-bearing JSON sites are exempt from extraction
   altogether, ruling 5.
5. **The strict_json API freezes at the rail.** `p14-r1-strict-json` owns it as a door; adoption
   phases consume it and never extend it. A missing primitive is a DECISION, not an edit.
6. **Extraction commits never change accepted input.** Each site keeps its own exception type and
   message through the `fail` callback. The two semantic drifts, trim and SHA case, are settled by
   the pre-phase rungs before any extraction lands.

---

<a id="p14-r1-strict-json"></a>

## p14-r1-strict-json

Create `kernel/codec/strict_json.py`: `strict_loads(raw, *, fail)` rejecting duplicate keys and
non-finite constants, `exact_keys(actual, required, optional)` returning `(missing, unknown)`,
`require_mapping(value, *, fail)`, `require_lowercase_sha256(value, *, fail)`, with
`fail: Callable[[str], NoReturn]`. Additive: no site adopts it here. Ruling 4 fixes the home.

<a id="p14-t-local"></a>

## p14-t-local

FOR-1585 moves the betas coercion into `TrainConfig.__post_init__` and restores `_build` to a
generic constructor. FOR-1602 validates through the `binding` property instead of a discarded
construction.

<a id="p14-j-register"></a>

## p14-j-register

The register trio and `gate/ladder_campaign.py` adopt `require_mapping`, `exact_keys` and
`strict_loads`. Each codec keeps its own `_text`, its own exception type and its own messages.
`d1_plan.py` adopts only the decoded-object primitives.

<a id="p14-t-composition"></a>

## p14-t-composition

FOR-1571 folds the six `if args.command` blocks into the existing `handlers` table. FOR-1583 types
the runtime and RunPod builders with the existing `D1RunPodPort` and one thin `D1R2Runtime`
Protocol. FOR-1582 and FOR-1584 point the three composition canonical dumps at `canonical_json`,
leaving `d1_cli.py:180` alone. FOR-1607 annotates the mailbox factory.

<a id="p14-t-trainer"></a>

## p14-t-trainer

FOR-1568 moves `_trainer_failure` verbatim into `d1_terminal_failure.py` and calls it from both
trainers. FOR-1611 shares the four byte-identical `state_moments` accessors of `D1EvalResult` and
`D1QualityEvidence` through one slotted mixin in `kernel/verdict/d1_evaluation_values.py`.

<a id="p14-t-historical"></a>

## p14-t-historical

FOR-1570 extracts the AdamW decay groups and the autocast selection into free functions beside
`cosine_lr`. FOR-1586 adds a local `_resource_phase` helper, never a null-object meter. FOR-1606
deletes `load_ladder_model`. FOR-1580 removes a partially materialized workspace on failure. The
phase also repoints the audited boundary SHA calls in `d_ladder_remote_request.py` and
`d_ladder_worker_command.py`, so it consumes the rail.

<a id="p14-t-campaign"></a>

## p14-t-campaign

FOR-1576 makes `D1BillingFinalizationRequest` the public contract of `finalize_d1_billing`.
FOR-1587 extracts only the custody decision. FOR-1603 types `restore` with all five keyword
parameters, including `max_power_gap_seconds`. FOR-1601 replaces the kwargs bag with one explicit
call.

<a id="p14-t-capture-sink"></a>

## p14-t-capture-sink

FOR-1578 splits MLflow startup into a create phase and a tag-and-params phase under one failure
policy, so `create_run` is never re-entered. FOR-1569 narrows each extractor's `extract` signature
and splits the Protocol, keeping `HTrainingStepExtractor`'s name and module.

<a id="p14-t-core"></a>

## p14-t-core

FOR-1593 deletes two dead aliases. FOR-1591 extracts the decode validator into
`core/observation/contracts.py`. FOR-1592 samples offsets once and dispatches devices once.
FOR-1595 adds one typed head accessor. FOR-1597 moves `GatedKappaNorm`, `KappaNormStats` and
`UnitNorm` into `core/substrate/blocks.py` and deletes `core/geometry/kappa_norm.py`.

<a id="p14-t-dead"></a>

## p14-t-dead

Deletes `d1_family_adapter.py`, `EncoderShape`, the `core/latent/targets.py` facade,
`d1_configs_dir` and the ACL lease `d1_output_access.py` with its test. FOR-1605 is RETIRE, ruling 10.

<a id="p14-j-kernel-capture"></a>

## p14-j-kernel-capture

Kernel verdict and evidence modules and the capture custody, ledger, records and resource codecs
adopt strict_json. `d1_parsing.object_exact` stays a thin D1 wrapper over `exact_keys`.
`kernel/artifacts.py` and `ladder/capture/capture.py` are exempt.

<a id="p14-j-analyze-review-file"></a>

## p14-j-analyze-review-file

The analyze, review and file boundary decoders adopt strict_json, routing every SHA check to
`require_lowercase_sha256`. FOR-1610 folds the binding and verification helpers. FOR-1579 deletes
the pre-filter so `D1TelemetryGap` propagates. Before that edit the lane re-runs the measured
precondition of ruling 11:

```text
grep -rnE 'D1RunReader|ladder_analysis' src/entropic_substrate/ladder/run/runner.py \
  src/entropic_substrate/ladder/run/worker_command.py \
  src/entropic_substrate/composition/wiring/h_*.py src/entropic_substrate/core/sentence/
# measured at eb93d5d (src == 651af30): no output, exit=1
```

`D1RunReader` has zero `src` constructors at the epoch; its only users are the three calls in
`tests/test_ladder_analysis.py` `[M, grep]`. `ladder_filing` and `ladder_predecessor` import
`ScientificRunReader`, a different class.

<a id="p14-j-run"></a>

## p14-j-run

The 17 `ladder/run` primitive files and `composition/wiring/d1_cli.py` adopt strict_json;
`d1_trainer_process_contracts.py` also carries FOR-1582's fourth canonical dump. The
identity-bearing `ladder/run/journal.py` is exempt.

<a id="p14-t-verdict-vocab"></a>

## p14-t-verdict-vocab

Ruling 9. Commit one audits every persisted D1 verdict artifact against the frozensets and records
the count. Commit two promotes the vocabularies to `StrEnum`, retypes the fields, adds
`__post_init__` rejection, and keeps a read-old window for one tested rollback window.

<a id="p14-z-identity"></a>

## p14-z-identity

FOR-1589 extracts the `dir_fd` descent with the full descriptor lifecycle owned inside the helper.
FOR-1588 adds one directory helper beside the two normalizers and uses it at all five sites.
FOR-1596 shares the score-gain summary arithmetic. The conductor re-pins both regime tables once.
Never land while an H run is in flight or pending resume: `checkpoint.py:304-305` refuses a
checkpoint whose reader digest differs.

---

## Guard teeth by mutation

House rule 5: break the guarded thing and show the test go RED, both arms in the PR body.

| phase | mutation | must go RED |
| --- | --- | --- |
| r1 | accept a duplicate key; accept `NaN`; accept uppercase hex; return from `fail` instead of raising | `tests/test_strict_json.py`, each arm |
| t-local | drop the `__post_init__` coercion | a YAML-list betas test expecting a float tuple |
| j-register, j-kernel-capture, j-analyze-review-file, j-run | re-add one local `def _pairs` in the owned packages | that phase's `tests/test_strict_json_adoption_<pkg>.py` AST guard |
| same four | wire one site's `fail` to a different exception type | the site's existing codec test asserting its exception type |
| t-composition | add a seventh `if args.command` branch in `main()` | an AST test counting dispatch branches in `main()` |
| t-trainer | change one payload field inside the shared failure helper | both the fresh and the resume failure tests |
| t-historical | delete the cleanup on failure | the RED-first partial-materialization test in `tests/test_d_ladder_remote.py` |
| t-campaign | drop `max_power_gap_seconds` from `restore` | a resume-with-power-registration test raising `TypeError` |
| t-capture-sink | re-enter `create_run` after a tag timeout | a fake-client test asserting exactly one `create_run` |
| t-capture-sink | make a non-cold `log_param` failure warn instead of raise | the startup-abort test expecting `MlflowStartupError` |
| t-core | skip the range check in the shared decode validator | both codec decode tests |
| t-dead | re-add any deleted symbol | an absence test over the deleted names |
| j-analyze-review-file | point one boundary site at `require_sha256_hex_any_case` | a mixed-case digest test at that site |
| j-analyze-review-file | restore the `(int, float)` pre-filter | a `D1CosineGap` test expecting `D1TelemetryGap` |
| t-verdict-vocab | construct `D1Contrast(decision="MAYBE", ...)` | the `__post_init__` rejection test |
| z-identity | fail `os.open` at the second descent step | a descriptor-accounting test: no leak, no double close |
| z-identity | type a regime-row digest instead of measuring it | `tests/test_identity_epoch.py` regime-row tests |

---

## The ratchet

Every phase ACCEPTANCE carries these, machine-checkable, at its tip:

1. **remap_check at or under the baseline.** Layer inversions ≤ 1 and package cycles ≤ 4, the
   p13 baseline in `docs/plans/measurements/p13-sweep-remap.txt`, which the epoch re-measured
   unchanged `[M]`. The mapping is re-derived at the tip with the procedure recorded in that file,
   never the committed `p13-sweep-mapping.json`, which already fails on one module added after p13
   `[M]`. The command pair is quoted in each brief.
2. **Mechanical counts at or under the epoch** for the six rows of the context table; each adoption
   phase shows a strict decrease in the primitive-file count for its own packages.
3. **Admissibility evidence** per abstraction introduced, as the rule above states.
4. **Behaviour tripwires unchanged**: identity fixtures, numerics tests, the CLI `--help` snapshot,
   and every caller-visible exception type and message not named by the brief.
5. **Guard teeth** by mutation, both arms, as tabled.
6. **Envelope**: `git diff --name-only <parent>...HEAD` is a subset of the brief's allowed files,
   and for every phase but the rail it has an empty intersection with the 51 reader files.

---

## Sequence

```text
pre: fix/for-1567 -> fix/for-1617 -> chore/for-1614 -> chore/for-1615   (each onto the trunk)
p14-r1-strict-json
w1  [ t-local | t-composition | j-register ]
w2  [ t-trainer | t-campaign | t-historical ]
p14-t-capture-sink                      (serial, live gate)
w3  [ t-core | t-dead | j-kernel-capture ]
w4  [ j-analyze-review-file | j-run ]
p14-t-verdict-vocab                     (serial tracer, door)
p14-z-identity                          (last, one conductor re-pin)
then: the Lean tail spike
```

Waves launch through `mass-patch-fanout` with `K ≤ 3`, parallel commit hooks serialized per the
2026-09-23 measurement. Inside a wave the zip restacks by ascending footprint and re-runs the exit
gate at every restacked tip; a zip conflict falsifies the wave. Serial phases launch the ported
cascade directly.

**Args budget.** One assembled serial cascade args string, for the longest-titled phase
`p14-j-analyze-review-file` with its brief-as-file scope, measures 807 characters `[M]`. The fanout
invocations measure 1169, 1190 and 1158 characters for w1, w2 and w3 with titles, over budget, and
944, 947 and 936 with bare phase ids as titles, under the 950 fanout ceiling; w4 measures 863 with
titles `[M]`. So w1 to w3 launch with bare-id titles.

---

## Rulings

**Operator rulings 2026-09-28, binding.**

| # | ruling |
| --- | --- |
| 1 | Trunk `refactor/p14-consolidation` ratified; Graphite trunk added by the conductor; pushed only after tag `v0.1.0` exists. |
| 2 | Green baseline: a full `just check` at the epoch with no other lanes active, SHA pinned by the conductor before launch. |
| 3 | FOR-1612 narrowed to package-level hygiene and held out of every phase; a dependabot and package-hygiene review follows the consolidation phases. |
| 4 | strict_json home is `kernel/codec/strict_json.py` with a `fail` callback, under the FOR-1613 umbrella. |
| 5 | Identity-bearing JSON sites, `ladder/capture/capture.py`, `ladder/run/journal.py` and `kernel/artifacts.py` `object_fields`, are exempt from extraction in p14. |
| 6 | One identity rail, landed last, one conductor re-pin. The phase is chosen by measured tuple membership, which selects `p14-z-identity`. |
| 7 | FOR-1567 and FOR-1617 are `fix/` branches stacked on the trunk before the phases, each with its own conductor re-pin; they are not manifest phases. |
| 8 | Any h-ladder flywheel experimental run needs a fresh operator green light; a prerequisite, never a phase. |
| 9 | FOR-1573: audit, then enforce with a read-old window. |
| 10 | FOR-1605: RETIRE, delete. |
| 11 | FOR-1579: fail-loud, after a measured grep shows `D1RunReader` on no h-ladder flywheel path; recorded above and in the brief. |
| 12 | FOR-1614 keeps per-site semantics now and becomes strict-trimmed after a persisted-record audit; FOR-1615 is lowercase-strict at every boundary with the loose kernel function renamed; both as `chore/` branches before the phases. |
| 13 | FOR-1598 out as a verification spike; FOR-1609 out, a `scripts/` split is not consolidation; FOR-1608 closed against FOR-1571. |
| 14 | The admissibility rule above. |
| 15 | Touch sets re-derived from the live tree; no path carried without verification. |

Also settled before the interview: FOR-1581 CANCELED as a byte-identical duplicate of FOR-1578.

**Planning-lane decisions 2026-09-28, pending ratification with the manifest.**

| id | decision | why |
| --- | --- | --- |
| P14-SHA-SEQ | FOR-1615 splits into the chore's rename plus audit, the rail's `require_lowercase_sha256`, and per-package repointing inside the four adoption phases | keeps one owner for the frozen strict_json API and removes the name collision instead of moving it |
| P14-1579-HOME | FOR-1579 rides `p14-j-analyze-review-file` | same slice, and `ladder_analysis.py` is also an audited SHA caller there, so one file keeps one owner |
| P14-LIVE | `p14-z-identity` has `liveRequired: false`, a deviation from the draft and the p4 precedent | task rule: only the phase touching `mlflow.py` is live; the rail still publishes through `just ship-it` |
| P14-1611-HOME | the FOR-1611 accessor mixin lives in `kernel/verdict/d1_evaluation_values.py` | both owning modules already import it; two concrete sites, so admissible |
| P14-1597-HOME | `GatedKappaNorm` and its companions move into `core/substrate/blocks.py`; `kappa_norm.py` is deleted | the issue's remedy; `build_norm` already owns the norm variants |
| P14-LEAN | the Lean tail runs after all phases, not inside a register/gate phase as the play's section 8a suggested | the task places it after the phases; recorded as a dated deviation |
| P14-ARCHIVE | the p0-p13 manifest is kept on disk as `sprint-manifest.hexagonal-refactor.json` | ruling O-2 deleted an absorbed manifest; this one was completed, so deletion is the operator's call |

---

**Ratified 2026-09-28 (operator, live, then autonomous mode until the morning):** the manifest as written at epoch 4e0ae6a, all planning-lane decisions (P14-SHA-SEQ, P14-1579-HOME, P14-LIVE, P14-1611-HOME, P14-1597-HOME, P14-LEAN, P14-ARCHIVE) as recommended; DN-ARCHIVE = delete the archived p0-p13 manifest (this commit); DN-LEDGER = rotate the gitignored launch queue (done by the conductor, `launch-queue.hexagonal-refactor.jsonl`); DN-LEAN-BOX set at that launch; DN-1614-READ-OLD deferred to the audit. Goal: complete the 15 phases end to end, tracer phases serial with the design gate, ratchet waves autonomous; notify only on DECISION-NEEDED / BLOCKED.

## DECISION-NEEDED register

| id | question |
| --- | --- |
| DN-ARCHIVE | Delete `sprint-manifest.hexagonal-refactor.json` from disk as O-2 did, or keep it. |
| DN-LEDGER | Rotate the main checkout's `launch-queue.jsonl` to `launch-queue.hexagonal-refactor.jsonl` at prelaunch; the p14 rows were appended to the live file because other lanes were still writing it. |
| DN-LEAN-BOX | The Lean tail timebox, set by the operator at launch. |
| DN-1614-READ-OLD | Only if the FOR-1614 audit finds untrimmed persisted records: the read-old window for them. |
| DN-1632-SEAM (2026-10-06) | FOR-1632: `closure/d1_runpod_billing_parser.py` keeps its local hooks because the strict-JSON door has no `parse_float` seam (billed amounts are `Decimal`); ruling W4-21 retained it and left the acceptance count at "falls by 16". Amend the door with a `parse_float` seam, or keep the one holdout. **RULED 2026-10-08 (operator): amend the door with the `parse_float` seam; the holdout is not retained.** FOR-1632 -> Todo with the scope; one small serial lifecycle off the merged trunk. |
| DN-1626-REPOINT (2026-10-05/06) | FOR-1626: three BOUNDARY SHA sites stay on `require_sha256_hex_any_case` by rulings W1-16, W3-21 and W4-26 (`d1_registered_design` blob_sha256, `d1_primitives` encoder-initialization digest, `d1_trainer_object_transport` attempt/request digests); the repoint changes a caller-visible exception type at two of them. Repoint with the tests restated, or keep. **RULED 2026-10-08 (operator): repoint all three sites with their tests restated**, exception-type changes pinned, audit rows 39/71/93-94 removed and the header restated 33/57/57/0 (conductor rulings R1-R6, 2026-10-08; the earlier wording "-> `repointed`" is retracted). FOR-1626 reopened (it was Done while row 71 was still retained); one small serial lifecycle off the merged trunk. |
| DN-1633-HOOK (2026-10-07) | FOR-1633: the commit-msg gitlint hook blocked over-length messages from an interactive shell but let four agent commits through this sprint (mechanism not isolated; repro recipe on the ticket). A `.pre-commit-config.yaml` or hook-install change is operator-ruled. **RULED 2026-10-08 (operator): repair the root cause.** Root cause MEASURED 2026-10-08 by the conductor: the global `exclude` in `.pre-commit-config.yaml` began with `\.git/`; prek applies it to the commit-msg stage's only file, the path git hands the hook, which is `.git/COMMIT_EDITMSG` when `git commit` starts at the repository root (gitdir relative) and an absolute path from a subdirectory or a linked worktree (gitdir absolute). From the root both commitizen and gitlint reported `(no files to check) Skipped` (reproduced with the installed hook on `.git/COMMIT_EDITMSG`: rc=0; the same file by an absolute path: gitlint FAILED rc=1); the four fail-open commits were all made from the root, the one agent commit whose hooks ran was made from a shell that had drifted into a subdirectory. Fix: drop the `\.git/` entry (git never stages anything under `.git/`, so it was inert for every other stage), pinned RED-first by `tests/test_commit_msg_hook_sees_message.py`; published as its own PR. The interim per-stage `gitlint --msg-filename` clause stays harmless. |
| DN-1573-DVC (2026-10-07) | FOR-1573's persisted-verdict audit is PARTIAL: `results/captured.dvc` (30 of 76 files), `results/h-image-publication-3b93854.dvc` and `results/field-pretrain-s42.jsonl.dvc` were never fetched. Authorize `dvc pull` of the three pointers and a re-run of the quoted audit script, or accept the zero over the scanned set. **RULED 2026-10-08 (operator): `dvc pull` authorized, pulled bytes removed from local storage once recorded.** Done 2026-10-08: pull 14:34:32-14:34:44Z (38 fetched + 43 added), audit re-run byte-identical at `da4d35c` 14:35:39-14:35:50Z: 0 artifacts / 0 out-of-vocabulary / 0 unreadable over the now-COMPLETE scan set, control still detects; completion addendum appended to `docs/plans/measurements/p14-verdict-vocab-audit.txt`; FOR-1573 stays Done, the audit is no longer PARTIAL. |

**Post-merge state (2026-10-08):** the operator merged the thirteen drafts `--no-ff` in one sweep
(#320 `5b42dd4` ... #332 `da4d35c`); trunk `refactor/p14-consolidation` = `da4d35c`. Linear: the 25
p14 In Review issues and the FOR-1613 umbrella are Done; FOR-1632 and FOR-1626 are Todo with their
rulings; FOR-1633 carries the root cause. The H image must be rebuilt from the merged tip before any
future green-lit run (no pods, no spend, no flywheel run without a fresh go-ahead and the balance
said aloud); the first measured publication block of that run gates FOR-1623.

**Post-merge state (2026-10-12):** the four follow-ups landed on `main` as #333 `fc31214`, #334
`008c179`, #335 `58e52a3`, #336 `ecc6e88`, each a single-parent commit: the ruled `--no-ff` sweep did
not happen, so the certified tips (`ee9df9a`, `a6cac92`, `ee2c13a`, `f3521f0`) and the DN-8DB6295-MSG
reword `5c521f7` are not in `main` history; the landed trees are identical to the certified trees.
The trunk reached `main` as merge `3ca01bd`; release-please cut v0.2.0 at `65542d4`. FOR-1633,
FOR-1632, FOR-1626 and FOR-1636 are Done (2026-10-12); the 2026-10-08 sentence "FOR-1632 and
FOR-1626 are Todo" is superseded. The H image was rebuilt from `65542d4` (PR #338, amd64 manifest
`762ed810...`, capture `results/h-image-publication-65542d4.dvc`). Planned-versus-actual
reconciliation: `docs/plans/refactor-retrospective-2026-10-12.md`.

---

## Lean tail: timeboxed exploratory spike after the phases

Class 6 exploratory spike, per the play of 2026-09-23. It runs only after `p14-z-identity` lands,
with a timebox the operator sets at launch, and it ends in one of three honest exits: succeeded,
failed, or exhausted-pending. Kill at the timebox, never extend. No pods, no GPU, no spend.

1. **Verified lifecycle state machine.** A Lean inductive `RungState` and `Step` relation for the
   ten happy and seven problem states of `docs/architecture.md` "Lifecycle state machine", with
   theorems for the "never" rules of `docs/rung-lifecycle-protocol.md`. A Python golden test in
   `ladder.gate` asserts the Python transition table equals the Lean-exported table; the ladder
   refuses transitions absent from it. Guard teeth: mutate one Python edge and the golden test
   goes RED.
2. **Math before code, items 1 and 2 only.** The isotropic reduction theorem, then density
   correctness of the directional plus log-magnitude law. Then STOP and file what was learned as a
   K0 entry and a verdict-style note under `docs/` before any ruling on the Fisher pullback or the
   retained E-series claims.
3. **Rules.** A proof that contradicts the plan text is a DECISION-NEEDED on the plan, never a
   silent code edit. Code adjusts to a proof only after the proof is filed and the plan amended.

---

## Out of scope

| ticket | reason |
| --- | --- |
| FOR-1612 | held out; package-hygiene review after the phases, ruling 3 |
| FOR-1598 | verification spike, ruling 13 |
| FOR-1609 | `scripts/` split, not consolidation, ruling 13 |
| FOR-1608 | closed against FOR-1571, ruling 13 |
| FOR-1581 | CANCELED duplicate of FOR-1578 |
| FOR-1614, FOR-1615, FOR-1567, FOR-1617 | pre-phase rungs, rulings 7 and 12 |
