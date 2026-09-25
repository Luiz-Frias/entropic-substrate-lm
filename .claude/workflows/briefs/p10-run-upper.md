# p10-run-upper — `ladder/run` upper stack (campaign, closure, ckptsync, D1 worker)

Authoritative brief for one `td-phase-lifecycle.cascade` lifecycle. v4, drafted 2026-09-24,
superseding v3 (same date) after verifier round 2 returned FAIL on one blocker: v3's
`P10-WORKER-SPLIT` ruling over-read design-gate ruling D-E and landed `build_worker_factory`
alongside `worker_resource_registration`, which would silently disarm an existing unit test's
monkeypatch (`tests/test_d_ladder_worker_runtime.py`). Corrected below, scope narrowed to the one
function D-E actually names for p10. v3 in turn superseded v2 (same date) after verifier round 1
returned FAIL on three blockers and one major finding — those corrections stand, re-confirmed at the
same epoch, and are not re-derived here. Plan: `docs/plans/src-hexagonal-refactor.plan.md` (§p10,
conductor-rulings table, now including `W2-ADMISSIBLE-DELTAS`). The lifecycle's `acceptance` arg
reads "Per the brief's ACCEPTANCE section"; that section is below.

**Epoch.** Every `[MEASURED]` count below was taken at `e5aa9d33fd4100b69fd170ce94f69bd1719a0aa6`
(`docs(w2): post-wave docs sweep, retractions and ratification stamps`), read-only via
`git show e5aa9d3:<path>` / `git grep <pat> e5aa9d3 -- <paths>` / `git ls-tree`. **RETRACTED (house
rule 4):** v2 measured at `12f9f4a` and described `e5aa9d3` as "riding on top, touching no `src/`" —
false. `[MEASURED]`: `git diff --stat 12f9f4a e5aa9d3 -- src tools tests` shows 8 files changed, 26
insertions / 25 deletions. Every changed file was read in full: `ladder/run/trainer/d1_calibration.py`,
`d1_training_step.py` and `tools/gen_shims.py` gained DOCSTRING-only corrections (retroactively
fixing "still under `__legacy/`" comments now that p8/p9 landed those record/torch halves);
`tests/test_d1_external_worker.py` gained a docstring provenance note; the other four
(`ladder/capture/adapters/{composite,jsonl,mlflow}.py`, `ladder/capture/ports/metric_sink.py`) were
spot-checked and are the same shape. **None touches any of p10's 27 targets, its 22 importer files,
`checkpoint.py`, `capture.py`, `tools/mapping.json`, or any landed-set fragment** — every number this
draft carries forward from v2's re-derivation is re-confirmed independently at `e5aa9d3`, not merely
copied. The **launch `startTip`** is the actual branch tip at launch — currently `e5aa9d3` per the
checkout's own status, but the lane always re-reads it, never a typed SHA baked into this file.

**Evidence classes.** `[MEASURED]`: command output at `e5aa9d3`. `[predicted]`: arithmetic on
measured inputs, never typed into a reproduction path. Never commingled in one row without a label.

## Branch

| field                                        | value                                                                                                                                                                                                     |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p10-run-upper`                                                                                                                                                                                  |
| parent                                       | `mythwave/w2-reconcile` @ `<startTip>` (**P10-PARENT**, conductor 2026-09-24). p10 consumes p9's `ladder/run/{values,transport}` at their real, post-reconciliation addresses, never through a forwarder. |
| mode (sprint manifest, parallel eligibility) | `ratchet`                                                                                                                                                                                                 |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived; feeding the manifest `mode` here yields a silent `normal` and no Opus design gate.                                                                                |
| `engine_mode`                                | `arch`                                                                                                                                                                                                    |
| wave                                         | `serial`, post-zip, post-rung. Not a wave lane: no lane worktree, no zip, no `publishMode:'defer'`.                                                                                                       |
| `gates.liveRequired`                         | `true`                                                                                                                                                                                                    |
| live gate                                    | `just ship-it` runs INSIDE the lifecycle (house rule 1). FOR-1541 branch-only fallback as ruled for p5..p7 and used by `w2-reconcile`.                                                                    |
| gates (lifecycle arg)                        | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}`                                                                                                           |
| launch                                       | SERIAL phase: the conductor launches the repo's ported cascade DIRECTLY. Not a wave member.                                                                                                               |
| `repoToplevel`                               | `/Users/luizfrias/CursorAI/entropic-substrate-lm`. The MAIN checkout, never a worktree. Launch only when quiescent: no publish or pre-push hook running, `git status --porcelain` empty.                  |
| trunk                                        | `refactor/hexagonal-redesign`                                                                                                                                                                             |

## Contract surface owned

| contract                                                                                                                                                                                                                                                                            | kind     | frozen_after |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `ladder/run/{campaign,closure,ckptsync}` (three NEW sub-packages), the D1 leaf added to the EXISTING `ladder/run/worker/` package (`d1_predecessor_loader`), and the SPLIT remainder of `d_ladder_worker_runtime` that D-E assigned to `ladder.run.worker` (see *P10-WORKER-SPLIT*) | contract | p13          |

p10 owns none of the fail-closed allowlist literals; see *P10-ALLOWLISTS ruling* under *Scope*.

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item
below is LANDED before this lane launches.

- `ladder/run/{values,transport}` (p9, real addresses post-reconciliation — **P10-PARENT**)
- the H half of `ladder/run/worker` (p4): `h_probe_allocation.py`, `h_probe_command.py`,
  `h_probe_controller.py`, `hardware.py` (`measure_hardware`, `verify_source_tree`,
  `SOURCE_PREFIXES`), plus the package's empty `__init__.py` `[MEASURED]`
- the frozen `h-checkpoint-reader-v2` reader identity, 51 members,
  `src/entropic_substrate/ladder/run/checkpoint.py:65-117` `[MEASURED]` (`ladder.run.worker_command`
  / `worker_completion` / `worker_execution` at lines 111/112/113)
- `evidence.resource_provenance` (p9 → `kernel.evidence_class.resource_provenance`) — LANDED,
  consumed by the worker-split leaf (`ResourceActivationMode`, `ResourceActivationRegistration`; see
  *P10-WORKER-SPLIT*). **RETRACTED (house rule 4):** v3 additionally cited
  `evidence.resource_meter_runtime` (p8), `d1_resource_activation` (p9) and `experiments.d1_primitives`
  (p9) as consumed here — those three were needed only by `build_worker_factory`, which v3 moved and
  this brief does not; `build_worker_factory` and its imports stay in the `__legacy` remainder,
  outside p10's envelope.
- `tools/gen_shims.py` (+ `--verify`, `--preflight`), `tools/import_graph.py`, `tools/regime_rows.py`
- `tests/test_legacy_coupling.py`, `tests/test_no_old_prefix.py`, `tests/test_module_identity.py`,
  `tests/test_identity_epoch.py`, `tests/test_regime_rows.py`, `tests/test_cli_surface.py` (READ-ONLY
  consumed control — see *Guard teeth*, this phase edits nothing in it)
- `.claude/workflows/landed-set/p11-analyze-review-file.json`, `legacy_edge_budget: 32` `[MEASURED]`
  at `e5aa9d3` (re-confirmed unchanged from the `12f9f4a` value; the fragment's `note` field is the
  only byte that moved between the two epochs, per `git diff 12f9f4a e5aa9d3 -- .claude/workflows/landed-set`)

## Sacred invariants

Quoted verbatim from `CLAUDE.md`. `touch_set ∩ core/** = ∅` for every file this phase edits.

| id   | verbatim invariant                                                                                                          | tripwire / guard                                                                                       |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | `git diff <startTip> -- src/entropic_substrate/core tests/test_kappa_autocast_island.py` must be EMPTY |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | same empty diff over `core/geometry`                                                                   |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | same empty diff over `core/substrate`                                                                  |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | same empty diff over `core/sentence`                                                                   |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | same empty diff over `core/latent`                                                                     |

House rules HR-1..HR-7 bind exactly as tabled in `.claude/workflows/briefs/p9-run-lower.md`.

## Scope

**P10-SET (conductor 2026-09-24).** The full-module target set is **27 modules**: the 26 quarantined
`tools/mapping.json` rows in the campaign(15)/closure(6)/ckptsync(5) categories, PLUS
`d1_predecessor_loader` as the D1 half of `ladder/run/worker`. **PLUS**, per **P10-WORKER-SPLIT**
below, a PARTIAL consumption of a 28th quarantined module (`d_ladder_worker_runtime`) — ONE
function, `worker_resource_registration`, corrected down from v3's two (see *P10-WORKER-SPLIT*). The
27 full moves are unchanged from v2/v3 and are re-confirmed at `e5aa9d3`.

**NOT p10's, each with its measured reason:**

- `entropic_substrate.predictive_state` facade (1-line `__init__.py`) — split sends the runtime half
  to `construction/__init__.py` (p9's, landed) and the domain half to `core/latent/__init__.py`
  (core layer). Not p10's at any point.
- `provider_launch`'s and `d1_runpod_adapter`'s allowlist literals — see *P10-ALLOWLISTS ruling*.
- `build_worker_factory`, `verify_worker_source`, and the rest of `d_ladder_worker_runtime`'s
  remainder after p10's (narrowed) split consumption — see *P10-WORKER-SPLIT*; both are p12's.

### The 27 full-module moves, paths and line counts `[MEASURED]` (`git show e5aa9d3:<path> | wc -l`

re-run at the new epoch; every value unchanged from v2's `12f9f4a` count, confirming the docs-sweep
commit touched none of these files)

| #  | sub-package (new dest)                          | legacy module                          | path under `src/entropic_substrate/__legacy/` | lines |
| -- | ----------------------------------------------- | -------------------------------------- | --------------------------------------------- | ----: |
| 1  | `ladder.run.campaign`                           | `d1_campaign_bootstrap`                | `d1_campaign_bootstrap.py`                    |   253 |
| 2  | `ladder.run.campaign`                           | `d1_campaign_capture`                  | `d1_campaign_capture.py`                      |   378 |
| 3  | `ladder.run.campaign`                           | `d1_campaign_checkpoint`               | `d1_campaign_checkpoint.py`                   |   420 |
| 4  | `ladder.run.campaign`                           | `d1_campaign_checkpoint_repository`    | `d1_campaign_checkpoint_repository.py`        |   251 |
| 5  | `ladder.run.campaign`                           | `d1_campaign_evidence_assembler`       | `d1_campaign_evidence_assembler.py`           |   166 |
| 6  | `ladder.run.campaign`                           | `d1_campaign_queue`                    | `d1_campaign_queue.py`                        |   275 |
| 7  | `ladder.run.campaign`                           | `d1_campaign_recovery`                 | `d1_campaign_recovery.py`                     |   284 |
| 8  | `ladder.run.campaign`                           | `d1_campaign_recovery_queue`           | `d1_campaign_recovery_queue.py`               |   245 |
| 9  | `ladder.run.campaign`                           | `d1_campaign_request`                  | `d1_campaign_request.py`                      |   363 |
| 10 | `ladder.run.campaign`                           | `d1_excluded_operator`                 | `d1_excluded_operator.py`                     |   291 |
| 11 | `ladder.run.campaign`                           | `d1_operator_controller`               | `d1_operator_controller.py`                   |   712 |
| 12 | `ladder.run.campaign`                           | `d1_operator_custody_adapter`          | `d1_operator_custody_adapter.py`              |   122 |
| 13 | `ladder.run.campaign`                           | `d1_operator_evidence`                 | `d1_operator_evidence.py`                     |   489 |
| 14 | `ladder.run.campaign`                           | `d1_operator_intent_codec`             | `d1_operator_intent_codec.py`                 |   267 |
| 15 | `ladder.run.campaign`                           | `d1_operator_values`                   | `d1_operator_values.py`                       |   394 |
| 16 | `ladder.run.closure`                            | `d1_billing_finalization_values`       | `d1_billing_finalization_values.py`           |   267 |
| 17 | `ladder.run.closure`                            | `d1_billing_finalizer`                 | `d1_billing_finalizer.py`                     |   228 |
| 18 | `ladder.run.closure`                            | `d1_excluded_capture`                  | `d1_excluded_capture.py`                      |   360 |
| 19 | `ladder.run.closure`                            | `d1_finalized_cell_custody`            | `d1_finalized_cell_custody.py`                |   551 |
| 20 | `ladder.run.closure`                            | `d1_finalized_cell_custody_repository` | `d1_finalized_cell_custody_repository.py`     |   141 |
| 21 | `ladder.run.closure`                            | `d1_runpod_billing_parser`             | `d1_runpod_billing_parser.py`                 |   160 |
| 22 | `ladder.run.ckptsync`                           | `d1_capture_finalize_coordinator`      | `d1_capture_finalize_coordinator.py`          |   119 |
| 23 | `ladder.run.ckptsync`                           | `d1_capture_finalize_request`          | `d1_capture_finalize_request.py`              |   145 |
| 24 | `ladder.run.ckptsync`                           | `d1_capture_prepare_coordinator`       | `d1_capture_prepare_coordinator.py`           |   108 |
| 25 | `ladder.run.ckptsync`                           | `d1_checkpoint_dvc_recovery`           | `d1_checkpoint_dvc_recovery.py`               |   285 |
| 26 | `ladder.run.ckptsync`                           | `d1_checkpoint_dvc_sync`               | `d1_checkpoint_dvc_sync.py`                   |   359 |
| 27 | `ladder.run.worker` (D1 leaf, existing package) | `d1_predecessor_loader`                | `d1_predecessor_loader.py`                    |   220 |

Totals: campaign 15/4910, closure 6/1707, ckptsync 5/1016, worker D1 1/220. **27 modules, 7853
lines**, plus the partial split below.

**Six-way collision check, `[MEASURED]` at `e5aa9d3`:** none of `ladder/run/{campaign,closure,ckptsync}.py`
exists as a file; `ladder/run/worker/` exists with 4 leaves + empty `__init__.py`
(unchanged from `12f9f4a`); the (narrowed, v4) split-leaf name `ladder/run/worker/resource_registration.py`
is also free `[MEASURED]`: `git cat-file -e e5aa9d3:src/entropic_substrate/ladder/run/worker/resource_registration.py`
fails, "does not exist".

## P10-WORKER-SPLIT ruling

**BLOCKER, corrected AGAIN (house rule 4 — v3's split is RETRACTED as over-broad; it moved
`build_worker_factory` alongside `worker_resource_registration`, which the D-E ruling's own text
forbids for a MEASURED reason v3 never read to the end).** `[MEASURED]`
`git show e5aa9d3:src/entropic_substrate/__legacy/d_ladder_worker_runtime.py` (105 lines): the
module's docstring records design-gate RULING D-E (conductor 2026-09-22): only `measure_hardware`,
`verify_source_tree` and `SOURCE_PREFIXES` moved to `ladder.run.worker.hardware` at p4. Three
functions remain in the `__legacy` file today: `worker_resource_registration` (lines 50-54),
`build_worker_factory` (lines 57-71), and `verify_worker_source` (lines 74-105).

**RETRACTION, one line (house rule 4).** v3 read `tools/mapping.json`'s disclaimed survey-era note
("measure_hardware/verify_source_tree/build_worker_factory/worker_resource_registration ->
ladder.run.worker") as agreeing with ruling D-E and assigned BOTH remaining functions to p10.
`[MEASURED]`, full text of `.claude/workflows/briefs/p4-h-tracer.md:106`, read to its end this time:
*"`build_worker_factory`, `worker_resource_registration` and `verify_worker_source` STAY in the
`__legacy` remainder, which keeps re-exporting the two moved names so
`__legacy/d_ladder_worker_command.py:41` and its tests resolve unchanged. The remainder is p12's
(with p10 consuming `worker_resource_registration` per its brief). MEASURED reason: moving
`build_worker_factory` would silently disarm `tests/test_d_ladder_worker_runtime.py`'s monkeypatch
of `build_torch_cuda_adapters` (module-globals binding; the guard would pass for the wrong reason on
CPU). The four-name enumeration in the mapping.json row note is a survey-era note, not a door."* The
ruling names exactly ONE p10 carve-out — `worker_resource_registration` — and explicitly, by name,
keeps `build_worker_factory` in the p12 remainder together with `verify_worker_source`, WITH a
measured mechanism for why. v3's "both sources agree" reasoning used the source the ruling itself
calls non-authoritative to override the ruling's own explicit, reasoned text. Struck.

**The MEASURED reason, independently re-verified at `e5aa9d3`.**
`git show e5aa9d3:tests/test_d_ladder_worker_runtime.py:29-36`:
`test_resource_registration_requires_the_existing_cuda_adapters` does
`import entropic_substrate.d_ladder_worker_runtime as module`,
`monkeypatch.setattr(module, "build_torch_cuda_adapters", lambda **_: (None, None))`, then calls
`build_worker_factory(worker_resource_registration())` and asserts a `ValueError`. The root-level
`entropic_substrate.d_ladder_worker_runtime` `[MEASURED]` is a `sys.modules[__name__] = _target`
shim to `entropic_substrate.__legacy.d_ladder_worker_runtime`, so `module` IS the `__legacy`
remainder module object. A Python function resolves free names (`build_torch_cuda_adapters`) from
its OWN defining module's globals at call time — so if `build_worker_factory` were DEFINED in a new
landed leaf (as v3 planned), the test's patch of the `__legacy` remainder's attribute would have
ZERO effect on the real call, and — because v3's own remainder-edit spec never re-imported
`build_torch_cuda_adapters` back into the remainder (nothing left there would need it) —
`monkeypatch.setattr(module, "build_torch_cuda_adapters", ...)` with its default `raising=True`
would raise `AttributeError` outright, an existing-test breakage v3 named nowhere in its Guard teeth,
ACCEPTANCE, or GAPS, and was independently forbidden from fixing by its own Non-goals ("Does NOT
edit any EXISTING test other than the identity mirrors named under R-PIN"). Both halves of the
hazard are real; this brief avoids it by not moving `build_worker_factory` at all.

**Fix: p10's worker commit lands `worker_resource_registration` ALONE**, as a new leaf,
`src/entropic_substrate/ladder/run/worker/resource_registration.py`. D-E's own text names no landed
file for it (only the destination PACKAGE, `ladder.run.worker`); this brief follows the conductor's
fallback naming (`resource_registration.py`) rather than reusing `d_ladder_worker_runtime.py`, which
would misname a leaf now carrying one function, not "the runtime". MOVED, verbatim: lines 50-54 of
the `__legacy` file, 5 lines — the function body only. `_PROVIDER` and `_FACTORY_DIGEST` (lines
38-47) STAY in the remainder, because `build_worker_factory` (staying) also needs them; the new leaf
imports them BACK from the remainder by its explicit `__legacy` address (an ordinary
landed-depends-on-quarantined edge, the same shape used everywhere else in this refactor for a
not-yet-landed dependency, just crossing the boundary in the less common direction — a landed leaf
consuming two names the convention would normally call "private" by their underscore prefix; flagged
here as a design point worth a second look against the project's lint config, not resolved by this
drafting session). The new leaf's two OTHER names (`ResourceActivationMode`,
`ResourceActivationRegistration`) come from the ALREADY-LANDED `evidence.resource_provenance` →
`kernel.evidence_class.resource_provenance` (confirmed under *Contracts consumed*). `[predicted]`
reconstructed content, since this drafting session writes no file:

```python
"""Worker CUDA-resource registration, split from d_ladder_worker_runtime (RULING D-E)."""

# RETRACTION (docs sweep 2026-09-24): the landed leaf imports these two names INSIDE the
# function body (see the cycle note in resource_registration.py), not at module level as
# this [predicted] block shows; and no .pyi stub was owed for the new leaf (it had no
# prior old-address shim), so the Allowed-files '(+ .pyi, NEW ...)' mention is retracted.
from entropic_substrate.__legacy.d_ladder_worker_runtime import _FACTORY_DIGEST, _PROVIDER
from entropic_substrate.kernel.evidence_class.resource_provenance import (
    ResourceActivationMode,
    ResourceActivationRegistration,
)


def worker_resource_registration() -> ResourceActivationRegistration:
    """This tracer measures CUDA duration/memory; FLOPs/power remain explicit gaps."""
    return ResourceActivationRegistration(
        _FACTORY_DIGEST, ResourceActivationMode.PRODUCTION, _PROVIDER, "cuda:0", None, None
    )
```

`[predicted]` **~13 lines total** (5 moved verbatim + a module docstring, blank lines and 2 new
import statements the original file never needed to write out separately). This REPLACES v3's
34-line, four-import, two-function leaf.

**The remainder file (`__legacy/d_ladder_worker_runtime.py`) keeps `build_worker_factory` AND
`verify_worker_source` and their bodies byte-identical.** Its ONLY edits: (a) remove
`worker_resource_registration`'s 5-line body; (b) its `evidence.resource_provenance` import narrows
from `(ResourceActivationMode, ResourceActivationRegistration)` to `(ResourceActivationRegistration)`
alone — `ResourceActivationMode` is unused in the remainder once the one function that used it
leaves (`build_worker_factory` never references it; only `ResourceActivationRegistration`, in its own
type signature); (c) gain ONE new re-export line,
`from entropic_substrate.ladder.run.worker.resource_registration import
worker_resource_registration as worker_resource_registration`, matching the exact re-export form
already used for `measure_hardware`/`verify_source_tree`/`SOURCE_PREFIXES` two paragraphs above it in
the same file — the p4/p5b remainder re-export convention ("a split's quarantined remainder
re-exports the moved names so old-dotted-name consumers keep working through the one-target shim").
`_PROVIDER`/`_FACTORY_DIGEST` and the `experiments.d1_primitives` import that builds them stay
untouched, since `build_worker_factory` still needs both. `d_ladder_worker_runtime` stays
`action: split`, `landed: false` in its own mapping row forever (or until p12 empties it); the row is
NOT part of p10's 27-item fragment `modules` map — p10's fragment instead records a
`split_consumption` note naming the ONE function and the destination file.

**Importers, both sides, `[MEASURED]`** (`git grep -n "d_ladder_worker_runtime" e5aa9d3 -- src tests`,
excluding the file's own body and generated `as _target` shim lines):

- `src/entropic_substrate/__legacy/d_ladder_worker_command.py:41` — imports
  `worker_resource_registration` via the flat pre-quarantine name. Still quarantined itself
  (legacy-internal import, not landed); keeps resolving through the root-level shim + the
  remainder's new re-export, no rewrite owed by p10.
- `src/entropic_substrate/ladder/run/worker/hardware.py:3` — a DOCSTRING mention of the split
  ruling, not an `import` statement. Not a rewrite target.
- `tests/test_d_ladder_worker_command.py:15`, `tests/test_d_ladder_worker_runtime.py:8,32`,
  `tests/test_h_probe_bootstrap.py:226`, `tests/test_identity_epoch.py:37,43` (docstring only),
  `tests/test_worker_source_anchor.py:2,44,76,145,276` — all TEST files. No phase before p13 edits an
  existing test; all keep resolving via the regenerated root-level shim AND the remainder's own
  unmoved `build_worker_factory`/`verify_worker_source` bodies (the monkeypatch hazard above is
  avoided precisely because these two stay put).

**Zero landed (non-test, non-`__legacy`) importers of `worker_resource_registration` exist at
`e5aa9d3`.** Nothing is added to *Allowed files* or `touch_set` on the importer side beyond the
remainder file and the new destination file themselves.

## P10-ALLOWLISTS ruling (assumption, flagged for operator ratification)

**BLOCKER, corrected (house rule 4 — v2 cited both allowlist files under their stale `__legacy` path;
both landed at p9).** `[MEASURED]` at `e5aa9d3`: `git cat-file -e e5aa9d3:src/entropic_substrate/__legacy/provider_launch.py`
and the same for `d1_runpod_adapter.py` both fail — neither exists under `__legacy` any more. Both
are LANDED, at `src/entropic_substrate/ladder/run/provider/provider_launch.py` (56 lines) and
`src/entropic_substrate/ladder/run/provider/d1_runpod_adapter.py` (601 lines), each with a
root-level generated alias shim (`src/entropic_substrate/provider_launch.py`,
`src/entropic_substrate/d1_runpod_adapter.py`) pointing at the landed address. This matches the
brief's own *Landed importers* table, which already cited the correct landed paths for the SAME two
files' `d1_operator_values` / `d1_runpod_billing_parser` imports — v2 contradicted itself between
that table and every allowlist-specific mention.

**ASSUMED, operator ratification pending.** The two fail-closed allowlist literals — now correctly
cited at `src/entropic_substrate/ladder/run/provider/provider_launch.py:32-35` (the `if command[2]
not in {...}:` set literal) and `src/entropic_substrate/ladder/run/provider/d1_runpod_adapter.py:349`
(the single dotted-name literal) — name `entropic_substrate.d1_trainer_mailbox_command` and
`entropic_substrate.d_ladder_worker_command`. `tools/mapping.json` maps the first to `composition`
(p12) and the second to `historical.d_ladder` (p12); NEITHER is a p10 module. **p10 does not edit
either literal.** `provider_launch.py` is BOTH an importer p10 rewrites (line 5, `d1_operator_values`)
AND the allowlist file p10 must not touch elsewhere; `d1_runpod_adapter.py` is BOTH an importer p10
rewrites (lines 12 and 17, `d1_operator_values` and `d1_runpod_billing_parser`) AND the allowlist
file. The diff scope named in ACCEPTANCE is explicit about which lines are whose.

**FOR-1488 is re-homed to p12** (unchanged from v2): both allowlists, `h_probe_launch.py:167` and
`ladder/review/d1_external_analysis_adapter.py:241` (both hardcode stale literals, both landed
already, both outside p10's touch set), the stale `ENTRYPOINT_H_WORKER` value, and the dangling
`justfile` reference to `patches/entrypoints-python.md` (`[MEASURED]` absent at `e5aa9d3`).

## ACCEPTANCE

Machine-checkable, one line each, each naming its instrument.

- The 27 full-move modules exist at the destination addresses under *Scope*, with regenerated shim
  pairs (`tools/gen_shims.py --verify`); `ladder/run/{campaign,closure,ckptsync}/__init__.py` new,
  docstring-only, zero re-exports; `ladder/run/worker/__init__.py` byte-identical (already 0 bytes).
- The worker-split leaf `ladder/run/worker/resource_registration.py` exists with exactly
  `worker_resource_registration`, importing `_FACTORY_DIGEST`/`_PROVIDER` back from the `__legacy`
  remainder by explicit address and `ResourceActivationMode`/`ResourceActivationRegistration` from
  the landed `kernel.evidence_class.resource_provenance`; `__legacy/d_ladder_worker_runtime.py` keeps
  `build_worker_factory` AND `verify_worker_source` byte-identical (only its
  `evidence.resource_provenance` import narrows by dropping the now-unused `ResourceActivationMode`)
  and gains ONE re-export line for the moved name; its own mapping row stays `action: split`,
  `landed: false`.
- `git diff -M --stat` shows every fully-moved module as a pure rename plus import-line rewrites
  only; the split leaf and the remainder are new/modified files, not renames, and are called out as
  such (a `git mv` rename-detector cannot and should not classify a split).
- **NO `remap_check` arm** (ruling P9-REMAP, FOR-1549, wave-wide, not re-opened here).
- **Allowlist byte-identity (P10-ALLOWLISTS), corrected instrument.**
  `git diff <startTip>..HEAD -- src/entropic_substrate/ladder/run/provider/provider_launch.py
  src/entropic_substrate/ladder/run/provider/d1_runpod_adapter.py` shows import-line hunks ONLY — the
  rewrite-on-arrival of `provider_launch.py:5` and `d1_runpod_adapter.py:12,17`'s `d1_operator_values`
  / `d1_runpod_billing_parser` imports — and the two literal blocks, quoted verbatim at `e5aa9d3`
  with line numbers (`provider_launch.py:32-35`; `d1_runpod_adapter.py:349`), are byte-identical in
  that diff; PLUS `tests/test_cli_surface.py::TestFrozenDottedNameLiteralSet::test_module_site_contains_its_frozen_literals_verbatim`
  green unedited for its `provider_launch` and `d1_runpod_adapter` parametrizations (see *Guard
  teeth*).
- **Cold import, fresh interpreter per moved leaf, entered FIRST, stdout marker not exit code**
  (`tests/test_run_upper_cold_import.py`, NEW, mirroring `tests/test_run_lower_cold_import.py`,
  RED-first on commit 1's destination addresses — including the split leaf — extended per subsequent
  commit). This drafting session did not execute this sweep (no code execution permitted); the lane
  runs it fresh at commit 1, per the DATA-BUILD-RUNG precedent of not assuming zero.
- **Topological check recorded before the first commit**, re-derived from the CORRECTED edge set
  (see *Scope — Intra-set import graph*): the lane asserts (a) the group about to land has no
  outgoing edge to an unlanded group, (b) the four landing commits partition the 27 full moves plus
  the split leaf lands with commit 1. A disagreement is a FREEZE and a `DECISION`.
- **R-PIN discharged in commit 1** (see *R-PIN*): `_CHECKPOINT_REGIME_TABLE` re-pinned in the SAME
  commit that rewrites `worker_command.py:14`, `worker_completion.py:8`, `worker_execution.py:8,9`;
  label and 51-member set UNCHANGED; `h-source-reader-v3` re-measured UNCHANGED as control;
  `docs/identity/read-old-table.md` gains a dated p10 row.
- **Every landing commit carries its own fragment rows AND its own MEASURED `legacy_edge_budget`**
  (BUDGET-PER-COMMIT). Baseline `[MEASURED]` at `e5aa9d3`: **32**, re-confirmed unchanged from
  `12f9f4a`. No per-commit delta is predicted in this brief (see *GAPS*); the lane measures live.
- `lint-imports` green; `just check`, `just pc-ci`, `just ship-it` green;
  `gh pr view --json headRefOid` equals `git rev-parse`.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms
  recorded.
- The actual diff's file set is a subset of *Allowed files*.

## Non-goals

- Does NOT touch `src/entropic_substrate/ladder/run/provider/provider_launch.py:32-35` or
  `.../d1_runpod_adapter.py:349` (P10-ALLOWLISTS, paths corrected from v2's stale `__legacy` citation).
  Does NOT author or edit `scripts/entrypoints.env`.
- Does NOT move `build_worker_factory` or `verify_worker_source`; both are p12's. MEASURED reason for
  `build_worker_factory` specifically (ruling D-E): moving it would silently disarm
  `tests/test_d_ladder_worker_runtime.py`'s monkeypatch of `build_torch_cuda_adapters` (a Python
  function resolves free names from its OWN defining module's globals, so a moved
  `build_worker_factory` would stop seeing the test's patched attribute on the `__legacy` remainder,
  and — since nothing left in the remainder would still need `build_torch_cuda_adapters` — the
  monkeypatch's default `raising=True` would then raise `AttributeError` outright). Leaves
  `d_ladder_worker_runtime`'s remainder holding `build_worker_factory` and `verify_worker_source`
  byte-identical, plus its now-four re-exported names (three already-landed-at-p4:
  `measure_hardware`, `verify_source_tree`, `SOURCE_PREFIXES`; one newly landed by p10:
  `worker_resource_registration`). Does NOT touch `tests/test_worker_source_anchor.py` or
  `tests/test_d_ladder_worker_runtime.py`.
- Does NOT rewrite `ladder/analyze/ladder_analysis.py:17`, `ladder/file/ladder_filing.py:12`, or
  `ladder/gate/ladder_predecessor.py:9` — the three landed importers of
  `entropic_substrate.__legacy.evidence.ladder_campaign`, not a p10 target (owner unresolved, see
  *GAPS*, kept DECISION-NEEDED).
- Does NOT touch `h_probe_launch.py` or `ladder/review/d1_external_analysis_adapter.py` (p12's).
- Does NOT edit `tools/mapping.json`, `.importlinter`, briefs, plan docs, the manifest, engine
  scripts, or CI.
- Does NOT edit any EXISTING test other than the identity mirrors named under *R-PIN*.
- Does NOT introduce a transient `entropic_substrate.__legacy.<x>` address for a module THIS phase
  itself moves: the peeled `d1_excluded_capture` lands with `d1_excluded_operator`'s commit, never
  via an intermediate quarantine hop.

## Allowed files

- the 27 full-move legacy sources enumerated under *Scope*
- `src/entropic_substrate/ladder/run/{campaign,closure,ckptsync}/**` (new, docstring-only
  `__init__.py` each) and `src/entropic_substrate/ladder/run/worker/d1_predecessor_loader.py` (+
  `.pyi`)
- `src/entropic_substrate/ladder/run/worker/resource_registration.py` (+ `.pyi`, NEW — the
  worker-split leaf, `worker_resource_registration` only) and
  `src/entropic_substrate/__legacy/d_ladder_worker_runtime.py` (remainder edit ONLY: remove
  `worker_resource_registration`'s 5-line body, narrow the `evidence.resource_provenance` import to
  drop the now-unused `ResourceActivationMode`, add ONE re-export line; `build_worker_factory` and
  `verify_worker_source` byte-identical)
- the 27 typed shim pairs at the old `__legacy` paths, regenerated, plus every regenerated
  ROOT-LEVEL old-prefix shim whose target moves
- the 22 distinct landed-importer files carrying the 30 rewritten lines (see *Landed importers*),
  each edited ONLY on the line(s) named there — including `ladder/run/provider/provider_launch.py`
  (line 5 only) and `ladder/run/provider/d1_runpod_adapter.py` (lines 12, 17 only), NEVER their
  allowlist lines
- `tests/test_run_upper_cold_import.py` (new, RED-first)
- `tests/test_identity_epoch.py`, `tools/regime_rows.py`, `tests/test_regime_rows.py`,
  `docs/identity/read-old-table.md` — R-PIN, commit 1 only
- `.claude/workflows/landed-set/p10-run-upper.json` — this phase's disjoint fragment
- generator outputs, `.secrets.baseline`, `.claude/workflows/landed-set/.emitted.json` (untracked,
  never `git add`) — as v2

**Deliberately EXCLUDED:** `src/entropic_substrate/ladder/run/provider/provider_launch.py:32-35`;
`src/entropic_substrate/ladder/run/provider/d1_runpod_adapter.py:349` (paths CORRECTED from v2's
stale `__legacy/provider_launch.py:33-34` / `__legacy/d1_runpod_adapter.py:349` citations — those
paths do not exist at `e5aa9d3`); `scripts/entrypoints.env`; `h_probe_launch.py`;
`ladder/review/d1_external_analysis_adapter.py`; `src/entropic_substrate/__legacy/predictive_state/__init__.py`;
`src/entropic_substrate/__legacy/evidence/ladder_campaign.py` and its three landed importers;
`tests/test_worker_source_anchor.py`; `tests/test_cli_surface.py`; `tools/mapping.json`;
`.importlinter`. A diff touching any of them is a `DECISION`, not an edit.

### Landed importers — 30 lines, 22 distinct files, 7 target modules `[MEASURED]`, re-confirmed at `e5aa9d3`

Unchanged from v2 (independently re-derived by this session and separately confirmed by the
verifier's own re-grep — both land on the same 30 lines / 22 files / per-target counts 19/4/2/2/1/1/1).

| target                           | importer file:line                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `d1_operator_values` (19)        | `ladder/run/construction/d1_r2_runtime_composition.py:18` · `ladder/run/h_probe_watchdog.py:11` · `ladder/run/h_runpod_transport.py:11` · `ladder/run/provider/d1_build_cloud_adapter.py:11` · `ladder/run/provider/d1_runpod_adapter.py:12` · `ladder/run/provider/d1_system_runtime_probe.py:10` · `ladder/run/provider/provider_launch.py:5` · `ladder/run/trainer/d1_trainer_stage.py:10` · `ladder/run/transport/d1_trainer_evidence_adapter.py:18` · `ladder/run/transport/d1_trainer_mailbox_contracts.py:12` · `ladder/run/transport/d1_trainer_mailbox_worker.py:12` · `ladder/run/transport/d1_trainer_object_transport.py:13` · `ladder/run/transport/d1_trainer_process_contracts.py:13` · `ladder/run/transport/d1_trainer_request_codec.py:14` · `ladder/run/values/d1_operator_ports.py:16` · `ladder/run/worker/h_probe_allocation.py:5` · `ladder/run/worker_command.py:14` **[READER]** · `ladder/run/worker_completion.py:8` **[READER]** · `ladder/run/worker_execution.py:8` **[READER]** |
| `d1_operator_evidence` (4)       | `ladder/run/trainer/d1_trainer_stage.py:5` · `ladder/run/transport/d1_trainer_evidence_adapter.py:9` · `ladder/run/transport/d1_trainer_output_publication.py:10` · `ladder/run/values/d1_operator_ports.py:11`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `d1_excluded_capture` (2)        | `ladder/analyze/ladder_registration.py:32`, `:380` — both inside `if TYPE_CHECKING:` blocks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `d1_predecessor_loader` (2)      | `ladder/gate/ladder_predecessor.py:8` · `ladder/run/worker_execution.py:9` **[READER]**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `d1_runpod_billing_parser` (1)   | `ladder/run/provider/d1_runpod_adapter.py:17`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `d1_checkpoint_dvc_sync` (1)     | `ladder/run/construction/d1_r2_runtime_composition.py:14`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `d1_checkpoint_dvc_recovery` (1) | `ladder/run/construction/d1_r2_runtime_composition.py:11`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

## R-PIN (OWED)

Unchanged in substance from v2, re-confirmed at `e5aa9d3`: `_READER_MODULES` in `checkpoint.py`
spans lines 65-117, 51 entries; `ladder.run.worker_command`/`worker_completion`/`worker_execution` at
lines 111/112/113. `d1_operator_values` and `d1_predecessor_loader` both rewrite bytes of these three
files and are bundled into commit 1 for exactly that reason (one re-pin, not two).

- p10 re-pins `_CHECKPOINT_REGIME_TABLE` with `tools/regime_rows.py` in commit 1. Label and 51-member
  SET unchanged. `docs/identity/read-old-table.md` gains a dated p10 row.
- **Control:** `h-source-reader-v3`'s 7-module tuple has ZERO overlap with p10's targets or importers
  `[MEASURED]`; re-measured UNCHANGED.
- **Classifier caveat (P10-RPIN):** if the auto-mode classifier refuses a direct digest-swap edit
  inside `tests/test_identity_epoch.py` (measured elsewhere this sprint for the identical file
  class), the lane computes the rows with `tools/regime_rows.py --python --torch --pysbd`
  (`[MEASURED]` all three flags `required=True` in its argparse definition) and returns `DECISION`
  carrying the computed rows verbatim; the conductor applies them.

## Guard teeth by mutation

| mutation                                                                                                                                               | must go RED (named instrument)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | GREEN arm                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| re-introduce ONE old-prefix import in a landed module                                                                                                  | `tests/test_no_old_prefix.py`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | zero landed→old-prefix edges, test unedited                                                                             |
| inflate or deflate the committed `legacy_edge_budget` by one, either direction                                                                         | `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | measured == committed, number never typed                                                                               |
| hand-delete one regenerated shim pair for a module THIS phase already landed                                                                           | `tools/gen_shims.py --verify` exits 1 naming the missing file and its expected target                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `--verify` exits 0 after regenerating                                                                                   |
| land a commit that moves modules WITHOUT adding their fragment rows                                                                                    | `tests/test_module_identity.py` and `gen_shims --preflight` / `merge_fragments` disjointness                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | every commit carries its own rows; `--preflight` clean                                                                  |
| hand-edit a `_CHECKPOINT_REGIME_TABLE` digest instead of re-pinning with `tools/regime_rows.py`                                                        | `tests/test_identity_epoch.py::TestComposedDigestsPerMeasuredRegime` and `tests/test_regime_rows.py`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | tool-written rows, label/set unchanged; `_SOURCE_REGIME_TABLE` re-measured UNCHANGED as control                         |
| **change one string in `provider_launch.py:32-35` or `d1_runpod_adapter.py:349`** — **BLOCKER, corrected: guard tooth EXISTS, not EXHIBIT-OR-RETRACT** | `tests/test_cli_surface.py::TestFrozenDottedNameLiteralSet::test_module_site_contains_its_frozen_literals_verbatim[provider_launch-...]` and the same test's `[d1_runpod_adapter-...]` parametrization. `[MEASURED]`: `_FROZEN_LITERAL_SITES_MODULE` (`tests/test_cli_surface.py:81-89`) maps `"provider_launch": ("entropic_substrate.d1_trainer_mailbox_command", "entropic_substrate.d_ladder_worker_command")` and `"d1_runpod_adapter": ("entropic_substrate.d1_trainer_mailbox_command",)`; the test resolves each dotted name `by importlib` ("to survive p3's shim rebind" per its own docstring, line 8-9), reads `Path(module.__file__).read_text(...)`, and asserts each literal `in text` — this FOLLOWS the root-level shim through to whichever file the row currently lands at, so it reaches the real, landed `provider_launch.py`/`d1_runpod_adapter.py` content, not a stale path. | the two literals byte-identical; both parametrized test cases green unchanged                                           |
| delete one forwarder-served `__legacy` file a moved module still imports before its own commit lands                                                   | `tests/test_run_upper_cold_import.py` (NEW) — cold-import arm only                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | all landed-so-far destination addresses cold-import clean in a fresh interpreter                                        |
| land `d1_excluded_capture` in the SAME commit as the rest of `closure`, ahead of `d1_excluded_operator`                                                | the topological check recorded before the first commit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | the peeled placement (commit 3, with `d1_excluded_operator`) shows zero forward references                              |
| hand-drop the remainder's one re-export line for `worker_resource_registration` after landing it                                                       | `tests/test_d_ladder_worker_runtime.py::8-13` — its own module-level import block, `from entropic_substrate.d_ladder_worker_runtime import (SOURCE_PREFIXES, build_worker_factory, measure_hardware, verify_source_tree, worker_resource_registration)`, resolves through the root shim into the `__legacy` remainder; with the re-export dropped that name is absent from the remainder's namespace and the import raises `ImportError` at COLLECTION time, RED for the whole file, not just one test. (`tests/test_d_ladder_worker_command.py:15`'s direct import of the same name would fail identically, but the collection-level failure above is the more direct instrument and is named as primary.)                                                                                                                                                                                          | both tests unedited, green; `__legacy/d_ladder_worker_runtime.py` re-exports `worker_resource_registration` by identity |

**RETRACTED (house rule 4).** v2's row for the allowlist literal read "must go RED: EXHIBIT OR
RETRACT... no test in `tests/` reaches that set literal". `[MEASURED]` at `e5aa9d3`:
`tests/test_cli_surface.py::TestFrozenDottedNameLiteralSet` DOES reach both literals, exactly as
tabled above — the earlier audit's grep for a literal-matching test missed this file. The tooth is
not absent; it is load-bearing.

**On the two `TYPE_CHECKING`-only importer lines** (`ladder_registration.py:32,380`, importing
`d1_excluded_capture`): unchanged from v2 — `ast.walk` sees them regardless of the guard, so
rewrite-on-arrival applies identically to a runtime import; no separate guard-teeth row needed.

## Observability conformance

Unchanged from v2, re-confirmed by re-running the same seven-pattern regex sweep across all 27
full-move targets at `e5aa9d3`: **ZERO hits on every pattern.** The worker-split leaf's one function
(`worker_resource_registration`, narrowed from v3's two) was ALSO swept `[MEASURED]`: zero hits.
Empty subject confirmed for the whole phase, split included.

## Scope — Intra-set import graph: 55 edges, DAG, THREE closure→campaign back-edges

**MAJOR, corrected (house rule 4 — v2's "exactly one" back-edge claim undercounted).** Re-derived by
this session at `e5aa9d3` (regex parse of every `from entropic_substrate.<x> import` / `import
entropic_substrate.<x>` line in the 27 full-move files, restricted to the other 26 as targets):
**55 edges, unchanged from `12f9f4a`.** Tarjan: 27 trivial singleton SCCs, zero module-level cycles —
the DAG claim holds.

`campaign → closure`: **12 edges** (unchanged from v2: `d1_campaign_bootstrap`,
`d1_campaign_evidence_assembler`, `d1_excluded_operator` ×3, `d1_operator_controller` ×3,
`d1_operator_custody_adapter` ×3, `d1_operator_evidence` ×1).

`closure → campaign`: **THREE edges, not one:**

1. `d1_excluded_capture.py:19` → `d1_excluded_operator` (v2's cited edge)
2. `d1_finalized_cell_custody.py:21` → `d1_operator_values` (`[MEASURED]`, omitted by v2)
3. `d1_runpod_billing_parser.py:12` → `d1_operator_values` (`[MEASURED]`, omitted by v2)

**Why only ONE of the three forces a peel.** Edges 2 and 3 both target `d1_operator_values`, which —
independently of this back-edge analysis — is already a pure SOURCE (zero outgoing intra-set edges)
and lands in commit 1, ahead of every closure and every campaign commit. Landing `d1_operator_values`
in commit 1 satisfies edges 2 and 3 by construction: `d1_finalized_cell_custody` (commit 2) and
`d1_runpod_billing_parser` (commit 2) both depend on a module that is ALREADY landed by the time
their own commit runs. Only edge 1 targets a module (`d1_excluded_operator`) that is NOT peeled into
commit 1 — it is an ordinary campaign member landing in commit 3 — so only `d1_excluded_capture`
itself needs to be peeled out of the closure commit and landed alongside it, in commit 3.

**Re-run topological order (Kahn's algorithm, alphabetical tie-break) over the corrected edge set:
identical result to v2's — the corrected count changes the EXPLANATION, not the PLAN.** The 4-commit
partition below has ZERO forward-reference violations against the full, now-three-edge-aware,
55-edge graph, independently re-verified by the verifier and by this session:

```text
commit 1 {d1_operator_values, d1_billing_finalization_values, d1_campaign_checkpoint,
          d1_predecessor_loader} + worker-split leaf (worker_resource_registration only)
                                                        — sources, zero outgoing intra-set edges
commit 2 {d1_billing_finalizer, d1_finalized_cell_custody, d1_finalized_cell_custody_repository,
          d1_runpod_billing_parser}                     — closure minus the one peeled exception
commit 3 {13 remaining campaign modules} + d1_excluded_capture   — campaign + the peeled exception
commit 4 {5 ckptsync modules}                            — pure consumers of campaign + closure
```

## Already ruled

| id                                                    | status                                                                       | ruling as it binds this phase                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P10-PARENT                                            | CONDUCTOR 2026-09-24                                                         | Parent is `mythwave/w2-reconcile` @ `<startTip>`.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| P10-SET                                               | CONDUCTOR 2026-09-24                                                         | 27 full moves. `predictive_state` is nobody's `ladder/run` business.                                                                                                                                                                                                                                                                                                                                                                                                             |
| P10-WORKER-SPLIT                                      | CONDUCTOR 2026-09-24, corrected TWICE                                        | v2: whole remainder p12's (RETRACTED). v3: `worker_resource_registration` AND `build_worker_factory` p10's (RETRACTED — over-read the disclaimed mapping.json note against ruling D-E's own explicit, reasoned text and would have disarmed `tests/test_d_ladder_worker_runtime.py`'s monkeypatch). v4 (this brief): `worker_resource_registration` ALONE is p10's; `build_worker_factory` and `verify_worker_source` are p12's, per D-E's literal text and its MEASURED reason. |
| P10-DAG                                               | CONDUCTOR 2026-09-24, corrected                                              | 27-node, 55-edge module graph is a DAG. THREE closure→campaign back-edges measured (not one); only `d1_excluded_capture`'s needs a peel, because the other two resolve via `d1_operator_values` already landing first in commit 1.                                                                                                                                                                                                                                               |
| P10-RPIN                                              | CONDUCTOR 2026-09-24                                                         | R-PIN owed, discharged in commit 1.                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| P10-ALLOWLISTS                                        | CONDUCTOR 2026-09-24, corrected paths, ASSUMED pending operator ratification | p10 owns neither allowlist literal; both files are LANDED at `ladder/run/provider/`, not `__legacy`. FOR-1488 whole is p12's.                                                                                                                                                                                                                                                                                                                                                    |
| P10-IMPORTERS                                         | CONDUCTOR 2026-09-24                                                         | 30 lines / 22 files are p10's. The 3 `evidence.ladder_campaign` lines are not p10's; owner unresolved (GAPS).                                                                                                                                                                                                                                                                                                                                                                    |
| W2-ADMISSIBLE-DELTAS                                  | CONDUCTOR 2026-09-23, plan-anchored 2026-09-24                               | `[MEASURED]` `docs/plans/src-hexagonal-refactor.plan.md:856`: governs w2's own p8/p9 zip slice deltas (the p9→p8 DEPENDS edge and a 17-file intersection). Not directly load-bearing for p10's own scope — cited here only to confirm it now exists and this brief makes no claim inconsistent with it.                                                                                                                                                                          |
| P9-REMAP                                              | CONDUCTOR 2026-09-23, FOR-1549                                               | No remap arm, guard row, or artifact.                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| BUDGET-PER-COMMIT                                     | CONDUCTOR 2026-09-23                                                         | Every landing commit writes its own MEASURED `legacy_edge_budget`.                                                                                                                                                                                                                                                                                                                                                                                                               |
| RUNG-OWNED                                            | CONDUCTOR 2026-09-23                                                         | `ladder/gate/ladder_predecessor.py` was rung-owned for four (now-rewritten) lines during w2. p10 owns a fifth, different line (`d1_predecessor_loader`), ordinary rewrite-on-arrival.                                                                                                                                                                                                                                                                                            |
| FACADE-EMPTY                                          | CONDUCTOR 2026-09-23                                                         | N/A: all 27 full-move targets are flat files, none under a `__legacy/<pkg>/` subdirectory.                                                                                                                                                                                                                                                                                                                                                                                       |
| DATA-BUILD-RUNG                                       | CONDUCTOR 2026-09-23                                                         | Precedent for why the cold-import sweep must actually run.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| E-4 / P-1 / rewrite-on-arrival / shim form / skeleton | RULED 2026-09-17                                                             | As v2, unchanged.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## Planned commit branches

**Ticket mapping, CORRECTED (house rule 4).** `[CONDUCTOR-STATED, not independently verifiable by
this drafting session — no Linear access]`: FOR-1483 ("author the parameterized entrypoint source")
is already LANDED, at p0, under ruling E-4; the reconciliation pass on 2026-09-24 put it In Review.
It is **NOT a p10 commit id.** p10's four tickets map by TITLE, not by commit order:
**FOR-1484 = campaign, FOR-1485 = closure, FOR-1486 = ckptsync, FOR-1487 = worker D1 half**
(`d1_predecessor_loader` + the D-E split). p10's ACCEPTANCE for FOR-1483 is verification-only: the
file exists, is consumed by the justfile and `run_ladder.sh`, and p10 makes no code change to it —
resolves v2's GAP 2.

Commit 1 lands SOURCE modules from three different sub-packages plus the worker split, so — per the
p9 absorbed-id precedent (surplus Linear ids annotated "absorbed into commit N") — its PRIMARY ticket
is **FOR-1487** (it is the worker commit: `d1_predecessor_loader` + the split), and its commit body
carries two ABSORPTION notes for the other sub-packages it happens to land a source module from.
Commit 3 similarly carries one absorption note, for the peeled `d1_excluded_capture` (FOR-1485's
module, landed here rather than in FOR-1485's own commit).

| # | ticket                                                                                                                                                              | scope                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |     modules |             lines |
| - | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------: | ----------------: |
| 1 | **FOR-1487** (primary); commit-body notes: "FOR-1484 partial: `d1_operator_values`, `d1_campaign_checkpoint`", "FOR-1485 partial: `d1_billing_finalization_values`" | R-PIN commit: the 4 SOURCE modules — `d1_operator_values`, `d1_billing_finalization_values`, `d1_campaign_checkpoint`, `d1_predecessor_loader` — plus the worker-split leaf (`worker_resource_registration` ONLY, corrected down from v3's two-function split). Rewrites all 19 `d1_operator_values` importer lines + both `d1_predecessor_loader` importer lines (21 of 30 total). Re-pins `_CHECKPOINT_REGIME_TABLE`. `tests/test_run_upper_cold_import.py` NEW, RED-first, on these 4 + split-leaf destination addresses. | 4 (+ split) | 1287 (+~13 split) |
| 2 | **FOR-1485** ("closure", first of two commits; completes with commit 3's absorbed module)                                                                           | `closure` minus the one peeled exception: `d1_billing_finalizer`, `d1_finalized_cell_custody`, `d1_finalized_cell_custody_repository`, `d1_runpod_billing_parser`. Rewrites `d1_runpod_billing_parser`'s 1 importer line. Cold-import test extended.                                                                                                                                                                                                                                                                         |           4 |              1080 |
| 3 | **FOR-1484** (primary); commit-body note: "FOR-1485 partial: `d1_excluded_capture` (peeled for the measured back-edge to `d1_excluded_operator`)"                   | `campaign` (the 13 remaining) plus the peeled `d1_excluded_capture`. Rewrites `d1_operator_evidence`'s 4 importer lines and both `d1_excluded_capture` TYPE_CHECKING lines. Cold-import test extended.                                                                                                                                                                                                                                                                                                                       |          14 |              4569 |
| 4 | **FOR-1486**                                                                                                                                                        | `ckptsync`, all 5. Rewrites `d1_checkpoint_dvc_sync`'s and `d1_checkpoint_dvc_recovery`'s 1 importer line each. Cold-import test extended to all 27 + split leaf. **Closing step, no new ticket:** final `legacy_edge_budget` measurement and dated note, allowlist byte-identity re-confirmation across all four commits, `.secrets.baseline` reconciliation, full certification record assembly.                                                                                                                           |           5 |              1016 |

Rows 1-4 sum to 27 full-move modules / 7853 lines, plus the ~13-line split leaf (narrowed from v3's
34-line, two-function split — see *P10-WORKER-SPLIT*), matching *Scope* exactly. Topological validity
re-verified against the corrected 55-edge, three-back-edge graph:
**zero forward-reference violations** (see *Scope — Intra-set import graph*, both this session's and
the verifier's independent re-derivation agree).

## GAPS / DECISION-NEEDED

1. **DECISION-NEEDED, unchanged, kept verbatim per conductor instruction.**
   `entropic_substrate.evidence.ladder_campaign`'s gate half has no owner at `e5aa9d3`. The p5b brief
   states `frozen_after: p7` and describes p7 as the phase that moves the remainder
   (`RunAuthorization`, `SuccessorAuthorization`, `FiledLadderGate`, etc.) to `ladder.gate`.
   `[MEASURED]`: `p7-register-gate.json`'s `modules` map carries NO row for `evidence.ladder_campaign`
   at `e5aa9d3` — well after p7, p8, p9, p11 and the w2-reconcile rung all landed. The module is still
   fully quarantined; its 3 landed importers still resolve through the `__legacy` address:
   - `ladder/analyze/ladder_analysis.py:17`
   - `ladder/file/ladder_filing.py:12`
   - `ladder/gate/ladder_predecessor.py:9`

   This is not p10's to pick up (not a `ladder.run` destination, not in the
   campaign/closure/ckptsync/worker categories), but someone needs to own it before p13's
   `__legacy`-empty postcondition. Recommend: the conductor files a dedicated small phase or folds it
   into p12/p13's scope, whichever their own audits confirm is free.
2. **RESOLVED (was GAP 2 in v2).** FOR-1483 is landed at p0 (E-4), reconciliation 2026-09-24 put it
   In Review; not a p10 commit id; p10 verification-only.
3. **`patches/entrypoints-python.md`, cited by `justfile:9`, does not exist at `e5aa9d3`.** Dangling
   reference, p12's to fix when it picks up FOR-1488.
4. **Whether a `TYPE_CHECKING`-only import counts toward `tests/test_no_old_prefix.py`'s edge
   classification is untested anywhere in this sprint.** p10's own two such lines are the first case;
   not blocking.
5. **Per-commit `legacy_edge_budget` deltas are not predicted in this brief.** This draft could not
   run `tools/import_graph.py` against a tree with p10's modules actually moved (no code execution or
   file writes permitted in this drafting session). The lane measures live at commit 1 from the
   `[MEASURED]` baseline of 32.
