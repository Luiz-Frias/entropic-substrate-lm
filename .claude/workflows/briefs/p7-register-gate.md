# p7-register-gate — the preregistration layer lands in `ladder/{register,gate}`

Authoritative brief for one `td-phase-lifecycle.cascade` lifecycle. v2, written 2026-09-23,
superseding the 2026-09-17 v1 (`p7-register-gate.md`) under the conductor's serial ruling
(D-WAVES r3, 2026-09-23): `tools/lane_slices.py` measured `p7 -> p6` import dependencies through
the `observation` facade, p6 has since LANDED serially, and every wave-lane / zip / `publishMode:'defer'`
/ CERTIFIED construction of v1 is struck. Plan: `docs/plans/src-hexagonal-refactor.plan.md`
(§p7-register-gate, conductor-rulings table). Every count below is MEASURED at `861d015` on
`mythwave/p6-core` with the command recorded beside it in `p7-notes.md`. The lifecycle's
`acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p7-register-gate`                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| parent                                       | `mythwave/p6-core`                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| mode (sprint manifest, parallel eligibility) | `tracer` (serial: `parallel_group` is `null`). CORRECTED 2026-09-23 from v1's `ratchet`: the schema's own enum description reads "ratchet = known set (parallel-eligible)", so a serial phase is `tracer`, as p6-core and p5b-ladder-campaign both are.                                                                                                                                                                                                          |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate.                                                                                                                                                                                                                                                                                                                                       |
| engine_mode                                  | `arch` (manifest field, machine-read by the launch composer; ruled 2026-09-18)                                                                                                                                                                                                                                                                                                                                                                                   |
| wave                                         | `serial`. There is no `w1`.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `gates.liveRequired`                         | `true`                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| live gate                                    | `just ship-it` runs INSIDE the lifecycle. House rule 1: never a bare `gt submit`. FOR-1541 workaround as ruled for p5, p5b and p6: branch-only `gt submit` after a green `just ship-it` whose sole failure is the commitizen-branch empty range on an up-to-date lower branch.                                                                                                                                                                                   |
| gates                                        | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}` — this is the LIFECYCLE `gates` arg. The sprint-manifest `gates` object keeps only `{"liveRequired": true}`: the schema sets `additionalProperties: false` on it and admits nothing but `liveRequired` / `liveDeferred`, and every landed phase entry (p0, p5b, p6) carries exactly that. `liveDeferred` is REMOVED here — nothing is deferred, there is no zip. |
| launch                                       | SERIAL phase: the conductor launches the ported cascade DIRECTLY. Not a wave member. There is no lane worktree, no zip, no `publishMode:'defer'`, and no CERTIFIED terminal state.                                                                                                                                                                                                                                                                               |
| `repoToplevel`                               | `/Users/luizfrias/CursorAI/entropic-substrate-lm`. The MAIN checkout, never a worktree.                                                                                                                                                                                                                                                                                                                                                                          |
| trunk                                        | `refactor/hexagonal-redesign`                                                                                                                                                                                                                                                                                                                                                                                                                                    |

## Contract surface owned

| contract                                                                                                                                                                                    | kind     | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| the 19 `ladder.register.*` / `ladder.gate.*` addresses enumerated under *Scope* (`ladder.register` 16, `ladder.gate` 3)                                                                     | contract | p13          |
| `ladder/register/rungs/` and `ladder/register/arms/` scaffolding (EMPTY, docstring-only `__init__.py` each)                                                                                 | contract | p13          |
| destination package `__init__.py` shape for `ladder/register`, `ladder/gate`: DOCSTRING-ONLY, no re-exports (p4/p6 precedent, `src/entropic_substrate/core/sentence/__init__.py`, 85 bytes) | contract | p13          |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item
below is LANDED before this phase launches.

- kernel addresses (p4, p5): MEASURED, the 19 moved modules import exactly six landed kernel
  addresses — `kernel.identity.d1_primitives` (15 lines; the Landed table below cites all 15, 13 was a typed count retracted 2026-09-23), `kernel.runtime.config` (3),
  `kernel.verdict.d1_contract` (3), `kernel.verdict.d1_types` (3), `kernel.lifecycle` (2),
  `kernel.evidence_class.resource_provenance` (1)
- **p6 core addresses** (the reason this phase follows p6 serially): `core.observation.byte_codec`,
  `core.observation.contracts`, `core.latent.dynamics`, `core.sentence.profile`,
  `core.sentence.records`, `core.sentence.recovery_state`
- `kernel.artifacts` as extended by p5b (the five artifact functions), plus `kernel.lifecycle`
- `tools/gen_shims.py`, the landed-set fragment convention, `tests/test_legacy_coupling.py`
  (EQUALITY arm; p5's monotone arm is RETRACTED — see *Already ruled*)
- `tests/test_no_old_prefix.py` (p3b); `tools/regime_rows.py` + `tests/test_regime_rows.py` +
  `tests/test_identity_epoch.py` (identity mirrors, ruling Q1) — consumed READ-ONLY here, see R-PIN
- `tools/lane_slices.py`, `tools/import_graph.py` (conductor measurement oracles)
- `kernel/runtime/logging.py` (p3b) — available; MEASURED to have no work in this phase

## Sacred invariants

The five below are quoted verbatim from `CLAUDE.md` (North star and *Non-negotiable numerics*).
They bind every phase without exception. `SI-*` ids are local to this brief set; the text is the
contract.

| id   | verbatim invariant                                                                                                          | tripwire                                                                                                                                                                                  | guard-teeth recipe (house rule 5, both arms)                                                                                                                                                                                                                                                                                                                                      |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | This phase moves NO numerics module: MEASURED, `touch_set ∩ core/** = ∅`. p6 landed `core/geometry`; its FP32 islands and `tests/test_kappa_autocast_island.py` must stay byte-identical. | The load-bearing arm is p6's autocast-STATE spy (`tests/test_kappa_autocast_island.py`), not a dtype arm (`_geometry_dtype()` floors to fp32 regardless of autocast state — p0 round-C, FOR-1517). This phase records the arm as UNTOUCHED and proves it by `git diff mythwave/p6-core -- tests/test_kappa_autocast_island.py src/entropic_substrate/core/geometry/` being EMPTY. |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | `touch_set ∩ core/geometry = ∅`.                                                                                                                                                          | Widen the κ clamp bound by one ulp and feed a boundary input; `tests/test_kappa_geometry.py`'s κ-domain arm must go RED. Revert in-branch.                                                                                                                                                                                                                                        |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | `touch_set ∩ core/substrate = ∅`.                                                                                                                                                         | Detach κ from the load-balance term; the gradient-reaches-κ unit test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                              |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | `touch_set ∩ core/sentence = ∅`. `h_preservation.py` moves and IMPORTS `core.sentence.{profile,records,recovery_state}`; it must not change one carrier line.                             | Append a `UnitNorm` to the magnitude-preserving carrier's output; the carrier magnitude test must go RED. Revert in-branch.                                                                                                                                                                                                                                                       |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | `touch_set ∩ core/latent = ∅`. `ladder.py` moves and IMPORTS `core.latent.dynamics`; the latent law is untouched.                                                                         | Substitute a `sqrt(p + eps)` for the latent law in the main path; the latent-path law test must go RED. Revert in-branch.                                                                                                                                                                                                                                                         |

House rules from `CLAUDE.md`, verbatim headline, each binding in this phase:

| id   | verbatim house rule                                                            | tripwire                                                                                        | guard-teeth recipe                                                                                                                                                                                   |
| ---- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HR-1 | "Publish with `just ship-it`, never a bare `gt submit`."                       | A serial phase that publishes without the live gate. `just ship-it` runs INSIDE this lifecycle. | Inspect the transcript for any `gt submit` not reached through `just ship-it` (FOR-1541 branch-only fallback excepted, and only after a green live gate); presence is a BLOCKED verdict, not a note. |
| HR-2 | "Probe sequentially. Never draw a conclusion from concurrent containers."      | The podman VM is 4 vCPU / 4 GB; two MLflow stacks at once make even a control appear hung.      | Any container conclusion in this phase must name the single container it came from. Two concurrent stacks in the transcript invalidate it.                                                           |
| HR-3 | "Never fix flakiness by raising a timeout or a retry budget."                  | A gate that goes green only after a timeout bump.                                               | A diff that raises any timeout or retry constant is a BLOCKED verdict unless it carries the measurement showing the cause was found.                                                                 |
| HR-4 | "Comments record measurements, with the measurement."                          | A comment in this phase's diff stating a fact with no measurement attached.                     | A wrong measured claim is retracted explicitly, in the commit that fixes it. Silent overwrite is a defect. This phase carries FIVE such retractions (see *Already ruled*).                           |
| HR-5 | "TDD is non-negotiable, and guard-teeth prove both arms."                      | A guard landed without its RED arm demonstrated.                                                | Every guard carries the mutation listed under *Guard teeth by mutation*, performed and reverted in-branch, evidence in the PR body.                                                                  |
| HR-6 | "Secrets never enter git."                                                     | gitleaks runs at pre-commit and pre-push.                                                       | No credential, token or `.env` value in any diff of this phase.                                                                                                                                      |
| HR-7 | "Upgrading an existing venv across the mlflow → mlflow-skinny swap breaks it." | `import mlflow` raising `ModuleNotFoundError`.                                                  | Fix with `uv sync --all-groups --reinstall-package mlflow-skinny`, never by pinning around it.                                                                                                       |

## Scope

The PREREGISTER and SUCCESSOR-AUTHORIZED layer: **19 leaf modules**, MEASURED as the
`tools/mapping.json` rows whose `dest` is `entropic_substrate.ladder.register` or
`entropic_substrate.ladder.gate`, whose legacy source is a leaf `.py` under
`src/entropic_substrate/__legacy/`, and which no landed-set fragment already marks `landed: true`.
Zero behaviour change: registration and admission logic move unchanged.

RETRACTION (house rule 4): v1 said "the 20 modules mapped to `ladder/{register,gate}`", the plan's
§p7 row says 20, and the manifest's `contracts_owned` says "ladder/register (17), ladder/gate (3)".
All three are WRONG and are retracted, not overwritten. There are 20 MAPPING ROWS, one of which —
`entropic_substrate.experiments` — is a PACKAGE KEY, never a movable target and never a fragment
row: `tools/gen_shims.py::merge_fragments` raises `ManifestError` on a fragment naming one, and
`tools/lane_slices.py::parse_lane` exits 2 on one. The correct counts are register 16, gate 3 = 19.

| legacy source (moves)                                                    | destination address                                          | typed shim pair at the old path (regenerated by `tools/gen_shims.py`)    |
| ------------------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `src/entropic_substrate/__legacy/evidence/d1_family_adapter.py`          | `entropic_substrate.ladder.gate.d1_family_adapter`           | `src/entropic_substrate/evidence/d1_family_adapter.py` + `.pyi`          |
| `src/entropic_substrate/__legacy/evidence/d1_verdict.py`                 | `entropic_substrate.ladder.gate.d1_verdict`                  | `src/entropic_substrate/evidence/d1_verdict.py` + `.pyi`                 |
| `src/entropic_substrate/__legacy/evidence/ladder_predecessor.py`         | `entropic_substrate.ladder.gate.ladder_predecessor`          | `src/entropic_substrate/evidence/ladder_predecessor.py` + `.pyi`         |
| `src/entropic_substrate/__legacy/experiments/d1.py`                      | `entropic_substrate.ladder.register.d1`                      | `src/entropic_substrate/experiments/d1.py` + `.pyi`                      |
| `src/entropic_substrate/__legacy/experiments/d1_config.py`               | `entropic_substrate.ladder.register.d1_config`               | `src/entropic_substrate/experiments/d1_config.py` + `.pyi`               |
| `src/entropic_substrate/__legacy/experiments/d1_config_profile.py`       | `entropic_substrate.ladder.register.d1_config_profile`       | `src/entropic_substrate/experiments/d1_config_profile.py` + `.pyi`       |
| `src/entropic_substrate/__legacy/experiments/d1_confirmatory_matrix.py`  | `entropic_substrate.ladder.register.d1_confirmatory_matrix`  | `src/entropic_substrate/experiments/d1_confirmatory_matrix.py` + `.pyi`  |
| `src/entropic_substrate/__legacy/experiments/d1_execution.py`            | `entropic_substrate.ladder.register.d1_execution`            | `src/entropic_substrate/experiments/d1_execution.py` + `.pyi`            |
| `src/entropic_substrate/__legacy/experiments/d1_family.py`               | `entropic_substrate.ladder.register.d1_family`               | `src/entropic_substrate/experiments/d1_family.py` + `.pyi`               |
| `src/entropic_substrate/__legacy/experiments/d1_launch_attempt_codec.py` | `entropic_substrate.ladder.register.d1_launch_attempt_codec` | `src/entropic_substrate/experiments/d1_launch_attempt_codec.py` + `.pyi` |
| `src/entropic_substrate/__legacy/experiments/d1_launch_attempts.py`      | `entropic_substrate.ladder.register.d1_launch_attempts`      | `src/entropic_substrate/experiments/d1_launch_attempts.py` + `.pyi`      |
| `src/entropic_substrate/__legacy/experiments/d1_plan.py`                 | `entropic_substrate.ladder.register.d1_plan`                 | `src/entropic_substrate/experiments/d1_plan.py` + `.pyi`                 |
| `src/entropic_substrate/__legacy/experiments/d1_provenance.py`           | `entropic_substrate.ladder.register.d1_provenance`           | `src/entropic_substrate/experiments/d1_provenance.py` + `.pyi`           |
| `src/entropic_substrate/__legacy/experiments/d1_registered_design.py`    | `entropic_substrate.ladder.register.d1_registered_design`    | `src/entropic_substrate/experiments/d1_registered_design.py` + `.pyi`    |
| `src/entropic_substrate/__legacy/experiments/d1_registration_codec.py`   | `entropic_substrate.ladder.register.d1_registration_codec`   | `src/entropic_substrate/experiments/d1_registration_codec.py` + `.pyi`   |
| `src/entropic_substrate/__legacy/experiments/d1_runs.py`                 | `entropic_substrate.ladder.register.d1_runs`                 | `src/entropic_substrate/experiments/d1_runs.py` + `.pyi`                 |
| `src/entropic_substrate/__legacy/experiments/d1_stabilization.py`        | `entropic_substrate.ladder.register.d1_stabilization`        | `src/entropic_substrate/experiments/d1_stabilization.py` + `.pyi`        |
| `src/entropic_substrate/__legacy/experiments/h_preservation.py`          | `entropic_substrate.ladder.register.h_preservation`          | `src/entropic_substrate/experiments/h_preservation.py` + `.pyi`          |
| `src/entropic_substrate/__legacy/experiments/ladder.py`                  | `entropic_substrate.ladder.register.ladder`                  | `src/entropic_substrate/experiments/ladder.py` + `.pyi`                  |

**Shim form (ruled 2026-09-17, quoted from the generator's own output,
`src/entropic_substrate/sentence_prediction/head.py`):**

```python
# Generated by gen_shims.py from .claude/workflows/landed-set -- do not hand-edit.
# Alias shim for entropic_substrate.sentence_prediction.head -> entropic_substrate.core.sentence.head (docs/plans/src-hexagonal-refactor.plan.md 'Shim form (required)').
import sys

import entropic_substrate.core.sentence.head as _target

sys.modules[__name__] = _target
```

The `_target` is the NEW `ladder.*` address, never the `__legacy` address. A string-only
`importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.
RETRACTION (house rule 4): v1's *Already ruled* "shim form" row read
`import entropic_substrate.__legacy.<x> as _target`. That is the pre-p3 form and is WRONG for a
phase that lands a destination; retracted here, not overwritten.

**Package `__init__.py` disposition** (package keys are NOT fragment rows; `tools/gen_shims.py`
fails loud on one):

| package       | moved facade `src/entropic_substrate/__legacy/<pkg>/__init__.py`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | live old-path facade `src/entropic_substrate/<pkg>/__init__.py` | new destination `__init__.py`                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `experiments` | **STAYS VERBATIM, UNTOUCHED.** MEASURED 2026-09-23: 286 bytes, a module docstring plus `from __future__ import annotations`, ZERO re-exports. `experiments` is NOT one of the seven facades covered by the p3 emptied-facade ruling (that set is `data`, `geometry`, `init`, `layers`, `models`, `observation`, `predictive_state`), so p6's "all seven" statement does not reach it — but the measurement shows it needs no emptying either. Its mapping row `entropic_substrate.experiments` (dest `ladder.register`) describes the p13 end state and is owned by p13. | stays VERBATIM. Untouched.                                      | `ladder/register/__init__.py` NEW, docstring-only                                                                           |
| `evidence`    | NOT emptied — `evidence` is not one of the seven facades and its rows fan out to four different dests (`kernel.verdict`, `kernel.evidence_class`, `ladder.gate`, `ladder.capture`); only its three gate leaves move here.                                                                                                                                                                                                                                                                                                                                                | stays VERBATIM. Untouched.                                      | `ladder/gate/__init__.py` NEW, docstring-only                                                                               |
| —             | —                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | —                                                               | `ladder/register/rungs/__init__.py` and `ladder/register/arms/__init__.py` NEW, docstring-only, EMPTY packages (no members) |

**Split-row rule (p4 D-E / p6 precedent, conductor 2026-09-23).** p7 writes DOCSTRING-ONLY
destination `__init__.py`s and touches neither the `__legacy/experiments` facade nor the
`__legacy/evidence` facade, exactly as p4 wrote `core/sentence/__init__.py` (85 bytes, docstring
only, zero re-exports) and p6 wrote `core/{geometry,substrate,observation,corpus}/__init__.py`.

**Importer sweep (rewrite-on-arrival) — no landed importer exists.** MEASURED 2026-09-23 with
`tools/lane_slices.py` and confirmed by grep over the five landed layers: **ZERO** landed modules
import any of the 19 targets by an explicit `entropic_substrate.__legacy.<x>` address. The lane
slice is therefore exactly 57 files = 19 legacy sources + 38 shim-pair files, with no importer
column at all. Consequence, stated explicitly because it inverts p5b's and p6's shape:

- **No `_READER_MODULES` member is touched, so NO regime re-pin is owed (R-PIN does not fire).**
  MEASURED: `h-checkpoint-reader-v2`'s 51-module tuple
  (`src/entropic_substrate/ladder/run/checkpoint.py`) and `h-source-reader-v3`'s seven-module tuple
  (`src/entropic_substrate/ladder/capture/capture.py`) contain NO p7 target under its old name and
  no `ladder.register.*` / `ladder.gate.*` address. The four identity files
  (`tests/test_identity_epoch.py`, `tools/regime_rows.py`, `tests/test_regime_rows.py`,
  `docs/identity/read-old-table.md`) are deliberately **OUT of Allowed files**.
- The CONTROL is an assertion, not an omission: `git diff mythwave/p6-core -- tests/test_identity_epoch.py`
  is EMPTY, i.e. both `_CHECKPOINT_REGIME_TABLE` and `_SOURCE_REGIME_TABLE` are byte-identical to
  the parent tip, and `tests/test_identity_epoch.py` + `tests/test_regime_rows.py` are green
  unchanged. Evidence in the PR body.
- RETRACTION (house rule 4): v1's *Wave slice rule* section named the five
  `zip_owned_regenerable` files as this lane's possible outputs. There is no zip and no lane; the
  section is struck, and with zero reader-set members touched those files are not this phase's at all.

**Rewrite-on-arrival for the 19 moved modules' own imports.** MEASURED 2026-09-23: the 19 legacy
sources carry **73** `entropic_substrate.*` import lines — 31 that become a LANDED address, 30
intra-set (sibling target) lines, and 12 that resolve to still-quarantined modules of which 2
rewrite to a p6 address, leaving **10 explicit `entropic_substrate.__legacy.<x>` lines**. Every
line, with the address it becomes:

*Intra-set (30 lines): `entropic_substrate.experiments.<leaf>` → `entropic_substrate.ladder.register.<leaf>`,
`entropic_substrate.evidence.<leaf>` → `entropic_substrate.ladder.gate.<leaf>`, for the 19 targets only.*
`d1_family_adapter.py:6` · `d1.py:9,37,46,47` · `d1_config.py:10,17` · `d1_config_profile.py:17,28,51` ·
`d1_confirmatory_matrix.py:11,16,23` · `d1_family.py:10,30,36,46,177,196,338` ·
`d1_launch_attempt_codec.py:11` · `d1_plan.py:9,19,27` · `d1_registered_design.py:18` ·
`d1_registration_codec.py:23,29` · `d1_runs.py:7,14` · `ladder.py:9`.

*Landed (31 lines), one row per address:*

| line(s)                                                                                                                                                                                                                                                                                                                                      | old import                                                                 | becomes                                                                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `d1.py:16` `d1_config.py:20` `d1_config_profile.py:33` `d1_confirmatory_matrix.py:17` `d1_execution.py:9` `d1_family.py:14` `d1_launch_attempt_codec.py:19` `d1_launch_attempts.py:10` `d1_plan.py:10` `d1_provenance.py:8` `d1_registered_design.py:19` `d1_registration_codec.py:24` `d1_runs.py:8` `d1_stabilization.py:8` `ladder.py:10` | `entropic_substrate.experiments.d1_primitives`                             | `entropic_substrate.kernel.identity.d1_primitives`                                                                                                            |
| `d1_config.py:9` `d1_config_profile.py:16` `d1_plan.py:8`                                                                                                                                                                                                                                                                                    | `entropic_substrate.config`                                                | `entropic_substrate.kernel.runtime.config`                                                                                                                    |
| `d1_verdict.py:11` `d1_registered_design.py:17` `d1_registration_codec.py:13`                                                                                                                                                                                                                                                                | `entropic_substrate.evidence.d1_contract`                                  | `entropic_substrate.kernel.verdict.d1_contract`                                                                                                               |
| `d1_family_adapter.py:5` `d1_verdict.py:21` `d1_registration_codec.py:21`                                                                                                                                                                                                                                                                    | `entropic_substrate.evidence.d1_types`                                     | `entropic_substrate.kernel.verdict.d1_types`                                                                                                                  |
| `ladder_predecessor.py:28` `d1_registration_codec.py:22`                                                                                                                                                                                                                                                                                     | `entropic_substrate.evidence.lifecycle`                                    | `entropic_substrate.kernel.lifecycle`                                                                                                                         |
| `d1_confirmatory_matrix.py:10`                                                                                                                                                                                                                                                                                                               | `entropic_substrate.evidence.resource_provenance`                          | `entropic_substrate.kernel.evidence_class.resource_provenance`                                                                                                |
| `d1_config_profile.py:58`                                                                                                                                                                                                                                                                                                                    | `entropic_substrate.observation` (`ByteObservationCodec`, `CodecIdentity`) | **SPLITS** into `entropic_substrate.core.observation.byte_codec` (`ByteObservationCodec`) + `entropic_substrate.core.observation.contracts` (`CodecIdentity`) |
| `d1_family.py:43`                                                                                                                                                                                                                                                                                                                            | `entropic_substrate.observation` (`CodecIdentity`)                         | `entropic_substrate.core.observation.contracts`                                                                                                               |
| `h_preservation.py:7`                                                                                                                                                                                                                                                                                                                        | `entropic_substrate.sentence_prediction.profile`                           | `entropic_substrate.core.sentence.profile`                                                                                                                    |
| `h_preservation.py:8`                                                                                                                                                                                                                                                                                                                        | `entropic_substrate.sentence_prediction.records`                           | `entropic_substrate.core.sentence.records`                                                                                                                    |
| `h_preservation.py:9`                                                                                                                                                                                                                                                                                                                        | `entropic_substrate.sentence_prediction.recovery_state`                    | `entropic_substrate.core.sentence.recovery_state`                                                                                                             |
| `ladder.py:11`                                                                                                                                                                                                                                                                                                                               | `entropic_substrate.predictive_state.dynamics`                             | `entropic_substrate.core.latent.dynamics`                                                                                                                     |

*Explicit `__legacy` (10 lines) — each one a NEW landed→`__legacy` edge:*

| line                       | old import                                              | becomes                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `d1_verdict.py:12`         | `entropic_substrate.evidence.d1_filing`                 | `entropic_substrate.__legacy.evidence.d1_filing`                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `d1_verdict.py:19`         | `entropic_substrate.evidence.d1_filing_graph`           | `entropic_substrate.__legacy.evidence.d1_filing_graph`                                                                                                                                                                                                                                                                                                                                                                                                            |
| `d1_verdict.py:20`         | `entropic_substrate.evidence.d1_gate`                   | `entropic_substrate.__legacy.evidence.d1_gate`                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `d1_verdict.py:43`         | `entropic_substrate.evidence.d1_validator`              | `entropic_substrate.__legacy.evidence.d1_validator`                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ladder_predecessor.py:8`  | `entropic_substrate.analysis.ladder_analysis`           | `entropic_substrate.__legacy.analysis.ladder_analysis`                                                                                                                                                                                                                                                                                                                                                                                                            |
| `ladder_predecessor.py:9`  | `entropic_substrate.analysis.ladder_filing`             | `entropic_substrate.__legacy.analysis.ladder_filing`                                                                                                                                                                                                                                                                                                                                                                                                              |
| `ladder_predecessor.py:10` | `entropic_substrate.d1_predecessor_loader`              | `entropic_substrate.__legacy.d1_predecessor_loader`                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ladder_predecessor.py:11` | `entropic_substrate.evidence.d1_capture_artifacts`      | `entropic_substrate.__legacy.evidence.d1_capture_artifacts`                                                                                                                                                                                                                                                                                                                                                                                                       |
| `ladder_predecessor.py:17` | `entropic_substrate.evidence.d1_capture_bundle`         | `entropic_substrate.__legacy.evidence.d1_capture_bundle`                                                                                                                                                                                                                                                                                                                                                                                                          |
| `ladder_predecessor.py:18` | `entropic_substrate.evidence.ladder_campaign` (9 names) | **SPLITS**: `artifact_from_mapping`, `object_fields`, `read_bound_artifact`, `read_json_artifact` → `entropic_substrate.kernel.artifacts` (p5b moved them there and the legacy module only re-exports them by identity); `FiledLadderGate`, `actor_from_mapping`, `read_campaign_design`, `verify_ladder_registration` → `entropic_substrate.__legacy.evidence.ladder_campaign` (the gate half, still quarantined, `landed: false` in `p5b-ladder-campaign.json`) |

All ten land in `ladder/gate/`; `ladder/register/**` adds ZERO landed→`__legacy` edges.

**`legacy_edge_budget` prediction `[predicted]`, conductor arithmetic.** Parent tip
`mythwave/p6-core` @ `861d015` MEASURED **35** landed→`__legacy` edges with `tools/import_graph.py`,
exactly equal to `p6-core.json`'s committed budget (lens check passed). p7 removes **0** (no landed
importer exists) and adds **10** distinct `(landed module, __legacy module)` pairs — 4 from
`ladder/gate/d1_verdict.py`, 6 from `ladder/gate/ladder_predecessor.py`. Predicted
**35 + 10 − 0 = 45**. Labelled `[predicted]`; the lane MEASURES its own number and reproduces
nothing from this paragraph. RETRACTION (house rule 4): the plan's BUDGET-MONOTONE row estimates
"p7: 0 removed, ~12 added". 12 is the count of quarantined-resolving import LINES; two of them
(`d1_config_profile.py:58`, `d1_family.py:43`) rewrite to p6's landed `core.observation.*`, so the
edge delta is 10, not 12. Retracted, not overwritten.

## ACCEPTANCE

Machine-checkable, one line each. The lifecycle's `acceptance` arg points here.

- The 19 moved modules exist at the destination addresses enumerated under *Scope*, each shim pair
  regenerated by `tools/gen_shims.py` with `--verify` byte-comparing a fresh re-derivation; every
  shim's `_target` is the `ladder.register.*` / `ladder.gate.*` address, never `__legacy`.
- `git diff -M --stat` shows every moved module as a pure rename plus import-line rewrites only.
  No admission predicate, no digest literal, no schema constant changes bytes. Evidence: the
  rename-detected diff in the PR body.
- `ladder/register/rungs/` and `ladder/register/arms/` exist with a docstring-only `__init__.py`
  each and ZERO members. `ladder/register/__init__.py` and `ladder/gate/__init__.py` are
  docstring-only, no re-exports.
- **The `d1_family` + `d1_runs` SCC lands in ONE commit** (`tools/mapping.json` puts both in
  `ladder.register`; the plan's §p7 and rulings table call it one of two module-level SCCs). See
  *Guard teeth by mutation* for the measured status of the RED arm — it is a DECISION, not an
  assumed `ImportError`.
- **`experiments/d1_launch_attempts.py`'s non-PILOT/non-CONFIRMATORY refusal is TEST-PINNED.**
  MEASURED 2026-09-23: NO existing test reaches it. `grep -rn "records only pilot or confirmatory" tests/`
  → zero matches; `tests/test_d1_launch_attempt_ledger.py:73`'s `pytest.raises(ValueError, match="pilot")`
  pins line 86 (`"pilot attempt forbids registered_design_digest"`), a different branch; no test
  anywhere passes `RunClass.SMOKE` — the only third member of `RunClass`
  (`src/entropic_substrate/kernel/identity/d1_primitives.py:102`, members `SMOKE`/`PILOT`/`CONFIRMATORY`)
  — into `D1LaunchAttemptIdentity`. This phase adds `tests/test_d1_launch_attempts_refusal.py`,
  RED-first: constructing the identity with `run_class=RunClass.SMOKE` raises `ValueError` matching
  `"records only pilot or confirmatory"`. RETRACTION (house rule 4): v1's ACCEPTANCE line "its test
  still passes" and its guard-teeth row citing `experiments/d1_launch_attempts.py:70` are both
  wrong — there is no such test, and the raise is at **:71** (`:70` is the `if`). Retracted, not overwritten.
- **Cold import at the NEW address, fresh interpreter per moved leaf**
  (`tests/test_register_gate_cold_import.py`, NEW, RED-first; the subprocess-plus-stdout-marker
  instrument of `tests/test_shim_layer.py` and p6's `tests/test_core_cold_import.py`, never exit
  code alone): each of the 19 destination addresses imports cleanly with NOTHING warming the chain
  first. MEASURED at `861d015`, honestly: this is NOT a live hazard today —
  `PYTHONPATH=src uv run python -c "import entropic_substrate.__legacy.<x>"` in a fresh interpreter
  succeeds for all 19 legacy sources AND for all 10 rewritten `__legacy` targets (0 failures of 29,
  against p6's 9 of 27). The arm is landed anyway as the p6-parity regression guard; the PR body
  records the measured zero rather than claiming a hazard the measurement does not show.
- `legacy_edge_budget` in `.claude/workflows/landed-set/p7-register-gate.json` equals the MEASURED
  landed→`__legacy` edge count at the tip, measured with `tools/import_graph.py` and never typed,
  gated by the EQUALITY arm of
  `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget`.
  DIRECTION for this phase: INCREASING (predicted 45). The strict-decrease arm is RETRACTED
  (BUDGET-MONOTONE, plan line 842, landed retraction `795f8e6`) — this phase is exactly the case
  that falsified it.
- `tests/test_no_old_prefix.py` green: ZERO landed→old-prefix edges, i.e. no moved module keeps an
  `entropic_substrate.<pkg>.<leaf>` import.
- **Regime-table control (no R-PIN owed):** `git diff mythwave/p6-core -- tests/test_identity_epoch.py`
  is EMPTY — both `_CHECKPOINT_REGIME_TABLE` and `_SOURCE_REGIME_TABLE` byte-identical to the
  parent tip — and `tests/test_identity_epoch.py`, `tests/test_regime_rows.py` are green unchanged.
  Zero of the 51 checkpoint reader members and zero of the 7 source reader members are touched.
  Command output in the PR body.
- `lint-imports` green: ZERO inversions among the landed layers
  `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until
  p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds
  them instead. MEASURED premise: every non-`__legacy` import in the 19 targets resolves to
  `kernel.*` or `core.*`, both BELOW `ladder` — no inversion is possible by construction.
- `just check`, `just pc-ci` (exit gate via the runner) and `just ship-it` green on the tip;
  `gh pr view --json headRefOid` equals `git rev-parse`.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the
  PR body.
- The actual diff's file set is a subset of *Allowed files*. (PR-body evidence; there is no zip
  predicted-vs-actual check — this phase is serial.)

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for
correctness is a `DECISION`, not an edit.

- Does NOT preregister H1 or author any rung spec. `rungs/` and `arms/` are scaffolded EMPTY so
  H1's spec has a home before H1 exists.
- ZERO behaviour change to registration or admission. `RunClass.PILOT` admission logic moves
  unchanged; the new refusal test pins existing behaviour, it does not alter it.
- Does NOT touch `core/**`. p6 landed it; this phase only IMPORTS six of its addresses.
- Does NOT touch `kernel/**`, including `kernel/artifacts.py` — `ladder_predecessor.py:18`'s split
  consumes p5b's five moved names, it does not move anything.
- Does NOT move the remaining `evidence.ladder_campaign` gate half, `evidence.d1_filing*`,
  `evidence.d1_gate`, `evidence.d1_validator`, `analysis.ladder_*` or `d1_predecessor_loader`.
  They stay quarantined and are reached by explicit `__legacy` address; p8/p11/p12 own them.
- Does NOT empty or edit `__legacy/experiments/__init__.py` or `__legacy/evidence/__init__.py`,
  and does NOT touch the live old-path facades `src/entropic_substrate/{experiments,evidence}/__init__.py`.
- Does NOT re-pin any regime row, bump an identity LABEL or change a module SET.
- Does NOT edit `tools/mapping.json`, `.importlinter`, briefs, plan docs, the manifest, engine
  scripts or CI. Does NOT edit any EXISTING test.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this
envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the
verification you RAN attached (command + observed output). RETRACTED 2026-09-18 (Q1 ruled (b)):
this section previously granted "`tests/test_d1_launch_attempts.py` and the register/gate tests,
for import-path rewrite only" as a prose test grant. Struck — no phase before p13 edits EXISTING
tests; shims keep them green under rewrite-on-arrival, and the prose form cannot discharge the
ACCEPTANCE subset-of-Allowed-files check anyway.

- the 19 legacy sources enumerated under *Scope* (moved with `git mv`, import lines rewritten once
  on arrival per the *Rewrite-on-arrival* tables)
- `src/entropic_substrate/ladder/register/**` (the 16 moved modules, a docstring-only
  `__init__.py`, and the two empty scaffolded subpackages `rungs/`, `arms/`)
- `src/entropic_substrate/ladder/gate/**` (the 3 moved modules plus a docstring-only `__init__.py`)
- the 19 typed shim pairs at the old paths (`.py` + `.pyi`), regenerated by `tools/gen_shims.py`,
  never hand-edited
- `tests/test_register_gate_cold_import.py` (new, RED-first: fresh interpreter per moved leaf at
  its NEW address)
- `tests/test_d1_launch_attempts_refusal.py` (new, RED-first: the `RunClass.SMOKE` refusal at
  `d1_launch_attempts.py:71`, which MEASURED has no existing test)
- `.claude/workflows/landed-set/p7-register-gate.json` — THIS phase's disjoint fragment (the 19
  rows `landed: true` with their `ladder.*` targets, the measured `legacy_edge_budget`, and a
  `note` recording the five retractions). No other phase's fragment. `.importlinter` is NOT in this
  envelope: it is edited only at p0 and p13.
- generator outputs (D-B precedent, p5b): any `.pyi` stub whose target this phase edits, listed in
  the PR body from `git status --porcelain` after generate + `--verify`. MEASURED SCOPE: exactly the
  19 stubs above and no others. `compute_expected_stub_map` yields 322 expected stubs at the parent
  tip, one per non-retired module row; a stub is re-derived from its TARGET's AST bound names, and
  the only targets this phase edits are the 19. An import-line rewrite does not change a bound name,
  so each of the 19 stubs changes ONLY in the module path of its `from ... import X as X` lines.
  There is no "third stub" surprise.
- `.claude/workflows/landed-set/.emitted.json` is UNTRACKED derived state (FOR-1539); never
  `git add` it, never treat it as a companion of an allowed file
- `.secrets.baseline` (STANDING RULE, conductor 2026-09-19, generalizing p0's ruling: a mechanical
  companion of allowed content. detect-secrets regenerates it whenever an allowed file's digest
  literals move or change, so its diff may contain ONLY line renumbering, removal of orphaned
  entries, the `generated_at` stamp, and new `Hex High Entropy String` entries whose flagged lines
  are sha256 / git-SHA literals inside this phase's own Allowed files; the lane verifies that in
  the PR body. New digest literals in TEST files carry an inline `# pragma: allowlist secret`
  instead of a baseline entry, so the baseline stops churning; JSON records keep baseline entries.
  Never a credential.)

**Deliberately EXCLUDED from Allowed files** (they are p5b's and p6's, and no reader-set member is
touched here): `tests/test_identity_epoch.py`, `tools/regime_rows.py`, `tests/test_regime_rows.py`,
`docs/identity/read-old-table.md`. A diff touching any of them is a `DECISION`, not an edit.
`tests/test_shim_layer.py` is likewise EXCLUDED: p6 ruled it in only because its hard-coded facade
children were all p6 targets. MEASURED 2026-09-23 at `861d015`:
`grep -n "legacy_dotted\|experiments\|evidence" tests/test_shim_layer.py` names NO `experiments` or
`evidence` child (its one "evidence" hit is the word inside a docstring at `:7`), so the p6 Q-1
collision does not recur here and the p3-era rule "no phase before p13 edits existing tests" holds
unqualified.

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                        | status                                              | ruling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| serial p7 (D-WAVES r3)    | CONDUCTOR 2026-09-23                                | p7 runs as a SERIAL lifecycle on parent `mythwave/p6-core`, `liveRequired: true`, publishing through `just ship-it` inside the lifecycle. `tools/lane_slices.py` MEASURED the `p7 -> p6` import dependency (2 edges, via the `observation` facade: `d1_config_profile.py:58`, `d1_family.py:43`), and a parallel lane has no valid address for such an import in its own worktree. Every wave-lane, zip, `publishMode:'defer'`, `liveDeferred` and CERTIFIED construction in v1 is struck, including its O-1 rows and its "Wave slice rule" section. |
| FOR-1541                  | CONDUCTOR 2026-09-22                                | Publish workaround as ruled for p5, p5b and p6: branch-only `gt submit` after a green `just ship-it` whose sole failure is the commitizen-branch empty range on an up-to-date lower branch.                                                                                                                                                                                                                                                                                                                                                          |
| target set                | tools/mapping.json (ratified) + MEASURED 2026-09-23 | 19 targets = `dest ∈ {entropic_substrate.ladder.register, entropic_substrate.ladder.gate}` ∧ leaf `.py` under `__legacy/` ∧ not already landed. `entropic_substrate.experiments` is a PACKAGE KEY, excluded because `tools/gen_shims.py::merge_fragments` raises on a fragment naming one and `tools/lane_slices.py::parse_lane` exits 2. v1's "20", the plan's §p7 row and the manifest's "register (17), gate (3)" are retracted, not overwritten (house rule 4).                                                                                  |
| R-PIN (NOT OWED HERE)     | CONDUCTOR 2026-09-22 (p5, p5b, p6 precedent)        | Rewriting a `_READER_MODULES` member re-pins the affected reader's regime rows in the same commit. MEASURED 2026-09-23: this phase rewrites ZERO landed importers, and none of the 19 targets or their new addresses appears in `h-checkpoint-reader-v2`'s 51-module tuple or `h-source-reader-v3`'s 7-module tuple. **No re-pin is owed.** The four identity files stay OUT of Allowed files; the control is a byte-identity assertion against the parent tip.                                                                                      |
| BUDGET-MONOTONE retracted | CONDUCTOR 2026-09-23 (landed `795f8e6`)             | p5's arm requiring each fragment's `legacy_edge_budget` to be STRICTLY BELOW its predecessor's is RETRACTED. p7 is the falsifying case: its moved modules legitimately import still-quarantined modules, so the count RISES on arrival. The binding arm is the EQUALITY arm of `tests/test_legacy_coupling.py` against a MEASURED number; direction is predicted by the conductor and never typed into the lane's reproduction path. v1's guard-teeth row "must be STRICTLY LESS than the previous phase's" is struck.                               |
| FOR-1539                  | 2026-09-22                                          | `gen_shims`' `.emitted.json` manifest is untracked derived state; no phase commits it and it is never a companion of an allowed file.                                                                                                                                                                                                                                                                                                                                                                                                                |
| SCC                       | RULED 2026-09-17, MEASUREMENT AMENDED 2026-09-23    | `experiments.d1_family` + `d1_runs` move as ONE commit. The pair is an SCC of the FULL import graph (`tools/import_graph.py` records nested imports). AMENDED: at MODULE-INITIALIZATION level the edge is one-way — `d1_runs.py:7` imports `d1_family` at module scope, while every `d1_family` → `d1_runs` edge is deferred (`TYPE_CHECKING` at `:45`, function-local at `:177`, `:196`, `:338`). The one-commit rule stands; the claimed `ImportError` RED arm does not follow from it. See *Guard teeth by mutation*.                             |
| scaffolding               | RULED 2026-09-17                                    | `rungs/` and `arms/` are scaffolded EMPTY here so a new rung has exactly seven possible touch points and the usual rung touches two.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| destination facade        | CONDUCTOR 2026-09-23 (p4, p6 precedent)             | Destination package `__init__.py` is DOCSTRING-ONLY, no re-exports. `__legacy/experiments/__init__.py` (286 bytes, docstring + `from __future__ import annotations`, zero re-exports) and `__legacy/evidence/__init__.py` stay VERBATIM, untouched; so do both live old-path facades. The `entropic_substrate.experiments` mapping row describes the p13 end state.                                                                                                                                                                                  |
| shim form                 | RULED 2026-09-17, CORRECTED 2026-09-23              | `import entropic_substrate.<new.landed.address> as _target` then `sys.modules[__name__] = _target`, as the generator emits it. v1's *Already ruled* row said `entropic_substrate.__legacy.<x>` as the `_target` — the pre-p3 form, WRONG for a phase that lands a destination; retracted, not overwritten (house rule 4). A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                                                                                                   |
| rewrite-on-arrival        | RULED 2026-09-17                                    | A file's import lines are rewritten exactly once, by the phase that moves its target. This phase never rewrites an importer outside its own touch set — and MEASURED, no such importer exists.                                                                                                                                                                                                                                                                                                                                                       |
| P-1                       | RULED 2026-09-17                                    | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                                                                                                                                                                                                                                                                                                                                    |
| E-4                       | RULED 2026-09-17                                    | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                                                                                                                                                                                                                                                                                                                                   |
| skeleton                  | RULED 2026-09-17                                    | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical < composition`.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Q-1 SCC arm               | CONDUCTOR 2026-09-23                                | MEASURED: `d1_family`/`d1_runs` is acyclic at module-initialization time (`d1_runs.py:7` is the only module-scope edge; the reverse edges are TYPE_CHECKING or function-local), so the plan's "collection-time ImportError" tooth was never load-bearing and is RETRACTED (house rule 4). The pair still moves in ONE commit as a granularity rule; the guard row is dropped, not claimed.                                                                                                                                                           |
| Q-2 refusal test          | CONDUCTOR 2026-09-23                                | `tests/test_d1_launch_attempts_refusal.py` is p7's, RED-first, in Allowed files; the raise is at `d1_launch_attempts.py:71`.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Q-3 cold-import arm       | CONDUCTOR 2026-09-23                                | Zero cold-import failures measured for the 29 addresses; `tests/test_register_gate_cold_import.py` stays as an ACCEPTANCE instrument (regression at the new addresses), not a guard-teeth row.                                                                                                                                                                                                                                                                                                                                                       |
| Q-4 empty packages        | CONDUCTOR 2026-09-23 (p0/p4 marker precedent)       | `ladder/register/{rungs,arms}/__init__.py` are docstring-only markers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Q-6 read-old table        | CONDUCTOR 2026-09-23                                | No re-pin is owed (zero reader-set members in the slice); no table row is written; the control is `git diff mythwave/p6-core -- tests/test_identity_epoch.py` empty, quoted in the PR body.                                                                                                                                                                                                                                                                                                                                                          |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test
goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never
reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                                                               | must go RED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| relax `d1_launch_attempts.py:71`'s non-PILOT/non-CONFIRMATORY refusal (drop `RunClass.SMOKE` from the rejected set)    | `tests/test_d1_launch_attempts_refusal.py` (NEW, RED-first). MEASURED: no existing test reaches this branch, so without the new test this mutation is INVISIBLE — that is why the test is in Allowed files.                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| leave one moved leaf importing a sibling by its OLD dotted name (`entropic_substrate.experiments.d1_family`)           | `tests/test_no_old_prefix.py` (landed→old-prefix edge count ≠ 0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| drop one rewritten `__legacy` import, or point it at a name the quarantined module no longer exports                   | `tests/test_legacy_coupling.py` equality arm (measured edges ≠ committed budget, off by one) and `tests/test_register_gate_cold_import.py`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| point one regenerated shim's `_target` at the `__legacy` address instead of the `ladder.*` address                     | `tests/test_gen_shims.py` (`--verify` byte-compare of a fresh re-derivation) and the layers contract                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| add a re-export to `ladder/register/__init__.py`, `ladder/gate/__init__.py`, `rungs/__init__.py` or `arms/__init__.py` | the destination-facade contract (docstring-only); the diff-review arm, plus `lint-imports` if the re-export crosses a layer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| hand-edit a regime-table digest instead of leaving the tables untouched                                                | `tests/test_identity_epoch.py::TestComposedDigestsPerMeasuredRegime` (and the byte-identity control against `mythwave/p6-core`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| split the `d1_family` / `d1_runs` SCC across two commits                                                               | **DECISION, not an asserted `ImportError`.** v1 claimed "collection-time `ImportError`". MEASURED 2026-09-23: at module-init the pair is ACYCLIC (`d1_runs.py:7` → `d1_family` at module scope; all four reverse edges deferred — `TYPE_CHECKING` `:45`, function-local `:177`/`:196`/`:338`), and the old-path shims keep both importable during a split, so the predicted collection error may never fire. The lane must EXHIBIT the RED arm it claims — a named failing test or a captured collection error from the actual split — or RETRACT this row in the PR body with the measurement, per house rule 4. It must not report the mutation as proven from the plan's wording. |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change
class permitted inside a structural phase, scoped to log routing, metric routing and entry-point
configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the
identity fixtures and the numerics tests are the tripwire.

- **Nothing to route, measured.** MEASURED 2026-09-23 across all 19 legacy sources: ZERO `print(`
  sites, ZERO `argparse` references, ZERO `__main__` guards, ZERO `def main(`, ZERO
  `configure_logging` calls, and ZERO `logging`/`logger` uses — the single `logging` token in the
  corpus is the word "logging" inside `d1_execution.py:222`'s docstring. The rewrite-on-arrival
  print obligation and the entry-point obligation both have an EMPTY SUBJECT here.
- No entry point moves, so no `conforming_entrypoints` change is claimed. A claim of entry-point
  conformance would need its own measurement. v1's "Any `python -m` entry point or CLI subcommand
  this phase moves calls `configure_logging` at ENTRY" is retained as a conditional with no
  instances, not as a task.
- Admission REFUSALS keep their exception text, which is the contract — `d1_launch_attempts.py:71`'s
  message is now pinned by a test, not only by the diff.
- No metric emission originates in register or gate (MEASURED: no `MetricSink` reference in the 19
  sources), so no sink construction site is created here.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm —
hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.
The id→content mapping below is ASSUMED from v1's 1..6 ordering: the Linear issue titles behind
FOR-1458..1463 were NOT read when this brief was drafted (no Linear access in that session). Confirm
before the design gate — `p7-notes.md` Q-8.

1. FOR-1458 Land `ladder/register` with the `d1_family` + `d1_runs` SCC as ONE commit (both
   modules, both import sets rewritten together)
2. FOR-1459 Land the remaining `ladder/register` value and spec modules (`d1`, `d1_config`,
   `d1_config_profile`, `d1_confirmatory_matrix`, `d1_execution`, `d1_plan`, `d1_provenance`,
   `d1_registered_design`, `d1_stabilization`, `h_preservation`, `ladder`) with
   `tests/test_register_gate_cold_import.py` RED-first
3. FOR-1460 Scaffold `ladder/register/rungs/` and `ladder/register/arms/` EMPTY (docstring-only
   `__init__.py` each)
4. FOR-1461 Land `ladder/register`'s codecs (`d1_launch_attempt_codec`, `d1_registration_codec`)
   and `d1_launch_attempts` with `tests/test_d1_launch_attempts_refusal.py` RED-first; fix the
   `:70` → `:71` line cite and retract v1's "its existing test still passes"
5. FOR-1462 Land `ladder/gate` (`d1_family_adapter`, `d1_verdict`, `ladder_predecessor`) — the ONLY
   commit that moves the `legacy_edge_budget`: 10 new explicit `__legacy` edges, including
   `ladder_predecessor.py:18`'s split across `kernel.artifacts` and the quarantined
   `ladder_campaign` gate half
6. FOR-1463 Fragment with the measured `legacy_edge_budget`, the regime-table byte-identity control
   (no R-PIN owed), gates, and the five retractions
