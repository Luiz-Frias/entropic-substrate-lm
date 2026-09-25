# p12-historical-composition — `historical/{e_series,d_ladder}`, `composition/`; `__legacy` empty

Authoritative brief for one `td-phase-lifecycle.cascade` lifecycle. **v5**, drafted 2026-09-24 (v3 + verifier round-2 and round-3 fixes by the conductor),
superseding v2 (same date) after verifier round 1 returned FAIL on one MAJOR plus four minors
(`.../scratchpad/p12/verify-1.json`). **RETRACTED, one sentence, house rule 4:** v2's P12-SCC
section and its ACCEPTANCE topological-check bullet both claimed the 58-edge residue graph was
"re-derived independently by this session... over the full residue edge graph." That was false —
v2 took `targets.json`'s precomputed `legacy_importers` field at face value; it did not itself parse
any source file. The verifier's own independent AST parse of all 44 residue files found five real
edges from `entropic_substrate.cli` (`d1_cli`, `d1_cli_composition`, `d_ladder_cli`,
`d_ladder_probe_cli`, `analysis`) missing from that field. **This v3 draft performs the actual
re-derivation** the retracted sentence falsely claimed — see *P12-SCC* and *Scope — Intra-set import
graph* below — confirming the verifier's finding exactly (5 missing edges, all from `cli.py`, zero
false positives in `targets.json`'s field) and extending it to a full, exhaustive count. Four minor
corrections (composition/cli.py's line count, a TYPE_CHECKING line range, an incomplete test
citation, a missing GAPS entry about the plan document's own stale figures) are applied inline,
each marked CORRECTED. Two new conductor rulings (P12-LEGACY-EMPTY-DEF, revised; P12-RETIRED-CANARY,
new) are added to *Already ruled*. Everything the verifier's `confirmed` list checked is carried
forward byte-identical — nothing there is re-derived or restated differently. v1 (dated
2026-09-17) remains superseded for the reasons v2 already gave. Plan:
`docs/plans/src-hexagonal-refactor.plan.md` (§p12, conductor-rulings table). The lifecycle's
`acceptance` arg reads "Per the brief's ACCEPTANCE section"; that section is below.

**Epoch.** Every `[MEASURED]` count below was taken at `8e5ef7df36c1ccd27ce01d679b19363ebc72834f`
(p10 tip, PR #302), read-only via `git show 8e5ef7d:<path>` / `git grep <pat> 8e5ef7d -- <paths>` /
`git ls-tree -r --name-only 8e5ef7d <dir>`. No file in the repo was written and no git-state command
was run. **Correction to v2's blanket claim:** this v3 draft DID execute code — a local, read-only
`python3` script (`ast.parse` over each of the 44 residue files' `git show 8e5ef7d:<path>` text,
walking `Import`/`ImportFrom` nodes tagged by lexical context: module-level, `TYPE_CHECKING`-guarded,
function-local) — matching the coordinator's own instruction that "a python3 ast walk over `git
show` output is fine and cheap." This is analysis of already-fetched text, not execution of
repository code, a repo command, or any git-state mutation; it is the SAME class of activity v1/v2's
Tarjan-by-hand arithmetic already performed, just automated and exhaustive instead of manual and
partial. `[predicted]`: arithmetic on measured inputs, never independently reproduced. `[ASSUMED]`:
a reading this session adopted where the ruling text itself is ambiguous or was itself found to
contain an error; flagged for operator ratification, never silently treated as settled. Evidence
classes are never commingled in one row without a label (epistemic-hygiene rule 3).

## Branch

| field                                        | value                                                                                                                                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p12-historical-composition`                                                                                                                                                    |
| parent                                       | `mythwave/p10-run-upper` @ `<startTip>` (**P12-PARENT**, conductor 2026-09-24; the lane re-reads its own launch tip, never a typed SHA baked into this file)                             |
| mode (sprint manifest, parallel eligibility) | `ratchet`                                                                                                                                                                                |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived — feeding the manifest `mode` here yields a silent `normal` and no Opus design gate.                                                              |
| `engine_mode`                                | `arch`                                                                                                                                                                                   |
| wave                                         | `serial`, post-p10, post-w2. Not a wave lane: no lane worktree, no zip, no `publishMode:'defer'`.                                                                                        |
| `gates.liveRequired`                         | `true`                                                                                                                                                                                   |
| live gate                                    | `just ship-it` runs INSIDE the lifecycle (house rule 1). FOR-1541 branch-only fallback as used by every serial phase this sprint.                                                        |
| gates (lifecycle arg)                        | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}`                                                                                          |
| launch                                       | SERIAL phase: the conductor launches the repo's ported cascade DIRECTLY. Not a wave member.                                                                                              |
| `repoToplevel`                               | `/Users/luizfrias/CursorAI/entropic-substrate-lm`. The MAIN checkout, never a worktree. Launch only when quiescent: no publish or pre-push hook running, `git status --porcelain` empty. |
| trunk                                        | `refactor/hexagonal-redesign`                                                                                                                                                            |

## Contract surface owned

| contract                                                                                                                                                                                                               | kind     | frozen_after |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `historical/e_series` (`train.py`, `field_pretrain.py`) :: governed READ-ONLY, zero behaviour change                                                                                                                   | contract | p13          |
| `historical/d_ladder` :: the 11 `historical.d_ladder`-dest modules (per mapping.json, corrected count — see *Scope*) plus the `d_ladder_worker_runtime` remainder, RUNNABLE, same governance                           | contract | p13          |
| `composition/{wiring,entrypoints}` and `composition/entrypoints/cli.py` (the E/D argparse root; **NOT** `composition/cli.py` — that path is occupied, see *P12-CLI-ROOT*) :: the one composition root for E/D adapters | contract | p13          |
| `ladder/gate/ladder_campaign.py` :: the `evidence.ladder_campaign` gate half (**P12-GATE-HALF**) — an EXTENSION to the already-existing, already-landed `ladder/gate` package, not a new contract surface              | contract | p13          |
| `__legacy/` EMPTY postcondition                                                                                                                                                                                        | door     | p12          |

p12 owns the fail-closed allowlist literals for `d1_trainer_mailbox_command` and
`d_ladder_worker_command` only (FOR-1488, re-homed here — see *P12-ALLOWLISTS*); it does not own
`h_probe_launch.py`'s or `ladder/review/d1_external_analysis_adapter.py`'s own module content
(both already landed, neither a p12 target), only their one literal line each.

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item
below is LANDED before this lane launches.

- every landed address (p0-p11 + w2-reconcile + p10)
- the parameterized entrypoint source, `scripts/entrypoints.env` (p0, ruling E-4) — p12 EDITS its
  six values in commit 6 (see *Planned commit branches*), it does not author the mechanism
- `tools/gen_shims.py` (+ `--verify`, `--preflight`), `tools/import_graph.py`, `tools/regime_rows.py`
- `tests/test_legacy_coupling.py`, `tests/test_no_old_prefix.py`, `tests/test_module_identity.py`,
  `tests/test_cli_surface.py` (READ-ONLY consumed control except the one door-amendment class named
  under *P12-ALLOWLISTS* — every other assertion in the file stays unedited)
- `.claude/workflows/landed-set/p10-run-upper.json`, `legacy_edge_budget: 5` `[MEASURED]` at
  `8e5ef7d` — this is p12's starting baseline, re-derived by this session by reading the field
  directly rather than assumed; the fragment's own `note` records it as "p10's FINAL fragment value"
  across its four commits (32 → 13 → 12 → 7 → 5)
- `ladder/gate/ladder_predecessor.py`, `ladder/analyze/ladder_analysis.py`,
  `ladder/file/ladder_filing.py`, `ladder/run/worker/d1_predecessor_loader.py`,
  `ladder/run/worker/resource_registration.py` — the five files this phase edits ONLY on the
  import lines named under *Scope* (three already existing `evidence.ladder_campaign` imports, one
  `d1_predecessor_loader`-owned partial import, one worker-split leaf digest/provider migration)

## Sacred invariants

Quoted verbatim from `CLAUDE.md`. `touch_set ∩ core/** = ∅` for every file this phase edits (the
core-layer facades this phase touches are `__legacy/`-physical DUPLICATES only — see *P12-FACADES* —
never `src/entropic_substrate/core/**` itself).

| id   | verbatim invariant                                                                                                          | tripwire / guard                                                                                       |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | `git diff <startTip> -- src/entropic_substrate/core tests/test_kappa_autocast_island.py` must be EMPTY |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | same empty diff over `core/geometry`                                                                   |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | same empty diff over `core/substrate`                                                                  |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | same empty diff over `core/sentence`                                                                   |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | same empty diff over `core/latent`                                                                     |

House rules HR-1..HR-7 bind exactly as tabled in `.claude/workflows/briefs/p9-run-lower.md` and
`p10-run-upper.md`, unchanged.

## Scope

**P12-RESIDUE (conductor 2026-09-24, CORRECTED one line by this session — house rule 4).** The
ruling's own text lists "the 11 package/root facades (entropic_substrate root **init** 12L,
analysis 21L, evidence 8L, experiments 8L, predictive_state, geometry, observation, init, models,
data, sentence_prediction, layers one-liners)" as part of "the 13 previously unowned." **[MEASURED]**
counting mapping.json rows: the root `entropic_substrate` row (dest `entropic_substrate.composition`,
12 lines) is ALREADY counted inside the original in-scope `composition` group's 18 modules / 4585
lines (verified by summing all 18 composition-dest rows' line counts: they total exactly 4585 only
when the root row is included). It is not an addition. The ruling names 12 facade identifiers but
labels them "11" — the arithmetic ("13 previously unowned") is still correct, because the true
enumeration is 2 (gate half + worker-runtime remainder) + 11 genuine facades (analysis, evidence,
experiments, predictive_state, geometry, observation, init, models, data, sentence_prediction,
layers — root EXCLUDED) = 13, 729 lines, independently reconciling exactly with the original audit's
"13 out-of-scope, 729 lines" table. Root's *treatment* still resembles a facade (see below), but its
*count* stays inside the original 31/8289 split. Restated cleanly:

| slice                                                                                                                                                       | modules | lines    | instrument                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| composition (dest `entropic_substrate.composition`, incl. root `entropic_substrate` 12L and `entropic_substrate.__main__` 8L)                               | 18      | 4585     | `[MEASURED]` sum of `wc -l` per file, `tools/mapping.json` dest filter                                                                                                                                                                                                                                                                       |
| historical.d_ladder                                                                                                                                         | 11      | 2300     | `[MEASURED]` same method                                                                                                                                                                                                                                                                                                                     |
| historical.e_series                                                                                                                                         | 2       | 1404     | `[MEASURED]` same method                                                                                                                                                                                                                                                                                                                     |
| **original in-scope subtotal**                                                                                                                              | **31**  | **8289** | matches v1's Scope prose numerically for the first time — v1's own "57 modules" and "11 d_ladder + 2 probes" claims were both CONTRADICTED by the audit (35 true composition/historical-dest rows; the 11 `historical.d_ladder` rows already INCLUDE `analysis.d1_probe`/`analysis.ladder_probe`, not two rows in addition to a separate 11) |
| gate half (`evidence.ladder_campaign`, dest `kernel.artifacts` in mapping.json, SPLIT — see *P12-GATE-HALF*)                                                | 1       | 371      | `[MEASURED]`                                                                                                                                                                                                                                                                                                                                 |
| `d_ladder_worker_runtime` remainder (`build_worker_factory`, `verify_worker_source`; dest `ladder.run` in mapping.json, SPLIT — see *P12-WORKER-REMAINDER*) | 1       | 98       | `[MEASURED]`                                                                                                                                                                                                                                                                                                                                 |
| 11 genuine package facades (dest `core.*` / `ladder.{analyze,capture,register}` — see *P12-FACADES*)                                                        | 11      | 260      | `[MEASURED]`                                                                                                                                                                                                                                                                                                                                 |
| **previously-unowned subtotal**                                                                                                                             | **13**  | **729**  | reconciles exactly with the audit's independent count                                                                                                                                                                                                                                                                                        |
| **TOTAL p12 residue**                                                                                                                                       | **44**  | **9018** | matches the audit's grand total exactly                                                                                                                                                                                                                                                                                                      |

**Six E/D CLI subcommands, CORRECTED (house rule 4, repeats an audit finding this brief must
carry forward since v1 asserted the uncorrected number in its own ACCEPTANCE and Already-ruled
table).** `[MEASURED]` tracing every subcommand handler in `__legacy/cli.py` for an import of a
residue historical-dest module: only **five** — `train`, `eval` (via `Trainer`), `field-pretrain`,
`d-probe`, `d-score-gain`. `init-fisher` (the obvious sixth candidate) calls `build_fisher_init`
from `entropic_substrate.init.fisher`, **already landed** at `core.substrate.fisher` (p6) — it does
not touch `historical/` at this epoch. DN-24 (ruled 2026-09-17, before p6 landed) is stale on this
one count; the underlying door — E0-E1.5c replay stays possible through `historical/` — is
unaffected, since all five real subcommands are inside p12's own `historical/{d_ladder,e_series}`
scope and none is deleted. ACCEPTANCE below states five, not six.

### P12-GATE-HALF

**RULED by operator 2026-09-24, FOR-1557.** `evidence.ladder_campaign` (371 lines, physically
`src/entropic_substrate/__legacy/evidence/ladder_campaign.py`) carries the gate half left behind
by p5b's split (artifact half landed at `kernel.artifacts`, p5b/p7) and p7's further extraction
(`d1_family_adapter`/`d1_verdict`/`ladder_predecessor` moved to `ladder.gate`, p7). What remains —
`RunAuthorization`, `SuccessorAuthorization`, `FiledLadderGate`, `validate_ladder_phase_transition`,
`gate_from_d1_filing`, `read_campaign_design`, `verify_ladder_registration`, `authorize_ladder_run`,
`actor_from_mapping` — lands at `src/entropic_substrate/ladder/gate/ladder_campaign.py`, an ADDITION
to the already-existing, already-landed `ladder/gate` package (`__init__.py`, `d1_family_adapter.py`,
`d1_verdict.py`, `ladder_predecessor.py` `[MEASURED]`), not a new subpackage.

**Precondition, not p12's own edit.** `tools/mapping.json`'s row for `entropic_substrate.evidence.
ladder_campaign` currently carries a SINGLE `dest: entropic_substrate.kernel.artifacts` with a note
saying the gate half was earmarked for `ladder.gate.campaign` and never executed. The door
amendment — adding the second destination `entropic_substrate.ladder.gate.ladder_campaign` for the
gate half — is applied by the conductor in the gap commit that installs this brief, BEFORE p12
launches (stated here as a precondition p12 verifies exists at launch, never as p12's own diff:
`mapping.json` stays read-only inside p12's own Allowed files).

**Landed importers rewritten, 4 lines, `[MEASURED]`:**

| importer                                        | current line                                                                                                                                                                                                                                                                                                                                                                                      | rewrite                                                                           |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `ladder/gate/ladder_predecessor.py:8`           | `import entropic_substrate.__legacy.evidence.ladder_campaign as ...`                                                                                                                                                                                                                                                                                                                              | → `entropic_substrate.ladder.gate.ladder_campaign`                                |
| `ladder/analyze/ladder_analysis.py:17`          | same shape                                                                                                                                                                                                                                                                                                                                                                                        | → same target                                                                     |
| `ladder/file/ladder_filing.py:12`               | same shape                                                                                                                                                                                                                                                                                                                                                                                        | → same target                                                                     |
| `ladder/run/worker/d1_predecessor_loader.py:12` | partial import of `FiledLadderGate`, `actor_from_mapping`, `gate_from_d1_filing` ONLY (3 of the gate half's 9 names; the other 4 names this file needs — `artifact_from_mapping`, `object_fields`, `read_bound_artifact`, `read_json_artifact` — already resolve to the LANDED `kernel.artifacts` and are untouched) `[MEASURED]`, per p10's own fragment note documenting this split-import case | → the 3-name import retargets to `entropic_substrate.ladder.gate.ladder_campaign` |

**Facade retraction rides this commit.** `src/entropic_substrate/__legacy/evidence/__init__.py`
(8 lines) is, **contrary to the general "1-line placeholder" pattern this session found does NOT
hold uniformly** (see *P12-FACADES*), a byte-identical FULL duplicate of the permanent top-level
`src/entropic_substrate/evidence/__init__.py`. Once `ladder_campaign.py` (the directory's only
other leaf) lands, `__legacy/evidence/` holds zero leaves — retract the duplicate `__init__.py` in
THIS SAME commit (FACADE-EMPTY) and remove the now-empty directory. The real facade content stays,
untouched, at the top-level address forever; nothing imports the `__legacy`-prefixed name
(`[MEASURED]`, zero hits).

**Budget.** `[MEASURED]` baseline at `8e5ef7d`: 5 landed→legacy edges (4 to `evidence.ladder_campaign`

- 1 to `d_ladder_worker_runtime`, per `targets.json`'s own `landed_importer_detail`). This commit
  discharges the 4 `ladder_campaign` edges. `[predicted]` post-commit budget: **1** (the remaining
  `d_ladder_worker_runtime` edge, discharged in commit 3). The lane measures live per BUDGET-PER-COMMIT.

### P12-WORKER-REMAINDER

`d_ladder_worker_runtime`'s remainder (`build_worker_factory`, `verify_worker_source`, 98 lines —
`measure_hardware`/`verify_source_tree`/`SOURCE_PREFIXES` already moved at p4; `worker_resource_
registration` already moved alone at p10, per ruling D-E) lands at
`historical/d_ladder/d_ladder_worker_runtime.py`.

**Layer-inversion fix, read in full at `8e5ef7d`.** `ladder/run/worker/resource_registration.py`
(landed p10) currently does a FUNCTION-LOCAL import:
`from entropic_substrate.__legacy.d_ladder_worker_runtime import (_FACTORY_DIGEST, _PROVIDER)`.
Once the remainder lands at `historical.d_ladder`, that import would become `ladder → historical`
— an INVERSION under the layer contract `kernel < core < ladder < historical < composition`. Per
the conductor ruling: `_FACTORY_DIGEST`/`_PROVIDER` (and the `canonical_digest({...})` computation
that builds `_FACTORY_DIGEST`, currently importing `experiments.d1_primitives.canonical_digest`,
landed at `kernel.identity.d1_primitives`) MIGRATE into `resource_registration.py` in THIS commit.
The landed remainder then imports them BACK from `ladder.run.worker.resource_registration`
(`historical → ladder`, downward, allowed). Both files are edited in the SAME commit (a
direction-flip, not an ordinary one-way rewrite-on-arrival) — flagged prominently because it is the
one edit in this phase to an ALREADY-LANDED file outside `historical/`/`composition/`.

**The remainder's own 10 import lines, all needing rewrite-on-arrival as IT moves (`[MEASURED]`, read
in full):**

| import in the remainder today                        | landed address `[MEASURED]`                                                                                                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entropic_substrate.d1_artifact_snapshot`            | `entropic_substrate.kernel.artifacts` (flat-module merge)                                                                                                                             |
| `entropic_substrate.d1_predecessor_loader`           | `entropic_substrate.ladder.run.worker.d1_predecessor_loader` (p10)                                                                                                                    |
| `entropic_substrate.d1_resource_activation`          | `entropic_substrate.ladder.run.resource.d1_resource_activation`                                                                                                                       |
| `entropic_substrate.d1_trainer_mailbox_contracts`    | `entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts`                                                                                                                |
| `entropic_substrate.d1_trainer_object_transport`     | `entropic_substrate.ladder.run.transport.d1_trainer_object_transport`                                                                                                                 |
| `entropic_substrate.d_ladder_remote_request`         | lands in THIS SAME commit, `historical.d_ladder.d_ladder_remote_request` (intra-set sibling)                                                                                          |
| `entropic_substrate.evidence.resource_meter_runtime` | `entropic_substrate.ladder.capture.resource.resource_meter_runtime`                                                                                                                   |
| `entropic_substrate.evidence.resource_provenance`    | `entropic_substrate.kernel.evidence_class.resource_provenance`                                                                                                                        |
| `entropic_substrate.experiments.d1_primitives`       | `entropic_substrate.kernel.identity.d1_primitives` (migrates into `resource_registration.py`, see above — the remainder no longer needs this import once `_FACTORY_DIGEST` moves out) |
| `entropic_substrate.paths`                           | `entropic_substrate.kernel.runtime.paths`                                                                                                                                             |

The `ladder.run` package-level dest names for the three `d1_resource_activation`/`d1_trainer_
mailbox_contracts`/`d1_trainer_object_transport` rows are `[MEASURED]` resolved to exact leaf
files by `git grep -l` on their defined class names, not derived from `mapping.json`'s
package-level `dest` field alone (per `tools/gen_shims.py`'s own docstring: the generator never
derives an exact leaf name).

**GAP, stated plainly.** This table is a full accounting for the ONE file this session read in
full (98 lines). A byte-exact import audit of the other 43 residue files (≈8900 more lines) was NOT
performed by this drafting session — no code execution or bulk-file-read budget covers that here,
and the exemplar briefs (p9, p10) that DID perform it were drafted against sets an order of
magnitude smaller. **The lane runs its own `tools/import_graph.py` sweep (or equivalent grep) per
commit, live, as the authoritative enumeration** — DATA-BUILD-RUNG precedent: measure, never
assume zero. Every commit's own landed-importer table in *Planned commit branches* below is
therefore `[predicted]` unless explicitly marked `[MEASURED]`.

**Zero landed (non-`__legacy`) importers of the remainder besides `resource_registration.py:13`
exist at `8e5ef7d`.** `[MEASURED]`, `targets.json`'s own `landed_importer_detail`.

### P12-SCC

**Instrument, `[MEASURED]` by THIS session (v3), corrected from v2's false claim of the same.** A
local `python3` script fetched `git show 8e5ef7d:<path>` for each of the 44 residue files, parsed
each with `ast.parse`, and walked every `Import`/`ImportFrom` node tagging it by lexical context
(inside a `FunctionDef`/`AsyncFunctionDef` → function-local; inside `if TYPE_CHECKING:` or `if
typing.TYPE_CHECKING:` → TYPE_CHECKING-guarded; neither → module-level), keeping only targets whose
dotted name (after stripping an `entropic_substrate.__legacy.` infix back to the bare
`entropic_substrate.` prefix, matching the shim convention) equals another residue module's own old
name. **Result: 63 distinct intra-residue edges** (up from v2's uncritically-copied 58), of which
**61 are RUNTIME edges** (at least one occurrence is module-level or function-local, i.e. actually
executes) and **2 are TYPE_CHECKING-only** (every occurrence of that edge is guarded, so it never
executes). Diffing against `targets.json`'s `legacy_importers` field: **zero false positives, five
false negatives, all five from `entropic_substrate.cli`** — `cli → d1_cli` (lines 34, 306, 1059;
`TYPE_CHECKING` at 34, function-local at 306/1059), `cli → d1_cli_composition` (line 1064,
function-local), `cli → d_ladder_cli` (lines 282, 1093, function-local), `cli → d_ladder_probe_cli`
(lines 286, 1097, function-local), `cli → analysis` (lines 666, 880, 902, 1031, function-local) —
exactly matching the verifier's independent finding. `targets.json`'s field is therefore
INCOMPLETE, not wrong, and only for one file.

`d_ladder_artifacts` ↔ `d_ladder_execution` land as ONE commit (the plan's "second module-level
SCC"). Tarjan over the full 63-edge graph: **exactly one non-trivial SCC**, `{d_ladder_artifacts,
d_ladder_execution}` — unchanged from v2's conclusion despite the corrected edge count, because
none of the 5 newly-found edges touches either module. Tarjan over the 61-edge RUNTIME-ONLY graph:
**zero cycles** — the runtime graph is a pure DAG.

**Both TYPE_CHECKING-only edges, `[MEASURED]`, corrected line range (house rule 4 — v2 cited
`d_ladder_artifacts.py:17-20`; the actual guard is one line, the import is the next):**
`d_ladder_artifacts.py:18` is `if TYPE_CHECKING:`; `:19` is
`from entropic_substrate.d_ladder_execution import SmokeRequest` (the SCC-forming return edge). A
SECOND `TYPE_CHECKING`-only edge this session's exhaustive parse found and v2 never named:
`d_ladder_artifacts.py:20`, `from entropic_substrate.d_ladder_request import ScientificRequest` —
this does NOT form a new cycle (`d_ladder_request` has no runtime or `TYPE_CHECKING` edge back to
`d_ladder_artifacts`; its only outgoing edge is to `evidence.ladder_campaign`), and both modules
land in the same commit (3) regardless, so it changes no plan decision — named here only because
"re-derived independently" must mean exhaustive, not "found the one edge v1/v2 already knew about."

**CONTRADICTED, carried forward from the audit (house rule 4 — v1's own guard-teeth row is wrong
and this correction is load-bearing, not cosmetic).** `d_ladder_execution.py:21` does a real,
module-level `from entropic_substrate.d_ladder_artifacts import ...` (the runtime edge).
`d_ladder_artifacts.py:18-19` does the RETURN edge only inside `if TYPE_CHECKING:` — it never
executes. **The runtime-only edge graph has ZERO cycles** (now independently confirmed over the
full 61-edge runtime graph, not just this one pair). Splitting the two files across separate
commits would NOT raise a `pytest`-collection-time `ImportError`, contra v1's guard-teeth row and
contra `docs/plans/src-hexagonal-refactor.plan.md:644`'s own master Guard-teeth table row ("p7, p12
| split an SCC pair across commits | collection-time `ImportError`") — that plan-document row is
ALSO stale; see *GAPS*. Fixed instrument, per EXHIBIT-OR-RETRACT: `basedpyright` (part of `just check`) resolves the
`TYPE_CHECKING`-only import and DOES fail on a missing forward reference if the two are split —
the guard tooth is `just check`'s typecheck step, not pytest collection. See *Guard teeth by
mutation*.

### P12-FACADES (FACADE-EMPTY applied, with a MEASURED correction to the audit's own claim)

**The audit's §1 states: "`__legacy/<pkg>/__init__.py` is a 1-line placeholder in every one of
these 11 cases." This session re-measured all 11 facades directly (`diff` against the top-level
file) and found this FALSE for 4 of them (house rule 4 — this is a genuine correction to the audit,
not merely to v1):**

| facade (old dotted name) | mapping.json dest        | `__legacy` copy                                                  | top-level copy | classification             |
| ------------------------ | ------------------------ | ---------------------------------------------------------------- | -------------- | -------------------------- |
| `geometry`               | `core.geometry`          | 1 line, "Retracted facade body" placeholder                      | 29 lines       | TRUE placeholder           |
| `init`                   | `core.substrate`         | 1 line, same pattern                                             | 23 lines       | TRUE placeholder           |
| `layers`                 | `core.geometry`          | 1 line, same pattern                                             | 7 lines        | TRUE placeholder           |
| `models`                 | `core.substrate`         | 1 line, same pattern                                             | 31 lines       | TRUE placeholder           |
| `observation`            | `core.observation`       | 1 line, same pattern                                             | 24 lines       | TRUE placeholder           |
| `data`                   | `ladder.capture` (SPLIT) | 1 line, same pattern                                             | 33 lines       | TRUE placeholder           |
| `predictive_state`       | `ladder.run` (SPLIT)     | 1 line, same pattern                                             | 75 lines       | TRUE placeholder           |
| `analysis`               | `ladder.analyze`         | **21 lines, BYTE-IDENTICAL to top-level**                        | 21 lines       | duplicate, never retracted |
| `evidence`               | `ladder.capture`         | **8 lines, BYTE-IDENTICAL**                                      | 8 lines        | duplicate, never retracted |
| `experiments`            | `ladder.register`        | **8 lines, BYTE-IDENTICAL**                                      | 8 lines        | duplicate, never retracted |
| `sentence_prediction`    | `core.sentence`          | **1 line, BYTE-IDENTICAL** (content is trivially 1 line already) | 1 line         | duplicate, never retracted |

**Why this matters for p12's own commits.** For the 7 TRUE placeholders, retracting the `__legacy`
copy is a pure deletion of dead weight (1 line each) with zero content loss — the "restored,
verbatim, at `entropic_substrate.<pkg>`" text in each placeholder's own docstring already documents
that the real content permanently lives OUTSIDE `__legacy/`, by the p3-quarantine ruling (moving
it would re-enter sibling shims mid-import, `ImportError` on partial init — measured 2026-09-20,
design-gate `wf_43ef5138-4bc`). For the 4 duplicates, retracting the `__legacy` copy is the SAME
operation (delete a redundant, never-imported-by-dotted-`__legacy`-prefix copy) but the deleted
content happens to be non-trivial (up to 21 lines) rather than a 1-line stub — no different in kind,
only in byte count. **None of the 11 facades' TOP-LEVEL content is touched, moved, or deleted by
p12** — only the physically-`__legacy`-resident duplicate. This keeps p12 inside its own
`historical/`/`composition/` contract surface: deleting a file physically under `__legacy/` is
squarely the "`__legacy/` empty" door's own business, never a write to `core/`, `ladder/`, or any
layer p12 doesn't own.

**Root's structural kinship to a facade, stated once (not a count change — see *Scope* above).**
`src/entropic_substrate/__init__.py` (the real, permanently-necessary package bootstrap file) is
byte-identical to `__legacy/__init__.py` (12 lines, `__version__` re-export). Exactly like the 11
facades, it can never be literally `git mv`'d to `composition/__init__.py` while remaining
importable as `entropic_substrate` — a package's own `__init__.py` cannot be shim-redirected before
it executes. p12's action on this row is identical in kind to the 11 facades: retire the `__legacy`
duplicate, leave the real top-level file untouched. `entropic_substrate.__main__` (8 lines) is
DIFFERENT and genuinely relocates — `python -m entropic_substrate` resolves `__main__.py` through
the SAME generated top-level shim mechanism already proven working for every other entrypoint
(`[MEASURED]`, its current shim already redirects to `__legacy.__main__` and contains a `runpy`
special-case for `if __name__ == "__main__"` — this generalizes to any new target unchanged), so
`__main__.py` lands at `composition/entrypoints/__main__.py` as an ordinary move (see commit 5).

**Per-commit facade timing.** `evidence` retracts in commit 1 (its directory empties there).
`experiments` retracts in commit 3 (`ladder_profile.py` is its only other leaf, lands there).
`analysis`'s two leaves (`d1_probe.py`, `ladder_probe.py`) ALSO land in commit 3, emptying its
directory of leaves at that point — but the conductor ruling's own text explicitly reserves
`analysis`'s retraction, together with root, for "the FINAL commit." **This session follows that
literal instruction** rather than applying FACADE-EMPTY's general "retract as soon as empty" timing
to `analysis` early: retracting LATER than the earliest-possible point is always safe (nothing
downstream depends on prompt removal of an already-dead duplicate), so there is no correctness
conflict, only a one-line note that `__legacy/analysis/__init__.py` sits leaf-empty-but-present
between commit 3 and commit 7. The remaining 7 TRUE placeholders + root are batched into commit 7
(FOR-1494) since none of their directories ever held a leaf in the first place (nothing forces
early removal, and bundling under the ticket named "facades retired" is the cleanest per-commit
story). This is a judgment call, stated as such — see *GAPS*.

**RESOLVED by conductor ruling, P12-RETIRED-CANARY (conductor 2026-09-24) — supersedes v2's open GAP
on `gen_shims.py`'s retired-row mechanics for these 12 facade-pattern rows.** The docstring supports
`retired: true` for "a module deleted outright before its move," which fits this case (nothing is
moving; the `__legacy` copy is deleted outright; the real top-level content's own eventual
relocation, if `core.*`/`ladder.{analyze,capture,register}` layers ever actually execute their
`mapping.json`-declared move, remains scoped to a phase this brief does not name and does not claim
to be) — but this session cannot execute `gen_shims.py --verify` against a synthetic `retired: true`
row to CONFIRM the generator accepts it (no code execution of repository tooling is permitted in a
drafting session). **Ruled procedure, binding on the lane, not this drafting session:** before
retiring any of the 12 facade rows, the lane runs a RED-first canary in its own tree — retire ONE
facade row (`evidence`, commit 1, is the earliest and therefore the recommended canary, per v2's
own suggestion) and delete its `__legacy` one-liner/duplicate, then run `tools/gen_shims.py --verify`
and `--preflight`. **If either refuses a `retired: true` row whose old-address facade still exists
OUTSIDE `__legacy/`, the lane STOPS and returns `DECISION` quoting the refusal verbatim** — the
conductor amends `gen_shims.py` itself under FOR-1555's follow-up; the lane never hand-edits the
generator in-lane. If the canary passes, the remaining 11 rows follow the same pattern without
re-running the full canary procedure for each (one canary stands for the class, since all 12 rows
share the identical shape: a deleted `__legacy` duplicate whose real content is untouched outside
`__legacy/`).

### P12-LEGACY-EMPTY-DEF (RULED by conductor 2026-09-24, revised from v2's own `[ASSUMED]` proposal)

**[MEASURED]**, `git grep -l "Alias shim for entropic_substrate\.__legacy\." 8e5ef7d -- 'src/entropic_substrate/__legacy/**'`:
four files — `__legacy/d1_calibration.py`, `__legacy/d1_calibration_records.py`,
`__legacy/d1_training_step.py`, `__legacy/d1_training_step_records.py` — plus their four `.pyi`
companions (8 files, 7+7+7+7 = 28 lines of `.py` + 29+14+109+32 = 184 lines of `.pyi`, 212 lines
total) are GENERATED shims that gen_shims emits at the OLD, `__legacy`-PREFIXED address itself
(`entropic_substrate.__legacy.d1_calibration`, not just the bare `entropic_substrate.d1_calibration`
every other module gets). Their four underlying modules are ALL `landed: true` already (p8, p9,
`[MEASURED]` directly from those fragments) — this is not p12 residue at all, it is a permanent
side effect of the shim mechanism. **Cause, `[MEASURED]`, CORRECTED citation (house rule 4 — v2
cited only two of the four import statements that create this dependency):**
`tests/test_d1_training_step_split.py:35-39` (`from entropic_substrate.__legacy import
(d1_calibration, d1_training_step, d1_training_step_records)`, the three-name block v2's citation
omitted entirely), `:40-42` (`from entropic_substrate.__legacy.d1_calibration_records import
(D1BaselineCalibrationAttestation,)`), and `:43-48` (`from entropic_substrate.__legacy.
d1_training_step_records import (D1ComponentGradientEvidence, D1GradientEvidenceKind,
D1GradientReachability, D1TrainStepResult)`) — three separate import statements, together covering
all four module names, all by the `__legacy`-prefixed dotted name, not through the ordinary
top-level shim. No phase before p13 may edit an existing test (repeated Non-goal across every prior
brief).

**RULED, conductor 2026-09-24, P12-LEGACY-EMPTY-DEF.** p12's postcondition is: every file under
`src/entropic_substrate/__legacy/` starts with `# Generated by gen_shims` — measured to be exactly
the 8 files above (the 4 test-served forwarder pairs: `d1_calibration`, `d1_calibration_records`,
`d1_training_step`, `d1_training_step_records`, each `.py`+`.pyi`) — no other file or directory
remains under `__legacy/`, and every `tools/mapping.json` row is `landed` or `retired` across the
merged fragments. p13 deletes the last 8 files together with the rewrite of
`tests/test_d1_training_step_split.py` off the `__legacy`-prefixed import. **`tests/test_legacy_
empty.py` (NEW, p12's own) asserts exactly this**, RED-first: the lane plants one non-generated file
under `__legacy/` (e.g. an empty `__legacy/_canary.py` with no generated-marker header) in a scratch
step, confirms the test goes RED, then removes the plant and re-runs green. This resolves v2's own
`[ASSUMED]`, self-proposed reading — the conductor's ruling is now the binding postcondition text,
not this session's invention.

## ACCEPTANCE

Machine-checkable, one line each, each naming its instrument.

- All 44 residue modules exist at their destination addresses (31 original + gate half + worker
  remainder + 11 facade duplicates retired), with regenerated shim pairs (`tools/gen_shims.py
  --verify`); `historical/{d_ladder,e_series}/__init__.py` new, docstring-only, zero re-exports;
  `composition/{wiring,entrypoints}/__init__.py` new, docstring-only.
- `d_ladder_artifacts` + `d_ladder_execution` land in ONE commit (P12-SCC); `just check`'s
  `basedpyright` step is the guard-teeth instrument (CORRECTED from a pytest-collection
  `ImportError`, which the runtime-only edge graph cannot produce — see *P12-SCC*).
- **Five**, not six, E/D CLI subcommands (`train`, `eval`, `field-pretrain`, `d-probe`,
  `d-score-gain`) still dispatch into `historical/` — `init-fisher` already landed at p6 and is not
  recounted here (CORRECTED from DN-24's stale six-subcommand claim).
- A checkable postcondition test (`tests/test_legacy_empty.py`, NEW, RED-first: plant one
  non-generated file, confirm RED, remove it, confirm green) asserts every remaining file under
  `__legacy/` starts with the `# Generated by gen_shims` marker (measured to be exactly the 8
  `d1_calibration*`/`d1_training_step*` forwarder pairs), no other file or directory remains, per
  the RULED P12-LEGACY-EMPTY-DEF postcondition (conductor 2026-09-24, no longer ASSUMED); it
  separately asserts every `mapping.json` row resolves to either `landed: true` or `retired: true`
  across the merged fragments (no silent third state).
  `src/entropic_substrate/__legacy/` itself is removed as a directory once its only remaining
  contents are the 8 generated `d1_calibration*`/`d1_training_step*` files (which move up to
  become direct children of no quarantine directory is NOT proposed — they stay exactly where they
  are, under whatever path gen_shims currently emits them; this brief does not touch that
  mechanism).
- `worker_resource_registration`'s digest/provider migration (P12-WORKER-REMAINDER): `ladder/run/
  worker/resource_registration.py` computes `_FACTORY_DIGEST`/`_PROVIDER` itself; the landed
  remainder (`historical/d_ladder/d_ladder_worker_runtime.py`) imports them back from it; zero
  `ladder → historical` edges remain anywhere in the tree (`lint-imports` zero inversions, the
  standing instrument every phase uses).
- `git diff -M --stat` shows every fully-moved module as a pure rename plus import-line rewrites
  only; the worker-remainder's direction-flip commit (3) and every facade-retraction diff are new
  deletions, not renames, and are called out as such.
- **NO `remap_check` arm** (ruling P9-REMAP, FOR-1549, wave-wide, not re-opened here).
- **Allowlist/literal byte-identity, per-commit** (P12-ALLOWLISTS): each of the three literal-edit
  sites (`provider_launch.py:35-38 (set literal block; the two strings at :36-37)`, `d1_runpod_adapter.py:349`, `scripts/entrypoints.env` +
  `h_probe_launch.py:167` + `ladder/review/d1_external_analysis_adapter.py:241`) is edited in the
  commit named for it under *Planned commit branches*, each paired with the matching
  `tests/test_cli_surface.py::TestFrozenDottedNameLiteralSet` parametrization staying green,
  UNEDITED for every site this phase does not touch, EDITED (door amendment) for every site it
  does, both arms named in the PR body.
- The p0 CLI `--help` snapshot (`tests/test_cli_surface.py::TestHelpSnapshotIsWidthPinned`) stays
  BYTE-IDENTICAL after `cli.py` relocates to `composition/entrypoints/cli.py`. `[MEASURED]`:
  `test_prog_name_is_frozen` asserts `build_parser().prog == "entropic-substrate"`, a hardcoded
  string literal INSIDE `build_parser()`, never derived from the module's own dotted address or
  `__main__`/argv0 — this resolves the audit's open question in P12-CLI-ROOT's favour: relocating
  the file does not, by itself, touch `prog`, and the snapshot instrument is unaffected as long as
  `build_parser()`'s own `prog=` argument is carried over verbatim.
- **Cold import, fresh interpreter per moved leaf, entered FIRST, stdout marker not exit code**
  (`tests/test_historical_composition_cold_import.py`, NEW, mirroring `tests/test_run_upper_cold_
  import.py`, RED-first on commit 1's destination addresses, extended per subsequent commit). This
  drafting session did not execute this sweep (no code execution permitted); the lane runs it fresh.
- **Topological check recorded before the first commit**, re-derived from the 63-edge residue graph
  (61 runtime + 2 `TYPE_CHECKING`-only) this session built by AST-parsing all 44 residue files
  directly (Tarjan: one SCC, the `d_ladder_artifacts`/`d_ladder_execution` pair; the runtime-only
  graph is a zero-cycle DAG). A disagreement
  between the lane's own live re-derivation and this brief's table is a FREEZE and a `DECISION`.
- **Every landing commit carries its own fragment rows AND its own MEASURED `legacy_edge_budget`**
  (BUDGET-PER-COMMIT). Baseline `[MEASURED]` at `8e5ef7d`: **5** (p10's final value, read directly
  from `.claude/workflows/landed-set/p10-run-upper.json`). `[predicted]` per-commit deltas are given
  under *Planned commit branches* but are not load-bearing; the lane measures live.
- `lint-imports` green (zero inversions among `kernel < core < ladder < historical < composition`;
  `__legacy` stays outside the contract until p13, edges to/from it bounded by `tests/test_legacy_
  coupling.py` instead); `just check`, `just pc-ci`, `just ship-it` green;
  `gh pr view --json headRefOid` equals `git rev-parse`.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms
  recorded.
- The actual diff's file set is a subset of *Allowed files*.

## Non-goals

- Deletes NOTHING from `historical/`. No module's BEHAVIOUR changes; `historical/` is governed
  read-only for `e_series`, runnable-but-frozen-behaviour for `d_ladder`.
- Does NOT delete the shim layer at the OLD top-level dotted address for any of the 11 facades or
  root — only their `__legacy`-physical DUPLICATE. `src/entropic_substrate/{geometry,init,layers,
  models,observation,data,predictive_state,analysis,evidence,experiments,sentence_prediction}/
  __init__.py` and `src/entropic_substrate/__init__.py` are byte-identical before and after this
  phase.
- Does NOT touch `src/entropic_substrate/core/**`, `src/entropic_substrate/ladder/run/**` (besides
  the one named `resource_registration.py` digest/provider migration), `src/entropic_substrate/
  ladder/capture/**`, `src/entropic_substrate/ladder/analyze/**` (besides the one named
  `ladder_analysis.py:17` import-line rewrite), `src/entropic_substrate/ladder/file/**` (besides
  `ladder_filing.py:12`), or `src/entropic_substrate/ladder/gate/**` besides the new
  `ladder_campaign.py` leaf and the `ladder_predecessor.py:8` import-line rewrite.
- Does NOT edit `tools/mapping.json`, `.importlinter`, other phases' briefs, the plan document, or
  CI. Does NOT edit `scripts/entrypoints.env`'s AUTHORING mechanism (E-4, p0's) — only its six
  literal values (commit 6).
- Does NOT edit any EXISTING test other than the one named door-amendment class inside
  `tests/test_cli_surface.py` (see *P12-ALLOWLISTS*, three separate sub-edits across commits 3, 5,
  6, each scoped to exactly the literal(s) that commit changes).
- Does NOT re-pin the deployment image (ruling recommendation E-5: once, at p13).
- Does NOT resolve the `train.py` second-`MetricSink`-construction-site tension the audit flagged
  (§8): `train.py`'s own default `JsonlSink()` fallback in `Trainer.__init__` is preserved
  byte-identical, per "historical governed read-only." Whether this is an accepted exception or an
  undetected conformance violation is not this brief's to rule — see *GAPS*.
- Does NOT introduce a transient `entropic_substrate.__legacy.<x>` address for a module THIS phase
  itself moves.

## Allowed files

- `.secrets.baseline` (STANDING RULE, unchanged from every prior phase's envelope)
- `.claude/workflows/landed-set/p12-historical-composition.json` — this phase's disjoint fragment
- the 44 residue sources enumerated under *Scope* (31 original + gate half + worker remainder + 11
  facade `__legacy`-physical duplicates, deletion only for the 11)
- `src/entropic_substrate/historical/{d_ladder,e_series}/**` (new)
- `src/entropic_substrate/composition/{wiring,entrypoints}/**` (new) — explicitly NOT
  `src/entropic_substrate/composition/cli.py` or `composition/__init__.py` (both occupied, unedited
  — see *P12-CLI-ROOT*)
- `src/entropic_substrate/ladder/gate/ladder_campaign.py` (new leaf in the existing package)
- `src/entropic_substrate/ladder/run/worker/resource_registration.py` (ONE edit: the digest/provider
  migration, P12-WORKER-REMAINDER)
- the 44 typed shim pairs at the old paths, regenerated, plus every regenerated ROOT-LEVEL
  old-prefix shim whose target moves
- the landed-importer files named across *Planned commit branches*, each edited ONLY on the named
  line(s): `ladder/gate/ladder_predecessor.py`, `ladder/analyze/ladder_analysis.py`,
  `ladder/file/ladder_filing.py`, `ladder/run/worker/d1_predecessor_loader.py`,
  `ladder/run/provider/provider_launch.py` (lines 35-38 only, the set literal; the two frozen strings at 36-37), `ladder/run/provider/
  d1_runpod_adapter.py` (line 349 only), `ladder/run/h_probe_launch.py` (line 167 only),
  `ladder/review/d1_external_analysis_adapter.py` (line 241 only)
- `scripts/entrypoints.env` (all six values, commit 6 — FOR-1488's widened scope, re-homed here)
- `justfile` (the one dangling `patches/entrypoints-python.md` comment only, commit 6) — HUB
  CONFLICT with `p0-contract-pin` on this same file, recorded both directions (see manifest)
- `scripts/run_ladder.sh`, `Containerfile` — verification-only per commit 6 (both already
  parameterized off `scripts/entrypoints.env` by p0's E-4; p12 makes no code change to either
  unless a live grep finds one, in which case `DECISION`)
- `tests/test_cli_surface.py` — DOOR AMENDMENT, three scoped sub-edits (see *P12-ALLOWLISTS*),
  never any other assertion in the file
- `tests/observability_conformance_config.py`'s consumers: p12's OWN fragment keys
  `sink_construction_modules` (adding `historical.e_series.{train,field_pretrain}`, P12-ESERIES-SINK)
  and `print_allowlist` (re-keying `entropic_substrate.cli` to its landed name, adding entries for
  `d1_probe_cli`, `d_ladder_cli`, `d_ladder_probe_cli`'s one `print()` site each once their landed
  names are known) — never the config module itself, which derives everything from fragments
- `tests/test_legacy_empty.py`, `tests/test_historical_composition_cold_import.py` (both new,
  RED-first)
- generator outputs, `.claude/workflows/landed-set/.emitted.json` (untracked, never `git add`)

**Deliberately EXCLUDED:** `src/entropic_substrate/composition/cli.py`,
`src/entropic_substrate/composition/__init__.py` (both p4-landed, unrelated H-series content — see
*P12-CLI-ROOT*); every OTHER assertion in `tests/test_cli_surface.py` beyond the three named literal
sub-edits; `src/entropic_substrate/core/**`; `tools/mapping.json`; `.importlinter`; any file not
named above. A diff touching any of them is a `DECISION`, not an edit.

## P12-ALLOWLISTS / P12-CLI-ROOT / P12-ESERIES-SINK

**P12-ALLOWLISTS (RATIFIED, FOR-1488 re-homed here as 7/7).** Two allowlist literals move with
their named module: `entropic_substrate.d1_trainer_mailbox_command` (commit 5, composition) at
`provider_launch.py:36` and `d1_runpod_adapter.py:349`; `entropic_substrate.d_ladder_worker_command`
(commit 3, historical.d_ladder) at `provider_launch.py:37`. Each rewrite pairs with a scoped door
amendment to `tests/test_cli_surface.py`'s `_FROZEN_DOTTED_LITERALS` tuple and the relevant
`_FROZEN_LITERAL_SITES_MODULE` entry, in the SAME commit. `scripts/entrypoints.env`'s six values,
`h_probe_launch.py:167`, `ladder/review/d1_external_analysis_adapter.py:241`, and the justfile's
dangling comment are bundled into commit 6, `[ASSUMED]` (this session's judgment call, not forced
by the ruling text — see *GAPS*): rewriting `entrypoints.env` requires EVERY moving module's final
address, so it is cleanest to do it once, after `cli.py`/`d1_trainer_mailbox_command`/`d_ladder_
worker_command` are all already landed (commits 3 and 5), rather than split across three earlier
commits. `h_probe_launch.py`'s and `d1_external_analysis_adapter.py`'s own literals
(`entropic_substrate.sentence_prediction.worker_command`, `entropic_substrate.analysis.
d1_external_worker`) name modules p12 does NOT move (both already landed elsewhere) — they are
edited ONLY because `scripts/entrypoints.env`'s corresponding values change to the landed address in
the same commit, and the allowlist string must match what actually gets invoked.

**P12-CLI-ROOT (ASSUMED, operator ratification pending).** `composition/cli.py` is p4's H-series
manifest CLI (**182 lines**, CORRECTED from v2's mistyped 227 — `[MEASURED]` `git show
8e5ef7d:src/entropic_substrate/composition/cli.py | wc -l` — house rule 4; `sentence_prediction.cli`'s landed target) AND a frozen
`h-checkpoint-reader-v2` member's sibling package — `[MEASURED]`, `checkpoint.py`'s
`_READER_MODULES` contains `composition.cli` among its `composition.*` entries, confirmed to refer
to THIS module, not the incoming E/D one. It is NOT renamed. The E/D argparse root (1133 lines,
`entropic_substrate.cli` today) lands at `composition/entrypoints/cli.py`, `__main__` beside it at
`composition/entrypoints/__main__.py`. `scripts/entrypoints.env`'s `ENTRYPOINT_CLI` becomes
`entropic_substrate.composition.entrypoints.cli`. The p0 `--help` snapshot is byte-identical
per the `_PROG` finding above. Wiring modules (`d1_cli_composition`, `d1_campaign_*`,
`d1_capture_cli_services`, `d1_trainer_cli_services`, `d1_inspect_service`, `d1_probe_cli`,
`d_ladder_cli`, `d_ladder_probe_cli`) go to `composition/wiring/`; worker entrypoints
(`d1_trainer_mailbox_command`, `d1_trainer_worker_command`, `d1_trainer_process_worker`,
`d1_r2_trainer_mailbox_worker`) go to `composition/entrypoints/`, alongside `cli.py`/`__main__.py`.

**P12-ESERIES-SINK (ASSUMED, operator ratification pending).** `train.py:220` and
`field_pretrain.py:362` keep constructing `JsonlSink` directly (governed read-only, zero behaviour
change). Same commit (2) adds `entropic_substrate.historical.e_series.train` and
`...field_pretrain` to `ALLOWED_SINK_CONSTRUCTION_MODULES` via p12's own fragment's
`sink_construction_modules` key, with the read-only rationale recorded in the fragment's `note`.
Their `entropic_substrate.sinks` imports rewrite to the landed adapter address on arrival
(`[MEASURED]`, need live confirmation of the exact landed leaf — not independently re-derived by
this drafting session; flagged for the lane). Print sites `[MEASURED]`: `__legacy/cli.py` 3 sites
(all `json.dumps` machine-consumed envelopes, lines 521/1025/1072 — 2 of these are the p0-pinned
`eval --json` and `telemetry final-eval` sites, unrelated to p12 since `telemetry` isn't a p12
target; the third needs classification, GAP); `d1_probe_cli.py`, `d_ladder_cli.py`, `d_ladder_
probe_cli.py` each have ONE `print()` site with no existing allowlist entry — p12 must add
`print_allowlist` entries for their LANDED names in commit 4 (KEYING ruling: print-ban entries are
re-keyed to the landed name, unlike `conforming_entrypoints`).

## Already ruled

| id                   | status                                                                                 | ruling as it binds this phase                                                                                                                                                                                                                                                                                                                                                               |
| -------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P12-PARENT           | CONDUCTOR 2026-09-24                                                                   | Parent is `mythwave/p10-run-upper` @ `<startTip>`.                                                                                                                                                                                                                                                                                                                                          |
| P12-RESIDUE          | CONDUCTOR 2026-09-24, CORRECTED one line by this session                               | p12 owns all 44 residue modules (31 original + 13 previously unowned); root `__init__` was already counted in the 31, not an addition — see *Scope*.                                                                                                                                                                                                                                        |
| P12-SCC edge count   | CORRECTED by this session (v3), RETRACTING v2's false "independently re-derived" claim | Exhaustive AST parse of all 44 residue files: 63 total edges (61 runtime + 2 TYPE_CHECKING-only), up from v2's uncritically-copied 58; five edges from `cli.py` were missing from `targets.json`'s field. SCC conclusion and 7-commit topological order both UNCHANGED (zero forward-reference violations under the corrected graph).                                                       |
| P12-GATE-HALF        | RULED by operator 2026-09-24, FOR-1557                                                 | `ladder_campaign.py`'s gate half lands at `ladder/gate/ladder_campaign.py`, commit 1. Mapping.json door amendment is a conductor precondition, not p12's edit.                                                                                                                                                                                                                              |
| P12-WORKER-REMAINDER | CONDUCTOR 2026-09-24                                                                   | Remainder lands at `historical/d_ladder/d_ladder_worker_runtime.py`, commit 3; digest/provider migrate to `resource_registration.py` in the same commit to avoid a `ladder → historical` inversion.                                                                                                                                                                                         |
| P12-FACADES          | CONDUCTOR 2026-09-24, applied with a MEASURED correction by this session               | Each `__legacy/<pkg>/__init__.py` retracts when its directory holds no leaf; root + analysis explicitly deferred to the final commit per the ruling's own text; this session found 4 of 11 facades are byte-identical duplicates, not 1-line placeholders, which does not change the action (delete the duplicate) but does change its size.                                                |
| P12-SCC              | CONDUCTOR 2026-09-24                                                                   | `d_ladder_artifacts`/`d_ladder_execution` land as one commit; the "collection-time `ImportError`" guard tooth is RETRACTED as unreachable (the cycle is `TYPE_CHECKING`-only); `basedpyright` is the corrected instrument.                                                                                                                                                                  |
| P12-ALLOWLISTS       | RATIFIED, FOR-1488 re-homed                                                            | Two literals move with their module (commits 3, 5); the wider bundle (`entrypoints.env`, `h_probe_launch.py`, `d1_external_analysis_adapter.py`, justfile comment) rides commit 6, `[ASSUMED]` timing — see *GAPS*.                                                                                                                                                                         |
| P12-CLI-ROOT         | ASSUMED, operator ratification pending                                                 | E/D `cli.py` lands at `composition/entrypoints/cli.py`, never `composition/cli.py` (occupied by p4's H-series CLI).                                                                                                                                                                                                                                                                         |
| P12-ESERIES-SINK     | ASSUMED, operator ratification pending                                                 | `train.py`/`field_pretrain.py` keep constructing `JsonlSink` directly; both modules join `ALLOWED_SINK_CONSTRUCTION_MODULES` in the same commit.                                                                                                                                                                                                                                            |
| P12-LEGACY-EMPTY-DEF | RULED by conductor 2026-09-24 (revised from v2's own `[ASSUMED]` proposal)             | Postcondition is: every remaining `__legacy/` file starts with the `# Generated by gen_shims` marker (exactly the 8 `d1_calibration*`/`d1_training_step*` forwarder pairs), no other file/directory remains, every mapping.json row landed or retired; p13 deletes the last 8 with the test rewrite. `tests/test_legacy_empty.py` asserts this, RED-first via a planted non-generated file. |
| P12-RETIRED-CANARY   | RULED by conductor 2026-09-24                                                          | Before retiring any facade row, the lane runs a RED-first canary (`evidence`, commit 1): retire the row, delete its `__legacy` one-liner, run `gen_shims --verify`/`--preflight`; a refusal is a `DECISION` quoting it verbatim, never a hand-edit of the generator in-lane. FOR-1555's follow-up owns any needed generator amendment.                                                      |
| E-6 (was DN-10)      | RATIFIED at sprint-preflight                                                           | `historical/d_ladder/`, runnable, same governance as `e_series`.                                                                                                                                                                                                                                                                                                                            |
| E-4                  | RULED 2026-09-17                                                                       | `scripts/entrypoints.env` is the one parameterized source; p12 edits its VALUES only, in commit 6.                                                                                                                                                                                                                                                                                          |
| DN-24                | RULED 2026-09-17, stale on ONE count                                                   | Six E/D subcommands is now measured five (`init-fisher` landed at p6); the underlying door is otherwise unaffected.                                                                                                                                                                                                                                                                         |
| P-1                  | RULED 2026-09-17                                                                       | Observability wired through during the refactor; the E-series pattern (governed read-only, sink allowlist entry) applies identically here.                                                                                                                                                                                                                                                  |
| O-1                  | RULED 2026-09-17                                                                       | `publishMode:'defer'` — N/A, p12 is a serial phase, not a wave lane.                                                                                                                                                                                                                                                                                                                        |
| rewrite-on-arrival   | RULED 2026-09-17                                                                       | A file's import lines are rewritten exactly once, by the phase that moves it.                                                                                                                                                                                                                                                                                                               |
| shim form            | RULED 2026-09-17                                                                       | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`.                                                                                                                                                                                                                                                                                                 |
| P11-FACADE           | CONDUCTOR 2026-09-23                                                                   | p11 deferred `__legacy/analysis/__init__.py`'s retraction to p12 (never emptied the directory itself); consistent with the deferral above.                                                                                                                                                                                                                                                  |
| FACADE-EMPTY         | CONDUCTOR 2026-09-23 (wave-wide)                                                       | A facade retracts only in the commit after which its directory holds no leaf; each brief states which phase does it — this one does, per commit, above.                                                                                                                                                                                                                                     |
| BUDGET-PER-COMMIT    | CONDUCTOR 2026-09-23                                                                   | Every landing commit writes its own MEASURED `legacy_edge_budget`.                                                                                                                                                                                                                                                                                                                          |
| P9-REMAP             | CONDUCTOR 2026-09-23, FOR-1549                                                         | No remap arm, guard row, or artifact.                                                                                                                                                                                                                                                                                                                                                       |
| DATA-BUILD-RUNG      | CONDUCTOR 2026-09-23                                                                   | Precedent for why the cold-import sweep must actually run, not be assumed.                                                                                                                                                                                                                                                                                                                  |

## Guard teeth by mutation

| mutation                                                                                                                                                                                                                      | must go RED (named instrument)                                                                                                                                                                         | GREEN arm                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| split `d_ladder_artifacts`/`d_ladder_execution` across two commits                                                                                                                                                            | `basedpyright` inside `just check` (CORRECTED from a pytest-collection `ImportError`, unreachable — the cycle is `TYPE_CHECKING`-only, see *P12-SCC*)                                                  | both land in one commit, typecheck clean                                                           |
| leave one file under `__legacy/` that is not one of the 8 P12-LEGACY-EMPTY-DEF generated exceptions                                                                                                                           | `tests/test_legacy_empty.py` (NEW)                                                                                                                                                                     | zero non-generated files, test unedited after commit 7                                             |
| reintroduce one old-prefix import in a landed module                                                                                                                                                                          | `tests/test_no_old_prefix.py`                                                                                                                                                                          | zero landed→old-prefix edges                                                                       |
| inflate or deflate the committed `legacy_edge_budget` by one, either direction                                                                                                                                                | `tests/test_legacy_coupling.py::test_measured_landed_to_legacy_edges_equals_committed_budget`                                                                                                          | measured == committed, number never typed                                                          |
| hand-delete one regenerated shim pair for a module this phase already landed                                                                                                                                                  | `tools/gen_shims.py --verify` exits 1 naming the missing file                                                                                                                                          | `--verify` exits 0 after regenerating                                                              |
| land a commit that moves modules without adding their fragment rows                                                                                                                                                           | `tests/test_module_identity.py` and `gen_shims --preflight` disjointness                                                                                                                               | every commit carries its own rows                                                                  |
| reintroduce the `ladder → historical` inversion via `resource_registration.py`                                                                                                                                                | `lint-imports` (the standing layer-contract instrument)                                                                                                                                                | zero inversions, `resource_registration.py` self-contained on the digest/provider                  |
| change one string in `provider_launch.py:35-38 (set literal block; the two strings at :36-37)`, `d1_runpod_adapter.py:349`, `h_probe_launch.py:167`, or `d1_external_analysis_adapter.py:241` outside the commit named for it | `tests/test_cli_surface.py::TestFrozenDottedNameLiteralSet` parametrized per site                                                                                                                      | each site's literal byte-identical except in its named commit, door-amended together with the test |
| delete one forwarder-served `__legacy` file a moved module still imports before its own commit lands                                                                                                                          | `tests/test_historical_composition_cold_import.py` (NEW)                                                                                                                                               | all landed-so-far destination addresses cold-import clean                                          |
| retract a facade duplicate before its directory holds no leaf                                                                                                                                                                 | a live `find __legacy/<pkg> -type f` check recorded in the PR body per commit (no dedicated test — FACADE-EMPTY's own instrument across the sprint has always been "listing it," not a pytest fixture) | directory empties exactly when claimed                                                             |
| construct a `MetricSink` inside a landed, non-allowlisted module                                                                                                                                                              | `tests/test_observability_conformance.py`'s sink-construction arm                                                                                                                                      | only `composition/**` and the two P12-ESERIES-SINK entries construct one                           |

## Observability conformance

Per P-1. `historical/` keeps E-series's exact pattern (governed read-only, allowlisted construction
site — P12-ESERIES-SINK). `composition/entrypoints/cli.py` becomes `entropic_substrate.composition.
entrypoints.cli`; `CONFORMING_ENTRYPOINTS`/`NOT_YET_CONFORMED_ENTRYPOINTS` stay KEYED to the
PRE-MOVE mapping.json name (`entropic_substrate.cli`) per the standing KEYING ruling — never
re-keyed before p13 — so this phase's fragment adds no `conforming_entrypoints` entry at all
(the row already exists, keyed correctly, from p3b/p4). `print_allowlist` DOES re-key
(`entropic_substrate.cli` → its landed name) and gains three new entries for `d1_probe_cli`,
`d_ladder_cli`, `d_ladder_probe_cli`'s one `print()` site each. `sink_construction_modules` gains
the two `historical.e_series` entries. The three machine-consumed raw-`print()` sites p0 pinned
(`telemetry final-eval`, `cli eval --json`) are UNTOUCHED — `telemetry` is not a p12 target; `cli
eval --json`'s print survives the file's relocation unchanged (pinned by substring, not by module
address, per `tests/test_cli_surface.py`'s own docstring).

## Planned commit branches

Ticket mapping per conductor instruction: FOR-1557 (gate half), FOR-1489 (e_series), FOR-1490
(d_ladder + remainder + SCC), FOR-1491 (composition wiring), FOR-1492 (entrypoints incl. cli root),
FOR-1493 (justfile/run_ladder.sh/Containerfile/entrypoints.env), FOR-1488 (literals, split across
commits 3/5/6), FOR-1494 (facades retired + postcondition, final). Order derived from this session's
own 63-edge AST-based topological re-derivation (61 runtime, 2 `TYPE_CHECKING`-only; Tarjan, one
SCC) — independently re-checked against the RUNTIME-only 61-edge graph after correcting the 12
facade-pattern modules' modeling (their old dotted address never changes resolution, so an import of
one is never a landing-order constraint): **zero forward-reference violations** against the table
below, including the 5 edges `targets.json` had omitted (all from `cli.py`, commit 5, all five
targets already landed by commit 4).

| # | ticket       | scope                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |                                                                  modules |               lines | `__legacy` dirs emptied                                                                                                                                          | `[predicted]` budget                                                                                                                                                                                                                |
| - | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -----------------------------------------------------------------------: | ------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **FOR-1557** | `evidence.ladder_campaign` → `ladder/gate/ladder_campaign.py`; retract `__legacy/evidence/__init__.py` (8L duplicate); rewrite 4 landed importers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                                             1 (+ facade) |    371 (+8 deleted) | `__legacy/evidence/`                                                                                                                                             | 5 → 1                                                                                                                                                                                                                               |
| 2 | **FOR-1489** | `train.py`, `field_pretrain.py` → `historical/e_series/`; new `__init__.py`; `sinks` import rewrite; `sink_construction_modules` fragment entries (P12-ESERIES-SINK)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                                                        2 |                1404 | none                                                                                                                                                             | 1 (unchanged; e_series is self-contained, zero landed-importer edges found)                                                                                                                                                         |
| 3 | **FOR-1490** | 11 `historical.d_ladder`-dest modules (incl. the `d_ladder_artifacts`/`d_ladder_execution` SCC pair) + worker-runtime remainder → `historical/d_ladder/`; new `__init__.py`; `resource_registration.py` digest/provider migration; retract `__legacy/experiments/__init__.py` (8L duplicate, `ladder_profile.py` was its last leaf); `d_ladder_worker_command`'s allowlist literal (`provider_launch.py:37`) + door amendment                                                                                                                                                                                                                                                                                                                                    |                                                                       12 |                2398 | `__legacy/experiments/` (`__legacy/analysis/` empties of LEAVES here but its `__init__.py` retraction is deferred to commit 7 per P12-FACADES)                   | 1 → `[predicted]` 0 (the one remaining edge, `resource_registration.py`'s import of the remainder, is discharged by the digest/provider migration in this same commit)                                                              |
| 4 | **FOR-1491** | 11 composition wiring modules (`d1_campaign_authority_loader`, `d1_cli`, `d1_campaign_cli_service`, `d1_campaign_controller_factory`, `d1_capture_cli_services`, `d1_inspect_service`, `d1_trainer_cli_services`, `d1_cli_composition`, `d1_probe_cli`, `d_ladder_cli`, `d_ladder_probe_cli`) → `composition/wiring/`; `print_allowlist` entries for `d1_probe_cli`/`d_ladder_cli`/`d_ladder_probe_cli`                                                                                                                                                                                                                                                                                                                                                          |                                                                       11 |                2510 | none                                                                                                                                                             | 0 (no landed importers of these found; the lane confirms live)                                                                                                                                                                      |
| 5 | **FOR-1492** | `d1_trainer_process_worker`, `d1_trainer_worker_command`, `d1_r2_trainer_mailbox_worker`, `d1_trainer_mailbox_command`, `cli.py`, `__main__.py` → `composition/entrypoints/`; `d1_trainer_mailbox_command`'s allowlist literal (`provider_launch.py:36`, `d1_runpod_adapter.py:349`) + door amendment                                                                                                                                                                                                                                                                                                                                                                                                                                                            |                  6 (+ root facade note, retraction deferred to commit 7) | 2063 (+12 deferred) | none                                                                                                                                                             | 0                                                                                                                                                                                                                                   |
| 6 | **FOR-1493** | `scripts/entrypoints.env` (all six values → landed addresses, `[ASSUMED]` timing, see *P12-ALLOWLISTS*); `h_probe_launch.py:167`; `ladder/review/d1_external_analysis_adapter.py:241`; `justfile`'s dangling `patches/entrypoints-python.md` comment; `tests/test_cli_surface.py` door amendment for these 3 sites; `scripts/run_ladder.sh`/`Containerfile` verification-only                                                                                                                                                                                                                                                                                                                                                                                    |                                                                0 modules |                   0 | none                                                                                                                                                             | 0 (no module moves; script/config only)                                                                                                                                                                                             |
| 7 | **FOR-1494** | Retract the 7 remaining TRUE-placeholder facades (`geometry`, `init`, `layers`, `models`, `observation`, `data`, `predictive_state`; 1L each) + `__legacy/sentence_prediction/__init__.py` (1L, byte-identical duplicate of the top-level one-liner) + root `__legacy/__init__.py` (12L) + `__legacy/analysis/__init__.py` (21L, deferred from commit 3); remove the `__legacy/` directory tree down to the 8 P12-LEGACY-EMPTY-DEF generated exceptions; author `tests/test_legacy_empty.py` (RED-first, then green) and finish `tests/test_historical_composition_cold_import.py`'s extension to all 44 modules + every `python -m` entrypoint; final `legacy_edge_budget` measurement; full allowlist byte-identity re-confirmation across all 6 prior commits | 0 modules (10 facade files, 41L: 8 one-liners + root 12L + analysis 21L) |  41 (+ dir removal) | `__legacy/{geometry,init,layers,models,observation,data,predictive_state,sentence_prediction,analysis}/`, `__legacy/` itself (except the 8 generated exceptions) | 0 → **0 or the count of the 8 generated-exception edges the lane's own classifier still attributes to them, whichever `tests/test_legacy_coupling.py` measures — this session cannot predict which without running the classifier** |

Rows 1-7 sum to 44 residue modules / 9018 lines (371+1404+2398+2510+2075+0+729 facade-lines-deleted
= reconciles: 371+1404+2398+2510+2075 = 8758 real-module lines landed across commits 1-5, plus
8+8+21+8×1+12 = 57 lines of `__legacy`-physical facade duplicate deleted across commits 1/3/7 —
these are TWO DIFFERENT counts, matching the *Scope* table's distinction between "residue accounted
for" (9018, top-level-file convention) and "bytes actually removed from `__legacy/`" (8758 + 57 =
8815, plus the gate half already counted in the 8758). No arithmetic gap: every one of the 44
mapping.json rows this brief owns is accounted for in exactly one commit above.

## GAPS / DECISION-NEEDED

1. **P12-LEGACY-EMPTY-DEF and P12-RETIRED-CANARY are now RULED (conductor 2026-09-24), closing v2's
   two biggest open gaps** — but neither ruling's PROCEDURE has been executed against the live tree
   by this drafting session (no code execution of repo tooling is permitted here). The lane still
   owes: (a) the RED-first plant/remove cycle for `tests/test_legacy_empty.py`; (b) the `evidence`
   canary's actual `gen_shims --verify`/`--preflight` run before any facade row is marked `retired:
   true`. Both are now MECHANICALLY SPECIFIED, not open design questions — this item tracks
   execution, not decision.
2. **P12-ALLOWLISTS' commit-6 bundling of `scripts/entrypoints.env` is a judgment call
   (`[ASSUMED]`), not forced by the ruling text**, which could equally be read as "rewrite each
   value the moment its own module lands" (spreading the entrypoints.env edit across commits 1, 2,
   3, 5 instead of bundling in 6). This brief chose the single-commit reading for auditability; the
   operator may prefer the other. Either is mechanically sound.
3. **A full byte-exact import audit of the other 43 residue files (beyond the one, `d_ladder_worker_
   runtime`'s remainder, this session read in full) was not performed for INTER-package imports of
   already-landed (non-residue) modules** — distinct from the intra-residue edge graph, which THIS
   v3 draft DID exhaustively AST-parse (see *P12-SCC*, 63 edges, `[MEASURED]`). The *Planned commit
   branches* table's per-commit `legacy_edge_budget` deltas beyond commit 1 (which IS measured, from
   the audit's own `landed_importer_detail`) are `[predicted]`, unverified against a live
   `tools/import_graph.py` run. The lane measures every one live, per BUDGET-PER-COMMIT.
4. **`train.py`'s second `MetricSink` construction site (audit §8) is preserved unreconciled.**
   Whether this is an accepted E-series exception or an undetected conformance violation once
   `historical.e_series.train` joins `ALLOWED_SINK_CONSTRUCTION_MODULES` is not resolved by this
   brief — the allowlist entry makes it PASS the conformance test mechanically either way, which
   means the test cannot itself distinguish "accepted" from "overlooked." Flagged for the operator,
   not silently accepted.
5. **The third `print()` site in `__legacy/cli.py` (line ≈521 or 1025 or 1072, whichever is not one
   of the two p0-pinned `eval --json`/`telemetry final-eval` sites) was not individually classified
   by this session** — `[MEASURED]` only that 3 print sites exist and 2 are pinned; the third's
   destination (an allowlist entry, or a `logger` rewrite) is not decided here.
6. **Whether `scripts/run_ladder.sh` and `Containerfile` need ANY code change is asserted
   verification-only, not independently re-derived line-by-line by this session** beyond the
   citations already in the audit (`run_ladder.sh:65,72,170,183,187` all reference `ENTRYPOINT_*`
   variables, never a hardcoded dotted name; `Containerfile:90` is a comment). If a live grep at
   commit-6 time finds a hardcoded dotted name this brief missed, that is a `DECISION`, not a silent
   edit.
7. **The exact landed leaf address for `entropic_substrate.sinks`'s current target (consumed by
   `train.py`/`field_pretrain.py`) was not independently re-derived** — P12-ESERIES-SINK's import
   rewrite direction is stated but the destination string is `[MEASURED: need live confirmation]`,
   not resolved by this drafting session.
8. **NEW (this v3, from verifier round 1's minor finding): the AUTHORITATIVE plan document itself,
   `docs/plans/src-hexagonal-refactor.plan.md`, still carries three stale claims uncorrected at
   `8e5ef7d`, independent of v1/v2's own now-corrected numbers** — `[MEASURED]`: the phase-index
   table's p12 row states `57` modules (`:363`); the p12 prose section and the master ACCEPTANCE
   list both state "The six E/D CLI subcommands still dispatch into `historical/`" (`:599`, `:922`);
   the master *Guard teeth by mutation* table's own p7/p12 row still names "collection-time
   `ImportError`" as the SCC-split guard tooth (`:644`), directly contradicted by this brief's own
   `basedpyright`-based P12-SCC finding. p12's Non-goals correctly exclude editing the plan
   document, so this is not a defect in what p12 DOES — but the plan document, the thing an operator
   or a future phase (p13) would consult as ground truth, remains internally inconsistent with this
   brief's corrected, better-supported numbers after p12 ships. Recommend a docs-sweep note or a
   dedicated conductor-owned plan-doc retraction commit — NOT lane work, consistent with this
   repo's "Docs sweep off the critical path" convention (prose-only findings are captured and
   deprioritized into one conductor commit after SUBMITTED, never blocking the critical path). A fourth stale plan claim (found by verifier round 2): the plan says p12 "asserts zero files under `__legacy/`", which P12-LEGACY-EMPTY-DEF supersedes (exactly the 8 generator-emitted forwarder files persist until p13's test rewrite); the conductor retracts it in the same plan-doc sweep.
