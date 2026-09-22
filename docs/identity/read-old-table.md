# Read-old table — reader identity bumps across the hexagonal refactor

**Status: FILLED at p4.** The `old digest` column was filled at **p0**, when the identity fixtures
were computed and pinned at the epoch. The `new digest` column is filled here, at **p4**, at the
tip where both readers were rewritten and recomputed (commit `31113d1a8f5904c0491a4cd4b8d8fcb5f77aad43`,
"feat(h-lane): pin both H reader identities (rung 3)").

This file is a **door** (`kind: door`, `frozen_after: p4`). What is frozen is the LABEL and the
module-set SHAPE of each reader (conductor ruling 2026-09-19: "frozen_after freezes label + shape,
not digest values"). The digest VALUES re-measure whenever a later phase rewrites a member module's
bytes (ruling R-PIN, 2026-09-22: 44 explicit `__legacy` imports sit inside the checkpoint reader's
51 members and are rewritten across p5..p10); each such phase appends a dated row below with the
tool-measured values. The sentence this replaces ("After p4 no phase may change either value") was
stricter than the ruling and is retracted here (house rule 4).

---

## What "read-old window" means here, and what it does not

The window is **not** an in-tree cross-reader shim. `docs/hierarchical-latent-predictor.plan.md`
rules that V1 source evidence "remains unchanged and requires its original reader; no automatic
migration or cross-reader resume is permitted."

So the read-old window is exactly this: **old artifacts replay against the p0-epoch commit, and
nowhere else.** The refactor mints new reader identities and bumps the version tag so the
discontinuity is legible **by name, not only by hash**. No compatibility layer is built, and none
should be proposed.

Measured state of the affected artifacts at the survey epoch:

| reader | artifacts bound to the old identity | DVC-tracked? |
|---|---|---|
| h-source-reader-v2 | three local `h-source-capture-v2` captures (`03d8f8b2...`) `[M, DN-1]` | **no** |
| h-checkpoint-reader-v1 | existing H checkpoints do not resume across the bump `[M, DN-2]` | no authoritative capture binds it `[M]` |

Neither reader has an authoritative DVC-captured artifact bound to it. That is why an outright bump
is admissible rather than a compatibility window.

---

## The table

| identity | old version | old digest (pinned at p0) | new version | new digest (computed at p4) | read-old window | re-certification owner |
|---|---|---|---|---|---|---|
| h-source-reader | `h-source-reader-v2` | per-regime rows in `tests/test_identity_epoch.py::_SOURCE_REGIME_TABLE` (keyed (python, pysbd); two regimes pinned at p0) | `h-source-reader-v3` | per-regime rows in `tests/test_identity_epoch.py::_SOURCE_REGIME_TABLE` (keyed (python, pysbd); local `(3.12.13, 0.3.4)` = `a6075295436de193087160f65fa2558fc28054115118946e53ecbdd4f22d510f`, CI `(3.12.14, 0.3.4)` = `214730ac2b370a7b67fc24a43077c49ecdcc59291505d80f249f235aaccf0467`) | Replay against the p0-epoch commit only. Three local `h-source-capture-v2` captures, none DVC-tracked. No cross-reader resume. | **p4-h-tracer** (serial, `liveRequired:true`, `just ship-it`) |
| h-checkpoint-reader | `h-checkpoint-reader-v1` | per-regime rows in `tests/test_identity_epoch.py::_CHECKPOINT_REGIME_TABLE` (keyed (python, torch); two regimes pinned at p0) | `h-checkpoint-reader-v2` | per-regime rows in `tests/test_identity_epoch.py::_CHECKPOINT_REGIME_TABLE` (keyed (python, torch); local `(3.12.13, 2.13.0)` = `8e4ca5a3840e5c2efa7c4e0f0a1521af576e54139588cef76169021276b8eaf3`, CI `(3.12.14, 2.13.0+cpu)` = `ea85d18b278423bc7de7095b8292ce224b6c7aa50428eecf443dec828eb1602a`) | Replay against the p0-epoch commit only. Existing H checkpoints do not resume across the bump. No cross-reader resume. | **p4-h-tracer** (serial, `liveRequired:true`, `just ship-it`) |

**p0 epoch commit:** `bbe7976df0710af5bdc00879b1b8145568169034`
**Shape amendment (conductor, 2026-09-21, p4 brief audit; door `frozen_after: p4` not yet frozen):** p0 pinned each reader's old digest as a COMPOSED value per measured regime (Python x torch for the checkpoint reader, Python x pysbd for the source reader), because the digest is regime-dependent (MEASURED at p0, see the module docstring of `tests/test_identity_epoch.py`). The 'old digest' column therefore points at those regime tables rather than carrying one literal; p4 fills the 'new digest' column the same way, one row per regime it measures, and records the tip SHA below. The read-old window itself is unchanged: replay against the p0-epoch commit only.

**p4 tip commit:** `31113d1a8f5904c0491a4cd4b8d8fcb5f77aad43` — "feat(h-lane): pin both H reader identities (rung 3)", the last commit whose diff touches a member of either reader's `_READER_MODULES` tuple. This commit (the read-old-table fill) is docs-only and touches no member file, so the digests above remain valid at this tip. Provenance is pre-merge: the SHA is recomputable from the tree it names (`git show 31113d1a8f5904c0491a4cd4b8d8fcb5f77aad43`) and is restated, never reinterpreted, if any in-lane restack or amend moves it before merge.

Both SHAs are full 40-hex, not short. A verdict binds to a derived immutable identifier; a moved
epoch voids the row and it is restated, never reinterpreted.

---

## Why each identity changes, measured

**`h-source-reader-v2` → `v3`.** `sentence_prediction/capture.py` (now `ladder/capture/capture.py`)
`_READER_MODULES` is seven literal dotted names, and the identity is
`content_identity(("h-source-reader-v2", python_version(), pysbd version, *"name:digest"))`
`[M, capture.py:25-48]`. p4 rewrites all seven names to their post-move dotted paths (including the
package member, `sentence_prediction` → `core.sentence`, resolving to its own minimal `__init__.py`
rather than the pre-move bare name — the p4 design-gate package-member constraint) and the digest
of every one of the seven files changes. Both the names and the bytes move, so the value cannot
survive.

**`h-checkpoint-reader-v1` → `v2`.** `sentence_prediction/checkpoint.py` `_reader_identity()` globs
`sentence_prediction/*.py` plus four named siblings, and hashes `path.relative_to(package.parent)`
into the identity string `[M, checkpoint.py:45-64]`. Once `package.parent` is `ladder/run`, the glob
**RAISES rather than changing value** (MEASURED 2026-09-22 by the p4 validator on the glob form left in place after the move: the four rail siblings resolved relative to `package.parent` no longer exist at the new root, and the reader raises `D1ArtifactSnapshotError` before any digest is composed). p4 therefore rewrites it to an
explicit dotted-name tuple, the idiom `capture.py` already uses, now living at
`ladder/run/checkpoint.py`.

**Membership is a faithful restatement (recommendation E-2).** The v2 tuple enumerates the **same
51-member file set** at the new dotted names (ACCEPTANCE RECONCILIATION, corrected here per the
architect's measured constraint, retracting the prior "core/sentence (18), ladder/run (29 including
the H-specific modules), composition (4) and core/probe (1)" figure, which folded the 7 `h_probe_*`
modules into `ladder/run` and omitted `ladder/capture` and the package member entirely — the earlier
figure counted 52 with an omission and a double-count, not 51):

| destination | count | members |
|---|---|---|
| `core.sentence` | 18 leaves + 1 package (`core.sentence` itself) | boundary, codec, consumer, corpus, head, law, model, numerics, objective, profile, records, recovery_state, rollout, runtime_precision, sampling, scoring, segmentation, spine |
| `ladder.run` | 22 | checkpoint, evaluation_records, execution, generation, journal, mechanism_evidence, native_verification, object_worker, runner, seals, span_diagnostics, span_probe, span_probe_artifacts, span_states, trace_capture, training, worker_command, worker_completion, worker_execution, worker_request, worker_snapshot, worker_transport |
| `ladder.capture` | 1 | capture |
| `composition` | 4 | capture_cli, cli, source_cli, span_cli |
| `core.probe` | 1 | score_gain |
| `kernel` | 2 | artifacts (`d1_artifact_snapshot`), lifecycle (`evidence.lifecycle`) |
| `core.latent` | 2 | metric, dynamics (`predictive_state.{metric,dynamics}`) |
| **total** | **51** | |

Narrowing to domain modules only would be a genuine narrowing of what the checkpoint binds, which
is a **scientific** change, not a structural one. It is not offered here.

---

## Guard teeth (house rule 5, both arms)

| mutation | must go RED |
|---|---|
| revert one of the four reader-closure rail modules (`d1_artifact_snapshot`, `evidence.lifecycle`, `predictive_state.metric`, `predictive_state.dynamics`) to `__legacy` | the checkpoint identity test, with a **resolution error**, not a digest mismatch |
| leave `checkpoint.py`'s package glob in place after the move | the checkpoint identity test raises; this is the failure mode p4 exists to convert into an explicit tuple |
| change either digest in a phase after p4 | the `frozen_after: p4` assertion on this door |
| fill a `new digest` cell without recomputing it at the p4 tip | the p4 ACCEPTANCE check, which compares the table's cell against a fresh computation |

The third row is the one that matters most for the whole sprint: these two values are frozen after
p4, and **no later phase may change either**.

**Row 1 performed (both arms).** Mutated `_READER_MODULES` in `ladder/run/checkpoint.py`,
replacing `"kernel.artifacts"` with `"__legacy.d1_artifact_snapshot"` (simulating the rail module
sitting back under quarantine): `uv run pytest tests/test_identity_epoch.py` went RED on
`test_checkpoint_reader_digest_matches_measured_regime_row` with
`ModuleNotFoundError: No module named 'entropic_substrate.__legacy.d1_artifact_snapshot'` — a
resolution error, not a digest mismatch, exactly the failure mode the row names. Reverted; the
same test file re-ran 12 passed, `git diff` on the mutated file empty.
