# Read-old table — reader identity bumps across the hexagonal refactor

**Status: TEMPLATE.** The `old digest` column is filled at **p0**, when the identity fixtures are
computed and pinned at the epoch. The `new digest` column is filled at **p4**, when both readers are
rewritten and recomputed. A row with an unfilled digest is an unmet precondition, not a placeholder:
p4's ACCEPTANCE asserts both columns are populated before the phase can go green.

This file is a **door** (`kind: door`, `frozen_after: p4`). After p4 no phase may change either
value.

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
| h-source-reader | `h-source-reader-v2` | per-regime rows in `tests/test_identity_epoch.py::_SOURCE_REGIME_TABLE` (keyed (python, pysbd); two regimes pinned at p0) | `h-source-reader-v3` | `<FILL AT p4>` | Replay against the p0-epoch commit only. Three local `h-source-capture-v2` captures, none DVC-tracked. No cross-reader resume. | **p4-h-tracer** (serial, `liveRequired:true`, `just ship-it`) |
| h-checkpoint-reader | `h-checkpoint-reader-v1` | per-regime rows in `tests/test_identity_epoch.py::_CHECKPOINT_REGIME_TABLE` (keyed (python, torch); two regimes pinned at p0) | `h-checkpoint-reader-v2` | `<FILL AT p4>` | Replay against the p0-epoch commit only. Existing H checkpoints do not resume across the bump. No cross-reader resume. | **p4-h-tracer** (serial, `liveRequired:true`, `just ship-it`) |

**p0 epoch commit:** `bbe7976df0710af5bdc00879b1b8145568169034`
**Shape amendment (conductor, 2026-09-21, p4 brief audit; door `frozen_after: p4` not yet frozen):** p0 pinned each reader's old digest as a COMPOSED value per measured regime (Python x torch for the checkpoint reader, Python x pysbd for the source reader), because the digest is regime-dependent (MEASURED at p0, see the module docstring of `tests/test_identity_epoch.py`). The 'old digest' column therefore points at those regime tables rather than carrying one literal; p4 fills the 'new digest' column the same way, one row per regime it measures, and records the tip SHA below. The read-old window itself is unchanged: replay against the p0-epoch commit only.

**p4 tip commit:** `<FILL AT p4 — the 40-hex SHA the new digests were computed at>`

Both SHAs are full 40-hex, not short. A verdict binds to a derived immutable identifier; a moved
epoch voids the row and it is restated, never reinterpreted.

---

## Why each identity changes, measured

**`h-source-reader-v2` → `v3`.** `sentence_prediction/capture.py` `_READER_MODULES` is seven literal
dotted names, and the identity is
`content_identity(("h-source-reader-v2", python_version(), pysbd version, *"name:digest"))`
`[M, capture.py:25-48]`. p4 rewrites all seven names to their post-move dotted paths and the digest
of every one of the seven files changes. Both the names and the bytes move, so the value cannot
survive.

**`h-checkpoint-reader-v1` → `v2`.** `sentence_prediction/checkpoint.py` `_reader_identity()` globs
`sentence_prediction/*.py` plus four named siblings, and hashes `path.relative_to(package.parent)`
into the identity string `[M, checkpoint.py:45-64]`. Once `package.parent` is `ladder/run`, the glob
**RAISES rather than changing value**: there is no valid root. p4 therefore rewrites it to an
explicit dotted-name tuple, the idiom `capture.py` already uses.

**Membership is a faithful restatement (recommendation E-2).** The v2 tuple enumerates the **same
file set** at the new dotted names. After p4 those files sit in `core/sentence` (18), `ladder/run`
(29 including the H-specific modules), `composition` (4) and `core/probe` (1) `[M]`. Narrowing to
domain modules only would be a genuine narrowing of what the checkpoint binds, which is a
**scientific** change, not a structural one. It is not offered here.

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
