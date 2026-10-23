# for-1673-structurally-absent-sentence-rows: say WHY a zero-sentence mechanism row has no range

DRAFT 2026-10-22 by the conductor (wave C, lane of a K=3 mass-patch-fanout; issue FOR-1673, from the H0 blind
review "unreported nulls" finding: 18 prediction rows per arm of HEpP7g carry `mechanism.sentences.range: null`
with `count: 0`). Anchors measured at origin/main `70d2b5f1` with `grep -n`; the implementer re-verifies them in
recon. The Linear ticket is authoritative. **DG-A lane: the touched module is a reader-tuple member.**

## Branch

| field                | value                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1673-structurally-absent-sentence-rows`                                                    |
| parent               | the wave base branch (wave B's zipped tip + the briefs commit)                                      |
| cascade `phase.mode` | `normal` (the vocabulary already exists in the same module; the row shape is extended additively)   |
| publish              | none in the lane; the lane returns `REPIN-HANDOFF` and the conductor lands the single commit (DG-A) |

## Why (measured at `70d2b5f1`)

- `src/entropic_substrate/ladder/run/mechanism_evidence.py`: `_norms` emits `"range": _range(norms) if norms.numel() else None` (`:30`); `collect_mechanism` (`:34`) already emits `{"schema": SCHEMA, "status": "structurally_absent", "reason": "codec_warmup"}` for warm-up rows (`:38-41`); the reader side at `:132-135` checks `count == 0` then `row["range"] is not None` ("empty history must not invent norm measurements") before `_read_range`.
- Readers measured: `collect_mechanism` / `_read_range` are referenced ONLY from `ladder/run/mechanism_evidence.py` and `ladder/run/execution.py` (grep over `src`). `ladder/analyze/h_aggregate.py` has ZERO hits for `mechanism`, `structurally_absent` or `"range"`: the ticket's "consumers to check" entry for the aggregate is measured EMPTY, so the aggregate regression check is a sanity run, not a code change.
- Tests: `tests/test_sentence_mechanism_evidence.py` (7 tests, `:29` step records + journal replays, `:128` missing readout not misclassified as warm-up, `:152` zero carrier measurements retained); `tests/test_sentence_execution.py` and `tests/test_sentence_training.py` also name the module.
- Reader tuple (`ladder/run/checkpoint.py:65-118`) MEMBER: `ladder.run.mechanism_evidence` (and `ladder.run.execution` if its reader call changes). DG-A protocol applies.
- The HEpP7g capture (`results/h0-plumbing-1k-v3-HEpP7g`, 4 files, provisioned by lanePolicy.installCmd) is the regression fixture: its 18 null-range rows per arm must still load; `structurally_absent` applies to NEW runs only (ticket out-of-scope line).

## Scope

1. **C1, tests RED at the parent** (`tests/test_sentence_mechanism_evidence.py`, additive): a prediction readout with zero sentence states yields a `sentences` entry carrying `"range": null`, `"status": "structurally_absent"`, `"reason": "no_sentence_states"` and `count: 0`; the reader accepts that row and still refuses a non-null range at count 0; a non-empty readout carries no `status` key (control).
2. **C2, implementation**: `_norms` (or its caller) adds the status/reason pair when `norms.numel() == 0`; the reader at `:132` accepts both the old bare-null shape (read-old window for HEpP7g) and the new shape; `SCHEMA` unchanged unless the gate rules a bump (then the read-old row is dated).
3. **C3, regression**: run the H aggregate over `results/h0-plumbing-1k-v3-HEpP7g` at the staged tip and quote its exit code and row counts (the 18 rows per arm are the fixture).
4. **C4, DG-A hand-off**: as in FOR-1672: stage, save the cached diff, quote the expected RED (digest tests only) and the regime rows, return `REPIN-HANDOFF`; the conductor lands the single commit with the re-pin.

## Guard teeth by mutation

| id | mutation                                             | expected RED                |
| -- | ---------------------------------------------------- | --------------------------- |
| M1 | emit the bare null again (no status/reason)          | the status pin RED          |
| M2 | emit `reason: codec_warmup` on the zero-sentence row | the reason pin RED          |
| M3 | reader rejects the old bare-null shape               | the HEpP7g read-old pin RED |
| M4 | emit `status` on a non-empty readout too             | the control pin RED         |

## Allowed files

`src/entropic_substrate/ladder/run/mechanism_evidence.py`, `tests/test_sentence_mechanism_evidence.py`, `tests/test_sentence_execution.py` (additive pins only); `src/entropic_substrate/ladder/run/execution.py` only if the reader call must change (report why). Never: `tests/test_identity_epoch.py` (the conductor re-pins), `ladder/analyze/*` (measured: no mechanism reader there), the capture files, `docs/*`.

## Acceptance

C1 RED at the parent, GREEN at the staged tip; M1-M4 RED one at a time and restored with the RED output quoted; the HEpP7g aggregate run quoted; `just check` at the staged tip RED ONLY on the regime-row digest tests and GREEN after the conductor's re-pin commit; the exit-gate runner green at the conductor's commit; diff confined to the Allowed files.
