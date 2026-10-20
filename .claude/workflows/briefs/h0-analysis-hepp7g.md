# H0 fresh-session analysis brief: wave h0-plumbing-1k-v3.HEpP7g

Authority: H0 "complete tracer" exit (plan `docs/hierarchical-latent-predictor.plan.md`, stage table).
This is a PLUMBING analysis with a test-only gate. It carries no scientific claim, no H1 eligibility
beyond "the machinery exists", and no spending authority. The session that runs it must be a FRESH
context that has not seen the live wave (evidence design ruled 2026-08-17).

## What the analysis session reads, and only that

- `results/h0-plumbing-1k-v3-HEpP7g.dvc` (this commit) pulled with `dvc pull`, then the four files under
  `results/h0-plumbing-1k-v3-HEpP7g/`: per arm one `metrics.jsonl` (1009 rows, one `run_start`, one
  `run_end`) and one `worker-log.jsonl`, named `h-worker-<request_sha256>-<mlflow run id>-*.jsonl`.
  Arms: `a163161f` = magnitude (pod `211tvsc7277d22`), `988d8388` = sphere (pod `euikqmf95pnrga`).
- Readers: `ladder/capture/reader/telemetry.py` for admissibility (the only read side), the H lane rules in
  `docs/architecture.md` "Evidence plane", `h-inspect-training` for receipts if R2 custody is consulted.
- Never the live MLflow server, never a pod, never the conductor's ledger narrative as evidence.

Captured 2026-10-20 14:37Z by `capture-results --experiment h-series --dest results/h0-plumbing-1k-v3-HEpP7g`
from this worktree with a private DVC site cache; pass admitted, pushed, cloud-verified (exit 0).

## Questions the analysis answers (each with the file and row that answers it)

1. Admissibility: do both arms pass the H lane predicate (one `run_start`, one `run_end` with `status: ok`,
   committed steps on one global axis through 1000, evaluation rows at 500/750/1000)?
2. Lineage: do the `run_end` rows and the worker logs show both carriers on the same source identity,
   profile config_hash, image and code commit (differing only in carrier and run identity)?
3. Final leg: per stage, the `admission.start`/`admission.end` deltas (memo fresh/shared, seals walked,
   evaluations, seconds) and stage wall seconds, both arms, in one table with evidence class per row.
4. Stochastic rollout: what the generation stage recorded (continuation count per prompt, boundary emitted
   or explicit truncation outcome), as far as the captured files show it; name what needs R2 custody.
5. Recovery: state plainly that no resume was exercised; this exit stays open.
6. Numerics: any nonfinite, nonmonotonic or missing field; any `FAILED` or stranded sink event.

## Output

`docs/h0/analysis-hepp7g-2026-10-20.md` (new directory), tables over prose, measured numbers only, each
table row labelled MEASURED or DERIVED_FROM_MEASURED, no estimate anywhere, no em-dashes. End with the
verdict proposal in the plan's vocabulary: `tracer_complete` (all exits proven), `tracer_partial` (naming
each unproven exit), or `numerical_defect` / `missing_evidence`, plus the exact list of what a reviewer
must re-derive. The review (a different fresh session) and the filing stay separate steps.
