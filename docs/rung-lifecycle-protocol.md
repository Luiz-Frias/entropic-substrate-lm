# Rung-lifecycle protocol — the preregistration flywheel

Ruled by the operator 2026-08-17/18. The tension this resolves: honest
preregistration requires inputs that only earlier rungs produce (measured
σ_seed sets stage sizes; the prior verdict sets arm assignment), while
post-hoc design invites bias. The flywheel fixes the **rules** early and binds
the **numbers** late: each rung's preregistration is committed after the prior
rung's verdict and before its own first confirmatory run — the same move
adaptive clinical designs make with pre-specified adaptation rules.

## The cycle, per rung N

1. **RUN** — confirmatory runs exactly per rung N's committed
   `docs/eN-preregistration.md` (registered seeds only; pilots declared and
   excluded). Pods stay credential-free; results reach the tracking server;
   `capture-results` pulls them into DVC, admissibility-gated.
2. **ANALYZE** — a FRESH context session (implementation sessions never
   analyze; analysis sessions never implement) reads ONLY the DVC-captured
   JSONLs through `telemetry.py`, computes the preregistered estimator, and
   writes the verdict artifact (format below).
3. **REVIEW** — the double-blind packet per `docs/peer-review-protocol.md`
   (arm relabeling, sealed mapping); the filed review travels with the
   verdict.
4. **FILE** — verdict artifact + review committed to git.
5. **PREREGISTER N+1** — drafted from N's measured quantities (σ_seed →
   stage sizes; verdict → arm assignment; endpoints declared primary vs
   exploratory — E3's must declare nats/byte vs κ-trajectories explicitly),
   committed BEFORE any N+1 confirmatory run.
6. **GATE** — automation consumes the filed verdict, never computes one:
   `run_ladder.sh` (FOR-889) refuses to advance past rung N without
   `docs/verdicts/eN-verdict.json` on record — a DECISION-NEEDED stop, with
   an explicit override env for deliberately ungated runs.

## Verdict artifact format

Two files per rung under `docs/verdicts/`: `eN-verdict.md` (the human
analysis narrative) and `eN-verdict.json` (the machine half the ladder
checks):

```json
{
  "rung": "e1",
  "preregistration": "docs/e1-preregistration.md",
  "preregistration_commit": "<sha>",
  "stage": 1,
  "n_pairs": 5,
  "mean_d": 0.0,
  "sd_d": 0.0,
  "ci": [0.0, 0.0],
  "alpha_per_look": 0.025,
  "verdict": "SUPPORTED | NOT_SUPPORTED | PRACTICALLY_NULL | INCONCLUSIVE",
  "inputs": [{ "path": "results/captured/<file>.jsonl", "md5": "<md5>" }],
  "review": "docs/reviews/eN-review.md"
}
```

The ladder checks existence and reads `verdict`; it never computes one.
Judgment stays human — gates are sanity + selection only.

## Protocol constants (carry across rungs unchanged)

- Paired-difference CI estimator family; total alpha split across looks.
- Exclusion rule: non-finite final loss drops the PAIR, always reported.
- Analysis inputs are content-addressed captures only — never the live
  tracking server.
- Blind review with arm relabeling on every confirmatory verdict.
- Pilots are declared at registration time and never enter confirmatory sets.

## Bounds

- E4 preregisters only after its implementation exists
  (`docs/e-4-instruction.md`) — a design for unbuildable runs is fiction.
- Any deviation from a committed preregistration is exploratory by
  definition and must say so where the result is reported.
