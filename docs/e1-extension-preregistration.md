# E1 extension pre-registration — replication seeds 6-9 (outcome-aware)

Committed BEFORE any extension run. This is NOT the registered stage 2 of
`docs/e1-preregistration.md`: that trigger (stage-1 CI straddling 0) was NOT
met — the stage-1 CI sits entirely above 0 (`docs/reviews/e1-review.md` §3),
so the original registration's confirmatory phase is CLOSED and its filed
verdict stands untouched. This extension is **outcome-aware by
construction**: it is motivated by the blind review's robustness findings
(leave-one-out fragility at conventional thresholds; the unexplained
{1,2}-vs-{3,4,5} clustering of paired differences) and by the review's own
"additional evidence" list. It therefore carries its OWN registered decision
rule — deliberately stricter than stage 1's — and reports under it.

## Design

- **Seeds {6, 7, 8, 9}**, arms `e1-random` / `e1-fisher`, configs unchanged,
  one run per pod (`--seed N --results results/<exp>-sN.jsonl`), A100-SXM4-80GB
  secure cloud (same class as stage 1). The container image digest is recorded
  in the launch record at pod-create time and must be identical across all 8
  pods.
- **Pairing** evidenced by `data_order_digest` exactly as in stage 1; a
  within-pair mismatch invalidates the pair (reported, excluded).
- **Exclusion rule** unchanged: non-finite final loss drops the whole pair,
  always reported.
- **Primary analysis:** cumulative paired analysis over seeds {1..9} —
  stage-1 captures plus extension captures, all read exclusively via
  `entropic_substrate.telemetry.final_eval_metric(path, "nats_per_byte")` on
  DVC-captured JSONLs (`capture-results`), never the live tracking server.
  `d_s = y_random,s − y_fisher,s`.

## Registered decision rule (3σ)

- **STRONGLY_SUPPORTED** iff the one-sided 99.865% lower confidence bound of
  mean(d) exceeds 0 — equivalently `mean(d)/(sd(d)/√9) > t_{0.99865, 8} =
  4.277`. Otherwise **NOT_STRONGLY_SUPPORTED**. This label is reported
  *alongside* — never replacing — the stage-1 filed verdict.
- Practical relevance is read against the ratified ε = 0.01 nats/byte
  (Amendments 1-2 of `docs/e1-preregistration.md`), same clause wording.
- **Power context** [projected from measured stage-1 sd 0.00261, mean
  0.00489]: if the extension pairs resemble stage 1's distribution, t ≈ 5.62
  → passes; if the clustering reflects real heterogeneity and sd doubles,
  t ≈ 2.81 → fails. The minimum homogeneous effect detectable at this bar is
  ≈ 0.0037 nats/byte. The bar is chosen precisely because it discriminates
  between those two worlds; a failure here is an informative result, not a
  nullity.

## Clause precedence (declared 2026-08-18, before any extension analysis)

The stage-1 cloud review (`docs/reviews/e1-review-cloud.md`, MAJOR-1) found
the original registration's verdict rule self-contradictory in the observed
region: the direction clause and the practical-relevance clause can both fire,
with no declared precedence. For this extension, declared before any
extension run completed or was analyzed: the two clauses answer DIFFERENT
questions and are BOTH always reported — direction
(STRONGLY_SUPPORTED / NOT_STRONGLY_SUPPORTED per the 3σ rule) is the primary
verdict label; practical relevance vs ε is a mandatory co-equal rider that
travels with every quotation of the verdict. Quoting the direction label
without its rider is a misquotation of this registration.

## Predeclared probes (report, never decide)

1. Covariates per pair: launch order, datacenter, pod id, run start/end from
   tracking-server metadata (labeled operator-supplied).
2. Spread comparison, seeds {1-5} vs {6-9} — does the two-cluster pattern
   replicate?
3. Train-vs-eval nats/byte gap per arm per run (overfit/saturation check;
   the byte arm makes ≈3.45 passes over the train split at these settings).
4. Convergence speed per arm: steps to reach the e0-byte terminal nats/byte
   (0.7738; directional readout, no decision rule).
5. Leave-one-out CI sensitivity at n = 9.

## Review

- Packet assembled by the committed `scripts/assemble_review_packet.py`
  extended over all 18 runs (A/B relabel, content scrub, fresh sealed
  mapping kept outside the repository until the review files).
- Reviewer runs in an **isolated cloud environment**: primary lane =
  Anthropic Files API + code-execution sandbox (scripted via the `ant` CLI);
  fallback lane = RunPod CPU pod running `claude --bare` with only the
  packet and an API key. Prompt verbatim from `docs/peer-review-protocol.md`.
- The extension's analysis session is again a fresh context and computes
  before reading any review artifact.

## Out of scope, queued behind this extension

- **FineWeb-Edu single-pass arm** — registered in the E2-era design as the
  gap-widening prediction (single-pass streaming is the regime where a
  statistics-first init should matter most: no repeated exposure to relearn
  what the init encodes).
- **Drift / substitutability probes** — require checkpoint capture (the
  embedding tables), a separate declared lane; JSONL captures cannot answer
  them.
