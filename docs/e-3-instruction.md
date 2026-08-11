# E-3 Instruction — running the ladder E0 → E3

Each rung changes exactly one variable. Do not skip rungs: a positive E3 without its E1/E2
controls is uninterpretable, and each rung is independently publishable as a negative result.

## Ground rules (all experiments)

- Metric: **nats/byte** on a held-out split. For the BPE baseline, convert with the *measured*
  bytes-per-token ratio on the eval split (never the folklore 4.0 constant).
- Matched parameter budgets (~85M) — the BPE arm spends ~38M on its embedding table and gets
  fewer layers; that asymmetry is the point being measured, so report both param and FLOP
  counts.
- Fixed seed per run; three seeds before believing any gap < 0.05 nats/byte.
- Every run appends to `results/<exp>.jsonl` (the seed is recorded in each run's
  `run_start` record — pass `--results results/<exp>-s<seed>.jsonl` to split files per seed);
  checkpoints go to `models/` and get `dvc add`-ed. Set the DVC remote before long runs (`dvc remote add -d <name> <url>`).
- Smoke-test any change with `--tiny` (CPU, minutes) before burning GPU hours.

## E0 — baselines

```
just train exp=e0-byte     # byte-level vanilla transformer, 256-way head
just train exp=e0-bpe      # GPT-2 BPE vocab, tied embeddings
```

Purpose: anchor numbers on enwik8 + FineWeb-Edu stream. Sanity targets: byte-level enwik8
transformers of this scale land in the ~1.0–1.3 bits/byte (≈0.7–0.9 nats/byte) neighborhood;
if E0 is wildly off, fix the harness before proceeding.

## E1 — substrate init swap (cheapest, highest information)

```
just train exp=e1-random   # uniform-random unit vectors (control)
just train exp=e1-fisher   # √p Fisher–Rao init from corpus transition stats
```

One variable: `init-fisher --corpus data/enwik8/enwik8` (the file, not the directory) writes
the (256, d_model) table to `data/fisher_init_d768.pt`, which the e1-fisher/e2/e3 configs read
(the trainer recomputes it from the corpus if the file is missing); both runs are otherwise
bit-identical. Read out:

1. **Convergence speed** — steps to reach the E0-byte terminal loss.
2. **Substitutability probe** — at init and at convergence, does pairwise Fisher–Rao angle
   predict byte substitutability (KL of context distributions) better than the learned
   embedding cosine of the random arm? (The falsifiable pillar: geometry is induced by
   statistics.)
3. **Drift** — how far the trained table moves from its init (mean angular displacement).
   Small drift + equal loss = the statistics already were the geometry.

## E2 — hypersphere-native backbone

```
just train exp=e2
```

Swaps blocks for nGPT-style Norm-as-retraction with learnable α residual interpolation.
Expected from the literature: materially faster convergence per step at equal quality — if E2
converges *slower*, suspect the α init or a missed normalization (check every matmul input is
unit-norm). Compare against E1-winner init; keep the winner for E3.

## E3 — GatedKappaNorm (the experiment)

```
just train exp=e3
```

Replaces the norm layer with the gated κ-stereographic MoE (K=3, κ init −1/0/+1, top-1 gate,
load-balance λ per config). Read out, in priority order:

1. **κ trajectories** (logged every eval): prediction — κ drifts negative on hierarchy-heavy
   data (code), positive on cyclic/periodic structure, ~0 where Euclidean suffices. Flat-at-init
   κ with healthy loss = the layer is being bypassed; check gate entropy.
2. **Gate usage histograms**: collapse to one component ⇒ raise load-balance λ (known MoE
   failure; the collapse target is usually the Euclidean component).
3. **nats/byte vs E2** at matched budget.
4. **Robustness**: eval on byte-corrupted held-out text (typo noise) — the continuity argument
   predicts the geometric arms degrade more gracefully than E0-bpe.

## Failure triage

| Symptom | First suspect |
|---|---|
| NaN/inf in E2/E3 | geometry op running under autocast — audit FP32 blocks |
| κ pinned at clamp boundary | clamp too tight or LR too high on κ (give κ its own param group, 10x lower LR) |
| Gate collapse | load-balance λ too low; try top-2 |
| E1-fisher ≈ E1-random exactly | lift not applied / init overwritten by weight init — assert at step 0 |
| BPE arm "wins" implausibly | bytes-per-token conversion — recompute on the actual eval split |

## After E3

E4 (H-Net dynamic chunking, `references/hnet`) replaces fixed sequence granularity with learned
boundaries — separate instruction to be written when the ladder gets there. Do not conflate it
with E3: chunking and geometry are different variables.
