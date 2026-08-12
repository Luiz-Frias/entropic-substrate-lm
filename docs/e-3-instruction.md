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

## E3 — κ-attention (the experiment)

```
just train exp=e3
```

Replaces the attention similarity kernel: per-head logits go from cosine `q·k` to the
κ-stereographic squared distance `−d_κ(q, k)²`, one learnable clamped curvature per head
(init spread over −1/0/+1). E2 is the exact κ = 0 member of this family — for unit vectors
`−d_0(q, k)²/2 = q·k − 1` and softmax kills the constant — so the rung still changes exactly
one variable. Heads are the mixture-over-geometries: no router, no load-balance machinery, no
collapse failure mode. Read out, in priority order:

1. **Per-head κ trajectories** (logged every eval): prediction — heads drift negative on
   hierarchy-heavy data (code), positive on cyclic/periodic structure, ~0 where Euclidean
   suffices. All heads pinned at 0 with healthy loss = the information landscape is flat at
   this scale; that is a *result* (it kills E3″ cheaply), not a failure.
2. **Head specialization**: do different heads settle at different κ on mixed corpora?
   Divergence is the measured trigger that unlocks E3″ (below).
3. **nats/byte vs E2** at matched budget.
4. **Robustness**: eval on byte-corrupted held-out text (typo noise) — the continuity argument
   predicts the geometric arms degrade more gracefully than E0-bpe.

### Design record — why not the norm layer (2026-08-11)

The original E3 artifact (gated κ-stereographic **norm** layer) is measurably a no-op:
`logmap0`/`expmap0` at the chart origin are radial maps, and the unit-sphere retraction the
nGPT backbone requires erases radius — so the layer output equals `x/‖x‖` to float precision
for every K/top-k/κ configuration, and ∂loss/∂κ ≡ 0 (exactly in top-1; ~1e-7 float residue
in top-2). Curvature is a second-order, *relational* property — unobservable from a single
point at a chart origin. `layers/kappa_norm.py` is retained with a regression test pinning
the equivalence (`GatedKappaNorm ≡ UnitNorm`); the geometry moved to where points actually
meet: the attention kernel. A second trap, recorded so nobody re-falls into it: a *shared*
basepoint in the distance kernel is also null — Möbius translation is an isometry, so
`d_κ((−p)⊕q, (−p)⊕k) = d_κ(q, k)` for every p.

### E3″ / E3‴ — anchored arms (code-ready; execution behind the measured trigger)

Both arms are implemented and test-pinned but **run only if E3 shows per-head κ
divergence** — the code landed early by explicit operator decision (2026-08-11) so the
parallel pod can fire the moment the trigger shows; the gate moved from "build" to "run"
and the ladder runner enforces it.

- **`e3-anchor` (E3″)** — per-head κ plus ONE global asymmetric anchor pair
  (`p_q ≠ p_k`, zero-init): the minimal symmetry break. Zero-init means the arm starts
  exactly at the E3 point and learns its vantage away from it (gradient-alive at zero,
  pinned by test).
- **`e3-vantage` (E3‴)** — global κ plus per-head anchors: the attribution ablation.
  Against `e3` (curvature only) and `e3-anchor` (curvature + shared vantage) it answers
  *is the win coming from curvature or from vantage?* It forfeits the per-head κ readout
  by design and is not a mainline rung.
- Two traps, both pinned as tests so nobody re-falls into them: symmetric anchors
  (`p_q = p_k`) leave the kernel **exactly** invariant (Möbius translation is an
  isometry), and the operating radius must stay a bounded fraction of the hyperbolic
  ball — at the ball edge, near-antipodal pairs put `artanh` in its quasi-divergent
  regime (fp32 noise amplified ~1e5×, saturated gradients).

Horizon beyond both: position-dependent curvature κ(x) with learned transport — the
atlas — which no rung below has yet earned.

## Failure triage

| Symptom | First suspect |
|---|---|
| NaN/inf in E2/E3 | geometry op running under autocast — audit FP32 blocks |
| κ pinned at clamp boundary | clamp too tight or LR too high on κ (give κ its own param group, 10x lower LR) |
| All κ stuck at exactly 0 | first verify the ∂κ ≠ 0 unit test is green; if it is, flat may be the honest answer — report it |
| E1-fisher ≈ E1-random exactly | lift not applied / init overwritten by weight init — assert at step 0 |
| BPE arm "wins" implausibly | bytes-per-token conversion — recompute on the actual eval split |

## After E3

E4 (H-Net dynamic chunking, `references/hnet`) replaces fixed sequence granularity with learned
boundaries — separate instruction to be written when the ladder gets there. Do not conflate it
with E3: chunking and geometry are different variables.
