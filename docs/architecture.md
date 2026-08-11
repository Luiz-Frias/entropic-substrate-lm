# Architecture

One thesis, three swappable components under a standard decoder-only transformer, plus an
experiment harness that isolates each swap. Nothing here is a framework — it is a controlled
ablation ladder (see `e-3-instruction.md`).

## The thesis being tested

Entropic initialization and hyperspherical initialization of symbols are not competing choices:
under the square-root map, a probability distribution `p` becomes a point `√p` on the positive
orthant of the unit hypersphere, and the Fisher–Rao metric between distributions is exactly
(4x) the angular metric on that sphere
([Fisher information metric](https://en.wikipedia.org/wiki/Fisher_information_metric)).
So "encode symbols by their statistics" *induces* spherical geometry rather than assuming it.
Downstream of the substrate, geometry is treated as a **layer operator** (projection /
retraction), not a container — and curvature is a **learnable parameter**, not a catalog of
shapes.

## Component 1 — √p Fisher substrate init (`init/fisher.py`)

- Count next-byte transition distributions `p(· | b)` over a corpus sample (256×256, Laplace
  smoothed).
- Row-wise `√p` places each byte on `S²⁵⁵ ∩ ℝ₊` with Fisher–Rao as arc length.
- A fixed semi-orthogonal lift (QR of a seeded Gaussian) maps 256 → `d_model` isometrically.
- Control arm: uniform-random unit vectors (`random_sphere_init`) — the maximum-entropy prior,
  which is only well-defined *because* the sphere is compact.

This is the E1 swap: identical architecture, only the embedding init differs.

## Component 2 — hypersphere-native backbone (`models/blocks.py`)

nGPT-style blocks per [Loshchilov et al.](https://arxiv.org/abs/2410.01131): all embedding-
dimension vectors kept unit-norm, `Norm()` without affine params acting as the retraction step
of Riemannian optimization, and residual recombination
`h ← Norm(h + α(h_block − h))` with learnable eigen-learning-rates `α`. Parameters stay
Euclidean and the optimizer stays AdamW — geometry lives entirely in the forward pass
(projection pattern, torch.compile-friendly). This is the E2 swap.

## Component 3 — GatedKappaNorm, the MoE-over-geometries layer (`layers/kappa_norm.py`)

The novel artifact of this repo. Replaces the block normalization layer in E3:

```
x (fp32, autocast disabled)
  ├─ gate(x) → sparse top-k weights gᵢ over K components      (Switch-style router)
  ├─ per component i: projx_κᵢ → logmap0_κᵢ                    (into tangent space)
  ├─ gated tangent-space mix → κ_eff = Σᵢ gᵢκᵢ
  └─ expmap0(κ_eff) → renorm                                  (single retraction out)
```

- Each component carries a learnable curvature `κᵢ` (init −1, 0, +1; clamped), using the
  [κ-stereographic model](http://proceedings.mlr.press/v119/bachmann20a/bachmann20a.pdf) so the
  sign of curvature is differentiable through zero (Taylor-guarded at `κ → 0`).
- Sparse gating + load-balancing auxiliary loss follows the
  [Switch Spaces](https://arxiv.org/abs/2102.08688) argument: route each input to the geometry
  that fits it instead of averaging all geometries.
- Product-of-constant-curvature-factors framing per
  [Gu et al., ICLR 2019](https://openreview.net/forum?id=HJxeWnCcF7).
- `geometry/kappa.py` implements the primitives directly (vendored math, geoopt used as a
  test-time oracle only — see `tests/`).

## Non-negotiable numerics

- **All geometric math runs in FP32** inside `torch.amp.autocast(enabled=False)` blocks —
  normalization, log/exp maps, cosine logits, distances. The HoloByte reference
  (`references/holobyte/holobyte.py`) demonstrates the failure mode this prevents.
- κ clamped to a bounded interval; inputs projected to the component domain
  (`‖x‖² < −1/κ` for κ < 0) with a boundary epsilon before any map.
- Load-balance loss must be finite and gradients must reach κ (unit-tested).

## Data flow

```
scripts/download_enwik8.py → data/enwik8  (dvc add data/enwik8; remote set by owner)
HuggingFaceFW/fineweb-edu sample-10BT     (streamed, no local copy)
        │
   data/ loaders (raw utf-8 bytes; BPE path via tiktoken gpt2 for E0-bpe only)
        │
   backbone (config-selected: init × block × norm)
        │
   train.py → results/*.jsonl (nats/byte, κ trajectories, gate histograms)
            → models/*.pt     (dvc-track large checkpoints)
```

## Deliberate exclusions

- **H-Net dynamic chunking (E4)** — deferred; requires `mamba_ssm` source builds and is a
  separate variable. The submodule is pinned at `references/hnet` for when the ladder gets
  there.
- **Riemannian optimizers** — v1 uses the projection pattern everywhere. If E3 shows signal,
  graduating κ-components to `geoopt.ManifoldParameter` + `RiemannianAdam` is the follow-up.
