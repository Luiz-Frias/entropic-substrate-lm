# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## North star

Prove or falsify one thesis: statistics-first symbol encoding _induces_ spherical geometry
rather than assuming it — under the square-root map, byte transition distributions land on
the positive orthant of the unit hypersphere with Fisher–Rao as (4×) the angular metric —
and downstream of the substrate, geometry belongs in the layers, not around the model:
projection/retraction as layer operators, curvature as a learnable parameter. The repo is a
controlled ablation ladder (E0→E3, one variable per rung; see `docs/e-3-instruction.md`),
not a framework. The readout is nats/byte plus per-head κ-trajectory diagnostics.

## Horizon

E4 in full (`docs/e-4-instruction.md`), then follow-on:

- **E4a** — fixed-W=8 chunk-superposition control on the settled E3 winner (HoloByte
  bind/unbind + micro-decoder pattern), isolating "hierarchy at all" from "learned
  boundaries". **Precondition:** E3 concluded with 3-seed nats/byte numbers.
- **E4b** — learned dynamic chunking per H-Net (`references/hnet`). Toolchain is
  quarantined to the container (`mamba_ssm`/`causal_conv1d`, possibly nvcc source builds);
  the pure-torch escape hatch (small transformer encoders in place of Mamba-2 stages) is
  acceptable for a first pass at ~85M scale.
- **Follow-on** — if E3 shows signal, graduate κ-components from the projection pattern to
  `geoopt.ManifoldParameter` + `RiemannianAdam`.

## Commands

```bash
mise trust && mise install   # pinned toolchain (python 3.12, uv, just, prek, ...)
just setup                   # tools + uv sync --all-groups + prek hooks
just check                   # quality gate: fmt-check + lint + typecheck + unit tests (= CI)
just test                    # unit tests only (CPU, <60s)
just test-all                # unit + integration
PYTHONPATH=src uv run pytest tests/test_kappa.py::test_name -m unit   # single test
just train exp=e1-fisher     # exp ∈ {e0-byte, e0-bpe, e1-random, e1-fisher, e2, e3, tiny}
just eval ckpt=models/e3-latest.pt
just download-data           # enwik8 → data/, then: dvc add data/enwik8 && dvc push
just container-build         # podman build (CUDA 12.4 image; deploy: docs/deploy-h100.md)
just pc-ci                   # prek commit + pre-push stages (CI parity)
```

mise loads `.env` into the environment; `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` there
are the R2 S3-protocol credentials the DVC remote needs (remote itself is committed in
`.dvc/config`, bucket `entropic-substrate-lm`). Commit messages are gitlint-checked:
Conventional Commits, title ≤ 72 chars.

## Architecture

Three swappable components under a standard decoder-only transformer, selected per
experiment by `configs/*.yaml` (init × block × norm) — `docs/architecture.md` is the
authoritative version:

1. **Substrate init** (`init/fisher.py`) — √p of Laplace-smoothed next-byte transition
   rows → points on S²⁵⁵ ∩ ℝ₊, lifted 256→`d_model` by a fixed semi-orthogonal QR map.
   Control arm: uniform-random unit vectors. This is the only thing E1 varies.
2. **Backbone** (`models/blocks.py`) — nGPT-style: all embedding-dim vectors unit-norm,
   affine-free `Norm()` as the retraction step, residual `h ← Norm(h + α(h_block − h))`
   with learnable eigen-learning-rates. Optimizer stays plain AdamW; geometry lives
   entirely in the forward pass. This is the E2 swap.
3. **κ-attention** (`models/blocks.py`) — the novel artifact (redesigned 2026-08-11, design
   record in `docs/e-3-instruction.md`): per-head attention logits `−d_κ(q,k)²` with one
   learnable clamped curvature per head (init spread −1/0/+1); E2 is the exact κ=0 member
   (softmax shift-invariance), heads are the mixture-over-geometries. The earlier
   `layers/kappa_norm.py` is retained but measured radially-annihilated (`≡ UnitNorm`,
   pinned by regression test — curvature is relational and cannot act on a single point at
   a chart origin under a unit-norm contract). The anchored arms `e3-anchor` (per-head κ +
   global asymmetric anchors) and `e3-vantage` (global κ + per-head anchors; attribution
   ablation) are code-ready but execution-gated on the E3 readout: per-head κ divergence.
   Symmetric anchors are exactly null (isometry) — pinned by test. `geometry/kappa.py` implements the
   κ-stereographic primitives directly; the geoopt submodule is a **test-time oracle
   only**, never a runtime dependency.

**Non-negotiable numerics:** all geometric math runs in FP32 inside
`torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast; κ is
clamped and inputs are projected into the component domain with a boundary epsilon before
any map; the load-balance loss must stay finite and gradients must reach κ (unit-tested).
`references/holobyte/holobyte.py` demonstrates the failure mode this prevents.

**Data flow:** enwik8 is DVC-tracked in `data/` (pointer `.dvc` files are git's half of the
contract — `data|models|results/*` are gitignored _except_ `*.dvc`); FineWeb-Edu is
streamed, never local. `train.py` writes `results/*.jsonl` (nats/byte, κ trajectories,
gate histograms) and `models/*.pt` checkpoints; both sync to the R2 remote via `dvc push`.

**Tests** (`tests/`, pytest markers `unit` / `integration`) check the vendored κ-math
against geoopt and pin the numerics invariants above.
