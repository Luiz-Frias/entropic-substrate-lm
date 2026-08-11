# References — prior art map

Pinned as git submodules under `references/` (`git submodule update --init --depth 1`).
Each entry states exactly what this repo takes from it and what it deliberately does not.

## references/holobyte — HoloByte (Khasia, 2026)

[Paper (arXiv 2603.16917)](https://arxiv.org/abs/2603.16917) ·
[Repo](https://github.com/VladimerKhasia/HoloByte)

Tokenizer-free modeling via bytes projected to a unit hypersphere, positional binding through
orthogonal rotations, chunk superposition, micro-decoder unbinding. **Taken:** the
FP32-geometry-inside-autocast pattern; the FineWeb-Edu streaming loader shape; the dual
CE + latent-MSE loss idea as a reference point; the parameter-parity comparison design.
**Not taken:** its `torch.randn` byte manifold init (that slot is exactly what E1 replaces) and
its assumed 4.0 bytes/token BPE conversion (we measure). Single-author, small-corpus results —
treat its numbers as directional.

## references/ngpt — nGPT (Loshchilov et al., NVIDIA, ICLR 2025)

[Paper (arXiv 2410.01131)](https://arxiv.org/abs/2410.01131) ·
[Repo](https://github.com/NVIDIA/ngpt)

Fully hypersphere-native transformer: all vectors unit-norm, normalization as Riemannian
retraction, learnable eigen-learning-rate residuals, 4–20x faster convergence. **Taken:** the
E2 block design; the "geometry as layer operator, params stay Euclidean, optimizer stays
AdamW" doctrine; and their post-step per-matrix renormalization — after every optimizer step,
each block's six projection matrices are re-projected to unit norm along the embedding axis
(dim=1 for stream-consuming matrices, dim=0 for the c_proj pair, mirroring the reference
convention). **Not taken:** SLERP residual recombination — E2 uses the LERP + renorm
approximation the reference itself trains with.

## references/geoopt — geoopt (Kochurov et al.)

[Paper (arXiv 2005.02819)](https://arxiv.org/abs/2005.02819) ·
[Repo](https://github.com/geoopt/geoopt)

Riemannian optimization for PyTorch; `geoopt.Stereographic(learnable=True)` is the reference
κ-stereographic implementation, plus `ProductManifold`/`Scaled` (the Gu et al. construction)
and `RiemannianAdam`. **Taken:** used as the *test oracle* — `geometry/kappa.py` must agree
with it within tolerance across κ < 0, κ → 0, κ > 0. **Not taken (yet):** manifold parameters
and Riemannian optimizers at train time; that is the post-E3 graduation path.

## references/hnet — H-Net (Hwang, Wang & Gu, 2025)

[Paper (arXiv 2507.07955)](https://arxiv.org/abs/2507.07955) ·
[Repo](https://github.com/goombalab/hnet)

End-to-end learned dynamic chunking; byte-level H-Net beats BPE transformers, recursion gives
multi-level abstraction. **Taken:** nothing at E0–E3 — pinned for E4, where its routing +
smoothing + ratio-loss mechanism replaces fixed granularity. Note its heavier toolchain
(PyTorch ≥ 2.5.1, `mamba_ssm` built from source).

## Papers without submodules

- [Gu, Sala, Gunel & Ré — Learning Mixed-Curvature Representations in Product Spaces, ICLR 2019](https://openreview.net/forum?id=HJxeWnCcF7):
  product manifolds of constant-curvature factors, joint curvature learning, signature
  estimation from sectional-curvature statistics. The E3 layer is this idea made into a
  normalization operator.
- [Bachmann, Bécigneul & Ganea — Constant Curvature Graph Convolutional Networks, ICML 2020](http://proceedings.mlr.press/v119/bachmann20a/bachmann20a.pdf):
  the κ-stereographic model; differentiable interpolation through κ = 0.
- [Zhang et al. — Switch Spaces: Learning Product Spaces with Sparse Gating, 2021](https://arxiv.org/abs/2102.08688):
  sparse gating over geometry components + specialization argument; E3's router follows this.
- [Fisher information metric](https://en.wikipedia.org/wiki/Fisher_information_metric):
  the √p / positive-orthant-of-the-sphere identity that grounds the E1 init.
- [Meng et al. — Spherical Text Embedding, NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/043ab21fc5a1607b381ac3896176dac6-Abstract.html):
  spherical embeddings post-tokenization — the gap (substrate-level spherical identity) this
  repo probes.
- [BLT — Byte Latent Transformer (Meta, 2024)](https://arxiv.org/abs/2412.09871) and
  [MambaByte (2024)](https://arxiv.org/abs/2401.13660): the tokenizer-free lineage E4 will
  rejoin.

## Datasets

- [enwik8](http://mattmahoney.net/dc/textdata.html) — canonical byte-level benchmark
  (100MB Wikipedia dump); enables bits/byte comparison against published numbers.
- [FineWeb-Edu `sample-10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) —
  streamed; the HoloByte comparison corpus.
