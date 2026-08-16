# entropic-substrate-lm

Experiment repo testing substrate-level symbol encoding for language models. It combines
three ideas: (1) **√p Fisher-Rao initialization** — byte identity vectors initialized from
corpus next-byte transition distributions, sqrt-mapped onto the positive orthant of the unit
hypersphere (Fisher-Rao metric = angular metric), then lifted 256→`d_model` via a fixed
semi-orthogonal matrix; (2) an **nGPT-style hypersphere-native backbone** (normalization-as-
retraction, no manifold-aware optimizer, plain AdamW); and (3) **per-head κ-attention** —
curvature in the comparison kernel rather than on individual states: each head scores with
`logits = −d_κₕ(q, k)² · s` under its own learnable, clamped curvature `κₕ` (initialized
spread over −1/0/+1), so the heads themselves are the mixture-over-geometries. Head mixing,
not a sparse router — no load-balance machinery and no gate-collapse mode. E2 is the exact
`κ = 0` member of that family, which keeps the ablation one-variable. Geometry math is always forced to
FP32 inside autocast-disabled blocks, regardless of the surrounding AMP precision. Prior art is
pinned as git submodules under [`references/`](references/) (holobyte, hnet, ngpt, geoopt).

> **A note on what changed, and why it is a result rather than a rename.** An earlier
> revision put curvature in a gated κ-stereographic _normalization_ layer (`logmap0` →
> sparse gate over `K` constant-curvature components → `expmap0`/`projx`). That layer is
> still in-tree (`layers/kappa_norm.py`) but is **measured to be radially annihilated**:
> origin log/exp maps are radial, and the unit-norm retraction erases radius, so under this
> backbone's unit-norm contract the layer equals `UnitNorm` to float precision — pinned by a
> regression test. Curvature is _relational_; it cannot act on a single point sitting at a
> chart origin, only where points meet. That is why it moved into the attention kernel.
> [`docs/architecture.md`](docs/architecture.md) is the authoritative description.

## Quickstart

```bash
# 1. Trust and install pinned tool versions (python 3.12, uv, just, prek, ...)
mise trust
mise install

# 2. Sync the Python environment
uv sync --all-groups

# 3. Run the full quality gate (fmt-check + lint + typecheck + unit tests)
just check

# 4. Run an experiment (see the experiment table below for valid `exp` values)
just train exp=e1-fisher

# 5. Build + run the training container for a single H100 (RunPod / Vast.ai friendly)
just container-build
podman run --gpus all \
  -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results \
  entropic-substrate-lm:latest train --exp e3
```

Other useful commands:

```bash
just download-data      # fetch enwik8 into data/ (then `dvc add data/enwik8`)
just test                # unit tests only (CPU, <60s)
just test-all            # unit + integration tests
just eval ckpt=models/e3-latest.pt
```

## Experiment ladder

| ID | Description                                                                                                                                                                                                                | Metric                                                                                        |
| -- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| E0 | Matched-param baselines: byte-level vanilla transformer **and** GPT-2-BPE transformer (~80–100M params)                                                                                                                    | nats/byte (BPE loss converted at the measured bytes/token on the eval split, not assumed 4.0) |
| E1 | E0 byte model with embedding init swapped: random-on-sphere vs. √p Fisher-Rao init. Identical arch/seed.                                                                                                                   | nats/byte                                                                                     |
| E2 | Hypersphere-native backbone (nGPT-style Norm-as-retraction blocks, SLERP/LERP residual with learnable eigen-LRs α)                                                                                                         | nats/byte                                                                                     |
| E3 | E2 with the attention similarity kernel swapped for per-head κ-distance: `logits = −d_κₕ(q,k)²·s`, one learnable clamped κ per head (init spread −1/0/+1). E2 is the exact κ=0 member, so the ablation stays one-variable. | nats/byte + per-head κ trajectories                                                           |

Datasets: [enwik8](http://mattmahoney.net/dc/textdata.html) (downloaded via
`just download-data` into `data/`, tracked with `dvc add data/enwik8`) and
[FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (streamed
via 🤗 `datasets`).

## Project layout

```text
entropic-substrate-lm/
├── src/entropic_substrate/   # package: data loaders, geometry, layers, models, train.py, cli.py
├── configs/                  # per-experiment YAML (e0..e3, tiny.yaml)
├── tests/                    # unit + integration tests (pytest markers: unit, integration)
├── scripts/                  # data download / utility scripts
├── docs/                     # design notes, ADRs, experiment writeups
├── references/               # pinned upstream submodules (holobyte, hnet, ngpt, geoopt)
├── data/ models/ results/    # DVC-tracked working dirs (empty dir placeholders only in git)
└── notebooks/                # scratch analysis notebooks (not tracked beyond .gitkeep)
```

## References

- Prior art is vendored as git submodules under [`references/`](references/): HoloByte,
  H-Net, nGPT, and [geoopt](https://github.com/geoopt/geoopt) (also used as the dev/test
  oracle that this repo's own κ-stereographic ops are checked against).
- See [`docs/`](docs/) for design notes, ADRs, and experiment write-ups.
- geoopt is installed directly from GitHub (`geoopt @ git+https://github.com/geoopt/geoopt.git`)
  per its own README's recommendation, ahead of its PyPI release cadence.
