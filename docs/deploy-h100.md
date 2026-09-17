# Deploy — H100 on RunPod / Vast.ai

## Current H-series image

The refreshed project image was built and published using Docker Build Cloud's
`cloud-luizffrias-small-lang-model` amd64 node. The canonical machine-readable
record is `publication.json` inside the DVC-backed publication capture, referenced
by [the capture pointer](../results/h-image-publication-79a4be2.dvc). Git stores this
human-readable pin and the pointer, not a second copy of the generated evidence.

```text
ghcr.io/luiz-frias/entropic-substrate-lm@sha256:bd41ff94266a51550c9b013bf963676b7f46b763087e383a385b0afeb67b8b42
```

Runtime source matches commit `79a4be270baca3bce077bad89d8937201c0e6b96` exactly.
The H worker imports, pySBD/boto3 are installed, and the committed R2 topology is
present. These are CPU-side packaging/provenance checks, not CUDA training,
scientific confirmation, or successor eligibility. The 6K probes still require
the promised prelaunch check-in. Do not substitute `latest` for this digest.

Publication capture used a fresh task-private DVC site cache through
`DVC_SITE_CACHE_DIR`; the pre-existing world-writable site cache was not trusted
or modified. Use a private cache for subsequent capture work too.

For H-probe template preparation, Luiz confirmed the Runpod secret-reference names
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` on 2026-09-17. These are names, not
credential values. The H object worker reads its endpoint from committed DVC topology;
it does not require the separate `R2_API_KEY` or `MLFLOW_S3_ENDPOINT_URL` references.
Template creation and the pre-allocation check-in remain pending.

## Historical E-series deployment

The following commands and sizing discussion retain their historical E-series scope.

The `Containerfile` builds a single image (base: `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`,
CUDA 12.6 covers H100 / sm_90). Everything below assumes the image is pushed to a registry both
platforms can pull from (GHCR or Docker Hub).

The image installs the frozen runtime plus the `deploy` dependency group from `uv.lock`.
That group includes DVC's S3 adapter for R2 synchronization while excluding dev-only test
and geometry-oracle dependencies. The project pins `torch==2.13.0` to the same release already
provided by the CUDA base image, preventing a later dependency resolution from replacing the
CUDA-matched wheel. The container export prunes Torch's dependency closure because the official
base image owns its CUDA 12.6 libraries; application packages cannot layer a second CUDA runtime
over them.

## Build & push

```bash
just container-publish   # native amd64 Docker Build Cloud builder → pushes to GHCR
```

`just container-build` (local podman) still exists for native-arch experiments, but is
NOT a publish lane on Apple Silicon: the dependency layer's uv installer segfaults under
qemu-user emulation (measured 2026-08-17 — see the justfile's retraction comment).
`container-publish` builds on a native amd64 cloud builder and pushes straight to GHCR,
so the multi-GB image never transits the local uplink. The GHCR package is private
(the repo is); pods authenticate with a **classic** PAT scoped to `read:packages` only —
GHCR does not accept fine-grained PATs (GitHub docs: "GitHub Packages only supports
authentication using a personal access token (classic)"), registered via
`runpodctl registry create`.

## RunPod

- Pod type: GPU Pod, H100 80GB (PCIe is fine at this scale; SXM unnecessary for ~85M params).
- Container image: `ghcr.io/luiz-frias/entropic-substrate-lm:latest`.
- Volume: mount a persistent volume at `/app/models` and `/app/results` — pods are ephemeral.
- Env: `HF_TOKEN` (only needed for gated datasets; FineWeb-Edu is public), optional
  `HF_HOME=/app/.cache/hf` pointed at the volume so streaming shards cache across restarts.
- Run command examples:

```bash
python -m entropic_substrate.cli train --exp e1-fisher --device cuda
python -m entropic_substrate.cli train --exp e3 --device cuda
bash scripts/run_ladder.sh --device cuda   # gated ladder E1 → E2 → E3, dvc push per rung
```

## Vast.ai

- Filter: H100, CUDA ≥ 12.6 driver, verified hosts, ≥ 200GB disk if caching FineWeb shards.
- Image: same GHCR image; on-start script:

```bash
cd /app && python -m entropic_substrate.cli train --exp e3 --device cuda \
  2>&1 | tee results/e3-console.log
```

- Vast instances die without warning — checkpointing is on by default; resume with
  `--resume models/<exp>-latest.pt` (checkpoints are written as
  `models/<name>-latest.pt` / `models/<name>-final.pt`).

## Sizing expectations

- E0–E3 at ~85M params, seq 1024, micro-batch 4–16 w/ grad accumulation: a single H100 is
  overkill on memory (< 20GB) and each 20k-step run is hours, not days. bf16 autocast is on by
  default; the geometry stays FP32 internally regardless (see `architecture.md`).
- Multi-seed sweeps: run seeds sequentially on one pod rather than multi-pod — data streaming
  is the bottleneck, and the HF cache is shared on-volume.

## Sync results home

The DVC remote is committed in `.dvc/config` (Cloudflare R2, bucket `entropic-substrate-lm`,
the embedding-conversion convention: topology in git, identity in env). A pod only needs
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from a bucket-scoped R2 Object Read & Write
token injected as env vars, then `dvc push`. For the tiny `results/` JSONL files,
`rsync`/`runpodctl send` remains a credential-free alternative.

`scripts/run_ladder.sh` creates DVC's site cache with mode `0700` under a `077` umask. This
mitigates the unpatched DiskCache local-deserialization advisory by preventing other pod users
from modifying serialized cache state; it does not make an untrusted or shared cache safe. Do
not mount `.cache/dvc-site-cache` from a shared writable volume.

## Observability — MLflow tracking stack (Railway)

The stack the local live gate rehearses (`compose.yaml`) is deployed on Railway, project
`entropic-substrate-lm`: `Postgres` (managed template, Railway volume) ← `mlflow` (built from
`Containerfile.mlflow`, **private-network only** — no public domain; dual-stack bind via
`MLFLOW_HOST=::`) ← `caddy` (built from `Containerfile.caddy`, the only public service —
`basic_auth` in front, `/health` exempt, upstream `mlflow.railway.internal:5000` via the
Caddyfile's `MLFLOW_UPSTREAM` placeholder). Every MLflow server option arrives as a native
env var (see `Containerfile.mlflow`'s CMD comment); artifacts land at
`s3://entropic-substrate-lm/mlflow/` on R2 — a prefix the deployed server OWNS. The local
live gate writes the venue-split `mlflow-live-gate/` prefix instead (see compose.yaml's
`--artifacts-destination` comment for the observed commingling that forced the split).
`MLFLOW_SERVER_ALLOWED_HOSTS` on the mlflow service pins the public Caddy domain exactly —
this deployment is the enforcement site for compose.yaml's never-`'*'` note. Per-service
build recipes + GitHub-source watch paths are config-as-code in `railway.mlflow.json` /
`railway.caddy.json` (each service's settings point at its file).

A training pod opts in with three variables (canonical values: Doppler project
`entropic-substrate-lm`; the server-side pair lives on the Railway services):

- `MLFLOW_TRACKING_URI=https://caddy-production-3bbd.up.railway.app`
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` — Caddy basic auth; the MLflow
  client sends these natively.

The sink stays best-effort and never gated on (`architecture.md`): URI unset ⇒ the pod logs
JSONL only. VERIFIED outside-in 2026-08-17 against the deployed stack: `/health` unauth →
200; API unauth → 401; API auth'd → 200 (DB-backed); an artifact logged through the public
URL landed at `mlflow/1/<run_id>/artifacts/probe.txt` on R2 and read back byte-identical.
