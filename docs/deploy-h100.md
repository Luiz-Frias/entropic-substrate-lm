# Deploy — H100 on RunPod / Vast.ai

## Current H-series image

The refreshed project image was built and published using Docker Build Cloud's
`cloud-luizffrias-small-lang-model` amd64 node. The canonical machine-readable
record is `publication.json` inside the DVC-backed publication capture, referenced
by [the capture pointer](../results/h-image-publication-cf4d9cb.dvc). Git stores this
human-readable pin and the pointer, not a second copy of the generated evidence.

```text
ghcr.io/luiz-frias/entropic-substrate-lm@sha256:e4e4a8608379a8e607edb27bcda7ae329c0175fc07e201139dc770547860ab12
```

Runtime source matches commit `cf4d9cb20a4607e6c7565dfd67a5ee34c31bf386` exactly.
Native validation includes source replay and allowlisted S3 failure diagnostics.
RE-PINNED (release v0.9.0, branch `fix/h-image-repin-0.9.0`, operator mandate 2026-10-28): the
previous pin (`b39b05b1...`, source `66d5fea`) is superseded; its capture `results/h-image-
publication-66d5fea.dvc` remains historical evidence. This image is built from the v0.9.0 release
tip of `main`. It carries every H run-class gate routed through the admission port with the port in
the 52-member checkpoint reader set (PR #400), the sealed RunAuthority that pilot and confirmatory
requests must present (rung B, PR #406), and the conductor's re-pin that adds the authority module
to the reader set (53 members) with the five regime rows recomputed over that tree (PR #407), so a
pod running this image admits a scientific request only under a verified authority while smoke
requests keep every earlier verdict. Same verifier and `h-prepare-source` manifest as the 66d5fea
publication; the source replay's in-image capture admitted 491 documents / 1954698 bytes, corpus
identity `cf938a0b...`. The nine-key RunPod template for this image is `k7gtsfkr77`
(`mythwave-h0-cf4d9cb-v0.9.0`; see LAUNCH SCHEMA v3 below).
RE-PINNED (release v0.8.0, branch `fix/h-image-repin-0.8.0`, operator go-ahead 2026-10-27): the
previous pin (`5452fbf4...`, source `e73d123`) is superseded; its capture `results/h-image-
publication-e73d123.dvc` remains historical evidence. This image is built from the v0.8.0 release
tip of `main`. It carries the H1 per-byte E-LM projection with the median ESS/k reading (PR #395),
the H run admission port with every gate routed through it (PRs #396 and #400 lineage), the
gradient-reconciliation guard rewritten as set identity with the preflight NOT_APPLICABLE verdict
for isotropic profiles (PR #402), the CI pin to CPython 3.12.15 with the 3.12.16 regime rows, and
the format-on-edit hook wiring, so the worker a quote run exercises completes codec warm-up at
development widths. Same verifier and `h-prepare-source` manifest as the e73d123 publication; the
source replay's in-image capture admitted 491 documents / 1954698 bytes, corpus identity
`cf938a0b...`. The nine-key RunPod template for this image is `x8zk48r24q`
(`mythwave-h0-66d5fea-v0.8.0`; see LAUNCH SCHEMA v3 below).
RE-PINNED (main tip `e73d123`, direct commit on `main`, operator go-ahead 2026-10-27): the previous
pin (`88e21c59...`, source `046cfb0`) is superseded; its capture `results/h-image-
publication-046cfb0.dvc` remains historical evidence. This image is built from the `main` tip that
merged the H0 final-probe stack (PRs #390 to #394), not from a release tip: Release Please could not
run (GitHub Actions billing, paused until early November 2026), so no tag names this tree and the
re-pin landed as a direct commit on `main` under the same ruling. It carries the H0 final-probe
stack (PRs #390 to #394: the runpodctl 3.0.1 pin across the launch path, the seal-targeted operator
stop marker with its phase and step target, the resumed-arm checkpoint progress fix, the 2026-10-26
hypothesis and systems reviews with the H0 to H5 horizon and the H1 skeleton rulings, and the H0
final-probe filing with its tracer_complete verdict), so the worker a pilot exercises carries the
stop and resume semantics the probe validated. Same verifier and `h-prepare-source` manifest as the
046cfb0 publication; the source replay's in-image capture admitted 491 documents / 1954698 bytes,
corpus identity `cf938a0b...`. The nine-key RunPod template for this image is `tvgoxbc7di`
(`mythwave-h0-e73d123-main`; see LAUNCH SCHEMA v3 below).
RE-PINNED (release v0.6.0, branch `fix/h-image-repin-0.6.0`, operator go-ahead 2026-10-23): the
previous pin (`e4986ac9...`, source `ba76c0a`) is superseded; its capture `results/h-image-
publication-ba76c0a.dvc` remains historical evidence. This image is built from the v0.6.0 release
tip of `main`. It carries waves B and C (PRs #375 to #386: the strict MLflow observer and CLI
handler families, the blind-review sandbox file collection, observed bf16 autocast in run_start,
structurally absent zero-sentence mechanism rows, and FOR-1674's generation receipt v3 with N
continuations per prompt and their sampling law plus the profile continuations field) and the H1
preregistration skeleton, so the worker a probe exercises can prove stochastic rollout and record
its autocast state. Same verifier and `h-prepare-source` manifest as the ba76c0a publication; the
source replay's in-image capture admitted 491 documents / 1954698 bytes, corpus identity
`cf938a0b...`. The nine-key RunPod template for this image is `w1rl2rxaqw`
(`mythwave-h0-046cfb0-v0.6.0`; see LAUNCH SCHEMA v3 below).
RE-PINNED 2026-10-22 (release v0.4.0, branch `fix/h-image-repin-0.4.0`, operator go-ahead
2026-10-22): the previous pin (`8181443b...`, source `9d353fc`) is superseded; its capture
`results/h-image-publication-9d353fc.dvc` remains historical evidence. This image is built from
the v0.4.0 release tip of `main`, the first tip that carries the experiment pre-requisite stack
(PRs #359 to #369: FOR-1670 evaluation read memo, FOR-1664 step-keyed evaluation reseed plus the
required `diagnostic_samples` field and profile schema `h-run-profile-v2`, FOR-1648 H analysis
surface, the FOR-1609/FOR-1598/FOR-1634 bug-squash fixes, the HEpP7g capture and analysis, and the
`BLIND_REVIEW_MODEL` env for the cloud blind review), so the worker's evaluation path, profile
schema and final leg a probe exercises match the released tree. Same verifier and
`h-prepare-source` manifest as the 9d353fc publication; the source replay's in-image capture
admitted 491 documents / 1954698 bytes, corpus identity `cf938a0b...`. The nine-key RunPod
template for this image is `tgk12wur3v` (`mythwave-h0-ba76c0a-v0.4.0`; see LAUNCH SCHEMA v3 below).
RE-PINNED 2026-10-19 (release v0.3.1, branch `fix/h-image-repin-0.3.1`, operator go-ahead
2026-10-19): the previous pin (`96e98504...`, source `dbe9709`) is superseded; its capture
`results/h-image-publication-dbe9709.dvc` remains historical evidence. This image is built from
the v0.3.1 release tip of `main`, the first tip that carries the two final-leg fixes the 2026-10-17
paired 1K probe exposed (FOR-1662: one admission walk per artifact root, later stages reuse the
memo; FOR-1665/FOR-1667: the host terminal probe tolerates recoverable R2 read errors and fails
fast only on permanent codes) plus the FOR-1666 workflow overlay, so the worker's final leg and
the host controller a probe exercises match the released tree. Same verifier and
`h-prepare-source` manifest as the dbe9709 publication; the source replay's in-image capture
admitted 491 documents / 1954698 bytes, corpus identity `cf938a0b...`. The nine-key RunPod
template for this image is `wjag3o82g3` (`mythwave-h0-9d353fc-v0.3.1`; see LAUNCH SCHEMA v3 below).
RE-PINNED 2026-10-17 (H observability parity, branch `fix/h-image-repin-parity`, operator
go-ahead 2026-10-17): the previous pin (`762ed810...`, source `65542d4`) is superseded; its
capture `results/h-image-publication-65542d4.dvc` remains historical evidence. This image is
built from the `main` tip that carries the whole H observability parity stack (PRs #341 to
#347: FOR-1637, FOR-1649, FOR-1639/1641, FOR-1650, FOR-1642/1643/1644/1653, FOR-1645/1646/1647/1638,
FOR-1657/1659), so the worker log vocabulary, the run surface, the launch schema v3 and the
request-digest run tag a probe exercises match the released tree. Same verifier and
`h-prepare-source` manifest as the 65542d4 publication; the source replay's in-image capture
admitted 491 documents / 1954698 bytes, corpus identity `cf938a0b...`. The nine-key RunPod
template for this image is `3n4y0ba76d` (`mythwave-h0-dbe9709-parity`; see LAUNCH SCHEMA v3 below).
RE-PINNED 2026-10-12 (release v0.2.0, branch `fix/h-image-repin-v0.2.0`, operator go-ahead
2026-10-12): the previous pin (`f2469dc9...`, source `3b93854`) is superseded; its capture
`results/h-image-publication-3b93854.dvc` remains historical evidence. This image is built
from the v0.2.0 release tip of `main`, the first tip that carries the whole p14 consolidation
sprint and its four follow-ups (FOR-1633, FOR-1632, FOR-1626, FOR-1636), so the worker,
transport and capture code a probe exercises matches the released tree. Same verifier and
`h-prepare-source` manifest as the 3b93854 publication; the source replay's in-image capture
admitted 491 documents / 1954698 bytes, corpus identity `cf938a0b...`.
RE-PINNED 2026-09-26 (exit-gate lane, branch `mythwave/h-exit-gate`, operator go-ahead for the
H tiny-probe wave): the previous pin (`6b4e68fa...`, source `88d1a25`) is superseded; its capture
`results/h-image-publication-88d1a25.dvc` remains historical evidence. This image carries the
two changes the wave depends on: the worker receives MLflow through the launch template's
secret references (`h0-probe-launch-v2`) and reports GLOBAL optimizer steps plus evaluation
rows to the metric sink. Same verifier and manifest as the 88d1a25 publication; the source
replay's in-image capture admitted 491 documents / 1,954,698 bytes, corpus identity `cf938a0b...`.
RE-PINNED 2026-09-26 (p13-sweep commit F, ruling P13-IMAGE, operator go-ahead OP-3): the
previous pin (`12ed1634...`, source `9a6300a`) is superseded; its capture
`results/h-image-publication-9a6300a.dvc` remains historical evidence. This publication's
verifier imports only landed addresses (`kernel.lifecycle`, `ladder.capture.capture`,
`ladder.run.worker.*`, `ladder.run.transport.*`, `ladder.run.worker_command`), which is itself
the packaging proof that the retired dotted namespace is gone from the image. The source
replay uses a fresh `h-prepare-source` capture (same manifest: 512 candidates, `sample-10BT`,
pinned FineWeb revision, split seed 42; 491 admitted / 1,954,698 bytes, corpus identity
identical to the 9a6300a-era capture) because the p4 reader-identity bump makes the earlier
capture unreadable by the current reader, by design.
Requests must bind their code commit and bundle to this publication, not a later
host-only controller commit. Earlier publication captures remain historical evidence.
The H worker imports, pySBD/boto3 are installed, and the committed R2 topology is
present. These are CPU-side packaging/provenance checks, not CUDA training,
scientific confirmation, or successor eligibility. Paid 6K validation is authorized,
but completion requires the worker's verified artifacts. Do not substitute `latest`
for this digest.

The pin above (source `cf4d9cb`, as did the 66d5fea, e73d123, 046cfb0, ba76c0a, 9d353fc and dbe9709 pins) carries FOR-1637: the H worker's log-event vocabulary and the
`worker-log.jsonl` MLflow artifact. (Until 2026-10-17 this paragraph said the pin did not
contain them, which was true of the 65542d4 pin.)

The pin above also carries the H run surface (FOR-1642): the grouping key and descriptor
params on every H run, the per-checkpoint evidence tags, and a `FAILED` MLflow status when the
worker raises, plus the `request_sha256` run tag (FOR-1657). (Until 2026-10-17 this paragraph
said the pin did not contain them, which was true of the 65542d4 pin.)

LAUNCH SCHEMA v3 (FOR-1647, DN-OBS-TEMPLATE; the dated RE-PINNED paragraphs above name `h0-probe-launch-v2`
as of their dates and are not rewritten). The pod environment is the seven-key base plus declared
extensions from a closed registry; the H0 wave declares `named_experiment("h-series")` and
`system_metrics()`, so the template carries NINE environment keys (`MLFLOW_EXPERIMENT_NAME=h-series`
and `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true` beyond the seven). The nine-key template is created by
the conductor at the image rebuild ceremony; no template is created, and no template id is recorded
here, until that ceremony has measured one. MEASURED 2026-10-17: template `3n4y0ba76d`
(`mythwave-h0-dbe9709-parity`, nine keys, image `96e985048fd9...`) was created at the dbe9709 rebuild
ceremony and read back whole. MEASURED 2026-10-28: template `k7gtsfkr77` (`mythwave-h0-cf4d9cb-v0.9.0`, nine keys, image `e4e4a8608379...`) was created at the cf4d9cb rebuild ceremony and read back whole; `x8zk48r24q` stays bound to the superseded 66d5fea image. MEASURED 2026-10-27: template `x8zk48r24q` (`mythwave-h0-66d5fea-v0.8.0`, nine keys, image `b39b05b1cb71...`) was created at the 66d5fea rebuild ceremony and read back whole; `tvgoxbc7di` stays bound to the superseded e73d123 image. MEASURED 2026-10-27: template `tvgoxbc7di` (`mythwave-h0-e73d123-main`, nine keys, image `5452fbf4fba6...`) was created at the e73d123 rebuild ceremony and read back whole; `w1rl2rxaqw` stays bound to the superseded 046cfb0 image. MEASURED: template `w1rl2rxaqw` (`mythwave-h0-046cfb0-v0.6.0`, nine keys, image `88e21c59b17b...`) was created at the 046cfb0 rebuild ceremony and read back whole; `tgk12wur3v` stays bound to the superseded ba76c0a image. MEASURED 2026-10-22: template `tgk12wur3v` (`mythwave-h0-ba76c0a-v0.4.0`,
nine keys, image `e4986ac9eaca...`) was created at the ba76c0a rebuild ceremony and read back whole;
`wjag3o82g3` stays bound to the superseded 9d353fc image. MEASURED 2026-10-19: template `wjag3o82g3` (`mythwave-h0-9d353fc-v0.3.1`,
nine keys, image `8181443b6c41...`) was created at the 9d353fc rebuild ceremony and read back whole;
`3n4y0ba76d` stays bound to the superseded dbe9709 image. `validate_template` compares the whole environment, so a
template missing either extension key is refused before any pod is created. Runs launched with
`named_experiment` land in the MLflow experiment `h-series`, not the default. `capture-results` selects
runs by experiment name and defaults to `$MLFLOW_EXPERIMENT_NAME` or `entropic-substrate-lm`
(`composition/entrypoints/cli.py`, `ladder/capture/reader/capture_results.py`), so capturing these runs
needs `--experiment h-series`; without it a capture finds none of them.

RUN ADMISSION (2026-10-27, rung A): which run class may hold a worker request, execute on the local
or remote worker, or be launched, and with which cost guard, is one constant table, `RUN_ADMISSION` in
`ladder/run/admission.py`; it admits `local_tiny` on the local worker at every stage and `remote_tiny`
on the remote worker and at launch only at H0 with `host_watchdog_tiny_only`, and refuses `pilot` and
`confirmatory` everywhere. `H0ProbeLaunch` asks the port, the five reader-identity members that gate a
class still carry inline checks held to the same table by `tests/test_h_run_admission_sites.py` until
the conductor's re-pin routes them through it, and rung B replaces the constant with a sealed
`RunAuthority` artifact under LAUNCH SCHEMA v4.

LAUNCH SCHEMA v4 (2026-10-27, rung B, operator ruling "option B"). A
`pilot` or `confirmatory` run at H1-H5 is admissible only under a sealed `RunAuthority`
(`ladder/run/run_authority.py`, schema `h-run-authority-v1`): stage, run class, the ruling
(ISO date, text, operator), the spend cap (USD and hours, both finite and positive), the RunPod
GPU type id (any safe identifier, recorded as given), the full 40-hex SHA of the committed
preregistration the run executes, and the SHA-256 of all of those as canonical JSON. The port
re-verifies that digest at every gate, so an authority edited after sealing is refused
("run authority fails its seal"), as are a missing one, one naming another stage or class, and
any authority on a smoke request. The worker request carries the authority as
`h-probe-worker-request-v2`; a smoke request still renders `h-probe-worker-request-v1`
byte-identically, so every captured `request_sha256` re-derives. A codec-sharing PAIR is its own
request, `h-probe-worker-request-v3` (FOR-1693): two cells donor first (the magnitude-spine arm, then
the sphere-spine arm), a block binding and one authority for both; v1 and v2 keep their bytes and
meaning (RETRACTED 2026-11-02: "no launch carries a pair before FOR-1694"; launch v5 below carries
one). The launch manifest
`h-probe-launch-v4` names the authority digest as both `scientific_authority` and
`spending_authority` (`"none"` for smoke). A scientific launch declares the cost guard
`host_watchdog_budgeted` (smoke keeps `host_watchdog_tiny_only`, the only kind it may declare),
must use the authority's GPU class, keeps `watchdog_timeout_seconds` inside the authority's hours,
and `h_probe_command` refuses a `--max-hourly-rate` whose product with the watchdog window exceeds
the authority's USD cap before any provider call. Its pod is named `<stage>-<class>-<digest
prefix>` (`h1-pilot-...`); smoke keeps `h0-tiny-...`. Execution receipts record kind
`remote_pilot`/`remote_confirmatory` and echo the authority digest in `runtime.authority`, which
rides into `worker/start.json`, the `started` event, the `run_start` lifecycle record and the
worker result; completion admission refuses a runtime that does not echo it.

The scientific pod environment is EIGHT keys for the H wave: the seven-key base minus
`ENTROPIC_MLFLOW_TINY`, plus `MLFLOW_EXPERIMENT_NAME=h-series` and
`MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true`. The worker builds the MLflow sink with `tiny=False`
for a request that carries an authority, so a tracking URI alone enables its mirror (with
`tiny=True` and no opt-in the mirror would be dropped silently by `sinks.mlflow_enabled`). The
eight-key template is created at the image rebuild ceremony for the first image that carries
this schema; none exists yet and no template id is recorded here until that ceremony measures one.
`validate_template` compares the whole environment, so a nine-key smoke template is refused for a
pilot and the converse. READ-OLD: a `h0-probe-launch-v3` (or v2) payload reads as the same class,
re-renders its own bytes and never compares equal to a v4 launch; a pre-v4 payload carries no
authority, so a v3 launch of a scientific request is refused. v1 stays refused.

LAUNCH SCHEMA v5 (2026-11-02, FOR-1694, h1-l4c). One bump, three moves. (1) The launch SEALS its
hourly price: a scientific launch declares `hourly_price_ceiling` (USD per hour) at construction, an
explicit field with no default that is never derived from the authority (`RunAuthority` carries a
USD cap and an hours cap and no hourly price, so `usd / hours` would replace the ceiling the invoker
approved with a looser number nobody typed). The product rule of rung B is unchanged and now runs at
construction and at parse: `ceiling * watchdog_timeout_seconds / 3600 <= spend_cap.usd`. The
manifest also records `hours_cap`, copied from `spend_cap.hours` and re-derived at parse; a smoke
launch carries explicit `null` for both. `h_probe_command` and the controller read the ceiling from
the launch: `--max-hourly-rate` is optional for a sealed launch and, when present, must equal the
sealed value exactly (no tolerance), else it is refused by name before any workspace or provider
call; the controller's `cost_per_hour` check compares against the sealed value. A smoke, v4 or v3
launch has no sealed ceiling and keeps the flag with its v3 semantics, required. (2) The launch
carries a codec-sharing pair (`h-probe-worker-request-v3`): the manifest records `pair` and both
cell identities (`cells`, donor first), re-derived from the embedded request at parse; a pair under
a v3 or v4 manifest is refused by name. (3) The v2 read retires, as the v3 section promised: a
`h0-probe-launch-v2` payload is refused naming `h0-probe-launch-v3` as the minimum supported
version. READ-OLD: v4 and v3 payloads (every H0 launch file sealed on R2 is v3) read as the same
class, re-render their own bytes and never compare equal to a v5 launch. Every launch built in
process renders v5, so its digest, pod name and workspace moved; the pod name rule
`<stage>-<class>-<digest prefix>` (smoke `h0-tiny-...`) is unchanged, and the pod listing filter
matches that exact name. The scientific pod environment is the same eight keys as v4.

Host side of a pair (FOR-1694). A pair is collected as `h-probe-terminal-collection-v2`: one entry
per cell (receipt, result entry, capture and verification roots), the pair verdict and the terminal
it was collected under (`complete`, `paused`, `stopped`, `excluded`, `failure`); result v2 is read
under every terminal that can carry it, the failure terminal included, and `workload_verified` is
keyed on the verdict, never on a status. A pair whose pod never began a cell collects as a typed
`pair_not_started` failure. A single-profile request keeps `h-probe-terminal-collection-v1` byte for
byte. The wave monitor lists one arm per (request, cell), each searched by the `request_sha256` and
`cell` run tags; `h_probe_stop --at-cell donor|recipient` (with `--at-phase` and `--at-step`)
publishes the write-once `h-probe-stop-marker-v2` and stops that cell only, and a pair without
`--at-cell` is refused by name. Objects above 64 MiB are published as verified multipart uploads
(`d1_trainer_s3_object_adapter`); Cloudflare R2's handling of conditional completion and part
checksums is UNMEASURED, so the first live pair run is the measurement.

Capture by reference (FOR-1694; lands with the staged reader-member DG-A patch, not on this
branch tip). A pair's completion snapshot is `h-worker-snapshot-v2`: a `capture/<cell>/<file>`
whose bytes equal the snapshot's own `run/<cell>/<file>` is a reference entry (`reference`, and
`key` naming the referenced object) with no second upload, so a pair pays for each large object
once. The host resolves a reference to the run twin's bytes and refuses a digest disagreement; a
reference that is not an equal run twin of its own cell is refused. Every consumer that lists a
snapshot's keys (for example `h_probe_stop.request_stop`) therefore reads a reference's twin key
once. A single-profile request keeps `h-worker-snapshot-v1` byte for byte, and a pre-L4c pair's
v1 markers still read.

Publication capture used a fresh task-private DVC site cache through
`DVC_SITE_CACHE_DIR`; the pre-existing world-writable site cache was not trusted
or modified. Use a private cache for subsequent capture work too.

For H-probe template preparation, Luiz confirmed the Runpod secret-reference names
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` on 2026-09-17. These are names, not
credential values. The H object worker reads its endpoint from committed DVC topology;
it does not require the separate `R2_API_KEY` or `MLFLOW_S3_ENDPOINT_URL` references.
RETRACTION (2026-09-17): supplied names did not establish that account secrets existed.
An authenticated metadata query returned zero stored secrets; pod `8raqb0mtr6alzz`
received literal references and failed before training. Its termination was verified.
With explicit approval, both named secrets were then created from locally verified
R2 credentials and their metadata read back. Values were not printed or captured.
Verify secret-name existence separately from template-reference spelling before launch.
The current v2 OpenAPI exposes `/v2/account/secrets`; the connected tool subset does not.

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
python -m entropic_substrate.composition.entrypoints.cli train --exp e1-fisher --device cuda
python -m entropic_substrate.composition.entrypoints.cli train --exp e3 --device cuda
bash scripts/run_ladder.sh --device cuda   # gated ladder E1 → E2 → E3, dvc push per rung
```

## Vast.ai

- Filter: H100, CUDA ≥ 12.6 driver, verified hosts, ≥ 200GB disk if caching FineWeb shards.
- Image: same GHCR image; on-start script:

```bash
cd /app && python -m entropic_substrate.composition.entrypoints.cli train --exp e3 --device cuda \
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
