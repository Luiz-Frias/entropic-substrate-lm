"""Local end-to-end tracer for a remote tiny H probe, with observability wired.

Why this exists (2026-09-27): both remote_tiny H0 probes raised at prediction
step 1, and the only run that had ever executed prediction steps was a width-16
CPU tracer with a different shape. This command runs the SAME profile shape
locally so the failure can be reproduced and instrumented without a pod.

What it changes, and nothing else: ``localize_profile`` rewrites only the
execution regime (``run_class`` -> ``local_tiny``, ``device`` -> ``cpu``, an
optional ``precision``, and the re-sealed ``source_digest``). Every scientific
field (codec, spine, law, consumer, solver, optimization, schedule, seeds,
permutation) is carried byte-for-byte. The run identity hashes the manifest, so
a localized run is a new identity by construction. ``cpu`` because
``HProfile`` admits only ``cpu``/``cuda`` and ``core.sentence.profile`` is a
checkpoint-reader identity member; admitting ``mps`` is an identity re-pin.

Source: the remote capture's reader identity is regime-bound (full interpreter
version), so the local reader refuses it. ``reseal_foreign_corpus`` replays the
captured candidates under the capture's own spec in THIS regime, requires the
replay to reproduce the foreign ``corpus_id`` and summary exactly, and only
then publishes a capture stamped with the local reader. Nothing is re-streamed.

Observability: the sink is composed exactly as ``ladder.run.worker_command``
composes it (``build_sink_for_run`` + ``HTrainingStepExtractor``), written to a
sibling ``observability/`` directory because ``run_probe_training`` refuses a
non-empty output root. The JSONL sink is always on; the MLflow mirror joins
only when ``MLFLOW_TRACKING_URI`` is set and ``ENTROPIC_MLFLOW_TINY=1``. This
tracer has no scientific or spending authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

from entropic_substrate.core.sentence.corpus import (
    PreparationSpec,
    SplitPolicy,
    prepare_documents,
)
from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.records import SourceDocument
from entropic_substrate.core.sentence.segmentation import SegmentationPolicy
from entropic_substrate.kernel.artifacts import normalize_d1_artifact_root
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.kernel.runtime.logging import configure_logging, logger
from entropic_substrate.ladder.capture.adapters.composite import build_sink_for_run
from entropic_substrate.ladder.capture.capture import read_prepared_corpus, write_prepared_corpus
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.capture.ports.run_descriptor import TrackingRunDescriptor
from entropic_substrate.ladder.capture.reader.sinks import (
    CONFIG_HASH_TAG,
    EXP_TAG,
    git_sha,
    h_config_hash,
)
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor
from entropic_substrate.ladder.run.runner import TrainingProgress, run_probe_training

__all__ = ["localize_profile", "main", "reseal_foreign_corpus", "run_tiny_trace"]

type Precision = Literal["fp32", "bf16"]
type SinkBuilder = Callable[..., MetricSink]

_DEFAULT_LIMIT = 67_108_864
_CAPTURE_FILE = "prepared-corpus.json"


def localize_profile(
    remote: HProfile, *, source_digest: str, precision: Precision | None = None
) -> HProfile:
    """Rewrite only the execution regime of a tiny profile for this host's CPU."""
    if remote.run_class not in ("remote_tiny", "local_tiny"):
        raise ValueError("only a tiny probe profile can be traced locally")
    manifest = remote.manifest | {
        "run_class": "local_tiny",
        "device": "cpu",
        "precision": remote.precision if precision is None else precision,
        "source_digest": source_digest,
    }
    return HProfile.from_manifest(manifest)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def reseal_foreign_corpus(capture: Path, output_root: Path, *, max_bytes: int) -> ArtifactIdentity:
    """Replay another regime's source capture here; publish only an exact replay."""
    encoded = capture.read_bytes()
    if len(encoded) > max_bytes:
        raise ValueError("foreign source capture exceeds its byte limit")
    payload = json.loads(encoded)
    if type(payload) is not dict or payload.get("format_version") != "h-source-capture-v2":
        raise ValueError("reseal requires an h-source-capture-v2 payload")
    raw = cast(dict[str, object], payload)
    values = cast(dict[str, object], raw["spec"])
    spec = PreparationSpec(
        candidate_limit=cast(int, values["candidate_limit"]),
        admitted_byte_limit=cast(int, values["admitted_byte_limit"]),
        revision=cast(str, values["revision"]),
        dataset_name=cast(str | None, values["dataset_name"]),
        segmentation=SegmentationPolicy(cast(int, values["max_sentence_bytes"])),
        splits=SplitPolicy(
            cast(int, values["split_seed"]),
            cast(int, values["validation_buckets"]),
            cast(int, values["test_buckets"]),
        ),
    )
    if values["engine_version"] != SegmentationPolicy.engine_version:
        raise ValueError("foreign capture segmentation version differs from this regime")
    candidates: list[SourceDocument] = []
    for item in cast(list[dict[str, str]], raw["candidates"]):
        document = SourceDocument(item["source_id"], item["text"], item["revision"])
        if document.identity != item["identity"]:
            raise ValueError("foreign candidate identity does not match its content")
        candidates.append(document)
    replayed = prepare_documents(candidates, spec)
    if replayed.identity != raw["corpus_id"]:
        raise ValueError("local replay does not reproduce the foreign corpus identity")
    staging = output_root.parent / f".{output_root.name}.reseal"
    staging.mkdir()
    try:
        staged = write_prepared_corpus(staging, replayed, max_bytes=max_bytes)
        local = json.loads((staging / _CAPTURE_FILE).read_bytes())
        if _canonical(local["summary"]) != _canonical(raw["summary"]):
            raise ValueError("local replay does not reproduce the foreign summary")
        read_prepared_corpus(staging, staged.content_hash, max_bytes=max_bytes)
        return write_prepared_corpus(output_root, replayed, max_bytes=max_bytes)
    finally:
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()


def run_tiny_trace(
    profile: HProfile,
    source_root: Path,
    trace_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    sink_builder: SinkBuilder = build_sink_for_run,
) -> TrainingProgress:
    """Train with the worker-shaped sink; this function owns the sink's lifecycle."""
    if profile.run_class != "local_tiny":
        raise ValueError("the local tracer runs only local_tiny profiles")
    output = trace_root / "run"
    output.mkdir(parents=True)
    # FOR-1642: the H run surface (params exp/seed/config_hash/git_sha/run_class/precision,
    # tags schema/exp/config_hash). `image` is omitted: a local tracer has no image, and a
    # placeholder would be unverified. The `-tiny` run-name suffix is TRACER-PARTITION
    # (conductor 2026-10-14): this producer bypasses object_worker and so writes no
    # h-run-lifecycle-v1 records; the suffix lets capture_results' include_tiny gate keep
    # it off the H lane, exactly as E smoke runs are kept off the E lane.
    chash = h_config_hash(profile)
    sink = sink_builder(
        TrackingRunDescriptor(
            run_name=f"h-tiny-trace-{profile.identity}-tiny",
            tiny=True,
            params={
                EXP_TAG: profile.name,
                "seed": profile.seed,
                CONFIG_HASH_TAG: chash,
                "git_sha": git_sha(),
                "run_class": profile.run_class,
                "precision": profile.precision,
            },
            tags={"schema": "h-tiny-trace-v1", EXP_TAG: profile.name, CONFIG_HASH_TAG: chash},
        ),
        results_path=trace_root / "observability" / "metrics.jsonl",
        metric_extractor=HTrainingStepExtractor(warmup_steps=profile.schedule.warmup_steps),
    )
    try:
        return run_probe_training(
            profile,
            source_root,
            output,
            max_source_bytes=max_source_bytes,
            max_artifact_bytes=max_artifact_bytes,
            sink=sink,
        )
    finally:
        sink.finish()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="h-tiny-trace", description="run a tiny H probe profile locally on CPU"
    )
    parser.add_argument("--profile", type=Path, required=True, help="remote or local profile")
    parser.add_argument("--corpus", type=Path, required=True, help="prepared-corpus.json")
    parser.add_argument("--out", type=Path, required=True, help="new or empty trace directory")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--max-source-bytes", type=int, default=_DEFAULT_LIMIT)
    parser.add_argument("--max-artifact-bytes", type=int, default=_DEFAULT_LIMIT)
    return parser


def _emit(root: Path, value: dict[str, object]) -> None:
    """File the command result beside the run; stdout stays free of raw prints."""
    (root / "result.json").write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    logger.info(f"h-tiny-trace {value['status']}: {root / 'result.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _parser().parse_args(argv)
    out = cast(Path, args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = normalize_d1_artifact_root(out)
    if any(root.iterdir()):
        raise SystemExit("h-tiny-trace: --out must be new or empty")
    profile_bytes = cast(Path, args.profile).read_bytes()
    remote = HProfile.from_manifest(json.loads(profile_bytes))
    source = root / "source"
    source.mkdir()
    corpus_bytes = cast(Path, args.corpus).read_bytes()
    artifact = reseal_foreign_corpus(args.corpus, source, max_bytes=args.max_source_bytes)
    local = localize_profile(
        remote, source_digest=artifact.content_hash.digest, precision=args.precision
    )
    corpus = read_prepared_corpus(source, artifact.content_hash, max_bytes=args.max_source_bytes)
    (root / "trace.json").write_text(
        json.dumps(
            {
                "schema": "h-tiny-trace-v1",
                "scientific_authority": "none",
                "from_profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
                "from_profile_identity": remote.identity,
                "from_corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
                "corpus_identity": corpus.identity,
                "local_source_digest": artifact.content_hash.digest,
                "local_profile_identity": local.identity,
                "precision": local.precision,
            },
            sort_keys=True,
            indent=2,
        )
    )
    try:
        progress = run_tiny_trace(
            local,
            source,
            root,
            max_source_bytes=args.max_source_bytes,
            max_artifact_bytes=args.max_artifact_bytes,
        )
    except Exception as error:  # the runner already filed its failure record
        failures = sorted((root / "run").glob("*-failure.json"))
        _emit(
            root,
            {
                "status": "scientific_gap"
                if isinstance(error, UndefinedDirection)
                else "execution_defect",
                "error_type": type(error).__name__,
                "error_code": getattr(error, "code", type(error).__name__),
                "detail": str(error),
                "failure": str(failures[-1]) if failures else None,
                "scientific_authority": "none",
            },
        )
        return 1
    _emit(
        root,
        {
            "status": progress.status,
            "last_seal": progress.last_seal.path,
            "receipt": progress.receipt.path,
            "scientific_authority": "none",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
