"""Bounded FineWeb sentence-source preparation in the consuming reader runtime."""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path

from entropic_substrate.core.sentence.corpus import (
    PreparationSpec,
    SplitPolicy,
    prepare_documents,
)
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.segmentation import SegmentationPolicy
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.capture import (
    read_prepared_corpus,
    write_prepared_corpus,
)
from entropic_substrate.ladder.capture.custody.fineweb import FINEWEB_REVISION
from entropic_substrate.ladder.run.journal import artifact_reference, read_json


def add_source_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    for name in ("h-prepare-source", "h-inspect-source"):
        parser = sub.add_parser(name, help="prepare or replay a bounded sentence-source capture")
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--manifest", required=True)


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("source policy counts must be integers, not booleans")
    return value


def cmd_source(args: argparse.Namespace) -> int:
    """Require every policy field and re-read the sealed source before returning."""
    stage = "validation"
    try:
        root = normalize_d1_artifact_root(args.root)
        _, _, digest = read_d1_artifact_bytes(
            root, args.manifest, label="H source command", max_bytes=1_048_576
        )
        raw = read_json(
            root,
            ArtifactIdentity(args.manifest, ContentHash("sha256", digest)),
            max_bytes=1_048_576,
        )
        preparing = args.command == "h-prepare-source"
        row = recovery_record(
            raw,
            "schema dataset_name revision output_root max_artifact_bytes candidate_limit "
            "admitted_byte_limit max_sentence_bytes segmentation_version split_seed "
            "validation_buckets test_buckets"
            if preparing
            else "schema source_root source max_artifact_bytes",
        )
        schema = "h-prepare-source-command-v1" if preparing else "h-inspect-source-command-v1"
        if row["schema"] != schema:
            raise ValueError("unsupported source command schema")
        limit = row["max_artifact_bytes"]
        if type(limit) is not int or limit < 1:
            raise ValueError("source capture byte limit must be a positive integer")
        relative = normalize_d1_relative_artifact_path(
            row["output_root" if preparing else "source_root"], label="source output directory"
        )
        output = normalize_d1_artifact_root(root / relative)
        if preparing:
            if row["revision"] != FINEWEB_REVISION:
                raise ValueError("source revision differs from the pinned FineWeb revision")
            if row["segmentation_version"] != SegmentationPolicy.engine_version:
                raise ValueError("segmentation construction differs from the declared version")
            name = row["dataset_name"]
            if type(name) is not str or not name.strip() or name != name.strip():
                raise ValueError("dataset name must be explicit and canonical")
            spec = PreparationSpec(
                _integer(row["candidate_limit"]),
                _integer(row["admitted_byte_limit"]),
                FINEWEB_REVISION,
                SegmentationPolicy(_integer(row["max_sentence_bytes"])),
                SplitPolicy(
                    _integer(row["split_seed"]),
                    _integer(row["validation_buckets"]),
                    _integer(row["test_buckets"]),
                ),
                dataset_name=name,
            )
            if any(output.iterdir()):
                raise ValueError("source preparation requires an empty owned output directory")
            from entropic_substrate.ladder.capture.custody.fineweb import stream_fineweb_documents

            stage = "execution"
            with closing(
                stream_fineweb_documents(max_documents=spec.candidate_limit, name=name)
            ) as documents:
                corpus = prepare_documents(documents, spec)
            source = write_prepared_corpus(output, corpus, max_bytes=limit)
        else:
            source = artifact_from_mapping(row["source"])
            if source.path != "prepared-corpus.json" or source.content_hash.algorithm != "sha256":
                raise ValueError("source inspection requires the canonical SHA-256 capture")
            stage = "execution"
        corpus = read_prepared_corpus(output, source.content_hash, max_bytes=limit)
        result = {
            "status": "source_verified",
            "scientific_authority": "none",
            "source": artifact_reference(output, source),
            "corpus_identity": corpus.identity,
            "corpus_status": corpus.status,
            "stopped_by": corpus.stopped_by,
            "examined_documents": corpus.examined_documents,
            "admitted_documents": len(corpus.records),
            "admitted_bytes": corpus.admitted_bytes,
            "rejected_documents": len(corpus.rejections),
        }
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "status": "invalid_request" if stage == "validation" else "execution_defect",
                    "error_type": type(error).__name__,
                    "scientific_authority": "none",
                },
                sort_keys=True,
            )
        )
        return 2 if stage == "validation" else 1
