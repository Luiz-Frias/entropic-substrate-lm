"""Immutable numerical journal shards; journals alone grant no scientific authority."""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.corpus import PreparedCorpus
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.sampling import SamplingSpec, SentenceSampler, VerifiedCorpus
from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    normalize_d1_relative_artifact_path,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
    relative_d1_artifact_path,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.run.execution import SentenceExecution, batch_coordinates
from entropic_substrate.ladder.run.mechanism_evidence import admit_mechanism

if TYPE_CHECKING:  # seals imports this module, so the memo type is named for the checker only
    from entropic_substrate.ladder.run.seals import CorpusMemo


def artifact_reference(root: Path, artifact: ArtifactIdentity) -> dict[str, str]:
    if artifact.content_hash.algorithm != "sha256":
        raise ValueError("H artifacts require SHA-256")
    name = (
        relative_d1_artifact_path(root, Path(artifact.path), label="H artifact")
        if Path(artifact.path).is_absolute()
        else normalize_d1_relative_artifact_path(artifact.path, label="H artifact")
    )
    return {"path": name, "sha256": artifact.content_hash.digest}


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate H artifact JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise ValueError(f"nonfinite H artifact JSON constant: {value}")


def write_json(root: Path, name: str, value: object, *, max_bytes: int) -> ArtifactIdentity:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("H artifact byte limit must be positive")
    payload = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > max_bytes:
        raise ValueError("H artifact exceeds its byte limit")
    return publish_d1_atomic_bytes(root, name, payload, label="H JSON artifact")


def read_json(root: Path, artifact: ArtifactIdentity, *, max_bytes: int) -> dict[str, object]:
    ref = artifact_reference(root, artifact)
    _, payload, digest = read_d1_artifact_bytes(
        root, ref["path"], label="H JSON artifact", max_bytes=max_bytes
    )
    if digest != ref["sha256"]:
        raise ValueError("H artifact hash mismatch")
    value: object = json.loads(payload, object_pairs_hook=_unique, parse_constant=_nonfinite)
    if type(value) is not dict:
        raise ValueError("H artifact must be an object")
    return cast(dict[str, object], value)


def _number(value: object, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise ValueError("journal measurement must be finite and not boolean")
    result = float(cast(float, value))
    if nonnegative and result < 0:
        raise ValueError("journal measurement must be nonnegative")
    return result


def _validate(
    payload: object, profile: HProfile, corpus: PreparedCorpus | VerifiedCorpus
) -> dict[str, object]:
    journal = recovery_record(
        payload,
        "schema run_identity run_class scientific_authority phase start_step end_step parent rows",
    )
    if (
        journal["schema"] not in ("h-journal-shard-v1", "h-journal-shard-v2")
        or journal["run_identity"] != profile.identity
        or journal["run_class"] != profile.run_class
        or journal["scientific_authority"] != "none"
    ):
        raise ValueError("journal construction or authority differs from profile")
    phase, start, end, rows = (journal[k] for k in ("phase", "start_step", "end_step", "rows"))
    mechanism = journal["schema"] == "h-journal-shard-v2"
    if (
        phase not in ("codec_warmup", "prediction")
        or type(start) is not int
        or type(end) is not int
        or not 1 <= start <= end
    ):
        raise ValueError("invalid journal phase or step range")
    budget = profile.schedule.warmup_steps if phase == "codec_warmup" else profile.schedule.steps
    if end > budget or not isinstance(rows, (list, tuple)) or len(rows) != end - start + 1:
        raise ValueError("journal must contain an in-budget consecutive prefix")
    if journal["parent"] is not None:
        artifact_from_mapping(journal["parent"])
    sampler = SentenceSampler(
        corpus,
        SamplingSpec(
            "train",
            profile.schedule.batch_size,
            0 if phase == "codec_warmup" else profile.spine.context_sentences,
            profile.source_seed,
        ),
        SentenceByteCodec(profile.permutation),
        torch.device("cpu"),
    )
    sampler.restore(sampler.snapshot() | {"cursor": start - 1})
    for index, value in enumerate(cast(list[object], rows), start):
        row = recovery_record(
            value,
            "schema run_identity phase step source_batch source_examples status objective aggregate_norm gradients meaning decoder_ce future_nll_per_dimension source_bytes boundary_symbols history_bytes encoder_bytes wall_seconds cuda_seconds cuda_peak_bytes peak_scope"
            + (" mechanism" if mechanism else ""),
        )
        if (
            row["schema"] != ("h-training-step-v2" if mechanism else "h-training-step-v1")
            or row["run_identity"] != profile.identity
            or row["phase"] != phase
            or type(row["step"]) is not int
            or row["step"] != index
            or row["status"] != "committed"
        ):
            raise ValueError("journal rows must be consecutive committed steps of this run")
        batch = sampler.next_batch()
        if mechanism:
            admit_mechanism(
                row["mechanism"],
                profile,
                phase=phase,
                batch=len(batch.examples),
                history=len(batch.examples[0].history),
            )
        if row["source_batch"] != batch.identity or json.dumps(
            row["source_examples"], sort_keys=True
        ) != json.dumps(batch_coordinates(batch), sort_keys=True):
            raise ValueError("journal source coordinates differ from independent replay")
        source = sum(len(e.target.source) for e in batch.examples)
        history = sum(len(s.source) for e in batch.examples for s in e.history)
        encoder = (
            source
            if phase == "codec_warmup"
            else history + (source if profile.optimization.future_weight else 0)
        )
        for key, expected in (
            ("source_bytes", source),
            ("history_bytes", history),
            ("encoder_bytes", encoder),
            ("boundary_symbols", len(batch.examples)),
        ):
            if type(row[key]) is not int or row[key] != expected:
                raise ValueError("journal exposure count differs from source replay")
        gradients = recovery_record(row["gradients"], "encoder decoder spine future_law")
        norms: list[float] = []
        for component, item in gradients.items():
            evidence = recovery_record(item, "status norm")
            absent = (
                "structurally_absent_codec_warmup"
                if phase == "codec_warmup" and component in ("spine", "future_law")
                else "structurally_absent_frozen_encoder"
                if phase == "prediction" and component == "encoder"
                else None
            )
            if absent is not None:
                if evidence != {"status": absent, "norm": None}:
                    raise ValueError("journal structural absence differs from phase")
            else:
                norm = _number(evidence["norm"], nonnegative=True)
                if evidence["status"] != ("measured_zero" if norm == 0 else "measured_nonzero"):
                    raise ValueError("journal gradient category differs from its norm")
                norms.append(norm)
        if not math.isclose(
            _number(row["aggregate_norm"], nonnegative=True),
            math.hypot(*norms),
            rel_tol=1e-5,
            abs_tol=1e-12,
        ):
            raise ValueError("journal gradient norms do not reconcile")
        decoder = _number(row["decoder_ce"], nonnegative=True)
        expected_loss = decoder
        if phase == "prediction" and profile.optimization.future_weight > 0:
            expected_loss += profile.optimization.future_weight * _number(
                row["future_nll_per_dimension"]
            )
        elif row["future_nll_per_dimension"] is not None:
            raise ValueError("disabled objective has a journal measurement")
        tolerance = (
            4 * torch.finfo(torch.float32).eps * max(1, decoder, abs(expected_loss - decoder))
        )
        if not math.isclose(_number(row["objective"]), expected_loss, rel_tol=0, abs_tol=tolerance):
            raise ValueError("journal objective does not match its scored components")
        meaning = (
            "oracle_reconstruction_training"
            if phase == "codec_warmup"
            else "prior_sample_decoder_training"
        )
        if row["meaning"] != meaning or row["peak_scope"] != "process_allocator_high_water":
            raise ValueError("journal score or resource meaning differs from execution")
        _number(row["wall_seconds"], nonnegative=True)
        if profile.device == "cuda":
            _number(row["cuda_seconds"], nonnegative=True)
            peak = row["cuda_peak_bytes"]
            if (
                not isinstance(peak, (list, tuple))
                or len(peak) != 2
                or any(type(n) is not int or n < 0 for n in peak)
                or peak[1] < peak[0]
            ):
                raise ValueError("journal CUDA memory counters are invalid")
        elif row["cuda_seconds"] is not None or row["cuda_peak_bytes"] is not None:
            raise ValueError("CPU journal cannot claim CUDA measurements")
    return journal


def write_journal(
    root: Path,
    name: str,
    engine: SentenceExecution,
    rows: tuple[dict[str, object], ...],
    *,
    parent: ArtifactIdentity | None,
    max_bytes: int,
) -> ArtifactIdentity:
    if not rows or engine.failed:
        raise ValueError("restart journals require successful training rows")
    payload = {
        "schema": "h-journal-shard-v2",
        "run_identity": engine.profile.identity,
        "run_class": engine.profile.run_class,
        "scientific_authority": "none",
        "phase": engine.phase,
        "start_step": rows[0]["step"],
        "end_step": engine.step,
        "parent": None if parent is None else artifact_reference(root, parent),
        "rows": rows,
    }
    expected = ContentHash("sha256", engine.profile.source_digest)
    corpus = (
        read_prepared_corpus(engine.source_root, expected, max_bytes=engine.max_source_bytes)
        if engine.corpora is None
        else engine.corpora.read(engine.source_root, expected, max_bytes=engine.max_source_bytes)
    )
    _validate(payload, engine.profile, corpus)
    return write_json(root, name, payload, max_bytes=max_bytes)


def read_journal(
    root: Path,
    artifact: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_bytes: int,
    max_source_bytes: int,
    corpora: "CorpusMemo | None" = None,
) -> dict[str, object]:
    payload = read_json(root, artifact, max_bytes=max_bytes)
    expected = ContentHash("sha256", profile.source_digest)
    corpus = (
        read_prepared_corpus(source_root, expected, max_bytes=max_source_bytes)
        if corpora is None
        else corpora.read(source_root, expected, max_bytes=max_source_bytes)
    )
    return _validate(payload, profile, corpus)
