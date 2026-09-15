"""Validated primitive/tensor payloads for RNG and bytewise generation recovery."""

import math
from typing import cast

import torch

from entropic_substrate.sentence_prediction.boundary import SentenceByteCodec
from entropic_substrate.sentence_prediction.model import SentencePredictor
from entropic_substrate.sentence_prediction.numerics import checked_coordinates
from entropic_substrate.sentence_prediction.rollout import (
    SentenceEmission,
    SentenceRollout,
)


def recovery_record(value: object, keys: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError("recovery payload fields differ from the versioned schema")
    return cast(dict[str, object], value)


def capture_rng() -> dict[str, object]:
    """Torch CPU and initialized CUDA streams; source sampling has its own cursor."""
    return {
        "schema": "h-torch-rng-v1",
        "cpu": torch.get_rng_state().clone(),
        "cuda": tuple(s.clone() for s in torch.cuda.get_rng_state_all())
        if torch.cuda.is_initialized()
        else (),
    }


def restore_rng(value: object) -> None:
    row = recovery_record(value, "schema cpu cuda")
    cuda = row["cuda"]
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_initialized() else 0
    if row["schema"] != "h-torch-rng-v1" or type(cuda) is not tuple or len(cuda) != expected_cuda:
        raise ValueError("RNG version or device topology mismatch")
    states = (row["cpu"], *cuda)
    for index, state in enumerate(states):
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.device.type != "cpu"
        ):
            raise ValueError("RNG state must be a CPU byte vector")
        device = "cpu" if index == 0 else f"cuda:{index - 1}"
        try:
            torch.Generator(device=device).set_state(state)
        except RuntimeError as error:
            raise ValueError("invalid RNG state bytes") from error
    # All streams are validated on isolated generators before global mutation.
    torch.set_rng_state(cast(torch.Tensor, row["cpu"]))
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def _vector(value: object, shape: tuple[int, ...], model: SentencePredictor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError("recovery latent shape mismatch")
    if value.dtype != model.spine.bos.dtype or not bool(torch.isfinite(value).all()):
        raise ValueError("recovery latent precision or finiteness mismatch")
    if model.codec.law_spec.mode == "sphere" and value.numel():
        if bool((checked_coordinates(value).log_radius.abs() > 1e-5).any()):
            raise ValueError("recovery spherical latent is not unit norm")
    return value.detach().to(model.spine.bos.device).clone()


def _integers(value: object, limit: int) -> tuple[int, ...]:
    if type(value) is not tuple or any(type(v) is not int or not 0 <= v < limit for v in value):
        raise ValueError("recovery observations must be bounded integer tuples")
    return cast(tuple[int, ...], value)


def rollout_from_payload(
    value: object, model: SentencePredictor, codec: SentenceByteCodec
) -> SentenceRollout | None:
    """Validate framing and feedback relationships before returning resumed state.

    Numerical equality to a fresh encoder run is not assumed across precision
    modes: the enclosing checkpoint must bind model weights and runtime policy.
    Completed operative vectors must match their recorded context slots exactly.
    """
    if value is None:
        return None
    row = recovery_record(
        value,
        "schema codec_identity history context pending sample emissions observations log_scores status",
    )
    if row["schema"] != "h-rollout-v1" or row["codec_identity"] != codec.identity:
        raise ValueError("recovery codec or rollout version mismatch")
    status = row["status"]
    if status not in (
        "active",
        "document_end",
        "empty_sentence",
        "invalid_utf8",
        "length_limit",
        "context_limit",
    ):
        raise ValueError("unknown recovered rollout status")
    history = row["history"]
    if type(history) is not tuple or len(history) > model.spine.spec.context_sentences + 1:
        raise ValueError("recovery history exceeds the bounded context")
    for source in history:
        if type(source) is not bytes or not 1 <= len(source) <= model.codec.spec.max_sentence_bytes:
            raise ValueError("invalid recovered sentence bytes")
        source.decode("utf-8", errors="strict")
    history = cast(tuple[bytes, ...], history)
    width = model.codec.spec.width
    context = _vector(row["context"], (1, len(history), width), model)
    pending = _integers(row["pending"], 256)
    observations = _integers(row["observations"], 258)
    scores = row["log_scores"]
    if (
        type(scores) is not tuple
        or len(scores) != len(observations)
        or any(type(s) is not float or not math.isfinite(s) or s > 0 for s in scores)
    ):
        raise ValueError("recovery token scores do not match observations")
    emissions = row["emissions"]
    if type(emissions) is not tuple or len(emissions) > len(history):
        raise ValueError("invalid recovered emission roster")
    offset = len(history) - len(emissions)
    restored: list[SentenceEmission] = []
    trace: list[int] = []
    for index, item in enumerate(emissions):
        entry = recovery_record(item, "source terminal shadow operative")
        terminal = entry["terminal"]
        if (
            type(terminal) is not int
            or terminal not in (256, 257)
            or entry["source"] != history[offset + index]
        ):
            raise ValueError("emission and recovered source history disagree")
        if terminal == 257 and (index != len(emissions) - 1 or status != "document_end"):
            raise ValueError("document-end does not match recovered status")
        shadow = _vector(entry["shadow"], (1, width), model)
        operative = _vector(entry["operative"], (1, width), model)
        if not torch.equal(context[:, offset + index], operative):
            raise ValueError("recovered operative feedback differs from context")
        source = history[offset + index]
        restored.append(SentenceEmission(source, terminal, shadow, operative))
        trace.extend((*[codec.permutation[b] for b in source], terminal))
    trace.extend(pending)
    if status in ("empty_sentence", "invalid_utf8"):
        if not observations or observations[-1] not in (256, 257):
            raise ValueError("failed boundary lacks its sampled control")
        trace.append(observations[-1])
    if tuple(trace) != observations:
        raise ValueError("recovered observation trace differs from feedback")
    if status == "empty_sentence" and pending:
        raise ValueError("empty-sentence failure contains bytes")
    if status == "invalid_utf8":
        try:
            bytes(codec.canonical_byte(b) for b in pending).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            invalid_utf8 = True
        else:
            invalid_utf8 = False
        if not invalid_utf8:
            raise ValueError("invalid-UTF8 status contains valid text")
    limit = model.codec.spec.max_sentence_bytes
    if (status == "length_limit" and len(pending) != limit + 1) or (
        status != "length_limit" and len(pending) > limit
    ):
        raise ValueError("recovered length outcome is inconsistent")
    if status == "document_end" and (not restored or restored[-1].terminal != 257):
        raise ValueError("recovered document-end has no terminal emission")
    if (status == "context_limit") != (
        len(history) > model.spine.spec.context_sentences and status != "document_end"
    ):
        raise ValueError("recovered context outcome is inconsistent")
    sample = None if row["sample"] is None else _vector(row["sample"], (1, width), model)
    needs_sample = bool(pending) or status in ("empty_sentence", "invalid_utf8", "length_limit")
    if (sample is not None) != needs_sample or (
        status in ("document_end", "context_limit") and pending
    ):
        raise ValueError("recovered pending prefix and sampled latent disagree")
    return SentenceRollout(
        codec.identity,
        history,
        context,
        pending,
        sample,
        tuple(restored),
        observations,
        cast(tuple[float, ...], scores),
        status,
    )


def rollout_to_payload(
    state: SentenceRollout | None, model: SentencePredictor, codec: SentenceByteCodec
) -> dict[str, object] | None:
    if state is None:
        return None
    result: dict[str, object] = {
        "schema": "h-rollout-v1",
        "codec_identity": state.codec_identity,
        "history": state.history,
        "context": state.context.detach().cpu().clone(),
        "pending": state.pending,
        "sample": None if state.sample is None else state.sample.detach().cpu().clone(),
        "emissions": tuple(
            {
                "source": e.source,
                "terminal": e.terminal,
                "shadow": e.shadow.detach().cpu().clone(),
                "operative": e.operative.detach().cpu().clone(),
            }
            for e in state.emissions
        ),
        "observations": state.observations,
        "log_scores": state.log_scores,
        "status": state.status,
    }
    rollout_from_payload(result, model, codec)
    return result
