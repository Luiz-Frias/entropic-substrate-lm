"""FineWeb-Edu streaming loader (HuggingFace ``datasets``, streaming=True).

Following the holobyte reference: pull documents from the ``sample-10BT``
config, join them with ``<|endoftext|>`` and treat the result as a utf-8 byte
stream. Streaming avoids materialising the 10BT shard on disk, which matters
on ephemeral GPU boxes (RunPod / Vast).
"""

from __future__ import annotations

from collections.abc import Iterator

__all__ = [
    "FINEWEB_EOT",
    "FINEWEB_REPO",
    "FINEWEB_REVISION",
    "fineweb_text",
    "stream_fineweb_bytes",
]

FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"  # pragma: allowlist secret
FINEWEB_EOT = "<|endoftext|>"


def _iter_documents(name: str, split: str) -> Iterator[str]:
    from datasets import load_dataset  # imported lazily: heavy + optional at test time

    dataset = load_dataset(  # nosec B615 -- immutable revision constant is forwarded below
        FINEWEB_REPO,
        name=name,
        split=split,
        streaming=True,
        revision=FINEWEB_REVISION,
    )
    for item in dataset:
        text = item.get("text") if isinstance(item, dict) else None
        if text:
            yield str(text)


def fineweb_text(
    num_chars: int = 20_000_000,
    *,
    name: str = "sample-10BT",
    split: str = "train",
) -> str:
    """Concatenate streamed documents until ``num_chars`` characters are held."""
    parts: list[str] = []
    total = 0
    for text in _iter_documents(name, split):
        chunk = text + FINEWEB_EOT
        parts.append(chunk)
        total += len(chunk)
        if total >= num_chars:
            break
    return "".join(parts)[:num_chars]


def stream_fineweb_bytes(
    max_bytes: int | None = None,
    *,
    name: str = "sample-10BT",
    split: str = "train",
) -> Iterator[bytes]:
    """Yield utf-8 encoded document chunks, capped at ``max_bytes`` total.

    Useful for :func:`entropic_substrate.init.fisher.count_transitions`, which
    consumes an iterable of byte chunks without buffering the whole corpus.
    """
    emitted = 0
    for text in _iter_documents(name, split):
        blob = (text + FINEWEB_EOT).encode("utf-8")
        if max_bytes is not None:
            remaining = max_bytes - emitted
            if remaining <= 0:
                return
            blob = blob[:remaining]
        emitted += len(blob)
        yield blob
        if max_bytes is not None and emitted >= max_bytes:
            return
