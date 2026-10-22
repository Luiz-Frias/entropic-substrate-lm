"""Cloud-isolated blind review: Anthropic Files API + code-execution sandbox.

The strongest isolation lane for the double-blind review
(docs/peer-review-protocol.md): the reviewer model runs Python inside
Anthropic's server-side sandbox, which has NO network access (enforced, not
instructed), and sees ONLY the uploaded packet files plus the verbatim
reviewer prompt. No repository, no memory, no operator machine.

Operator-side tool; fail-loud. Wire contract (verified against the claude-api
skill 2026-08-18): code_execution_20260120 tool (GA, no beta header) +
files-api-2025-04-14 beta on upload AND the messages call; container_upload
content blocks mount files into the sandbox; a server-side loop that stops
with stop_reason "pause_turn" is resumed by resending user + assistant turns.

Run: ANTHROPIC_API_KEY=... uv run --with anthropic python \
    scripts/blind_review_cloud.py <packet_dir> <out_md> <prompt_txt>
The reviewer model is BLIND_REVIEW_MODEL when set (e.g. claude-opus-5-5),
else DEFAULT_MODEL; a blank value is refused.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

MODEL_ENV = "BLIND_REVIEW_MODEL"
DEFAULT_MODEL = "claude-opus-5"
BETAS = ["files-api-2025-04-14"]
CODE_TOOL: dict[str, str] = {"type": "code_execution_20260120", "name": "code_execution"}

# The API caps container uploads at 16 files per request (measured 2026-08-18:
# the 21-file n=9 packet was rejected with "Cannot add more than 16 files").
# Ship ONE zip instead — relative paths survive inside it, so the manifest
# verifies against the real layout with no renaming.
ZIP_NOTE = (
    "\n\nNOTE ON DELIVERY: the packet arrives in your execution container as "
    "a single file, packet.zip. First extract it (Python's zipfile module) "
    "into ./packet — the archive preserves the packet's relative layout, so "
    "manifest.txt paths resolve directly against the extracted tree."
)


def resolve_model() -> str:
    """The reviewer model: ``$BLIND_REVIEW_MODEL`` when set, else ``DEFAULT_MODEL``.

    A set-but-blank value is refused rather than silently falling back, so a
    typo in the operator's environment never runs the review on a model they
    did not choose (fail-loud, house rule 5).
    """
    raw = os.environ.get(MODEL_ENV)
    if raw is None:
        return DEFAULT_MODEL
    value = raw.strip()
    if not value:
        raise ValueError(f"{MODEL_ENV} is set but blank; unset it or name a model")
    return value


def build_packet_zip(packet_dir: pathlib.Path, zip_path: pathlib.Path) -> list[str]:
    """Zip the packet preserving relative paths; return the sorted arcnames."""
    import zipfile

    arcnames: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in packet_dir.rglob("*") if p.is_file()):
            arcname = path.relative_to(packet_dir).as_posix()
            archive.write(path, arcname)
            arcnames.append(arcname)
    return arcnames


def build_content(prompt_text: str, file_ids: list[str]) -> list[dict[str, Any]]:
    """One user message: verbatim prompt first, then the packet archive."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text + ZIP_NOTE}]
    content += [{"type": "container_upload", "file_id": fid} for fid in file_ids]
    return content


def main() -> int:
    import anthropic  # deferred: `uv run --with anthropic` supplies it

    packet, out, prompt_file = (pathlib.Path(a) for a in sys.argv[1:4])
    # Resolve configuration before any client, upload or spend exists, so a bad
    # BLIND_REVIEW_MODEL fails here and is never masked by an upload failure.
    model = resolve_model()
    client = anthropic.Anthropic()

    file_ids: list[str] = []
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            zip_path = pathlib.Path(scratch) / "packet.zip"
            arcnames = build_packet_zip(packet, zip_path)
            with zip_path.open("rb") as handle:
                uploaded = client.beta.files.upload(file=("packet.zip", handle, "application/zip"))
        file_ids.append(uploaded.id)
        print(f"uploaded packet.zip ({len(arcnames)} files) -> {uploaded.id}")

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": build_content(prompt_file.read_text(encoding="utf-8"), file_ids),
            }
        ]
        print(f"reviewer model: {model}")
        for round_index in range(12):  # pause_turn resume bound, fail loud past it
            with client.beta.messages.stream(
                model=model,
                max_tokens=32000,
                betas=BETAS,
                messages=messages,
                tools=[CODE_TOOL],
            ) as stream:
                response = stream.get_final_message()
            print(f"round {round_index}: stop_reason={response.stop_reason}")
            if response.stop_reason == "pause_turn":
                messages = [
                    messages[0],
                    {"role": "assistant", "content": response.content},
                ]
                continue
            if response.stop_reason == "refusal":
                msg = f"reviewer request refused: {response.stop_details}"
                raise RuntimeError(msg)
            break
        else:
            msg = "server-side loop never completed within the resume bound"
            raise RuntimeError(msg)

        review = "\n".join(block.text for block in response.content if block.type == "text")
        if not review.strip():
            msg = "reviewer returned no text content"
            raise RuntimeError(msg)
        out.write_text(review, encoding="utf-8")
        print(f"wrote {out} ({len(review)} chars)")
        return 0
    finally:
        for fid in file_ids:
            try:
                client.beta.files.delete(fid)
            except Exception as cleanup_error:  # best-effort cleanup
                print(f"cleanup: could not delete {fid}: {cleanup_error}")


if __name__ == "__main__":
    raise SystemExit(main())
