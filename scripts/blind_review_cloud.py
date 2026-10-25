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

Files the reviewer writes in its sandbox (FOR-1671). The full review often
lands in ``/tmp/outputs/review.md`` inside the container rather than in the
text blocks, so the files are collected too. Wire shape VERIFIED, not
inferred, against the claude-api skill (python/claude-api/tool-use.md,
"Retrieve Generated Files", skill bundle 2.1.308, read 2026-10-22), whose
example sits under ``code_execution_20260120`` (the tool pinned here):

- result block: ``block.type == "bash_code_execution_tool_result"``;
- ``block.content.type == "bash_code_execution_result"`` (an error result has
  another type and no ``.content`` list: it contributes no file);
- ``block.content.content`` lists entries with
  ``.type == "bash_code_execution_output"`` and ``.file_id``;
- the filename is NOT on the block: ``client.beta.files.retrieve_metadata(id)
  .filename``, sanitized with ``os.path.basename`` as the skill instructs;
- download: ``client.beta.files.download(id).write_to_file(path)``.

Each file lands in ``<out_md stem>-files/<name>`` beside ``<out_md>`` and its
path is printed. The skill is silent on whether a ``pause_turn`` resume
re-carries earlier rounds' blocks and the resume loop below resends only the
last assistant turn, so refs are gathered from EVERY round's response, not
the final one alone. Divergence recorded, not migrated: the skill documents
the Files API as out of beta (``client.files``, no header); this script stays
on ``client.beta.files`` + ``files-api-2025-04-14``, unchanged by this fix.
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


def collect_output_file_ids(content: list[Any]) -> list[str]:
    """File ids referenced by code-execution result blocks, in order.

    Shape per the module docstring. Blocks of any other type, and execution
    results that failed (no ``bash_code_execution_result`` payload), yield none.
    """
    ids: list[str] = []
    for block in content:
        if getattr(block, "type", None) != "bash_code_execution_tool_result":
            continue
        result = getattr(block, "content", None)
        if getattr(result, "type", None) != "bash_code_execution_result":
            continue
        for entry in getattr(result, "content", None) or []:
            if getattr(entry, "type", None) == "bash_code_execution_output":
                ids.append(entry.file_id)
    return ids


def download_output_files(
    client: Any, file_ids: list[str], dest: pathlib.Path
) -> list[pathlib.Path]:
    """Download each Files API id into ``dest``; return the saved paths.

    The sandbox controls the filename, so it is reduced to its basename; an
    empty or dot-only name falls back to the file id. A name already taken in
    ``dest`` is prefixed with the file id, and if that is taken too (the sandbox
    may write a file literally named ``<file_id>-<name>``) a counter is added
    until the name is free, so no download overwrites a file already in ``dest``.
    """
    if not file_ids:
        return []
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[pathlib.Path] = []
    for file_id in file_ids:
        name = os.path.basename(client.beta.files.retrieve_metadata(file_id).filename)
        if name in {"", ".", ".."}:
            name = file_id
        target = dest / name
        if target.exists():
            target = dest / f"{file_id}-{name}"
        counter = 1
        while target.exists():
            target = dest / f"{file_id}-{counter}-{name}"
            counter += 1
        client.beta.files.download(file_id).write_to_file(target)
        saved.append(target)
    return saved


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
        output_file_ids: list[str] = []
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
            for fid in collect_output_file_ids(response.content):
                if fid not in output_file_ids:
                    output_file_ids.append(fid)
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
        if not review.strip() and not output_file_ids:
            msg = "reviewer returned no text content"
            raise RuntimeError(msg)
        if review.strip():
            out.write_text(review, encoding="utf-8")
            print(f"wrote {out} ({len(review)} chars)")
        else:
            print("reviewer returned no text content; collecting output files only")
        files_dir = out.with_name(f"{out.stem}-files")
        for saved in download_output_files(client, output_file_ids, files_dir):
            print(f"saved {saved}")
        return 0
    finally:
        for fid in file_ids:
            try:
                client.beta.files.delete(fid)
            except Exception as cleanup_error:  # best-effort cleanup
                print(f"cleanup: could not delete {fid}: {cleanup_error}")


if __name__ == "__main__":
    raise SystemExit(main())
