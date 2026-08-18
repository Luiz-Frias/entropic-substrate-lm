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
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

MODEL = "claude-opus-5"
BETAS = ["files-api-2025-04-14"]
CODE_TOOL: dict[str, str] = {"type": "code_execution_20260120", "name": "code_execution"}

# Files API uploads are flat; packet paths are nested. '/' -> '__' keeps the
# layout recoverable and leaves file CONTENT (and so every manifest md5)
# untouched. The prompt suffix below tells the reviewer the mapping.
NAME_MAPPING_NOTE = (
    "\n\nNOTE ON FILE NAMES: the packet files are provided to your execution "
    "container with '/' in their relative paths replaced by '__' (e.g. "
    "data/A-seed1.jsonl arrives as data__A-seed1.jsonl). manifest.txt refers "
    "to the original layout; file CONTENTS are byte-identical, so every md5 "
    "in the manifest must still verify against the renamed files."
)


def flatten_name(relative_path: pathlib.PurePath) -> str:
    """Container-safe flat filename for a nested packet path."""
    return "__".join(relative_path.parts)


def build_content(prompt_text: str, file_ids: list[str]) -> list[dict[str, Any]]:
    """One user message: verbatim prompt first, then every packet file."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text + NAME_MAPPING_NOTE}]
    content += [{"type": "container_upload", "file_id": fid} for fid in file_ids]
    return content


def main() -> int:
    import anthropic  # deferred: `uv run --with anthropic` supplies it

    packet, out, prompt_file = (pathlib.Path(a) for a in sys.argv[1:4])
    client = anthropic.Anthropic()

    file_ids: list[str] = []
    try:
        for path in sorted(p for p in packet.rglob("*") if p.is_file()):
            name = flatten_name(path.relative_to(packet))
            with path.open("rb") as handle:
                uploaded = client.beta.files.upload(file=(name, handle, "text/plain"))
            file_ids.append(uploaded.id)
            print(f"uploaded {name} -> {uploaded.id}")

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": build_content(prompt_file.read_text(encoding="utf-8"), file_ids),
            }
        ]
        for round_index in range(12):  # pause_turn resume bound, fail loud past it
            with client.beta.messages.stream(
                model=MODEL,
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
