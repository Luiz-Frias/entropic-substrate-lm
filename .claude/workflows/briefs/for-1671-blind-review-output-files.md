# for-1671-blind-review-output-files: collect the files the blind reviewer writes in its sandbox

DRAFT 2026-10-22 by the conductor (wave C, lane of a K=3 mass-patch-fanout; issue FOR-1671, filed from the
H0 blind review round 1 where the reviewer's full review in `/tmp/outputs/review.md` was dropped and only a
4,924-character summary survived). Anchors measured at origin/main `70d2b5f1` with `grep -n`; the implementer
re-verifies them in recon. The Linear ticket is authoritative.

## Branch

| field                | value                                                                 |
| -------------------- | --------------------------------------------------------------------- |
| branch               | `fix/for-1671-blind-review-output-files`                              |
| parent               | the wave base branch (wave B's zipped tip + the briefs commit)        |
| cascade `phase.mode` | `normal` (an internal script; no port or published surface changes)   |
| publish              | none in the lane (`publishMode: defer`, CERTIFIED); the zip publishes |

## Why (measured at `70d2b5f1`)

- `scripts/blind_review_cloud.py`: `CODE_TOOL` (`:31`, `code_execution_20260120`), `build_content` (`:74-77`, prompt then `container_upload` blocks), `main` (`:81`), upload `:98`, the stream call `:110-113` (`betas=BETAS`), the `pause_turn` resume loop (`for round_index in range(12)`), and the text-only collection `review = "\n".join(block.text for block in response.content if block.type == "text")` followed by the `"reviewer returned no text content"` RuntimeError and `out.write_text(review)`. Files the sandbox writes are never read.
- `tests/test_blind_review_cloud.py`: 4 tests (`:29` zip layout, `:44` prompt-first content, `:53` model from env, `:68` blank model refused before any client); the fake client pattern at `:77` is the one to extend.
- Not a reader-tuple member (`ladder/run/checkpoint.py` `_READER_MODULES` lists no `scripts.*`); ordinary lane commit. `scripts/*` is normally conductor-frozen (wave ruling 6); this lane has an explicit carve-out for `scripts/blind_review_cloud.py` only (same shape as the FOR-1609 R2 carve-out).

## Scope

1. **C1, tests RED at the parent**: a stubbed final response carrying one text block and one code-execution tool-result block that references an output file; the script writes the text to `<out_md>` AND downloads the referenced file into `<out_md stem>-files/<name>` through the Files API, printing each saved path. RED at the parent: the directory is never created.
2. **C2, implementation**: after the loop, walk `response.content` for the code-execution result blocks that carry file references; download each via the Files API (`client.beta.files.download` or the verified equivalent) into the sibling directory; keep the text path byte-identical in behaviour; fail loud (RuntimeError) only when there is neither text nor any output file. The wire shape (block `type` names, the field holding the file id) is VERIFIED against the claude-api skill's Files API / code-execution reference (`shared/live-sources.md`) before coding, never inferred; the verified names and the date go in the module docstring.
3. **C3**: no prompt-protocol or packet-assembler change (out of scope per the ticket).

## Guard teeth by mutation

| id | mutation                                          | expected RED                                   |
| -- | ------------------------------------------------- | ---------------------------------------------- |
| M1 | drop the download loop                            | the file-landing pin RED                       |
| M2 | write the file but skip the text path             | the existing text pin (`<out_md>` content) RED |
| M3 | raise when text is present but no file is written | the text-only (no files) control case RED      |
| M4 | ignore a second output file (only download one)   | the two-file pin RED                           |

## Allowed files

`scripts/blind_review_cloud.py` (explicit carve-out), `tests/test_blind_review_cloud.py`. Never: any other `scripts/*`, `tools/*`, `.github/workflows/*`, `justfile`, docs, reader-tuple members, `pyproject.toml`, `uv.lock`.

## Acceptance

C1 RED at the parent, GREEN at the tip; M1-M4 RED one at a time and restored with the RED output quoted; `just check` exit 0 and the exit-gate runner green under the suite lock; the verified block/field names recorded with their source and date in the module docstring and the commit message; diff confined to the Allowed files.
