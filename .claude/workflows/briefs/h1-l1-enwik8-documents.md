# FOR-1689 [h1-l1] Add the enwik8 document adapter to ladder/capture/custody (one <page> per document, SplitPolicy buckets, bounded capture)

Source of truth: Linear FOR-1689 (project P-FOR-44). In-repo copy for the W1 lane (preflight corrections of 2026-10-28 applied on the Linear issue; anchors measured at main 45330219). The Linear issue text is authoritative where the two differ.

Wave W1 (K=3, non-member). Seat Opus xhigh. Class: feature (ruled: documents = one <page> element each; capture cap raised to the corpus size; corpus held in memory on the pod).

## Touch-set (non-member)

- NEW src/entropic_substrate/ladder/capture/custody/enwik8_documents.py (page splitter + document stream adapter beside the raw loader ladder/capture/custody/enwik8.py; reuses core/sentence/corpus.py:21-66 SplitPolicy by content-hash bucket per document)
- NEW scripts/prepare_enwik8_source.py (thin wrapper in the shape of scripts/h0_final_probe_checks.py; writes the same capture shape as h-prepare-source, read back through read_prepared_corpus)
- NEW tests/test_enwik8_documents.py
- NOT touched: composition/source_cli.py, ladder/capture/capture.py, core/sentence/* (reader members; a dispatch there is W2's call)

## Why

The H1 sealed corpus is the full enwik8 as a sentence capture. h-prepare-source is FineWeb-only (composition/source_cli.py:79-80 revision pin, :100-104 stream); ladder/capture/custody/enwik8.py is a raw-byte memmap with no document notion; nothing turns data/enwik8 (DVC, 100,000,073 bytes) into the prepared-corpus.json shape read_prepared_corpus admits.

## RED-first tests

page splitting on <page> boundaries; the final page truncated at the 10^8-byte cut and at a split UTF-8 character; exactly one revision string for the whole capture; unique source ids; write then read round trip through read_prepared_corpus; a bounded 10 MB capture.

UTF-8 truncation rule, as shipped: the 10^8-byte cut drops only an incomplete trailing UTF-8 sequence (at most 3 bytes), backing up to a code-point boundary, and any other invalid byte fails loud as a ValueError. The trimmed byte count is exposed as a "trimmed" segment and as `Enwik8Partition.trimmed_tail_bytes` (src/entropic_substrate/ladder/capture/custody/enwik8_documents.py: `_trimmed_tail_bytes` :105-115, `scan_enwik8` :145-148, `trimmed_tail_bytes` :82-84). tests/test_enwik8_documents.py:197-215 pins the 2-, 3- and 4-byte split characters, asserting the trimmed count, the final byte offset and the decoded text, and the fail-loud cases start at :218 (invalid bytes at the bound, test at :241), :251 (invalid bytes inside a complete page) and :260 (invalid bytes in the header, a gap or the trailer). On the real file the cut lands on an ASCII byte (the last byte of the pinned file, offset 99,999,999, is the space 0x20, measured on data/enwik8/enwik8), so the trim branch is exercised by the synthetic fixture, not by the real corpus.

## Guard teeth (mutation, each RED)

source-id reuse; revision mutated; tail-handling dropped; splitting on title instead of content.

## Acceptance (measured, in the PR body)

pysbd seconds per MB; whole-document rejection rate for oversize sentences (core/sentence/corpus.py:207-213); capture JSON bytes vs max_object_bytes; replay seconds for read_prepared_corpus; peak RSS of the in-memory capture, measured on the pod regime against a declared pod-memory budget before sealing; max_sentence_bytes agreed with L2.

Peak RSS is a separate measurement from the artifact size: `max_object_bytes` bounds the artifact, not peak memory, because `write_prepared_corpus` replays `prepare_documents` and encodes the full payload before the byte check (src/entropic_substrate/ladder/capture/capture.py:98, :117-130). The budget is declared before the run, and the number is measurement-gated with the other full-corpus capture numbers until FOR-1706.

The size check is an acceptance invariant, as shipped: the capture is serialized through `write_prepared_corpus` with the configured limit (scripts/prepare_enwik8_source.py:139, with `MAX_OBJECT_BYTES` 2 GiB at :61 and as the `--max-artifact-bytes` default at :250), the writer refuses overflow (ladder/capture/capture.py:129-130) so a passing run implies `capture_json_bytes <= max_object_bytes`, and the capture is then read back through `read_prepared_corpus` and compared to the prepared corpus (scripts/prepare_enwik8_source.py:142-145). The round trip is pinned at tests/test_enwik8_documents.py:383-398. The full-corpus `capture_json_bytes` number stays exhausted-pending until FOR-1706 merges.

## Out of scope

the full in-image capture (W4), any reader-member edit, the corpus memo (W2 L4b).
