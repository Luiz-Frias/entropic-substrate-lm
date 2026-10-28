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

## Guard teeth (mutation, each RED)

source-id reuse; revision mutated; tail-handling dropped; splitting on title instead of content.

## Acceptance (measured, in the PR body)

pysbd seconds per MB; whole-document rejection rate for oversize sentences (core/sentence/corpus.py:207-213); capture JSON bytes vs max_object_bytes; replay seconds for read_prepared_corpus; max_sentence_bytes agreed with L2.

## Out of scope

the full in-image capture (W4), any reader-member edit, the corpus memo (W2 L4b).
