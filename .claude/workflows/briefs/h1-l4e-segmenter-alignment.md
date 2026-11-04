# FOR-1706 [h1-l4e] Align pysbd sentence spans sequentially from the cursor in segment_document (keeps every enwik8 page; segmentation identity bump, golden replay; DG-A re-pin row 5)

Fifth and last serial W2 slice, after L4d; arch mode (owns the segmentation identity door); Opus xhigh. Defect fix in a reader member under DH-35 (operator 2026-10-28: the fix, explicitly NOT an exclusion). It moves the segmentation identity and therefore the source reader identity, so the W4 enwik8 capture stays after it. Every anchor below was re-measured at main e3d1524b (v0.10.0) on 2026-10-31. Correction to the issue text: `segment_document` and the overlap guard live in core/sentence/segmentation.py, not corpus.py.

## Measured (L1 design gate 2026-10-28; anchors 2026-10-31)

pysbd 0.3.4 (pyproject.toml:17, uv.lock:3457, `SegmentationPolicy.engine_version` at segmentation.py:21) `sentences_with_char_spans` (site-packages pysbd/segmenter.py:66-69) re-scans the text from offset 0 with `re.finditer` and accepts the first match whose END passes the previous span's end even when it STARTS before it, so short repeated-punctuation sentences (`. .`, `...`, `???`) get overlapping spans. `segment_document` (segmentation.py:84) guards `cursor <= start < end <= len(document.text)` at :118 and raises `RuntimeError("segmenter produced invalid, altered, or overlapping spans")` at :121; corpus.py calls it at :208 without catching. In the first 10,000,000 bytes of data/enwik8, 4 of 1,325 complete pages raised at the L1 gate (indices 261, 569, 766, 1280); the `RejectionReason` Literal (corpus.py:17) is closed to duplicate_content, oversize_sentence and source_budget, so no typed exclusion slot exists. The W1 adapter never catches a segmenter defect (enwik8_documents.py:8-9); scripts/prepare_enwik8_source.py raises `SegmenterDefect` at :129-132 naming source id, title and span, exits 1 from `main` (:245) with the output dir empty, and `--measure-segmenter` (:212) lists defects per page and exits 0. The "page 261" figure is the L1 gate's measurement and is re-measured by this slice with `--measure-segmenter` on the same prefix before and after the fix.

## Ruling (DH-35)

Keep the affected pages: in `segment_document`, align pysbd's sentence TEXTS sequentially from the running cursor (each sentence located at or after the previous span's end), never trusting pysbd's `char_span` offsets. No `segmenter_overlap` rejection reason; no page is dropped.

## Touch-set

- member (DG-A): src/entropic_substrate/core/sentence/segmentation.py (sequential alignment; the fail-loud invariant at :118-121 stays as the guard). segmentation.py and corpus.py are members of BOTH readers (checkpoint.py:97 and :84; capture.py:32-33), so this edit moves all nine digest rows: five checkpoint rows (tests/test_identity_epoch.py:904-931) and four source rows (:933-951), recomputed by the conductor with `tools/regime_rows.py`; the source reader identity hashes python, pysbd and the seven member files (capture.py:40-52, `h-source-reader-v3` at :51).
- tests: the segmentation tests (RED-first fixtures with `. .`, `...`, `???`; the four real pages reduced to their failing paragraphs; the guard keeps its teeth on a text that does not occur at or after the cursor); golden replay over every existing fixture document: the synthetic corpus is `tests/test_sentence_sampling.py::corpus()` (:21-30, 30 documents, revision `tiny-v1`, `SegmentationPolicy(64)`), imported by test_sentence_execution.py:23 and test_sentence_seals.py:10; the H0 result fixtures carry telemetry only, no span data, so they are NOT golden sources here; tests/test_enwik8_documents.py gains the real-page fixtures; tests/test_identity_epoch.py and docs/identity/read-old-table.md (conductor re-pin row 5).
- NOT touched: any pysbd version move (pinned in three places: pyproject.toml:17, segmentation.py:21, capture.py:51; a bump is its own identity event); the L1 adapter and script (W1, merged #417; their three capture-dependent measurements leave exhausted-pending once this slice merges); the full enwik8 capture (W4).

## RED-first tests

A fixture document with `. .`, `...` and `???` sentences segments without raising and its spans are monotone and non-overlapping (RED at the parent: RuntimeError at segmentation.py:121); the four real pages' failing paragraphs segment; every existing fixture document produces the same spans as before (golden); a pysbd result whose text does not occur at or after the cursor still raises.

## Guard teeth (mutation, each RED)

search from offset 0 instead of the cursor; accept a span that starts before the cursor; drop the text-identity check; skip the identity bump.

## Handoff

DG-A as in L4a: the lane stages the patch, the proposed message and the evidence (RED/GREEN, mutations, the before/after `--measure-segmenter` counts on the 10 MB prefix, local CodeRabbit round); the conductor lands patch plus the nine-row re-pin plus the read-old-table row as one commit.

## Out of scope

the pysbd upgrade; the L1 adapter; the W4 capture; any exclusion path.
