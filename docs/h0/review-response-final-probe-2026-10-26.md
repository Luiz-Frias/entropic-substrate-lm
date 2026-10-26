# H0 final probe, blind review round 1: author response

Response to the independent, carrier-blind review of
`docs/h0/analysis-final-probe-2026-10-26.md`. The reviewer (model `claude-opus-5-5`,
network-isolated code-execution sandbox, one server-side round, 426 s, 2026-10-26T19:31Z) saw
only the packet: abstract, brief, the analysis with carriers relabelled, the four pods' metrics
and worker logs, 24 artifacts (generation and progress receipts, collections, worker results,
carrier-redacted profiles and requests, the two wave records, the in-image check report with
its metadata, regime and exit) and a standalone copy of the H lane reader. The reviewer's full
six-section file (27,026 characters, md5 `897b5d48861c7be593ccf47458fa975a`) is the review of
record, archived with the prompt, packet, manifest, run log and sealed mapping at
`~/.claude/projects/-Users-luizfrias-CursorAI-entropic-substrate-lm/refactor-survey-2026-09-17/h-sprint/h0-review/final-probe-round1/`
(operator machine, outside git). Every reviewer statement below is REVIEWER-REPORTED; every
number marked MEASURED was recomputed by this session from the DVC captures.

Measured cost of the review (API usage, `response.usage` of the single round): 1,911,284
input tokens and 38,245 output tokens on `claude-opus-5-5`. At the first-party rates read from
the Claude API reference on 2026-10-26 ($4 and $20 per million tokens) that is $7.65 plus
$0.76, about $8.41; sandbox container time is not reported by the usage object and is not
counted. The review key's account balance was not readable from this session.

Dispositions: `corrected` means the analysis file was fixed in place with a dated retraction;
`explained` means no change to the analysis beyond a pointer; `packet` means the concern is
about what the packet carried, resolved by evidence in the captures that the reviewer did not
see; `open` means the item stays unresolved.

## Reviewer's verdict

`tracer_partial`. Exits PROVEN by the packet alone: stochastic rollout, recovery and resume
(functionally), autocast observability (as a recorded configuration), capture admissibility,
lineage, final leg (worker side), numerics. UNPROVEN: the test-only gate, for the two blocking
reasons below. The reviewer reports that every number recomputed from `data/` matches the
analysis, including the resume seam on 54 non-timing leaves over 1000 rows, the three
evaluations and the two carrier-A receipts, and that `invalid_utf8` is expected content, not a
tracer defect, after replaying every continuation's outcome from its observations.

## Major concerns

| # | concern | author response | evidence | disposition |
| --- | --- | --- | --- | --- |
| M1 (BLOCKING) | The in-image check report's sha256 (`46781617…`) does not equal the `report_sha256` its metadata records (`2e104321…`), so the K1 chain verdict cannot be tied to the run it claims to come from | The packet copy of `report.jsonl` was carrier-redacted for blinding: K2's `facts.carriers` lists the carrier names and K7's arm label is `wave1/sphere`, and the blinding pass rewrote those strings. The captured file is unaltered: `sha256sum` of `results/h0-final-probe-checks-1755b0/report.jsonl` is `2e10432125a1ff3a7cd52941467e2dac5e50eaefb59fbf989e79e7712db0289a`, equal to `metadata.json` `report_sha256`. The analysis now states this (section K). The packet should have carried the pre-redaction hash beside the redacted copy; it did not, which is a packaging defect of this round, not an evidence defect | MEASURED (`sha256sum` over the DVC-captured file); the redaction is in the packet build script, archived | packet; corrected (hash check added to section K) |
| M2 (BLOCKING) | Controller and watchdog records (`terminal_collected`, `termination_verified`, `error_types`) are absent from the packet, so K2's inputs and the termination claims cannot be checked | The records are in the two host-side captures (`<arm>/host/control/controller-result.json`, `watchdog-result.json`, `controller-pod.json`, `controller-stop.json`) and were read by the analysis (Q3 clocks, labelled RECORD) and consumed by the local K2 re-executions (plans m1 and m2, PASS over {sphere, W1 magnitude} and {sphere, W2 B}). They were left out of the packet by the author's choice of scope, not because they are missing. The controller payloads for the three completed arms are archived with the plans | MEASURED (K2 PASS twice, `scripts/h0_final_probe_checks.py` on this Mac); records read verbatim | packet |
| M3 (NOT BLOCKING) | An unlogged 378 s gap on the resumed pod between `publish.end … started.json` and `phase.start prediction`, plus the 99.9 s first seal, with no resume or restore event | Agreed and re-measured: 378.134 s on the resumed pod against 34.288 / 34.441 / 34.565 s on the other three, so the restore costs about 343.7 s that no row itemises. Added to the analysis as a Q3 row and a finding for the gate author | S-log, M-log, A-log, B-log L17 to L18 timestamps (MEASURED) | corrected (row added) |

## Minor concerns

| # | concern | author response | evidence | disposition |
| --- | --- | --- | --- | --- |
| m1 | "Bit-for-bit on every logged metric" overstates: `cuda_peak_bytes` differs on 136 rows and the timing fields on every row | Agreed. Re-measured over every flattened field of every training row (31 dotted keys): 28 identical on all 1000 rows, 3 differ (`wall_seconds`, `cuda_seconds` on every row; `cuda_peak_bytes` on stitched rows 501 to 636, the resumed pod's first 136 steps, a fresh process's lower allocator high-water mark). The reviewer's count of 54 differs from 28 only in counting list elements as separate leaves | flattened row comparison, MEASURED 2026-10-26 | corrected (row added to Q4, retraction in the seam conclusion and the verdict proposal) |
| m2 | The two carrier-A receipts differ in more than the snapshot fields | Agreed: `training.{path, sha256}`, `wall_seconds`, `cuda_seconds` and the per-continuation snapshots also differ; `log_scores`, `sentences`, `outcome`, `seed`, `sampling_law` and `summary` are identical | flattened receipt diff, MEASURED | corrected (Q4 row restated with a retraction) |
| m3 | The elapsed × cost row is an estimate mislabelled DERIVED_FROM_MEASURED | Agreed. It is watchdog elapsed time times the observed hourly rate, not a bill; relabelled with a retraction | Q3 | corrected |
| m4 | Several MEASURED rows rest on files the packet did not carry (HEpP7g files, `terminal.json`, DVC pointers, local plans m1 to m4) | The rows are measured over the DVC captures, which carry all of them; the packet carried a subset. The local plans, their reports and stderr are archived beside the review. For the reviewer these rows are author-reported, as the review says | archive listing | packet; explained |
| m5 | `admission.end` timestamps do not bracket `seconds` on four shared-memo callers | Agreed and re-measured on all three completed pods: the bracket equals the stage seconds, so the event is emitted at stage end while `seconds` is the admission's own clock. Added to Q3 as a finding for the gate author | admission brackets, MEASURED | corrected (paragraph added) |
| m6 | `summary.outcome` reports only the last continuation | Already in the analysis (Q5, second finding) | Q5 | explained |
| m7 | The predicate does not reject non-finite values | Agreed; Q6's finiteness scan is the separate check. Noted in the Q1 predicate row | Q1 | corrected (note added) |
| m8 | "Six metrics files" stated, four in the packet | True of the captures (the two HEpP7g files are in the same MLflow capture and were run through the predicate and plan m4); the packet carried four by design | section K, plan m4 | explained |

## Evidence-class audit items

| item | response | disposition |
| --- | --- | --- |
| Request-file to `request_sha256` binding not checkable (redacted, re-serialized) | Correct; the redaction breaks the hash by construction. In the captures the binding holds through `wave.json`, `collection.json`, `worker/start.json` and the `run_start` row, all carrying the same digest per arm | explained |
| K1 chain presented as RECORD-only | Yes, and the review agrees it is candid; the hash anchor is now verified in section K (M1) | corrected |
| "Autocast observability CLOSED" proves a recorded configuration, not an effective one | Agreed as stated. The HEpP7g exit asked for "an autocast-enabled field in the runtime or step record"; that is what closed. Whether bf16 arithmetic took effect on the live step is not shown by any captured row and is not claimed | explained; the exit as written is closed, the stronger statement is not made |
| Code-read statements inside MEASURED rows (`rng_digest` dependence) | The row's measured content is the digest equality; the dependence clause is code-read and says so in the evidence cell | explained |

## Sealed carrier mapping, disclosed after the verdict was written

The packet named the carriers A and B. Mapping sealed in the scratchpad note
`sealed-mapping-final-probe.md` before the review ran, archived beside the review:
**Carrier A = magnitude** (`h0-magnitude-b64-1k-v4-fisher-tiny`, identity `5ece5d1b…`, pods
`xm1c6coypgd2by` complete, `i11npiybn4nwnt` stopped, `m8ootdwxmkuveg` resumed); **Carrier B =
sphere** (`h0-sphere-b64-1k-v4-fisher-tiny`, identity `3390f577…`, pod `g8087m4ah3ops2`). The
letters are the reverse of the HEpP7g round. The reviewer's packet also carried the analysis's
own pod labels "W2 A" and "W2 B" for the stopped and resumed pods, which the brief and prompt
flagged as pod names, not carriers.

## Open after this response

| item | why it stays open |
| --- | --- |
| Test-only gate, as judged by an independent reviewer | Both blocking concerns are resolved by evidence in the captures (M1, M2), but by the author, not the reviewer. A second blind round on a packet that carries the host control records and the pre-redaction report hash is the evidence that would let the reviewer close it; round 1 measured about $8.41 of API spend, so round 2 is an operator spend decision |
| Resume cost observability | about 444 s of restore-related time with no itemised row (M3); not an exit, a finding for the gate author |
| Declared measure of the future density per carrier | a docstring question carried from HEpP7g; nothing in this wave bears on it |
