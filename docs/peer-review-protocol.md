# Peer-review protocol — double-blind adversarial review of E-ladder results

Goal: replicate a double-blind review as closely as a single-operator project
can. Two blinding layers: (1) the reviewer session has NO project context —
fresh agent, no repository access, no narrative docs; (2) the ARMS ARE
RELABELED A/B in the packet — the reviewer cannot favor the "interesting"
init because they do not know which is which. The A/B mapping is sealed
(kept outside the packet) and disclosed only after the review is filed.

## The review packet (everything the reviewer gets, nothing else)

| File | Content |
|---|---|
| `abstract.md` | ≤100 neutral words: "paired comparison of two embedding initializations (A, B) for a byte-level language model; n pairs; primary metric nats/byte" — no project name, no init names, no hypothesis direction |
| `preregistration.md` | e1-preregistration.md with arm names REPLACED by A/B (direction restated as "A better than B") and repo paths stripped |
| `data/` | captured result JSONLs, filenames relabeled `A-seed<i>.jsonl` / `B-seed<i>.jsonl` |
| `reader.py` | self-contained copy of the telemetry admissibility reader (no package imports) |
| `manifest.txt` | md5 of every packet file + the preregistration COMMIT TIMESTAMP |

Assembly checklist (operator, before opening the reviewer session): relabel,
strip, hash, seal the mapping in a separate note outside the packet.

## The reviewer prompt (paste verbatim into a FRESH agent session)

```text
You are an anonymous, independent statistical reviewer. You have a review
packet for a paired two-arm machine-learning experiment. You have no other
context about the project or its authors, and you must not attempt to
identify or research them — no web searches about the work. Treat every
claim as unverified until you have recomputed it yourself.

Work in this order:
1. INTEGRITY — verify every md5 in manifest.txt. Any mismatch: report and
   stop.
2. ADHERENCE — enumerate the seeds present in data/; compare against the
   registered stage lists; verify any exclusions match the registered rule
   (and drop whole pairs); verify the analysis metric and estimator match
   the registration. Any deviation is a MAJOR finding.
3. RECOMPUTATION — parse data/ ONLY through reader.py. Recompute the paired
   differences, mean, sd, and the registered CI at the registered per-look
   alpha. Trust no number you did not compute.
4. OPTIONAL-STOPPING AUDIT — from record timestamps, verify no confirmatory
   run predates the preregistration commit timestamp in manifest.txt, and
   that stage 2 data (if present) exists only under the registered trigger.
5. ROBUSTNESS PROBES (report, do not decide) — sign test on the paired
   differences; leave-one-out CI sensitivity; describe the per-pair spread;
   flag anomalies (non-finite values, truncated series, irregular eval
   cadence, duplicate digests across seeds — which would indicate the seeds
   were not actually varied).
6. VERDICT — exactly one of SUPPORTED / NOT SUPPORTED / INVALID, quoting the
   registered criterion; then MAJOR and MINOR concern lists; then the
   specific additional evidence that would change your verdict.

Skepticism is the default posture: your task is to find the ways the claimed
conclusion could be wrong.
```

## After the review

The sealed A/B mapping is disclosed, the review is filed alongside the
analysis (both DVC- or git-tracked), and any MAJOR finding reopens the
analysis — the verdict is the reviewer's to give, not the operator's to
negotiate.
