# H0 verdict: wave h0-plumbing-1k-v3.HEpP7g

## Verdict

`tracer_partial`. The author analysis and the independent blind review agree. This is a
plumbing gate with a test-only gate: it carries no scientific claim, and no H1 eligibility
beyond "the machinery exists". It authorizes no spend.

Vocabulary used: `tracer_complete` / `tracer_partial` / `numerical_defect` / `missing_evidence`
(H0 row of `docs/hierarchical-latent-predictor.plan.md`). Not `numerical_defect`: no nonfinite,
nonmonotonic, failed or stranded evidence. Not `missing_evidence`: the capture is complete and
admissible for what it was built to carry. Not `tracer_complete`: the open exits below.

## Exits proven

| exit | basis | evidence class |
| --- | --- | --- |
| Capture admissibility under the H lane predicate | both arms: one `run_start`, one `run_end`, committed step rows; worker logs pass the JSON-object-lines check | analysis Q1 counts MEASURED; predicate executed by the reviewer through a standalone reader, REVIEWER-REPORTED |
| Lineage | one code commit `9d353fcc`, one image digest, identical ordered per-row source fields over 1000 rows; profiles identical except `law.mode`, `spine.carrier`, `name` | analysis Q2 MEASURED; profile diff re-run from the review packet, MEASURED |
| Carrier assignment (`988d8388` sphere, `a163161f` magnitude) | three links: published profile size equals the logged size, generation receipt `run_identity` equals the metrics file, sphere sentence states have norm 1.000000. The profile link rests on file size only | size and norm links REVIEWER-REPORTED; the identity and carrier links re-measured by the conductor 2026-10-22 from the archived packet: receipt `run_identity` prefixes `ce3740e5` / `ed98fc2e` equal the metrics file profile-identity prefixes, and the fetched profiles carry `spine.carrier` and `law.mode` `sphere` / `magnitude` with names `h0-sphere-b64-1k-v3-fisher-tiny` / `h0-magnitude-b64-1k-v3-fisher-tiny`, MEASURED |
| Final leg | all eight stages on both arms, seven admission pairs, zero retries, zero stop markers, JSONL sink settled with zero stranded | analysis Q3 and Q6 MEASURED |
| Numerics | 0 nonfinite values across 207,382 numeric leaves (102,534 sphere, 104,848 magnitude; metrics plus worker logs); steps and timestamps monotone; `max_residual_ratio` below 1 on all 1500 prediction rows | MEASURED (analysis Q6, recount by the docs lane) |
| Generation produced finished continuations | both receipts `generation_complete`, `outcome: sentence_budget`, `pending` empty, nothing truncated (reviewer) | receipts re-read by the docs lane from the packet, MEASURED; "token 256 is the boundary" is a reviewer inference, not a schema fact |

## Exits open

| exit | why it is open | next evidence that would close it |
| --- | --- | --- |
| Stochastic rollout | one continuation per prompt; no sampler, temperature or seed recorded; a deterministic decoder would give the same receipts | a receipt carrying the sampling law and seed, and at least two continuations per prompt that differ |
| Recovery and resume | no resume event in either log (analysis Q5) | a remote run that restores from a sealed checkpoint and reaches a completion receipt |
| Test-only gate against this capture | not exercised | the gate run against `results/h0-plumbing-1k-v3-HEpP7g/` |
| Autocast observability | profiles declare `bf16`, runtime and mechanism rows report float32; whether autocast was enabled on the live step is not recorded anywhere | an autocast-enabled field in the runtime or step record (ticketed: FOR-1672) |
| Item 7 schema questions | sign and null-gap are explained from code (`objective.py`, `h_aggregate.py:29-31` and `:261-282`); the declared measure of the future density per carrier is not stated in either file | the density's measure written into a schema or docstring |

## Evidence chain

| link | value | evidence class |
| --- | --- | --- |
| Capture pointer | `results/h0-plumbing-1k-v3-HEpP7g.dvc`, md5 `2dc65e2bdf918d43858c6b2d92a63c58.dir`, 4 files, 5,597,931 bytes, commit `1d3d67d1e481f871f4f58917b6dda0cadf93f05d` | MEASURED (analysis Inputs) |
| Wave | `h0-plumbing-1k-v3.HEpP7g`, pods `euikqmf95pnrga` (sphere) and `211tvsc7277d22` (magnitude) | MEASURED (run_start rows) |
| Image pin at the time of the run | `sha256:8181443b6c410c45a0e39eb472fe77ace8793c1dbe84cc38ce86de4b1af801ea`, code commit `9d353fcc2d6f82e19927ad8fec77eb78a75b4df0` | MEASURED (run_start L1 both arms) |
| Analysis | `docs/h0/analysis-hepp7g-2026-10-20.md`, first at main `50a4951e` (its pre-review text), corrected in the commit that adds this file | file in git |
| Review | blind round 1, reviewer model `claude-opus-5-5`, network-isolated sandbox, 2026-10-22T16:12Z; verdict `tracer_partial`; the reviewer's full six-section file was lost by the tooling, so the chat summary is the review of record | REVIEWER-REPORTED |
| Review archive | `~/.claude/projects/-Users-luizfrias-CursorAI-entropic-substrate-lm/refactor-survey-2026-09-17/h-sprint/h0-review/review-cloud-round1-summary.md` plus `packet-round1/` beside it (operator machine, outside git) | path, not in git |
| Author response | `docs/h0/review-response-hepp7g-2026-10-22.md` | file in git |

## Scope statement

This is a plumbing gate with no scientific claim. No decoder-quality, predictive or geometric
statement follows from it, and the generated text in the receipts is not evidence of model
quality. H1 eligibility beyond "the machinery exists" is not implied. The open exits above
carry forward unchanged.
