# h1-p0-design-identity (W1 prelude, FOR-1685)

Authoritative brief for the W1 prelude phase. Anchors MEASURED at main `453302199feee3c026ff68f0cf0645de7dc44565`
(tag v0.9.1) on 2026-10-28. Owner: the conductor (sprint-preflight Q1, ledger DH-23). Engine mode: arch (this
phase owns doors). Parent: `h1-sprint-manifest`. Linear: FOR-1685 (the W1 wave parent hosts this prelude).

## Why this phase exists

L3 (FOR-1691) maps the exclusion reason `independent_warmup` and L2 (FOR-1690) and L3 both key the
statistical design by a "construction digest" that the plan rules as "profile identity with `name` and
`seed` removed" (DH-14), but no issue owned either definition and L4a (FOR-1692) owns the defect type that
produces `independent_warmup` one wave later. This prelude lands the three shared definitions first, as
plain functions beside `_shared_profile`, so the three W1 lanes consume fixtures instead of inventing
their own and nothing in W1 waits on W2.

## Touch-set (non-member; `ladder/register/h_preservation.py` is absent from `ladder/run/checkpoint.py:72-126`)

- `src/entropic_substrate/ladder/register/h_preservation.py` (105 lines today): add three public names.
  Nothing in `H1PreservationBlock` (:29-105) changes, and the `h1-preservation-block-v1` manifest bytes
  (:85-91) and `identity` (:93-95) stay byte-identical (pinned by the existing tests).
- `tests/test_h_preservation.py` (reuses the `profiles()` fixture at :15-35, which builds four H1 profiles
  from `tests.test_sentence_profile.profile()` at `tests/test_sentence_profile.py:19`).
- NOT touched: `core/sentence/profile.py`, `ladder/analyze/*`, `ladder/run/*`, any reader member, the
  skeleton `docs/h1/preregistration-skeleton.md`, `docs/architecture.md`, `PROGRESS.md`.

## The three definitions

1. `h1_construction_digest(profile: HProfile) -> str`
   The H construction identity for the statistical design (`h-ladder-statistical-design-v1` family and
   construction keys). Build it from `profile.manifest` (`core/sentence/profile.py:259-262`, the
   `h-run-profile-v3` dict) with the top-level keys `name`, `seed` and `source_seed` removed (the ruling
   below), canonically serialized (`json.dumps(..., sort_keys=True, separators=(",", ":"))`) and hashed
   through `content_identity` (`core/sentence/records.py:8-10`), the same shape `HProfile.identity` uses
   (`profile.py:264-266`).
   RULING (as shipped, the DH-14 reading): the digest removes exactly the top-level keys `name`, `seed`
   and `source_seed` and nothing nested (`h1_construction_digest`, `h_preservation.py:30-53`, docstring
   `:31-47`). `source_seed` is a sibling top-level key (the quote profile carries `seed: 42,
   source_seed: 42`; `tests/test_sentence_profile.py:25-26`) and the pilot sets it per block (one source
   stream per block, FOR-1695; L2 sets `source_seed = seed`, `ladder/register/rungs/h1_pilot.py:184`), so
   leaving it in would read the eight blocks as eight constructions and `plan_seed_count` would see n=1
   per construction. Pinned by `tests/test_h_preservation.py:191-200` (a changed name, seed or
   `source_seed` leaves the digest equal; golden value at `:169-174`) and by
   `tests/test_h1_pilot_manifests.py:277-281` (the digest is invariant across the eight blocks and the
   four arms differ). `sinks.h_config_hash` keeps `source_seed` by design
   (`ladder/capture/reader/sinks.py:226-245`, conductor ruling 2026-10-14) and is untouched.
   This function is DISTINCT from `_shared_profile` (:20-25, which removes `name`, `law.mode` and
   `spine.carrier` and KEEPS `seed`: it is the pair-block predicate) and from L4a's pair rule (name and
   carrier removed, same law.mode): three predicates, three functions (preflight Q2).
2. `H1ExclusionReason` (`StrEnum`, closed): `NONFINITE_ENDPOINT`, `MISSING_ENDPOINT`, `READER_REFUSAL`,
   `INADMISSIBLE_ESS`, `INDEPENDENT_WARMUP`. Precedent: `D1PairExclusionReason` at
   `ladder/capture/custody/d1_capture_roster.py:167`. These are the whole-block exclusion reasons FOR-1700
   lists; a block drops whole on any of them and the reason is reported, never silent. `INDEPENDENT_WARMUP`
   is the name L4a's `CodecHandoffDefect` will map to in W2; this enum owns the vocabulary, L4a owns the
   detector. No other string is admitted (a lookup of an unknown reason raises).
3. `h1_exposure_bytes(rows: Iterable[Mapping[str, object]]) -> int`
   The exposure rule L3's `StatisticalEvaluation` needs (cumulative canonical bytes): the sum of
   `source_bytes` over committed prediction-phase journal step rows. Journal step rows carry `phase`
   (`ladder/run/execution.py:203`), `source_bytes` and `status="committed"` (`execution.py:261-271`).
   Rule (ledger DH-25 default, open for the operator before L8 seals): prediction-phase rows only, for
   every cell, so the donor and the recipient of a codec hand-off are counted the same way; warm-up rows
   and uncommitted rows contribute nothing; a row with `phase == "prediction"` and no `source_bytes` raises
   (measurement path, fail loud).
   Order as shipped and ruled fail-loud (`h1_exposure_bytes`, `h_preservation.py:74-100`): every
   prediction-phase row of any status must carry a nonnegative `int` `source_bytes` or the helper raises
   (`:95-97`), and only then are committed rows summed (the status test comes last, `:98-99`); a
   non-committed prediction row with bytes contributes nothing and one without bytes raises
   (`tests/test_h_preservation.py:313-316`, `:346-349`). An unrecognized `phase` or `status` raises too
   (`:91-92`). This is deliberately not committed-first filtering: persisted journals hold only committed
   rows (`ladder/run/journal.py:127-135`) and the training loop re-raises on failure
   (`ladder/run/execution.py:274-285`), so a failure-shaped row in the input means attempt records were
   fed in, which is the intended refusal.

## RED-first tests (write first, watch them fail at the parent tip, then implement)

- digest equal for two profiles that differ only in `name`, `seed` and `source_seed`; digest differs
  when `spine.carrier`, `law.mode`, any codec/spine width, `source_digest` or the schedule differs;
  one golden value pinned for the `profiles()[0]` fixture with a comment naming the inputs.
- `H1ExclusionReason` has exactly the five members above; `H1ExclusionReason("anything_else")` raises.
- `h1_exposure_bytes`: warm-up rows excluded, uncommitted rows excluded, prediction rows summed, a
  prediction row without `source_bytes` raises, an empty iterable is 0.
- The existing `h1-preservation-block-v1` identity and manifest pins stay green (no byte moves).

## Guard teeth (mutation, each must go RED, quoted in the handoff)

- drop the `seed` deletion from the digest (two blocks then read as different constructions);
- include warm-up rows in the exposure;
- add a sixth exclusion reason without the enum (a bare string accepted);
- make the digest read `_shared_profile` instead of the manifest (law.mode and carrier vanish).

## Acceptance

`just check` exit 0 and the exit-gate runner green under the suite lock; the diff confined to the two files
above; a golden digest value, the five reason names and the exposure rule stated in the handoff; the design
gate's `source_seed` ruling recorded in the docstring and echoed back to the conductor.

## Out of scope

The adapter (L3), the generator (L2), any consumer wiring, any reader-member edit, the defect detector
(L4a), the registration values (L8).

## Protocol

Suite lock for every gate and hook (`/tmp/esl-suite.lock`, mkdir + holder, never remove a lock you do not
hold; wait for 1-minute load under 8); `-p no:xdist` for single-file TDD runs; never a bare `gt submit`;
publishMode defer (the W1 zip publishes once through `just ship-it`); sha256 hex fixtures in tests take an
inline `# pragma: allowlist secret`; `.secrets.baseline` is frozen.
