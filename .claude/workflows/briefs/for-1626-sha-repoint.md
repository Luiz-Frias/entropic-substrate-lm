# for-1626-sha-repoint: repoint the three retained BOUNDARY SHA sites to `require_lowercase_sha256`

Authoritative brief for one `td-phase-lifecycle.cascade` run, written 2026-10-08 by the conductor from operator ruling DN-1626-REPOINT (2026-10-08: "repoint > keep"). Plan: `docs/plans/p14-consolidation.plan.md` DECISION-NEEDED register, row DN-1626-REPOINT. Linear: FOR-1626 (reopened to Todo; the ruling and the three sites are recorded on the ticket). Premises were measured at the merged p14 trunk `da4d35c`; this branch stacks on `mythwave/for-1632-parse-float-seam` (the published head of the FOR-1632 draft), whose touch set is disjoint from this one.

## Branch

| field                | value                                                                                                                                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch               | `mythwave/for-1626-sha-repoint`                                                                                                                                                  |
| parent               | `mythwave/for-1632-parse-float-seam` (the published FOR-1632 draft head, recorded in the launch args as `startTip`); the operator merges the whole stack in one sweep at the end |
| cascade `phase.mode` | `arch`, always explicit (caller-visible exception types change at two sites)                                                                                                     |
| wave                 | serial, main checkout, no lanes                                                                                                                                                  |
| `gates.liveRequired` | `false` (no MLflow adapter in the touch set)                                                                                                                                     |
| gates                | `{"offline":"just check","exit":"just pc-ci","liveRequired":false}`                                                                                                              |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                                                                                           |
| publish              | the cascade's publish stage: `gt submit --draft --no-interactive` once, stacked on the parent                                                                                    |

The branch EXISTS at launch: the conductor created it with `git checkout -b mythwave/for-1626-sha-repoint a6cac92` (the published FOR-1632 head, passed explicitly as the start point so the branch base is the declared `startTip`, not whatever HEAD happened to be: CodeRabbit finding on PR #334) and committed this brief correction as its first commit; `gt track --parent mythwave/for-1632-parse-float-seam` records the parent. Do not recreate it; verify `git merge-base --is-ancestor a6cac92 HEAD` and implement on top of the `startTip` in your args.

## Contract surface owned

| contract                                                                                                                                                  | kind                                                                   | frozen_after |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------ |
| Three BOUNDARY digest sites refuse anything but a lowercase SHA-256 through `kernel.codec.strict_json.require_lowercase_sha256`, each with its own `fail` | boundary contract (caller-visible exception type/message at two sites) | true         |
| `D1ObjectMailboxTransportError` gains the code `("start", "request_noncanonical")`, mirroring `("identity", "attempt_noncanonical")`                      | error vocabulary extension                                             | true         |

## Contracts consumed

- `kernel.codec.strict_json.require_lowercase_sha256(value, *, fail) -> str` (door, frozen after the rail; `fail` receives `"value must be 64 lowercase hexadecimal characters"`).
- `kernel.identity.d1_primitives.require_sha256_hex_any_case` stays defined (its own test `tests/test_sha256_caller_audit.py::test_any_case_accepts_and_rejects` and the in-process callers keep it); this phase only removes the three boundary callers.
- `docs/plans/measurements/p14-sha-callers.txt` is parsed by `tests/test_sha256_caller_audit.py::test_caller_audit_is_committed_and_covers_the_live_tree` (header `rows` / `boundary` / `in-process` / `files` must equal the row counts; every live call site must be audited).

## Sacred invariants

| invariant                                     | verbatim                                                                                                                                                            | tripwire                                                                                                                                                                                                                                                                                               | guard-teeth recipe                              |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| House rule 5                                  | "TDD is non-negotiable, and guard-teeth prove both arms."                                                                                                           | every restated or new pin is shown RED against the parent form before the repoint                                                                                                                                                                                                                      | the mutations below, both arms quoted per site  |
| Boundary lowercase contract (FOR-1615 ruling) | "every untrusted-boundary site requires lowercase. Never point boundary sites at the looser kernel function."                                                       | `tests/test_strict_json_adoption_run.py::test_every_boundary_sha_site_refuses_a_mixed_case_digest_with_its_own_error` green with three new `_Site` rows                                                                                                                                                | mutation per site                               |
| Exception surface                             | changes only where the remedy text states: the two transport sites move from `ValueError` to `D1ObjectMailboxTransportError`; the other two sites keep `ValueError` | every existing pin passes unmodified except the ones this brief names                                                                                                                                                                                                                                  | reviewer diffs the test edits against this list |
| Reader identity (R-PIN)                       | the 51 reader-tuple files are never edited outside the identity rail                                                                                                | `git diff --name-only <startTip>...HEAD` has an empty intersection with the 51 files listed in `.claude/workflows/briefs/p14-t-verdict-vocab.md` under *Identity members* (`kernel/identity/d1_primitives.py` is NOT a member; `kernel/artifacts.py` and `kernel/lifecycle.py` are and stay untouched) | proven at p0                                    |
| Registered identity constant                  | `default_d1_encoder_initialization_identity()` still constructs and `d1_config_profile._build_encoder` still admits the registered profile                          | `tests/test_d1_initialization.py`, `tests/test_d1_primitives.py`, `tests/test_d1_family.py` green                                                                                                                                                                                                      | mutation M2                                     |
| Layer contract                                | `kernel < core < ladder < historical < composition`                                                                                                                 | `lint-imports` green                                                                                                                                                                                                                                                                                   | proven at p0 and p13                            |

## Identity members

None. The three source files and the test files are outside the 51 reader-tuple files (measured). Editing a member is a HALT (DECISION), never a hand-off.

## Scope

One commit per site is admissible; one commit for all three is also admissible. Sites, with the ruled shape:

1. `src/entropic_substrate/ladder/register/d1_registered_design.py:88-90` (`D1PreregistrationArtifact.__post_init__`, field `blob_sha256`, audit row 71, ruling W1-16): replace the `require_sha256_hex_any_case(...)` call AND the explicit `.lower()` branch with `require_lowercase_sha256(self.blob_sha256, fail=<raise ValueError>)`; the message keeps the substring `lowercase` (pinned at `tests/test_d1_registered_design.py:111`, which stays as is) and names the field; add one RED-first row to that parametrize for the 63-character arm and one for a non-hex 64-character arm. Drop the now-unused `require_sha256_hex_any_case` import.
2. `src/entropic_substrate/kernel/identity/d1_primitives.py:204` (`EncoderInitializationIdentity.__post_init__`, `construction_digest`, audit row 39, ruling W3-21): repoint to `require_lowercase_sha256(..., fail=<raise ValueError>)`; the message keeps the substring `SHA-256` (pinned at `tests/test_d1_initialization.py:131`) and gains `lowercase`; add a RED-first pin that a mixed-case 64-character digest is refused (behaviour change: the any-case door admitted it). The in-process callers of `require_sha256_hex_any_case` elsewhere in `d1_primitives.py` are NOT touched.
3. `src/entropic_substrate/ladder/run/transport/d1_trainer_object_transport.py:75` (`_attempt`, audit row 93) and `:256` (`start_trainer`, `request_sha256`, audit row 94; ruling W4-26): `_attempt` repoints with a `fail` that raises `_error("identity", "attempt_noncanonical")` (the malformed arm moves from `ValueError` to `D1ObjectMailboxTransportError`, same code as the mixed-case arm; the `.lower()` branch goes); `start_trainer` repoints with a `fail` that raises `_error("start", "request_noncanonical")` (new code), BEFORE the session lookup. `tests/test_strict_json_adoption_run.py::test_object_transport_request_digest_retains_its_parent_exception_type` and its W4-26(b) comment are restated: a mixed-case digest now raises `("start", "request_noncanonical")`, a non-hex digest the same, the genuine lowercase digest still starts the trainer; add `object_transport_request` to `_SITES` with `D1ObjectMailboxTransportError` and `request_noncanonical`.
4. `docs/plans/measurements/p14-sha-callers.txt`: remove rows 39, 71, 93 and 94 (they are no longer call sites), restate the header counts (`rows`, `boundary`, `files`: the register and transport files leave the audited set; `d1_primitives.py` stays because of its in-process calls), and replace the 2026-10-05/07 RETAINED comment block with a dated note that FOR-1626 repointed the three sites under DN-1626-REPOINT. `tests/test_sha256_caller_audit.py` must stay green unmodified.

## ACCEPTANCE

- `grep -rn require_sha256_hex_any_case src/entropic_substrate/ladder/register/d1_registered_design.py src/entropic_substrate/ladder/run/transport/d1_trainer_object_transport.py` is empty; in `d1_primitives.py` the only remaining references are the definition and in-process calls (quote them).
- RED-first pins per site as listed in *Scope*, each shown RED against the parent form.
- `tests/test_strict_json_adoption_run.py`: `_SITES` carries three new rows (`registered_design_blob_sha256`, `encoder_initialization_digest`, `object_transport_request`) and the restated transport test; green.
- `tests/test_sha256_caller_audit.py` green with the audit file restated; `tests/test_d1_registered_design.py`, `tests/test_d1_initialization.py`, `tests/test_d1_primitives.py`, `tests/test_d1_trainer_object_transport.py` green.
- Guard teeth by mutation, both arms quoted per site (below).
- remap_check at the tip over the re-derived mapping: inversions <= 1, package cycles <= 4; mechanical counts at or under `da4d35c` (forbidden-hook files 2 after FOR-1632, `def _mapping(` 23, `separators=(",", ":")` 64, lowercase-SHA files 29, `type: ignore[arg-type]` 9, files > 1000 lines 2).
- `git diff --name-only <startTip>...HEAD` is a subset of *Allowed files*; the identity tripwire is empty.
- `just check` green at the tip under the host lock; exit gate certified by the runner JSON at HEAD; draft PR stacked on `mythwave/for-1632-parse-float-seam` with a filled body (Linear ref FOR-1626, the ruling, per-site arms, the audit header before/after, remap stdout, counts).

## Non-goals

- Does NOT retire or rename `require_sha256_hex_any_case`; in-process callers stay.
- Does NOT touch `d1_registered_design.py` rows other than the blob_sha256 site, nor any other `_SITES` entry.
- Does NOT change the transport's existing codes or stages beyond the one addition.
- Does NOT launch pods, spend, build containers or start an MLflow server.

## Allowed files

- `src/entropic_substrate/ladder/register/d1_registered_design.py`
- `src/entropic_substrate/kernel/identity/d1_primitives.py`
- `src/entropic_substrate/ladder/run/transport/d1_trainer_object_transport.py`
- `tests/test_d1_registered_design.py`
- `tests/test_d1_initialization.py`
- `tests/test_d1_primitives.py`
- `tests/test_d1_trainer_object_transport.py`
- `tests/test_strict_json_adoption_run.py`
- `docs/plans/measurements/p14-sha-callers.txt`
- `.secrets.baseline` (mechanical companion, digest literals only, stated in the PR body)

Never edit: `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/regime_rows.py`, `tests/test_sha256_caller_audit.py`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `tools/mapping.json`, the manifest, the plan, briefs, `pyproject.toml`, `uv.lock`.

## Already ruled

- DN-1626-REPOINT (operator, 2026-10-08): repoint all three sites with their tests restated; the exception-type changes at the transport sites are ruled, not DECISIONs.
- W1-16, W3-21, W4-26 (the retentions) are superseded by DN-1626-REPOINT; standing ruling 17 ("no caller-visible type change") yields to it at exactly these sites.
- Door messages (W1-17): `fail` receives the door's single reason string; each site composes its own message around it.
- `request_noncanonical` is the ruled code name for the start-stage refusal.
- The commit-msg hook now lints every commit from the repo root (#333); the per-stage `gitlint --msg-filename` pre-check stays.

## Guard teeth by mutation

| id | mutation                                                                             | expected RED                                                                                                                                | expected GREEN after revert |
| -- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| M1 | `d1_registered_design.py`: remove the `require_lowercase_sha256` call                | `test_preregistration_artifact_identity_is_canonical` rows for uppercase, 63-char and non-hex RED                                           | GREEN                       |
| M2 | `d1_primitives.py`: remove the call in `EncoderInitializationIdentity.__post_init__` | the malformed-digest pin and the new mixed-case pin RED; `default_d1_encoder_initialization_identity()` still constructs (positive control) | GREEN                       |
| M3 | `d1_trainer_object_transport.py`: point `_attempt`'s `fail` at `ValueError`          | `_SITES` row `object_transport_attempt` RED (type must be `D1ObjectMailboxTransportError`)                                                  | GREEN                       |
| M4 | `d1_trainer_object_transport.py`: drop the `start_trainer` check                     | the restated transport test RED (mixed-case reaches `request_mismatch` again)                                                               | GREEN                       |
| M5 | restore row 71 in the audit file without the call site                               | `test_caller_audit_is_committed_and_covers_the_live_tree` header-count arm RED                                                              | GREEN                       |
