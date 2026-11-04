# Series plugin topology plan: review against the rules corpus and the language packs

Written 2026-11-01 on the survey worktree at `07129a58f35975acdf7c22f036bccf6b3b7f7b30`
(plan rung four, `docs/plans/series-plugin-topology.plan.md`). Inputs read in full this
session: `~/.claude/rules/*.md` (13 files, three of them modified 2026-11-01),
`rules-lib/doctrine/*` (7), `rules-lib/pattern-cards/{families,shared,agent-development}/*`
(24 + 13 + 5 cards), `rules-lib/playbooks/*` (44 playbooks plus index),
`rules-lib/typescript/**` (19 modules), `rules-lib/rust/modules/**` (24 modules),
`rules-lib/python/master-python-patterns.md`, `rules-lib/reference/*` (2). Measurements of this
tree are labelled `[M, grep at 07129a58]`; everything else is doctrine citation.

Verdict in one line: **the plan's rulings stand; nine of its sections take amendments, two
rows of the placement table are reshaped, and three new operator decisions are opened (D8 to
D10). Nothing is redesigned.** The amendments come from two sources the plan could not have
read: the 2026-11-01 policy entries (pull-forward at the open door, the pod-to-host wire ladder,
sealed evidence never migrates) and the language packs, whose Rust and TypeScript forms map
onto Python forms this tree already uses in 496 frozen value types and 95 Protocol ports.

---

## 1. Verdict by plan section

| plan section | verdict | what changes |
| --- | --- | --- |
| Context, pins table | stands, amend | classify the five environment-bound names (section 3.11); define the spec digest (3.4) |
| R1 ladder protocol is platform | stands | |
| T1 own sprint after H1 | stands | |
| R2 explicit registry | amend | closed `SeriesId` enumeration in the CLI member; capability version, duplicate policy, declared order, isolated import failure, resolved registry printed and digested; a string-import tooth (3.2) |
| R3 retirement flag plus admission row | stands | |
| C1 pins from spec files | amend | `spec_version` key; digest over the parsed frozen value, never file bytes; the numerics key leaves the spec (D10) |
| R4 engines are platform | stands, amend P3b proof | the step loop must be linear and total in one module (3.6) |
| R5 workspace members | amend | series members are packaging boundaries, not hexagons: no `ports/` under a series; `deptry` per member as the undeclared-dependency refusal; the python pin travels with a member; extras as the feature-flag form (3.7) |
| R6 platform as library | reshape the consequence | consumption by pinned dependency, copy only for a class 6 spike (D8); the member horizon must host one foreign harness shape (3.9) |
| R7 member `CLAUDE.md` | amend | each member names its system class and the operator STOP list for it, its gate command and its public surface (3.10) |
| C2 observability ports | amend | tracker failure is a recorded gap, never silence; `RunRegistry` is a projection over the capture ledger; tracker SDKs behind extras; Trackio must clear the `exclude-newer` anchor (3.8) |
| D1 TOML | stands, amend | `spec_version`; canonical digest definition (3.4) |
| D2 reader lists in `series.toml` | amend | registry order is defined as sorted by `id`; the composed identity is generated inside the gate and diffed (3.4) |
| D3 schema versions in code | stands | |
| D5 unified checkpoint custody | amend | the payload slot is a typed port with a per-series payload schema version, not an opaque mapping; the unified format is a new named version under the wire ladder; sealed H evidence never migrates and the read-old table gains the rows (3.5) |
| D6 Trackio | stands, prerequisite | the adapter's pinned release must clear the `exclude-newer` anchor or the anchor advances forward (3.8) |
| D7 member north star and horizon | amend | the horizon names its oracle and includes a fourth, non-series fixture (3.9) |
| Seam section | amend | numerics are a promise, not a selection (D10) |
| Target topology | stands | |
| Placement table | reshape two rows | `Curriculum` and `Environment` fold into one `Source` port with two implementation families (D9); `Admission` keeps D1's confirmatory matrix as its second implementation (3.3) |
| Typestate paragraph | amend | name the Python form and its compile-fail test (3.1) |
| Three teeth | amend, add one | canonical digest plus a round-trip property test; the layers contract is blind to string imports, so a syntactic tooth joins it; a fourth tooth refuses `ports` under any series member (3.12) |
| Phases table | amend | a vehicle column; hub-file owners; the pull-forward rule cited on P1, P3b, P5, P6; the worktree hook installs every member root; the prek hook edit is an operator ask (3.13) |
| Risks | stands, add two | file-bytes digest drift; silent environment layering (3.11) |

---

## 2. Cross-language transfer table

Columns: the pattern as the Rust pack states it, the TypeScript pack's form, the Python form
this tree uses, the transfer cardinality (1:1 literal, 1:* one source form to several Python
forms, *:* several forms collapse), what exists here today, and what the plan must say.

| # | pattern | Rust form (pack) | TypeScript form (pack) | Python form here | cardinality | today `[M, grep at 07129a58]` | plan touch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | newtype for ids and invariants | `struct Id(u64)` (20, 25) | branded type (10) | frozen slotted one-field dataclass with a validated constructor | 1:* (Rust newtype splits into `NewType` for static-only and a frozen dataclass for runtime refusal) | 0 `NewType`, 421 `@dataclass(frozen=True, slots=True)` | `SeriesId`, `SpecDigest`, `PortVersion` are frozen value types, never `NewType`: strict codecs need runtime refusal |
| 2 | typestate, consume self and return the next type | `Builder<HasHost>` plus `PhantomData` (35) | branded state tokens, state-specific interfaces (10, 80) | distinct frozen `init=False` classes minted only by a factory, a transition method that returns the next class, runtime guards kept | *:1 (Rust typestate and Rust enum FSM both land on this one form) | 0 phantom markers, 14 `init=False` frozen classes, lifecycle FSM in kernel | registry typestate `SeriesSpec` to `AdmittedSeries` to `BoundSeries` in this form, with the compile-fail test in row 15 |
| 3 | sealed trait | private supertrait (25) | module-private plus discriminated union | `@final` classes, a closed `StrEnum`, a registry allowlist | 1:* fuzzy | 111 `StrEnum`, 0 `@final` on ports | the series set is closed: `SeriesId` is an enumeration in the CLI member and `series.toml` may only name a member of it (R2 amendment) |
| 4 | enum FSM with per-variant data, exhaustive matching | enum plus `match` (35) | discriminated union, exhaustive `switch`, `assertNever` (80) | tagged union of frozen classes, `match`, `typing.assert_never` | 1:1 literal | 5 `match` statements, 0 `assert_never` | kernel lifecycle transitions become a total transition function checked by basedpyright strict (`reportMatchNotExhaustive`) |
| 5 | expected failure as data, invariant as panic | `Result<T, E>`, `thiserror` in libraries, `anyhow` in binaries (30) | `Result` discriminated union, throw only for invariants (30) | typed exception hierarchy for invariants and refusals; a frozen outcome value only where a partial result is contract-authorized | 1:* (Rust `Result` splits by trigger) | 0 `Result` types, 100 exception subclasses, 24 `*Result` value classes | spec load and codec refusal raise; tracker outcomes and per-arm gaps are values (C2 amendment) |
| 6 | error cause chain | `source` field, `.context()` (30) | `.cause` unwrapped at the boundary (playbook keep-invariant-throw-bare) | `raise X(...) from e` with typed fields; invariant raises stay bare outside any `try` | 1:1 | exception classes carry fields; not audited for bare raises | member CLAUDE.md local rulings name the invariant-raise rule |
| 7 | builder at a public surface | consuming builder, `build()` only on the complete state (35) | options object, staged interfaces (card builder) | the strict spec loader: parse, validate the whole invariant set once, freeze; no partial product escapes | 1:1 fuzzy | 8 `yaml.safe_load` loaders, 0 `tomllib` | the loader is the builder; no defaults, no partial spec (C1) |
| 8 | strategy behind a trait, static in the engine | generics for static dispatch, `dyn` at the interchange (20, 60) | interface plus constructor injection (140) | `typing.Protocol`, implementations injected at the composition root | 1:1 literal | 95 Protocol classes, 25 `@runtime_checkable`, all series-named | `ports.py` Protocols; the 95 existing Protocols are inventoried into platform port, series-internal, or deleted (P3a design gate) |
| 9 | plugin registry | explicit factory map; enums or features for a closed set (card plugin-registry, Rust form) | typed manifest, allowlisted dynamic import at composition (card, TS form) | explicit registry over `series.toml`, allowlisted module path, isolated import failure | *:1 (the TS manifest form plus the Rust closed-set rule) | 0 `entry_points`, 0 `importlib.import_module`, no registry | R2 amendment: closed enumeration plus manifest; the card's five fitness functions become registry tests |
| 10 | workspace of inward-dependent crates | Cargo workspace, `[dependencies]` refused when undeclared (card modular-monolith) | monorepo packages with export maps (50, 60) | uv workspace members, per-member `pyproject.toml`, import-linter for internal layers, `deptry` for third-party declarations | 1:1 literal, two tools for one compiler check | `deptry` declared in `pyproject.toml`, not per member; import-linter with three contracts | R5 amendment: `deptry` runs per member in the member gate |
| 11 | cargo features | feature flags, scalar fallback compiled by default (50, 05) | conditional exports | `[project.optional-dependencies]` extras per member | 1:1 | `mlflow-skinny` is a hard dependency | C2 amendment: tracker SDKs behind `entropic-platform[mlflow]`, `[trackio]`; the jsonl adapter always on |
| 12 | `#[non_exhaustive]`, additive extension | new builder method is non-breaking (architecture-investment carry tiers) | new optional field on an options object | a new spec version; strict exact-key tables are not additive | fuzzy | strict codecs refuse unknown keys | C1 amendment: `spec_version`, admitted versions enumerated in code (D3 form) |
| 13 | detect once, dispatch forever | `OnceLock` selector (15, 0105) | module-level memo | resolve the registry once in the CLI main; `Final` module constants | 1:1 fuzzy | 138 `Final` | `BoundSeries` resolved once; no lazy import inside the engine (plugin-registry failure mode: import-time side effects) |
| 14 | structured concurrency, declared budgets, three exits | `CancellationToken`, `JoinSet`, bounded channels (40) | `AbortSignal`, bounded parallelism (40) | watchdog process with a declared budget; every result schema carries an exhausted-pending arm | 1:1 fuzzy | H watchdog exists; D1 controller has an attempt flow | P3c: the `Provider` contract names its three terminal states (axiom Terminal) |
| 15 | compile-fail tests | `trybuild`, compile-fail (70) | `@ts-expect-error` type tests (100) | a `tests/typing/` fixture tree run by basedpyright with `reportUnnecessaryTypeIgnoreComment` on, so each expected-error comment is load-bearing | 1:1 literal | basedpyright strict on; no typing fixtures | the typestate card's fitness function for the registry and the lifecycle FSM |
| 16 | property tests, mutation tests | `proptest`, `cargo-mutants` (70) | property-based tests (100) | `hypothesis` round trips; guard teeth proven by hand mutation (playbook prove-guard-teeth-by-mutation) | 1:1 | `hypothesis` absent, no mutation tool | tooth 1 gains a spec canonicalization round trip |
| 17 | deterministic generated code, regenerated in the gate | codegen determinism (50) | generated files stable for diffs (150); playbook tracked-means-shipped | the reader-identity generator regenerates inside the gate and diffs | 1:1 | `tools/regime_rows.py` exists; regeneration not in the gate | D2 amendment |
| 18 | schema first, one source of truth | serde types | Zod schema then `z.infer` (20, 120) | the frozen dataclass is the schema; the strict codec validates into it | 1:1 fuzzy | 2 `TypedDict`, 496 frozen classes | spec tables map one to one onto frozen classes in `kernel/spec` |
| 19 | poor man's DI at the composition root | generics or trait objects chosen outside the domain (card hexagonal) | `createX` factories returning typed bundles (140) | adapters passed into platform factories; one composition root | 1:1 literal | `composition/wiring/*` | stands as written |
| 20 | public surface is explicit | `pub` versus `pub(crate)` | export only what is supported; no barrel sprawl (50) | member root `__init__.__all__` is the surface; an import-linter forbidden contract keeps series out of platform submodules | 1:1 fuzzy (no `pub(crate)` in Python) | not enforced | tooth 2 amendment |
| 21 | file-size ratchet | not stated | 600 to 800 line window, ratchets down only, tests excluded (50) | a per-member file-ceiling lint | 1:1 | no ceiling; largest file 1007 lines | P1 sets the tightest fitting threshold and P3b must not merge three trainers into one 2000-line engine |
| 22 | CLI exit codes and config layering | flags over env over file; meaningful exit codes (45) | validate env at startup (60) | the CLI maps exception classes to exit codes; env never layers silently over a spec | 1:1 | `cli.py` 967 lines | section 3.11 |
| 23 | actor with a mailbox | task plus mailbox (40) | not stated | the D1 mailbox over an object store is an actor across machines | *:* fuzzy | `ladder/run/transport` | P3c keeps the mailbox as the platform actor form |
| 24 | static shape in the hot loop | SIMD and dispatch rules (0100 to 0110, 60) | not applicable | torch modules stay concrete; the `Objective` port is called once per step, never per op | 1:1 | not applicable | a note on the `Objective` row |

Rows 25 and beyond do not transfer: FFI (80), embedded (90), database (47), SIMD lane policy,
and the Python master patterns' experimental sections have no counterpart in this plan.

---

## 3. Findings and amendments

Each finding names the plan section, the corpus source, the defect or gap, and the amendment
text ready to fold into the plan.

### 3.1 Typestate paragraph: name the Python form

Source: typestate card (Python form, fitness functions); Rust 35; TS 80.

The plan says typestate survives and names three registry states, but not the form. In this
tree the form is measured: frozen, slotted, `init=False` classes minted by a factory token. The
card's Python caveat applies: the runtime guard stays because Python cannot make the static
guarantee. The compile-fail lane is row 15.

Amendment: "`SeriesSpec`, `AdmittedSeries` and `BoundSeries` are three frozen `init=False`
classes under `entropic_cli.registry`, minted only by the registry's own factories; each
transition is a method that returns the next class; a `tests/typing/` fixture asserts that
`BoundSeries` cannot be constructed from a `SeriesSpec` directly, run by basedpyright with
`reportUnnecessaryTypeIgnoreComment` on so the expected-error comment goes red when the guard
weakens. The kernel lifecycle FSM moves to a total transition function with `assert_never`."

### 3.2 R2 registry: close the set, then validate the manifest

Source: plugin-registry card (fitness functions, failure modes, Rust and TS forms); Rust 20
sealed traits; TS 70 "avoid stringly-typed identifiers"; the `.importlinter` header in this
tree, which measured that a string-only `importlib.import_module` shim is GREEN-BLIND to the
layers contract.

The series set is closed and known at build time. The Rust form for a closed set is an
enumeration, not a registry; the TS form is a typed manifest with an allowlisted import. The
plan has the manifest half. Four of the card's five fitness functions are missing: capability
version, duplicate policy, deterministic observable discovery, and isolated import failure.
The allowlisted import by module string is exactly the form the layers contract cannot see,
which matters because a series plugin could reach another series the same way.

Amendment to R2's consequence: "The set of series is a promise: `SeriesId` is a `StrEnum` in
`entropic_cli` and `series.toml` may only name a member of it. The spec selects status and
admission. The registry (a) refuses two specs with one `id`, (b) checks `platform_api` in each
spec against the version `ports.py` exports, (c) binds in order sorted by `id`, (d) wraps the
plugin import so a failure raises `PluginLoadError(series_id, cause)` and never a bare
`ImportError`, (e) hands each plugin the ports it implements and nothing else, and (f) prints
the resolved registry with every spec digest at startup, without secrets, and that record enters
the run identity."

Tooth, see 3.12: an ast-grep rule in prek refuses `importlib.import_module` and `__import__`
outside `entropic_cli`.

### 3.3 Placement table: `Environment` has zero implementations

Source: hexagon-placement card litmus 3 ("zero measures nothing; the port is speculative");
architecture-investment "seam sized for the next waypost, machinery for the current one" and
"one-way doors at final level now".

The plan's own row says "no RL source today". A port with zero implementations is speculative
under the card. What is a door today is the engine's step contract: what `Objective.step`
receives. That is final-level work now. The reach to RL is kept by making the batch source one
port with two implementation families, the second arriving when measured.

Reshape, opened as D9: fold `Curriculum` and `Environment` into `Source` (next batch for a
step, with an order digest); `Curriculum` is its supervised implementation family with the two
measured implementations (`SequenceSampler`, the H sentence sampler); an RL environment is the
second family, written when the first RL objective exists. The port list goes from nine to
eight.

`Admission` keeps two implementations: `ladder/run/admission.py` for H and the D1
confirmatory matrix (`ladder/register/d1_confirmatory_matrix.py` already exposes a Protocol
`[M]`). The plan's "(H only)" annotation is corrected.

### 3.4 C1, D1, D2: define the digest and version the spec

Source: axiom Epoch (verdicts bind to derived immutable identifiers such as a config hash);
playbook pin-tip-before-claim; TS 20 "versioned schemas with explicit migration at the
boundary"; TS 150 and playbook tracked-means-shipped on regenerating derived artifacts inside
the gate.

TOML has no canonical serialization. A digest over file bytes re-pins the identity on a
whitespace edit and is not what the Epoch axiom means by a derived identifier.

Amendment to the seam section: "The spec digest is SHA-256 over the canonical JSON of the
parsed, frozen value (sorted keys, typed scalars, no floats where an integer or string is
declared). File bytes are not the identity. Every spec carries `spec_version = N`; the loader
enumerates the versions it admits in code (D3 form) and refuses others by name."

Amendment to D2: "Registry order is sorted by `id`. The composed reader identity is generated
by the identity tool inside the gate and diffed; a stale generated row fails the gate, it is
never hand-edited."

Tooth 1 gains a `hypothesis` round trip: parse, freeze, canonicalize, digest is stable under
key reordering and whitespace, and changes on any value edit.

### 3.5 D5 unified custody: the payload slot is the under-determined port tell

Source: architecture-investment "warning sign that a port is under-determined: a `dict` field
documented provider-specific"; policies.md 2026-11-01 entries (pod-to-host wire ladder, sealed
evidence never migrates); `docs/identity/read-old-table.md` exists in this tree `[M]`.

"A per-series payload slot" documented as series-specific is the smell the doctrine names. The
fix is the one it prescribes: type it. The unified format is also a new version of a named
contract, so the wire ladder binds, and sealed H artifacts from H0 and H1 bind the old reader
identity and never migrate.

Amendment to D5: "The payload slot is a port, `CheckpointPayload`, with `encode`, `decode` and
a per-series `payload_schema` version string kept in code beside its codec (D3). The unified
format is `platform-training-checkpoint-v3`, written only by the engine; readers admit v3 and
`h-training-checkpoint-v2` by name for the persisted-artifact rollback window; the D1 15-key
mapping and the E `format_version` file are admitted read-old for the same window and deleted
by a removal PR whose test proves no retained artifact needs them. Sealed H0 and H1 checkpoints
never migrate: P3b and P6 add their rows to `docs/identity/read-old-table.md`, and replay
happens at the epoch commit."

### 3.6 R4 and P3b: thin means linear and total

Source: architecture-investment "thin is not hidden"; TS 50 file-size ratchet; largest file
today 1007 lines `[M]`.

Merging three trainers invites two failure modes the doctrine names: a 2000-line engine, or a
dozen short files that scatter the step loop. The acceptance criterion is stated once.

Amendment to P3b's proof: "The engine's run loop is one function readable top to bottom with no
policy branching of its own; every series difference enters through a port call at a named
point. The engine member carries the file ceiling from P1."

### 3.7 R5 members: packaging boundaries, not hexagons

Source: hexagon-placement card (formula, fitness "every module-level `ports/` directory can
name its second implementor"); modular-monolith card Rust form; the plan's own sentence that
Python does not refuse an undeclared import; memory `floating-python-pin-drifts-regime`
(identity regime keys on the patch version).

The platform earns its own hexagon by measured axes. The series do not: they are
implementations of platform ports and share its domain language. A series member is a
packaging and gate boundary, which is what R5 asks for, and must not grow its own ports.
Cargo refuses an undeclared dependency at build time; import-linter covers internal layers only,
so the third-party half of that check needs `deptry` per member. A copied member must carry
its own python pin or its identity regime drifts.

Amendment to R5's consequence: "Series members own no `ports` module; only `entropic-platform`
does (tooth 4). Each member's gate runs `deptry` against its own `pyproject.toml`, so an import
of `mlflow` from a series member fails the way an undeclared crate fails `cargo build`. Each
member carries a `.python-version` pinned to the regime patch version so the copy test runs in
the same regime. Optional adapters are extras (`entropic-platform[mlflow]`,
`entropic-platform[trackio]`), the Python form of a cargo feature; the jsonl adapter is never
optional."

### 3.8 C2 observability: silence is inadmissible, projections are not truth

Source: trigger-sensitive-failure card ("partial results omit the failed item" is a failure
mode; expected failures cross seams as typed data); axiom Artifact (silence inadmissible);
state-and-persistence family and append-only-ledger card (a ledger becomes truth only on a
promotion signal); official-sdk-adapter card ("on provider removal delete factory, adapter,
live probe and credentials together"); `exclude-newer = 2026-08-13` in `pyproject.toml` `[M]`
with playbook advance-cooldown-anchor-forward.

Amendment to C2: "A tracker outcome is a value, never an exception that escapes the composite
sink: the authoritative run record carries `mirrors = [{tracker, status, last_step}]` with
status in `ok | degraded | refused`, so a mirror that fell behind is a recorded gap. The D1
strict adapter is the one raise-through member and says so in its name. `RunRegistry` is a
projection over the capture ledger; a tracker's own registry is a mirror of that projection and
never a source. Tracker SDKs live behind extras and the `mlflow-only-from-adapter` contract
generalizes to one contract per tracker SDK. D6 prerequisite: the pinned Trackio release must
resolve under the `exclude-newer` anchor; if it does not, the anchor advances forward on the
schedule, never backdated."

### 3.9 R6 and D7: the horizon needs a foreign fixture, and copy is not the consumption mode

Source: architecture-investment horizon gate ("names its oracle"; "a like-X horizon resolves to
a measured parity matrix"); epistemic hygiene rule 2 (a missing side is instrumented, reframed
or declined); portfolio-and-platform family (second consumer names the seam, third extracts
with one owner); playbook wrap-sibling-never-patch; system class 6.

R6 claims the platform generalizes beyond E, D and H. Three series fixtures cannot measure that
claim; they measure that the platform hosts the series it was carved from. The claim needs one
fixture that is none of them. The plan's Tinker comparison is phrased "like"; the doctrine turns
that into a parity matrix or drops it.

Amendment to D7: "The platform member's horizon names its oracle: the three series tiny fixtures
plus one foreign fixture (a LoRA fine-tune of a tiny public checkpoint, the smallest shape R6
names) run through the one engine with telemetry compared through the admissible reader. The
Tinker sentence either becomes a four-row parity matrix (forward-backward, optimizer step,
sample, save state) or is removed. The horizon closes with the plan-versus-actual audit."

Opened as D8: the sanctioned way a scratch repository consumes the platform. The user's words
were "copy, then refactor lightly"; the operator's own playbook says wrap a sibling, never patch
it, and a consumer-side fork is a defect signal. Both hold if scoped: a copy is admissible for a
class 6 spike (timeboxed, kill-don't-extend); anything that survives the spike consumes the
platform as a pinned dependency (`[tool.uv.sources]` git tag or path) and sends its refactor
upstream. The copy test in P7 remains the proof that the boundary is clean; it is not the
consumption mode. The first scratch consumer is the second consumer under the paved-road rule and
names the seam with a review; extraction to its own repository with one owner waits for the
third.

### 3.10 R7 member rules: name the class and the STOP list

Source: system-class-policies (classify at the subsystem; each class has an operator STOP list);
collaboration-protocol decision classes; TS 50 "public API surfaces are explicit".

Amendment to R7: "Each member `CLAUDE.md` names its system class and the operator-ruled STOP
list that class carries: `entropic-kernel` is class 1 for numerics and class 5 for the spec
loader; `entropic-core` class 1; `entropic-platform/engine` and `/pod` class 2 (contracts,
checkpoint wire shape, pipeline promises are operator's); `entropic-platform/capture` through
`/gate` class 4 (post-seal analysis never changes; verdict filing by a fresh reviewer); series
members class 1 plus 4 (the algorithm and its parameters are operator's); `entropic-cli` class 3
and 5. The boundaries part also names the member's gate command and its public surface, the
`__all__` of the member root."

### 3.11 Environment-bound settings and the numerics key

Source: the plan's own seam (promises in code); repo `CLAUDE.md` non-negotiable numerics;
playbook gate-numerics-on-hardware; Rust 45 config layering; TS 60 validate env at startup;
measured `os.environ` reads `[M, grep at 07129a58]`: `MLFLOW_EXPERIMENT_NAME` (3),
`ENTROPIC_SUBSTRATE_DEVICE`, `RUNPOD_POD_ID`, `TORCH_ALLOW_TF32`, `NVIDIA_TF32_OVERRIDE` (1 each).

The plan lists five environment-bound settings and does not say where they go. It also places
`execution_precision` in `platform.toml`, which its own seam forbids: the FP32 islands are a
repo promise, not an operator selection. What hardware may select is the ambient dtype.

Amendment to the pins table and the seam, opened as D10 for the numerics half: "Environment
names are classified once. Selection moves into a spec and is digested:
`MLFLOW_EXPERIMENT_NAME` to `platform.toml [observability]`, `ENTROPIC_SUBSTRATE_DEVICE` to the
run profile. Runtime identity injected by the provider (`RUNPOD_POD_ID`) is recorded, never
configured. Numerics promises (`TORCH_ALLOW_TF32`, `NVIDIA_TF32_OVERRIDE`) are refused by a code
gate when set, never read as configuration. Secrets stay out of band. Nothing layers silently:
the CLI prints the effective configuration with its digest at startup. `[numerics]` leaves
`platform.toml`; `require_execution_precision` stays a code gate; the only selectable numerics
value is the ambient autocast dtype in the run profile, branched on compute capability per the
hardware playbook, and the islands stay FP32 regardless."

### 3.12 Teeth: one amended, one added

Source: `.importlinter` header measurement (string shims are GREEN-BLIND); playbook
prove-guard-teeth-by-mutation; hexagon-placement fitness.

Tooth 2 amendment: "The mutation proof for the `platform` never imports `series` contract and
the mutual-independence contract includes the string-import form; because the layers contract
stays green on it, a prek ast-grep rule refuses `importlib.import_module` and `__import__`
outside `entropic_cli`, and that rule is itself proven by mutation. A forbidden contract keeps
series members out of platform submodules: a series may import `entropic_platform` and
`entropic_platform.ports` only, the Python approximation of `pub(crate)`."

Tooth 4 added: "A test fails if any `entropic-series-*` member contains a `ports` module or a
Protocol class whose name ends in `Port`."

Tooth 3 note: adding the prek hook edits `.pre-commit-config.yaml`, which is an operator ask
under the non-negotiable rule; P1 lists it as a decision, not a lane action.

### 3.13 Phases: vehicles, hub owners, the open-door rule

Source: playbook pick-vehicle-by-size; conflict-minimized-worktree-stacks card (every hub file
has one owning stack); policies.md pull-forward-at-open-door (2026-11-01, this repo, DH-44);
playbook install-every-package-root-in-fresh-worktree; memory `worktree-write-guard-hook`.

The plan predates the pull-forward policy by two days. Four of its phases move a door: P1 the
configuration door, P3b the checkpoint door, P5 and P6 the reader identity. Each must leave
every producer and consumer truthful at its tip, with no stub reader and no placeholder status.

Amendment to the phases table: a `vehicle` column (P1, P3a, P3b, P3c: one cascade-arch lane
each, a door or design judgment; P2 and P4: mechanical, one lane or hand-authored under the
measured band; P5, P6: one lane plus the conductor's re-pin; P7: one lane), and a `hub owner`
column naming the single stack that may touch `pyproject.toml`, `uv.lock`, `.importlinter`,
`platform.toml`, `tests/test_identity_epoch.py` and `.pre-commit-config.yaml` in each wave.
P1 declares all seven members up front so the root `pyproject.toml` and the lockfile are touched
once. P1 also updates the worktree-create hook to `uv sync --all-packages`, since a fresh
worktree with one package root installed fails the member gates with errors that look like base
defects. P1, P3b, P5 and P6 cite `pull-forward-at-open-door` in their rail cell.

---

## 4. New decisions for the operator

| # | decision | recommendation | why |
| --- | --- | --- | --- |
| D8 | how a scratch repository consumes the platform | pinned dependency (`[tool.uv.sources]` git tag or path); copy admissible only for a class 6 spike with a timebox; survivors come upstream | wrap-sibling-never-patch; paved-road rule of three; the copy test stays as the boundary proof |
| D9 | fold `Curriculum` and `Environment` into one `Source` port | yes; `Curriculum` is the supervised implementation family, RL environments the second, written when the first RL objective exists | a port with zero implementations is speculative; the step contract is the door and gets its final shape now |
| D10 | remove `[numerics]` from `platform.toml` | yes; the FP32 islands are a code promise and `require_execution_precision` stays a gate; the ambient dtype is the only selectable numerics value and lives in the run profile | the plan's own seam; repo non-negotiable numerics; gate-numerics-on-hardware |

Rulings on D8 to D10 are recorded in the plan on the commit that records them; this review is
transport until then.

Ruled 2026-11-01, recorded in the plan as R8 to R10 on the same commit as this review: D8, both
modes are first class, pinned dependency by default and a lift (a clone or fork that keeps
history, refactored to the new repository's goals) as the operator's alternate, so the
recommendation above is superseded on the lift half and stands on the pinned half; D9, yes, with
the RL family implemented now as a typed refusing placeholder with an expiry test; D10, yes.
D11, which external environment API the RL adapter speaks, was opened in the plan's register and
ruled the same day as R11 on the recommendation as written: Gymnasium's `reset` and `step`
contract as the port shape, TorchRL `EnvBase` as the first adapter, TRL deferred to the first
LLM-RL objective.

---

## 5. Measured facts this review rests on

All at `07129a58` in this worktree, by `grep -rn` over `src/entropic_substrate`.

| fact | value |
| --- | --- |
| Protocol classes / runtime_checkable | 95 / 25 |
| frozen dataclasses / of which slotted / `init=False` | 496 / 421 / 14 |
| `Final` / `Literal[` / `StrEnum` / `Enum` | 138 / 57 / 111 / 113 |
| `NewType` / `TypeVar` / `Generic[` / PEP 695 generic defs | 0 / 0 / 0 / 10 |
| `match` statements / `assert_never` | 5 / 0 |
| `Result` types / exception subclasses / `*Result` value classes | 0 / 100 / 24 |
| `tomllib` / `yaml.safe_load` / `entry_points` / `importlib.import_module` | 0 / 8 / 0 / 0 |
| `os.environ` names read | 5 distinct, listed in 3.11 |
| largest source file | `ladder/capture/adapters/mlflow.py`, 1007 lines |
| `deptry` / `import-linter` / `basedpyright` declared | yes / yes / yes (strict) |
| `hypothesis` / mutation tool / `trackio` declared | no / no / no |
| `exclude-newer` anchor | 2026-08-13 |
| `docs/identity/read-old-table.md` | present |
