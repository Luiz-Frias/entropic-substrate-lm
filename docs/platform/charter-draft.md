# Platform member charter (draft)

**Status: DRAFT, 2026-11-27.** Drafted in horizon M0 of the platform refactor under register
rows D7 and D16 of `docs/plans/series-plugin-topology.plan.md`. The `entropic-platform` member
does not exist yet. Horizon M3 creates it and moves this text into the member's own `CLAUDE.md`,
where it becomes the member's bearing. Until then this file binds nothing, and the plan governs
wherever the two disagree.

This is the platform member's own north star and horizon. It is separate from the repo's, which
stay in the root `CLAUDE.md` and govern the science. It is also separate from the refactor
initiative's, which ends when the sprint closes. This charter outlives the sprint.

## North star

A library for training machinery that can be lifted into another repository and used there
without this repo's science (ruling R6).

The platform owns what sits around the optimizer step: admission, launch, custody, telemetry,
capture and the evidence protocol. It does not own what PyTorch does, autograd and modules, or
what a cluster scheduler does. A consumer brings a model, an objective and a data source, and
gets a run whose every claim can be traced to a sealed artifact.

Its motive is reuse for other training work: other schedules and curricula, other backbones,
mixture-of-experts machinery, reinforcement-learning environments, fine-tuning and adapter
methods. So its ports are named by what a training harness needs, and none is named after E, D
or H.

Three properties are the bearing. Each is a promise the code keeps, not a preference:

| property | what it means | ruling |
| --- | --- | --- |
| liftable | `entropic_kernel` and `entropic_platform` never import `entropic_substrate`; the member's gate passes from a bare directory | R5, R14 |
| owned once | one engine, one custody, one pod machinery, one capture path, one observability plane, one evidence protocol; a series adds adapters and never a second copy | R4 |
| evidence first | the authoritative record is the sealed chain and the captured JSONL; every tracker is a mirror, and a failed mirror is a recorded gap | C2, D19 |

## Horizon

The platform is sufficient when it has hosted four harness shapes through the one engine, with
telemetry compared through the admissible reader:

| shape | fixture | what it proves |
| --- | --- | --- |
| byte-level language model | the E series tiny fixture | a plain supervised loop with field pre-training as an adapter |
| registered four-arm experiment | the D1 series tiny fixture | journal-first custody, exact resume and the filing protocol |
| hierarchical latent predictor | the H series tiny tracer | phase tuples, pair runs and the sealed checkpoint chain |
| foreign fine-tune | a LoRA adapter on `autogluon/chronos-bolt-tiny` | the platform hosts a model it did not build, from a library it does not wrap |

The oracle is the fine-tuning rubric of ruling R21 under the hardware gates of R20, plus the
member copy test: the member, copied to a bare directory with its declared dependencies, passes
its own gate.

Primitive parity, for orientation. A managed fine-tuning service such as Tinker exposes four
primitives. The platform is sufficient as its self-hosted analogue when each has a port and a
passing contract test:

| primitive | platform port | contract test exists |
| --- | --- | --- |
| forward and backward | `Objective` over a `ModelFactory` product | not yet, M6 |
| optimizer step | the engine's checked step with `LearningRateLaw` and `PhasePlan` | not yet, M7 |
| sample | `Source` next batch, snapshot and restore | not yet, M6 |
| save state | custody `Journal` append and seal with `CheckpointPayload` | not yet, M7 |

Port names in this table follow the plan's topology table and are estimates until the M6 design
gate fixes them.

Out of reach on purpose. The reinforcement-learning arm of `Source` and the mixture-of-experts
adapter method ship as typed placeholders that refuse by name, each with an expiry test. A
second tracker is write-only. Extraction to a repository with its own owner waits for a third
consumer (ruling R6, amended 2026-11-01). Reaching the horizon and continuing because the work
is interesting is a violation.

The horizon closes with a plan-versus-actual audit that marks each ruling honored, partial or
lapsed and proposes the next star.

## Boundaries (to be completed at M3 and M6)

| item | value | fixed at |
| --- | --- | --- |
| import root | `entropic_platform` | R14 |
| may import | `entropic_kernel`, and third-party libraries it declares | R13 |
| never imports | `entropic_substrate` in any form, core included | R13, R14 |
| series-facing surface | `entropic_platform.ports`; each cell keeps its own ports file behind it | plan, "Topology, restated" |
| testing surface | `entropic_platform.testing`: one fake per port, a toy series, the contract suites | R17 |
| system class | orchestration apparatus for engine and pod; evidence plane for capture through gate | R7 |
| operator stop list | contracts and schemas, checkpoint wire shape, pipeline promises, one-way doors | R7 |
| gate command | not defined yet | M3 |
| consumption | pinned dependency by default; lift by clone or fork as the operator's alternate | R8 |

## Open at draft time

- The port list above is an estimate. The M6 design gate fixes it.
- The gate command and the member's public surface are written when the member exists, at M3.
- Whether nested `CLAUDE.md` files load is measured at M3. If they do not, this text moves to a
  path-scoped rule and the member keeps a copy for lifted lineages.
