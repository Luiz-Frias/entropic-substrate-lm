"""p0 identity-epoch door (frozen_after: p0, except the two ruled bump phases:
p4 bumps h-checkpoint-reader-v1 and h-source-reader-v2; p2 retires the D1
closure manifest row). Every other phase's fixture value here must byte-match
this file.

Design-gate ruling (2026-09-18): the composed H digests hash
``platform.python_version()`` and (for the checkpoint reader only)
``torch.__version__`` -- MEASURED, they are NOT the same regime key. This
fixture therefore asserts the SOURCE-DERIVED components exactly (reader
module tuple, format version, tau eras, codec construction inputs, the D1
closure manifest) and records the COMPOSED digest per measured regime
starting with the local one. An unmeasured regime fails loud, by name, in its
own test function -- never a skip (house rule 1's false-green class).

Retraction (house rule 4, MEASURED under .venv/bin/python 3.12.13): the
brief's and the plan's "52-entry D1 closure manifest" figure is WRONG against
p0's tip. ``installed_d1_reader_source_files()`` returned 73 entries there,
not 52.

p1 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p1-anchors): the count above is ALSO now wrong against THIS lane's
tip. ``config.py`` gained a real, static ``import entropic_substrate.paths``
(the new layout-invariant repo_root resolver, E-3/DN-8); ``paths.py`` is
itself reachable from the D1 closure's roots through that one edge, so the
closure grew 73 -> 74 members with ``entropic_substrate.paths`` the sole
addition. This file pins 74, the measured value, and this paragraph is the
explicit retraction the ruling requires -- the 73 literal is not carried
forward anywhere in this fixture. p1 also re-pins the composed
h-checkpoint-reader-v1 rows (COMPOSED VALUE ONLY, the label itself does not
bump before p4): ``sentence_prediction/checkpoint.py::_reader_identity()``
sweeps ``sentence_prediction/*.py`` by content, and this lane edited
``worker_execution.py``'s anchor (still E-3/DN-8), changing its bytes and
therefore every regime row's digest. ``_SOURCE_REGIME_TABLE`` is untouched:
the source reader's fixed module tuple never included ``worker_execution``.

p4 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p4-h-tracer, rung 1 sub-commit "split d_ladder_worker_runtime"):
this commit's D-E split rewrote the import lines of
``sentence_prediction/worker_execution.py`` and
``sentence_prediction/native_verification.py`` (now importing
``measure_hardware``/``verify_source_tree`` from the newly landed
``entropic_substrate.ladder.run.worker.hardware`` instead of the old
top-level ``entropic_substrate.d_ladder_worker_runtime`` name) -- changing
those two files' bytes and therefore, via ``checkpoint.py``'s
``package.glob("*.py")`` sweep of ``sentence_prediction/``, every
h-checkpoint-reader-v1 COMPOSED regime row (same mechanism p1 already
recorded above; the label itself still does not bump until this phase's
identity-pin commit, per plan). Both rows below were recomputed with
``tools/regime_rows.py`` (local: ``--python 3.12.13 --torch 2.13.0``; CI:
``--python 3.12.14 --torch 2.13.0+cpu``) and the OLD values they replace,
``5cef5371d10a6136ef466a7454efaecf1c0054ecd4fcdafacb1caf85343cff46``
(local) and
``cdf8cd5e7cecc5866f160c1776de78d0c09f7314a1e9b06a21e8251a872a57e6`` (CI),
are retracted here, not silently overwritten. ``_SOURCE_REGIME_TABLE`` is
untouched: none of the seven ``h-source-reader-v2`` member files
(``sentence_prediction`` package, ``records``, ``segmentation``,
``corpus``, ``capture``, ``d1_artifact_snapshot``, ``evidence.lifecycle``)
were touched by this split, confirmed by re-measuring both rows unchanged.

p4 re-pin #2 (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p4-h-tracer, rung 1 batch (a) "core.sentence: 18 leaves"): this
batch ``git mv``'d 18 of ``sentence_prediction``'s leaves (boundary, codec,
consumer, corpus, head, law, model, numerics, objective, profile, records,
recovery_state, rollout, runtime_precision, sampling, scoring, segmentation,
spine) to ``entropic_substrate.core.sentence.*`` and rewrote their import
lines onto the landed or ``__legacy`` addresses (never the old prefix, per
the zero-old-prefix invariant). Both composed regime rows change: (1)
h-checkpoint-reader-v1 changes because ``checkpoint.py``'s
``package.glob("*.py")`` sweep of ``sentence_prediction/`` now sees 29
files instead of 47 (the 18 moved leaves no longer live there); (2)
h-source-reader-v2 changes because three of its seven fixed members --
``records``, ``segmentation`` and ``corpus`` -- are in this batch's 18 and
their bytes changed (import rewrites). The ``sentence_prediction`` package
member and ``capture``/``d1_artifact_snapshot``/``evidence.lifecycle`` are
untouched this batch (capture.py moves in a later batch, the rail modules
move last); ``_FROZEN_SOURCE_READER_MODULES`` below is therefore still
byte-identical -- only the SOURCE BYTES behind three of its entries moved,
not the tuple of dotted names itself, so this fixture's frozen-tuple
assertion needed no edit. Both rows were recomputed with
``tools/regime_rows.py`` (local: ``--python 3.12.13 --torch 2.13.0 --pysbd
0.3.4``; CI: ``--python 3.12.14 --torch 2.13.0+cpu --pysbd 0.3.4``), and the
values they replace --
``04af5bd2e34b3ee050f9f13938115268f26266cceb8499ef028f7e555772b140``
(checkpoint, local), ``bc09822a7d2fa55a50d43de7a8227b9a26a9c9b160e21344110811973bd830e9``
(checkpoint, CI), ``da8205c91989bfc09feafcd56ff46d6fd6c2b366f452308e8353e425cfeae8be``
(source, local) and ``8e973c8702bd20d9c8cfc25331346674c7070dd7cbb9e4b6488a1881f1b25165``
(source, CI) -- are retracted here, not silently overwritten. The label
itself still does not bump (h-checkpoint-reader-v1, h-source-reader-v2)
until this phase's identity-pin commit (rung 3), per plan.

p4 re-pin #3 (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p4-h-tracer, rung 1 batch (b) "core.probe: score_gain"): this
batch ``git mv``'d ``sentence_prediction/score_gain.py`` to
``entropic_substrate.core.probe.score_gain`` (a new package; D-C's
zero-byte ``core/probe/__init__.py`` marker, not a docstring facade --
``score_gain`` is not a mapping.json package-key member). Only
h-checkpoint-reader-v1 changes: ``checkpoint.py``'s ``package.glob("*.py")``
sweep of ``sentence_prediction/`` drops from 29 files to 28 once
``score_gain.py`` leaves. h-source-reader-v2 is untouched --
``score_gain`` was never one of its seven fixed members -- confirmed by
re-measuring the source row unchanged (``701d2b5b...`` local,
``f8354ee1...`` CI, byte-identical to the batch-(a) re-pin above).
Recomputed with ``tools/regime_rows.py`` (same two regimes as above); the
values replaced,
``5a3cd776bd8a8cbc8add95c772e01f259a8f00f31d95b26bd5a2283e6bb7855f``
(local) and
``2a66a7b1758d1b370d851a333e882b29c85426025fe501d20bceb9d89fd751c3``
(CI), are retracted here, not silently overwritten. Label still does not
bump until rung 3.

p4 re-pin #4 (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p4-h-tracer, rung 1 batch (c) "ladder.run: 21 leaves + capture.py"):
this batch ``git mv``'d the 21 remaining non-checkpoint ``sentence_prediction``
leaves (evaluation_records, execution, generation, journal,
mechanism_evidence, native_verification, object_worker, runner, seals,
span_diagnostics, span_probe, span_probe_artifacts, span_states,
trace_capture, training, worker_command, worker_completion, worker_execution,
worker_request, worker_snapshot, worker_transport) to flat
``entropic_substrate.ladder.run.*``, plus ``capture.py`` to
``entropic_substrate.ladder.capture.capture`` (mapping.json's dest for
``sentence_prediction.capture`` is the package-level ``ladder.capture``;
leaf restated per fragment-author convention, same as batch (b)'s
``core.probe.score_gain``). ``checkpoint.py`` itself stays in
``__legacy/sentence_prediction/`` this batch (conductor ruling 2026-09-22:
its ``_reader_identity()`` resolves ``package = Path(__file__).parent``, so
moving it before rewriting that glob to an explicit tuple would silently
widen the swept set to everything under its new directory; the rewrite and
the move both land together at rung 3). Both composed regime rows change:
(1) h-checkpoint-reader-v1 changes because ``checkpoint.py``'s
``package.glob("*.py")`` sweep of ``sentence_prediction/`` now sees 6 files
(``__init__.py``, ``checkpoint.py``, and the 4 CLI modules) instead of 28;
(2) h-source-reader-v2 changes because ``capture`` -- one of its seven fixed
members -- physically left ``sentence_prediction/`` (its digest is now read
from the landed file's bytes, which differ from the old ones by the import
rewrites above, including a second pass that caught this same batch's own
old-prefix imports of ``evidence.lifecycle``/``d1_artifact_snapshot`` --
``entropic_substrate.evidence.lifecycle``/``entropic_substrate.
d1_artifact_snapshot`` resolve through the root-level compat shim rather
than the explicit ``entropic_substrate.__legacy.*`` address, which
``tests/test_no_old_prefix.py`` caught RED across all 22 files this batch
lands, not only ``capture.py`` -- rewritten to the explicit form, per the
zero-old-prefix invariant); the ``sentence_prediction`` package member and
``d1_artifact_snapshot``/``evidence.lifecycle`` are untouched this batch
(rail modules move last, per the same rung-3 ordering). Both rows were
recomputed with ``tools/regime_rows.py`` (local: ``--python 3.12.13 --torch
2.13.0 --pysbd 0.3.4``; CI: ``--python 3.12.14 --torch 2.13.0+cpu --pysbd
0.3.4``), and the values they replace --
``9d08ecb44633e6a45b23049d635af9202db21138ec1582b3620d00fa8c2565e3``
(checkpoint, local),
``23fa1e8919bc9f35477aad0ba480d630257c41c16bc306c6e6c2176474726777``
(checkpoint, CI),
``701d2b5b85906b906242f1e6bbfbcbec88e3faefb6f639a5e89df27811dedfa4``
(source, local) and
``f8354ee1916c9e85a9e2f8dc8ce05e355881e2f1a5c9630bf2d526f3e3c13610``
(source, CI) -- are retracted here, not silently overwritten. The label
itself still does not bump until this phase's identity-pin commit (rung 3),
per plan.

p4 re-pin #5 (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p4-h-tracer, rung 1 batch (d) "composition: 4 CLI modules"): this
batch ``git mv``'d the 4 remaining ``sentence_prediction`` CLI modules
(capture_cli, cli, source_cli, span_cli) to
``entropic_substrate.composition.*`` (mapping.json's package-level dest for
each; ruling D-A's ``print_allowlist`` fragment entries use these same
landed dotted names). Only h-checkpoint-reader-v1 changes: ``checkpoint.py``'s
``package.glob("*.py")`` sweep of ``sentence_prediction/`` drops from 6
files to 2 (``__init__.py``, ``checkpoint.py``) once the 4 CLI modules
leave. h-source-reader-v2 is untouched -- none of the 4 CLI modules were
ever one of its seven fixed members -- confirmed by re-measuring the source
row unchanged (``b78ad25a...`` local, ``c63ffeb3...`` CI, byte-identical to
the batch-(c) re-pin above). Recomputed with ``tools/regime_rows.py`` (same
two regimes as above); the values replaced,
``49e9a1b11dbde1097c65dc11ef62621fb3e7b6173ca677bafdfa6cb6aef92f6d``
(checkpoint, local) and
``e6d01c13dd3f0d36c922be5528b1921e0d7553d8b36f3ea082c959c627e90aed``
(checkpoint, CI), are retracted here, not silently overwritten. Label still
does not bump until rung 3.

p4 identity pin (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p4-h-tracer, rung 3, the LAST code commit): both labels bump here.
``ladder/run/checkpoint.py``'s glob (which by this point resolved only
``__init__.py``/``checkpoint.py`` in ``sentence_prediction/``, per re-pin #5
above) is rewritten to an explicit ``_READER_MODULES`` tuple of 51 landed
dotted names (ruling "checkpoint glob"; faithful restatement, ruling E-2) --
label bumps ``h-checkpoint-reader-v1`` -> ``-v2``. ``ladder/capture/capture.py``'s
``_READER_MODULES`` tuple is rewritten from its 7 old-prefix/``__legacy``-prefixed
names to their 7 landed dotted names (the package member now ``core.sentence``,
resolving to its own minimal ``__init__.py``, not the pre-move bare
``sentence_prediction`` name) -- label bumps ``h-source-reader-v2`` -> ``-v3``.
Both rows recomputed with ``tools/regime_rows.py`` (local: ``--python 3.12.13
--torch 2.13.0 --pysbd 0.3.4``; CI: ``--python 3.12.14 --torch 2.13.0+cpu
--pysbd 0.3.4``); the OLD (still-``-v1``/``-v2``-labelled) values they
replace --
``d9be9f47178a557263ad64b0a397049db94a6554e8d877a39eba4237c6bcde6d`` (checkpoint,
local), ``af9268d28f5cd62db26d6b78800c86ef84141e57af967afbf94163e7ddeff5a1``
(checkpoint, CI), ``b78ad25a7d1865c6e0923acc55f5988f5e77a2382e74633cbd769e32efeb87d8``
(source, local) and ``c63ffeb3ad285f841ea3bbf31b54ad02b2da28c48f10c78478006a1139b34427``
(source, CI) -- are retracted here, not silently overwritten. Neither value
changes later (ACCEPTANCE).

RETRACTION (house rule 4, p5-kernel): the sentence above, "Neither value
changes later (ACCEPTANCE)", is WRONG against ruling R-PIN (conductor,
2026-09-22, precedent p1 AMENDED 2026-09-19: "frozen_after freezes label +
shape, not digest values"). What is frozen after p4 is the LABEL
(``h-checkpoint-reader-v2``) and the module-set SHAPE (the 51-member
``_READER_MODULES`` tuple) -- not the composed digest VALUE, which
re-measures whenever a later phase rewrites one member's bytes. This is not
a new exception; p1's own re-pin (above) already changed a composed value
under the still-``-v1``-labelled reader before p4 bumped the label. The
retracted sentence is not carried forward.

p5 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p5-kernel, kernel/runtime rung): this phase's runtime commit
rewrites ``ladder/run/worker_execution.py`` line 10 (and its line-31
docstring mention) from ``entropic_substrate.__legacy.paths`` to
``entropic_substrate.kernel.runtime.paths`` -- the one landed->``__legacy``
edge among the 14 p5-kernel modules (MEASURED 2026-09-22, ruling R-PIN).
``worker_execution`` is one of ``_READER_MODULES``' 51 members, so its
changed bytes move the composed h-checkpoint-reader-v2 digest; label and
membership are unchanged (R-PIN). Both rows recomputed with
``tools/regime_rows.py`` (local: ``--python 3.12.13 --torch 2.13.0``; CI:
``--python 3.12.14 --torch 2.13.0+cpu``); the OLD values they replace,
``8e4ca5a3840e5c2efa7c4e0f0a1521af576e54139588cef76169021276b8eaf3``
(local) and
``ea85d18b278423bc7de7095b8292ce224b6c7aa50428eecf443dec828eb1602a``
(CI), are retracted here, not silently overwritten. ``_SOURCE_REGIME_TABLE``
is untouched: ``worker_execution`` was never one of h-source-reader-v3's
seven fixed members, confirmed by re-measuring both source rows unchanged
via the same tool invocation (with ``--pysbd 0.3.4``).

p5b re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p5b-ladder-campaign, DESIGN-GATE RULING 2026-09-22, wf_d8e28935-075,
D-A): this phase appends five functions (``artifact_mapping``,
``artifact_from_mapping``, ``read_bound_artifact``, ``object_fields``,
``read_json_artifact``, plus the private ``_unique`` helper) to
``kernel/artifacts.py``, changing that module's bytes.
``entropic_substrate.kernel.artifacts`` is a member of BOTH composed
readers' fixed module tuples -- 1 of h-source-reader-v3's 7
(``_FROZEN_SOURCE_READER_MODULES`` below) and 1 of h-checkpoint-reader-v2's
51 (``_READER_MODULES`` in ``ladder/run/checkpoint.py``, unchanged in shape)
-- so BOTH composed digests move this phase, not only the checkpoint row as
every prior re-pin paragraph in this file recorded. RETRACTION (house rule
4): the pattern in every re-pin paragraph above this one, "``_SOURCE_REGIME_TABLE``
is untouched", was true for EACH of those phases' own specific mutated
member (``worker_execution``, ``score_gain``, the four CLI modules, and so
on -- none of which was ever one of h-source-reader-v3's seven fixed
members) -- it was never a standing guarantee that no future phase could
move the source row, and this phase is the first to touch a member common
to both tuples. Both rows recomputed with ``tools/regime_rows.py`` (local:
``--python 3.12.13 --torch 2.13.0 --pysbd 0.3.4``; CI: ``--python 3.12.14
--torch 2.13.0+cpu --pysbd 0.3.4``); the OLD values they replace,
``d4bde6d4bb4c0f6c154450b070a2336a1a34c5b4a949bb4fefaa322522dcaf42``
(checkpoint, local),
``d7c521786523b3aa0855e14413c830cf4cb7cf031b6914412041fbfa0853199f``
(checkpoint, CI),
``a6075295436de193087160f65fa2558fc28054115118946e53ecbdd4f22d510f``
(source, local) and
``214730ac2b370a7b67fc24a43077c49ecdcc59291505d80f249f235aaccf0467``
(source, CI) -- are retracted here, not silently overwritten. Label and
both module-set shapes (``h-checkpoint-reader-v2``, 51 members;
``h-source-reader-v3``, 7 members) are unchanged, per ruling R-PIN
("frozen_after freezes label + shape, not digest values"). No metric, loss
or scientific behaviour changes.

p6-core re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p6-core, FOR-1455): this phase rewrites the six landed importers of
the two ``core.probe`` targets (``composition/span_cli.py``,
``ladder/run/{span_probe,span_probe_artifacts,span_diagnostics,
worker_request}.py``, ``core/probe/score_gain.py`` -- 7 edges,
``span_probe.py`` imports both) from their ``entropic_substrate.__legacy.
analysis.*`` addresses to ``entropic_substrate.core.probe.*``, changing all
six files' bytes. All six are ``h-checkpoint-reader-v2``'s ``_READER_MODULES``
members (51-member tuple, unchanged shape), so the composed checkpoint digest
moves; label and membership are unchanged (R-PIN). Both rows recomputed with
``tools/regime_rows.py`` (local: ``--python 3.12.13 --torch 2.13.0 --pysbd
0.3.4``; CI: ``--python 3.12.14 --torch 2.13.0+cpu --pysbd 0.3.4``); the OLD
values they replace,
``518eefec27900d62d9469f36fac875b8f0b8c52bec3971c7a14f042a1c366e82``
(local) and
``1ee4c5384db632c55222f1a3b2d04d0f4e43e4c8252f2b4c795b9cc54a43f78c``
(CI), are retracted here, not silently overwritten. ``_SOURCE_REGIME_TABLE``
is untouched: none of the six importers is one of ``h-source-reader-v3``'s
seven fixed members (``core.sentence``, ``core.sentence.records``,
``core.sentence.segmentation``, ``core.sentence.corpus``,
``ladder.capture.capture``, ``kernel.artifacts``, ``kernel.lifecycle``),
confirmed by re-measuring both source rows unchanged via the same tool
invocation.

RETRACTION (house rule 4, p6-core, same commit): the checkpoint values
first computed above,
``da5d88e7d19ea7dcd4e5de3f84b20c23abdcc4c99a0ef7916b252bd9a52e34b6``
(local) and
``61281588e897462cfc9c8eae4cd0ae9217aaf9d0f0060044595278c6665dbaea``
(CI), were measured BEFORE ``just lint``'s ``ruff check --fix`` reordered
``ladder/run/span_diagnostics.py``'s two import lines (a reader-module
member) into isort order, which moved its bytes and therefore the composed
digest a SECOND time within this one commit -- caught by ``just test``
going RED after the gate's lint step ran. The values pinned in
``_CHECKPOINT_REGIME_TABLE`` above,
``992bbd0f199fc1efb3d089ce657d3381974a403bce1285f948bbdad4f4931be3``
(local) and
``f41275db894d927b39d9ff75124d77607a618a6e44bea0da41cb1947651d5758``
(CI), are the values measured AFTER lint, on the tree this commit actually
ships. Lesson recorded, not just fixed: a `_READER_MODULES` member's
regime row must be measured only after `just check`'s formatting/lint
step has run to a fixed point, never immediately after a hand rewrite.

p7b re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13, lane
mythwave/p7b-freeze-and-verify, FOR-1550): this phase extracts the checkpoint
reader's label into a named constant, ``ladder/run/checkpoint.py::
_READER_IDENTITY_LABEL``, and pins BOTH halves of the door here
(``_FROZEN_CHECKPOINT_READER_IDENTITY_LABEL``,
``_FROZEN_CHECKPOINT_READER_MODULES``). The label VALUE is unchanged
(``h-checkpoint-reader-v2``) and the 51-member tuple is unchanged, so the door
itself does not move -- but ``ladder.run.checkpoint`` is member 30 of its OWN
``_READER_MODULES``, so extracting the constant changes that member's bytes and
therefore the composed digest (ruling R-PIN: "frozen_after freezes label +
shape, not digest values"). Both rows recomputed with ``tools/regime_rows.py``
AFTER ``just fmt``/``just lint`` ran to a fixed point (the p6-core lesson
directly above; measured here: both reported no change, so the pre-lint and
post-lint values coincide this time). The OLD values they replace,
``992bbd0f199fc1efb3d089ce657d3381974a403bce1285f948bbdad4f4931be3``
(local) and
``f41275db894d927b39d9ff75124d77607a618a6e44bea0da41cb1947651d5758``
(CI), are retracted here, not silently overwritten. ``_SOURCE_REGIME_TABLE`` is
untouched: ``ladder.run.checkpoint`` is not one of h-source-reader-v3's seven
fixed members, confirmed by re-measuring both source rows unchanged with the
same tool invocation (``aa4702a2...`` local, ``5ea5ca80...`` CI).

pre-for-1567 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-09-28, p14 pre-phase rung FOR-1567, operator ruling 7
+ ruling R-PIN, design-gate ruling DG-A): h-checkpoint-reader-v2 changes
because seven of its 51 members changed bytes -- ``ladder.run.seals`` (seal
memo, incremental verification against the memoised parent),
``ladder.run.worker_snapshot`` (in-process publication ledger),
``ladder.run.worker_transport`` (bounded retry on verification READS only),
``ladder.run.runner`` / ``ladder.run.worker_execution`` /
``ladder.run.object_worker`` (per-leg publication timing carrier + the
``publication`` block on checkpoint events) and
``ladder.run.evaluation_records`` (memo passthrough, DG-B). Label and member
set unchanged (the door is the label and the shape). Both rows recomputed with
``tools/regime_rows.py`` AFTER ``ruff format --check`` reported a fixed point
(the p6-core lesson). The OLD values they replace,
``300334a9cce05c443e469e68ed1ce3bfd0ac008a2f68ed2f51403a0b2a3794c1``
(local) and
``c42a3d1adba0b4542abb66f6137be53ce5b6a8ca31ea7c00a8e1a576c4069a9f``
(CI), are retracted here, not silently overwritten. ``_SOURCE_REGIME_TABLE`` is
untouched: none of the seven touched members is one of h-source-reader-v3's
seven fixed members, confirmed by re-measuring both source rows unchanged with
the same tool invocation (``aa4702a2...`` local, ``5ea5ca80...`` CI).

pre-for-1617 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-09-29, p14 pre-phase rung FOR-1617, operator ruling 7
+ ruling R-PIN, design-gate ruling DG-A carried over): h-checkpoint-reader-v2
changes because two of its 51 members changed bytes --
``ladder.run.object_worker`` (stop-marker schema, encoder and classifier, the
bounded ``stop_requested`` poll at seal boundaries over the existing
``VerificationReadRetry`` fields, and the ``stopped`` terminal wiring) and
``ladder.run.worker_execution`` (the ``operator_stop`` poll composed with
``request.pause`` inside ``stop_after_seal``). ``ladder.run.runner`` needed no
byte change: it already exposed ``stop_after_seal`` once per seal boundary.
Label and member set unchanged. Both rows recomputed with
``tools/regime_rows.py`` AFTER ``ruff format --check`` reported a fixed point.
The OLD values they replace,
``ae100745d962684bfc82fc1f01e145b45045ad705185dd8bcbd64dab4b35e7d8``
(local) and
``37590887e88ca26e499a54453f16ad033af4f6f037a4cb55a8427836dd420b05``
(CI), are retracted here, not silently overwritten. ``_SOURCE_REGIME_TABLE`` is
untouched: neither member is one of h-source-reader-v3's seven fixed members,
confirmed by re-measuring both source rows unchanged with the same tool
invocation (``aa4702a2...`` local, ``5ea5ca80...`` CI).

pre-for-1567c re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-09-29, p14 pre-phase rung FOR-1621, operator ruling
DN-1567-C option C + ruling R-PIN, design-gate ruling DG-A carried over):
h-checkpoint-reader-v2 changes because two of its 51 members changed bytes --
``ladder.run.object_worker`` (one shared ``_publication`` block builder; the
final-snapshot leg now reuses the run's publication ledger and a fresh
``VerificationReadRetry`` and writes the ``publication`` block on every
terminal event) and ``ladder.run.worker_completion`` (``admit_publication``,
the single validator both readers reuse; the ``complete`` event admits the
optional ``publication`` key, read-old/write-new). ``h_probe_events`` is not
a member. Label and member set unchanged. Both rows recomputed with
``tools/regime_rows.py`` AFTER ``ruff format --check`` reported a fixed point.
The OLD values they replace,
``44052b3f34f8f16843b5a35094583929932dfeff6ab42b9f3d9202063c6ca127``
(local) and
``072ef7770347234d6ca3045b349d69a608e002783b7d7b41e4956c3e0098c215``
(CI), are retracted here, not silently overwritten. ``_SOURCE_REGIME_TABLE`` is
untouched: neither member is one of h-source-reader-v3's seven fixed members,
confirmed by re-measuring both source rows unchanged with the same tool
invocation (``aa4702a2...`` local, ``5ea5ca80...`` CI).

FOR-1670 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-10-20, experiment-pre-reqs wave lane FOR-1670, design-gate
ruling DG-A carried over, operator ruling "file it" + fanout 2026-10-20):
h-checkpoint-reader-v2 changes because three of its 51 members changed bytes --
``ladder.run.seals`` (``SealMemo`` gains a digest-keyed verified-evaluation map,
``get_evaluation`` / ``remember_evaluation`` and the ``evaluations_memo_hits``
counter; ``admission.end`` reports the hits beside the disk reads),
``ladder.run.evaluation_records`` (``read_evaluation`` serves a remembered
record on a memo hit and reads from disk otherwise; the disk-read counter and
the remember call run together after the whole verification) and
``ladder.run.runner`` (the roster loop's own ``evaluations_read`` increment is
removed, so the counter advances at exactly one site). Label and member set
unchanged. All three rows recomputed with ``tools/regime_rows.py`` AFTER
``ruff format --check`` reported a fixed point (828 files). The OLD values they
replace,
``b6f2b84b30c12ad0397d6ac0c394c33d978b7a80468bbbd823353841eea9f9f1``
(local), ``a2e29fd82341388dd005a228246dd97a1fe6a142ac57ae8ebdec6ad20c1e7a66``
(CI 3.12.14) and
``4f7d16f3270ec5ed331201ae23c3390efdf5da07e7f20ce67d9434ac59ebaa1b``
(CI 3.12.15), are retracted here, not silently overwritten.
``_SOURCE_REGIME_TABLE`` is untouched: none of the three members is one of
h-source-reader-v3's seven fixed members, confirmed by re-measuring all three
source rows unchanged with the same tool invocation (``9978611d...`` local,
``d85b3144...`` and ``fc7fad81...`` CI).

FOR-1664 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-10-21, experiment-pre-reqs serial sub-wave lane FOR-1664,
design-gate ruling DG-A carried over, operator door rulings A (ii) and B
2026-10-20, conductor rulings DN-1664-GOLDEN and DN-1664-FIXTURES):
h-checkpoint-reader-v2 changes because four of its 51 members changed bytes --
``core.sentence.profile`` (``RunSchedule`` gains the required last field
``diagnostic_samples``, the derived ``evaluated_samples`` tuple, the
development constants and the smoke-only run-class predicate; manifest
schema literal ``h-run-profile-v1`` becomes ``h-run-profile-v2``, v1 refused),
``ladder.run.evaluation_records`` (envelope and ``by_sample_count`` read the
derived tuple), ``ladder.run.execution`` (the evaluation seed is keyed by
step, ``(seed << 32) | step``; ``sample_counts`` reads the derived tuple) and
``ladder.run.trace_capture`` (the schema literal at its two sites). Label and
member set unchanged. All three rows recomputed with ``tools/regime_rows.py``
AFTER ``ruff format --check`` reported a fixed point (828 files). The OLD
values they replace,
``79f77aa2195ddc29672299926e70550333e76f07aef34c3a888f686a0728b852``
(local), ``19fcb786c23dd15e48245940790473d342921e0e606862bb6171871dbee8b6f9``
(CI 3.12.14) and
``60163f0f8f5392fdea4e36a07f2ac7795bfec54e1bf797396db80bdf1e28439b``
(CI 3.12.15), are retracted here, not silently overwritten.
``_SOURCE_REGIME_TABLE`` is untouched: none of the four members is one of
h-source-reader-v3's seven fixed members, confirmed by re-measuring all three
source rows unchanged with the same tool invocation (``9978611d...`` local,
``d85b3144...`` and ``fc7fad81...`` CI).

FOR-1664 L2-fix re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-10-21, experiment-pre-reqs serial sub-wave lane FOR-1664,
second DG-A hand-off after the lane's L2 gate): h-checkpoint-reader-v2 changes
because one of its 51 members changed bytes -- ``core.sentence.profile``
(``HProfile.__post_init__`` refuses ``seed >= 2**32`` because the step-keyed
evaluation seed ``(seed << 32) | step`` is injective and inside
``torch.manual_seed``'s range only below that bound, measured: the largest
in-range pair is accepted and ``2**32 << 32`` overflows; the tiny-name rule
reads ``SMOKE_RUN_CLASSES`` so the smoke-class set has one definition). Label
and member set unchanged. All three rows recomputed with
``tools/regime_rows.py`` AFTER ``ruff format --check`` reported a fixed point
(828 files). The OLD values they replace,
``ebbb2ee8f4dcbe264345457edfc17ab34e05d152164c9a9870ca96d6c394ab51``
(local), ``57292f2a3c0ab80b4ab6ada9cd418909eb849851632cc61728da23f7d36c75dc``
(CI 3.12.14) and
``57ad3ee6a8e730c47210644a15d8cdb22b5100dad96ffcef7ec5600c1c1bb8d4``
(CI 3.12.15), are retracted here, not silently overwritten.
``_SOURCE_REGIME_TABLE`` is untouched (``core.sentence.profile`` is not one of
h-source-reader-v3's seven fixed members; source rows re-measured unchanged).

FOR-1664 review-fix re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-10-21, experiment-pre-reqs serial sub-wave lane FOR-1664,
third DG-A hand-off after review round A, conductor ruling DN-1664-SEED):
h-checkpoint-reader-v2 changes because two of its 51 members changed bytes --
``ladder.run.execution`` (RETRACTION: the evaluation seed ``(seed << 32) | step``
claimed injectivity over (seed, step), but the CPU Mersenne Twister consumes
only the low 32 bits of ``torch.manual_seed``, measured: seeds 42 and 43 drew
identically, so the noise was a function of step alone; the seed is now
``(seed * 0x9E3779B1 + step) % 2**32``, a bijection on seed at a fixed step and
on step at a fixed seed, pure, pinned on a seed-by-step grid) and
``core.sentence.profile`` (the ``seed < 2**32`` guard's comment names the new
mixing and its aliasing bound). Label and member set unchanged. All three
rows recomputed with ``tools/regime_rows.py`` AFTER ``ruff format --check``
reported a fixed point (828 files). The OLD values they replace,
``91aed75258445b5e3d412353eb90a09cfedd8a8b796f444af386e78b65860143``
(local), ``e6a17011df9d58705fdf7d00243a526fad83f8dc646142705eec20da3ee73a0d``
(CI 3.12.14) and
``8893b5e40bfa86cf596ba3744c80d85905033b0f84724d2c8a0c7e2d84d4268f``
(CI 3.12.15), are retracted here, not silently overwritten.
``_SOURCE_REGIME_TABLE`` is untouched (neither member is one of
h-source-reader-v3's seven fixed members; source rows re-measured unchanged).

FOR-1673 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-10-23, wave C ``wf_3ce4ed9e-710`` lane FOR-1673,
design-gate ruling DG-A, gate ruling: schema bump ``h-mechanism-evidence-v1``
to ``-v2`` with a v1 read-old window; alternative recorded, not taken:
SCHEMA unchanged with additive keys): h-checkpoint-reader-v2 changes because
one of its 51 members changed bytes -- ``ladder.run.mechanism_evidence`` (a
zero-sentence prediction row's ``sentences`` entry is labelled
``status: structurally_absent, reason: no_sentence_states`` beside
``range: null``; the emitter writes v2 only; ``admit_mechanism`` accepts
exactly four shapes: v2 and v1 warm-up literals, v2 labelled and v1 bare-null
count-0 prediction entries, so the HEpP7g capture replays against its own
epoch). Label and member set unchanged. All three rows recomputed with
``tools/regime_rows.py`` after ``ruff format --check`` reported a fixed point.
The OLD values they replace,
``6b012375ecd8220652d82ed9f1c21328990659d9a4112fb961e015af34cb06e9``
(local), ``13df3a88f97d949e83207144e451c6ac6a8fa743045f9d174300b6c044cdc167``
(CI 3.12.14) and
``06092e3131f952a2be7ee6158f65e4fc7701ac5673a543182cf671f2ce03acb2``
(CI 3.12.15), are retracted here, not silently overwritten.
``_SOURCE_REGIME_TABLE`` is untouched: the member is not one of
h-source-reader-v3's seven fixed members, confirmed by re-measuring all three
source rows unchanged with the same tool invocation (``9978611d...`` local,
``d85b3144...`` and ``fc7fad81...`` CI).
Rebase onto the python 3.12.15 pin (house rule 4, MEASURED 2026-10-23, operator GO):
this re-pin also sets the local ``('3.12.15', '2.13.0')`` row added on main by the pin,
recomputed over THIS tree's reader members with ``tools/regime_rows.py``:
``c98a7c070a24509adad1004f560aaf0739b03a01a2c1d4e76192e592cf759a67``. The value
main carried, ``3cdea7f20c6725eb9dd459b9653fa5adf554751237be40c88499989cf02828aa``,
is main's reader set and is retracted here for this tree, not inherited.

FOR-1672 re-pin (house rule 4, MEASURED under .venv/bin/python 3.12.13,
conductor commit, 2026-10-23, wave C ``wf_3ce4ed9e-710`` lane FOR-1672,
design-gate ruling DG-A, gate ruling D1: the observed autocast state is
measured once at run start by a probe context opened with the exact
expression the execution sites use; alternative D2, observing at the first
execution step, recorded and not taken): h-checkpoint-reader-v2 changes
because two of its 51 members changed bytes -- ``core.sentence.runtime_precision``
(new sibling ``require_observed_execution_precision(profile)`` composes the
unchanged zero-argument guard and adds ``declared_precision`` and
``autocast = {enabled, dtype, device_type}`` to the record; the FP32-defaults
refusal is unchanged) and ``ladder.run.worker_execution`` (``_runtime()``
feeds the observed record into ``run_start.runtime.precision``).
``ladder.run.execution`` is byte-identical. Label and member set unchanged.
All three rows recomputed with ``tools/regime_rows.py`` after
``ruff format --check`` reported a fixed point. The OLD values they replace,
``6b012375ecd8220652d82ed9f1c21328990659d9a4112fb961e015af34cb06e9``
(local), ``13df3a88f97d949e83207144e451c6ac6a8fa743045f9d174300b6c044cdc167``
(CI 3.12.14) and
``06092e3131f952a2be7ee6158f65e4fc7701ac5673a543182cf671f2ce03acb2``
(CI 3.12.15), are retracted here, not silently overwritten.
``_SOURCE_REGIME_TABLE`` is untouched: neither member is one of
h-source-reader-v3's seven fixed members, confirmed by re-measuring all three
source rows unchanged with the same tool invocation (``9978611d...`` local,
``d85b3144...`` and ``fc7fad81...`` CI).
Rebase onto the python 3.12.15 pin (house rule 4, MEASURED 2026-10-23, operator GO):
this re-pin also sets the local ``('3.12.15', '2.13.0')`` row added on main by the pin,
recomputed over THIS tree's reader members with ``tools/regime_rows.py``:
``9c188cbdca1c6d12c635c95371de07599e9ad1e4ec10354e4d63f2a1dfc58ed1``. The value
main carried, ``3cdea7f20c6725eb9dd459b9653fa5adf554751237be40c88499989cf02828aa``,
is main's reader set and is retracted here for this tree, not inherited.

Zip restack (house rule 4, MEASURED 2026-10-24 by the conductor, wave C zip, stack
1671, 1673, 1672, 1674 on PR #380 59ddc948; level 3 of 4): FOR-1672's change set now
sits on FOR-1673, so this tree's reader members carry every lane below it. The four
checkpoint rows were recomputed with ``tools/regime_rows.py`` over THIS tree:
(3.12.13, 2.13.0)
``1c5ab1c684346db048e08b8e4d8c81e8eda94a7c888924ff243b191c0f63dff4``, (3.12.14,
2.13.0+cpu) ``772843ac05e6e1d0a91f55f597276c1bcc9297417db0837dcda940feb7bfc985``,
(3.12.15, 2.13.0)
``32d2f51cf6d44f23a35c3cc2bfc8ce8ab4845965572d475c628efd29636bb795``, (3.12.15,
2.13.0+cpu) ``a6f5ba48be57197b03321c17f7f8d5b99f3eebebae2c6aa3b6660198632a1161``. The
lane-local values FOR-1672 pinned on the bare wave base ((3.12.13, 2.13.0)
``dfa86d64``, (3.12.14, 2.13.0+cpu) ``916f62e8``, (3.12.15, 2.13.0) ``9c188cbd``,
(3.12.15, 2.13.0+cpu) ``b7514208``) and the FOR-1673 level values this commit's parent
carries ((3.12.13, 2.13.0) ``96f2512f``, (3.12.14, 2.13.0+cpu) ``e03d4f8d``, (3.12.15,
2.13.0) ``c98a7c07``, (3.12.15, 2.13.0+cpu) ``1f901dd7``) are retracted for this tree,
not inherited. Source rows re-measured unchanged.

FOR-1674 re-pin (house rule 4, MEASURED 2026-10-23, conductor commit, wave C
``wf_3ce4ed9e-710`` lane FOR-1674, design-gate ruling DG-A; gate ruling G2, a
``torch.random.fork_rng`` island per continuation seeded once with
``(profile.seed * 0x9E3779B1 + ((schedule.steps << 8) | i)) % 2**32``; G1, an explicit
generator threaded through ``advance_rollout``, recorded as the alternative and
rejected because ``Beta.rsample`` takes no generator; operator ruling D1
``DEVELOPMENT_GENERATION_CONTINUATIONS = 5``): h-checkpoint-reader-v2 changes because
three of its 51 members changed bytes -- ``core.sentence.profile``
(``RunSchedule.generation_continuations``, schema ``h-run-profile-v2`` -> ``-v3``),
``ladder.run.generation`` (receipt ``h-generation-receipt-v3`` with ``continuations``
and a top-level ``sampling`` block; v1/v2 receipts refused by name) and
``ladder.run.trace_capture`` (schema literals). Label and member set unchanged. All
FOUR checkpoint rows recomputed with ``tools/regime_rows.py`` over this tree after a
``ruff format --check`` fixed point, matching the lane's independent measurement:
``4573549b...`` (3.12.13 local), ``f598741b...`` (CI 3.12.14), ``eb6f0288...`` (CI
3.12.15) and ``05f60906...`` (3.12.15 local). The OLD values they replace,
``6b012375...``, ``13df3a88...``, ``06092e31...`` and ``3cdea7f2...``, are retracted
here, not silently overwritten. ``_SOURCE_REGIME_TABLE`` is untouched (none of the
three members is a h-source-reader-v3 member; source rows re-measured unchanged).

Zip restack (house rule 4, MEASURED 2026-10-24 by the conductor, wave C zip, stack
1671, 1673, 1672, 1674 on PR #380 59ddc948; level 4 of 4): FOR-1674's change set now
sits on FOR-1672, so this tree's reader members carry every lane below it. The four
checkpoint rows were recomputed with ``tools/regime_rows.py`` over THIS tree:
(3.12.13, 2.13.0)
``f518ccbf3c7e54fb5c8b04ed55f9eaccc740a23d377e3787b8fc31c06bbe751e``, (3.12.14,
2.13.0+cpu) ``a284e8fd544b321201e5cf37dbcf17a03f3985bed7cfd285b693649c49512e1a``,
(3.12.15, 2.13.0)
``5d28628730388bfd64dcdeb511f7dfac91d7043eeb56a4c431dd95e3cee1ebc0``, (3.12.15,
2.13.0+cpu) ``d45302eb616689b61e03404f1d8b98862c9dc58b558df6ffa78ebe4f83e0266b``. The
lane-local values FOR-1674 pinned on the bare wave base ((3.12.13, 2.13.0)
``4573549b``, (3.12.14, 2.13.0+cpu) ``f598741b``, (3.12.15, 2.13.0) ``05f60906``,
(3.12.15, 2.13.0+cpu) ``eb6f0288``) and the FOR-1672 level values this commit's parent
carries ((3.12.13, 2.13.0) ``1c5ab1c6``, (3.12.14, 2.13.0+cpu) ``772843ac``, (3.12.15,
2.13.0) ``32d2f51c``, (3.12.15, 2.13.0+cpu) ``a6f5ba48``) are retracted for this tree,
not inherited. Source rows re-measured unchanged.

Review round 1 re-pin (house rule 4, MEASURED 2026-10-25 by the conductor, follow-up
branch ``fix/wave-c-review-r1`` on FOR-1674 ``3f563f89``): PR #385 review (CodeRabbit
4369526052) found that ``RunSchedule`` admitted ``steps >= 2**24`` although
``continuation_seed`` carries only 24 step bits, so the refusal came after training.
``core.sentence.profile`` gains ``MAX_RUN_STEPS = 2**24 - 1`` checked in
``RunSchedule.__post_init__``, and ``ladder.run.generation`` reads the same constant;
label and member set unchanged, schema ids unchanged. The four checkpoint rows were
recomputed with ``tools/regime_rows.py`` over THIS tree: (3.12.13, 2.13.0)
``25140936058fe296772af5875138d1195a0b674c7786de8af315f5b13f89a2f7``, (3.12.14,
2.13.0+cpu) ``2ef403d110c334a3dbc81ed53897df6b1fc6f1a25e48849a46e06ffcbb3c6063``,
(3.12.15, 2.13.0)
``fd307002c6904b101d79a65d21568688deab159306bf1246b76a5888f551b6ff``, (3.12.15,
2.13.0+cpu) ``941f3aef0d9c2542419e6d775c9374e3636a322fda4f4704bd5bce27cb3864a3``. The
FOR-1674 level values they replace ((3.12.13, 2.13.0) ``f518ccbf``, (3.12.14,
2.13.0+cpu) ``a284e8fd``, (3.12.15, 2.13.0) ``5d286287``, (3.12.15, 2.13.0+cpu)
``d45302eb``) are retracted for this tree, not silently overwritten. Source rows
re-measured unchanged.

Stop-at-seal re-pin (house rule 4, MEASURED 2026-10-26, branch
``feat/h-probe-stop-at-seal`` on ``fix/runpodctl-3.0.1``): the operator stop marker
gains an optional ``target`` ``{phase, step}`` so a stop published early lands at a named
seal (wave h0-final-probe-1k-v4.92c9db: the verb had no way to name a seal and the
5-minute window between two polls was missed). h-checkpoint-reader-v2 changes because
two of its members changed bytes -- ``ladder.run.object_worker`` (marker encoder,
classifier, ``stop_marker_hits``, the poll now receives the seal) and
``ladder.run.worker_execution`` (``operator_stop`` is called with ``(phase, step)``).
Label and member set unchanged. The four checkpoint rows were recomputed with
``tools/regime_rows.py`` after a ``ruff format`` fixed point: (3.12.13, 2.13.0)
``5b4bbf26...``, (3.12.14, 2.13.0+cpu) ``4bba1114...``, (3.12.15, 2.13.0) ``77ba4f8b...``,
(3.12.15, 2.13.0+cpu) ``b60b61b7...``. The values they replace (``25140936``,
``2ef403d1``, ``fd307002``, ``941f3aef``) are retracted for this tree, not silently
overwritten. Source rows are untouched (neither member is an h-source-reader-v3 member).

Stop-at-seal follow-up re-pin (house rule 4, MEASURED 2026-10-26, same branch, PR #391 review):
``ladder.run.object_worker`` now refuses a stop target outside the request schedule at encode
time (CodeRabbit finding), so its bytes moved again and the four checkpoint rows were recomputed
with ``tools/regime_rows.py``. Retracted for this tree, the 92af16b6 values: (3.12.13, 2.13.0) ``5b4bbf26...``, (3.12.14, 2.13.0+cpu) ``4bba1114...``, (3.12.15, 2.13.0) ``77ba4f8b...``, (3.12.15, 2.13.0+cpu) ``b60b61b7...``. Source rows unchanged.

Stop-at-seal review round 2 re-pin (house rule 4, MEASURED 2026-10-26, PR #391): the target
check now applies the full seal-boundary rule of ``ProbeRequest.pause`` (an in-budget non-seal
step is refused; parity pinned by a grid test), so ``ladder.run.object_worker`` moved again.
Retracted for this tree, the df51f73e values: (3.12.13, 2.13.0) ``fd4d9f47...``, (3.12.14, 2.13.0+cpu) ``53854d6b...``, (3.12.15, 2.13.0) ``283bcaad...``, (3.12.15, 2.13.0+cpu) ``1bbb94b9...``. Source rows unchanged.

Gradient-guard re-pin (house rule 4, MEASURED 2026-10-27, branch ``fix/grad-reconciliation-guard``
on main 1228d088; lane grad-guard-fix handoff, conductor DG-A commit): ``ladder.run.training``
``apply_sentence_step`` replaces the ``gradient_reconciliation`` norm comparison (an fp32
``clip_grad_norm_`` aggregate against a float64 hypot at rel_tol 1e-5, which was tautological
once ``expected`` derives from the roles and false-positive at development widths: rel gap
3.253e-4 at 15.36M parameters over 100 tensors, measured; 6.85e-6 at step 1 on the dev-width
profile) with exact set identity: no parameter in two roles (``optimizer_ownership``) and the
set of parameters carrying gradients equal to the role set (``gradient_reconciliation``), so a
tied or stray parameter with a zero gradient is now seen. Defect names unchanged; receipts
unchanged. One member moved bytes; label and 52-member set unchanged. The five checkpoint rows
were recomputed with ``tools/regime_rows.py`` over the staged tree: (3.12.13, 2.13.0)
``65209d55...``, (3.12.14, 2.13.0+cpu) ``a0320b1f...``, (3.12.15, 2.13.0) ``283935bd...``
(equal to the lane's independently measured digest), (3.12.15, 2.13.0+cpu) ``bbf6aa86...``,
(3.12.16, 2.13.0+cpu) ``c692a7b6...``. The values they replace (``7a02d741``, ``b1ee8ce8``,
``5b0daf50``, ``d6e7edc8``, ``1ba2025e``) are retracted for this tree, not silently overwritten.
Source rows are untouched.

Admission-port re-pin (house rule 4, MEASURED 2026-10-27, branch ``pilot-admission-repin`` on
``pilot-admission-port-25e262`` c689b1c6, PR #396; conductor DG-A commit under the operator
ruling "Handoff route + amend the door"): every H run-class gate now reads ONE admission port,
``ladder.run.admission`` (class, stage, surface, cost-guard policy as a frozen table; rung A of
the pilot-admission stack, behaviour unchanged, 389-case site characterization green before and
after). The port joins ``_READER_MODULES`` and the frozen tuple (51 to 52 members) because the
admitted-class policy is part of what a checkpoint reader proves, and five members moved bytes
(``ladder.run.worker_request``, ``runner``, ``worker_command``, ``worker_execution``,
``worker_completion``), plus ``checkpoint.py`` itself. Label unchanged. The four checkpoint rows
were recomputed with ``tools/regime_rows.py`` over the staged tree: (3.12.13, 2.13.0)
``e922ca52...``, (3.12.14, 2.13.0+cpu) ``82bf09bc...``, (3.12.15, 2.13.0) ``a6e5353f...``,
(3.12.15, 2.13.0+cpu) ``563afc50...``. The values they replace (``7a02d741``, ``b1ee8ce8``,
``5b0daf50``, ``d6e7edc8``) are retracted for this tree, not silently overwritten. Rebased onto
main d59bd588 after #395 and #396 squash-merged (2026-10-27): the four rows held (no reader
member differs between c689b1c6 and that main) and the 3.12.16 row that #399 added on the
pre-port tree was recomputed here: (3.12.16, 2.13.0+cpu) ``21d022bb...``, replacing
``1ba2025e``. Source rows are untouched (no h-source-reader-v3 member changed).

Rebase of the admission-port re-pin onto main 66d5feac (house rule 4, MEASURED 2026-10-28,
branch ``pilot-admission-repin``, PR #400; conductor rebase under the operator ruling "rebase
onto main, gt submit --publish, skip ship-it for this PR"): main had meanwhile taken #402's
gradient-guard re-pin (one member moved bytes: ``ladder.run.training``) and the 0.8.0 release,
so the two 2026-10-27 entries above now compose in one tree: the 52-member tuple with the
admission port AND the set-identity gradient guard. The five checkpoint rows were recomputed
with ``tools/regime_rows.py`` over the resolved tree: (3.12.13, 2.13.0) ``5f992472...``,
(3.12.14, 2.13.0+cpu) ``199b1cbb...``, (3.12.15, 2.13.0) ``28c8cb5a...``, (3.12.15, 2.13.0+cpu)
``9f6d6de2...``, (3.12.16, 2.13.0+cpu) ``fcc64eda...``. Retracted for this tree: main's guard-only
values (``65209d55``, ``a0320b1f``, ``283935bd``, ``bbf6aa86``, ``c692a7b6``) and the branch's
admission-only values (``e922ca52``, ``82bf09bc``, ``a6e5353f``, ``563afc50``, ``21d022bb``).
Source rows are untouched (no h-source-reader-v3 member changed; recomputed and equal).

p2 retraction (house rule 4, MEASURED 2026-09-20): the D1 closure manifest
row (``_FROZEN_D1_CLOSURE_COUNT``, ``_FROZEN_D1_CLOSURE_MANIFEST_DIGEST``,
``_closure_manifest_digest()`` and their pinning test) is DROPPED here, not
re-pinned. ``analysis/d1_reader_source_closure.py`` -- the module this row
measured -- is retired outright this phase (operator ruling E-1); `git grep`
over tracked source confirms zero remaining importers of
``installed_d1_reader_source_files`` once this row is removed. This is the
one D1 closure manifest row this phase's ruled bump touches; every other
fixture value below stays byte-identical to p0/p1.
CI regime row added (house rule 4, MEASURED 2026-10-02): the GitHub runners
resolve mise's floating ``python = "3.12"`` to CPython 3.12.15 since
2026-10-02 (run 37006785236 on PR #323, and the same signature on #320-#322),
while the last green trunk run (36870794433 at b11855d, 2026-10-01) installed
3.12.14, so ``h-checkpoint-reader-v2 has no fixture row for regime
('3.12.15', '2.13.0+cpu')`` failed loud by name as this fixture is designed
to. No reader member changed: the 3.12.14 rows are RETAINED (that regime is
still real), and the 3.12.15 rows are ADDED, computed with
``tools/regime_rows.py --python 3.12.15 --torch 2.13.0+cpu --pysbd 0.3.4``
(the same instrument that produced the 3.12.14 rows, run on the macOS
checkout; the first CI run at this tip is the live confirmation). The
source-derived assertions below are untouched. The floating patch pin is the
root cause and is recorded as a separate decision for the toolchain owner.
p14-z-identity RE-PIN (ruling R-PIN, design-gate DG-A, MEASURED 2026-10-07):
the identity rail's ONE conductor commit carries the lane's staged change
set for FOR-1589 (a dir_fd descent helper owning the descriptor lifecycle
in ``kernel/artifacts.py``), FOR-1588 (one artifact-directory helper for
the five CLI sites) and FOR-1596 (shared score-gain summary arithmetic,
moved out of the reader-hashed bytes of ``core.probe.score_gain`` into
``core/probe/predictive_score_gain.py``, which is not a member). 6 of the
51 checkpoint members changed bytes (``composition.capture_cli``,
``composition.cli``, ``composition.source_cli``, ``composition.span_cli``,
``core.probe.score_gain``, ``kernel.artifacts``) and 1 of the 7 source
members (``kernel.artifacts``); both labels and both module sets are
unchanged. All six rows were re-measured with ``tools/regime_rows.py`` on
the staged tree after ``ruff format --check`` reported a fixed point, and
cross-checked by the lane with plain hashlib over the member files. The
previous values (checkpoint 15ca5961 / 5bc5602e / 18823965, source
aa4702a2 / 5ea5ca80 / 904eaa93) are retracted here, not silently
overwritten (house rule 4). Both regime-row tests were RED on the staged
tree before this re-pin (the brief's "type a digest instead of measuring
it" arm, measured 2 failed / 5909 passed) and are GREEN after it.
Local regime row added (house rule 4, MEASURED 2026-10-23, operator ruling the
same day): mise's floating ``python = "3.12"`` began resolving to CPython 3.12.15
on the operator's machine on 2026-10-23 (a machine-wide toolchain update), so a
venv built after that date ran ``h-checkpoint-reader-v2 has no fixture row for
regime ('3.12.15', '2.13.0')`` and failed loud by name, exactly as this fixture
is designed to. The operator ruled to pin ``mise.toml`` to the exact patch
``python = "3.12.15"`` (the floating pin is the root cause the CI paragraph above
already named) and to add the one missing row rather than hold the old patch.
No reader member changed and no existing row moves: the 3.12.13 rows are
RETAINED (venvs and local artifacts made under that regime stay readable), and
``_SOURCE_REGIME_TABLE`` already carried ``('3.12.15', '0.3.4')`` from the CI
paragraph. The added value was measured three ways under ``.venv/bin/python``
3.12.15 with torch 2.13.0 and all agree: ``checkpoint._reader_identity()`` live,
``tools/regime_rows.py --python 3.12.15 --torch 2.13.0 --pysbd 0.3.4``, and
that tool's control (fed the 3.12.13 strings it reproduces the pinned 3.12.13
row exactly; its 3.12.15 source row equals the already-pinned value). The full
unit suite under that venv was 1 regime failure plus 2 packet-assembly
failures that are an instrument artifact (the assembler shells out to
``ruff``, absent from PATH when the venv's python is invoked directly; both
pass with the venv on PATH, as every repo gate runs them).
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch

from entropic_substrate.core.observation.byte_codec import ByteObservationCodec
from entropic_substrate.core.observation.permuted_byte_codec import PermutedByteObservationCodec
from entropic_substrate.kernel.runtime.checkpoint import _TAU_ERAS, CHECKPOINT_FORMAT_VERSION
from entropic_substrate.ladder.capture.capture import (
    _READER_MODULES as _SOURCE_READER_MODULES,
)
from entropic_substrate.ladder.capture.capture import (
    _reader_identity as _source_reader_identity,
)
from entropic_substrate.ladder.run.checkpoint import (
    _READER_IDENTITY_LABEL as _CHECKPOINT_READER_IDENTITY_LABEL,
)
from entropic_substrate.ladder.run.checkpoint import (
    _READER_MODULES as _CHECKPOINT_READER_MODULES,
)
from entropic_substrate.ladder.run.checkpoint import (
    _reader_identity as _checkpoint_reader_identity,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Source-derived components (regime-free): frozen at p0, byte-exact -- except
# the D1 closure count/digest below, re-pinned at p1 (see module docstring).
# ---------------------------------------------------------------------------

_FROZEN_SOURCE_READER_MODULES = (
    "core.sentence",
    "core.sentence.records",
    "core.sentence.segmentation",
    "core.sentence.corpus",
    "ladder.capture.capture",
    "kernel.artifacts",
    "kernel.lifecycle",
)

# FOR-1550: the h-checkpoint-reader-v2 DOOR itself -- label + module-set shape,
# the two things docs/identity/read-old-table.md freezes after p4 (ruling R-PIN:
# "frozen_after freezes label + shape, not digest values"). Until this phase the
# checkpoint reader's label lived only as an inline literal inside
# _reader_identity() and its 51-member tuple had no frozen counterpart here, so
# the source reader's shape was pinned and the checkpoint reader's was not.
_FROZEN_CHECKPOINT_READER_IDENTITY_LABEL = "h-checkpoint-reader-v2"

_FROZEN_CHECKPOINT_READER_MODULES = (
    "composition.capture_cli",
    "composition.cli",
    "composition.source_cli",
    "composition.span_cli",
    "core.latent.dynamics",
    "core.latent.metric",
    "core.probe.score_gain",
    "core.sentence",
    "core.sentence.boundary",
    "core.sentence.codec",
    "core.sentence.consumer",
    "core.sentence.corpus",
    "core.sentence.head",
    "core.sentence.law",
    "core.sentence.model",
    "core.sentence.numerics",
    "core.sentence.objective",
    "core.sentence.profile",
    "core.sentence.records",
    "core.sentence.recovery_state",
    "core.sentence.rollout",
    "core.sentence.runtime_precision",
    "core.sentence.sampling",
    "core.sentence.scoring",
    "core.sentence.segmentation",
    "core.sentence.spine",
    "kernel.artifacts",
    "kernel.lifecycle",
    "ladder.capture.capture",
    "ladder.run.admission",
    "ladder.run.checkpoint",
    "ladder.run.evaluation_records",
    "ladder.run.execution",
    "ladder.run.generation",
    "ladder.run.journal",
    "ladder.run.mechanism_evidence",
    "ladder.run.native_verification",
    "ladder.run.object_worker",
    "ladder.run.runner",
    "ladder.run.seals",
    "ladder.run.span_diagnostics",
    "ladder.run.span_probe",
    "ladder.run.span_probe_artifacts",
    "ladder.run.span_states",
    "ladder.run.trace_capture",
    "ladder.run.training",
    "ladder.run.worker_command",
    "ladder.run.worker_completion",
    "ladder.run.worker_execution",
    "ladder.run.worker_request",
    "ladder.run.worker_snapshot",
    "ladder.run.worker_transport",
)

_FROZEN_CHECKPOINT_FORMAT_VERSION = 1
_FROZEN_TAU_ERAS = ("softplus", "temperature")

_FROZEN_BYTE_CODEC_IDENTITY_LABEL = "identity-byte-v1"
_FROZEN_BYTE_CODEC_CARDINALITY = 256
_FROZEN_BYTE_CODEC_CONSTRUCTION_DIGEST = (
    "b1f2d8afd749dd6570dc043358cfa5c97d674b8812df095b4d2959068ebaf89c"
)

_FROZEN_PERMUTED_BYTE_CODEC_LABEL = "permuted-byte-v1"
_FROZEN_PERMUTED_BYTE_CODEC_SEED = 0
_FROZEN_PERMUTED_BYTE_CODEC_CONSTRUCTION_DIGEST = (
    "b1b23d918cb4b13b763a7b8243508972d9a905a36e21810407b35fba02de58f5"
)

# ---------------------------------------------------------------------------
# Regime table: composed digest per (measured environment inputs) -> digest.
# Each row names EXACTLY the inputs that identity hashes (measured, not the
# same key for both readers -- see ruling above).
# ---------------------------------------------------------------------------

_CHECKPOINT_REGIME_TABLE: dict[tuple[str, str], str] = {
    (
        "3.12.13",
        "2.13.0",
    ): "5f992472f4d81757b9d0fd60f80dddb79560519b53ee7569437943bc26cfab4d",  # pragma: allowlist secret
    (
        "3.12.14",
        "2.13.0+cpu",
    ): "199b1cbba33cf6f066119e1139d32676666aa9b9dd1a4b536af69199f58f73df",  # pragma: allowlist secret
    (
        "3.12.15",
        "2.13.0",
    ): "28c8cb5a5377b76ac8c5d2cb8550ad37fe1b5a615a658e1445541df8c0d53944",  # pragma: allowlist secret
    (
        "3.12.15",
        "2.13.0+cpu",
    ): "9f6d6de2fad74516d5269c47285ce874e719ffd046e2c7694c05eab92d4b2703",  # pragma: allowlist secret
    # MEASURED 2026-10-27: the floating `uv python install 3.12` in ci.yaml resolved CPython
    # 3.12.16 on the runners after the Actions budget was restored (runs 40519114224,
    # 40538710441, 40540752238), the third drift of this kind; ci.yaml now pins 3.12.15 to
    # match mise, and this row keeps the 3.12.16 regime real for anyone running that patch.
    # Computed with tools/regime_rows.py --python 3.12.16 --torch 2.13.0+cpu --pysbd 0.3.4,
    # whose control reproduced the two 3.12.15 rows above exactly.
    (
        "3.12.16",
        "2.13.0+cpu",
    ): "fcc64edad6a4d711ee41645c2298855dfd446a312a95704d0ba32bfdfe8250a3",  # pragma: allowlist secret
}

_SOURCE_REGIME_TABLE: dict[tuple[str, str], str] = {
    (
        "3.12.13",
        "0.3.4",
    ): "9978611df372f0b21e9ee7317b78a61859c4d94a8acef9d3dab3f8d93161e231",  # pragma: allowlist secret
    (
        "3.12.14",
        "0.3.4",
    ): "d85b31442840d6e4c466f00e170d976d3621956a174089c645fcb065f1ba5fbc",  # pragma: allowlist secret
    (
        "3.12.15",
        "0.3.4",
    ): "fc7fad81f3adb884236cb7918ea7c13966f71127e17de3b5aa8040ba633dda68",  # pragma: allowlist secret
    # MEASURED 2026-10-27: companion of the 3.12.16 checkpoint row above, same instrument run.
    (
        "3.12.16",
        "0.3.4",
    ): "23da6f3e89b348572be0b137cb6ad75fbfbb66c0341328fc93821d116cff9fbb",  # pragma: allowlist secret
}


class TestSourceDerivedComponentsFrozenAtP0:
    """These assertions carry no regime key -- they are pure source facts."""

    def test_source_reader_module_tuple_is_frozen(self) -> None:
        assert _SOURCE_READER_MODULES == _FROZEN_SOURCE_READER_MODULES

    def test_checkpoint_format_version_is_frozen(self) -> None:
        assert CHECKPOINT_FORMAT_VERSION == _FROZEN_CHECKPOINT_FORMAT_VERSION

    def test_tau_eras_are_frozen(self) -> None:
        assert _TAU_ERAS == _FROZEN_TAU_ERAS

    def test_byte_codec_construction_digest_is_frozen(self) -> None:
        codec = ByteObservationCodec()
        assert codec.construction_label == _FROZEN_BYTE_CODEC_IDENTITY_LABEL
        assert codec.cardinality == _FROZEN_BYTE_CODEC_CARDINALITY
        assert codec.construction_digest == _FROZEN_BYTE_CODEC_CONSTRUCTION_DIGEST

    def test_permuted_byte_codec_construction_digest_is_frozen_for_seed_zero(self) -> None:
        codec = PermutedByteObservationCodec.from_seed(_FROZEN_PERMUTED_BYTE_CODEC_SEED)
        assert codec.construction_label == _FROZEN_PERMUTED_BYTE_CODEC_LABEL
        assert codec.construction_digest == _FROZEN_PERMUTED_BYTE_CODEC_CONSTRUCTION_DIGEST


class TestComposedDigestsPerMeasuredRegime:
    """The composed H digests key on DIFFERENT regime inputs (measured):
    checkpoint reader keys on (python, torch); source reader keys on
    (python, pysbd). Neither shares a key with the other."""

    def test_checkpoint_reader_label_and_module_set_are_frozen(self) -> None:
        """Carries NO regime key, deliberately, and sits next to the digest row
        it governs (FOR-1550): the door frozen after p4 is the LABEL and the
        module-set SHAPE, while the composed digest below re-measures on every
        phase that edits a member's bytes (ruling R-PIN). Pinning the two here
        is what makes a silent label bump or a dropped/added member a named
        failure instead of a digest mismatch indistinguishable from an
        ordinary byte move."""
        assert _CHECKPOINT_READER_IDENTITY_LABEL == _FROZEN_CHECKPOINT_READER_IDENTITY_LABEL
        assert _CHECKPOINT_READER_MODULES == _FROZEN_CHECKPOINT_READER_MODULES

    def test_checkpoint_reader_digest_matches_measured_regime_row(self) -> None:
        regime = (platform.python_version(), str(torch.__version__))
        expected = _CHECKPOINT_REGIME_TABLE.get(regime)
        if expected is None:
            pytest.fail(
                f"h-checkpoint-reader-v2 has no fixture row for regime {regime!r}. "
                "Add the row: measure _reader_identity() under this exact "
                "(python_version, torch.__version__) pair and pin it in "
                "_CHECKPOINT_REGIME_TABLE -- do not skip this test."
            )
        assert _checkpoint_reader_identity() == expected

    def test_source_reader_digest_matches_measured_regime_row(self) -> None:
        regime = (platform.python_version(), metadata.version("pysbd"))
        expected = _SOURCE_REGIME_TABLE.get(regime)
        if expected is None:
            pytest.fail(
                f"h-source-reader-v3 has no fixture row for regime {regime!r}. "
                "Add the row: measure _reader_identity() under this exact "
                "(python_version, pysbd version) pair and pin it in "
                "_SOURCE_REGIME_TABLE -- do not skip this test."
            )
        assert _source_reader_identity() == expected


class TestUnmeasuredRegimeFailsLoudNeverSkips:
    """Isolated from the frozen assertions above (house rule: an unmeasured
    regime's failure must never mask or block the source-derived pins)."""

    def test_checkpoint_regime_absent_from_table_raises_named_failure(self) -> None:
        bogus_regime = ("9.9.9", "0.0.0-does-not-exist")
        assert bogus_regime not in _CHECKPOINT_REGIME_TABLE
        with pytest.raises(AssertionError):
            expected = _CHECKPOINT_REGIME_TABLE.get(bogus_regime)
            assert expected is not None, (
                f"h-checkpoint-reader-v2 has no fixture row for regime {bogus_regime!r}"
            )

    def test_source_regime_absent_from_table_raises_named_failure(self) -> None:
        bogus_regime = ("9.9.9", "0.0.0-does-not-exist")
        assert bogus_regime not in _SOURCE_REGIME_TABLE
        with pytest.raises(AssertionError):
            expected = _SOURCE_REGIME_TABLE.get(bogus_regime)
            assert expected is not None, (
                f"h-source-reader-v3 has no fixture row for regime {bogus_regime!r}"
            )


# ---------------------------------------------------------------------------
# D1, mapping coverage (architect ruling, generalized 2026-09-19): tools/mapping.json's
# coverage assertion is graph-nodes SUBSET-OF mapping-entries, one directional, never set
# equality. Admissible exclusions (ruling): (1) any subpackage of a landed layer
# (`entropic_substrate.kernel`, `entropic_substrate.kernel.runtime`, etc.), derived at
# test time from mapping.json's `layers` key; (2) any node that after stripping exactly
# one `entropic_substrate.__legacy.` prefix resolves to (1) or is a mapping entry.
# Never a hand-written ignore list, so a sixth unmapped top-level package outside
# layers still goes RED while the five ruled layer markers do not.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _graph_module_names() -> set[str]:
    tools_dir = _REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from import_graph import build_graph  # local import: tools/ is not a package

    mods, _edges = build_graph(_REPO_ROOT / "src" / "entropic_substrate", "entropic_substrate")
    return set(mods.keys())


def _mapping() -> dict:
    with (_REPO_ROOT / "tools" / "mapping.json").open() as f:
        return json.load(f)


def _is_layer_or_subpackage(node: str, layers: set[str]) -> bool:
    """Check if node is a landed layer marker or a subpackage of one."""
    for layer in layers:
        layer_prefix = f"entropic_substrate.{layer}"
        if node == layer_prefix or node.startswith(f"{layer_prefix}."):
            return True
    return False


def _canonicalize_node(node: str) -> str:
    """Strip exactly one leading __legacy. segment if present."""
    legacy_prefix = "entropic_substrate.__legacy."
    if node.startswith(legacy_prefix):
        return "entropic_substrate." + node[len(legacy_prefix) :]
    return node


class TestMappingCoversEveryGraphNodeAtThisEpoch:
    def test_mapping_is_committed_with_zero_edits_to_it(self) -> None:
        mapping = _mapping()
        assert set(mapping.keys()) >= {"layers", "modules"}
        assert mapping["layers"] == ["kernel", "core", "ladder", "historical", "composition"]

    def test_graph_nodes_are_a_subset_of_mapping_entries_modulo_layer_markers(self) -> None:
        mapping = _mapping()
        mapping_entries = set(mapping["modules"].keys())
        layers = set(mapping["layers"])
        graph_nodes = _graph_module_names()

        # A node is covered if:
        # 1. It is in the mapping, OR
        # 2. It is a layer marker or subpackage of one, OR
        # 3. It is the bare quarantine root (entropic_substrate.__legacy), OR
        # 4. After stripping one __legacy. prefix, it meets condition 1 or 2
        uncovered = set()
        for node in graph_nodes:
            # Check condition 1: in mapping
            if node in mapping_entries:
                continue
            # Check condition 2: is a layer or subpackage
            if _is_layer_or_subpackage(node, layers):
                continue
            # Check condition 3: bare quarantine root is always covered
            if node == "entropic_substrate.__legacy":
                continue
            # Check condition 4: strip __legacy. and recheck
            canonical = _canonicalize_node(node)
            if canonical != node:
                if canonical in mapping_entries or _is_layer_or_subpackage(canonical, layers):
                    continue
            uncovered.add(node)

        assert not uncovered, f"graph nodes not covered by tools/mapping.json: {sorted(uncovered)}"

    def test_exclusion_set_is_derived_from_the_layers_key_not_hand_written(self) -> None:
        """The base layer-marker set that seeds `_is_layer_or_subpackage` and
        `_canonicalize_node` is exactly the five entries of `layers`, no
        more -- both helpers are derived from this set at test time, never a
        hand-written ignore list, so a sixth unmapped top-level package
        (with or without a `__legacy.` prefix) still goes RED."""
        mapping = _mapping()
        layers = set(mapping["layers"])
        excluded = {f"entropic_substrate.{layer}" for layer in layers}
        assert excluded == {
            "entropic_substrate.kernel",
            "entropic_substrate.core",
            "entropic_substrate.ladder",
            "entropic_substrate.historical",
            "entropic_substrate.composition",
        }
        # A synthetic sixth "unmapped package" is NOT in this set, so it
        # would surface as uncovered rather than being silently absorbed --
        # neither directly nor via a __legacy. strip.
        assert "entropic_substrate.probe_unmapped_sixth_package" not in excluded
        assert not _is_layer_or_subpackage(
            "entropic_substrate.probe_unmapped_sixth_package", layers
        )
        # A subpackage of a landed layer IS covered, one level deep.
        assert _is_layer_or_subpackage("entropic_substrate.kernel.probe", layers)
        # Exactly one leading __legacy. segment is stripped -- a second
        # occurrence is left in place, never recursively unwrapped.
        assert (
            _canonicalize_node("entropic_substrate.__legacy.__legacy.x")
            == "entropic_substrate.__legacy.x"
        )
