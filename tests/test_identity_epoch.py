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

p2 retraction (house rule 4, MEASURED 2026-09-20): the D1 closure manifest
row (``_FROZEN_D1_CLOSURE_COUNT``, ``_FROZEN_D1_CLOSURE_MANIFEST_DIGEST``,
``_closure_manifest_digest()`` and their pinning test) is DROPPED here, not
re-pinned. ``analysis/d1_reader_source_closure.py`` -- the module this row
measured -- is retired outright this phase (operator ruling E-1); `git grep`
over tracked source confirms zero remaining importers of
``installed_d1_reader_source_files`` once this row is removed. This is the
one D1 closure manifest row this phase's ruled bump touches; every other
fixture value below stays byte-identical to p0/p1.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch

from entropic_substrate.checkpoint import _TAU_ERAS, CHECKPOINT_FORMAT_VERSION
from entropic_substrate.ladder.capture.capture import (
    _READER_MODULES as _SOURCE_READER_MODULES,
)
from entropic_substrate.ladder.capture.capture import (
    _reader_identity as _source_reader_identity,
)
from entropic_substrate.ladder.run.checkpoint import (
    _reader_identity as _checkpoint_reader_identity,
)
from entropic_substrate.observation.byte_codec import ByteObservationCodec
from entropic_substrate.observation.permuted_byte_codec import PermutedByteObservationCodec

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
    ): "8e4ca5a3840e5c2efa7c4e0f0a1521af576e54139588cef76169021276b8eaf3",  # pragma: allowlist secret
    (
        "3.12.14",
        "2.13.0+cpu",
    ): "ea85d18b278423bc7de7095b8292ce224b6c7aa50428eecf443dec828eb1602a",  # pragma: allowlist secret
}

_SOURCE_REGIME_TABLE: dict[tuple[str, str], str] = {
    (
        "3.12.13",
        "0.3.4",
    ): "a6075295436de193087160f65fa2558fc28054115118946e53ecbdd4f22d510f",  # pragma: allowlist secret
    (
        "3.12.14",
        "0.3.4",
    ): "214730ac2b370a7b67fc24a43077c49ecdcc59291505d80f249f235aaccf0467",  # pragma: allowlist secret
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
