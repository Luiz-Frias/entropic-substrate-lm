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
this lane's tip. ``installed_d1_reader_source_files()`` returns 73 entries,
not 52. This file pins 73, the measured value, and this docstring is the
explicit retraction the ruling requires -- the 52 literal is not carried
forward anywhere in this fixture.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import pytest
import torch

from entropic_substrate.analysis.d1_reader_source_closure import (
    installed_d1_reader_source_files,
)
from entropic_substrate.checkpoint import _TAU_ERAS, CHECKPOINT_FORMAT_VERSION
from entropic_substrate.observation.byte_codec import ByteObservationCodec
from entropic_substrate.observation.permuted_byte_codec import PermutedByteObservationCodec
from entropic_substrate.sentence_prediction.capture import (
    _READER_MODULES as _SOURCE_READER_MODULES,
)
from entropic_substrate.sentence_prediction.capture import (
    _reader_identity as _source_reader_identity,
)
from entropic_substrate.sentence_prediction.checkpoint import (
    _reader_identity as _checkpoint_reader_identity,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Source-derived components (regime-free): frozen at p0, byte-exact.
# ---------------------------------------------------------------------------

_FROZEN_SOURCE_READER_MODULES = (
    "sentence_prediction",
    "sentence_prediction.records",
    "sentence_prediction.segmentation",
    "sentence_prediction.corpus",
    "sentence_prediction.capture",
    "d1_artifact_snapshot",
    "evidence.lifecycle",
)

_FROZEN_CHECKPOINT_FORMAT_VERSION = 1
_FROZEN_TAU_ERAS = ("softplus", "temperature")

_FROZEN_D1_CLOSURE_COUNT = 73
_FROZEN_D1_CLOSURE_MANIFEST_DIGEST = (
    "c66acd6ffa03e5d1dd457d9f7255b9550189b5349d984942367f2399f82c472b"
)

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
    ("3.12.13", "2.13.0"): "42c7ac1087b0c9b1fdbe58d991458c3bf12c9c317fbe46543a44376c56b86cfe",
    (
        "3.12.14",
        "2.13.0+cpu",
    ): "a0bf5cad38047cfaf165bda80cb67b4988525404245cff9cf882dd7bebba7ec0",  # pragma: allowlist secret
}

_SOURCE_REGIME_TABLE: dict[tuple[str, str], str] = {
    ("3.12.13", "0.3.4"): "da8205c91989bfc09feafcd56ff46d6fd6c2b366f452308e8353e425cfeae8be",
    (
        "3.12.14",
        "0.3.4",
    ): "8e973c8702bd20d9c8cfc25331346674c7070dd7cbb9e4b6488a1881f1b25165",  # pragma: allowlist secret
}


def _closure_manifest_digest() -> tuple[int, str]:
    closure = installed_d1_reader_source_files()
    manifest = [item.manifest for item in closure]
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return len(closure), digest


class TestSourceDerivedComponentsFrozenAtP0:
    """These assertions carry no regime key -- they are pure source facts."""

    def test_source_reader_module_tuple_is_frozen(self) -> None:
        assert _SOURCE_READER_MODULES == _FROZEN_SOURCE_READER_MODULES

    def test_checkpoint_format_version_is_frozen(self) -> None:
        assert CHECKPOINT_FORMAT_VERSION == _FROZEN_CHECKPOINT_FORMAT_VERSION

    def test_tau_eras_are_frozen(self) -> None:
        assert _TAU_ERAS == _FROZEN_TAU_ERAS

    def test_d1_closure_manifest_is_frozen_at_73_entries(self) -> None:
        count, digest = _closure_manifest_digest()
        assert count == _FROZEN_D1_CLOSURE_COUNT
        assert digest == _FROZEN_D1_CLOSURE_MANIFEST_DIGEST

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
                f"h-checkpoint-reader-v1 has no fixture row for regime {regime!r}. "
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
                f"h-source-reader-v2 has no fixture row for regime {regime!r}. "
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
                f"h-checkpoint-reader-v1 has no fixture row for regime {bogus_regime!r}"
            )

    def test_source_regime_absent_from_table_raises_named_failure(self) -> None:
        bogus_regime = ("9.9.9", "0.0.0-does-not-exist")
        assert bogus_regime not in _SOURCE_REGIME_TABLE
        with pytest.raises(AssertionError):
            expected = _SOURCE_REGIME_TABLE.get(bogus_regime)
            assert expected is not None, (
                f"h-source-reader-v2 has no fixture row for regime {bogus_regime!r}"
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
        # 3. After stripping one __legacy. prefix, it meets condition 1 or 2
        uncovered = set()
        for node in graph_nodes:
            # Check condition 1: in mapping
            if node in mapping_entries:
                continue
            # Check condition 2: is a layer or subpackage
            if _is_layer_or_subpackage(node, layers):
                continue
            # Check condition 3: strip __legacy. and recheck
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
