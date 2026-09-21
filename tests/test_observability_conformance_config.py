"""Tests for observability_conformance_config module.

Verifies that _load_merged_fragment_sets and _load_all_entrypoints raise on
malformed input (matching tools/gen_shims.py's fail-loud error model) and
that _load_merged_fragment_sets excludes dotfiles.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import gen_shims  # noqa: E402

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import observability_conformance_config as occ  # noqa: E402

pytestmark = pytest.mark.unit


def test_load_merged_fragment_sets_raises_on_malformed_json() -> None:
    """Verify that _load_merged_fragment_sets raises ManifestError on
    malformed JSON, matching tools/gen_shims.py:load_fragments behavior."""
    with tempfile.TemporaryDirectory() as td:
        fragments_dir = Path(td)
        bad_file = fragments_dir / "bad.json"
        bad_file.write_text("{ invalid json")

        with pytest.raises(gen_shims.ManifestError, match="is not valid JSON"):
            occ._load_merged_fragment_sets(fragments_dir)


def test_load_merged_fragment_sets_raises_on_read_error() -> None:
    """Verify that _load_merged_fragment_sets raises ManifestError on an
    OSError from a fragment path. A directory named ``*.json`` is a
    deterministic, platform-independent way to trigger an unreadable-path
    OSError -- unlike ``chmod(0o000)``, this does not depend on the test
    process running as a non-root user (root and Windows ignore POSIX mode
    bits), matching gen_shims behavior."""
    with tempfile.TemporaryDirectory() as td:
        fragments_dir = Path(td)
        unreadable = fragments_dir / "unreadable.json"
        unreadable.mkdir()

        with pytest.raises(gen_shims.ManifestError, match="cannot read fragment"):
            occ._load_merged_fragment_sets(fragments_dir)


def test_load_merged_fragment_sets_excludes_dotfiles() -> None:
    """Verify that _load_merged_fragment_sets excludes dotfiles (e.g.,
    .emitted.json) that Path.glob("*.json") would otherwise match."""
    with tempfile.TemporaryDirectory() as td:
        fragments_dir = Path(td)

        # Write a dotfile that should be ignored
        dotfile = fragments_dir / ".emitted.json"
        dotfile.write_text(json.dumps({"conforming_entrypoints": ["should.not.appear"]}))

        # Write a valid fragment that should be processed
        valid = fragments_dir / "valid.json"
        valid.write_text(
            json.dumps(
                {
                    "conforming_entrypoints": ["real.entrypoint"],
                    "print_allowlist": ["real.module"],
                    "sink_construction_modules": ["real.sink"],
                }
            )
        )

        conforming, allowlist, sinks = occ._load_merged_fragment_sets(fragments_dir)

        # Verify dotfile was excluded
        assert "should.not.appear" not in conforming
        assert "real.entrypoint" in conforming

        # Verify valid file was processed
        assert "real.module" in allowlist
        assert "real.sink" in sinks


def test_load_merged_fragment_sets_merges_multiple_fragments() -> None:
    """Verify that _load_merged_fragment_sets correctly merges multiple
    valid fragment files."""
    with tempfile.TemporaryDirectory() as td:
        fragments_dir = Path(td)

        # Write two fragments
        frag1 = fragments_dir / "p3b.json"
        frag1.write_text(
            json.dumps(
                {
                    "conforming_entrypoints": ["phase1.ep"],
                    "print_allowlist": ["phase1.print"],
                    "sink_construction_modules": ["phase1.sink"],
                }
            )
        )

        frag2 = fragments_dir / "p4.json"
        frag2.write_text(
            json.dumps(
                {
                    "conforming_entrypoints": ["phase2.ep"],
                    "print_allowlist": ["phase2.print"],
                    "sink_construction_modules": ["phase2.sink"],
                }
            )
        )

        conforming, allowlist, sinks = occ._load_merged_fragment_sets(fragments_dir)

        # Verify both fragments were merged
        assert "phase1.ep" in conforming
        assert "phase2.ep" in conforming
        assert "phase1.print" in allowlist
        assert "phase2.print" in allowlist
        assert "phase1.sink" in sinks
        assert "phase2.sink" in sinks


def test_load_merged_fragment_sets_handles_missing_optional_keys() -> None:
    """Verify that _load_merged_fragment_sets handles fragments with
    missing optional keys gracefully."""
    with tempfile.TemporaryDirectory() as td:
        fragments_dir = Path(td)

        # Write a fragment with only some keys
        frag = fragments_dir / "partial.json"
        frag.write_text(json.dumps({"conforming_entrypoints": ["ep"]}))

        conforming, allowlist, sinks = occ._load_merged_fragment_sets(fragments_dir)

        assert "ep" in conforming
        assert len(allowlist) == 0
        assert len(sinks) == 0


def test_load_merged_fragment_sets_handles_empty_directory() -> None:
    """Verify that _load_merged_fragment_sets returns empty frozensets
    when the fragments directory is empty."""
    with tempfile.TemporaryDirectory() as td:
        fragments_dir = Path(td)

        conforming, allowlist, sinks = occ._load_merged_fragment_sets(fragments_dir)

        assert len(conforming) == 0
        assert len(allowlist) == 0
        assert len(sinks) == 0


def test_load_merged_fragment_sets_handles_nonexistent_directory() -> None:
    """Verify that _load_merged_fragment_sets returns empty frozensets
    when the fragments directory does not exist."""
    nonexistent = Path("/tmp/this_does_not_exist_12345")

    conforming, allowlist, sinks = occ._load_merged_fragment_sets(nonexistent)

    assert len(conforming) == 0
    assert len(allowlist) == 0
    assert len(sinks) == 0


def test_load_all_entrypoints_raises_on_malformed_json() -> None:
    """Verify that _load_all_entrypoints raises ManifestError on malformed
    JSON, matching tools/gen_shims.py:load_mapping behavior. Prior to this
    fix it silently returned an empty frozenset() on the identical
    exception pair -- the same error-model divergence already fixed on
    _load_merged_fragment_sets above."""
    with tempfile.TemporaryDirectory() as td:
        mapping_file = Path(td) / "mapping.json"
        mapping_file.write_text("{ invalid json")

        with pytest.raises(gen_shims.ManifestError, match="is not valid JSON"):
            occ._load_all_entrypoints(mapping_file)


def test_load_all_entrypoints_raises_on_read_error() -> None:
    """Verify that _load_all_entrypoints raises ManifestError on an OSError
    from the mapping path. A directory named mapping.json is a
    deterministic, platform-independent way to trigger this -- see
    test_load_merged_fragment_sets_raises_on_read_error's docstring for why
    chmod(0o000) does not reliably do so."""
    with tempfile.TemporaryDirectory() as td:
        unreadable = Path(td) / "mapping.json"
        unreadable.mkdir()

        with pytest.raises(gen_shims.ManifestError, match="cannot read mapping"):
            occ._load_all_entrypoints(unreadable)


def test_load_all_entrypoints_handles_nonexistent_file() -> None:
    """Verify that _load_all_entrypoints returns an empty frozenset when
    the mapping file does not exist -- a missing file is a distinct,
    pre-checked case from a malformed/unreadable one and stays a soft
    fallback, matching _load_merged_fragment_sets' own
    fragments_dir.exists() gate."""
    nonexistent = Path("/tmp/this_mapping_does_not_exist_12345.json")

    assert occ._load_all_entrypoints(nonexistent) == frozenset()
