"""D1 reader identity closes over executable local source, not a hand list."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.analysis.d1_external_analysis_control import (
    D1_READER_SOURCE_MODULES,
    derive_d1_analysis_code_digest,
)
from entropic_substrate.analysis.d1_reader_source_closure import (
    D1_READER_ROOT_MODULES,
    D1ReaderSourceClosureError,
    D1ReaderSourceFile,
    discover_d1_reader_source_files,
    installed_d1_reader_source_files,
)

pytestmark = pytest.mark.unit


def _write(root: Path, relative: str, source: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def _fixture(root: Path) -> None:
    _write(root, "entropic_substrate/__init__.py", "")
    _write(
        root,
        "entropic_substrate/reader.py",
        "from entropic_substrate.impl import read\n",
    )
    _write(
        root,
        "entropic_substrate/impl.py",
        "from entropic_substrate.codec import decode\n\ndef read(): return decode()\n",
    )
    _write(root, "entropic_substrate/codec.py", "def decode(): return 1\n")


def _digest(files: tuple[D1ReaderSourceFile, ...]) -> str:
    return derive_d1_analysis_code_digest([item.manifest for item in files])


def test_transitive_change_moves_digest_but_unrelated_source_does_not(tmp_path: Path) -> None:
    _fixture(tmp_path)
    roots = ("entropic_substrate.reader",)

    first = discover_d1_reader_source_files(tmp_path, roots=roots)
    repeated = discover_d1_reader_source_files(tmp_path, roots=roots)
    assert first == repeated
    assert tuple(item.identity for item in first) == (
        "entropic_substrate.__init__",
        "entropic_substrate.codec",
        "entropic_substrate.impl",
        "entropic_substrate.reader",
    )

    _write(tmp_path, "entropic_substrate/unrelated.py", "UNRELATED = 2\n")
    assert discover_d1_reader_source_files(tmp_path, roots=roots) == first

    _write(tmp_path, "entropic_substrate/codec.py", "def decode(): return 2\n")
    changed = discover_d1_reader_source_files(tmp_path, roots=roots)
    assert tuple(item.identity for item in changed) == tuple(item.identity for item in first)
    assert _digest(changed) != _digest(first)


def test_literal_dynamic_import_is_closed_and_nonliteral_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "entropic_substrate/__init__.py", "")
    _write(
        tmp_path,
        "entropic_substrate/reader.py",
        "from importlib import import_module\nimpl = import_module('entropic_substrate.impl')\n",
    )
    _write(tmp_path, "entropic_substrate/impl.py", "VALUE = 1\n")
    files = discover_d1_reader_source_files(
        tmp_path,
        roots=("entropic_substrate.reader",),
    )
    assert "entropic_substrate.impl" in {item.identity for item in files}

    _write(
        tmp_path,
        "entropic_substrate/reader.py",
        "from importlib import import_module\nname = 'entropic_substrate.impl'\nimpl = import_module(name)\n",
    )
    with pytest.raises(D1ReaderSourceClosureError, match="literal"):
        discover_d1_reader_source_files(
            tmp_path,
            roots=("entropic_substrate.reader",),
        )


@pytest.mark.parametrize(
    "source",
    [
        "from importlib import import_module as load\nimpl = load('entropic_substrate.impl')\n",
        "import importlib as loader\nimpl = loader.import_module('entropic_substrate.impl')\n",
    ],
)
def test_aliased_literal_dynamic_import_is_closed(tmp_path: Path, source: str) -> None:
    _write(tmp_path, "entropic_substrate/__init__.py", "")
    _write(tmp_path, "entropic_substrate/reader.py", source)
    _write(tmp_path, "entropic_substrate/impl.py", "VALUE = 1\n")

    files = discover_d1_reader_source_files(
        tmp_path,
        roots=("entropic_substrate.reader",),
    )

    assert "entropic_substrate.impl" in {item.identity for item in files}


def test_installed_reader_set_is_the_discovered_code_owned_closure() -> None:
    files = installed_d1_reader_source_files()

    assert tuple(item.identity for item in files) == D1_READER_SOURCE_MODULES
    assert set(D1_READER_ROOT_MODULES) <= set(D1_READER_SOURCE_MODULES)
    assert "entropic_substrate.telemetry" in D1_READER_SOURCE_MODULES
    assert "entropic_substrate.d1_evaluation_values" in D1_READER_SOURCE_MODULES
    assert "entropic_substrate.d1_evaluation" not in D1_READER_SOURCE_MODULES
    assert all(item.path.startswith("entropic_substrate/") for item in files)


def test_empty_external_or_missing_roots_fail_loudly(tmp_path: Path) -> None:
    _write(tmp_path, "entropic_substrate/__init__.py", "")
    with pytest.raises(TypeError, match="nonempty"):
        discover_d1_reader_source_files(tmp_path, roots=())
    with pytest.raises(D1ReaderSourceClosureError, match="inside"):
        discover_d1_reader_source_files(tmp_path, roots=("outside.reader",))
    with pytest.raises(D1ReaderSourceClosureError, match="no source"):
        discover_d1_reader_source_files(
            tmp_path,
            roots=("entropic_substrate.missing",),
        )


def test_symlinked_source_root_and_source_file_are_refused(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    _fixture(real_root)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(D1ReaderSourceClosureError, match="real directory"):
        discover_d1_reader_source_files(
            root_link,
            roots=("entropic_substrate.reader",),
        )

    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    reader = real_root / "entropic_substrate/reader.py"
    reader.unlink()
    reader.symlink_to(outside)
    with pytest.raises(D1ReaderSourceClosureError, match="regular file"):
        discover_d1_reader_source_files(
            real_root,
            roots=("entropic_substrate.reader",),
        )
