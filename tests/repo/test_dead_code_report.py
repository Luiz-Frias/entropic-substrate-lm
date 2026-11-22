"""Teeth for ``scripts/dead_code_report.py``: a WARNING report of source
modules nothing reaches. It never fails a gate; it names files to look at
when a refactor is in scope.
"""

from __future__ import annotations

from pathlib import Path

import dead_code_report as report
import pytest

pytestmark = pytest.mark.unit


def _package(root: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        target = root / "pkg" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return root / "pkg"


def test_a_module_reached_only_by_import_is_alive_and_an_unreached_one_is_reported(
    tmp_path: Path,
) -> None:
    package = _package(
        tmp_path,
        {
            "__init__.py": "",
            "main.py": "from pkg import used\nif __name__ == '__main__':\n    used.run()\n",
            "used.py": "from pkg.deep import leaf\ndef run():\n    return leaf.X\n",
            "deep/__init__.py": "",
            "deep/leaf.py": "X = 1\n",
            "orphan.py": "Y = 2\n",
        },
    )
    assert report.unreached(package, extra_roots=()) == ["pkg.orphan"]


def test_a_module_named_only_in_a_string_is_alive(tmp_path: Path) -> None:
    package = _package(
        tmp_path,
        {
            "__init__.py": "",
            "main.py": "ARGV = ['python', '-m', 'pkg.spawned']\nif __name__ == '__main__':\n    pass\n",
            "spawned.py": "Z = 3\n",
        },
    )
    assert report.unreached(package, extra_roots=()) == []


def test_a_module_named_by_an_outside_file_is_a_root(tmp_path: Path) -> None:
    package = _package(tmp_path, {"__init__.py": "", "tool.py": "A = 1\n", "orphan.py": "B = 2\n"})
    recipe = tmp_path / "justfile"
    recipe.write_text("run:\n    python -m pkg.tool\n")
    assert report.unreached(package, extra_roots=(recipe,)) == ["pkg.orphan"]


def test_a_cycle_nothing_enters_is_reported_whole(tmp_path: Path) -> None:
    package = _package(
        tmp_path,
        {
            "__init__.py": "",
            "a.py": "from pkg import b\n",
            "b.py": "from pkg import a\n",
            "main.py": "if __name__ == '__main__':\n    pass\n",
        },
    )
    assert report.unreached(package, extra_roots=()) == ["pkg.a", "pkg.b"]


def test_the_cli_warns_and_still_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _package(tmp_path, {"__init__.py": "", "orphan.py": "B = 2\n"})
    assert report.main(["--package", str(package)]) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "pkg/orphan.py" in out


def test_a_file_that_cannot_be_parsed_warns_and_still_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "broken.py").write_text("def (:\n", encoding="utf-8")
    assert report.main(["--package", str(package)]) == 0
    assert "incomplete" in capsys.readouterr().out
