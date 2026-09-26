"""Teeth for tools/lane_slices.py (D-WAVES r2, operator ruling 2026-09-22).

A lane's slice is its targets plus every LANDED importer of those targets plus the shim pair
at each old path; two lanes of one wave must be file-disjoint outside the zip-owned
regenerable set. Each arm below builds a tiny tree in tmp_path: the RED arm is a shared
importer (exit 1 naming the file), the GREEN control is a disjoint pair (exit 0), the
regenerable arm proves the exclusion is load-bearing, and the malformed arm proves a lane
with no targets never reads as an empty, trivially disjoint slice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:  # tools/ is not a package; import the tool directly
    sys.path.insert(0, str(TOOLS))
import lane_slices  # noqa: E402

_A = "entropic_substrate.alpha"
_B = "entropic_substrate.beta"


def _tree(tmp_path: Path, importer_lines: dict[str, list[str]]) -> tuple[Path, Path]:
    src = tmp_path / "src"
    for layer in lane_slices.LANDED_LAYERS:
        (src / "entropic_substrate" / layer).mkdir(parents=True)
    for rel, lines in importer_lines.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "modules": {
                    _A: {
                        "dest": "entropic_substrate.core.alpha",
                        "layer": "core",
                        "action": "move",
                    },
                    _B: {
                        "dest": "entropic_substrate.ladder.beta",
                        "layer": "ladder",
                        "action": "move",
                    },
                }
            }
        )
    )
    return src, mapping


@pytest.mark.unit
def test_shared_landed_importer_is_an_overlap_RED(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src, mapping = _tree(
        tmp_path,
        {
            "entropic_substrate/ladder/run/both.py": [
                "from entropic_substrate.__legacy.alpha import x",
                "from entropic_substrate.__legacy.beta import y",
            ]
        },
    )
    rc = lane_slices.main([str(src), str(mapping), f"p6={_A}", f"p7={_B}"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "OVERLAP p6 x p7" in out
    assert "entropic_substrate/ladder/run/both.py" in out


@pytest.mark.unit
def test_disjoint_importers_pass_GREEN_control(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src, mapping = _tree(
        tmp_path,
        {
            "entropic_substrate/ladder/run/only_a.py": [
                "from entropic_substrate.__legacy.alpha import x"
            ],
            "entropic_substrate/core/only_b.py": ["import entropic_substrate.__legacy.beta"],
        },
    )
    rc = lane_slices.main([str(src), str(mapping), f"p6={_A}", f"p7={_B}", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert doc["overlaps"] == []
    # the slice carries the legacy source, the shim pair and the importer
    assert set(doc["slices"]["p6"]) == {
        "entropic_substrate/__legacy/alpha.py",
        "entropic_substrate/alpha.py",
        "entropic_substrate/alpha.pyi",
        "entropic_substrate/ladder/run/only_a.py",
    }


@pytest.mark.unit
def test_regenerable_files_are_excluded_from_the_check(tmp_path: Path) -> None:
    src, mapping = _tree(
        tmp_path,
        {
            "entropic_substrate/ladder/run/both.py": [
                "from entropic_substrate.__legacy.alpha import x",
                "from entropic_substrate.__legacy.beta import y",
            ]
        },
    )
    shared = "entropic_substrate/ladder/run/both.py"
    assert lane_slices.main([str(src), str(mapping), f"p6={_A}", f"p7={_B}"]) == 1
    assert (
        lane_slices.main([str(src), str(mapping), f"p6={_A}", f"p7={_B}", "--regenerable", shared])
        == 0
    )


@pytest.mark.unit
def test_old_prefix_imports_are_not_edges(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only explicit __legacy addresses are rewrite-on-arrival edges; an old-prefix import in a
    landed file is a separate defect (tests/test_no_old_prefix.py), never a slice member."""
    src, mapping = _tree(
        tmp_path,
        {"entropic_substrate/ladder/run/old.py": ["from entropic_substrate.alpha import x"]},
    )
    rc = lane_slices.main([str(src), str(mapping), f"p6={_A}", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert "entropic_substrate/ladder/run/old.py" not in doc["slices"]["p6"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "lane",
    ["p6=", "p6", f"p6={_A},not_a_dotted_name", "p6=@missing-fragment.json"],
)
def test_malformed_lane_never_reads_as_an_empty_slice(tmp_path: Path, lane: str) -> None:
    src, mapping = _tree(tmp_path, {})
    assert lane_slices.main([str(src), str(mapping), lane]) == 2


@pytest.mark.unit
def test_unmapped_target_is_refused(tmp_path: Path) -> None:
    src, mapping = _tree(tmp_path, {})
    assert lane_slices.main([str(src), str(mapping), "p6=entropic_substrate.ghost"]) == 2
