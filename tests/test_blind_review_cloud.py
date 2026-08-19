"""Pins for the cloud blind-review packet transport.

The blinding contract survives the Files API's flat namespace only if the
name flattening is content-preserving and the reviewer prompt precedes the
files in the single user message (the model reads the instructions before
any data). These pins hold those two properties.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "blind_review_cloud",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "blind_review_cloud.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["blind_review_cloud"] = _MOD
_SPEC.loader.exec_module(_MOD)


def test_flatten_name_is_layout_recoverable() -> None:
    assert _MOD.flatten_name(pathlib.PurePath("data/A-seed1.jsonl")) == "data__A-seed1.jsonl"
    assert _MOD.flatten_name(pathlib.PurePath("manifest.txt")) == "manifest.txt"


def test_content_leads_with_prompt_then_all_files() -> None:
    content = _MOD.build_content("REVIEW PROMPT", ["file_a", "file_b"])
    assert content[0]["type"] == "text"
    assert content[0]["text"].startswith("REVIEW PROMPT")
    assert "NOTE ON FILE NAMES" in content[0]["text"]
    assert [c["file_id"] for c in content[1:]] == ["file_a", "file_b"]
    assert all(c["type"] == "container_upload" for c in content[1:])
