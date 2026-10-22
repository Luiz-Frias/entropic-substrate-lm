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


def test_packet_zip_preserves_relative_layout(tmp_path: pathlib.Path) -> None:
    import zipfile

    packet = tmp_path / "packet"
    (packet / "data").mkdir(parents=True)
    (packet / "manifest.txt").write_text("m", encoding="utf-8")
    (packet / "data" / "A-seed1.jsonl").write_text("a", encoding="utf-8")
    zip_path = tmp_path / "packet.zip"
    arcnames = _MOD.build_packet_zip(packet, zip_path)
    assert arcnames == ["data/A-seed1.jsonl", "manifest.txt"]
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.read("data/A-seed1.jsonl") == b"a"
        assert archive.read("manifest.txt") == b"m"


def test_content_leads_with_prompt_then_the_archive() -> None:
    content = _MOD.build_content("REVIEW PROMPT", ["file_zip"])
    assert content[0]["type"] == "text"
    assert content[0]["text"].startswith("REVIEW PROMPT")
    assert "NOTE ON DELIVERY" in content[0]["text"]
    assert [c["file_id"] for c in content[1:]] == ["file_zip"]
    assert all(c["type"] == "container_upload" for c in content[1:])


def test_reviewer_model_comes_from_the_environment_with_the_pinned_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer model is an operator input: BLIND_REVIEW_MODEL wins, the
    pinned default holds when it is unset, and a blank value is refused rather
    than silently falling back (fail-loud, house rule 5)."""
    monkeypatch.delenv("BLIND_REVIEW_MODEL", raising=False)
    assert _MOD.resolve_model() == _MOD.DEFAULT_MODEL == "claude-opus-5"
    monkeypatch.setenv("BLIND_REVIEW_MODEL", "claude-opus-5-5")
    assert _MOD.resolve_model() == "claude-opus-5-5"
    monkeypatch.setenv("BLIND_REVIEW_MODEL", "   ")
    with pytest.raises(ValueError, match="BLIND_REVIEW_MODEL"):
        _MOD.resolve_model()


def test_a_blank_model_is_refused_before_any_client_or_upload_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Configuration errors must surface before spend or network, not after an upload."""
    import types

    touched: list[str] = []

    class _Anthropic:
        def __init__(self) -> None:
            touched.append("client")

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))
    monkeypatch.setenv("BLIND_REVIEW_MODEL", "   ")
    packet = tmp_path / "packet"
    packet.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("review", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["blind_review_cloud", str(packet), str(tmp_path / "out.md"), str(prompt)]
    )
    with pytest.raises(ValueError, match="BLIND_REVIEW_MODEL"):
        _MOD.main()
    assert touched == []
