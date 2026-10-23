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


# --- FOR-1671: files the reviewer writes in its sandbox ----------------------
#
# Wire shape mirrored from the claude-api skill (python/claude-api/tool-use.md,
# "Retrieve Generated Files", bundle 2.1.308, read 2026-10-22): a
# ``bash_code_execution_tool_result`` block whose ``.content`` is a
# ``bash_code_execution_result`` whose ``.content`` lists
# ``bash_code_execution_output`` entries carrying ``.file_id``.


def _text(text: str) -> object:
    import types

    return types.SimpleNamespace(type="text", text=text)


def _result_with_files(*file_ids: str) -> object:
    import types

    outputs = [
        types.SimpleNamespace(type="bash_code_execution_output", file_id=fid) for fid in file_ids
    ]
    inner = types.SimpleNamespace(
        type="bash_code_execution_result",
        stdout="",
        stderr="",
        return_code=0,
        content=outputs,
    )
    return types.SimpleNamespace(type="bash_code_execution_tool_result", content=inner)


def _result_error() -> object:
    import types

    inner = types.SimpleNamespace(
        type="bash_code_execution_tool_result_error", error_code="unavailable"
    )
    return types.SimpleNamespace(type="bash_code_execution_tool_result", content=inner)


def _response(content: list[object], stop_reason: str = "end_turn") -> object:
    import types

    return types.SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=None)


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    responses: list[object],
    filenames: dict[str, str] | None = None,
) -> tuple[pathlib.Path, list[str], list[str]]:
    """Drive ``main()`` against a fake client replaying ``responses`` in order.

    Returns ``(out_md, downloaded_ids, deleted_ids)``.
    """
    import types

    names = filenames or {}
    downloaded: list[str] = []
    deleted: list[str] = []
    queue = list(responses)

    class _Stream:
        def __init__(self, response: object) -> None:
            self._response = response

        def __enter__(self) -> _Stream:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def get_final_message(self) -> object:
            return self._response

    class _Download:
        def __init__(self, file_id: str) -> None:
            self._file_id = file_id

        def write_to_file(self, path: object) -> None:
            pathlib.Path(str(path)).write_bytes(f"content-of-{self._file_id}".encode())

    class _Files:
        def upload(self, **_: object) -> object:
            return types.SimpleNamespace(id="file_zip")

        def delete(self, file_id: str) -> None:
            deleted.append(file_id)

        def retrieve_metadata(self, file_id: str) -> object:
            return types.SimpleNamespace(filename=names.get(file_id, f"{file_id}.md"))

        def download(self, file_id: str) -> _Download:
            downloaded.append(file_id)
            return _Download(file_id)

    class _Messages:
        def stream(self, **_: object) -> _Stream:
            return _Stream(queue.pop(0))

    class _Anthropic:
        def __init__(self) -> None:
            self.beta = types.SimpleNamespace(files=_Files(), messages=_Messages())

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))
    monkeypatch.delenv("BLIND_REVIEW_MODEL", raising=False)
    packet = tmp_path / "packet"
    packet.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("review", encoding="utf-8")
    out = tmp_path / "review.md"
    monkeypatch.setattr(sys, "argv", ["blind_review_cloud", str(packet), str(out), str(prompt)])
    assert _MOD.main() == 0
    return out, downloaded, deleted


def test_text_and_one_output_file_both_land(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, downloaded, _ = _run_main(
        monkeypatch,
        tmp_path,
        [_response([_text("the review"), _result_with_files("file_a")])],
        {"file_a": "review.md"},
    )
    assert out.read_text(encoding="utf-8") == "the review"
    saved = tmp_path / "review-files" / "review.md"
    assert saved.read_bytes() == b"content-of-file_a"
    assert downloaded == ["file_a"]
    assert str(saved) in capsys.readouterr().out


def test_every_output_file_is_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _, downloaded, _ = _run_main(
        monkeypatch,
        tmp_path,
        [_response([_text("r"), _result_with_files("file_a", "file_b")])],
        {"file_a": "one.md", "file_b": "two.csv"},
    )
    assert downloaded == ["file_a", "file_b"]
    files = tmp_path / "review-files"
    assert (files / "one.md").read_bytes() == b"content-of-file_a"
    assert (files / "two.csv").read_bytes() == b"content-of-file_b"


def test_text_only_writes_the_review_and_creates_no_files_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    out, downloaded, _ = _run_main(monkeypatch, tmp_path, [_response([_text("only text")])])
    assert out.read_text(encoding="utf-8") == "only text"
    assert downloaded == []
    assert not (tmp_path / "review-files").exists()


def test_output_files_without_text_still_land_and_no_review_md_is_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    out, downloaded, _ = _run_main(
        monkeypatch,
        tmp_path,
        [_response([_result_with_files("file_a")])],
        {"file_a": "review.md"},
    )
    assert downloaded == ["file_a"]
    assert (tmp_path / "review-files" / "review.md").exists()
    assert not out.exists()


def test_neither_text_nor_output_file_fails_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(RuntimeError, match="no text content"):
        _run_main(monkeypatch, tmp_path, [_response([_text("   "), _result_error()])])
    assert not (tmp_path / "review-files").exists()


def test_a_failed_execution_result_block_contributes_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    out, downloaded, _ = _run_main(
        monkeypatch, tmp_path, [_response([_text("t"), _result_error()])]
    )
    assert out.read_text(encoding="utf-8") == "t"
    assert downloaded == []


def test_a_file_written_in_a_paused_round_is_not_lost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A resumed response carries only the new round's blocks, so refs must be
    gathered per round, not from the final response alone."""
    out, downloaded, _ = _run_main(
        monkeypatch,
        tmp_path,
        [
            _response([_result_with_files("file_early")], stop_reason="pause_turn"),
            _response([_text("final text")]),
        ],
        {"file_early": "early.md"},
    )
    assert out.read_text(encoding="utf-8") == "final text"
    assert downloaded == ["file_early"]
    assert (tmp_path / "review-files" / "early.md").exists()


def test_the_same_file_referenced_twice_is_downloaded_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _, downloaded, _ = _run_main(
        monkeypatch,
        tmp_path,
        [
            _response([_result_with_files("file_a")], stop_reason="pause_turn"),
            _response([_text("t"), _result_with_files("file_a")]),
        ],
    )
    assert downloaded == ["file_a"]


@pytest.mark.parametrize(
    ("hostile", "expected"),
    [("../../escape.md", "escape.md"), ("/etc/passwd", "passwd"), ("..", "file_a"), ("", "file_a")],
)
def test_hostile_filenames_cannot_escape_the_files_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, hostile: str, expected: str
) -> None:
    _run_main(
        monkeypatch,
        tmp_path,
        [_response([_text("t"), _result_with_files("file_a")])],
        {"file_a": hostile},
    )
    files = tmp_path / "review-files"
    assert [p.name for p in files.iterdir()] == [expected]
    assert not (tmp_path.parent / "escape.md").exists()


def test_two_files_with_the_same_name_do_not_overwrite_each_other(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _run_main(
        monkeypatch,
        tmp_path,
        [_response([_text("t"), _result_with_files("file_a", "file_b")])],
        {"file_a": "same.md", "file_b": "same.md"},
    )
    contents = sorted(p.read_bytes() for p in (tmp_path / "review-files").iterdir())
    assert contents == [b"content-of-file_a", b"content-of-file_b"]
