"""Native verification and host-side receipt admission are different operations."""

from pathlib import Path

import pytest
from tests.test_sentence_custody import LIMITS, build_capture

from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.run import native_verification
from entropic_substrate.ladder.run.journal import read_json, write_json

pytestmark = pytest.mark.unit
REQUEST = ContentHash("sha256", "1" * 64)


def test_native_receipt_is_bound_and_can_be_checked_without_native_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root, capture = build_capture(tmp_path, monkeypatch)
    output = tmp_path / "verification"
    output.mkdir()
    receipt = native_verification.verify_native_capture(
        capture_root, capture, output, request_id=REQUEST, **LIMITS
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("host receipt reader attempted native admission")

    monkeypatch.setattr(native_verification, "read_model_trace", forbidden)
    recovered = native_verification.verify_native_capture(
        capture_root, capture, output, request_id=REQUEST, **LIMITS
    )
    assert recovered.content_hash == receipt.content_hash
    value = native_verification.read_native_verification(
        output, receipt, capture_root, capture, expected_request=REQUEST, **LIMITS
    )
    assert value["status"] == "native_reader_passed"
    assert value["scientific_authority"] == "none"
    assert value["runtime"]["device"] == "cpu" and value["runtime"]["accelerator"] is None
    assert value["wall_seconds"] >= 0
    with pytest.raises(ValueError, match="request"):
        native_verification.read_native_verification(
            output,
            receipt,
            capture_root,
            capture,
            expected_request=ContentHash("sha256", "2" * 64),
            **LIMITS,
        )
    forged = read_json(output, receipt, max_bytes=LIMITS["max_bytes"])
    forged["status"] = "byte_custody_verified"
    wrong = write_json(output, "wrong.json", forged, max_bytes=LIMITS["max_bytes"])
    with pytest.raises(ValueError, match="native verification"):
        native_verification.read_native_verification(
            output, wrong, capture_root, capture, expected_request=REQUEST, **LIMITS
        )
    for index, (field, value, message) in enumerate(
        (
            ("wall_seconds", -1, "duration"),
            ("verified_at", "2026-09-16", "UTC"),
            ("capture", {"path": "capture.json", "sha256": "0" * 64}, "capture binding"),
            ("run_identity", "0" * 64, "run identity"),
            (
                "runtime",
                {
                    "device": "cpu",
                    "python": "3.12.13",
                    "torch": "2.13.0",
                    "accelerator": {"model": "GPU", "count": 1},
                },
                "GPU execution",
            ),
        )
    ):
        record = read_json(output, receipt, max_bytes=LIMITS["max_bytes"])
        record[field] = value
        malformed = write_json(
            output, f"malformed-{index}.json", record, max_bytes=LIMITS["max_bytes"]
        )
        with pytest.raises(ValueError, match=message):
            native_verification.read_native_verification(
                output, malformed, capture_root, capture, expected_request=REQUEST, **LIMITS
            )
    with pytest.raises(ValueError, match="outside"):
        native_verification.verify_native_capture(
            capture_root, capture, capture_root, request_id=REQUEST, **LIMITS
        )
    tensor_file = next(capture_root.glob("*.pt"))
    with tensor_file.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="hash"):
        native_verification.read_native_verification(
            output, receipt, capture_root, capture, expected_request=REQUEST, **LIMITS
        )


def test_native_failure_cannot_emit_a_pass_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root, capture = build_capture(tmp_path, monkeypatch)
    output = tmp_path / "verification"
    output.mkdir()

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("native reader rejected model")

    monkeypatch.setattr(native_verification, "read_model_trace", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="rejected model"):
            native_verification.verify_native_capture(
                capture_root, capture, output, request_id=REQUEST, **LIMITS
            )
    assert not list(output.glob("*-native-verification.json"))
    assert len(list(output.glob("*-native-failure.json"))) == 2


def test_rejected_receipt_never_becomes_a_reusable_success_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root, capture = build_capture(tmp_path, monkeypatch)
    output = tmp_path / "verification"
    output.mkdir()
    original = native_verification.read_model_trace

    def inconsistent(*args: object, **kwargs: object) -> dict[str, object]:
        return original(*args, **kwargs) | {"run_identity": "0" * 64}

    monkeypatch.setattr(native_verification, "read_model_trace", inconsistent)
    with pytest.raises(ValueError, match="run identity"):
        native_verification.verify_native_capture(
            capture_root, capture, output, request_id=REQUEST, **LIMITS
        )
    assert not list(output.glob("*-native-verification.json"))
    monkeypatch.setattr(native_verification, "read_model_trace", original)
    receipt = native_verification.verify_native_capture(
        capture_root, capture, output, request_id=REQUEST, **LIMITS
    )
    assert (
        read_json(output, receipt, max_bytes=LIMITS["max_bytes"])["status"]
        == "native_reader_passed"
    )
