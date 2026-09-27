"""Bounded, observed first-32-batch source-offset custody for D-series resume."""

import hashlib
import struct
from typing import cast


class SourceOrderAudit:
    def __init__(self, batch_size: int) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("source audit requires positive batch size")
        self.batch_size = batch_size
        self._offsets: list[tuple[int, ...]] = []

    def record(self, offsets: tuple[int, ...]) -> None:
        """Record actual sampled offsets; never regenerate them from a seed."""
        if (
            type(offsets) is not tuple
            or len(offsets) != self.batch_size
            or any(type(value) is not int or not 0 <= value < 2**64 for value in offsets)
        ):
            raise ValueError("source offsets must be a complete nonnegative integer batch")
        if len(self._offsets) < 32:
            self._offsets.append(offsets)

    @property
    def attestation(self) -> tuple[str, int] | None:
        if not self._offsets:
            return None
        payload = b"".join(struct.pack("<Q", value) for batch in self._offsets for value in batch)
        return hashlib.sha256(payload).hexdigest(), len(self._offsets)

    @property
    def manifest(self) -> dict[str, object]:
        attestation = self.attestation
        return {
            "schema": "d-ladder-source-offsets-v1",
            "batch_size": self.batch_size,
            "offsets": [list(batch) for batch in self._offsets],
            "sha256": None if attestation is None else attestation[0],
        }

    @classmethod
    def restore(cls, value: object, *, step: int, batch_size: int) -> "SourceOrderAudit":
        if type(step) is not int or step < 0:
            raise ValueError("source audit cursor must be nonnegative")
        if type(value) is not dict or set(value) != {"schema", "batch_size", "offsets", "sha256"}:
            raise ValueError("source audit requires exact versioned fields")
        row = cast(dict[str, object], value)
        if row["schema"] != "d-ladder-source-offsets-v1" or row["batch_size"] != batch_size:
            raise ValueError("source audit schema or batch size mismatch")
        offsets = row["offsets"]
        if type(offsets) is not list or len(offsets) != min(step, 32):
            raise ValueError("source audit lacks the complete observed prefix")
        result = cls(batch_size)
        for batch in offsets:
            if type(batch) is not list:
                raise ValueError("source audit offset batch must be a list")
            result.record(tuple(batch))
        if result.manifest != value:
            raise ValueError("source audit digest or canonical fields mismatch")
        return result
