"""Identity-bound weak attestation for factory-only capability values."""

from __future__ import annotations

import threading
from collections.abc import Callable
from weakref import ReferenceType, ref


class FactoryAttestation[Value: object, Fingerprint: object]:
    """Trust only the exact weak-referenceable object registered by a factory."""

    def __init__(self, fingerprint: Callable[[Value], Fingerprint]) -> None:
        self._lock = threading.Lock()
        self._fingerprint = fingerprint
        self._references: dict[
            int,
            tuple[ReferenceType[Value], Fingerprint],
        ] = {}

    def register(self, value: Value) -> None:
        identifier = id(value)

        def discard(reference: ReferenceType[Value]) -> None:
            with self._lock:
                current = self._references.get(identifier)
                if current is not None and current[0] is reference:
                    self._references.pop(identifier, None)

        reference = ref(value, discard)
        fingerprint = self._fingerprint(value)
        with self._lock:
            self._references[identifier] = (reference, fingerprint)

    def registered(self, value: Value) -> bool:
        with self._lock:
            current = self._references.get(id(value))
            return current is not None and current[0]() is value

    def contains(self, value: Value) -> bool:
        with self._lock:
            current = self._references.get(id(value))
            if current is None or current[0]() is not value:
                return False
            expected = current[1]
        return self._fingerprint(value) == expected
