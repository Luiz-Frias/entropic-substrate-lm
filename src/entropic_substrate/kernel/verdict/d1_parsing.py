"""Strict primitive parsers shared by the D1 verdict validator."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType

from entropic_substrate.kernel.codec.strict_json import exact_keys
from entropic_substrate.kernel.lifecycle import (
    ArtifactIdentity,
    ContentHash,
    LifecycleContractError,
)
from entropic_substrate.kernel.verdict.d1_types import (
    CLAUSE_STATUS,
    D1VerdictValidationError,
    JsonValue,
)


def error(code: str, path: str, detail: str) -> D1VerdictValidationError:
    return D1VerdictValidationError(code=code, path=path, detail=detail)


def object_exact(value: object, *, path: str, fields: frozenset[str]) -> Mapping[str, object]:
    return object_optional(value, path=path, required=fields, optional=frozenset())


def object_optional(
    value: object,
    *,
    path: str,
    required: frozenset[str],
    optional: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error("invalid_type", path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise error("invalid_field_name", path, "object field names must be strings")
    missing, unknown = exact_keys(value, required, optional)
    if missing:
        raise error(
            "missing_field", path, f"missing required field(s): {', '.join(sorted(missing))}"
        )
    if unknown:
        raise error("unknown_field", path, f"unknown field(s): {', '.join(sorted(unknown))}")
    return value


def string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise error("invalid_string", path, "must be a nonempty string")
    return value


def finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error("invalid_number", path, "must be a finite numeric value, not a boolean")
    result = float(value)
    if not isfinite(result):
        raise error("invalid_number", path, "must be finite")
    return result


def number_or_none(value: object, *, path: str) -> float | None:
    return None if value is None else finite_number(value, path=path)


def positive_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error("invalid_integer", path, "must be a positive integer, not a boolean")
    return value


def status(value: object, *, path: str) -> str:
    result = string(value, path=path)
    if result not in CLAUSE_STATUS:
        raise error("invalid_status", path, f"must be one of {sorted(CLAUSE_STATUS)}")
    return result


def hash_value(value: object, *, path: str) -> ContentHash:
    data = object_exact(value, path=path, fields=frozenset({"algorithm", "digest"}))
    algorithm = string(data["algorithm"], path=f"{path}.algorithm")
    digest = string(data["digest"], path=f"{path}.digest")
    try:
        return ContentHash(algorithm=algorithm, digest=digest)
    except LifecycleContractError as exc:
        raise error(exc.code, path, str(exc)) from exc


def plain_hash(value: object, *, algorithm: str, path: str) -> ContentHash:
    digest = string(value, path=path)
    try:
        return ContentHash(algorithm=algorithm, digest=digest)
    except LifecycleContractError as exc:
        raise error(exc.code, path, str(exc)) from exc


def sha256_digest(value: object, *, path: str) -> str:
    return plain_hash(value, algorithm="sha256", path=path).digest


def artifact(value: object, *, path: str) -> ArtifactIdentity:
    data = object_exact(value, path=path, fields=frozenset({"path", "content_hash"}))
    artifact_path = string(data["path"], path=f"{path}.path")
    content_hash = hash_value(data["content_hash"], path=f"{path}.content_hash")
    try:
        return ArtifactIdentity(path=artifact_path, content_hash=content_hash)
    except LifecycleContractError as exc:
        raise error(exc.code, path, str(exc)) from exc


def assert_hash_stable(artifacts: tuple[ArtifactIdentity, ...]) -> None:
    seen: dict[str, ContentHash] = {}
    for artifact_identity in artifacts:
        prior = seen.get(artifact_identity.path)
        if prior is not None and prior != artifact_identity.content_hash:
            raise error(
                "hash_drift",
                artifact_identity.path,
                "hash drift: one artifact path carries more than one content hash",
            )
        seen[artifact_identity.path] = artifact_identity.content_hash


def freeze_json(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise error("invalid_registered_constant", path, "floating values must be finite")
        return value
    if isinstance(value, list):
        return tuple(freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise error(
                    "invalid_registered_constant", path, "mapping keys must be nonempty strings"
                )
            frozen[key] = freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    raise error(
        "invalid_registered_constant", path, "must be a finite JSON scalar, list, or object"
    )


def registered_constants(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or not value:
        raise error(
            "invalid_registered_constants", "registered_constants", "must be a nonempty object"
        )
    frozen = freeze_json(value, path="registered_constants")
    if not isinstance(frozen, Mapping):
        raise error("invalid_registered_constants", "registered_constants", "must be an object")
    return frozen
