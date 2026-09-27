"""Strict canonical codec for one registered D1 operator launch intent."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Final, NoReturn, cast

from entropic_substrate.kernel.evidence_class.resource_provenance import PurchaseOption, QuotedRate
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import (
    ResourceProvenanceCodecGap,
    quoted_rate_from_mapping,
    resource_activation_from_mapping,
)
from entropic_substrate.kernel.identity.d1_primitives import RunClass, canonical_json
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile
from entropic_substrate.ladder.register.d1_plan import (
    D1PlanManifestError,
    parse_d1_confirmatory_plan,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import (
    D1A100FormFactor,
    D1A100ResourceRequest,
    D1ImmutableImage,
    D1OperatorLaunchIntent,
)

__all__ = [
    "D1OperatorIntentCodecError",
    "parse_d1_confirmatory_operator_intent",
    "serialize_d1_operator_intent",
]

_SCHEMA: Final = "d1-operator-launch-intent-v2"
_ROOT_KEYS: Final = frozenset(
    {
        "execution_segment_id",
        "execution_segment_index",
        "image_reference",
        "plan",
        "plan_digest",
        "quoted_rate",
        "registered_design_digest",
        "resource",
        "resource_activation",
        "resume_checkpoint_sha256",
        "schema_version",
        "segment_start_mode",
        "terminate_after_seconds",
    }
)
_RESOURCE_KEYS: Final = frozenset(
    {
        "accelerator_count",
        "accelerator_form_factor",
        "market",
        "provider",
        "provider_gpu_type_id",
        "provider_sku",
        "purchase_option",
        "region",
    }
)


class D1OperatorIntentCodecError(ValueError):
    """A launch-intent wire value is malformed, substituted, or noncanonical."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1OperatorIntentCodecError(message)
    raise D1OperatorIntentCodecError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"D1 operator intent {label} must be an object with string keys")
    return cast(dict[str, object], value)


def _exact(value: Mapping[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"D1 operator intent {label} keys must be exact")


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"D1 operator intent {label} must be a nonempty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label=label)


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        _fail(f"D1 operator intent {label} must be a {qualifier} integer")
    return value


def _profile(
    plan: Mapping[str, object],
    catalog: Iterable[D1LaunchProfile],
) -> D1LaunchProfile:
    values = tuple(catalog)
    if any(type(item) is not D1LaunchProfile for item in values):
        _fail("D1 operator intent profile catalog contains a non-profile value")
    name = _text(plan.get("profile_name"), label="plan.profile_name")
    candidates = tuple(item for item in values if item.name == name)
    if len(candidates) != 1:
        _fail("D1 operator intent profile identity is absent or ambiguous")
    return candidates[0]


def _image(value: object, *, source_commit: str) -> D1ImmutableImage:
    reference = _text(value, label="image_reference")
    repository, separator, digest = reference.rpartition("@")
    if not separator:
        _fail("D1 operator intent image_reference must be digest-addressed")
    try:
        image = D1ImmutableImage(repository, digest, source_commit)
    except (TypeError, ValueError) as error:
        _fail(f"D1 operator intent image_reference is invalid: {error}", error)
    if image.reference != reference:
        _fail("D1 operator intent image_reference is noncanonical")
    return image


def _resource(value: object) -> D1A100ResourceRequest:
    item = _mapping(value, label="resource")
    _exact(item, _RESOURCE_KEYS, label="resource")
    try:
        return D1A100ResourceRequest(
            accelerator_form_factor=D1A100FormFactor(item["accelerator_form_factor"]),
            provider_gpu_type_id=cast(str, item["provider_gpu_type_id"]),
            accelerator_count=cast(int, item["accelerator_count"]),
            provider=cast(str, item["provider"]),
            provider_sku=cast(str, item["provider_sku"]),
            region=cast(str, item["region"]),
            market=cast(str, item["market"]),
            purchase_option=PurchaseOption(item["purchase_option"]),
        )
    except (TypeError, ValueError) as error:
        _fail(f"D1 operator intent resource is invalid: {error}", error)


def _quoted_rate(value: object) -> QuotedRate:
    item = _mapping(value, label="quoted_rate")
    observed = _text(item.get("observed_at"), label="quoted_rate.observed_at")
    try:
        timestamp = datetime.fromisoformat(observed)
    except ValueError as error:
        _fail("D1 operator intent quoted_rate.observed_at is invalid", error)
    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() != timedelta(0)
        or timestamp.isoformat() != observed
    ):
        _fail("D1 operator intent quoted_rate.observed_at must be canonical UTC ISO-8601")
    normalized = dict(item)
    normalized["observed_at"] = observed.replace("+00:00", "Z")
    try:
        return quoted_rate_from_mapping(normalized)
    except ResourceProvenanceCodecGap as error:
        _fail(f"D1 operator intent quoted_rate is invalid: {error}", error)


def operator_intent_from_mapping(
    value: object,
    *,
    catalog: Iterable[D1LaunchProfile],
) -> D1OperatorLaunchIntent:
    root = _mapping(value, label="root")
    _exact(root, _ROOT_KEYS, label="root")
    if root["schema_version"] != _SCHEMA:
        _fail("D1 operator intent schema_version is unsupported")
    plan_mapping = _mapping(root["plan"], label="plan")
    if plan_mapping.get("run_class") != RunClass.CONFIRMATORY.value:
        _fail("D1 operator intent requires a confirmatory plan")
    profile = _profile(plan_mapping, catalog)
    try:
        plan = parse_d1_confirmatory_plan(profile, plan_mapping)
    except D1PlanManifestError as error:
        _fail(f"D1 operator intent plan is invalid: {error}", error)
    code = plan.run.execution.code
    if code is None:
        _fail("D1 operator intent confirmatory plan lacks code identity")
    try:
        activation = resource_activation_from_mapping(root["resource_activation"])
    except ResourceProvenanceCodecGap as error:
        _fail(f"D1 operator intent resource_activation is invalid: {error}", error)
    try:
        intent = D1OperatorLaunchIntent(
            plan=plan,
            registered_design_digest=_optional_text(
                root["registered_design_digest"],
                label="registered_design_digest",
            ),
            image=_image(root["image_reference"], source_commit=code.commit),
            resource=_resource(root["resource"]),
            quoted_rate=_quoted_rate(root["quoted_rate"]),
            resource_activation=activation,
            execution_segment_index=_integer(
                root["execution_segment_index"],
                label="execution_segment_index",
            ),
            resume_checkpoint_sha256=_optional_text(
                root["resume_checkpoint_sha256"],
                label="resume_checkpoint_sha256",
            ),
            terminate_after_seconds=_integer(
                root["terminate_after_seconds"],
                label="terminate_after_seconds",
                positive=True,
            ),
        )
    except (TypeError, ValueError) as error:
        _fail(f"D1 operator intent is invalid: {error}", error)
    if intent.manifest != root:
        _fail("D1 operator intent manifest differs from recomputed canonical values")
    return intent


def serialize_d1_operator_intent(intent: D1OperatorLaunchIntent) -> str:
    if type(intent) is not D1OperatorLaunchIntent:
        raise TypeError("operator intent serializer requires D1OperatorLaunchIntent")
    return canonical_json(intent.manifest)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate D1 operator intent key {key!r}")
        result[key] = value
    return result


def parse_d1_confirmatory_operator_intent(
    payload: str,
    *,
    catalog: Iterable[D1LaunchProfile],
) -> D1OperatorLaunchIntent:
    if type(payload) is not str or not payload:
        _fail("D1 operator intent payload must be nonempty text")
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(
                f"non-finite D1 operator intent constant {constant}"
            ),
        )
    except D1OperatorIntentCodecError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        _fail("D1 operator intent payload is invalid JSON", error)
    intent = operator_intent_from_mapping(value, catalog=catalog)
    if canonical_json(value) != payload:
        _fail("D1 operator intent JSON is noncanonical")
    return intent
