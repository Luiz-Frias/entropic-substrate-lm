"""Strict shared request for D1 campaign execution and capture commands."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, cast

from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    canonical_json,
    require_sha256_hex_any_case,
)
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile
from entropic_substrate.ladder.register.d1_registered_design import (
    decision_contract_digest,
)
from entropic_substrate.ladder.register.d1_registration_codec import (
    D1RegisteredDesignDeclaration,
    D1RegistrationCodecError,
    decision_contract_from_mapping,
    decision_contract_to_mapping,
    registered_design_declaration_from_mapping,
)
from entropic_substrate.ladder.run.campaign.d1_operator_intent_codec import (
    D1OperatorIntentCodecError,
    parse_d1_confirmatory_operator_intent,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1OperatorLaunchIntent

__all__ = [
    "D1CampaignCommandRequest",
    "D1CampaignLocations",
    "D1CampaignRequestError",
    "D1FreshCampaignCursor",
    "D1PersistedCampaignCursor",
    "parse_d1_campaign_request",
    "serialize_d1_campaign_request",
]

_SCHEMA: Final = "d1-campaign-command-request-v1"
_ROOT_KEYS: Final = frozenset(
    {
        "cursor",
        "decision_contract",
        "intents",
        "locations",
        "registered_design",
        "schema_version",
    }
)
_LOCATION_KEYS: Final = frozenset(
    {
        "artifact_root",
        "campaign_checkpoint_path",
        "capture_preparation_custody_root",
        "finalized_cell_custody_root",
        "lifecycle_path",
        "repo_root",
    }
)


class D1CampaignRequestError(ValueError):
    """The campaign command request is malformed, contradictory, or noncanonical."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1CampaignRequestError(message)
    raise D1CampaignRequestError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"D1 campaign {label} must be an object with string keys")
    return cast(dict[str, object], value)


def _exact(value: Mapping[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"D1 campaign {label} keys must be exact")


def _path(value: object, *, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        _fail(f"D1 campaign {label} must be a nonempty NUL-free path")
    return Path(value)


@dataclass(frozen=True, slots=True)
class D1FreshCampaignCursor:
    @property
    def manifest(self) -> dict[str, object]:
        return {"kind": "fresh"}


@dataclass(frozen=True, slots=True)
class D1PersistedCampaignCursor:
    sha256: str

    def __post_init__(self) -> None:
        require_sha256_hex_any_case(self.sha256, label="D1 persisted campaign checkpoint")

    @property
    def manifest(self) -> dict[str, object]:
        return {"kind": "persisted", "sha256": self.sha256}


type D1CampaignCursor = D1FreshCampaignCursor | D1PersistedCampaignCursor


@dataclass(frozen=True, slots=True)
class D1CampaignLocations:
    repo_root: Path
    artifact_root: Path
    lifecycle_path: Path
    checkpoint_path: Path
    finalized_cell_custody_root: Path
    capture_preparation_custody_root: Path

    def __post_init__(self) -> None:
        values = (
            self.repo_root,
            self.artifact_root,
            self.lifecycle_path,
            self.checkpoint_path,
            self.finalized_cell_custody_root,
            self.capture_preparation_custody_root,
        )
        if any(
            not isinstance(cast(object, value), Path) or not str(value) or "\x00" in str(value)
            for value in values
        ):
            raise ValueError("D1 campaign locations must be nonempty NUL-free Paths")
        if len(set(values)) != len(values):
            raise ValueError("D1 campaign locations must be distinct")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "artifact_root": str(self.artifact_root),
            "campaign_checkpoint_path": str(self.checkpoint_path),
            "capture_preparation_custody_root": str(self.capture_preparation_custody_root),
            "finalized_cell_custody_root": str(self.finalized_cell_custody_root),
            "lifecycle_path": str(self.lifecycle_path),
            "repo_root": str(self.repo_root),
        }

    def resolve(self, base: Path) -> D1CampaignLocations:
        if not isinstance(cast(object, base), Path) or not base.is_dir() or base.is_symlink():
            raise ValueError("D1 campaign manifest base must be a real directory")

        def resolved(value: Path) -> Path:
            candidate = value if value.is_absolute() else base / value
            return Path(os.path.abspath(candidate))

        return D1CampaignLocations(
            *(
                resolved(value)
                for value in (
                    self.repo_root,
                    self.artifact_root,
                    self.lifecycle_path,
                    self.checkpoint_path,
                    self.finalized_cell_custody_root,
                    self.capture_preparation_custody_root,
                )
            )
        )


def _validate_request(request: D1CampaignCommandRequest) -> None:
    design = request.design
    contract = request.decision_contract
    if decision_contract_digest(contract) != design.decision_contract_digest:
        raise ValueError("D1 campaign decision contract differs from registered design")
    if contract.family_identity.family_digest != design.family_identity.family_digest or dict(
        contract.family_identity.arm_construction_digests
    ) != dict(design.family_identity.arm_construction_digests):
        raise ValueError("D1 campaign decision family differs from registered design")
    expected = tuple((arm, seed) for seed in design.registered_seeds for arm in D1Arm)
    actual = tuple((intent.plan.profile.arm, intent.plan.seed) for intent in request.intents)
    if actual != expected:
        raise ValueError("D1 campaign intent roster differs from registered seed/arm order")
    for intent in request.intents:
        plan = intent.plan
        preregistration = plan.run.preregistration
        if (
            intent.registered_design_digest != design.design_digest
            or intent.execution_segment_index != 0
            or plan.profile.family.family_digest != design.family_identity.family_digest
            or plan.run.construction_digest
            != design.family_identity.arm_construction_digests[plan.profile.arm.value]
            or preregistration is None
            or (
                preregistration.path,
                preregistration.commit,
                preregistration.family_digest,
            )
            != (
                design.preregistration.path,
                design.preregistration.commit,
                design.family_identity.family_digest,
            )
        ):
            raise ValueError("D1 campaign intent authority differs from registered design")


@dataclass(frozen=True, slots=True)
class D1CampaignCommandRequest:
    design: D1RegisteredDesignDeclaration
    decision_contract: D1DecisionContract
    intents: tuple[D1OperatorLaunchIntent, ...]
    cursor: D1CampaignCursor
    locations: D1CampaignLocations

    def __post_init__(self) -> None:
        if type(self.design) is not D1RegisteredDesignDeclaration:
            raise TypeError("D1 campaign request requires design declaration")
        if type(self.decision_contract) is not D1DecisionContract:
            raise TypeError("D1 campaign request requires decision contract")
        if type(self.intents) is not tuple or any(
            type(item) is not D1OperatorLaunchIntent for item in self.intents
        ):
            raise TypeError("D1 campaign request requires immutable operator intents")
        if type(self.cursor) not in (D1FreshCampaignCursor, D1PersistedCampaignCursor):
            raise TypeError("D1 campaign request requires a nominal cursor")
        if type(self.locations) is not D1CampaignLocations:
            raise TypeError("D1 campaign request requires campaign locations")
        _validate_request(self)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "cursor": self.cursor.manifest,
            "decision_contract": decision_contract_to_mapping(self.decision_contract),
            "intents": [item.manifest for item in self.intents],
            "locations": self.locations.manifest,
            "registered_design": self.design.manifest,
            "schema_version": _SCHEMA,
        }


def _cursor(value: object) -> D1CampaignCursor:
    item = _mapping(value, label="cursor")
    kind = item.get("kind")
    if kind == "fresh":
        _exact(item, frozenset({"kind"}), label="fresh cursor")
        return D1FreshCampaignCursor()
    if kind == "persisted":
        _exact(item, frozenset({"kind", "sha256"}), label="persisted cursor")
        try:
            return D1PersistedCampaignCursor(cast(str, item["sha256"]))
        except (TypeError, ValueError) as error:
            _fail(f"D1 campaign persisted cursor is invalid: {error}", error)
    _fail("D1 campaign cursor kind is unsupported")


def _locations(value: object) -> D1CampaignLocations:
    item = _mapping(value, label="locations")
    _exact(item, _LOCATION_KEYS, label="locations")
    try:
        return D1CampaignLocations(
            _path(item["repo_root"], label="repo_root"),
            _path(item["artifact_root"], label="artifact_root"),
            _path(item["lifecycle_path"], label="lifecycle_path"),
            _path(item["campaign_checkpoint_path"], label="campaign_checkpoint_path"),
            _path(
                item["finalized_cell_custody_root"],
                label="finalized_cell_custody_root",
            ),
            _path(
                item["capture_preparation_custody_root"],
                label="capture_preparation_custody_root",
            ),
        )
    except (TypeError, ValueError) as error:
        _fail(f"D1 campaign locations are invalid: {error}", error)


def campaign_request_from_mapping(
    value: object,
    *,
    catalog: Iterable[D1LaunchProfile],
) -> D1CampaignCommandRequest:
    root = _mapping(value, label="root")
    _exact(root, _ROOT_KEYS, label="root")
    if root["schema_version"] != _SCHEMA:
        _fail("D1 campaign request schema_version is unsupported")
    raw_intents = root["intents"]
    if type(raw_intents) is not list or not raw_intents:
        _fail("D1 campaign intents must be a nonempty array")
    try:
        design = registered_design_declaration_from_mapping(root["registered_design"])
        contract = decision_contract_from_mapping(root["decision_contract"])
        intents = tuple(
            parse_d1_confirmatory_operator_intent(
                canonical_json(item),
                catalog=catalog,
            )
            for item in raw_intents
        )
        return D1CampaignCommandRequest(
            design,
            contract,
            intents,
            _cursor(root["cursor"]),
            _locations(root["locations"]),
        )
    except D1CampaignRequestError:
        raise
    except (
        D1OperatorIntentCodecError,
        D1RegistrationCodecError,
        TypeError,
        ValueError,
    ) as error:
        _fail(f"D1 campaign request authority is invalid: {error}", error)


def serialize_d1_campaign_request(request: D1CampaignCommandRequest) -> str:
    if type(request) is not D1CampaignCommandRequest:
        raise TypeError("campaign request serializer requires D1CampaignCommandRequest")
    return canonical_json(request.manifest)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate D1 campaign request key {key!r}")
        result[key] = value
    return result


def parse_d1_campaign_request(
    payload: str,
    *,
    catalog: Iterable[D1LaunchProfile],
) -> D1CampaignCommandRequest:
    if type(payload) is not str or not payload:
        _fail("D1 campaign request payload must be nonempty text")
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: _fail(
                f"non-finite D1 campaign request constant {constant}"
            ),
        )
    except D1CampaignRequestError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        _fail("D1 campaign request payload is invalid JSON", error)
    request = campaign_request_from_mapping(value, catalog=catalog)
    if canonical_json(value) != payload:
        _fail("D1 campaign request JSON is noncanonical")
    return request
