"""Persisted design declarations must re-earn factory admission from evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import pytest

from d1_launch_admission_fixtures import PREREG_COMMIT, PREREG_PATH, SEEDS, contract, family
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.ladder.register.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)
from entropic_substrate.ladder.register.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    D1RegistrationCodecError,
    hydrate_registered_design,
    parse_registered_design_declaration,
    registered_design_declaration_from_mapping,
    registered_design_to_mapping,
    serialize_registered_design,
)

pytestmark = pytest.mark.unit

_RAW_PREREGISTRATION = b"# D1 preregistration\n\nBinding test bytes.\n"


def _design() -> D1RegisteredDesign:
    preregistration = D1PreregistrationArtifact(
        path=PREREG_PATH,
        blob_sha256=hashlib.sha256(_RAW_PREREGISTRATION).hexdigest(),
        commit=PREREG_COMMIT,
    )
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=contract(),
            registered_seeds=SEEDS,
            preregistration=preregistration,
        )
    )
    assert type(admitted) is D1RegisteredDesign
    return admitted


def _evidence(payload: bytes = _RAW_PREREGISTRATION) -> D1PreregistrationByteEvidence:
    return D1PreregistrationByteEvidence.from_bytes(payload)


def _hydrate(payload: str, *, expected_digest: str | None = None) -> D1RegisteredDesign:
    return hydrate_registered_design(
        payload,
        expected_design_digest=_design().design_digest
        if expected_digest is None
        else expected_digest,
        family=family(),
        decision_contract=contract(),
        preregistration_evidence=_evidence(),
    )


def test_canonical_design_declaration_rehydrates_only_through_existing_admission() -> None:
    original = _design()
    mapping = registered_design_to_mapping(original)
    payload = serialize_registered_design(original)
    declaration = parse_registered_design_declaration(payload)
    hydrated = _hydrate(payload)

    assert declaration.manifest == mapping == original.manifest
    assert hydrated == original
    assert hydrated.design_digest == original.design_digest
    assert payload == json.dumps(
        original.manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _unknown(value: dict[str, object]) -> None:
    value["invented"] = True


def _missing(value: dict[str, object]) -> None:
    value.pop("registered_seeds")


def _prereg_unknown(value: dict[str, object]) -> None:
    value["preregistration"]["invented"] = True  # type: ignore[index]


def _seed_collection(value: dict[str, object]) -> None:
    value["registered_seeds"] = tuple(value["registered_seeds"])  # type: ignore[arg-type]


@pytest.mark.parametrize("mutate", [_unknown, _missing, _prereg_unknown, _seed_collection])
def test_design_mapping_rejects_unknown_missing_or_non_json_collection_fields(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    mapping = registered_design_to_mapping(_design())
    mutate(mapping)

    with pytest.raises(D1RegistrationCodecError):
        registered_design_declaration_from_mapping(mapping)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"d1-registered-design-v1","schema_version":"x"}',
        " {}",
        "{}\n",
        "[]",
        '{"schema_version":NaN}',
    ],
)
def test_design_text_parser_rejects_duplicate_noncanonical_nonobject_or_nonfinite_json(
    payload: str,
) -> None:
    with pytest.raises(D1RegistrationCodecError):
        parse_registered_design_declaration(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("family_digest", "0" * 64),
        ("decision_contract_digest", "1" * 64),
        ("registered_seeds", [1, 2, 3, 4]),
    ],
)
def test_hydration_rejects_family_contract_or_seed_drift_even_with_matching_outer_digest(
    field: str, replacement: object
) -> None:
    mapping = registered_design_to_mapping(_design())
    mapping[field] = replacement
    payload = json.dumps(mapping, separators=(",", ":"), sort_keys=True)

    with pytest.raises(D1RegistrationCodecError):
        _hydrate(payload, expected_digest=canonical_digest(mapping))


def test_hydration_rejects_arm_and_preregistration_identity_drift() -> None:
    mapping = registered_design_to_mapping(_design())
    mapping["arm_construction_digests"]["sg"] = "2" * 64  # type: ignore[index]
    payload = json.dumps(mapping, separators=(",", ":"), sort_keys=True)
    with pytest.raises(D1RegistrationCodecError, match="arm"):
        _hydrate(payload, expected_digest=canonical_digest(mapping))

    for field, replacement in (
        ("path", "docs/other-preregistration.md"),
        ("commit", "3" * 40),
        ("blob_sha256", "4" * 64),
    ):
        mapping = registered_design_to_mapping(_design())
        mapping["preregistration"][field] = replacement  # type: ignore[index]
        payload = json.dumps(mapping, separators=(",", ":"), sort_keys=True)
        with pytest.raises(D1RegistrationCodecError):
            _hydrate(payload)


def test_raw_preregistration_bytes_and_expected_design_digest_are_independent_guards() -> None:
    payload = serialize_registered_design(_design())
    with pytest.raises(D1RegistrationCodecError, match="raw preregistration"):
        hydrate_registered_design(
            payload,
            expected_design_digest=_design().design_digest,
            family=family(),
            decision_contract=contract(),
            preregistration_evidence=_evidence(b"different bytes"),
        )
    with pytest.raises(D1RegistrationCodecError, match="design digest"):
        _hydrate(payload, expected_digest="5" * 64)


def test_preregistration_byte_evidence_rejects_size_or_sha_forgery() -> None:
    digest = hashlib.sha256(_RAW_PREREGISTRATION).hexdigest()
    with pytest.raises(ValueError, match="size"):
        D1PreregistrationByteEvidence(_RAW_PREREGISTRATION, len(_RAW_PREREGISTRATION) + 1, digest)
    with pytest.raises(ValueError, match="sha256"):
        D1PreregistrationByteEvidence(
            _RAW_PREREGISTRATION,
            len(_RAW_PREREGISTRATION),
            "6" * 64,
        )


def test_design_and_expected_digests_require_lowercase_canonical_spelling() -> None:
    mapping = registered_design_to_mapping(_design())
    mapping["decision_contract_digest"] = str(mapping["decision_contract_digest"]).upper()
    with pytest.raises(D1RegistrationCodecError, match="lowercase"):
        registered_design_declaration_from_mapping(mapping)

    with pytest.raises(D1RegistrationCodecError, match="lowercase"):
        _hydrate(
            serialize_registered_design(_design()),
            expected_digest=_design().design_digest.upper(),
        )


def test_preregistration_byte_evidence_rejects_uppercase_sha_identity_explicitly() -> None:
    digest = hashlib.sha256(_RAW_PREREGISTRATION).hexdigest().upper()
    with pytest.raises(ValueError, match="lowercase"):
        D1PreregistrationByteEvidence(
            _RAW_PREREGISTRATION,
            len(_RAW_PREREGISTRATION),
            digest,
        )
