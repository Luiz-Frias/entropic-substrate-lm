"""p14-j-register: the register and gate codecs adopt kernel.codec.strict_json.

Three groups, one per ticket facet (FOR-1575/FOR-1599, FOR-1577, FOR-1613):

1. adoption  -- an AST walk finds no local ``_pairs``-style strict-JSON hook
   definition left under ``ladder/register`` or ``ladder/gate``.
2. codec boundary -- the two JSON codecs raise their OWN exception type for
   duplicate keys, non-finite constants, overflow floats and malformed JSON,
   because ``fail=`` is wired to the codec's own ``_fail``.
3. SHA boundary -- every repointed digest site refuses a mixed-case digest
   with that site's own exception type.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from d1_launch_admission_fixtures import (
    CODE_COMMIT,
    PREREG_COMMIT,
    PREREG_PATH,
    SAMPLER_DIGEST,
    SEEDS,
    SPLIT_DIGEST,
    contract,
    family,
    plan,
    profiles,
    start,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.ladder.gate import ladder_campaign
from entropic_substrate.ladder.register.d1_confirmatory_matrix import D1ConfirmatoryLaunch
from entropic_substrate.ladder.register.d1_launch_attempt_codec import (
    D1LaunchAttemptCodecError,
    parse_launch_attempt_ledger,
)
from entropic_substrate.ladder.register.d1_launch_attempts import D1LaunchAttemptIdentity
from entropic_substrate.ladder.register.d1_plan import D1PlanManifestError
from entropic_substrate.ladder.register.d1_provenance import (
    CodeIdentity,
    ContainerIdentity,
    DataIdentity,
    DataOrderAttestation,
    PreregistrationIdentity,
)
from entropic_substrate.ladder.register.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)
from entropic_substrate.ladder.register.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    D1RegisteredDesignDeclaration,
    D1RegistrationCodecError,
    hydrate_registered_design,
    parse_decision_contract,
    parse_registered_design_declaration,
    serialize_registered_design,
)
from entropic_substrate.ladder.register.ladder import ParameterMatchReference

pytestmark = pytest.mark.unit

_REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "entropic_substrate" / "ladder"
_SCANNED = (_REPO_SRC / "register", _REPO_SRC / "gate")
_FORBIDDEN = frozenset(
    {"_pairs", "_design_pairs", "_strict_pairs", "_nonfinite", "_reject_constant"}
)

_DIGEST = "1" * 64
_MIXED = "A" + "1" * 63
_RAW_PREREGISTRATION = b"# D1 preregistration\n\nBinding test bytes.\n"


# --------------------------------------------------------------------------
# 1. adoption: no local strict-JSON hook definitions


def _forbidden_definitions(source: str) -> list[str]:
    return [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in _FORBIDDEN
    ]


def test_the_ast_walk_detects_a_local_hook_definition_at_any_depth() -> None:
    nested = "def outer():\n    def _pairs(pairs):\n        return {}\n"
    assert _forbidden_definitions(nested) == ["_pairs"]
    assert _forbidden_definitions("def parse():\n    return 1\n") == []


def test_register_and_gate_define_no_local_strict_json_hooks() -> None:
    files = [path for root in _SCANNED for path in sorted(root.rglob("*.py"))]
    assert len(files) > 10, "scan is misrooted: positive control on the file count failed"
    found = {
        path.relative_to(_REPO_SRC).as_posix(): names
        for path in files
        if (names := _forbidden_definitions(path.read_text(encoding="utf-8")))
    }
    assert found == {}


# --------------------------------------------------------------------------
# 2. codec boundary: strict_loads(fail=<codec _fail>) keeps each exception type

_LEDGER_PREFIX = '{"events":[],"schema_version":"d1-launch-ledger-v1"'


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ('{"events":[],"events":[],"schema_version":"x"}', "'events'"),
        ('{"events":[],"schema_version":NaN}', "NaN"),
        ('{"events":[],"schema_version":-Infinity}', "Infinity"),
        ('{"events":[],"schema_version":1e999}', "1e999"),
    ],
)
def test_launch_attempt_codec_refuses_strict_json_violations_with_its_own_error(
    payload: str, needle: str
) -> None:
    with pytest.raises(D1LaunchAttemptCodecError) as caught:
        parse_launch_attempt_ledger(payload)
    assert needle in str(caught.value)


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ('{"arm_ids":[],"arm_ids":[]}', "'arm_ids'"),
        ('{"schema_version":NaN}', "NaN"),
        ('{"schema_version":Infinity}', "Infinity"),
        ('{"schema_version":1e999}', "1e999"),
    ],
)
def test_registration_codec_refuses_strict_json_violations_with_its_own_error(
    payload: str, needle: str
) -> None:
    with pytest.raises(D1RegistrationCodecError) as caught:
        parse_decision_contract(payload)
    assert needle in str(caught.value)
    with pytest.raises(D1RegistrationCodecError) as design_caught:
        parse_registered_design_declaration(payload)
    assert needle in str(design_caught.value)


def test_decoder_level_errors_are_wrapped_at_each_codec_boundary() -> None:
    with pytest.raises(D1LaunchAttemptCodecError, match="invalid JSON") as launch_caught:
        parse_launch_attempt_ledger("{")
    assert isinstance(launch_caught.value.__cause__, json.JSONDecodeError)
    with pytest.raises(D1RegistrationCodecError, match="invalid JSON") as contract_caught:
        parse_decision_contract("{")
    assert isinstance(contract_caught.value.__cause__, json.JSONDecodeError)
    with pytest.raises(D1RegistrationCodecError, match="invalid JSON") as design_caught:
        parse_registered_design_declaration("{")
    assert isinstance(design_caught.value.__cause__, json.JSONDecodeError)


# --------------------------------------------------------------------------
# 3. SHA boundary: a mixed-case digest raises the site's own exception


def _design() -> D1RegisteredDesign:
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=contract(),
            registered_seeds=SEEDS,
            preregistration=D1PreregistrationArtifact(
                path=PREREG_PATH,
                blob_sha256=hashlib.sha256(_RAW_PREREGISTRATION).hexdigest(),
                commit=PREREG_COMMIT,
            ),
        )
    )
    assert type(admitted) is D1RegisteredDesign
    return admitted


def _launch() -> D1ConfirmatoryLaunch:
    run_plan = plan(profiles()[0], SEEDS[0])
    return D1ConfirmatoryLaunch(
        plan=run_plan,
        start=start(run_plan),
        registered_design_digest=_DIGEST,
        decision_contract_digest=_DIGEST,
    )


def _attempt(**overrides: object) -> D1LaunchAttemptIdentity:
    fields: dict[str, object] = {
        "run_class": RunClass.CONFIRMATORY,
        "family_digest": _DIGEST,
        "arm": D1Arm.SG,
        "seed": 1,
        "logical_run_id": "run-1",
        "registered_design_digest": _DIGEST,
    }
    fields.update(overrides)
    return D1LaunchAttemptIdentity(**fields)  # type: ignore[arg-type]


def _declaration() -> D1RegisteredDesignDeclaration:
    return parse_registered_design_declaration(serialize_registered_design(_design()))


def _hydrate_with_expected_digest(digest: str) -> object:
    return hydrate_registered_design(
        serialize_registered_design(_design()),
        expected_design_digest=digest,
        family=family(),
        decision_contract=contract(),
        preregistration_evidence=D1PreregistrationByteEvidence.from_bytes(_RAW_PREREGISTRATION),
    )


_SITES: list[tuple[str, type[Exception], str, Callable[[], object]]] = [
    (
        "matrix.registered_design_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: replace(_launch(), registered_design_digest=_MIXED),
    ),
    (
        "matrix.decision_contract_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: replace(_launch(), decision_contract_digest=_MIXED),
    ),
    (
        "attempts.family_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: _attempt(family_digest=_MIXED),
    ),
    (
        "attempts.registered_design_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: _attempt(registered_design_digest=_MIXED),
    ),
    (
        "plan.data_dataset_digest",
        D1PlanManifestError,
        "data.dataset_digest",
        lambda: plan(
            profiles()[0],
            SEEDS[0],
            data={
                "dataset_name": "enwik8",
                "dataset_digest": _MIXED,
                "split_digest": SPLIT_DIGEST,
                "sampler_construction_digest": SAMPLER_DIGEST,
            },
        ),
    ),
    (
        "provenance.dataset_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: DataIdentity("enwik8", _MIXED, _DIGEST, _DIGEST),
    ),
    (
        "provenance.split_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: DataIdentity("enwik8", _DIGEST, _MIXED, _DIGEST),
    ),
    (
        "provenance.sampler_construction_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: DataIdentity("enwik8", _DIGEST, _DIGEST, _MIXED),
    ),
    (
        "provenance.data_order_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: DataOrderAttestation(_MIXED),
    ),
    (
        "provenance.repository_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: CodeIdentity(CODE_COMMIT, _MIXED),
    ),
    (
        "provenance.image_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: ContainerIdentity("sha256:" + _MIXED),
    ),
    (
        "provenance.preregistration_family_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: PreregistrationIdentity(PREREG_PATH, PREREG_COMMIT, _MIXED),
    ),
    (
        "design.family_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: replace(_design(), family_digest=_MIXED),
    ),
    (
        "design.decision_contract_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: replace(_design(), decision_contract_digest=_MIXED),
    ),
    (
        "design.arm_construction_digest",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: replace(
            _design(),
            arm_construction_digests={
                **_design().arm_construction_digests,
                D1Arm.SG.value: _MIXED,
            },
        ),
    ),
    (
        "ladder.construction_identity",
        ValueError,
        "must be a 64-character hexadecimal SHA-256 digest",
        lambda: ParameterMatchReference(_MIXED, 1),
    ),
    (
        "codec.byte_evidence_sha256",
        ValueError,
        "preregistration sha256 must be lowercase SHA-256",
        lambda: replace(
            D1PreregistrationByteEvidence.from_bytes(_RAW_PREREGISTRATION), sha256=_MIXED
        ),
    ),
    (
        "codec.declaration_decision_contract_digest",
        ValueError,
        "decision_contract_digest",
        lambda: replace(_declaration(), decision_contract_digest=_MIXED),
    ),
    (
        "codec.expected_design_digest",
        D1RegistrationCodecError,
        "expected design digest must be lowercase SHA-256",
        lambda: _hydrate_with_expected_digest(_MIXED),
    ),
    (
        "campaign.ladder_identity",
        ValueError,
        "ladder identity",
        lambda: ladder_campaign._sha(_MIXED),  # pyright: ignore[reportPrivateUsage]
    ),
]


@pytest.mark.parametrize(
    ("exception", "message", "build"),
    [pytest.param(exc, rx, build, id=site) for site, exc, rx, build in _SITES],
)
def test_boundary_sha_sites_refuse_a_mixed_case_digest(
    exception: type[Exception], message: str, build: Callable[[], object]
) -> None:
    with pytest.raises(exception, match=message):
        build()


def test_every_boundary_site_still_accepts_its_lowercase_digest() -> None:
    _launch()
    _attempt()
    plan(profiles()[0], SEEDS[0])
    DataIdentity("enwik8", _DIGEST, _DIGEST, _DIGEST)
    DataOrderAttestation(_DIGEST)
    CodeIdentity(CODE_COMMIT, _DIGEST)
    ContainerIdentity("sha256:" + _DIGEST)
    PreregistrationIdentity(PREREG_PATH, PREREG_COMMIT, _DIGEST)
    ParameterMatchReference(_DIGEST, 1)
    _declaration()
    assert ladder_campaign._sha(_DIGEST) == _DIGEST  # pyright: ignore[reportPrivateUsage]
    _hydrate_with_expected_digest(_design().design_digest)


@pytest.mark.parametrize("value", [_MIXED, "1" * 63, "1" * 65, "g" * 64, "", 12345, None])
def test_require_lowercase_digest_refuses_with_a_labelled_value_error(value: object) -> None:
    from entropic_substrate.ladder.register.d1_provenance import require_lowercase_digest

    with pytest.raises(ValueError, match="my digest must be a 64-character hexadecimal"):
        require_lowercase_digest(value, label="my digest")


def test_require_lowercase_digest_accepts_a_lowercase_digest() -> None:
    from entropic_substrate.ladder.register.d1_provenance import require_lowercase_digest

    require_lowercase_digest(_DIGEST, label="my digest")
