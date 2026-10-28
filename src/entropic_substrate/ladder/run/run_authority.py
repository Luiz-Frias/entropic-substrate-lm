"""The sealed mandate a scientific H run executes under.

Rung B (operator ruling 2026-10-27, "option B"): a ``pilot`` or ``confirmatory`` run is
admissible only when its request carries a ``RunAuthority`` naming the stage and class it
runs, the ruling that granted it, the spend cap and GPU class it may consume, and the git
SHA of the committed preregistration it executes. The fields are the door.

The artifact is sealed by its own content digest: SHA-256 over the canonical JSON of
every field except the digest. Canonical means this module's ``json.dumps`` rules: sorted
keys, ``(",", ":")`` separators, non-ASCII escaped as ``\\uXXXX``, NaN and infinity
refused, and floats in Python's shortest round-trip ``repr`` (``40.0`` stays ``40.0``, an
int stays an int). It is NOT RFC 8785 JSON Canonicalization Scheme; a reviewer recomputing
a digest outside Python must reproduce these rules. That digest is tamper evidence, NOT
authentication:
construction refuses a digest that does not match its content, so an authority edited
without resealing cannot be built or parsed, and ``verify`` recomputes it, so an
instance altered after sealing (``object.__setattr__`` defeats ``frozen``) is caught by
the admission port, which verifies at every gate rather than trusting construction. But
``seal_run_authority`` is public and takes no secret, so anyone running this code can
mint a correctly sealed authority. The trust model is that the invoker is the operator:
the mandate is real only when its ruling and its ``registration_sha`` point at a
committed ruling and preregistration a reviewer can read, and the digest echoed in every
receipt is what lets that reviewer tie a capture back to them.

Stdlib only, like the admission port that reads it. It decides the bytes of a scientific
worker request, so it is read by the same checkpoint reader set as the port.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import cast

RUN_AUTHORITY_SCHEMA = "h-run-authority-v1"
SCIENTIFIC_RUN_CLASSES = ("pilot", "confirmatory")
_STAGES = ("H0", "H1", "H2", "H3", "H4", "H5")
_KEYS = frozenset(
    {"schema", "stage", "run_class", "ruling", "spend_cap", "gpu_class", "registration_sha"}
)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _digest(content: dict[str, object]) -> str:
    return hashlib.sha256(_json(content)).hexdigest()


def _text(value: object) -> bool:
    return type(value) is str and bool(value) and value.strip() == value


def _record(value: object, keys: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or frozenset(cast(dict[str, object], value)) != keys:
        raise ValueError("run authority fields differ from the versioned schema")
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class AuthorityRuling:
    """The operator ruling that granted the run: its date, its words and who gave it."""

    date: str
    text: str
    operator: str

    def __post_init__(self) -> None:
        valid = type(self.date) is str and re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date)
        try:
            if not valid:
                raise ValueError
            date.fromisoformat(self.date)
        except ValueError:
            raise ValueError("ruling date must be an ISO calendar date") from None
        if not _text(self.text):
            raise ValueError("ruling text must be non-empty text")
        if not _text(self.operator):
            raise ValueError("ruling operator must be non-empty text")


@dataclass(frozen=True, slots=True)
class SpendCap:
    """The most the run may spend, in US dollars and in wall-clock hours."""

    usd: float
    hours: float

    def __post_init__(self) -> None:
        if not all(_finite_positive(v) for v in (self.usd, self.hours)):
            raise ValueError("spend cap must be finite positive numbers")


def _finite_positive(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        # An int beyond float range (10**400 parses from JSON) overflows the conversion.
        return math.isfinite(cast(float, value)) and cast(float, value) > 0
    except OverflowError:
        return False


def _content(
    stage: str,
    run_class: str,
    ruling: AuthorityRuling,
    spend_cap: SpendCap,
    gpu_class: str,
    registration_sha: str,
) -> dict[str, object]:
    return {
        "schema": RUN_AUTHORITY_SCHEMA,
        "stage": stage,
        "run_class": run_class,
        "ruling": {"date": ruling.date, "text": ruling.text, "operator": ruling.operator},
        "spend_cap": {"usd": spend_cap.usd, "hours": spend_cap.hours},
        "gpu_class": gpu_class,
        "registration_sha": registration_sha,
    }


@dataclass(frozen=True, slots=True)
class RunAuthority:
    stage: str
    run_class: str
    ruling: AuthorityRuling
    spend_cap: SpendCap
    # The RunPod GPU type id. Any safe identifier is admitted and recorded: the provider
    # catalogue changes faster than this module, and the launch binds it to the pod.
    gpu_class: str
    registration_sha: str
    digest: str

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError("run authority stage is not an H stage")
        if self.run_class not in SCIENTIFIC_RUN_CLASSES:
            raise ValueError("run authority names a non-scientific run class")
        if type(self.ruling) is not AuthorityRuling:
            raise TypeError("run authority ruling must be an AuthorityRuling")
        if type(self.spend_cap) is not SpendCap:
            raise TypeError("run authority spend cap must be a SpendCap")
        gpu = self.gpu_class
        if (
            not _text(gpu)
            or not gpu[0].isalnum()
            or len(gpu) > 128
            or any(not (c.isascii() and (c.isalnum() or c in " ._-")) for c in gpu)
        ):
            raise ValueError("run authority GPU class must be a safe explicit identifier")
        if (
            type(self.registration_sha) is not str
            or re.fullmatch(r"[0-9a-f]{40}", self.registration_sha) is None
        ):
            raise ValueError("registration SHA must be a full 40-hex git SHA")
        if type(self.digest) is not str or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None:
            raise ValueError("run authority digest must be 64 lowercase hex")
        if self.digest != self.content_digest:
            raise ValueError("run authority digest does not match its content")

    @property
    def content(self) -> dict[str, object]:
        """Every sealed field except the digest, as the digest reads them."""
        return _content(
            self.stage,
            self.run_class,
            self.ruling,
            self.spend_cap,
            self.gpu_class,
            self.registration_sha,
        )

    @property
    def content_digest(self) -> str:
        return _digest(self.content)

    @property
    def manifest(self) -> dict[str, object]:
        return {**self.content, "digest": self.digest}

    @property
    def payload(self) -> bytes:
        return _json(self.manifest)

    def verify(self) -> bool:
        """Whether this instance still holds what was sealed; never raises."""
        try:
            AuthorityRuling(self.ruling.date, self.ruling.text, self.ruling.operator)
            SpendCap(self.spend_cap.usd, self.spend_cap.hours)
            RunAuthority(
                self.stage,
                self.run_class,
                self.ruling,
                self.spend_cap,
                self.gpu_class,
                self.registration_sha,
                self.digest,
            )
        except (TypeError, ValueError, AttributeError):
            return False
        return True


def seal_run_authority(
    *,
    stage: str,
    run_class: str,
    ruling: AuthorityRuling,
    spend_cap: SpendCap,
    gpu_class: str,
    registration_sha: str,
) -> RunAuthority:
    """Seal the fields under their own content digest."""
    if type(ruling) is not AuthorityRuling:
        raise TypeError("run authority ruling must be an AuthorityRuling")
    if type(spend_cap) is not SpendCap:
        raise TypeError("run authority spend cap must be a SpendCap")
    try:
        content = _content(stage, run_class, ruling, spend_cap, gpu_class, registration_sha)
        digest = _digest(content)
    except (TypeError, ValueError):
        digest = ""  # unserializable fields: construction below names the bad one
    return RunAuthority(stage, run_class, ruling, spend_cap, gpu_class, registration_sha, digest)


def parse_run_authority(value: object) -> RunAuthority:
    """Rebuild a sealed authority from its manifest; the digest must match the content."""
    row = _record(value, _KEYS | {"digest"})
    if row["schema"] != RUN_AUTHORITY_SCHEMA:
        raise ValueError("run authority schema is not admitted")
    ruling = _record(row["ruling"], frozenset({"date", "text", "operator"}))
    cap = _record(row["spend_cap"], frozenset({"usd", "hours"}))
    return RunAuthority(
        stage=cast(str, row["stage"]),
        run_class=cast(str, row["run_class"]),
        ruling=AuthorityRuling(
            cast(str, ruling["date"]), cast(str, ruling["text"]), cast(str, ruling["operator"])
        ),
        spend_cap=SpendCap(cast(float, cap["usd"]), cast(float, cap["hours"])),
        gpu_class=cast(str, row["gpu_class"]),
        registration_sha=cast(str, row["registration_sha"]),
        digest=cast(str, row["digest"]),
    )
