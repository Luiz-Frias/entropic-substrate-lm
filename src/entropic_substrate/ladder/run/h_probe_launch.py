"""Explicit H deployment choices for the pinned Runpod command boundary.

LAUNCH SCHEMA v5 (FOR-1694, h1-l4c, C-L4c-1, 2026-11-02). Three moves, one bump. (1) The
launch SEALS its price: a scientific launch declares ``hourly_price_ceiling`` at construction
(an explicit field, no default, never derived from the authority: ``RunAuthority`` carries a
USD cap and an hours cap and no hourly price, so ``usd / hours`` would quietly replace the
ceiling the invoker approved with a looser number nobody typed) and the existing product rule
``ceiling * watchdog_timeout_seconds / 3600 <= spend_cap.usd`` is re-checked at construction
and at parse. The manifest also records ``hours_cap``, copied from ``spend_cap.hours`` and
re-derived at parse. A smoke launch carries explicit ``null`` for both. ``h_probe_command``
and the controller read the ceiling from here: a ``--max-hourly-rate`` flag (or controller
argument) that is present and not EXACTLY equal to the sealed value is refused by name, before
any workspace or provider call; a smoke, v4 or v3 launch keeps the flag with its v3 semantics.
(2) The launch carries a codec-sharing PAIR (``PairProbeRequest``): the manifest records the
pair flag and both cell identities (donor first), re-derived from the embedded request at
parse; a pair under a v3 or v4 manifest is refused by name, because pairs exist only in v5.
(3) The v2 read retires, as the v3 section below promised at the next bump: a
``h0-probe-launch-v2`` payload is refused by name and the refusal names
``h0-probe-launch-v3`` as the minimum supported version. v4 and v3 still read as the same
class (every H0 launch file sealed on R2 is v3), re-render their own bytes and never compare
equal to a v5 launch. Every launch built in-process renders v5, so the digest, the pod name
(``<stage>-<class>-<digest prefix>``, the rule is unchanged) and the workspace of a freshly
built launch moved; the frozen launch fixtures in ``tests/test_h_probe_launch.py`` were
re-frozen once with the old digests retracted in the test docstring (house rule 4).

LAUNCH SCHEMA v4 (rung B, operator ruling 2026-10-27, "option B"). A launch is either
a smoke launch (``remote_tiny`` at H0, exactly the v3 shape and rules) or a scientific
launch (``pilot``/``confirmatory`` at H1-H5) whose request carries a sealed
``RunAuthority``. The manifest names that authority's digest as both its
``scientific_authority`` and its ``spending_authority`` (``"none"`` for smoke). A
scientific launch declares the ``host_watchdog_budgeted`` cost guard, runs on exactly
the authority's GPU class, keeps its watchdog inside the authority's hours, and its pod
environment drops the tiny MLflow opt-in: the seven-key base minus
``ENTROPIC_MLFLOW_TINY`` (six keys) plus the declared extensions, eight keys for the H
wave. Its pod is named ``<stage>-<class>-<digest prefix>``; a smoke pod keeps
``h0-tiny-``. The class keeps its ``H0ProbeLaunch`` name (72 references); the rename
is a follow-up, not part of this schema move.

READ-OLD v3 (documentation of an in-tree parse, NOT a reader-identity bump): a
``h0-probe-launch-v3`` payload reads as the same class with
``manifest_schema == LAUNCH_SCHEMA_V3``, re-renders its own bytes, and never compares
equal to a v4 or v5 launch; a pre-v4 payload carries no run authority, so a v3 launch of a
scientific request is refused. (v4 was the in-process rendering until v5.)

LAUNCH SCHEMA v3 (FOR-1647, DN-OBS-TEMPLATE, operator 2026-10-13). The pod
environment is a seven-key BASE (four R2/MLflow secret references, the S3
endpoint reference, the tiny opt-in and the tracking URI) plus DECLARED
EXTENSIONS drawn from a CLOSED registry (``EXTENSION_REGISTRY``: an immutable
module constant, no registration function, no discovery, no env or CLI
selection; adding an extension is a source edit). Each extension contributes
exactly one environment key. The manifest records each extension's name AND
its parameters, so ``parse_probe_launch`` rebuilds an identical environment.
The H0 wave launches with ``named_experiment("h-series")`` and
``system_metrics()``: nine keys.

READ-OLD (RETIRED by v5, FOR-1694): ``parse_probe_launch`` used to read a
``h0-probe-launch-v2`` payload as the same class with no extensions. That window closed at the
v5 bump it was promised for: a v2 payload is now refused by name, naming
``h0-probe-launch-v3`` as the minimum supported version. v1 stays refused.
"""

import hashlib
import json
import math
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import cast

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.ladder.run.admission import RUN_ADMISSION
from entropic_substrate.ladder.run.worker_request import (
    AnyProbeRequest,
    PairProbeRequest,
    ProbeRequest,
    parse_probe_request,
    refuse_version,
)

RUNPODCTL_VERSION = "runpodctl 3.0.1-551d6a4"
RUNPODCTL_SHA256 = (
    "da72de4c4a2deaeb3642378e7fbb71304f89463a5c7b3af7c7459d6683391696"  # pragma: allowlist secret
)


LAUNCH_SCHEMA_V2 = "h0-probe-launch-v2"
LAUNCH_SCHEMA_V3 = "h0-probe-launch-v3"
LAUNCH_SCHEMA_V4 = "h-probe-launch-v4"
LAUNCH_SCHEMA_V5 = "h-probe-launch-v5"
# v2 is NOT admitted: its read retired at the v5 bump (``parse_probe_launch`` refuses it by name).
_SCHEMAS = (LAUNCH_SCHEMA_V3, LAUNCH_SCHEMA_V4, LAUNCH_SCHEMA_V5)
# The manifest keys only a v5 launch carries; ``manifest`` and ``parse_probe_launch`` add and
# require them by schema, so a v3 or v4 payload re-renders its own bytes.
_V5_KEYS = "hourly_price_ceiling hours_cap pair cells"
# The one environment key a scientific launch does not carry.
_TINY_OPT_IN = "ENTROPIC_MLFLOW_TINY"
# The H lane's experiment. Exactly one definition in the tree: the wave monitor
# reads this constant, and a launch selects it with ``named_experiment``.
H_SERIES_EXPERIMENT = "h-series"

_EXTENSION_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


@dataclass(frozen=True, slots=True)
class _ExtensionSpec:
    """One registry row: the single env key an extension contributes."""

    env_key: str
    parameters: tuple[str, ...]
    value: Callable[[Mapping[str, str]], str]


# CLOSED registry, in canonical (registry) order. The env keys equal the consumer
# constants ``ladder.capture.reader.sinks.MLFLOW_EXPERIMENT_ENV`` and
# ``MLFLOW_SYSTEM_METRICS_ENV``; tests/test_h_probe_launch.py pins that equality
# and that "true" is a literal ``adapters.mlflow._system_metrics_logging_enabled``
# accepts, so a rename on the consumer side cannot silently orphan a launch.
EXTENSION_REGISTRY: Mapping[str, _ExtensionSpec] = MappingProxyType(
    {
        "named_experiment": _ExtensionSpec(
            "MLFLOW_EXPERIMENT_NAME", ("name",), lambda params: params["name"]
        ),
        "system_metrics": _ExtensionSpec(
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", (), lambda _: "true"
        ),
    }
)

_BASE_ENVIRONMENT_KEYS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_S3_ENDPOINT_URL",
        _TINY_OPT_IN,
    }
)
_extension_keys = [spec.env_key for spec in EXTENSION_REGISTRY.values()]
if len(set(_extension_keys)) != len(_extension_keys) or _BASE_ENVIRONMENT_KEYS & set(
    _extension_keys
):
    raise RuntimeError("launch extension environment keys must not collide")


@dataclass(frozen=True, slots=True)
class LaunchExtension:
    """A declared, validated member of the closed registry; its name and parameters
    are what the manifest records."""

    name: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        spec = EXTENSION_REGISTRY.get(self.name) if type(self.name) is str else None
        if spec is None:
            raise ValueError("launch extension is not in the closed registry")
        if type(self.parameters) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or _EXTENSION_VALUE.fullmatch(pair[1]) is None
            for pair in self.parameters
        ):
            raise ValueError("launch extension parameters must be safe identifier text")
        if tuple(key for key, _ in self.parameters) != spec.parameters:
            raise ValueError("launch extension parameters differ from its registry declaration")

    @property
    def env_value(self) -> str:
        return EXTENSION_REGISTRY[self.name].value(dict(self.parameters))

    @property
    def manifest(self) -> dict[str, object]:
        return {"name": self.name, "params": dict(self.parameters)}


def named_experiment(name: str) -> LaunchExtension:
    """Route the pod's MLflow runs into the named experiment."""
    return LaunchExtension("named_experiment", (("name", name),))


def system_metrics() -> LaunchExtension:
    """Turn on MLflow's own system-metrics monitor in the pod."""
    return LaunchExtension("system_metrics")


@dataclass(frozen=True, slots=True)
class H0ProbeLaunch:
    """A reproducible command, not permission to allocate or proof of a watchdog.

    The controller must establish a separately supervised watchdog before create.
    Both time budgets start at that arming event, not at trainer readiness. R2
    remains authoritative custody; this tiny lane does not attach shared volumes.
    """

    request: AnyProbeRequest
    template_id: str
    registry_auth_id: str
    gpu_type_id: str
    data_center_id: str
    cloud_type: str
    container_disk_gb: int
    r2_access_secret_name: str
    r2_private_secret_name: str
    mlflow_uri_secret_name: str
    mlflow_username_secret_name: str
    mlflow_password_secret_name: str
    mlflow_s3_endpoint_secret_name: str
    controller_timeout_seconds: int
    watchdog_timeout_seconds: int
    cost_guard: str
    # FOR-1694: the SEALED hourly price ceiling (USD per hour). ``None`` for a smoke launch
    # and for every pre-v5 schema; a v5 scientific launch declares it explicitly, with no
    # default and no derivation. Excluded from ``_BASE_FIELDS`` (a v5-only manifest key).
    hourly_price_ceiling: float | None = None
    # Schema state, not a subclass: a launch read from v3 is the same class with
    # ``manifest_schema == LAUNCH_SCHEMA_V3``. Neither name may equal one of the six
    # fixed manifest keys, because ``manifest`` and ``parse_probe_launch`` both
    # iterate ``fields()``.
    extensions: tuple[LaunchExtension, ...] = ()
    manifest_schema: str = LAUNCH_SCHEMA_V5

    def __post_init__(self) -> None:
        if type(self.request) is not ProbeRequest and type(self.request) is not PairProbeRequest:
            raise TypeError("H0 deployment requires a validated worker request")
        request = parse_probe_request(self.request.payload, max_bytes=len(self.request.payload))
        if type(request) is not type(self.request):
            raise ValueError("H0 deployment request changed kind on re-parse")
        if self.manifest_schema not in _SCHEMAS:
            raise ValueError("launch manifest schema is not admitted")
        if isinstance(request, PairProbeRequest) and self.manifest_schema != LAUNCH_SCHEMA_V5:
            raise ValueError(
                f"a pair launch requires {LAUNCH_SCHEMA_V5}; {self.manifest_schema} carries no pair"
            )
        profile = request.donor if isinstance(request, PairProbeRequest) else request.profile
        run_class, stage = profile.run_class, profile.stage
        authority = request.authority
        refusal = RUN_ADMISSION.authority_refusal(run_class, stage, authority)
        if refusal is not None:
            raise ValueError(refusal)
        if not RUN_ADMISSION.admits(run_class, stage, "launch", authority):
            raise ValueError("deployment is restricted to remote H0 tiny probes")
        object.__setattr__(self, "request", request)
        for value in (self.registry_auth_id, self.data_center_id, self.template_id):
            if (
                type(value) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None
            ):
                raise ValueError("provider placement identity is unsafe")
        gpu = self.gpu_type_id
        if (
            type(gpu) is not str
            or not gpu
            or gpu.strip() != gpu
            or not gpu[0].isalnum()
            or len(gpu) > 128
            or any(not (c.isascii() and (c.isalnum() or c in " ._-")) for c in gpu)
        ):
            raise ValueError("provider GPU type must be a safe explicit identifier")
        if self.cloud_type not in ("SECURE", "COMMUNITY"):
            raise ValueError("provider cloud type must be explicit")
        for value in (
            self.container_disk_gb,
            self.controller_timeout_seconds,
            self.watchdog_timeout_seconds,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("deployment storage and time bounds must be positive integers")
        if self.controller_timeout_seconds >= self.watchdog_timeout_seconds:
            raise ValueError(
                "watchdog must leave time after the controller deadline for capture/teardown"
            )
        if not RUN_ADMISSION.admits_cost_guard(run_class, self.cost_guard):
            if authority is not None:
                raise ValueError("scientific deployment requires the budgeted host watchdog")
            raise ValueError("H0 deployment requires the declared tiny-only watchdog exception")
        if authority is not None:
            if gpu != authority.gpu_class:
                raise ValueError("launch GPU differs from the run authority's GPU class")
            if self.watchdog_timeout_seconds > authority.spend_cap.hours * 3600:
                raise ValueError("launch watchdog outlasts the run authority's hours")
        secret_names = (
            self.r2_access_secret_name,
            self.r2_private_secret_name,
            self.mlflow_uri_secret_name,
            self.mlflow_username_secret_name,
            self.mlflow_password_secret_name,
            self.mlflow_s3_endpoint_secret_name,
        )
        for name in secret_names:
            if type(name) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) is None:
                raise ValueError(
                    "R2 and MLflow credentials must use provider secret-reference names"
                )
        if len(set(secret_names)) != len(secret_names):
            raise ValueError("provider secret-reference names must be distinct")
        if self.manifest_schema == LAUNCH_SCHEMA_V3 and authority is not None:
            raise ValueError("a pre-v4 launch carries no run authority")
        self._check_sealed_ceiling()
        if type(self.extensions) is not tuple or any(
            type(item) is not LaunchExtension for item in self.extensions
        ):
            raise ValueError("launch extensions must be a tuple of declared extensions")
        declared = [item.name for item in self.extensions]
        if declared != [name for name in EXTENSION_REGISTRY if name in declared] or len(
            set(declared)
        ) != len(declared):
            raise ValueError("launch extensions must be unique and in registry order")

    def _check_sealed_ceiling(self) -> None:
        """The sealed ceiling is explicit exactly for a v5 scientific launch, and is bounded by the
        authority's USD cap through the SAME product rule the flag was always held to."""
        authority = self.request.authority
        ceiling = self.hourly_price_ceiling
        if self.manifest_schema != LAUNCH_SCHEMA_V5 or authority is None:
            if ceiling is not None:
                raise ValueError(
                    "only a v5 scientific launch seals an hourly price ceiling"
                    if self.manifest_schema == LAUNCH_SCHEMA_V5
                    else "a pre-v5 launch seals no hourly price ceiling"
                )
            return
        if ceiling is None:
            raise ValueError("a v5 scientific launch must declare its hourly price ceiling")
        self._require_within_cap(ceiling)

    def _require_within_cap(self, ceiling: float) -> None:
        authority = self.request.authority
        if authority is None:
            return
        if (
            type(ceiling) not in (int, float)
            or not math.isfinite(ceiling)
            or ceiling <= 0
            or ceiling * self.watchdog_timeout_seconds / 3600 > authority.spend_cap.usd
        ):
            raise ValueError("hourly price ceiling over the watchdog window exceeds the spend cap")

    @property
    def manifest(self) -> dict[str, object]:
        mandate = self.request.manifest["spending_authority"]
        row: dict[str, object] = {
            "schema": self.manifest_schema,
            "scientific_authority": mandate,
            "spending_authority": mandate,
            "provider_deadline": None,
            "runpodctl_version": RUNPODCTL_VERSION,
            "runpodctl_sha256": RUNPODCTL_SHA256,
            **{f.name: getattr(self, f.name) for f in fields(self) if f.name in _BASE_FIELDS},
            "request": self.request.manifest,
        }
        row["extensions"] = [item.manifest for item in self.extensions]
        if self.manifest_schema == LAUNCH_SCHEMA_V5:
            authority = self.request.authority
            pair = isinstance(self.request, PairProbeRequest)
            row["hourly_price_ceiling"] = self.hourly_price_ceiling
            row["hours_cap"] = None if authority is None else authority.spend_cap.hours
            row["pair"] = pair
            row["cells"] = (
                [self.request.donor.identity, self.request.recipient.identity]
                if isinstance(self.request, PairProbeRequest)
                else None
            )
        return row

    def require_hourly_rate_within_cap(self, max_hourly_rate: float) -> None:
        """Refuse a price ceiling that could spend past the run authority's cap.

        The controller admits a pod only at a measured price at or under
        ``max_hourly_rate`` and the watchdog tears it down by
        ``watchdog_timeout_seconds``, so their product bounds the run's spend; a
        scientific launch requires that bound inside ``spend_cap.usd`` (its hours are
        checked at construction). A smoke launch has no authority and no cap here: its
        ceiling is the invoker's approval alone, as before rung B.

        FOR-1694: a launch that SEALS its ceiling (v5, scientific) admits no other value at
        all: a ``max_hourly_rate`` that is not exactly the sealed one is refused by name, with
        no tolerance, so the flag can never widen what the invoker sealed.
        """
        sealed = self.hourly_price_ceiling
        if sealed is not None:
            if (
                type(max_hourly_rate) not in (int, float)
                or isinstance(max_hourly_rate, bool)
                or max_hourly_rate != sealed
            ):
                raise ValueError(
                    "the hourly price ceiling differs from the launch's sealed hourly price ceiling"
                )
            return
        self._require_within_cap(max_hourly_rate)

    def hourly_ceiling(self, flag: float | None) -> float:
        """The ONE ceiling the host enforces: the sealed one, or (unsealed launch) the flag.

        A sealed launch accepts an absent flag and refuses a disagreeing one; an unsealed launch
        (smoke, v4, v3) keeps the flag's v3 semantics and refuses its absence by name.
        """
        sealed = self.hourly_price_ceiling
        if sealed is None:
            if flag is None:
                raise ValueError(
                    "this launch seals no hourly price ceiling: --max-hourly-rate is required"
                )
            self.require_hourly_rate_within_cap(flag)
            return flag
        if flag is not None:
            self.require_hourly_rate_within_cap(flag)
        return sealed

    @property
    def payload(self) -> bytes:
        return _json(self.manifest)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def pod_name(self) -> str:
        authority = self.request.authority
        if authority is None:
            return f"h0-tiny-{self.digest[:24]}"
        return f"{authority.stage.lower()}-{authority.run_class}-{self.digest[:24]}"

    @property
    def workspace(self) -> str:
        return f"/app/h-probe-{self.digest}"

    @property
    def template_environment(self) -> dict[str, str]:
        """The pod's entire environment, every value a provider secret reference
        except the one literal opt-in.

        EXIT-GATE LANE (conductor, operator-cleared 2026-09-26, schema v2): the H
        worker builds ``CompositeSink(JsonlSink, MlflowSink, tiny=True)`` and
        ``sinks.mlflow_enabled`` admits the MLflow mirror only when
        ``MLFLOW_TRACKING_URI`` is set AND ``ENTROPIC_MLFLOW_TINY == "1"``; with
        either missing it falls back to JSONL-only SILENTLY (measured 2026-09-26
        on the pinned 6b4e68fa image regime). The v1 contract carried only the
        two R2 references, so no v1 launch could ever reach the Railway MLflow.
        The username/password references feed the mlflow client's basic-auth
        env contract; the S3 endpoint reference feeds the artifact upload of the
        JSONL mirror. Exactly seven keys; validate_template compares the whole
        mapping, so a template missing any reference or the opt-in is refused.

        v3: the seven base keys, then each declared extension's one key in registry
        order (nine keys for the H0 wave). ``validate_template`` compares the union.

        v4: a smoke launch is the v3 environment. A scientific launch drops the tiny
        opt-in (its worker builds the MLflow sink with ``tiny=False``), so the H wave's
        scientific template is eight keys; it is created at the image rebuild ceremony.
        """
        environment = {
            "AWS_ACCESS_KEY_ID": f"{{{{ RUNPOD_SECRET_{self.r2_access_secret_name} }}}}",
            "AWS_SECRET_ACCESS_KEY": f"{{{{ RUNPOD_SECRET_{self.r2_private_secret_name} }}}}",
            "MLFLOW_TRACKING_URI": f"{{{{ RUNPOD_SECRET_{self.mlflow_uri_secret_name} }}}}",
            "MLFLOW_TRACKING_USERNAME": f"{{{{ RUNPOD_SECRET_{self.mlflow_username_secret_name} }}}}",
            "MLFLOW_TRACKING_PASSWORD": f"{{{{ RUNPOD_SECRET_{self.mlflow_password_secret_name} }}}}",
            "MLFLOW_S3_ENDPOINT_URL": f"{{{{ RUNPOD_SECRET_{self.mlflow_s3_endpoint_secret_name} }}}}",
            _TINY_OPT_IN: "1",
        }
        if self.request.authority is not None:
            del environment[_TINY_OPT_IN]
        for item in self.extensions:
            environment[EXTENSION_REGISTRY[item.name].env_key] = item.env_value
        return environment

    def validate_template(self, value: object) -> None:
        """Check a pinned-CLI template get result before creating the pod.

        CLI 2.14.0 omits false booleans, zero volumes and empty command/port lists.
        Their absence here follows that serialization contract, not a guessed
        scientific default. A provider read must still succeed independently.
        Measured 2026-10-26: 3.0.1 returns byte-identical ``template get``,
        ``user`` and ``pod list`` JSON to 2.14.0 for this template, so the
        contract carries over unchanged.
        """
        if type(value) is not dict:
            raise ValueError("H0 template must be a provider object")
        row = cast(dict[str, object], value)
        if (
            row.get("id") != self.template_id
            or row.get("imageName") != self.request.image
            or row.get("env") != self.template_environment
            or row.get("containerRegistryAuthId") != self.registry_auth_id
            or type(row.get("containerDiskInGb")) is not int
            or row.get("containerDiskInGb") != self.container_disk_gb
            or type(row.get("volumeInGb", 0)) is not int
            or row.get("volumeInGb", 0) != 0
            or any(row.get(k, False) is not False for k in ("isPublic", "isServerless", "isRunpod"))
            or any(row.get(k, []) != [] for k in ("ports", "dockerEntrypoint", "dockerStartCmd"))
        ):
            raise ValueError("H0 template differs from the declared private worker configuration")

    @property
    def create_arguments(self) -> tuple[str, ...]:
        """Arguments for verified runpodctl 3.0.1; execution belongs to the controller.

        ``pod create --help`` is byte-identical between 2.14.0 and 3.0.1 (measured
        2026-10-26); the create response itself is only exercised by a live launch.

        The fixed shell composition only creates a new owned workspace and execs
        the existing image worker. No installs, source overlays or SSH readiness.
        An existing workspace is a failure, never an implicit resume or overwrite.
        """
        request = self.request
        if request.image is None:
            raise ValueError("remote H0 deployment requires an immutable image")
        worker = (
            "python",
            "-m",
            "entropic_substrate.ladder.run.worker_command",
            "run",
            "--repo-root",
            "/app",
            "--workspace-root",
            self.workspace,
            "--max-object-bytes",
            str(request.limits.max_object_bytes),
            "--max-objects",
            str(request.limits.max_objects),
            "--request-key",
            f"{request.prefix}/request.json",
            "--request-sha256",
            request.digest,
            "--request-size",
            str(len(request.payload)),
        )
        script = f"mkdir -m 700 -- {shlex.quote(self.workspace)} && exec {shlex.join(worker)}"
        return (
            "pod",
            "create",
            "--name",
            self.pod_name,
            "--template-id",
            self.template_id,
            "--image",
            request.image,
            "--gpu-id",
            self.gpu_type_id,
            "--gpu-count",
            "1",
            "--cloud-type",
            self.cloud_type,
            "--data-center-ids",
            self.data_center_id,
            "--container-disk-in-gb",
            str(self.container_disk_gb),
            "--volume-in-gb",
            "0",
            "--min-cuda-version",
            "12.6",
            "--ssh=false",
            "--registry-auth-id",
            self.registry_auth_id,
            "--docker-args",
            shlex.join(("/bin/sh", "-c", script)),
            "--output",
            "json",
        )


_BASE_FIELDS = frozenset(
    f.name
    for f in fields(H0ProbeLaunch)
    if f.name not in ("request", "extensions", "manifest_schema", "hourly_price_ceiling")
)


def parse_probe_launch(payload: bytes, *, max_bytes: int) -> H0ProbeLaunch:
    """Require every deployment choice and canonical bytes; never fill defaults.

    Schema-first: the key list is chosen by the declared schema before any key is checked.
    A ``h0-probe-launch-v2`` payload is refused by name (its read retired at v5) and a
    schema outside the ladder keeps the generic scope refusal. v3 and v4 read as the same
    class and re-render their own bytes; a v5 payload must carry its four v5 keys, and
    every one that duplicates a value derivable from the embedded request or authority (the
    hours cap, the pair flag, both cell identities) is re-derived and compared, so a copied
    field is never a second, unsealed source of truth.
    """
    if (
        type(max_bytes) is not int
        or max_bytes < 1
        or type(payload) is not bytes
        or not 0 < len(payload) <= max_bytes
    ):
        raise ValueError("H0 deployment exceeds its explicit byte budget")
    decoded = json.loads(payload)
    schema = decoded.get("schema") if type(decoded) is dict else None
    if schema == LAUNCH_SCHEMA_V2:
        refuse_version(LAUNCH_SCHEMA_V2, LAUNCH_SCHEMA_V3, "launch")
    if schema not in _SCHEMAS:
        raise ValueError("H0 deployment scope or authority mismatch")
    names = " ".join(sorted(_BASE_FIELDS | {"request"}))
    keys = f"schema scientific_authority spending_authority provider_deadline runpodctl_version runpodctl_sha256 {names} extensions"
    if schema == LAUNCH_SCHEMA_V5:
        keys += f" {_V5_KEYS}"
    row = recovery_record(decoded, keys).copy()
    request = parse_probe_request(_json(row["request"]), max_bytes=max_bytes)
    row["request"] = request
    mandate = request.manifest["spending_authority"]
    if (
        row.pop("schema") != schema
        or row.pop("scientific_authority") != mandate
        or row.pop("spending_authority") != mandate
        or row.pop("provider_deadline") is not None
        or row.pop("runpodctl_version") != RUNPODCTL_VERSION
        or row.pop("runpodctl_sha256") != RUNPODCTL_SHA256
    ):
        raise ValueError("H0 deployment scope or authority mismatch")
    ceiling: float | None = None
    if schema == LAUNCH_SCHEMA_V5:
        authority = request.authority
        pair = isinstance(request, PairProbeRequest)
        declared_ceiling = row.pop("hourly_price_ceiling")
        # Named re-derivation of the copied fields. Measured 2026-11-02 (probes P19a/P19b): the
        # payload re-render below (``result.payload != payload``) refuses every tamper here too, so
        # these three checks are redundant guards that keep the refusal's field named.
        if (
            row.pop("hours_cap") != (None if authority is None else authority.spend_cap.hours)
            or row.pop("pair") is not pair
            or row.pop("cells")
            != ([request.donor.identity, request.recipient.identity] if pair else None)
        ):
            raise ValueError("H0 deployment scope or authority mismatch")
        if declared_ceiling is not None and (
            type(declared_ceiling) not in (int, float) or isinstance(declared_ceiling, bool)
        ):
            raise ValueError("launch hourly price ceiling must be a number or null")
        ceiling = cast(float | None, declared_ceiling)
    extensions: list[LaunchExtension] = []
    declared = row.pop("extensions")
    if type(declared) is not list:
        raise ValueError("launch extensions must be a list")
    for item in declared:
        entry = recovery_record(item, "name params")
        params = entry["params"]
        if type(params) is not dict or any(type(v) is not str for v in params.values()):
            raise ValueError("launch extension parameters must be text")
        extensions.append(
            LaunchExtension(cast(str, entry["name"]), tuple(cast(dict[str, str], params).items()))
        )
    factory = cast(Callable[..., H0ProbeLaunch], H0ProbeLaunch)
    result = factory(
        **row,
        hourly_price_ceiling=ceiling,
        extensions=tuple(extensions),
        manifest_schema=cast(str, schema),
    )
    if result.payload != payload:
        raise ValueError("H0 deployment must use canonical JSON")
    return result
