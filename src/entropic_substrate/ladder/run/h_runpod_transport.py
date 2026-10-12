"""Pinned, bounded Runpod transport; controller authority and watchdog are external."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, cast

from entropic_substrate.kernel.codec.strict_json import strict_loads
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.h_probe_launch import (
    RUNPODCTL_SHA256,
    RUNPODCTL_VERSION,
    H0ProbeLaunch,
    parse_probe_launch,
)
from entropic_substrate.ladder.run.values.d1_operator_ports import D1CommandResult, D1CommandRunner


class ProbeProviderError(RuntimeError):
    """Redacted failure; a possible allocation must be reconciled, never retried."""

    def __init__(
        self, code: str, *, allocation_uncertain: bool = False, possible_pod_id: str | None = None
    ) -> None:
        self.code = code
        self.allocation_uncertain = allocation_uncertain
        self.possible_pod_id = possible_pod_id
        super().__init__(f"H probe provider: {code}; allocation_uncertain={allocation_uncertain}")


@dataclass(frozen=True, slots=True)
class ProbeAccount:
    account_id: str
    balance_usd: float
    current_spend_per_hour: float


@dataclass(frozen=True, slots=True)
class ProbePod:
    pod_id: str
    name: str
    image: str
    gpu_count: int
    desired_status: str
    runtime_status: str | None
    cost_per_hour: float | None
    response_sha256: str


def _reject(reason: str) -> NoReturn:
    if reason.startswith("duplicate key"):
        raise ValueError("duplicate provider field")
    raise ValueError("provider response contains a non-finite number")


def _object(raw: str) -> dict[str, object]:
    value = strict_loads(raw, fail=_reject)
    # Reject nonfinite numbers at every depth, including otherwise ignored fields.
    json.dumps(value, allow_nan=False)
    if type(value) is not dict:
        raise ValueError("provider response is not an object")
    return cast(dict[str, object], value)


def _money(value: object) -> float:
    if type(value) is not int and type(value) is not float:
        raise ValueError("provider monetary measurement is invalid or missing")
    try:
        measured = float(value)
    except OverflowError:
        raise ValueError("provider monetary measurement exceeds finite range") from None
    if not math.isfinite(measured) or measured < 0:
        raise ValueError("provider monetary measurement is invalid or missing")
    return measured


class HRunpodTransport:
    """No retries, implicit credential logging, readiness claims, or scientific gates.

    The injected runner must bound time/output and execute argv without a shell.
    The application must establish spending authority and an independently armed
    watchdog before create_once. This low-level adapter grants neither.
    """

    def __init__(
        self,
        *,
        executable: Path,
        runner: D1CommandRunner,
        max_response_bytes: int,
        max_launch_bytes: int,
    ) -> None:
        if not isinstance(cast(object, executable), Path) or not executable.is_absolute():
            raise ValueError("provider executable must be an explicit absolute path")
        if any(type(v) is not int or v < 1 for v in (max_response_bytes, max_launch_bytes)):
            raise ValueError("provider response and launch byte limits must be positive")
        self._executable, self._runner = executable, runner
        self._maximum = max_response_bytes
        self._launch_maximum = max_launch_bytes

    def _launch(self, launch: H0ProbeLaunch) -> H0ProbeLaunch:
        try:
            if type(launch) is not H0ProbeLaunch:
                raise TypeError("nominal launch required")
            return parse_probe_launch(launch.payload, max_bytes=self._launch_maximum)
        except (TypeError, ValueError):
            raise ProbeProviderError("invalid_launch") from None

    def _invoke(self, arguments: tuple[str, ...], *, creating: bool = False) -> str:
        try:
            with self._executable.open("rb") as stream:
                binary = stream.read(64 * 1024 * 1024 + 1)
            if hashlib.sha256(binary).hexdigest() != RUNPODCTL_SHA256:
                raise ValueError("executable mismatch")
        except (OSError, ValueError):
            raise ProbeProviderError("executable_mismatch") from None
        try:
            result = self._runner.run((str(self._executable), *arguments), cwd=None, stdin=None)
        except Exception:
            raise ProbeProviderError("process_failure", allocation_uncertain=creating) from None
        if (
            type(result) is not D1CommandResult
            or len(result.stdout.encode()) + len(result.stderr.encode()) > self._maximum
        ):
            raise ProbeProviderError("response_budget_or_type", allocation_uncertain=creating)
        if result.returncode == 0:
            return result.stdout
        try:
            rows = [line for line in result.stderr.splitlines() if line.startswith("{")]
            if len(rows) != 1:
                raise ValueError("ambiguous error envelope")
            error = _object(rows[0])
            if type(error.get("error")) is not str or set(error) - {
                "error",
                "code",
                "status",
                "id",
            }:
                raise ValueError("invalid error envelope")
            code = error.get("code")
            if type(code) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) is None:
                raise ValueError("missing error code")
            possible = None if error.get("id") is None else require_d1_pod_id(error["id"])
        except (TypeError, ValueError):
            raise ProbeProviderError("malformed_error", allocation_uncertain=creating) from None
        raise ProbeProviderError(code, allocation_uncertain=creating, possible_pod_id=possible)

    def _read(
        self, arguments: tuple[str, ...], *, creating: bool = False
    ) -> tuple[dict[str, object], str]:
        raw = self._invoke((*arguments, "--output", "json"), creating=creating)
        try:
            return _object(raw), hashlib.sha256(raw.encode()).hexdigest()
        except (TypeError, ValueError):
            raise ProbeProviderError("malformed_response", allocation_uncertain=creating) from None

    def account(self) -> ProbeAccount:
        if self._invoke(("version",)).rstrip("\n") != RUNPODCTL_VERSION:
            raise ProbeProviderError("version_mismatch")
        row, _ = self._read(("user",))
        try:
            identity = require_d1_pod_id(row.get("id"))
            return ProbeAccount(
                identity, _money(row.get("clientBalance")), _money(row.get("currentSpendPerHr"))
            )
        except (TypeError, ValueError):
            raise ProbeProviderError("invalid_account_measurements") from None

    def check_template(self, launch: H0ProbeLaunch) -> None:
        launch = self._launch(launch)
        row, _ = self._read(("template", "get", launch.template_id))
        try:
            launch.validate_template(row)
        except (TypeError, ValueError):
            raise ProbeProviderError("template_mismatch") from None

    @staticmethod
    def _pod(
        launch: H0ProbeLaunch,
        row: dict[str, object],
        digest: str,
        *,
        expected_id: str | None,
        creating: bool,
    ) -> ProbePod:
        possible = None
        try:
            possible = require_d1_pod_id(row.get("id"))
            if (
                (expected_id is not None and possible != expected_id)
                or row.get("name") != launch.pod_name
                or row.get("imageName") != launch.request.image
            ):
                raise ValueError("pod identity mismatch")
            count, desired, runtime = (
                row.get("gpuCount"),
                row.get("desiredStatus"),
                row.get("runtimeStatus"),
            )
            if (
                type(count) is not int
                or count != 1
                or type(desired) is not str
                or not desired
                or (runtime is not None and type(runtime) is not str)
            ):
                raise ValueError("pod observation is malformed")
            price = None if row.get("costPerHr") is None else _money(row["costPerHr"])
            return ProbePod(
                possible,
                launch.pod_name,
                str(launch.request.image),
                count,
                desired,
                runtime,
                price,
                digest,
            )
        except (TypeError, ValueError):
            raise ProbeProviderError(
                "pod_identity_or_measurement_mismatch",
                allocation_uncertain=creating,
                possible_pod_id=possible,
            ) from None

    def create_once(self, launch: H0ProbeLaunch) -> ProbePod:
        """Perform one allocation call only; failed/ambiguous calls must be reconciled."""
        launch = self._launch(launch)
        arguments = launch.create_arguments
        if arguments[-2:] != ("--output", "json"):
            raise ProbeProviderError("invalid_launch")
        row, digest = self._read(arguments[:-2], creating=True)
        return self._pod(launch, row, digest, expected_id=None, creating=True)

    def get_owned(self, launch: H0ProbeLaunch, pod_id: str) -> ProbePod | None:
        launch = self._launch(launch)
        pod_id = require_d1_pod_id(pod_id)
        try:
            row, digest = self._read(("pod", "get", pod_id, "--include-machine"))
        except ProbeProviderError as error:
            if error.code == "not_found":
                if error.possible_pod_id not in (None, pod_id):
                    raise ProbeProviderError("not_found_identity_mismatch") from error
                return None
            raise
        return self._pod(launch, row, digest, expected_id=pod_id, creating=False)

    def list_owned(self, launch: H0ProbeLaunch) -> tuple[ProbePod, ...]:
        """Reconcile the exact launch name, including stopped resources.

        An empty snapshot does not disprove a delayed allocation after a timed-out
        create. Multiple candidates stay visible; this method never selects one,
        retries creation, deletes resources, or grants a termination certificate.
        """
        launch = self._launch(launch)
        raw = self._invoke(("pod", "list", "--all", "--name", launch.pod_name, "--output", "json"))
        try:
            rows = strict_loads(raw, fail=_reject)
            json.dumps(rows, allow_nan=False)
            if type(rows) is not list:
                raise ValueError("provider inventory must be an array")
            digest = hashlib.sha256(raw.encode()).hexdigest()
            found: list[ProbePod] = []
            seen: set[str] = set()
            for value in cast(list[object], rows):
                if type(value) is not dict:
                    raise ValueError("provider inventory row must be an object")
                row = cast(dict[str, object], value)
                pod_id = require_d1_pod_id(row.get("id"))
                name = row.get("name")
                if pod_id in seen or type(name) is not str or not name.strip():
                    raise ValueError("provider inventory identity is invalid or duplicated")
                seen.add(pod_id)
                if name == launch.pod_name:
                    found.append(self._pod(launch, row, digest, expected_id=pod_id, creating=False))
            return tuple(found)
        except (TypeError, ValueError):
            raise ProbeProviderError("invalid_inventory") from None

    def delete_owned(
        self, launch: H0ProbeLaunch, pod_id: str
    ) -> Literal["already_absent", "deletion_acknowledged"]:
        """Recheck ownership; acknowledgement alone does not establish termination."""
        if self.get_owned(launch, pod_id) is None:
            return "already_absent"
        row, _ = self._read(("pod", "delete", pod_id))
        if set(row) != {"deleted", "id"} or row["deleted"] is not True or row["id"] != pod_id:
            raise ProbeProviderError("deletion_acknowledgement_mismatch", possible_pod_id=pod_id)
        return "deletion_acknowledged"
