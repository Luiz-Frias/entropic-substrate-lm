"""Provider responses are not readiness, and ambiguous creates are never retried."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest
from tests.test_h_probe_launch import launch

from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_runpod_transport import HRunpodTransport, ProbeProviderError
from entropic_substrate.ladder.run.values.d1_operator_ports import D1CommandResult

pytestmark = pytest.mark.unit


@dataclass
class Runner:
    results: list[D1CommandResult | Exception]
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, argv: tuple[str, ...], *, cwd: Path | None, stdin: str | None) -> D1CommandResult:
        assert cwd is None and stdin is None
        self.calls.append(argv)
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def success(value: object) -> D1CommandResult:
    return D1CommandResult(json.dumps(value), "")


def failed(code: str) -> D1CommandResult:
    return D1CommandResult("", json.dumps({"error": "redacted fixture detail", "code": code}), 1)


def pod(value: H0ProbeLaunch) -> dict[str, object]:
    return {
        "id": "owned-pod",
        "name": value.pod_name,
        "imageName": value.request.image,
        "gpuCount": 1,
        "desiredStatus": "RUNNING",
        "runtimeStatus": "initializing",
        "costPerHr": 1.0,
        "env": {"credential": "must-not-escape"},
    }


def transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: Runner) -> HRunpodTransport:
    binary = tmp_path / "runpodctl"
    binary.write_bytes(b"test-only executable identity; runner is injected")
    binary.chmod(0o700)
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.h_runpod_transport.RUNPODCTL_SHA256",
        hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    return HRunpodTransport(
        executable=binary, runner=runner, max_response_bytes=10_000, max_launch_bytes=10_000
    )


def test_constructor_is_lazy_and_changed_binary_stops_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Runner([])
    adapter = transport(tmp_path, monkeypatch, runner)
    assert runner.calls == []
    (tmp_path / "runpodctl").write_bytes(b"changed")
    with pytest.raises(ProbeProviderError, match="executable_mismatch"):
        adapter.account()
    assert runner.calls == []


def test_account_preserves_measured_zero_and_requires_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Runner(
        [
            D1CommandResult("runpodctl 3.0.1-551d6a4\n", ""),
            success({"id": "account-fixture", "clientBalance": 22.0, "currentSpendPerHr": 0}),
        ]
    )
    account = transport(tmp_path, monkeypatch, runner).account()
    assert account.balance_usd == 22 and account.current_spend_per_hour == 0
    assert [call[1] for call in runner.calls] == ["version", "user"]


@pytest.mark.parametrize("reported", ["unknown-build", "2.14.0-dd55bcf"])
def test_changed_version_stops_before_the_account_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reported: str
) -> None:
    runner = Runner([D1CommandResult(f"runpodctl {reported}\n", "")])
    with pytest.raises(ProbeProviderError, match="version_mismatch"):
        transport(tmp_path, monkeypatch, runner).account()
    assert [call[1] for call in runner.calls] == ["version"]


def test_response_budget_does_not_change_launch_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Runner([success(pod(launch()))])
    transport(tmp_path, monkeypatch, runner)
    adapter = HRunpodTransport(
        executable=tmp_path / "runpodctl",
        runner=runner,
        max_response_bytes=1,
        max_launch_bytes=10_000,
    )
    with pytest.raises(ProbeProviderError, match="response_budget"):
        adapter.create_once(launch())
    assert len(runner.calls) == 1


@pytest.mark.parametrize("cost", [None, 0, 1])
def test_create_returns_only_allowlisted_observation_not_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cost: int | None
) -> None:
    value = launch()
    response = pod(value)
    response["costPerHr"] = cost
    del response["runtimeStatus"]  # GraphQL create has no native runtime readiness.
    runner = Runner([success(response)])
    observed = transport(tmp_path, monkeypatch, runner).create_once(value)
    assert observed.pod_id == "owned-pod" and observed.runtime_status is None
    assert observed.cost_per_hour == cost and "must-not-escape" not in repr(asdict(observed))
    assert runner.calls[0][1:] == value.create_arguments and len(runner.calls) == 1


@pytest.mark.parametrize(
    "result",
    [
        failed("network_error"),
        failed("timeout"),
        D1CommandResult("not-json", ""),
        OSError("must-not-escape"),
    ],
)
def test_ambiguous_create_is_explicit_and_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: D1CommandResult | Exception
) -> None:
    runner = Runner([result])
    with pytest.raises(ProbeProviderError) as error:
        transport(tmp_path, monkeypatch, runner).create_once(launch())
    assert error.value.allocation_uncertain is True
    assert "must-not-escape" not in str(error.value)
    assert len(runner.calls) == 1


def test_reported_create_id_survives_a_mismatched_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = launch()
    response = pod(value)
    response["name"] = "foreign"
    runner = Runner([success(response)])
    with pytest.raises(ProbeProviderError) as error:
        transport(tmp_path, monkeypatch, runner).create_once(value)
    assert error.value.possible_pod_id == "owned-pod"
    assert error.value.allocation_uncertain
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "field,value", [("name", "foreign"), ("imageName", "foreign:latest"), ("id", "foreign-id")]
)
def test_deletion_refuses_foreign_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    spec = launch()
    response = pod(spec)
    response[field] = value
    runner = Runner([success(response)])
    with pytest.raises(ProbeProviderError):
        transport(tmp_path, monkeypatch, runner).delete_owned(spec, "owned-pod")
    assert all("delete" not in call for call in runner.calls)


def test_deletion_acknowledgement_requires_separate_absence_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = launch()
    runner = Runner(
        [
            success(pod(spec)),
            success({"deleted": True, "id": "owned-pod"}),
            success(pod(spec)),
            failed("not_found"),
        ]
    )
    adapter = transport(tmp_path, monkeypatch, runner)
    assert adapter.delete_owned(spec, "owned-pod") == "deletion_acknowledged"
    assert adapter.get_owned(spec, "owned-pod") is not None
    assert adapter.get_owned(spec, "owned-pod") is None
    assert sum("delete" in call for call in runner.calls) == 1


@pytest.mark.parametrize("code", ["network_error", "timeout", "forbidden"])
def test_lookup_failure_is_not_confirmed_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    runner = Runner([failed(code)])
    with pytest.raises(ProbeProviderError):
        transport(tmp_path, monkeypatch, runner).get_owned(launch(), "owned-pod")


def test_absent_pod_is_not_deleted_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = Runner([failed("not_found")])
    assert (
        transport(tmp_path, monkeypatch, runner).delete_owned(launch(), "owned-pod")
        == "already_absent"
    )
    assert all("delete" not in call for call in runner.calls)


@pytest.mark.parametrize("payload", ['{"id":"a","id":"b"}', '{"id":NaN}', "null", "[]"])
def test_malformed_provider_objects_fail_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    runner = Runner([D1CommandResult(payload, "")])
    with pytest.raises(ProbeProviderError):
        transport(tmp_path, monkeypatch, runner).get_owned(launch(), "owned-pod")


def test_preflight_reads_and_validates_template_without_allocating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = launch()
    runner = Runner(
        [
            success(
                {
                    "id": spec.template_id,
                    "name": "private",
                    "imageName": spec.request.image,
                    "env": spec.template_environment,
                    "containerDiskInGb": spec.container_disk_gb,
                    "containerRegistryAuthId": spec.registry_auth_id,
                }
            )
        ]
    )
    transport(tmp_path, monkeypatch, runner).check_template(spec)
    assert runner.calls[0][1:4] == ("template", "get", spec.template_id)
    assert all("create" not in call for call in runner.calls)


def test_mutated_launch_cannot_reach_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = launch()
    object.__setattr__(spec, "cost_guard", "disabled")
    runner = Runner([])
    with pytest.raises(ProbeProviderError, match="invalid_launch"):
        transport(tmp_path, monkeypatch, runner).create_once(spec)
    assert runner.calls == []


@pytest.mark.parametrize("amount", [None, True, float("inf"), 10**400])
def test_invalid_account_measurement_is_not_a_zero_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, amount: object
) -> None:
    runner = Runner(
        [
            D1CommandResult("runpodctl 3.0.1-551d6a4\n", ""),
            success({"id": "account-fixture", "clientBalance": amount, "currentSpendPerHr": 0}),
        ]
    )
    with pytest.raises(ProbeProviderError):
        transport(tmp_path, monkeypatch, runner).account()


def test_error_resource_id_survives_without_credential_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = D1CommandResult(
        "", json.dumps({"error": "must-not-escape", "code": "timeout", "id": "possible-pod"}), 1
    )
    runner = Runner([result])
    with pytest.raises(ProbeProviderError) as error:
        transport(tmp_path, monkeypatch, runner).create_once(launch())
    assert error.value.possible_pod_id == "possible-pod"
    assert "must-not-escape" not in str(error.value)


def test_output_budget_failure_preserves_allocation_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Runner([D1CommandResult("x" * 10_001, "")])
    with pytest.raises(ProbeProviderError, match="response_budget") as error:
        transport(tmp_path, monkeypatch, runner).create_once(launch())
    assert error.value.allocation_uncertain


def test_incomplete_not_found_error_does_not_prove_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Runner([D1CommandResult("", '{"code":"not_found"}', 1)])
    with pytest.raises(ProbeProviderError, match="malformed_error"):
        transport(tmp_path, monkeypatch, runner).get_owned(launch(), "owned-pod")


def test_foreign_not_found_id_does_not_prove_requested_pod_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = D1CommandResult(
        "", json.dumps({"error": "missing", "code": "not_found", "id": "foreign-pod"}), 1
    )
    with pytest.raises(ProbeProviderError, match="not_found_identity_mismatch"):
        transport(tmp_path, monkeypatch, Runner([result])).get_owned(launch(), "owned-pod")


def test_changed_output_tail_stops_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(H0ProbeLaunch, "create_arguments", property(lambda _: ("pod", "create")))
    runner = Runner([])
    with pytest.raises(ProbeProviderError, match="invalid_launch"):
        transport(tmp_path, monkeypatch, runner).create_once(launch())
    assert runner.calls == []


def test_named_reconciliation_includes_stopped_pods_and_does_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = launch()
    stopped = pod(spec) | {"desiredStatus": "EXITED", "runtimeStatus": "stopped"}
    foreign = pod(spec) | {"id": "foreign-pod", "name": "other-job"}
    runner = Runner([success([stopped, foreign])])
    observed = transport(tmp_path, monkeypatch, runner).list_owned(spec)
    assert [p.pod_id for p in observed] == ["owned-pod"]
    assert observed[0].runtime_status == "stopped"
    assert runner.calls[0][1:] == (
        "pod",
        "list",
        "--all",
        "--name",
        spec.pod_name,
        "--output",
        "json",
    )
    assert all("create" not in call and "delete" not in call for call in runner.calls)


@pytest.mark.parametrize(
    "kind", ["null", "object", "duplicate", "malformed", "wrong_image", "row_type", "nonfinite"]
)
def test_invalid_inventory_never_becomes_an_empty_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    spec = launch()
    cases = {
        "null": None,
        "object": {},
        "duplicate": [pod(spec), pod(spec)],
        "malformed": [{}],
        "wrong_image": [pod(spec) | {"imageName": "different"}],
        "row_type": ["not-an-object"],
        "nonfinite": [pod(spec) | {"ignored": float("nan")}],
    }
    runner = Runner([success(cases[kind])])
    with pytest.raises(ProbeProviderError):
        transport(tmp_path, monkeypatch, runner).list_owned(spec)


def test_empty_inventory_is_only_a_snapshot_and_multiple_candidates_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = launch()
    runner = Runner([success([]), success([pod(spec), pod(spec) | {"id": "second-pod"}])])
    adapter = transport(tmp_path, monkeypatch, runner)
    assert adapter.list_owned(spec) == ()
    assert len(adapter.list_owned(spec)) == 2
