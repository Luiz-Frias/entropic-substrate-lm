"""p14-j-run: ladder/run and the D1 CLI wiring adopt kernel.codec.strict_json.

Five groups, one per acceptance bullet (FOR-1577, FOR-1582, FOR-1613):

1. adoption -- an AST walk finds no local ``_pairs``-style strict-JSON hook
   definition left under ``ladder/run`` or in ``composition/wiring/d1_cli.py``
   (the identity-bearing ``ladder/run/journal.py`` is exempt, ruling 5; the
   billing parser is a stated, test-pinned exception, see ``_PENDING``).
2. codec boundary -- every adopted decoder raises its OWN exception type for
   duplicate keys, non-finite constants, overflow floats and non-objects,
   because ``fail=`` is wired to the decoder's own error channel.
3. d1_cli -- ``_checked_payload`` stays the strict ``json.dumps`` admission
   gate (it is NOT folded into ``canonical_json``): a payload nesting a
   ``MappingProxyType`` is still an ``unexpected_defect``.
4. canonical -- ``d1_trainer_process_contracts._canonical`` now delegates to
   ``canonical_json`` and is byte-equal to the inline dump it replaced.
5. SHA boundary -- every repointed digest site refuses a mixed-case digest
   with that site's own exception type.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

import entropic_substrate.composition.wiring.d1_cli as d1_cli_module
from d1_launch_admission_fixtures import plan, profiles, start
from entropic_substrate.composition.entrypoints.d1_r2_trainer_mailbox_worker import (
    D1R2TrainerMailboxWorker,
    D1R2TrainerMailboxWorkerError,
)
from entropic_substrate.composition.wiring.d1_cli import (
    D1_CLI_MANIFEST_SCHEMA,
    D1CliApplication,
    D1CliCommand,
    D1CliExitCode,
    D1CliInvocation,
)
from entropic_substrate.ladder.run.campaign import d1_campaign_checkpoint
from entropic_substrate.ladder.run.campaign.d1_campaign_checkpoint import (
    D1CampaignCheckpointError,
    parse_campaign_checkpoint,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_request import (
    D1CampaignRequestError,
    D1PersistedCampaignCursor,
    parse_d1_campaign_request,
)
from entropic_substrate.ladder.run.campaign.d1_operator_evidence import attest_d1_trainer_start
from entropic_substrate.ladder.run.campaign.d1_operator_intent_codec import (
    D1OperatorIntentCodecError,
    parse_d1_confirmatory_operator_intent,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.ladder.run.ckptsync import (
    d1_checkpoint_dvc_recovery,
    d1_checkpoint_dvc_sync,
)
from entropic_substrate.ladder.run.ckptsync.d1_capture_finalize_request import (
    D1CaptureFinalizeRequestError,
    parse_d1_capture_finalize_request,
)
from entropic_substrate.ladder.run.ckptsync.d1_checkpoint_dvc_recovery import (
    D1CheckpointDvcRecoveryError,
    D1CheckpointDvcRecoveryService,
)
from entropic_substrate.ladder.run.ckptsync.d1_checkpoint_dvc_sync import (
    D1CheckpointDvcSyncAdapter,
    parse_d1_checkpoint_dvc_sync_receipt,
)
from entropic_substrate.ladder.run.closure import d1_finalized_cell_custody
from entropic_substrate.ladder.run.closure.d1_finalized_cell_custody import (
    D1FinalizedCellCustodyError,
    parse_d1_finalized_cell_custody,
)
from entropic_substrate.ladder.run.construction.d1_r2_runtime_composition import (
    D1BoundR2CheckpointSync,
)
from entropic_substrate.ladder.run.h_runpod_transport import _object as h_runpod_object
from entropic_substrate.ladder.run.provider import d1_build_cloud_adapter, d1_runpod_adapter
from entropic_substrate.ladder.run.trainer.d1_evaluation_codec import (
    D1EvaluationCodecError,
    parse_d1_evaluation,
)
from entropic_substrate.ladder.run.transport import (
    d1_trainer_evidence_adapter,
    d1_trainer_mailbox_contracts,
    d1_trainer_object_transport,
    d1_trainer_process_contracts,
)
from entropic_substrate.ladder.run.transport.d1_trainer_handoff_codec import (
    D1TrainerHandoffCodecError,
    parse_d1_trainer_handoff,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    parse_d1_trainer_mailbox_command,
    parse_d1_trainer_mailbox_error,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_worker import (
    D1TrainerMailboxWaitPolicy,
    D1TrainerMailboxWorker,
    D1TrainerMailboxWorkerError,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1ObjectMailboxTransportError,
)
from entropic_substrate.ladder.run.transport.d1_trainer_output_publication import (
    D1TrainerOutputPublicationError,
)
from entropic_substrate.ladder.run.transport.d1_trainer_output_publication import (
    _decode_marker as decode_completion_marker,
)
from entropic_substrate.ladder.run.transport.d1_trainer_process_contracts import (
    d1_trainer_request_path,
    d1_trainer_runtime_report_path,
    d1_trainer_start_report_path,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    D1TrainerRequestError,
    parse_d1_trainer_request,
)
from entropic_substrate.ladder.run.values.d1_operator_ports import D1OperatorTransportError

from .test_d1_operator_values import _image, _resource

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[1] / "src" / "entropic_substrate"
_RUN = _SRC / "ladder" / "run"
_D1_CLI = _SRC / "composition" / "wiring" / "d1_cli.py"
_EXEMPT = frozenset({"ladder/run/journal.py"})
# FOR-1577 finding, verified by running the adoption against it: strict_loads
# has no parse_float hook, so the billing parser (parse_float=Decimal, a
# 19-digit amount pinned in tests/test_d1_runpod_billing_parser.py) cannot
# adopt it without silently rounding billed amounts to float. The file keeps
# its local hooks until the door grows a parse_float seam. Ruled W4-21
# (option a, 2026-10-05): the exemption stands, the seam is FOR-1632.
_PENDING = frozenset({"ladder/run/closure/d1_runpod_billing_parser.py"})
_FORBIDDEN = frozenset({"_pairs", "_strict_pairs", "_nonfinite", "_reject_constant"})

_DIGEST = "1" * 64
_MIXED = "A" + "1" * 63
_SHA_MESSAGE = "must be a 64-character hexadecimal SHA-256 digest"


# --------------------------------------------------------------------------
# 1. adoption: no local strict-JSON hook definitions


def _forbidden_definitions(source: str) -> list[str]:
    return [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in _FORBIDDEN
    ]


def _scanned_files() -> list[Path]:
    return [*sorted(_RUN.rglob("*.py")), _D1_CLI]


def test_the_ast_walk_detects_a_local_hook_definition_at_any_depth() -> None:
    nested = "def outer():\n    def _reject_constant(value):\n        return 0\n"
    assert _forbidden_definitions(nested) == ["_reject_constant"]
    assert _forbidden_definitions("def parse():\n    return 1\n") == []


def test_run_and_d1_cli_define_no_local_strict_json_hooks() -> None:
    files = _scanned_files()
    assert len(files) > 100, "scan is misrooted: positive control on the file count failed"
    assert _D1_CLI in files, "d1_cli.py is part of the scan"
    assert _RUN / "journal.py" in files, "exempt file must be scanned-then-skipped"
    assert _RUN / "provider" / "d1_runpod_adapter.py" in files
    found = {
        rel: names
        for path in files
        if (rel := path.relative_to(_SRC).as_posix()) not in _EXEMPT | _PENDING
        if (names := _forbidden_definitions(path.read_text(encoding="utf-8")))
    }
    assert found == {}


def test_the_pending_exemption_is_exactly_the_billing_parser_and_still_needed() -> None:
    # A stale exemption would hide a regression: when the billing parser adopts
    # the door, this test goes RED and demands that _PENDING be emptied.
    assert _PENDING == {"ladder/run/closure/d1_runpod_billing_parser.py"}
    for rel in _PENDING:
        names = _forbidden_definitions((_SRC / rel).read_text(encoding="utf-8"))
        assert names, f"{rel} no longer defines a local hook: drop it from _PENDING"


# --------------------------------------------------------------------------
# 2. codec boundary: strict_loads(fail=<decoder channel>) keeps each exception type

_DUPLICATE = '{"schema_version":"x","schema_version":"y"}'
_NAN = '{"schema_version":NaN}'
_NEG_INFINITY = '{"schema_version":-Infinity}'
_OVERFLOW = '{"schema_version":1e999}'


def _build_cloud_metadata(payload: str) -> object:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.json"
        path.write_text(payload, encoding="utf-8")
        return d1_build_cloud_adapter._metadata(  # pyright: ignore[reportPrivateUsage]
            path, subject="image"
        )


_TEXT_DECODERS: list[tuple[str, Callable[[str], object], type[Exception]]] = [
    ("checkpoint", parse_campaign_checkpoint, D1CampaignCheckpointError),
    (
        "request",
        lambda payload: parse_d1_campaign_request(payload, catalog=()),
        D1CampaignRequestError,
    ),
    (
        "intent",
        lambda payload: parse_d1_confirmatory_operator_intent(payload, catalog=()),
        D1OperatorIntentCodecError,
    ),
    (
        "finalize",
        lambda payload: parse_d1_capture_finalize_request(payload, catalog=()),
        D1CaptureFinalizeRequestError,
    ),
    ("custody", parse_d1_finalized_cell_custody, D1FinalizedCellCustodyError),
    (
        "build_cloud",
        lambda payload: _build_cloud_metadata(payload),
        D1OperatorTransportError,
    ),
    (
        "runpod",
        lambda payload: d1_runpod_adapter._decode(  # pyright: ignore[reportPrivateUsage]
            payload, operation="op", subject="s"
        ),
        D1OperatorTransportError,
    ),
    ("h_runpod", h_runpod_object, ValueError),
]

_BYTES_DECODERS: list[tuple[str, Callable[[bytes], object], type[Exception]]] = [
    ("evaluation", parse_d1_evaluation, D1EvaluationCodecError),
    (
        "handoff",
        lambda payload: parse_d1_trainer_handoff(payload, artifact_root=Path(".")),
        D1TrainerHandoffCodecError,
    ),
    (
        "trainer_request",
        lambda payload: parse_d1_trainer_request(payload, catalog=()),
        D1TrainerRequestError,
    ),
    ("mailbox_command", parse_d1_trainer_mailbox_command, ValueError),
    ("mailbox_error", parse_d1_trainer_mailbox_error, ValueError),
    ("dvc_sync", parse_d1_checkpoint_dvc_sync_receipt, ValueError),
    ("completion_marker", decode_completion_marker, D1TrainerOutputPublicationError),
    (
        "process_report",
        lambda payload: d1_trainer_process_contracts._decode(  # pyright: ignore[reportPrivateUsage]
            payload, keys=frozenset({"schema_version"}), schema="s"
        ),
        ValueError,
    ),
]


# Some decoders (the pod report, the provider adapters) deliberately render a
# generic message instead of the door's reason, which names the offending key.
_VIOLATIONS = [
    (_DUPLICATE, "duplicate|repeats a key"),
    (_NAN, "non-?finite"),
    (_NEG_INFINITY, "non-?finite"),
    (_OVERFLOW, "non-?finite"),
]


@pytest.mark.parametrize(("payload", "needle"), _VIOLATIONS)
@pytest.mark.parametrize(
    ("decode", "exception"),
    [pytest.param(decode, exc, id=name) for name, decode, exc in _TEXT_DECODERS],
)
def test_text_decoders_refuse_strict_json_violations_with_their_own_error(
    decode: Callable[[str], object],
    exception: type[Exception],
    payload: str,
    needle: str,
) -> None:
    with pytest.raises(exception) as caught:
        decode(payload)
    # The two provider adapters surface the door's reason as their own detail.
    assert (
        "is invalid" in str(caught.value) or "invalid JSON" in str(caught.value)
        if exception is D1OperatorTransportError
        else re.search(needle, str(caught.value)) is not None
    )


@pytest.mark.parametrize(("payload", "needle"), _VIOLATIONS)
@pytest.mark.parametrize(
    ("decode", "exception"),
    [pytest.param(decode, exc, id=name) for name, decode, exc in _BYTES_DECODERS],
)
def test_bytes_decoders_refuse_strict_json_violations_with_their_own_error(
    decode: Callable[[bytes], object],
    exception: type[Exception],
    payload: str,
    needle: str,
) -> None:
    with pytest.raises(exception) as caught:
        decode(payload.encode())
    assert re.search(needle, str(caught.value)) is not None


# --------------------------------------------------------------------------
# 3. d1_cli: the strict admission gate is not folded into canonical_json

_CHECKED_PAYLOAD_SOURCE = """\
def _checked_payload(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TypeError("D1 command service must return a string-keyed mapping")
    payload = dict(cast(Mapping[str, object], value))
    json.dumps(payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return MappingProxyType(payload)
"""


def test_checked_payload_source_is_pinned_to_the_parent_bytes() -> None:
    assert inspect.getsource(d1_cli_module._checked_payload) == _CHECKED_PAYLOAD_SOURCE  # pyright: ignore[reportPrivateUsage]


class _Service:
    def __init__(self, result: object) -> None:
        self._result = result

    def execute(self, invocation: D1CliInvocation) -> object:
        del invocation
        return self._result


def _execute(tmp_path: Path, result: object) -> d1_cli_module.D1CliOutcome:
    command = next(iter(D1CliCommand))
    services = {item: _Service(result) for item in D1CliCommand}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "command": command.value,
                "payload": {"subject": "s"},
                "schema_version": D1_CLI_MANIFEST_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    return D1CliApplication(services).execute(command, manifest)


def test_a_payload_nesting_a_mapping_proxy_is_an_unexpected_defect(tmp_path: Path) -> None:
    outcome = _execute(tmp_path, {"nested": MappingProxyType({"k": 1})})
    assert outcome.exit_code is D1CliExitCode.UNEXPECTED_DEFECT
    assert outcome.status == "unexpected_defect"
    assert outcome.error_type == "TypeError"


def test_a_plain_json_payload_is_still_admitted(tmp_path: Path) -> None:
    outcome = _execute(tmp_path, {"nested": {"k": 1}})
    assert outcome.exit_code is D1CliExitCode.SUCCESS
    assert outcome.payload == {"nested": {"k": 1}}


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        ('{"command":"x","command":"y"}', "duplicate key: 'command'"),
        ('{"command":NaN}', "non-finite constant: 'NaN'"),
        ('{"command":1e999}', "non-finite float literal: '1e999'"),
    ],
)
def test_the_manifest_reader_refuses_strict_json_violations_as_invalid_evidence(
    tmp_path: Path, raw: str, needle: str
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(raw, encoding="utf-8")
    command = next(iter(D1CliCommand))
    outcome = D1CliApplication({item: _Service({}) for item in D1CliCommand}).execute(
        command, manifest
    )
    assert outcome.exit_code is D1CliExitCode.INVALID_EVIDENCE
    assert outcome.error_type == "D1CliInvalidEvidence"
    assert needle in (outcome.detail or "")


# --------------------------------------------------------------------------
# 4. canonical: the fourth canonical dump delegates to canonical_json


def _inline_canonical(value: dict[str, object]) -> bytes:
    """The dump d1_trainer_process_contracts._canonical carried before FOR-1582."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


_CANONICAL_CASES: list[dict[str, object]] = [
    {},
    {"b": 1, "a": [1, 2, {"z": None, "y": True}], "c": "café ☃"},
    {"schema_version": "x", "t": 0.1, "n": -0.0, "big": 10**30, "nested": {"k": [[], {}]}},
    {"quote": '"\\', "ctl": "\x00\x1f\n", "astral": "\U0001f600"},
]


@pytest.mark.parametrize("value", _CANONICAL_CASES)
def test_the_process_contract_canonical_dump_is_byte_equal_to_the_inline_dump(
    value: dict[str, object],
) -> None:
    canonical = d1_trainer_process_contracts._canonical  # pyright: ignore[reportPrivateUsage]
    assert canonical(value) == _inline_canonical(value)


def test_the_process_contract_canonical_dump_refuses_non_finite_like_the_inline_dump() -> None:
    canonical = d1_trainer_process_contracts._canonical  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="Out of range float"):
        canonical({"x": float("nan")})


def test_the_process_contract_canonical_dump_has_no_inline_json_dumps() -> None:
    source = inspect.getsource(d1_trainer_process_contracts._canonical)  # pyright: ignore[reportPrivateUsage]
    assert "json.dumps" not in source
    assert "canonical_json" in source


# --------------------------------------------------------------------------
# 5. SHA boundary: every repointed digest site refuses a mixed-case digest

_POD = "pod-sha-123"
_ATTEMPT = "a" * 64


class _Port:
    """One object answering every port the SHA-site fixtures need."""

    def add(self, **_kwargs: object) -> None: ...
    def push(self, **_kwargs: object) -> None: ...
    def pull(self, **_kwargs: object) -> None: ...
    def object_is_present(self, **_kwargs: object) -> bool:
        return True

    def publish(self, *_args: object, **_kwargs: object) -> object: ...
    def read(self, *_args: object, **_kwargs: object) -> object: ...
    def list(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
        return ()

    def restore(self, **_kwargs: object) -> None: ...


def _bound_r2_sync(attempt: str) -> object:
    with tempfile.TemporaryDirectory() as directory:
        adapter = D1CheckpointDvcSyncAdapter(
            repo_root=Path(directory), dvc=_Port(), cloud=_Port(), mailbox=_Port()
        )
        return D1BoundR2CheckpointSync(adapter, _POD, attempt, "run-1")


def _r2_worker_attempt(attempt: str) -> object:
    from .test_d1_trainer_object_transport import _ObjectStore

    store = _ObjectStore()
    store.seed(f"d1-mailbox/v1/{_POD}/{attempt}/commands/bootstrap.json", b"{}")
    with tempfile.TemporaryDirectory() as directory:
        worker = D1R2TrainerMailboxWorker(
            repo_root=Path(directory),
            pod_id=_POD,
            objects=store,
            catalog=tuple(profiles())[:1],
            worker_factory=lambda publisher: None,
            recovery=_Port(),
            wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
        )
        return worker._attempt()  # pyright: ignore[reportPrivateUsage]


def _mailbox_worker_attempt(attempt: str) -> object:
    with tempfile.TemporaryDirectory() as directory:
        volume = Path(directory).resolve()
        (volume / "d1-mailbox" / "v1" / _POD / attempt).mkdir(parents=True)
        worker = D1TrainerMailboxWorker(
            volume_root=volume,
            pod_id=_POD,
            worker_factory=lambda artifact_root: None,
            wait_policy=D1TrainerMailboxWaitPolicy(1.0, 0.01),
        )
        return worker._attempt()  # pyright: ignore[reportPrivateUsage]


def _sync_step(attempt: str) -> object:
    key = f"d1-mailbox/v1/{_POD}/{attempt}/sync/checkpoint-000000001.json"
    return D1CheckpointDvcRecoveryService._sync_step(  # pyright: ignore[reportPrivateUsage]
        D1TrainerMailboxObjectIdentity(key, 1, _DIGEST)
    )


def _trainer_start(field: str) -> Callable[[str], object]:
    def attest(value: str) -> object:
        digests = {
            "attempt_sha256": _DIGEST,
            "request_sha256": _DIGEST,
            "runtime_sha256": _DIGEST,
            "command_sha256": _DIGEST,
        }
        digests[field] = value
        return attest_d1_trainer_start(
            pod_id=_POD,
            process_id="1",
            trainer_started_at=datetime(2026, 1, 1, tzinfo=UTC),
            **digests,
        )

    return attest


def _launch_intent(field: str) -> Callable[[str], object]:
    def build(value: str) -> object:
        run = plan(profiles()[0], 101)
        base = D1OperatorLaunchIntent(
            plan=run,
            registered_design_digest=_DIGEST,
            image=_image(Path(".")),
            resource=_resource(),
            quoted_rate=start(run).quoted_rate,
            resource_activation=start(run).resource_activation,
            execution_segment_index=1,
            resume_checkpoint_sha256=_DIGEST,
            terminate_after_seconds=7200,
        )
        return replace(base, **{field: value})

    return build


@dataclass(frozen=True)
class _Site:
    name: str
    call: Callable[[str], object]
    exception: type[Exception]
    needle: str = _SHA_MESSAGE


_SITES = [
    # campaign
    _Site(
        "campaign_checkpoint_sha",
        lambda value: d1_campaign_checkpoint._sha(  # pyright: ignore[reportPrivateUsage]
            value, label="design_digest"
        ),
        ValueError,
    ),
    _Site("persisted_cursor", D1PersistedCampaignCursor, ValueError),
    _Site("operator_start_attempt", _trainer_start("attempt_sha256"), ValueError),
    _Site("operator_start_request", _trainer_start("request_sha256"), ValueError),
    _Site("operator_start_runtime", _trainer_start("runtime_sha256"), ValueError),
    _Site("operator_start_command", _trainer_start("command_sha256"), ValueError),
    _Site(
        "operator_intent_registered_design",
        _launch_intent("registered_design_digest"),
        ValueError,
    ),
    _Site(
        "operator_intent_resume_checkpoint",
        _launch_intent("resume_checkpoint_sha256"),
        ValueError,
    ),
    # ckptsync
    _Site(
        "dvc_sync_sha256",
        lambda value: d1_checkpoint_dvc_sync._sha256(  # pyright: ignore[reportPrivateUsage]
            value, label="checkpoint_sha256"
        ),
        ValueError,
    ),
    _Site(
        "dvc_recovery_checkpoint",
        d1_checkpoint_dvc_recovery._checkpoint_sha256,  # pyright: ignore[reportPrivateUsage]
        ValueError,
    ),
    _Site(
        "dvc_recovery_attempt",
        _sync_step,
        D1CheckpointDvcRecoveryError,
        "receipt_key",
    ),
    # closure
    _Site(
        "finalized_cell_sha",
        lambda value: d1_finalized_cell_custody._sha(  # pyright: ignore[reportPrivateUsage]
            value, "registered_design_digest"
        ),
        D1FinalizedCellCustodyError,
        "must be lowercase SHA-256",
    ),
    # construction
    _Site("bound_r2_sync_attempt", _bound_r2_sync, ValueError),
    # transport
    _Site(
        "evidence_adapter_attempt",
        d1_trainer_evidence_adapter._attempt,  # pyright: ignore[reportPrivateUsage]
        ValueError,
    ),
    _Site(
        "mailbox_contracts_digest",
        lambda value: d1_trainer_mailbox_contracts._digest(  # pyright: ignore[reportPrivateUsage]
            value, label="attempt_sha256"
        ),
        ValueError,
    ),
    _Site(
        "mailbox_worker_attempt", _mailbox_worker_attempt, D1TrainerMailboxWorkerError, "invalid"
    ),
    _Site(
        "object_transport_attempt",
        d1_trainer_object_transport._attempt,  # pyright: ignore[reportPrivateUsage]
        D1ObjectMailboxTransportError,
        "attempt_noncanonical",
    ),
    _Site("process_request_path", d1_trainer_request_path, ValueError),
    _Site("process_runtime_report_path", d1_trainer_runtime_report_path, ValueError),
    _Site(
        "process_start_report_attempt",
        lambda value: d1_trainer_start_report_path(value, _DIGEST),
        ValueError,
    ),
    _Site(
        "process_start_report_request",
        lambda value: d1_trainer_start_report_path(_DIGEST, value),
        ValueError,
    ),
    _Site(
        "process_report_sha",
        lambda value: d1_trainer_process_contracts._sha(  # pyright: ignore[reportPrivateUsage]
            value, "observed start attempt"
        ),
        ValueError,
    ),
    # composition
    _Site("r2_entrypoint_attempt", _r2_worker_attempt, D1R2TrainerMailboxWorkerError, "attempt"),
]


# W4-26(b): the object-transport request digest is RETAINED at its parent form
# (FOR-1626, third retained site). A mixed-case digest there is not refused as a
# malformed digest: it passes the any-case door and is refused by the session's
# own request_mismatch check, a D1ObjectMailboxTransportError. Repointing it at
# the lowercase-only door would turn that into a bare ValueError (a type change
# that standing ruling 17 forbids), so it is pinned here, apart from _SITES.
def test_object_transport_request_digest_retains_its_parent_exception_type() -> None:
    from .test_d1_trainer_object_transport import _ObjectStore, _transport

    payload = b"request-payload"
    real = hashlib.sha256(payload).hexdigest()
    mixed = real[:-1] + ("A" if real[-1] != "A" else "B")
    assert mixed != real and mixed.lower() != mixed
    with tempfile.TemporaryDirectory() as directory:
        transport = _transport(Path(directory).resolve(), _ObjectStore())
        transport.bootstrap(_POD, _ATTEMPT)
        transport.write_request(_POD, _ATTEMPT, d1_trainer_request_path(_ATTEMPT), payload)
        with pytest.raises(D1ObjectMailboxTransportError) as caught:
            transport.start_trainer(_POD, _ATTEMPT, mixed)
        assert type(caught.value) is D1ObjectMailboxTransportError
        assert (caught.value.stage, caught.value.code) == ("start", "request_mismatch")
        # Positive control: the genuine lowercase digest starts the trainer.
        transport.start_trainer(_POD, _ATTEMPT, real)
        # A digest that is not hex at all still fails at the any-case door.
        with pytest.raises(ValueError) as malformed:
            transport.start_trainer(_POD, _ATTEMPT, "z" * 64)
        assert type(malformed.value) is ValueError


@pytest.mark.parametrize("site", [pytest.param(site, id=site.name) for site in _SITES])
def test_every_boundary_sha_site_refuses_a_mixed_case_digest_with_its_own_error(
    site: _Site,
) -> None:
    with pytest.raises(site.exception) as caught:
        site.call(_MIXED)
    assert type(caught.value) is site.exception
    assert site.needle in str(caught.value) or site.needle in repr(caught.value)


@pytest.mark.parametrize("site", [pytest.param(site, id=site.name) for site in _SITES])
def test_every_boundary_sha_site_still_admits_a_lowercase_digest(site: _Site) -> None:
    # Positive control: the refusal above is the digest check, not the fixture.
    # The call must RETURN. The site's own exception is the digest refusal and is
    # named as such; any other exception is a fixture defect and propagates, so a
    # message that merely mentions a short needle ("invalid", "attempt") can
    # neither pass this control by accident nor fail it for the wrong reason.
    try:
        site.call(_DIGEST)
    except site.exception as error:
        pytest.fail(f"{site.name} refused a lowercase digest: {error!r}")
