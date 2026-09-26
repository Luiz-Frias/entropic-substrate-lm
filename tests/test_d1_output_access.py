"""Linux ACL output access grants only the fixed D1 container identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.ladder.analyze.d1_output_access import (
    D1AclCommandResult,
    D1OutputAccessError,
    D1SetfaclOutputAccess,
)

pytestmark = pytest.mark.unit


@dataclass
class _AclRunner:
    entry: str | None = None
    resolved_name: bool = False
    returncode: int = 0
    ignore_mutation: bool = False
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, argv: tuple[str, ...]) -> D1AclCommandResult:
        self.calls.append(argv)
        if argv[0] == "getfacl":
            identity = "nobody" if self.resolved_name else "65532"
            line = "" if self.entry is None else f"user:{identity}:{self.entry}\n"
            return D1AclCommandResult(argv, self.returncode, line, "")
        if not self.ignore_mutation:
            self.entry = "rwx" if argv[1:3] == ("-m", "u:65532:rwx") else None
        return D1AclCommandResult(argv, self.returncode, "", "")


def _access(
    tmp_path: Path,
    **options: object,
) -> tuple[Path, _AclRunner, D1SetfaclOutputAccess]:
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    runner = _AclRunner(**options)
    return output, runner, D1SetfaclOutputAccess(runner=runner)


def test_acl_grant_and_revoke_are_exact_and_preserve_base_mode(tmp_path: Path) -> None:
    output, runner, access = _access(tmp_path)

    access.grant(output, uid=65532, gid=65532)
    assert output.stat().st_mode & 0o777 == 0o700
    access.revoke(output, uid=65532, gid=65532)

    assert output.stat().st_mode & 0o777 == 0o700
    assert runner.entry is None
    assert runner.calls == [
        ("getfacl", "-cpn", "--", str(output)),
        ("setfacl", "-m", "u:65532:rwx", "--", str(output)),
        ("getfacl", "-cpn", "--", str(output)),
        ("setfacl", "-x", "u:65532", "--", str(output)),
        ("getfacl", "-cpn", "--", str(output)),
    ]


def test_acl_refuses_existing_entry_wrong_identity_and_lifecycle_drift(tmp_path: Path) -> None:
    output, runner, access = _access(tmp_path, entry="r-x")
    with pytest.raises(D1OutputAccessError, match="pre-existing"):
        access.grant(output, uid=65532, gid=65532)
    assert len(runner.calls) == 1

    runner.entry = None
    for uid, gid in ((0, 65532), (65532, 0)):
        with pytest.raises(D1OutputAccessError, match="65532"):
            access.grant(output, uid=uid, gid=gid)

    access.grant(output, uid=65532, gid=65532)
    with pytest.raises(D1OutputAccessError, match="active"):
        access.grant(output, uid=65532, gid=65532)
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    with pytest.raises(D1OutputAccessError, match="lease"):
        access.revoke(other, uid=65532, gid=65532)
    access.revoke(output, uid=65532, gid=65532)

    output.chmod(0o755)
    with pytest.raises(D1OutputAccessError, match="0700"):
        access.grant(output, uid=65532, gid=65532)


def test_acl_rejects_resolved_name_output_despite_numeric_command(tmp_path: Path) -> None:
    output, runner, access = _access(tmp_path, entry="r-x", resolved_name=True)
    with pytest.raises(D1OutputAccessError, match="nonnumeric"):
        access.grant(output, uid=65532, gid=65532)
    assert runner.calls == [("getfacl", "-cpn", "--", str(output))]


@pytest.mark.parametrize(("returncode", "ignore_mutation"), [(9, False), (0, True)])
def test_acl_fails_on_command_or_verification_failure(
    tmp_path: Path,
    returncode: int,
    ignore_mutation: bool,
) -> None:
    output, _, access = _access(
        tmp_path,
        returncode=returncode,
        ignore_mutation=ignore_mutation,
    )
    with pytest.raises(D1OutputAccessError):
        access.grant(output, uid=65532, gid=65532)
