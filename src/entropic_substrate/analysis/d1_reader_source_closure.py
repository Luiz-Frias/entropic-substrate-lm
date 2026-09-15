"""Deterministic local-source closure for the authoritative D1 readers."""

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path
from typing import NoReturn

__all__ = [
    "D1_READER_ROOT_MODULES",
    "D1ReaderSourceClosureError",
    "D1ReaderSourceFile",
    "discover_d1_reader_source_files",
    "installed_d1_reader_source_files",
]

_PACKAGE = "entropic_substrate"
D1_READER_ROOT_MODULES = (
    "entropic_substrate.analysis.d1_reader_source_closure",
    "entropic_substrate.telemetry",
)


class D1ReaderSourceClosureError(ValueError):
    """The local reader dependency closure cannot be proven exactly."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1ReaderSourceClosureError(message)
    raise D1ReaderSourceClosureError(message) from error


@dataclass(frozen=True, slots=True)
class D1ReaderSourceFile:
    identity: str
    path: str
    sha256: str

    @property
    def manifest(self) -> dict[str, str]:
        return {"identity": self.identity, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class _ResolvedSource:
    module: str
    identity: str
    target: Path
    relative: str
    package: bool
    source: bytes


def _source_root(value: object) -> Path:
    if not isinstance(value, Path):
        raise TypeError("D1 reader source root must be Path")
    lexical = Path(os.path.abspath(value))
    if lexical.is_symlink() or not lexical.is_dir() or lexical.resolve() != lexical:
        _fail("D1 reader source root must be an existing real directory")
    package = lexical / _PACKAGE
    if package.is_symlink() or not package.is_dir():
        _fail("D1 reader source root must contain a real entropic_substrate package")
    return lexical


def _local(module: str) -> bool:
    return module == _PACKAGE or module.startswith(f"{_PACKAGE}.")


def _module_name(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or any(not part.isidentifier() for part in value.split("."))
    ):
        _fail("D1 reader closure module identities must be dotted Python names")
    if not _local(value):
        _fail("D1 reader closure roots must remain inside entropic_substrate")
    return value


def _regular_target(root: Path, target: Path) -> None:
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
        _fail(f"D1 reader source must be a regular file below the source root: {target}")
    cursor = target.parent
    while cursor != root:
        if cursor.is_symlink():
            _fail(f"D1 reader source path may not traverse a symlink: {target}")
        cursor = cursor.parent


def _resolve(root: Path, module: str, *, required: bool) -> _ResolvedSource | None:
    module = _module_name(module)
    stem = root.joinpath(*module.split("."))
    plain = stem.with_suffix(".py")
    package = stem / "__init__.py"
    candidates = tuple(path for path in (plain, package) if path.exists())
    if not candidates:
        if required:
            _fail(f"D1 reader local import has no source file: {module}")
        return None
    if len(candidates) != 1:
        _fail(f"D1 reader local import is ambiguous between module and package: {module}")
    target = candidates[0]
    _regular_target(root, target)
    is_package = target == package
    relative = target.relative_to(root).as_posix()
    identity = f"{module}.__init__" if is_package else module
    try:
        source = target.read_bytes()
        source.decode("utf-8")
    except (OSError, UnicodeError) as error:
        _fail(f"D1 reader source is not readable UTF-8: {relative}", error)
    return _ResolvedSource(module, identity, target, relative, is_package, source)


def _package(resolved: _ResolvedSource) -> str:
    return resolved.module if resolved.package else resolved.module.rpartition(".")[0]


def _relative_module(node: ast.ImportFrom, resolved: _ResolvedSource) -> str:
    if node.level == 0:
        if node.module is None:
            _fail("absolute D1 reader import must name a module")
        return node.module
    current = _package(resolved)
    try:
        return resolve_name(f"{'.' * node.level}{node.module or ''}", current)
    except (ImportError, ValueError) as error:
        _fail(f"D1 reader relative import cannot be resolved in {resolved.identity}", error)


def _importlib_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    functions = {"__import__"}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name for alias in node.names if alias.name == "importlib"
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "importlib":
            functions.update(
                alias.asname or alias.name for alias in node.names if alias.name == "import_module"
            )
    return functions, modules


def _dynamic_name(
    node: ast.Call,
    resolved: _ResolvedSource,
    *,
    function_aliases: set[str],
    module_aliases: set[str],
) -> str | None:
    function = node.func
    dynamic = (isinstance(function, ast.Name) and function.id in function_aliases) or (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id in module_aliases
        and function.attr == "import_module"
    )
    if not dynamic:
        return None
    if (
        not node.args
        or not isinstance(node.args[0], ast.Constant)
        or not isinstance(node.args[0].value, str)
    ):
        _fail(f"D1 reader dynamic import in {resolved.identity} must use a literal module name")
    name = node.args[0].value
    if name.startswith("."):
        try:
            return resolve_name(name, _package(resolved))
        except (ImportError, ValueError) as error:
            _fail(f"D1 reader dynamic relative import cannot be resolved: {name}", error)
    return name


def _dependencies(root: Path, resolved: _ResolvedSource) -> set[str]:
    try:
        tree = ast.parse(resolved.source.decode("utf-8"), filename=resolved.relative)
    except SyntaxError as error:
        _fail(f"D1 reader source cannot be parsed: {resolved.relative}", error)
    function_aliases, module_aliases = _importlib_aliases(tree)
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _relative_module(node, resolved)
            names = (base,)
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                if (
                    alias.name != "*"
                    and _local(candidate)
                    and _resolve(root, candidate, required=False)
                ):
                    dependencies.add(candidate)
        elif isinstance(node, ast.Call):
            dynamic = _dynamic_name(
                node,
                resolved,
                function_aliases=function_aliases,
                module_aliases=module_aliases,
            )
            names = () if dynamic is None else (dynamic,)
        dependencies.update(name for name in names if _local(name))
    return dependencies


def _ancestors(root: Path, module: str) -> set[str]:
    result: set[str] = set()
    parts = module.split(".")
    for end in range(1, len(parts)):
        candidate = ".".join(parts[:end])
        resolved = _resolve(root, candidate, required=False)
        if resolved is not None and resolved.package:
            result.add(candidate)
    return result


def discover_d1_reader_source_files(
    source_root: Path,
    *,
    roots: tuple[str, ...],
) -> tuple[D1ReaderSourceFile, ...]:
    """Close literal local imports beneath an explicit source boundary."""
    root = _source_root(source_root)
    if type(roots) is not tuple or not roots:
        raise TypeError("D1 reader closure requires a nonempty tuple of root modules")
    pending = {_module_name(module) for module in roots}
    resolved_by_identity: dict[str, _ResolvedSource] = {}
    while pending:
        module = min(pending)
        pending.remove(module)
        resolved = _resolve(root, module, required=True)
        if resolved is None:
            _fail("D1 reader required module resolution returned no source")
        if resolved.identity in resolved_by_identity:
            continue
        resolved_by_identity[resolved.identity] = resolved
        pending.update(_ancestors(root, module))
        pending.update(_dependencies(root, resolved))
    return tuple(
        D1ReaderSourceFile(
            identity=resolved.identity,
            path=resolved.relative,
            sha256=hashlib.sha256(resolved.source).hexdigest(),
        )
        for resolved in sorted(resolved_by_identity.values(), key=lambda item: item.identity)
    )


def installed_d1_reader_source_files() -> tuple[D1ReaderSourceFile, ...]:
    """Discover the closure from the installed package's real source tree."""
    source_root = Path(__file__).resolve().parents[2]
    return discover_d1_reader_source_files(source_root, roots=D1_READER_ROOT_MODULES)
