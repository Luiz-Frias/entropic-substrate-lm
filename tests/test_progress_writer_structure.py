"""FOR-1718 (h1-l4b residual): ``runner.write_progress_receipt`` is the one writer, structurally.

Before this file the claim held "by grep and convention only": a module that built an inline
``h-probe-training-progress-v2`` mapping and wrote it with its own ``write_json`` stayed GREEN
(T6j2). ``tests/test_h_pair_request.py::test_only_the_runner_assigns_the_progress_v2_schema_name``
pins the two commonest shapes (a dict display and a keyword argument). This scan closes the
rest of the surface and proves its own teeth, because a scan that finds nothing is also what a
broken scan reports:

* a write-shaped use of the schema name is a dict display keyed ``"schema"``, a keyword
  ``schema=``, an assignment to a ``["schema"]`` subscript, or ``setdefault("schema", ...)``,
  whose value is the literal, the ``PROGRESS_SCHEMA_V2`` constant, an ``import ... as`` alias of
  it, an attribute ``<module>.PROGRESS_SCHEMA_V2`` or a name bound to any of those;
* every other use (a comparison, a positional argument, a set member, an f-string, a docstring)
  is read side or prose, and the scan says so (see ``READ_SIDE``).

The literal itself is also listed: outside ``runner`` it may appear in code only where this
file names it, so a reader that grows a second spelling is a visible edit here.
"""

import ast
from pathlib import Path

import pytest

import entropic_substrate
from entropic_substrate.ladder.run.runner import PROGRESS_SCHEMA_V2

pytestmark = pytest.mark.unit

CONSTANT = "PROGRESS_SCHEMA_V2"
WRITER = "ladder/run/runner.py:write_progress_receipt"
# The literal outside ``runner``, in code (docstrings excluded). ``trace_capture`` lists the
# name in its closed read-side vocabulary (``_SCHEMAS``): the capture closure admits it, and
# nothing there writes it. Anything else is a new spelling of the name and must be added HERE.
READ_SIDE: frozenset[str] = frozenset({"ladder/run/trace_capture.py:<module>"})
SRC = Path(entropic_substrate.__file__).parent


class _Scan(ast.NodeVisitor):
    """The write-shaped uses and the code literals of the progress-v2 name in one module."""

    def __init__(self, relative: str, tree: ast.Module) -> None:
        self.relative = relative
        self.owners: list[str] = []
        self.writers: list[str] = []
        self.literals: list[str] = []
        self.names = self._names_bound_to_the_schema(tree)
        self.docstrings = {
            id(body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and (body := node.body)
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        }

    @staticmethod
    def _names_bound_to_the_schema(tree: ast.Module) -> set[str]:
        names = {CONSTANT}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names |= {a.asname for a in node.names if a.name == CONSTANT and a.asname}
        grew = True
        while grew:  # a name bound to the constant, the literal or another such name
            grew = False
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and _is_schema(node.value, names)
                    and (fresh := {t.id for t in node.targets if isinstance(t, ast.Name)} - names)
                ):
                    names |= fresh
                    grew = True
        return names

    @property
    def owner(self) -> str:
        return self.owners[-1] if self.owners else "<module>"

    def _site(self) -> str:
        return f"{self.relative}:{self.owner}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.owners.append(node.name)
        self.generic_visit(node)
        self.owners.pop()

    # the same body serves ``async def``; only the node type in the annotation is narrower, and
    # ``NodeVisitor`` dispatches by class name at run time, not by that annotation
    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value == PROGRESS_SCHEMA_V2 and id(node) not in self.docstrings:
            self.literals.append(self._site())

    def visit_Dict(self, node: ast.Dict) -> None:
        if any(
            isinstance(k, ast.Constant) and k.value == "schema" and _is_schema(v, self.names)
            for k, v in zip(node.keys, node.values, strict=True)
        ):
            self.writers.append(self._site())
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        keyword = any(k.arg == "schema" and _is_schema(k.value, self.names) for k in node.keywords)
        default = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "schema"
            and _is_schema(node.args[1], self.names)
        )
        if keyword or default:
            self.writers.append(self._site())
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_schema(node.value, self.names) and any(_is_schema_slot(t) for t in node.targets):
            self.writers.append(self._site())
        self.generic_visit(node)


def _is_schema(node: ast.AST | None, names: set[str]) -> bool:
    return (
        (isinstance(node, ast.Name) and node.id in names)
        or (isinstance(node, ast.Attribute) and node.attr == CONSTANT)
        or (isinstance(node, ast.Constant) and node.value == PROGRESS_SCHEMA_V2)
    )


def _is_schema_slot(target: ast.AST) -> bool:
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "schema"
    )


def scan(source: str, relative: str = "x.py") -> _Scan:
    tree = ast.parse(source)
    found = _Scan(relative, tree)
    found.visit(tree)
    return found


# ------------------------------------------------------------------------- the real tree
def scanned_tree() -> list[_Scan]:
    return [
        scan(path.read_text(encoding="utf-8"), path.relative_to(SRC).as_posix())
        for path in sorted(SRC.rglob("*.py"))
    ]


def test_the_runner_is_the_only_writer_of_a_progress_v2_receipt() -> None:
    sites = sorted(site for found in scanned_tree() for site in found.writers)
    assert sites == [WRITER], sites


def test_the_progress_v2_literal_appears_in_code_only_where_it_is_listed() -> None:
    sites = {
        site
        for found in scanned_tree()
        for site in found.literals
        if not site.startswith("ladder/run/runner.py:")
    }
    assert sites == READ_SIDE, sites
    # and the runner's own spelling is the constant that every other module imports
    assert [s for f in scanned_tree() for s in f.literals if s.startswith("ladder/run/runner.py")]


# ----------------------------------------------------------------- the scan's own teeth
WRITER_SHAPES = {
    "dict display": 'row = {"schema": PROGRESS_SCHEMA_V2, "step": 1}',
    "dict display, literal": 'row = {"schema": "h-probe-training-progress-v2"}',
    "keyword": "row = dict(schema=PROGRESS_SCHEMA_V2)",
    "keyword, literal": 'row = dict(schema="h-probe-training-progress-v2")',
    "subscript": 'row["schema"] = PROGRESS_SCHEMA_V2',
    "subscript, literal": 'row["schema"] = "h-probe-training-progress-v2"',
    "setdefault": 'row.setdefault("schema", PROGRESS_SCHEMA_V2)',
    "import alias": (
        "from entropic_substrate.ladder.run.runner import PROGRESS_SCHEMA_V2 as NAME\n"
        'row = {"schema": NAME}'
    ),
    "attribute": 'row = {"schema": runner.PROGRESS_SCHEMA_V2}',
    "bound name": 'NAME = PROGRESS_SCHEMA_V2\nrow = {"schema": NAME}',
    "bound literal": 'NAME = "h-probe-training-progress-v2"\nrow = {"schema": NAME}',
    "inside a function": 'def spill(row):\n    return {"schema": PROGRESS_SCHEMA_V2}',
}


@pytest.mark.parametrize("shape", sorted(WRITER_SHAPES))
def test_the_scan_flags_each_shape_of_inline_writer(shape: str) -> None:
    assert scan(WRITER_SHAPES[shape]).writers, shape


def test_the_scan_names_the_function_that_owns_a_writer() -> None:
    found = scan(WRITER_SHAPES["inside a function"], "ladder/run/other.py")
    assert found.writers == ["ladder/run/other.py:spill"]


READER_SHAPES = {
    "comparison": 'if row["schema"] != PROGRESS_SCHEMA_V2:\n    raise ValueError',
    "get comparison": 'if row.get("schema") == PROGRESS_SCHEMA_V2:\n    pass',
    "positional argument": "found = schema_records(output, PROGRESS_SCHEMA_V2, max_bytes=1)",
    "other key": 'row = {"kind": PROGRESS_SCHEMA_V2}',
    "set member": 'NAMES = frozenset(("h-probe-training-progress-v2",))',
    "f-string": 'message = f"version is {PROGRESS_SCHEMA_V2}"',
    "another schema": 'row = {"schema": "h-checkpoint-seal-v1"}',
    "docstring": 'def f():\n    """Writes h-probe-training-progress-v2."""\n',
    "module docstring": '"""h-probe-training-progress-v2."""\nx = 1',
}


@pytest.mark.parametrize("shape", sorted(READER_SHAPES))
def test_the_scan_leaves_read_side_uses_and_prose_alone(shape: str) -> None:
    assert scan(READER_SHAPES[shape]).writers == [], shape


def test_the_scan_lists_a_code_literal_but_never_a_docstring_one() -> None:
    assert scan(READER_SHAPES["set member"]).literals == ["x.py:<module>"]
    assert scan(READER_SHAPES["docstring"]).literals == []
    assert scan(READER_SHAPES["module docstring"]).literals == []
