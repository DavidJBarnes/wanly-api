"""A module-private helper that is called must also be defined.

`_resolve_trigger` was deleted from `app/routes/segments.py` as collateral in #216 — the
diff sliced from `_resolve_loras` to `_resolve_wildcards` and it sat between them. Its call
site survived, so every `POST /jobs/{id}/segments` raised NameError from the moment that
merged: Next Segment could not work at all, and nothing failed until a request arrived.

Python does not catch this. The name is resolved when the line executes, so the module
imports cleanly, every other endpoint works, and the tests that never call the route pass.

That is the same shape as the `GET /jobs` 500 (a dropped column still named by a query) and
it went unnoticed for the same reason: the failure is per request, not per import. This walks
the AST and asserts every `_helper(...)` a module calls is defined, imported, or assigned in
it.
"""

import ast
import builtins
import pathlib

import pytest

APP = pathlib.Path(__file__).parent.parent / "app"


def _bound_names(tree: ast.AST) -> set[str]:
    """Everything a module binds at any level: defs, classes, imports, assignments, params."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _called_private_names(tree: ast.AST) -> set[tuple[str, int]]:
    """`_helper(...)` call targets. Attribute calls (`x._m()`) are somebody else's problem."""
    out = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("_")
            and not node.func.id.startswith("__")
        ):
            out.add((node.func.id, node.lineno))
    return out


@pytest.mark.parametrize(
    "path", sorted(APP.rglob("*.py")), ids=lambda p: str(p.relative_to(APP))
)
def test_every_private_helper_called_is_also_defined(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    bound = _bound_names(tree) | set(dir(builtins))
    missing = sorted(
        f"{path.name}:{line} {name}()"
        for name, line in _called_private_names(tree)
        if name not in bound
    )
    assert not missing, (
        "private helper called but never defined — this raises NameError at request time, "
        "not at import, so the module loads and only the endpoint breaks:\n  "
        + "\n  ".join(missing)
    )
