"""A column dropped by a migration must not still be named anywhere in app/.

Two production 500s came from this in one day:

* `GET /jobs` aggregated `Segment.faceswap_enabled` after #219 dropped it, so the job list
  was empty for everyone.
* `DELETE /segments/{id}` walked `segment.faceswap_image` in its S3 cleanup, so deleting a
  segment failed with AttributeError.

The AST guard added with the first one only caught `Segment.attr` — the class. The second
was `segment.attr` on an instance, which no amount of class-name matching sees.

So this works the other way round: read the column names that migrations actually DROP, and
assert none of them is used as an attribute anywhere in app/. Attribute names are checked
regardless of what they hang off, which is the only way to catch an instance access.

Only `upgrade()` counts. Most `drop_column` calls in this tree live in `downgrade()`, where
they undo an add — treating those as dropped would flag half the schema.

A dropped name that still exists on some model is excluded: a column can be dropped from one
table and live on in another, and `videos.error_message` going does not make
`segments.error_message` suspect.
"""

import ast
import pathlib

import pytest

from app import models

ROOT = pathlib.Path(__file__).parent.parent
APP = ROOT / "app"
VERSIONS = ROOT / "alembic" / "versions"


def _strings_in(value) -> set[str]:
    """Every string in a literal, however nested. `("faceswap_image", sa.Text(), None)` in a
    module constant yields its name even though the row also holds a type and a default."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        out: set[str] = set()
        for v in value:
            out |= _strings_in(v)
        return out
    return set()


def _module_constants(tree: ast.AST) -> dict[str, set[str]]:
    """Module-level list/tuple constants, as the set of strings each contains.

    `sa.Text()` inside a row makes the row unevaluable, so each element is tried on its own
    and the parts that are literals are kept.
    """
    consts: dict[str, set[str]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        found: set[str] = set()
        for element in node.value.elts:
            # Rows are ("column_name", sa.Type(), default). Only the first entry is a column
            # name -- taking every string in the row also collected server_defaults, which put
            # "false" in the dropped set.
            head = element.elts[0] if isinstance(element, (ast.List, ast.Tuple)) and element.elts else element
            try:
                found |= _strings_in(ast.literal_eval(head))
            except (ValueError, SyntaxError):
                pass
        consts[node.targets[0].id] = found
    return consts


def _dropped_in_upgrades() -> set[str]:
    names: set[str] = set()
    for path in VERSIONS.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        consts = _module_constants(tree)
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef) and node.name == "upgrade"):
                continue

            # Which names are loop variables over a module constant, and what strings that
            # constant holds. #077 drops nine faceswap columns as
            # `for name, _type, _default in _COLUMNS: op.drop_column("segments", name)`,
            # so the column name is never a literal at the call site.
            loop_sources: dict[str, set[str]] = {}
            for loop in ast.walk(node):
                if not isinstance(loop, ast.For) or not isinstance(loop.iter, ast.Name):
                    continue
                held = consts.get(loop.iter.id)
                if held is None:
                    continue
                targets = (
                    loop.target.elts if isinstance(loop.target, ast.Tuple) else [loop.target]
                )
                for t in targets:
                    if isinstance(t, ast.Name):
                        loop_sources[t.id] = held

            for call in ast.walk(node):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "drop_column"
                    and len(call.args) >= 2
                ):
                    continue
                arg = call.args[1]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.add(arg.value)
                elif isinstance(arg, ast.Name) and arg.id in loop_sources:
                    names |= loop_sources[arg.id]
    return names


def _live_column_names() -> set[str]:
    live: set[str] = set()
    for obj in vars(models).values():
        if isinstance(obj, type) and hasattr(obj, "__table__"):
            live |= {c.name for c in obj.__table__.columns}
            live |= {r.key for r in obj.__mapper__.relationships}
    return live


GONE = sorted(_dropped_in_upgrades() - _live_column_names())


def test_the_migrations_actually_drop_something():
    """Guards the guard: a parsing change that silently found nothing would pass everything."""
    assert GONE, "no dropped columns found — the migration parsing has broken"


@pytest.mark.parametrize("path", sorted(APP.rglob("*.py")), ids=lambda p: str(p.relative_to(APP)))
def test_no_module_references_a_dropped_column(path):
    gone = set(GONE)
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = sorted(
        f"{path.name}:{n.lineno} .{n.attr}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in gone
    )
    assert not hits, (
        "references a column that a migration dropped — this raises at request time, not at "
        "import, so the module loads and only the endpoint breaks:\n  " + "\n  ".join(hits)
    )
