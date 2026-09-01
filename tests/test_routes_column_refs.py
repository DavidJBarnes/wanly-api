"""Every Model.attribute anywhere in app/ must actually exist on that model.

`GET /jobs` returned 500 in production for a day because #219 dropped
`Segment.faceswap_enabled` from the model and the migration, but left a
`func.bool_or(Segment.faceswap_enabled)` aggregation in `list_jobs`. The result
was built into a dict that nothing read, so no test touched it and no import
failed -- SQLAlchemy only resolves the attribute when the query is constructed,
which happens per request.

It was invisible for an extra reason worth remembering: the block is guarded by
`if job_ids:`, so an empty jobs table returned 200. The endpoint only broke once
a job existed again, which made it look like a data problem rather than a code
one.

A dropped column is normal. A route still naming it is not, and it costs a
production outage rather than a red test, so this walks the AST of every route
module and checks each `Model.attr` against the mapped model.
"""
import ast
import pathlib

import pytest

from app import models

APP = pathlib.Path(__file__).parent.parent / "app"

# Mapped classes by name, e.g. {"Segment": Segment, "Job": Job, ...}
MODELS = {
    name: obj
    for name, obj in vars(models).items()
    if isinstance(obj, type) and hasattr(obj, "__table__")
}


def _model_attribute_refs(tree: ast.AST):
    """Yield (model_name, attr, lineno) for every `SomeModel.attr` in the tree."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in MODELS
        ):
            yield node.value.id, node.attr, node.lineno


@pytest.mark.parametrize(
    "path", sorted(APP.rglob("*.py")), ids=lambda p: str(p.relative_to(APP))
)
def test_module_only_references_columns_that_exist(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    missing = [
        f"{path.name}:{lineno} {model}.{attr}"
        for model, attr, lineno in _model_attribute_refs(tree)
        if not hasattr(MODELS[model], attr)
    ]
    assert not missing, (
        "module references attributes that no longer exist on the model — a dropped "
        "column left behind will 500 at request time, not at import:\n  "
        + "\n  ".join(missing)
    )
