"""The seed is locked across a continuation chain, and _resolve_trigger exists.

Two things that had both broken at once, found when a "Next Segment" click did nothing.
"""

import ast
import inspect
from pathlib import Path

from app.routes import segments


def test_resolve_trigger_still_exists():
    """It was removed as collateral damage, and nothing noticed for two merges.

    _resolve_loras was deleted by slicing from its `def` to the next one — and
    _resolve_trigger sat between them. The call site survived, so every
    POST /jobs/{id}/segments raised NameError: Next Segment could never have worked.

    Both halves are asserted: a call with no definition is exactly the state that shipped.
    """
    src = Path(inspect.getfile(segments)).read_text()
    tree = ast.parse(src)
    defined = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    called = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_resolve_trigger" in defined, "_resolve_trigger is called but not defined"
    assert "_resolve_trigger" in called, "nothing resolves <TRIGGER> before wildcards run"


def test_no_module_level_function_is_called_but_undefined():
    """The general form. A slice that removes a definition and leaves its call site is not a
    syntax error, so nothing catches it until the endpoint is exercised."""
    src = Path(inspect.getfile(segments)).read_text()
    tree = ast.parse(src)
    defined = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    private_calls = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id.startswith("_") and not n.func.id.startswith("__")
    }
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported |= {a.asname or a.name for a in n.names}
    missing = private_calls - defined - imported
    assert not missing, f"called but never defined or imported: {sorted(missing)}"


def test_the_claim_does_not_vary_the_seed_by_index():
    """The seed is LOCKED across a chain: every segment runs on the job's seed.

    It used to be `job.seed + segment.index`, justified as decorrelating later segments "so
    they don't repeat the same motion pattern" — WAN reasoning, where drift was spread around
    deliberately. Under LTX a continuation should look like the same shot continuing, and the
    seed is the single biggest determinant of what a take looks like.

    Verified against a real Postgres too: two segments of one job both claim on job.seed, and
    segment 1's start image resolves to segment 0's last frame.
    """
    src = Path(inspect.getfile(segments)).read_text()
    assert "job.seed + segment.index" not in src, (
        "the claim varies the seed by index again — a continuation would run on a different "
        "seed from the segment it continues from"
    )
    assert "seed=segment.seed if segment.seed is not None else job.seed," in src
