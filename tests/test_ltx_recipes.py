"""LTX recipes as data.

The .ods was a test harness that became load-bearing — the POC built guards AROUND it
("never regenerate the sheet") rather than replacing it. These tests cover what replaced it.
"""

import ast
from pathlib import Path

from app.ltx_stack import LTX_STACK

MIGRATION = Path("alembic/versions/071_ltx_recipes_are_data.py")


def _migration_const(name: str):
    """Read a literal out of the migration without importing alembic."""
    tree = ast.parse(MIGRATION.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:  # type: ignore[attr-defined]
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {MIGRATION}")


def test_seed_carries_every_recipe_from_the_sheet():
    chars = _migration_const("_CHARACTERS")
    recipes = _migration_const("_RECIPES")
    assert len(chars) == 3
    assert len(recipes) == 24
    assert {c["name"] for c in chars} == {"k3lly2026", "k3llydw", "p@y"}
    # 8 poses per character, which is what was validated.
    for c in chars:
        assert sum(1 for r in recipes if r["character"] == c["name"]) == 8


def test_prompts_are_stored_resolved_not_as_sheet_keys():
    """The sheet stored keys like P.MISS_POV against a definitions tab.

    That indirection existed to avoid retyping text in a spreadsheet. Carrying it into a
    database would mean the console could not simply edit the prompt, which is the entire
    reason recipes became rows.
    """
    for r in _migration_const("_RECIPES"):
        assert not r["prompt"].startswith("P."), f"{r['name']} still holds a sheet key"
        assert len(r["prompt"]) > 40, f"{r['name']} prompt looks like a key, not text"


def test_the_global_stack_is_stored_once_not_per_recipe():
    """Measured across all 24: only char_lora and prompt varied.

    Every other field had exactly ONE value, so it lives in LTX_STACK. Storing a global 24
    times is how it silently stops being global — one row gets edited, nothing complains, and
    two recipes that should be identical are not.
    """
    recipe_fields = set(_migration_const("_RECIPES")[0])
    assert recipe_fields == {"character", "name", "prompt", "validated"}
    for global_only in ("checkpoint", "content_lora", "distill", "cfg", "steps_stage_1"):
        assert global_only in LTX_STACK
        assert global_only not in recipe_fields


def test_content_lora_is_none_and_that_is_deliberate():
    """Dropping DR34ML4Y is what removed the motion horror.

    It was a third LoRA competing for the same layers as the character LoRA, and the
    checkpoint already carries the NSFW training it was providing. If this ever reads as a
    filename again, that is a regression to a stack that produced body horror.
    """
    assert LTX_STACK["content_lora"] == "none"


def test_per_stage_strengths_survive_as_two_numbers():
    """0.8 / 1.5 is validated. Stage 1 decides body and anatomy; stage 2 resolves the face.

    Collapsing them to one number is a different configuration, not a simplification — and
    the flat-0.8 panel is on record as worse on both axes.
    """
    for c in _migration_const("_CHARACTERS"):
        assert c["strength_stage_1"] == 0.8
        assert c["strength_stage_2"] == 1.5
        assert c["strength_stage_1"] != c["strength_stage_2"]


def test_migration_data_is_inlined_not_imported():
    """A migration is a record of what happened, so its data must be frozen.

    Importing the rows from a module would mean a later edit to that module silently changes
    what this migration did.
    """
    src = MIGRATION.read_text()
    assert "_CHARACTERS = [" in src and "_RECIPES = [" in src
    assert "from app.ltx_seed" not in src
    assert "import LTX_RECIPES" not in src


def test_the_sheet_parser_is_gone():
    """The harness does not come with us. Leaving it would invite a second source of truth."""
    assert not Path("app/ltx_sheet.py").exists()
