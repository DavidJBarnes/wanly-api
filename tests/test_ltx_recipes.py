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
    for global_only in ("checkpoint", "distill", "cfg", "steps_stage_1"):
        assert global_only in LTX_STACK
        assert global_only not in recipe_fields

    # content_lora used to be in that list. It is no longer global-ONLY: a pose can now
    # override it, because "what is happening" is a property of the pose and one global
    # cannot say "sfbehind for the from-behind poses and nothing for the rest" (console#395).
    # It stays in the stack as the DEFAULT, and no seeded recipe carries one — which is the
    # part that still matters here, and is what keeps every existing pose rendering the same.
    assert "content_lora" in LTX_STACK
    assert "content_lora" not in recipe_fields


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


# --- 072: a recipe is a POSE, not a pose-per-character ------------------------------------

POSES_MIGRATION = Path("alembic/versions/072_recipes_are_poses.py")


def _poses_const(name: str):
    tree = ast.parse(POSES_MIGRATION.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {POSES_MIGRATION}")


def test_twenty_four_rows_collapse_to_eight_poses():
    """8 poses x 3 characters was 24 rows of which 16 were duplicates.

    Verified against the seeded data before the change: strip the leading trigger token and
    all 8 poses are character-agnostic.
    """
    assert len(_poses_const("_POSES")) == 8


def test_every_pose_carries_the_trigger_placeholder():
    """Without it the prompt names no character and the LoRA has nothing to anchor to."""
    for p in _poses_const("_POSES"):
        assert "<TRIGGER>" in p["prompt_template"], p["name"]


def test_no_pose_hardcodes_a_character_name():
    """The bug this migration fixes.

    A pose naming k3lly2026 in its text is a pose only k3lly2026 can use, which is how new
    LoRAs got locked out.
    """
    for p in _poses_const("_POSES"):
        for character in ("k3lly2026", "k3llydw", "p@y"):
            assert character not in p["prompt_template"], f"{p['name']} hardcodes {character}"


def test_placeholder_is_braces_not_angle_brackets():
    """`<...>` is the wildcard resolver's syntax (segments.py::_resolve_wildcards).

    A trigger in angle brackets would be treated as a wildcard and randomly substituted —
    the render would silently name a different character, or nothing at all.
    """
    for p in _poses_const("_POSES"):
        assert "<trigger>" not in p["prompt_template"]


def test_render_prompt_substitutes_the_trigger():
    from app.routes.ltx_recipes import render_prompt
    assert render_prompt("<TRIGGER>, a woman walks", "k3lly2026") == "k3lly2026, a woman walks"
    # A template without the placeholder is returned unchanged rather than rejected.
    assert render_prompt("a woman walks", "k3lly2026") == "a woman walks"


def test_a_new_character_gets_every_pose_without_adding_rows():
    """The whole point. Adding a character is a row, not eight prompt copies.

    Poses are returned for every character because they are not attached to one, so this is a
    property of the schema rather than something the API has to remember to do.
    """
    import inspect
    from app.routes import ltx_recipes
    src = inspect.getsource(ltx_recipes.get_recipe_book)
    assert '"poses"' in src
    # Poses must NOT be nested inside characters — that is the shape that locked LoRAs out.
    assert "c.recipes" not in src


# --------------------------------------------------------------------------------------
# Editing a character (console#361)
# --------------------------------------------------------------------------------------
import uuid as _uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import get_current_user
from app.database import get_db
from app.main import app
from app.models import LtxCharacter, LtxRecipe, User


async def _user(db):
    u = User(id=_uuid.uuid4(), username=f"u{_uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(u)
    await db.flush()
    return u


async def _call(db, user, method, path, **kw):
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            return await getattr(c, method)(path, **kw)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestUpdateCharacter:
    async def test_a_strength_can_be_tuned_without_restating_the_lora(self, db):
        user = await _user(db)
        c = LtxCharacter(id=_uuid.uuid4(), name="k9", char_lora="k9_v2", trigger="k9",
                         strength_stage_1=0.8, strength_stage_2=1.5)
        db.add(c)
        await db.flush()

        r = await _call(db, user, "patch", f"/ltx/characters/{c.id}",
                        json={"strength_stage_2": 1.2})

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["strength_stage_2"] == 1.2
        assert body["strength_stage_1"] == 0.8      # untouched
        assert body["char_lora"] == "k9_v2"         # untouched

    async def test_a_rename_leaves_the_trigger_alone(self, db):
        """The trigger is what every pose's prompt renders with.

        On create an absent trigger defaults to the name. Carrying that rule into update
        would mean renaming a character silently rewrote the text of every pose rendered
        for it — a side effect nobody asked for, on the one field the LoRA responds to.
        """
        user = await _user(db)
        c = LtxCharacter(id=_uuid.uuid4(), name="old", char_lora="l", trigger="realtrigger",
                         strength_stage_1=0.8, strength_stage_2=1.5)
        db.add(c)
        await db.flush()

        r = await _call(db, user, "patch", f"/ltx/characters/{c.id}", json={"name": "new"})

        assert r.status_code == 200, r.text
        assert r.json()["name"] == "new"
        assert r.json()["trigger"] == "realtrigger"

    async def test_deleting_a_character_leaves_the_poses(self, db):
        """Poses are not owned by a character — that is the whole point of #212.

        The route's docstring claimed a cascade to "its recipes" long after the FK went,
        which is the kind of thing that stops someone deleting a bad character.
        """
        user = await _user(db)
        c = LtxCharacter(id=_uuid.uuid4(), name="doomed", char_lora="l", trigger="t",
                         strength_stage_1=0.8, strength_stage_2=1.5)
        pose = LtxRecipe(id=_uuid.uuid4(), name=f"pose-{_uuid.uuid4().hex[:6]}",
                         prompt_template="<TRIGGER>, standing", validated=True)
        db.add_all([c, pose])
        await db.flush()

        r = await _call(db, user, "delete", f"/ltx/characters/{c.id}")

        assert r.status_code == 204, r.text
        assert await db.get(LtxRecipe, pose.id) is not None

    async def test_an_unknown_character_is_404_not_500(self, db):
        user = await _user(db)
        r = await _call(db, user, "patch", f"/ltx/characters/{_uuid.uuid4()}",
                        json={"strength_stage_1": 1.0})
        assert r.status_code == 404
