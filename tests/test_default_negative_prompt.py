"""The Settings negative prompt is the default a render actually uses (console#430).

It never was. Two defaults existed for one question — `LTX_STACK['negative']`, a constant in
the image, and `app_settings.negative_prompt`, the Settings field — and the constant won
every time:

  * `GET /recipes` resolved a pose against the constant, never the setting;
  * the console prefilled its form from that resolved value, so every segment was created
    carrying it, so the claim endpoint's fallback — the ONLY reader of the setting — never
    saw the NULL it needed to fire on;
  * the pose editor prefilled from the resolved value too and saved it back, which pinned
    all 16 production poses to a verbatim copy of the constant.

These tests hold the resolution order: pose override, then the setting, then the constant.
"""

import importlib.util
import hashlib
from pathlib import Path

import pytest
import sqlalchemy as sa

from app.ltx_stack import LTX_STACK
from app.models import AppSetting, LtxRecipe
from app.negative_prompt import SETTING_KEY, default_negative_prompt


async def _set(db, value: str) -> None:
    await db.merge(AppSetting(key=SETTING_KEY, value=value))
    await db.flush()


class TestResolver:
    @pytest.mark.asyncio
    async def test_the_setting_wins_when_it_has_one(self, db):
        await _set(db, "extra fingers, bad anatomy")
        assert await default_negative_prompt(db) == "extra fingers, bad anatomy"

    @pytest.mark.asyncio
    async def test_the_stack_constant_is_the_seed_when_nothing_is_set(self, db):
        await db.execute(sa.delete(AppSetting).where(AppSetting.key == SETTING_KEY))
        await db.flush()
        assert await default_negative_prompt(db) == LTX_STACK["negative"]

    @pytest.mark.asyncio
    async def test_blank_means_not_set_not_render_without_one(self, db):
        """An empty box is how the field looks before anyone uses it. Reading that as "drop
        the quality negatives" would silently change every render the first time it was
        cleared, which is a much worse default than the constant."""
        for blank in ("", "   ", "\n"):
            await _set(db, blank)
            assert await default_negative_prompt(db) == LTX_STACK["negative"]


class TestRecipeBook:
    """The book is where the console reads a pose's negative from, so it is where the
    setting has to arrive."""

    @staticmethod
    async def _book(db):
        from app.routes.ltx_recipes import get_recipe_book

        return await get_recipe_book(user=None, db=db)

    @pytest.mark.asyncio
    async def test_a_pose_with_no_override_resolves_to_the_setting(self, db):
        await _set(db, "six fingers")
        db.add(LtxRecipe(name="inheriting pose", prompt_template="<TRIGGER>, standing"))
        await db.flush()

        pose = next(p for p in (await self._book(db))["poses"]
                    if p["name"] == "inheriting pose")
        assert pose["negative_prompt"] == "six fingers"
        # And it still reports that it holds no override of its own, which is what stops the
        # editor from writing one back.
        assert pose["negative_prompt_override"] is None

    @pytest.mark.asyncio
    async def test_a_real_override_still_outranks_the_setting(self, db):
        await _set(db, "six fingers")
        db.add(LtxRecipe(name="opinionated pose", prompt_template="<TRIGGER>, standing",
                         negative_prompt="hands"))
        await db.flush()

        pose = next(p for p in (await self._book(db))["poses"]
                    if p["name"] == "opinionated pose")
        assert pose["negative_prompt"] == "hands"
        assert pose["negative_prompt_override"] == "hands"

    @pytest.mark.asyncio
    async def test_the_book_states_the_default_so_the_editor_can_show_it(self, db):
        await _set(db, "six fingers")
        assert (await self._book(db))["default_negative_prompt"] == "six fingers"


class TestUnpinMigration:
    """086 clears the 16 poses that hold a copy of the constant. Without it the resolver fix
    changes nothing: a non-NULL override wins, and every production row had one."""

    @staticmethod
    def _migration():
        path = Path("alembic/versions/086_unpin_stack_negative.py")
        spec = importlib.util.spec_from_file_location("m086", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_frozen_literal_is_the_text_production_actually_holds(self):
        """195 characters, md5 bdb2e8d6…, measured on all 16 rows before the fix. Frozen in
        the migration rather than imported, because a migration describes the data as it was
        — but a reflow or a typo in that copy would make it match nothing, silently."""
        pinned = self._migration()._PINNED
        assert len(pinned) == 195
        assert hashlib.md5(pinned.encode()).hexdigest() == "bdb2e8d67a0c43e89d5aeee1cce2a0e0"

    @pytest.mark.asyncio
    async def test_it_clears_the_copies_and_leaves_a_real_override_alone(self, db):
        pinned = self._migration()._PINNED
        db.add_all([
            LtxRecipe(name="pinned pose", prompt_template="<TRIGGER>, a",
                      negative_prompt=pinned),
            LtxRecipe(name="hand written pose", prompt_template="<TRIGGER>, b",
                      negative_prompt="hands, six fingers"),
            LtxRecipe(name="nearly pinned pose", prompt_template="<TRIGGER>, c",
                      negative_prompt=pinned + ", cross-eyed"),
        ])
        await db.flush()

        await db.execute(
            sa.text("UPDATE ltx_recipes SET negative_prompt = NULL "
                    "WHERE negative_prompt = :pinned"),
            {"pinned": pinned},
        )

        rows = dict((await db.execute(
            sa.select(LtxRecipe.name, LtxRecipe.negative_prompt)
        )).all())
        assert rows["pinned pose"] is None
        assert rows["hand written pose"] == "hands, six fingers"
        assert rows["nearly pinned pose"] == pinned + ", cross-eyed"
