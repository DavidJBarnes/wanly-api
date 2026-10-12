"""wanly-console#675: a character keeps a list of good working prompts, each with its date.

David: "I want many prompts, not one long running textarea prompt. Capture date added."
"""
import asyncio
import uuid

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.models import LtxCharacter
from app.routes import character_prompts as cp
from app.schemas.character_prompts import CharacterPromptBody


async def _char(db, name=None):
    c = LtxCharacter(name=name or f"Char{uuid.uuid4().hex[:6]}", trigger="t@g", gender="woman")
    db.add(c)
    await db.flush()
    return c


@pytest.mark.asyncio
class TestCharacterPrompts:
    async def test_many_prompts_newest_first_with_dates(self, db):
        c = await _char(db)
        a = await cp.add_prompt(c.name, CharacterPromptBody(text="first"), db=db)
        await asyncio.sleep(0.01)
        b = await cp.add_prompt(str(c.id), CharacterPromptBody(text="  second  "), db=db)
        rows = await cp.list_prompts(c.name.lower(), db=db)
        assert [r.text for r in rows] == ["second", "first"]
        assert a.created_at is not None and b.created_at > a.created_at

    async def test_prompts_belong_to_their_character(self, db):
        c, other = await _char(db), await _char(db)
        await cp.add_prompt(c.name, CharacterPromptBody(text="mine"), db=db)
        assert await cp.list_prompts(other.name, db=db) == []

    async def test_edit_keeps_the_date_added(self, db):
        c = await _char(db)
        p = await cp.add_prompt(c.name, CharacterPromptBody(text="old"), db=db)
        added = p.created_at
        e = await cp.edit_prompt(p.id, CharacterPromptBody(text="new"), db=db)
        assert e.text == "new" and e.created_at == added and e.updated_at is not None

    async def test_delete(self, db):
        c = await _char(db)
        p = await cp.add_prompt(c.name, CharacterPromptBody(text="gone"), db=db)
        await cp.delete_prompt(p.id, db=db)
        assert await cp.list_prompts(c.name, db=db) == []
        with pytest.raises(HTTPException) as e:
            await cp.delete_prompt(p.id, db=db)
        assert e.value.status_code == 404

    async def test_unknown_character_is_404(self, db):
        with pytest.raises(HTTPException) as e:
            await cp.add_prompt("NoSuchCharacter", CharacterPromptBody(text="x"), db=db)
        assert e.value.status_code == 404

    async def test_deleting_the_character_deletes_its_prompts(self, db):
        from sqlalchemy import text
        c = await _char(db)
        await cp.add_prompt(c.name, CharacterPromptBody(text="x"), db=db)
        await db.execute(text("DELETE FROM ltx_characters WHERE id = :i"), {"i": c.id})
        n = (await db.execute(text("SELECT count(*) FROM character_prompts WHERE character_id = :i"),
                              {"i": c.id})).scalar()
        assert n == 0


def test_a_blank_prompt_is_refused():
    with pytest.raises(ValidationError):
        CharacterPromptBody(text="   \n ")
