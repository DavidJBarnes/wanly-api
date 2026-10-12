"""A character's list of good working prompts (wanly-console#675).

Many short entries, each with the date it was added -- not one long textarea. The list is the
character page's Prompts tab: add, copy, edit, delete.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import CharacterPrompt
from app.routes.characters import _find
from app.schemas.character_prompts import CharacterPromptBody, CharacterPromptResponse

router = APIRouter(tags=["characters"])


@router.get("/ltx/characters/{key}/prompts", response_model=list[CharacterPromptResponse],
            dependencies=[Depends(get_current_user)])
async def list_prompts(key: str, db: AsyncSession = Depends(get_db)):
    """Newest first. `key` is the id or the name, as everywhere on the character page."""
    c = await _find(db, key)
    return (await db.execute(select(CharacterPrompt)
                             .where(CharacterPrompt.character_id == c.id)
                             .order_by(CharacterPrompt.created_at.desc()))).scalars().all()


@router.post("/ltx/characters/{key}/prompts", response_model=CharacterPromptResponse,
             status_code=201, dependencies=[Depends(get_current_user)])
async def add_prompt(key: str, body: CharacterPromptBody, db: AsyncSession = Depends(get_db)):
    c = await _find(db, key)
    p = CharacterPrompt(character_id=c.id, text=body.text,
                        created_at=datetime.now(timezone.utc))
    db.add(p)
    await db.commit()
    await db.refresh(p)
    return p


async def _prompt(db: AsyncSession, prompt_id: uuid.UUID) -> CharacterPrompt:
    p = await db.get(CharacterPrompt, prompt_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return p


@router.patch("/ltx/character-prompts/{prompt_id}", response_model=CharacterPromptResponse,
              dependencies=[Depends(get_current_user)])
async def edit_prompt(prompt_id: uuid.UUID, body: CharacterPromptBody,
                      db: AsyncSession = Depends(get_db)):
    """The text only. The date added stays: it says when the prompt joined the list."""
    p = await _prompt(db, prompt_id)
    if p.text != body.text:
        p.text = body.text
        p.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(p)
    return p


@router.delete("/ltx/character-prompts/{prompt_id}", status_code=204,
               dependencies=[Depends(get_current_user)])
async def delete_prompt(prompt_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    p = await _prompt(db, prompt_id)
    await db.delete(p)
    await db.commit()
    return Response(status_code=204)
