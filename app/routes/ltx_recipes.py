"""LTX characters and recipes: CRUD, plus the assembled book the Storyboard page reads.

Recipes are DATA. The POC authored them in an .ods — a test harness that became load-bearing,
complete with guards built around it ("never regenerate the sheet") rather than replacing it.
They are rows now.

The schema encodes what was measured rather than what was assumed. Across all 24 seeded
recipes only `char_lora` and `prompt` varied; everything else had exactly one value and lives
once in the global stack. Storing a global value 24 times is how it silently stops being
global — one row gets edited, nothing complains, and two recipes that should be identical
are not.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.ltx_stack import LTX_STACK
from app.models import LtxCharacter, LtxRecipe, User
from app.schemas.ltx import (
    LtxCharacterCreate,
    LtxCharacterResponse,
    LtxRecipeCreate,
    LtxRecipeResponse,
    LtxRecipeUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def _character(db: AsyncSession, character_id: uuid.UUID) -> LtxCharacter:
    c = await db.get(LtxCharacter, character_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return c


@router.get("/recipes")
async def get_recipe_book(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Everything the Storyboard page needs, in one call.

    Assembled rather than stored: characters, their recipes, and the one global stack. The
    stack is returned alongside so the page can SHOW what a recipe pins without each recipe
    carrying a copy of it.
    """
    rows = (
        await db.execute(
            select(LtxCharacter).options(selectinload(LtxCharacter.recipes))
            .order_by(LtxCharacter.name)
        )
    ).scalars().all()

    return {
        "stack": LTX_STACK,
        "characters": [
            {
                "id": str(c.id),
                "name": c.name,
                "char_lora": c.char_lora,
                "strength_stage_1": c.strength_stage_1,
                "strength_stage_2": c.strength_stage_2,
                "recipes": [
                    {
                        "id": str(r.id),
                        "name": r.name,
                        "prompt": r.prompt,
                        "negative_prompt": r.negative_prompt or LTX_STACK["negative"],
                        "frames": r.frames or LTX_STACK["frames"],
                        "validated": r.validated,
                    }
                    for r in c.recipes
                ],
            }
            for c in rows
        ],
    }


@router.post("/ltx/characters", response_model=LtxCharacterResponse, status_code=201)
async def create_character(
    body: LtxCharacterCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    c = LtxCharacter(**body.model_dump())
    db.add(c)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail=f"Character {body.name!r} already exists")
    await db.refresh(c)
    return c


@router.get("/ltx/characters", response_model=list[LtxCharacterResponse])
async def list_characters(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(select(LtxCharacter).order_by(LtxCharacter.name))).scalars().all()
    return list(rows)


@router.delete("/ltx/characters/{character_id}", status_code=204)
async def delete_character(
    character_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes the character and its recipes (FK cascade).

    Renders already produced are untouched: a segment records what it ran in its own
    ltx_recipe blob, so history does not depend on the recipe still existing.
    """
    await db.delete(await _character(db, character_id))
    await db.commit()


@router.post("/ltx/characters/{character_id}/recipes",
             response_model=LtxRecipeResponse, status_code=201)
async def create_recipe(
    character_id: uuid.UUID,
    body: LtxRecipeCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _character(db, character_id)
    r = LtxRecipe(character_id=character_id, **body.model_dump())
    db.add(r)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"Recipe {body.name!r} already exists for this character",
        )
    await db.refresh(r)
    return r


@router.patch("/ltx/recipes/{recipe_id}", response_model=LtxRecipeResponse)
async def update_recipe(
    recipe_id: uuid.UUID,
    body: LtxRecipeUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Edit a recipe.

    Editing the prompt is editing the recipe — there is no separate "is this still the
    validated one" state to keep in sync, because `validated` is a field the author sets
    when they have watched it. Changing the prompt does NOT silently clear it: that would
    be the system overruling a human's judgement about their own edit.
    """
    r = await db.get(LtxRecipe, recipe_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(r, k, v)
    r.updated_at = datetime.now(timezone.utc)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="A recipe with that name already exists")
    await db.refresh(r)
    return r


@router.delete("/ltx/recipes/{recipe_id}", status_code=204)
async def delete_recipe(
    recipe_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    r = await db.get(LtxRecipe, recipe_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    await db.delete(r)
    await db.commit()
