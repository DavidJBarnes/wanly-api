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

from app.auth import get_current_user
from app.database import get_db
from app.ltx_stack import LTX_STACK
from app.models import LtxCharacter, LtxRecipe, User
from app.schemas.ltx import (
    LtxCharacterCreate,
    LtxCharacterUpdate,
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


# The placeholder a pose carries and a character's trigger fills.
#
# This shares syntax with the wildcard resolver (`<([^<>]+)>` in
# app/routes/segments.py::_resolve_wildcards), which matters: a Wildcard named TRIGGER would
# make the resolver substitute a RANDOM option here, and the render would quietly name the
# wrong character or none at all.
#
# Two things close that. The trigger is substituted BEFORE wildcard resolution runs, so a
# correctly-built prompt never reaches the resolver carrying this. And the name is reserved
# in the wildcard routes, so the shadowing wildcard cannot be created in the first place.
TRIGGER_PLACEHOLDER = "<TRIGGER>"


def render_prompt(template: str, trigger: str) -> str:
    """Fill a pose's placeholder with a character's trigger word.

    A template with no placeholder is returned unchanged rather than rejected — a pose whose
    prompt genuinely does not name the character is unusual but not wrong, and failing here
    would block a render for a stylistic choice.
    """
    return template.replace(TRIGGER_PLACEHOLDER, trigger)


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
    chars = (await db.execute(
        select(LtxCharacter).order_by(LtxCharacter.name)
    )).scalars().all()
    poses = (await db.execute(
        select(LtxRecipe).order_by(LtxRecipe.name)
    )).scalars().all()

    # Poses are character-agnostic, so EVERY pose is offered for EVERY character. That is the
    # point: a new LoRA is never locked out for want of rows in a table — add the character
    # and all of them work immediately.
    return {
        "stack": LTX_STACK,
        "poses": [
            {
                "id": str(r.id),
                "name": r.name,
                "prompt_template": r.prompt_template,
                "negative_prompt": r.negative_prompt or LTX_STACK["negative"],
                "frames": r.frames or LTX_STACK["frames"],
                # `or` is wrong here and `is None` is right: img_compression 0 is a REAL
                # setting -- it bypasses the conditioning-frame encode entirely -- and `or`
                # would silently replace it with the stack default.
                "img_compression": (r.img_compression if r.img_compression is not None
                                    else LTX_STACK["img_compression"]),
                "validated": r.validated,
            }
            for r in poses
        ],
        "characters": [
            {
                "id": str(c.id),
                "name": c.name,
                "char_lora": c.char_lora,
                "trigger": c.trigger,
                "strength_stage_1": c.strength_stage_1,
                "strength_stage_2": c.strength_stage_2,
            }
            for c in chars
        ],
    }


@router.post("/ltx/characters", response_model=LtxCharacterResponse, status_code=201)
async def create_character(
    body: LtxCharacterCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    data = body.model_dump()
    # A character without a trigger renders a prompt containing a literal "<TRIGGER>", which
    # is worse than any default. The name is what all three seeded characters use.
    data["trigger"] = data.get("trigger") or data["name"]
    c = LtxCharacter(**data)
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


@router.patch("/ltx/characters/{character_id}", response_model=LtxCharacterResponse)
async def update_character(
    character_id: uuid.UUID,
    body: LtxCharacterUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Edit a character.

    Without this the only way to correct a strength or a trigger typo was delete and
    recreate, which is a worse trade than it looks: the character's id changes, and a
    per-stage strength is exactly the field someone tunes repeatedly.
    """
    c = await _character(db, character_id)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(c, k, v)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="A character with that name already exists")
    await db.refresh(c)
    return c


@router.delete("/ltx/characters/{character_id}", status_code=204)
async def delete_character(
    character_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deletes the character. Poses are NOT touched.

    The docstring here used to claim it cascaded to "its recipes". That was true of the
    old per-character shape and false since #212: `ltx_recipes` has no character_id and
    no relationship to this table, because a pose belongs to every character. Deleting a
    character removes one LoRA + trigger pairing and nothing else.

    Renders already produced are untouched either way: a segment records what it ran in
    its own ltx_recipe blob, so history does not depend on the recipe still existing.
    """
    await db.delete(await _character(db, character_id))
    await db.commit()


@router.post("/ltx/recipes", response_model=LtxRecipeResponse, status_code=201)
async def create_recipe(
    body: LtxRecipeCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a pose. Not attached to a character — every character gets it."""
    r = LtxRecipe(**body.model_dump())
    db.add(r)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"A pose named {body.name!r} already exists",
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
