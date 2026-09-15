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
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.s3 import list_bucket
from app.auth import get_current_user, verify_api_key_or_bearer
from app.database import get_db
from app.checkpoint_sources import CHECKPOINT_SOURCES
from app.ltx_stack import LTX_STACK
from app.models import LtxBook, LtxCharacter, LtxRecipe, User, Worker
from app.negative_prompt import default_negative_prompt
from app.schemas.ltx import (
    LtxBookCreate,
    LtxBookResponse,
    LtxBookUpdate,
    LtxCharacterCreate,
    LtxCharacterUpdate,
    LtxCharacterResponse,
    LtxRecipeCreate,
    LtxRecipeResponse,
    LtxRecipeUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# The book a new pose lands in when the caller names none. Must match the migration's default
# (100_books) and the stack's checkpoint family: "10eros" holds the 10Eros poses, including
# the NULL-checkpoint ones. A pose is NEVER refused for lack of a book — that is the whole
# reason book_id is server-defaulted rather than required from the client.
DEFAULT_BOOK_NAME = "10eros"


async def _default_book(db: AsyncSession) -> LtxBook:
    """The book a new pose is filed into when none is given.

    Falls back to the first book by name if the default is somehow absent, so a create can
    still succeed rather than 500 on a database whose books were rearranged.
    """
    book = (await db.execute(
        select(LtxBook).where(LtxBook.name == DEFAULT_BOOK_NAME)
    )).scalar_one_or_none()
    if book is None:
        book = (await db.execute(
            select(LtxBook).order_by(LtxBook.name).limit(1)
        )).scalar_one_or_none()
    if book is None:
        raise HTTPException(
            status_code=409,
            detail="No books exist; create a book before creating a pose",
        )
    return book


async def _book(db: AsyncSession, book_id: uuid.UUID) -> LtxBook:
    b = await db.get(LtxBook, book_id)
    if b is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return b


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
#
# The placeholder lives in app/recipe_blob.py with the one reader of the blob's people;
# re-exported here because this is where it was.
from app.recipe_blob import (  # noqa: E402
    TRIGGER_PLACEHOLDER, TRIGGER_PLACEHOLDERS, render_prompt,
)

__all__ = ["router", "TRIGGER_PLACEHOLDER", "TRIGGER_PLACEHOLDERS",
           "render_prompt"]


@router.get("/recipes")
async def get_recipe_book(
    book_id: Annotated[uuid.UUID | None, Query(description="Limit poses to one book")] = None,
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

    # The Settings field, not LTX_STACK['negative']. Resolving against the constant is what
    # made the setting unreachable: the console prefills its form from the resolved value,
    # so every segment was created carrying the constant and the claim-time fallback to the
    # setting never had a NULL to fire on (console#430).
    default_negative = await default_negative_prompt(db)

    # Books, with a pose count each. The count is a query, not a column: adding a pose must
    # not require remembering to increment a number, and the console wants it to grey out a
    # delete before the API's 409 answers.
    counts = dict((await db.execute(
        select(LtxRecipe.book_id, func.count(LtxRecipe.id)).group_by(LtxRecipe.book_id)
    )).all())

    # Ordered by book then pose, so the console can render grouped headings straight from the
    # list without sorting, and the same order holds whether or not it filters to one book.
    poses_q = select(LtxRecipe).join(LtxBook).order_by(LtxBook.name, LtxRecipe.name)
    if book_id is not None:
        # An unknown book is an empty list, not a 404: the caller asked for a filter, and a
        # filter that matches nothing is not an error. Somebody deleting the book while the
        # console holds the id should show an empty page, not a failed request.
        poses_q = poses_q.where(LtxRecipe.book_id == book_id)
    poses = (await db.execute(poses_q)).scalars().all()

    books = (await db.execute(select(LtxBook).order_by(LtxBook.name))).scalars().all()

    # Poses are character-agnostic, so EVERY pose is offered for EVERY character. That is the
    # point: a new LoRA is never locked out for want of rows in a table — add the character
    # and all of them work immediately.
    return {
        "stack": LTX_STACK,
        # What an un-overridden pose resolves to. Returned alongside the stack rather than
        # folded into it: `stack` is the constant configuration the image ships with, and
        # this one is a setting that can differ from it.
        "default_negative_prompt": default_negative,
        # The shelves, so the console's picker can group without a second call.
        "books": [
            {
                "id": str(b.id),
                "name": b.name,
                "description": b.description,
                "recipe_count": counts.get(b.id, 0),
            }
            for b in books
        ],
        "poses": [
            {
                "id": str(r.id),
                "name": r.name,
                "prompt_template": r.prompt_template,
                "negative_prompt": r.negative_prompt or default_negative,
                # The RAW override, so the editor can tell "inherits the default" from
                # "overridden". Without it the editor could only show the resolved value,
                # and saving an untouched pose wrote that back as an override -- which is
                # how all 16 poses ended up pinned to a copy of the stack constant.
                "negative_prompt_override": r.negative_prompt,
                "frames": r.frames or LTX_STACK["frames"],
                # `or` is wrong here and `is None` is right: img_compression 0 is a REAL
                # setting -- it bypasses the conditioning-frame encode entirely -- and `or`
                # would silently replace it with the stack default.
                "img_compression": (r.img_compression if r.img_compression is not None
                                    else LTX_STACK["img_compression"]),
                # In application order, and returned as stored. No stack fallback: the
                # stack's content_lora is "none", and an empty list already says that more
                # directly than a one-element list naming "none" would.
                #
                # Strengths are whatever was stored, including 0 — which is a REAL setting:
                # it loads the LoRA and gives it no weight, which is how you measure what it
                # contributes. Nothing here may silently promote a 0 to 0.6.
                "content_loras": r.content_loras or [],
                # `or` is right here: NULL and "" both mean "not set", and there is no
                # falsy checkpoint name that means something different.
                "checkpoint": r.checkpoint or LTX_STACK["checkpoint"],
                "book_id": str(r.book_id),
                "book_name": r.book_name,
            }
            for r in poses
        ],
        "characters": [
            {
                "id": str(c.id),
                "name": c.name,
                "char_lora": c.char_lora,
                "trigger": c.trigger,
                "gender": c.gender,
                "strength_stage_1": c.strength_stage_1,
                "strength_stage_2": c.strength_stage_2,
                #: Which datasets trained this LoRA (migration 099), group order.
                "trained_from": c.trained_from,
            }
            for c in chars
        ],
    }


@router.get("/ltx/checkpoints", dependencies=[Depends(verify_api_key_or_bearer)])
async def list_checkpoints(db: AsyncSession = Depends(get_db)):
    """Base models a pose can actually be rendered on (console#404).

    The union of what live workers report, not a list held here. A checkpoint is a 46 GB
    file on a GPU box; whether one is loadable is a fact about that box, and the engine
    binds to 127.0.0.1 inside its container so nothing upstream can ask directly. Workers
    report it through the heartbeat instead.

    OFFLINE WORKERS ARE EXCLUDED. Offering a checkpoint that only exists on a box which is
    not running means picking it produces a job nothing can claim — a queue that silently
    stops rather than an error. What is offered should be what can be rendered now.

    The stack's default is always included even when no worker is up, so the dropdown is
    never empty and always contains the value every existing pose already uses.
    """
    rows = (await db.execute(
        select(Worker.checkpoints).where(Worker.status != "offline")
    )).scalars().all()

    names: set[str] = {LTX_STACK["checkpoint"]}
    for row in rows:
        for name in row or []:
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return {"checkpoints": sorted(names), "default": LTX_STACK["checkpoint"]}


@router.get("/ltx/checkpoints/catalog", dependencies=[Depends(verify_api_key_or_bearer)])
async def checkpoint_catalog():
    """Where each known checkpoint can be fetched from (console#423).

    Asked by a WORKER, not the console: a worker handed a pose whose base model it does not
    hold downloads it rather than failing the claim. It must not guess a URL, so the mapping
    from name to {repo, path} lives here -- one entry adds a base model for the whole fleet,
    against redeploying every worker to teach it a new name.

    Separate from /ltx/checkpoints, which answers "what can be rendered NOW" from live worker
    inventory. This answers "where does one come from", which is a fact about the file and
    true whether any worker is up.

    `size_bytes` is the point of the response as much as the URL. A partial safetensors is a
    valid header over missing data -- it passes every existence check and fails only at load,
    inside a claimed segment -- so the fetcher needs the expected length to verify against
    before it renames the file into place.
    """
    return {"sources": CHECKPOINT_SOURCES}


@router.get("/loras", dependencies=[Depends(verify_api_key_or_bearer)])
async def list_available_loras():
    """Every LoRA in the bucket — the worker's sync list AND the console's dropdowns.

    `kind` comes from the key prefix: `character/` are identity LoRAs (which character this
    is), `content/` are motion/act LoRAs (what is happening), and they are chosen in
    different places in the console — a character row versus a recipe. A LoRA sitting at the
    root reports "unfiled" rather than being hidden, because a file nobody can see is worse
    than one that is merely mis-shelved.

    Two callers, deliberately one endpoint: what the console offers to pick from is exactly
    what a worker can actually obtain. A dropdown listing a LoRA no worker can fetch is a
    job that fails ten minutes into a claimed segment.

    Hence verify_api_key_or_bearer and not verify_api_key_or_token: the worker authenticates
    with X-API-Key, the console with a Bearer JWT. The _or_token variant accepts a ?token=
    query param for <img src> media loads and would 401 the console outright.

    Exists because workers deliberately carry NO AWS credentials — a rented pod should not
    hold them — and so they cannot list the bucket themselves. They already download through
    GET /files, which 307s to a presigned URL; this is the missing half that tells them what
    is there.

    `etag` is the md5 of the content for these objects and is what a worker should compare
    against, NOT the name: a retrained LoRA republished under the same name would otherwise
    never be picked up, and the worker would render old weights while the console showed the
    new character. Size is not sufficient either — two of the current three are byte-identical
    in size and completely different in content.

    `multipart: true` means the etag is not an md5 of the whole object and cannot be compared;
    a worker should fall back to size and say so rather than re-downloading forever.
    """
    objs = await asyncio.to_thread(list_bucket, settings.s3_loras_bucket)
    out = []
    for o in objs:
        key = o["name"]
        if not key.endswith(".safetensors"):
            continue
        prefix, _, base = key.rpartition("/")
        out.append({
            # `name` is the BASENAME, deliberately. It is what a ComfyUI LoraLoader takes and
            # what ltx_characters.char_lora stores, and neither should have to learn about
            # how the bucket is organised. The prefix is filing, not identity.
            "name": base,
            "kind": prefix or "unfiled",
            "key": key,
            "size": o["size"],
            "etag": o["etag"],
            "multipart": o["multipart"],
            "uri": f"s3://{settings.s3_loras_bucket}/{key}",
        })
    return out


@router.post("/ltx/books", response_model=LtxBookResponse, status_code=201)
async def create_book(
    body: LtxBookCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create an empty book. Poses are filed into it later, or never — an empty book is fine."""
    b = LtxBook(**body.model_dump())
    db.add(b)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail=f"A book named {body.name!r} already exists")
    await db.refresh(b)
    return b


@router.get("/ltx/books", response_model=list[LtxBookResponse])
async def list_books(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Every book with its pose count, for the console's dropdown and management list."""
    counts = dict((await db.execute(
        select(LtxRecipe.book_id, func.count(LtxRecipe.id)).group_by(LtxRecipe.book_id)
    )).all())
    books = (await db.execute(select(LtxBook).order_by(LtxBook.name))).scalars().all()
    out = []
    for b in books:
        dto = LtxBookResponse.model_validate(b)
        dto.recipe_count = counts.get(b.id, 0)
        out.append(dto)
    return out


@router.patch("/ltx/books/{book_id}", response_model=LtxBookResponse)
async def update_book(
    book_id: uuid.UUID,
    body: LtxBookUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename or re-describe a book. Never touches its poses."""
    b = await _book(db, book_id)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(b, k, v)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="A book with that name already exists")
    await db.refresh(b)
    return b


@router.delete("/ltx/books/{book_id}", status_code=204)
async def delete_book(
    book_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an empty book. A non-empty one is refused.

    The refusal is explicit rather than a cascade, because deleting a book is the one action
    here that could silently destroy poses someone wrote. The FK is ON DELETE RESTRICT as a
    second line of defence -- the count check answers the common case with a message the
    console can show, and the constraint catches the race where a pose is added between the
    two.
    """
    b = await _book(db, book_id)
    n = (await db.execute(
        select(func.count(LtxRecipe.id)).where(LtxRecipe.book_id == b.id)
    )).scalar_one()
    if n:
        raise HTTPException(
            status_code=409,
            detail=f"Book {b.name!r} still holds {n} pose(s); move or delete them first",
        )
    await db.delete(b)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Book gained a pose while it was being deleted; try again",
        )


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
    data = body.model_dump()
    if data.get("book_id") is None:
        # The whole point of defaulting here rather than requiring it: a pose created by any
        # caller that has not been taught about books still lands somewhere sensible. The
        # column is NOT NULL with no server default, so this is the only place the default
        # is applied -- any other construction path (tests, scripts) must supply one.
        data["book_id"] = (await _default_book(db)).id
    else:
        await _book(db, data["book_id"])
    r = LtxRecipe(**data)
    db.add(r)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"A pose named {body.name!r} already exists in that book",
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

    Editing the prompt is editing the recipe. There is no separate provenance or quality
    state to keep in sync — a pose is just its prompt and its settings.
    """
    r = await db.get(LtxRecipe, recipe_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    data = body.model_dump(exclude_unset=True)
    if data.get("book_id") is not None:
        await _book(db, data["book_id"])
    for k, v in data.items():
        setattr(r, k, v)
    r.updated_at = datetime.now(timezone.utc)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail="A recipe with that name already exists in that book"
        )
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
