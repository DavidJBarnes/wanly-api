"""Characters own their dataset (wanly-api#452).

David, 2026-10-09: Characters and Datasets collapse into one thing -- a character HAS its
images (one living set), its runs (both arches) and its history (archived version sets). The
character's render LoRA is the checkpoint he STARS, never "whichever run finished last".
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import run_datasets
from app.auth import get_current_user, verify_api_key_or_bearer
from app.database import get_db
from app.models import Dataset, LtxCharacter, TrainingJob, User
from app.routes.datasets import _respond, _respond_one
from app.routes.training import checkpoint_uri, star_checkpoint
from app.schemas.characters import CharacterFull, CharacterRef
from app.schemas.datasets import DatasetRun
from app.schemas.ltx import CharacterStar, LtxCharacterResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["characters"])


async def _find(db: AsyncSession, key: str) -> LtxCharacter:
    """By id, or by name (case-insensitive) -- the console's URLs use the name."""
    row = None
    try:
        row = await db.get(LtxCharacter, uuid.UUID(key))
    except ValueError:
        row = (await db.execute(select(LtxCharacter).where(
            func.lower(LtxCharacter.name) == key.lower()))).scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no character {key!r}")
    return row


def _set_kind(c: LtxCharacter) -> str:
    return "composition" if (c.kind or "solo") == "pair" else "character"


def _ref(c: LtxCharacter) -> CharacterRef:
    return CharacterRef(id=c.id, name=c.name, kind=c.kind or "solo", hidden=bool(c.hidden))


@router.get("/ltx/characters/{key}/full", response_model=CharacterFull,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def character_full(key: str, db: AsyncSession = Depends(get_db)):
    """The character page in one call: the row, its living set, its archived sets and every
    run its page lists. One set is read in full (the living one); the archived are summaries
    of the same shape, because History shows their counts and runs, not their images."""
    c = await _find(db, key)
    sets = (await db.execute(select(Dataset).where(
        Dataset.character == c.name, Dataset.kind == _set_kind(c)))).scalars().all()
    living = next((d for d in sets if d.archived_at is None), None)
    archived = sorted((d for d in sets if d.archived_at is not None),
                      key=lambda d: d.archived_at, reverse=True)
    runs: list[dict] = []
    if living is not None:
        runs = await run_datasets.runs_for_dataset(db, living)
    else:
        # No living set (registered before its images, or deleted): the runs still belong
        # here. Gathered from the archived sets, each run once.
        seen: set[str] = set()
        for d in archived:
            for r in await run_datasets.runs_for_dataset(db, d):
                if r["job_id"] not in seen:
                    seen.add(r["job_id"])
                    runs.append(r)
    members: list[CharacterRef] = []
    if c.members:
        rows = {r.name: r for r in (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name.in_(list(c.members))))).scalars().all()}
        members = [_ref(rows[m]) for m in c.members if m in rows]
    pairs = [_ref(p) for p in (await db.execute(select(LtxCharacter).where(
        LtxCharacter.kind == "pair"))).scalars().all() if c.name in (p.members or [])]
    return CharacterFull(
        character=LtxCharacterResponse.model_validate(c),
        dataset=await _respond_one(db, living) if living is not None else None,
        archived=[_respond(d, None) for d in archived],
        runs=[DatasetRun(**r) for r in runs],
        members=members, pairs=pairs)


@router.post("/ltx/characters/{character_id}/star", response_model=LtxCharacterResponse)
async def star(
    character_id: uuid.UUID,
    body: CharacterStar,
    response: Response,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Make one LTX checkpoint the character's render LoRA.

    In the bucket: applied now (char_lora, provenance, base model; a pair's phrase). Still on
    the trainer (publish "none" is the default, #413): its upload is requested, the star is
    recorded as pending, and it applies when the file lands -- 202, and the row says so.
    """
    c = await db.get(LtxCharacter, character_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Character not found")
    job = await db.get(TrainingJob, body.training_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    if (job.config or {}).get("arch") == "sdxl":
        raise HTTPException(status_code=422, detail="an SDXL checkpoint is for A1111 — renders "
                                                    "use an LTX one, so only LTX can be starred")
    if job.character != c.name:
        raise HTTPException(status_code=422,
                            detail=f"that run trained {job.character!r}, not {c.name!r}")
    if not any(e.get("label") == body.label for e in (job.epochs or [])):
        raise HTTPException(status_code=404, detail=f"this run has no checkpoint {body.label}")
    uri = checkpoint_uri(job, body.label)
    if uri is not None:
        await star_checkpoint(db, c, job, uri)
        await db.commit()
        await db.refresh(c)
        logger.info("character %s: starred %s", c.name, uri)
        return c
    wanted = list(job.publish_requests or [])
    if body.label not in wanted:
        job.publish_requests = wanted + [body.label]
    c.star_pending = {"training_job_id": str(job.id), "label": body.label}
    await db.commit()
    await db.refresh(c)
    response.status_code = 202
    logger.info("character %s: starred %s v%s %s, upload requested", c.name, job.character,
                job.version, body.label)
    return c


@router.delete("/ltx/characters/{character_id}/star", response_model=LtxCharacterResponse)
async def cancel_pending_star(
    character_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Drop a pending star (the upload, if already asked for, still happens)."""
    c = await db.get(LtxCharacter, character_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Character not found")
    c.star_pending = None
    await db.commit()
    await db.refresh(c)
    return c
