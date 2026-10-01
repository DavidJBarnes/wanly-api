"""Build a character's sheet in the console (wanly-console#582). See app/sheet_gen.py.

    GET  /ltx/characters/sheet/presets          pre-filled outfit/hair, counts, crop padding
    POST /ltx/characters/{id}/sheet/generate    202 + a job on the image-edit queue
    GET  /ltx/characters/sheet/jobs/{job_id}    state, why it waits, candidates so far
    POST /ltx/characters/{id}/sheet/compose     {job_id, seed}: save that candidate's sheet to
                                                the repo, set it as the character's sheet
    GET  /ltx/characters/{id}/sheets            the provenance of every sheet saved for it
"""
import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import s3, sheet_gen
from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import CharacterSheet, LtxCharacter
from app.routes.image_edit import _fetch
from app.routes.ltx_recipes import _character, _settle_identity
from app.schemas.character_sheets import (
    CharacterSheetResponse, SheetComposeRequest, SheetComposeResponse,
    SheetGenerateRequest, SheetJob, SheetPresets,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["characters"])


def _http(e: sheet_gen.SheetError) -> HTTPException:
    return HTTPException(status_code=e.status_code, detail=e.detail)


def _no_pair(c: LtxCharacter) -> None:
    if (c.kind or "solo") == "pair":
        raise HTTPException(422, f"{c.name} is a pair: a pair renders with its first member's "
                                 f"sheet -- build the sheet on that character")


@router.get("/ltx/characters/sheet/presets", response_model=SheetPresets,
            dependencies=[Depends(get_current_user)])
async def sheet_presets():
    return SheetPresets(
        defaults=sheet_gen.DEFAULTS, default_count=sheet_gen.DEFAULT_COUNT,
        max_count=sheet_gen.MAX_COUNT, crop_padding=sheet_gen.CROP_PADDING,
    )


@router.post("/ltx/characters/{character_id}/sheet/generate", response_model=SheetJob,
             status_code=202, dependencies=[Depends(get_current_user)])
async def generate_sheet(character_id: uuid.UUID, body: SheetGenerateRequest,
                         db: AsyncSession = Depends(get_db)):
    """Queue N turnaround candidates (default 3) from one photo of her (console#585).
    Everything that can be refused is refused before the photo is fetched or queued."""
    c = await _character(db, character_id)
    _no_pair(c)
    if not settings.image_edit_url and not settings.image_edit_standing_url:
        raise HTTPException(503, "no image-edit service is configured (image_edit_url is empty)")
    try:
        request = sheet_gen.service_request(
            body.outfit, body.hair, sheet_gen.gender_for(c.gender, body.gender), body.subject,
            body.crop_padding)
        seeds = sheet_gen.pick_seeds(body.count, body.seeds)
    except sheet_gen.SheetError as e:
        raise _http(e) from e
    photo = await _fetch(body.photo_uri)
    job = await sheet_gen.submit(c, body.photo_uri, photo, request, seeds)
    rec, live = await sheet_gen.load(job.id)
    return JSONResponse(status_code=202,
                        content=SheetJob(**sheet_gen.view(rec, live)).model_dump(mode="json"))


@router.get("/ltx/characters/sheet/jobs/{job_id}", response_model=SheetJob,
            dependencies=[Depends(get_current_user)])
async def sheet_job(job_id: str):
    try:
        rec, live = await sheet_gen.load(job_id)
    except sheet_gen.SheetError as e:
        raise _http(e) from e
    return SheetJob(**sheet_gen.view(rec, live))


@router.post("/ltx/characters/{character_id}/sheet/compose", response_model=SheetComposeResponse,
             dependencies=[Depends(get_current_user)])
async def compose_sheet(character_id: uuid.UUID, body: SheetComposeRequest,
                        db: AsyncSession = Depends(get_db)):
    """Approve one candidate: its composed 1536x1024 sheet is copied into the Image Repo under a
    fresh name, becomes the character's sheet (identity_mode 'sheet'), and its provenance is
    recorded. Approving again (another seed, or the same one) writes another image; nothing is
    overwritten."""
    c = await _character(db, character_id)
    _no_pair(c)
    try:
        rec, live = await sheet_gen.load(body.job_id)
        cand = sheet_gen.candidate(rec, body.seed)
    except sheet_gen.SheetError as e:
        raise _http(e) from e
    if rec.get("character_id") != str(c.id):
        raise HTTPException(409, f"that job built a sheet for {rec.get('character_name')}, "
                                 f"not {c.name}")
    try:
        data = await asyncio.to_thread(s3.download_bytes, cand["sheet_uri"])
    except Exception as e:                          # noqa: BLE001
        raise HTTPException(404, f"the candidate's sheet is gone from the jobs bucket: {e}") from e
    uri = await asyncio.to_thread(s3.upload_bytes, data, sheet_gen.sheet_key(c.name, body.seed),
                                  settings.s3_images_bucket)
    req = rec.get("request") or {}
    c.sheet_uri = uri
    c.identity_mode = "sheet"
    _settle_identity(c)
    one_photo = rec.get("photo_mode") == sheet_gen.PHOTO_MODE
    row = CharacterSheet(
        character_id=c.id, character_name=c.name, sheet_uri=uri,
        candidate_uri=cand.get("candidate_uri"),
        face_uri=sheet_gen.photo_uri(rec) or "",
        outfit=req.get("outfit") or "", hair=req.get("hair"), body=req.get("body"),
        gender=req.get("gender"), prompt=cand.get("prompt") or "", seed=int(body.seed),
        model=cand.get("model"), settings=cand.get("settings"), files=cand.get("files"),
        face_panel=cand.get("face_panel"), identity=cand.get("identity"), job_id=rec["id"],
        # The face panel was auto-cropped from the same photo (console#585). A job from
        # before it (no photo_mode in its manifest) is recorded as it was made.
        photo_mode=sheet_gen.PHOTO_MODE if one_photo else None,
        face_panel_crop=cand.get("face_panel_crop") if one_photo else None,
    )
    db.add(row)
    await db.commit()
    await db.refresh(c)
    await db.refresh(row)
    await sheet_gen.record_saved(rec["id"], live, rec, {"seed": int(body.seed), "sheet_uri": uri,
                                                        "sheet_id": str(row.id)})
    logger.info("sheet for %s: seed %s of job %s -> %s (identity_mode sheet)", c.name,
                body.seed, rec["id"], uri)
    return SheetComposeResponse(character=c, sheet=row)


@router.get("/ltx/characters/{character_id}/sheets", response_model=list[CharacterSheetResponse],
            dependencies=[Depends(get_current_user)])
async def character_sheets(character_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Every sheet saved for this character, newest first, with where it came from."""
    await _character(db, character_id)
    rows = (await db.execute(
        select(CharacterSheet).where(CharacterSheet.character_id == character_id)
        .order_by(CharacterSheet.created_at.desc()))).scalars().all()
    return list(rows)
