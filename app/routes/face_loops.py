"""Character face loops (wanly-api#495, wanly-console#742).

A face loop is made in two steps:
1. An ordinary LTX job: Prompt recipe, the character's own LoRA and strengths, a calm preset
   prompt, from a close-up photo. It goes through the normal queue like any render.
2. When that render's segment uploads (files.upload_segment_output calls `on_render_uploaded`),
   a background task prepares the loop on this box (app/face_loop_prep.py, CPU) and uploads it
   under s3://<jobs bucket>/ar-face/<character>/<loop id>/.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.config import settings
from app.database import async_session, get_db
from app.enums import SegmentStatus
from app.face_loop_presets import FPS, FRAMES, PRESETS, build_prompt
from app.models import Dataset, FaceLoop, Job, LtxCharacter, LtxRecipe, Segment, User
from app.regularization import REG_PRIORITY_BASE
from app.routes.characters import _find
from app.s3 import delete_prefix, download_bytes, download_file, head_object, upload_bytes, upload_file
from app.schemas.face_loops import FaceLoopCreate, FaceLoopPhoto, FaceLoopPreset, FaceLoopResponse, FaceLoopUpdate

logger = logging.getLogger(__name__)
router = APIRouter(tags=["face-loops"])

PROMPT_RECIPE = "Prompt"
# One preparation at a time: the box is a t3.small, and each run peaks around 300 MB.
_prep_lock = asyncio.Lock()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "character"


async def _response(db: AsyncSession, loops: list[FaceLoop]) -> list[FaceLoopResponse]:
    if not loops:
        return []
    names = dict((await db.execute(select(LtxCharacter.id, LtxCharacter.name).where(
        LtxCharacter.id.in_({x.character_id for x in loops})))).all())
    jobs = [x.job_id for x in loops if x.job_id]
    renders = dict((await db.execute(select(Segment.job_id, Segment.status).where(
        Segment.job_id.in_(jobs), Segment.index == 0))).all()) if jobs else {}
    out = []
    for x in loops:
        r = FaceLoopResponse.model_validate(x)
        r.character = names.get(x.character_id)
        r.render_status = renders.get(x.job_id)
        out.append(r)
    return out


@router.get("/face-loops/presets", response_model=list[FaceLoopPreset], dependencies=[Depends(get_current_user)])
async def list_presets():
    return [FaceLoopPreset(id=k, label=v["label"], motion=v["motion"]) for k, v in PRESETS.items()]


@router.get("/ltx/characters/{key}/face-loops", response_model=list[FaceLoopResponse],
            dependencies=[Depends(get_current_user)])
async def list_character_loops(key: str, db: AsyncSession = Depends(get_db)):
    """Newest first."""
    c = await _find(db, key)
    rows = (await db.execute(select(FaceLoop).where(FaceLoop.character_id == c.id)
                             .order_by(FaceLoop.created_at.desc()))).scalars().all()
    return await _response(db, list(rows))


@router.get("/face-loops", response_model=list[FaceLoopResponse], dependencies=[Depends(get_current_user)])
async def list_loops(status: str | None = "ready", db: AsyncSession = Depends(get_db)):
    """Every character's loops (the AR page's picker). `status=` (empty) lists all states."""
    q = select(FaceLoop).order_by(FaceLoop.created_at.desc())
    if status:
        q = q.where(FaceLoop.status == status)
    return await _response(db, list((await db.execute(q)).scalars().all()))


# uri -> rating (None = no face). Photos do not change under a URI, so ratings are kept for the
# life of the process; the first open of a character's picker pays for them, later ones are free.
_photo_ratings: dict[str, dict | None] = {}
_PHOTO_CACHE_MAX = 5000
_rate_lock = asyncio.Lock()


async def candidate_photos(db: AsyncSession, c: LtxCharacter) -> list[str]:
    """Every image in the character's sets (living + archived) and in the datasets its LoRA was
    trained from — the training photos are often the sharpest close-ups."""
    sets = (await db.execute(select(Dataset).where(Dataset.character == c.name))).scalars().all()
    trained = [t.get("dataset_id") for t in (c.trained_from or []) if isinstance(t, dict) and t.get("dataset_id")]
    ids = []
    for t in trained:
        try:
            ids.append(uuid.UUID(str(t)))
        except ValueError:
            pass
    if ids:
        sets += (await db.execute(select(Dataset).where(Dataset.id.in_(ids)))).scalars().all()
    seen, out = set(), []
    for d in sets:
        for uri in d.images or []:
            if uri not in seen:
                seen.add(uri)
                out.append(uri)
    return out


def _rate_all(uris: list[str]) -> None:
    from app.face_loop_prep import photo_detector, rate_photo

    todo = [u for u in uris if u not in _photo_ratings]
    if not todo:
        return
    det = photo_detector()
    try:
        for u in todo:
            try:
                _photo_ratings[u] = rate_photo(download_bytes(u), det)
            except Exception:  # noqa: BLE001 - one unreadable image must not sink the list
                logger.warning("face loop photo rating failed for %s", u, exc_info=True)
                _photo_ratings[u] = None
    finally:
        det.close()
    while len(_photo_ratings) > _PHOTO_CACHE_MAX:
        _photo_ratings.pop(next(iter(_photo_ratings)))


@router.get("/ltx/characters/{key}/face-loop-photos", response_model=list[FaceLoopPhoto],
            dependencies=[Depends(get_current_user)])
async def rank_photos(key: str, db: AsyncSession = Depends(get_db)):
    """The character's photos, best start photo first. Photos with no face are left out."""
    c = await _find(db, key)
    uris = await candidate_photos(db, c)
    async with _rate_lock:
        await asyncio.to_thread(_rate_all, uris)
    rated = [FaceLoopPhoto(uri=u, **_photo_ratings[u]) for u in uris if _photo_ratings.get(u)]
    return sorted(rated, key=lambda p: p.score, reverse=True)


async def _cropped_start(uri: str, character: str) -> tuple[str, int, int]:
    """Crop the chosen photo to a face-centred square (face_loop_prep.crop_start) and store it,
    so every loop renders with the face filling the frame. Returns (uri, width, height)."""
    import hashlib

    from app.face_loop_prep import crop_start, photo_detector

    if await asyncio.to_thread(head_object, uri) is None:
        raise HTTPException(status_code=400, detail=f"start image not found: {uri}")
    data = await asyncio.to_thread(download_bytes, uri)

    def work():
        det = photo_detector()
        try:
            return crop_start(data, det)
        finally:
            det.close()

    try:
        png, w, h = await asyncio.to_thread(work)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    key = f"ar-face/{_slug(character)}/starts/{hashlib.sha256(png).hexdigest()[:16]}.png"
    return await asyncio.to_thread(upload_bytes, png, key, settings.s3_jobs_bucket), w, h


async def refresh_start_for_retry(db: AsyncSession, job: Job | None) -> None:
    """Before a face-loop render is retried: jobs made before #500 can carry a start crop that
    is not 64-divisible (831x831), which LTX's two-stage graph refuses on every retry. Re-crop
    the chosen photo (now always START_SIZE²) onto the job. Other jobs are left alone."""
    if job is None or (job.width % 64 == 0 and job.height % 64 == 0):
        return
    x = (await db.execute(select(FaceLoop).where(FaceLoop.job_id == job.id))).scalars().first()
    if x is None:
        return
    name = (await db.execute(select(LtxCharacter.name).where(LtxCharacter.id == x.character_id))).scalar_one()
    job.starting_image, job.width, job.height = await _cropped_start(x.start_image, name)


@router.post("/ltx/characters/{key}/face-loops", response_model=FaceLoopResponse, status_code=201)
async def create_loop(key: str, body: FaceLoopCreate, user: User = Depends(get_current_user),
                      db: AsyncSession = Depends(get_db)):
    c = await _find(db, key)
    if c.kind != "solo" or not c.char_lora:
        raise HTTPException(status_code=400, detail="face loops need a solo character with a LoRA")
    recipe = (await db.execute(select(LtxRecipe).where(LtxRecipe.name == PROMPT_RECIPE)
                               .order_by(LtxRecipe.is_default.desc()))).scalars().first()
    if recipe is None:
        raise HTTPException(status_code=409, detail=f"no {PROMPT_RECIPE!r} recipe to render with")
    start_uri, width, height = await _cropped_start(body.start_image, c.name)

    preset = PRESETS[body.preset]
    motion = body.motion.strip() if body.preset == "custom" else preset["motion"]
    prompt = build_prompt(c.trigger, c.gender, motion)
    label = (body.label or "").strip() or preset["label"]
    s1, s2 = float(c.strength_stage_1), float(c.strength_stage_2)
    ltx_recipe = {
        "recipe": recipe.name,
        "characters": [{"name": c.name, "trigger": c.trigger, "gender": c.gender, "char_lora": c.char_lora,
                        "s1": s1, "s2": s2}],
        "character": c.name, "trigger": c.trigger, "char_lora": c.char_lora, "char_s1": s1, "char_s2": s2,
        "frames": FRAMES, "img_compression": recipe.img_compression, "content_loras": recipe.content_loras or [],
        "checkpoint": recipe.checkpoint, "edited": ["prompt", "negative"], "face_loop": body.preset,
    }
    next_priority = (await db.execute(select(func.coalesce(func.max(Job.priority), -1)).where(
        Job.user_id == user.id, Job.priority < REG_PRIORITY_BASE))).scalar_one() + 1
    job = Job(user_id=user.id, name=f"{c.name} - Face Loop ({label})", width=width, height=height, fps=FPS,
              seed=uuid.uuid4().int % (2 ** 47), continuation_mode="traditional", priority=next_priority,
              tags="FaceLoop", starting_image=start_uri)
    db.add(job)
    await db.flush()
    db.add(Segment(job_id=job.id, index=0, prompt=prompt, duration_seconds=FRAMES / FPS, speed=1.0,
                   negative_prompt=preset["negative"], ltx_recipe=ltx_recipe, auto_finalize=False))
    # start_image records the photo that was CHOSEN; the job carries the cropped one.
    loop = FaceLoop(character_id=c.id, label=label, preset=body.preset, prompt=prompt,
                    start_image=body.start_image, job_id=job.id, status="rendering",
                    created_at=datetime.now(timezone.utc))
    db.add(loop)
    await db.commit()
    await db.refresh(loop)
    return (await _response(db, [loop]))[0]


async def _loop(db: AsyncSession, loop_id: uuid.UUID) -> FaceLoop:
    x = await db.get(FaceLoop, loop_id)
    if x is None:
        raise HTTPException(status_code=404, detail="Face loop not found")
    return x


@router.patch("/face-loops/{loop_id}", response_model=FaceLoopResponse, dependencies=[Depends(get_current_user)])
async def rename_loop(loop_id: uuid.UUID, body: FaceLoopUpdate, db: AsyncSession = Depends(get_db)):
    x = await _loop(db, loop_id)
    x.label = body.label.strip()
    x.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(x)
    return (await _response(db, [x]))[0]


@router.delete("/face-loops/{loop_id}", status_code=204, dependencies=[Depends(get_current_user)])
async def delete_loop(loop_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Removes the prepared files and the row. The render job stays (it is an ordinary job)."""
    x = await _loop(db, loop_id)
    if x.asset_prefix:
        bucket, prefix = x.asset_prefix.replace("s3://", "").split("/", 1)
        await asyncio.to_thread(delete_prefix, prefix.rstrip("/") + "/", bucket)
    await db.delete(x)
    await db.commit()
    return Response(status_code=204)


@router.post("/face-loops/{loop_id}/reprocess", response_model=FaceLoopResponse,
             dependencies=[Depends(get_current_user)])
async def reprocess_loop(loop_id: uuid.UUID, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    """Run the preparation again on the finished render (after a failure, or a restart mid-way).
    If the render itself failed, queue the render again instead (re-cropping an old start)."""
    x = await _loop(db, loop_id)
    seg = (await db.execute(select(Segment).where(Segment.job_id == x.job_id, Segment.index == 0))).scalars().first()
    if seg is not None and seg.status == SegmentStatus.FAILED and not seg.discarded:
        from app.routes.segments import requeue_failed_segment

        job = await db.get(Job, x.job_id)
        await refresh_start_for_retry(db, job)
        await requeue_failed_segment(db, seg, job)
        x.status, x.error, x.updated_at = "rendering", None, datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(x)
        return (await _response(db, [x]))[0]
    if seg is None or seg.status != SegmentStatus.COMPLETED or not seg.output_path:
        raise HTTPException(status_code=409, detail="the render has not finished yet")
    x.status, x.error, x.updated_at = "processing", None, datetime.now(timezone.utc)
    await db.commit()
    background_tasks.add_task(prepare_loop, x.id)
    await db.refresh(x)
    return (await _response(db, [x]))[0]


async def on_render_uploaded(db: AsyncSession, segment: Segment, background_tasks: BackgroundTasks) -> None:
    """Called from upload_segment_output: a face loop's render just landed, so prepare it."""
    if segment.index != 0:
        return
    x = (await db.execute(select(FaceLoop).where(FaceLoop.job_id == segment.job_id,
                                                 FaceLoop.status == "rendering"))).scalars().first()
    if x is None:
        return
    x.status, x.updated_at = "processing", datetime.now(timezone.utc)
    await db.commit()
    background_tasks.add_task(prepare_loop, x.id)


async def prepare_loop(loop_id: uuid.UUID) -> None:
    """Background: download the render, prepare it, upload the loop files, record the result."""
    from app.face_loop_prep import prepare

    async with _prep_lock:
        async with async_session() as db:
            x = await db.get(FaceLoop, loop_id)
            if x is None:
                return
            seg = (await db.execute(select(Segment).where(Segment.job_id == x.job_id,
                                                          Segment.index == 0))).scalars().first()
            c = await db.get(LtxCharacter, x.character_id)
            try:
                if seg is None or not seg.output_path:
                    raise ValueError("the render has no output")
                with tempfile.TemporaryDirectory(prefix="faceloop-") as tmp:
                    src = os.path.join(tmp, "render.mp4")
                    await asyncio.to_thread(download_file, seg.output_path, src)
                    out = os.path.join(tmp, "out")
                    result = await asyncio.to_thread(prepare, src, out)
                    prefix = f"ar-face/{_slug(c.name if c else 'character')}/{x.id}"
                    for name in ("loop.mp4", "face.json", "face.bin", "poster.jpg"):
                        await asyncio.to_thread(upload_file, os.path.join(out, name), f"{prefix}/{name}",
                                                settings.s3_jobs_bucket)
                x.asset_prefix = f"s3://{settings.s3_jobs_bucket}/{prefix}"
                x.window_start, x.window_end = result.window
                x.stats = {**result.stats, "frames": result.frames, "fps": result.fps}
                x.status, x.error = "ready", None
            except Exception as e:  # noqa: BLE001 - any failure is recorded on the row, not lost in a log
                logger.exception("face loop %s preparation failed", loop_id)
                x.status, x.error = "failed", f"{type(e).__name__}: {e}"[:2000]
            x.updated_at = datetime.now(timezone.utc)
            await db.commit()
