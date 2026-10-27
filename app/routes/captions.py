"""Describe an image, so a prompt can stop contradicting its own start frame.

console#405. The console calls this for segment 0, where a person is present to read the
caption before it is used; the API resolves <SCENE> itself for continuations, where nobody
is. Same placeholder, same meaning, two resolution points — exactly the convention
_resolve_trigger already documents for <TRIGGER>.
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app import s3, worker_modes
from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.joycaption import (CaptionError, CaptionerBusy, CaptionerUnreachable, NoGpuInMode,
                            busy_render_beside_the_captioner, captioner_for, describe,
                            instruction_for, mark_scene_down, mark_scene_up,
                            scene_captioner, scene_marked_down)
from app.models import User
from app.routes.app_settings import _get_all_settings
from app.schemas.captions import CaptionRequest, CaptionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


async def _caption_base(db: AsyncSession, interactive: bool) -> str:
    """Where to send an image-description caption -- a dataset clip sheet, or a scene caption
    the scene service could not take -- refusing if the box that captioner shares a card with is rendering.

    The scene service never comes through here (wanly-console#572): it shares a card with no
    render stack, so it has nothing to be refused for. See caption_scene.

    THE CAPTIONER SHARES A CARD WITH A RENDER WORKER (wanly-gpu-docker#83). Loading the
    vision model beside a 720p render OOMs one of them. While the box is rendering an
    interactive caption goes to the fallback captioner when one is configured, and is
    otherwise refused with the box's name. Claim-time <SCENE> resolution passes
    interactive=False and prefers the fallback outright: the claiming box is about to load
    the render. See captioner_for.

    ROUTED BY MODE FIRST (wanly-api#392). When the boxes say what mode they are in, such a
    caption goes to a box in "motion" mode (the wire name of the image-description mode) --
    whichever one it is, no config change -- and never to a box in another mode. None in motion mode: the fallback captioner if one is configured,
    else NoGpuInMode, which a held job waits out with that reason (no time limit: switching a
    box is what ends it). Only when no box reports a mode does the old single-URL path below run.
    """
    pick = await worker_modes.pick(db, "motion")
    if pick.box:
        return worker_modes.motion_url(pick.box)
    if pick.reporting:
        fallback = (settings.image_description_fallback_url or "").strip()
        if fallback:
            logger.info("No box in motion mode (%s); captioning on the fallback captioner %s",
                        pick.wait, fallback)
            return fallback
        raise NoGpuInMode(pick.wait or "no GPU in motion mode")

    busy = await busy_render_beside_the_captioner(db) if interactive else None
    base = captioner_for(busy, interactive)
    if base is None:
        # Name the REASON, because the two are fixed by different actions: a render in
        # flight is waited out, a mode is switched.
        from app.joycaption import _render_mode
        in_render_mode = busy and await _render_mode(
            type("W", (), {"friendly_name": busy})()) == "ltx-engine"
        raise CaptionerBusy(
            f"{busy} is in render mode, and the GPU does one job at a time. "
            f"Switch it to captions on the Workers page."
            if in_render_mode else
            f"{busy} is rendering, and the captioner shares its GPU. "
            f"Try again when the render finishes.")
    if busy:
        logger.info("%s is rendering; captioning on the fallback captioner %s", busy, base)
    return base


async def describe_scene(image: bytes, instruction: str, fallback_base) -> str:
    """The scene: on the scene service, or on the image-description captioner if that is down.

    `fallback_base` is an async callable returning the image-description captioner's URL --
    with its render refusal, which can raise CaptionerBusy -- and is called only when it is
    needed, so a scene caption is never refused because a render is running on that box
    (wanly-console#572).

    Every fallback is logged with its reason (app/joycaption.py mark_scene_down), so "scene
    captions are slow" can be read straight off the log: the scene service was down, and the
    old single-captioner path did the work. An EMPTY caption is not a fallback case -- the
    service answered, and its answer is the caller's to refuse.
    """
    cap = scene_captioner()
    if cap is not None:
        why = scene_marked_down()
        if why is None:
            try:
                text = await describe(image, instruction, captioner=cap)
            except CaptionerUnreachable as e:
                mark_scene_down(cap, e)
            else:
                mark_scene_up(cap)
                return text
        else:
            logger.info("Scene captioner %s still marked down (%s); this scene caption goes "
                        "to the single captioner", cap.url, why)
    base = await fallback_base()
    return await describe(image, instruction, base_url=base)


async def caption_image_bytes(db: AsyncSession, image: bytes,
                              style: str | None = None,
                              instruction: str | None = None,
                              interactive: bool = True) -> tuple[str, str]:
    """Caption bytes using the configured style. Returns (caption, instruction_used).

    The instruction is returned as well as the caption so a segment can record HOW it was
    described, not just what was said. A caption written under "terse" and one written under
    "rich" are different artefacts, and a rated panel should be able to tell them apart.
    """
    if instruction is None:
        cfg = await _get_all_settings(db)
        instruction = instruction_for(style or cfg.get("caption_style", ""),
                                      cfg.get("caption_instruction", ""))

    async def fallback_base() -> str:
        return await _caption_base(db, interactive)
    return await describe_scene(image, instruction, fallback_base), instruction


async def caption_clip_sheet(db: AsyncSession, sheet: bytes, instruction: str) -> str:
    """Caption a clip's contact sheet (wanly-api#411) on the image-description captioner.

    Never the scene captioner: JoyCaption describes one photograph, and a 2x2 grid reads to it
    as four photographs. Qwen3-VL reads the grid as moments in order, which is the only reason
    to send a sheet. Same render gate and fallback as any image-description caption.
    """
    base = await _caption_base(db, True)
    return await describe(sheet, instruction, base_url=base)


async def caption_image_scene(db: AsyncSession, image: bytes,
                              interactive: bool = True) -> tuple[str, str]:
    """The scene half alone, with the saved settings: (scene, instruction_used).

    What tagging an image spends (console#590).
    """
    cfg = await _get_all_settings(db)

    async def base_for() -> str:
        return await _caption_base(db, interactive)
    return await caption_scene(image, cfg, base_for)


async def caption_scene(image: bytes, cfg: dict, fallback_base,
                        style: str | None = None,
                        instruction: str | None = None) -> tuple[str, str]:
    """The static half alone: (scene, instruction_used).

    On the scene service (wanly-console#572), never refused for a render; on the
    image-description captioner -- through `fallback_base`, which carries the render refusal
    -- only when the scene service is down or not configured. The caption tickets call this
    in the scene lane.
    """
    if instruction is None:
        instruction = instruction_for(style or cfg.get("caption_style", ""),
                                      cfg.get("caption_instruction", ""))
    return await describe_scene(image, instruction, fallback_base), instruction


@router.post("/captions/describe", response_model=CaptionResponse)
async def describe_image(
    body: CaptionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Caption one image for the console's preview.

    This is the "a human is present" half of console#405: the console shows the caption,
    the person edits or accepts it, and the RESOLVED text is sent with the segment. Nothing
    is stored here — a preview the user rejects must leave no trace.
    """
    try:
        # boto3 is synchronous; off the event loop so one slow fetch does not stall the API.
        image = await asyncio.to_thread(s3.download_bytes, body.image_uri)
    except Exception as e:
        raise HTTPException(status_code=404,
                            detail=f"could not read {body.image_uri}: {e}") from e

    try:
        caption, instruction = await caption_image_bytes(
            db, image, style=body.style, instruction=body.instruction)
    except CaptionError as e:
        # 503 rather than 500: the captioner being down is a temporary condition on another
        # host, not a bug in this request. The console can say "try again" and mean it.
        logger.warning("caption failed for %s: %s", body.image_uri, e)
        raise HTTPException(status_code=503, detail=str(e)) from e

    return CaptionResponse(caption=caption, instruction=instruction,
                           words=len(caption.split()))
