"""Describe an image, so a prompt can stop contradicting its own start frame.

console#405. The console calls this for segment 0, where a person is present to read the
caption before it is used; the API resolves <SCENE> itself for continuations, where nobody
is. Same placeholder, same meaning, two resolution points — exactly the convention
_resolve_trigger already documents for <TRIGGER>.
"""
import asyncio
import logging
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app import s3
from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.joycaption import (CaptionError, CaptionerBusy, busy_render_beside_the_captioner,
                            captioner_for, describe, describe_motion, instruction_for)
from app.models import User
from app.routes.app_settings import _get_all_settings
from app.schemas.captions import CaptionRequest, CaptionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


async def _caption_base(db: AsyncSession, interactive: bool) -> str:
    """Where to send a caption, refusing if the sharing box is rendering.

    THE CAPTIONER SHARES A CARD WITH A RENDER WORKER (wanly-gpu-docker#83). Loading the
    vision model beside a 720p render OOMs one of them. While the box is rendering an
    interactive caption goes to the fallback captioner when one is configured, and is
    otherwise refused with the box's name. Claim-time <SCENE> resolution passes
    interactive=False and prefers the fallback outright: the claiming box is about to load
    the render. See captioner_for.
    """
    busy = await busy_render_beside_the_captioner(db) if interactive else None
    base = captioner_for(busy, interactive)
    if base is None:
        raise CaptionerBusy(
            f"{busy} is rendering, and the captioner shares its GPU. "
            f"Try again when the render finishes.")
    if busy:
        logger.info("%s is rendering; captioning on the fallback captioner %s", busy, base)
    return base


async def caption_image_bytes(db: AsyncSession, image: bytes,
                              style: str | None = None,
                              instruction: str | None = None,
                              interactive: bool = True) -> tuple[str, str]:
    """Caption bytes using the configured style. Returns (caption, instruction_used).

    The instruction is returned as well as the caption so a segment can record HOW it was
    described, not just what was said. A caption written under "terse" and one written under
    "rich" are different artefacts, and a rated panel should be able to tell them apart.
    """
    base = await _caption_base(db, interactive)
    if instruction is None:
        cfg = await _get_all_settings(db)
        instruction = instruction_for(style or cfg.get("caption_style", ""),
                                      cfg.get("caption_instruction", ""))
    return await describe(image, instruction, base_url=base), instruction


@dataclass
class ScenePair:
    """What one describe call produced. Motion None + error set is a partial success.

    A plain tuple of five was rejected: two of the four strings are instructions and a swap
    is invisible at the call site and in the response.
    """
    scene: str
    scene_instruction: str
    motion: str | None = None
    motion_instruction: str | None = None
    motion_error: str | None = None


async def caption_image_pair(db: AsyncSession, image: bytes,
                             style: str | None = None,
                             instruction: str | None = None,
                             motion_style: str | None = None,
                             motion_instruction: str | None = None,
                             interactive: bool = True) -> ScenePair:
    """Caption bytes twice: the static scene, then the motion half grounded on it.

    Two sequential Ollama calls, not one combined one: the combined-call motion half
    measurably degraded in the #326 prototype, and two calls keep per-output instruction
    provenance the way the static half already has it. Both calls go to the same captioner
    in the same keep_alive window, so the model is warm for the second one.

    A motion failure is NOT this call's failure. The static half is what <SCENE> consumes
    today and what claim-time resolution reuses; a captioner hiccup on the second call must
    not throw away the first (the ticket's partial-failure rule). The caller persists what
    came back and surfaces the gap.
    """
    base = await _caption_base(db, interactive)
    cfg = await _get_all_settings(db)
    if instruction is None:
        instruction = instruction_for(style or cfg.get("caption_style", ""),
                                      cfg.get("caption_instruction", ""))
    scene = await describe(image, instruction, base_url=base)

    # Kill-switch (#326): with a captioner whose model cannot do the motion half
    # (joycaption on the 2070 answers the directional prompt with plausible junk), the
    # motion call is skipped and motion stays None WITHOUT an error — an absent section,
    # not a failure, and the lightbox renders it that way.
    if not settings.motion_caption_enabled:
        return ScenePair(scene=scene, scene_instruction=instruction)

    custom = (motion_instruction if motion_instruction is not None
              else cfg.get("motion_instruction", ""))
    try:
        motion, motion_instr = await describe_motion(
            image, scene, style=motion_style or cfg.get("motion_style", ""),
            custom=custom, base_url=base)
    except CaptionError as e:
        logger.warning("motion caption failed (static half kept): %s", e)
        return ScenePair(scene=scene, scene_instruction=instruction, motion_error=str(e))
    return ScenePair(scene=scene, scene_instruction=instruction,
                     motion=motion, motion_instruction=motion_instr)


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
