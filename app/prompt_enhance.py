"""The client for the Sulphur-2 prompt enhancer (wanly-api#480).

The service (wanly-gpu-docker#225, `prompt-enhance` on the 2070 beside the scene captioner)
takes what the model card says to send -- the text you want enhanced and, optionally, the
image -- and answers with the long scene script LTX needs.

THE TEXT IS WHAT HAPPENS, NOT WHAT THE FRAME IS (wanly-api#486). The enhancer stands in for
LTX-2's own i2v enhancer, whose spec is "follow the user's raw input prompt: all requested
motion, actions, audio" and "describe only changes from the image". It sees the image itself.
Sending the scene description (as #480 did) got back long descriptions of a still frame. So
the text is the image's Action tags' descriptions -- every one, in the image's tag order.
"""
from __future__ import annotations

import base64
import logging

import httpx

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import ACTION_GROUP, TitleTag

logger = logging.getLogger(__name__)


async def action_prompt(db: AsyncSession, tags: str | None) -> str | None:
    """The "what happens" text for an image: its Action tags' descriptions, ALL of them, in the
    order the image carries the tags. None when it has no Action tag with a description.

    Image tags are stored by name, matched case-insensitively. A tag with no description
    contributes nothing (a non-motion Action tag, or one not written yet).
    """
    names = [t.strip() for t in (tags or "").split(",") if t.strip()]
    if not names:
        return None
    rows = (await db.execute(
        select(TitleTag).where(TitleTag.group == ACTION_GROUP,
                               func.lower(TitleTag.name).in_([n.lower() for n in names]))
    )).scalars().all()
    by_name = {r.name.lower(): (r.description or "").strip() for r in rows}
    seen: set[str] = set()
    lines = []
    for n in names:
        key = n.lower()
        if key in seen or not by_name.get(key):
            continue
        seen.add(key)
        lines.append(by_name[key])
    return "\n".join(lines) or None


class EnhanceError(RuntimeError):
    """The enhancer is off, unreachable, or answered with nothing."""


def enabled() -> bool:
    return bool((settings.prompt_enhance_url or "").strip())


async def _post(url: str, payload: dict) -> httpx.Response:
    async with httpx.AsyncClient() as c:
        return await c.post(url, json=payload, timeout=settings.prompt_enhance_timeout_s)


async def enhance(image: bytes, text: str) -> dict:
    """{prompt, ms, cold} from the enhancer, or EnhanceError."""
    url = (settings.prompt_enhance_url or "").strip().rstrip("/")
    if not url:
        raise EnhanceError("the prompt enhancer is not configured (PROMPT_ENHANCE_URL is empty)")
    payload = {"text": text, "image_b64": base64.b64encode(image).decode()}
    try:
        r = await _post(f"{url}/enhance", payload)
    except httpx.HTTPError as e:
        raise EnhanceError(f"the prompt enhancer at {url} did not answer: {e!r}") from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail") or r.text
        except ValueError:
            detail = r.text
        raise EnhanceError(f"the prompt enhancer answered {r.status_code}: {str(detail)[:500]}")
    body = r.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise EnhanceError("the prompt enhancer answered with nothing")
    logger.info("enhanced prompt: %d words in %s ms (cold=%s)", len(prompt.split()),
                body.get("ms"), body.get("cold"))
    return {"prompt": prompt, "ms": body.get("ms"), "cold": bool(body.get("cold"))}
