"""The client for the Sulphur-2 prompt enhancer (wanly-api#480).

The service (wanly-gpu-docker#225, `prompt-enhance` on the 2070 beside the scene captioner)
takes what the model card says to send -- the text you want enhanced and, optionally, the
image -- and answers with the long scene script LTX needs. We send the image and its saved
scene description, so the enhancer starts from words the person already accepted.
"""
from __future__ import annotations

import base64
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class EnhanceError(RuntimeError):
    """The enhancer is off, unreachable, or answered with nothing."""


def enabled() -> bool:
    return bool((settings.prompt_enhance_url or "").strip())


async def _post(url: str, payload: dict) -> httpx.Response:
    async with httpx.AsyncClient() as c:
        return await c.post(url, json=payload, timeout=settings.prompt_enhance_timeout_s)


async def enhance(image: bytes, scene_description: str) -> dict:
    """{prompt, ms, cold} from the enhancer, or EnhanceError."""
    url = (settings.prompt_enhance_url or "").strip().rstrip("/")
    if not url:
        raise EnhanceError("the prompt enhancer is not configured (PROMPT_ENHANCE_URL is empty)")
    payload = {"text": scene_description, "image_b64": base64.b64encode(image).decode()}
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
