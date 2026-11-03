"""The prompt enhancer's queue: ask for many images and walk away (console#719).

One image at a time, oldest first. The queue is two columns on image_meta rather than a list in
this process, unlike the caption tickets: a batch takes long enough that a deploy landing in the
middle of it is the normal case, and a walk-away queue that silently empties is no queue at all.
An image stays queued until its result is saved, so one interrupted by a restart simply runs again.

Same #559 rule as the tickets: no connection is held across the enhancer call.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select, update

from app import prompt_enhance, s3
from app.database import async_session
from app.models import ImageMeta

logger = logging.getLogger(__name__)

IDLE_POLL_S = 3.0
#: Tries per image, and the wait after a failed one. The enhancer is a service on another host,
#: so a restart there should cost a pause, not a batch of failures.
MAX_ATTEMPTS = 3
RETRY_WAIT_S = 60.0

#: The image the runner is on now. Display only; the row is what decides.
running: str | None = None
_attempts: dict[str, int] = {}


class _Permanent(Exception):
    """Trying again cannot help: the image is unreadable or has nothing to expand."""


async def _next() -> tuple[str, datetime, str | None] | None:
    async with async_session() as db:
        meta = (await db.execute(
            select(ImageMeta).where(ImageMeta.enhance_queued_at.is_not(None))
            .order_by(ImageMeta.enhance_queued_at, ImageMeta.path).limit(1)
        )).scalars().first()
        if meta is None:
            return None
        return meta.path, meta.enhance_queued_at, meta.tags


async def _write(path: str, queued_at: datetime, values: dict) -> None:
    """Settle one turn, unless it was cancelled or asked for again while the enhancer worked."""
    async with async_session() as db:
        await db.execute(
            update(ImageMeta)
            .where(ImageMeta.path == path, ImageMeta.enhance_queued_at == queued_at)
            .values(**values))
        await db.commit()


async def _enhance(path: str, tags: str | None) -> str:
    async with async_session() as db:
        action = await prompt_enhance.action_prompt(db, tags)
    if not action:
        raise _Permanent("no Action tag with a description")
    try:
        image = await asyncio.to_thread(s3.download_bytes, path)
    except Exception as e:  # noqa: BLE001 - whatever S3 says, the image cannot be read
        raise _Permanent(f"could not read the image: {e}") from e
    return (await prompt_enhance.enhance(image, action))["prompt"]


async def run_one() -> bool:
    """Take one turn. False when the queue is empty."""
    global running
    entry = await _next()
    if entry is None:
        return False
    path, queued_at, tags = entry
    running = path
    try:
        prompt = await _enhance(path, tags)
    except _Permanent as e:
        _attempts.pop(path, None)
        await _write(path, queued_at, {"enhance_queued_at": None, "enhance_error": str(e)})
    except prompt_enhance.EnhanceError as e:
        attempts = _attempts.get(path, 0) + 1
        logger.warning("prompt enhance failed for %s (try %d): %s", path, attempts, e)
        if attempts >= MAX_ATTEMPTS:
            _attempts.pop(path, None)
            await _write(path, queued_at, {"enhance_queued_at": None, "enhance_error": str(e)})
        else:
            _attempts[path] = attempts
            await asyncio.sleep(RETRY_WAIT_S)
    else:
        _attempts.pop(path, None)
        await _write(path, queued_at, {
            "enhanced_prompt": prompt, "enhanced_prompt_at": datetime.now(timezone.utc),
            "enhance_queued_at": None, "enhance_error": None})
    finally:
        running = None
    return True


async def enhance_queue_monitor() -> None:
    """Lives for the app's lifetime."""
    while True:
        try:
            if not await run_one():
                await asyncio.sleep(IDLE_POLL_S)
        except Exception:  # noqa: BLE001 - a bad turn must not end the queue
            logger.exception("Enhance queue turn failed")
            await asyncio.sleep(IDLE_POLL_S)
