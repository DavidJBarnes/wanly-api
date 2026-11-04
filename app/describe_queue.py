"""The scene describer's queue: every uploaded image gets described, with nobody watching
(console#727).

Describing used to start when an image was tagged, as a background loop that stopped at the
first refusal and never came back for the rest. Most images arrive from A1111's upload script in
batches, and the scene service on the 2070 is there for exactly this, so now an upload joins
this queue and is described whether or not anyone tags it.

Built like the enhancer's queue (app/enhance_queue.py): two columns on image_meta, oldest
first, one image at a time, so a deploy in the middle of a batch loses nothing. Each turn is an
ordinary caption ticket (app/caption_tickets.py), which keeps it single-flight with Describe,
the lightbox and held jobs; it also means a person's Describe never waits behind a whole upload
batch, only behind the one image this queue has in the lane.

Refusals: busy (the box beside the captioner is rendering, or in a mode with no GPU) is waited
out with no limit -- it ends by itself, and a walk-away queue that fails a batch because a render
ran is no queue at all. Any other caption failure gets MAX_ATTEMPTS tries, an unreadable image
none; then the image is marked failed and the queue moves on.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import func, select, update

from app import caption_tickets
from app.database import async_session
from app.models import ImageMeta

logger = logging.getLogger(__name__)

IDLE_POLL_S = 3.0
MAX_ATTEMPTS = 3
RETRY_WAIT_S = 60.0
#: What an upload may be described as. Anything else uploaded is not a picture.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

#: The image the runner is on now. Display only; the row is what decides.
running: str | None = None
#: The line's size and its failures as of the runner's last look, for the toolbar's caption
#: poll, which must not touch the database (it runs every 3 s on every page). queue() bumps it so
#: an upload shows at once; the runner's next turn corrects it.
counts = {"waiting": 0, "failed": 0}
_attempts: dict[str, int] = {}


def is_image(path: str) -> bool:
    return path.lower().endswith(IMAGE_SUFFIXES)


def queue(meta: ImageMeta, at: datetime | None = None) -> bool:
    """Put this row in line unless it is described or already waiting. Returns whether it was.
    The caller commits."""
    if (meta.scene_description or "").strip() or meta.describe_queued_at is not None:
        return False
    meta.describe_queued_at = at or datetime.now(timezone.utc)
    meta.describe_error = None
    counts["waiting"] += 1
    return True


async def status(db) -> dict:
    """How many are in line (the one running included), and which failed."""
    waiting = (await db.execute(select(func.count()).select_from(ImageMeta)
                                .where(ImageMeta.describe_queued_at.is_not(None)))).scalar_one()
    failed = (await db.execute(
        select(ImageMeta.path, ImageMeta.describe_error)
        .where(ImageMeta.describe_error.is_not(None)).order_by(ImageMeta.path))).all()
    counts.update(waiting=waiting, failed=len(failed))
    return {"running": running, "waiting": waiting,
            "failed": [{"path": p, "error": e} for p, e in failed]}


def backlog() -> tuple[int, int]:
    """(waiting not yet in the lane, failed), from the cached counts. No database."""
    return max(0, counts["waiting"] - (1 if running else 0)), counts["failed"]


async def _next() -> tuple[str, datetime, bool] | None:
    async with async_session() as db:
        await status(db)
        meta = (await db.execute(
            select(ImageMeta).where(ImageMeta.describe_queued_at.is_not(None))
            .order_by(ImageMeta.describe_queued_at, ImageMeta.path).limit(1)
        )).scalars().first()
        if meta is None:
            return None
        return meta.path, meta.describe_queued_at, bool((meta.scene_description or "").strip())


async def _write(path: str, queued_at: datetime, values: dict) -> None:
    """Settle one turn, unless the queue was cleared or the image re-queued meanwhile."""
    async with async_session() as db:
        await db.execute(
            update(ImageMeta)
            .where(ImageMeta.path == path, ImageMeta.describe_queued_at == queued_at)
            .values(**values))
        await db.commit()


def _described(path: str):
    """The ticket's precheck: whatever was ahead of it in the lane may have been this image."""
    async def check() -> bool:
        async with async_session() as db:
            return bool(await caption_tickets._saved_scene(db, path))
    return check


async def run_one() -> bool:
    """Take one turn. False when the queue is empty."""
    global running
    entry = await _next()
    if entry is None:
        return False
    path, queued_at, described = entry
    done = {"describe_queued_at": None, "describe_error": None}
    if described:
        # Described some other way while it waited: a person's Describe, or a held job.
        _attempts.pop(path, None)
        await _write(path, queued_at, done)
        return True
    running = path
    try:
        ticket, _ = caption_tickets.request(path, origin="auto", precheck=_described(path))
        await ticket.wait()
    finally:
        running = None
    if ticket.status == caption_tickets.DONE:
        _attempts.pop(path, None)
        await _write(path, queued_at, done)
    elif ticket.withdrawn:
        pass  # only ever because the words were saved; the next turn sees them
    elif ticket.busy:
        logger.info("describe queue: captioner busy at %s, waiting (%s)", path, ticket.error)
        await asyncio.sleep(RETRY_WAIT_S)
    elif ticket.unreadable:
        _attempts.pop(path, None)
        await _write(path, queued_at, {"describe_queued_at": None, "describe_error": ticket.error})
    else:
        attempts = _attempts.get(path, 0) + 1
        logger.warning("describe failed for %s (try %d): %s", path, attempts, ticket.error)
        if attempts >= MAX_ATTEMPTS:
            _attempts.pop(path, None)
            await _write(path, queued_at,
                         {"describe_queued_at": None, "describe_error": ticket.error})
        else:
            _attempts[path] = attempts
            await asyncio.sleep(RETRY_WAIT_S)
    return True


async def describe_queue_monitor() -> None:
    """Lives for the app's lifetime."""
    while True:
        try:
            if not await run_one():
                await asyncio.sleep(IDLE_POLL_S)
        except Exception:  # noqa: BLE001 - a bad turn must not end the queue
            logger.exception("Describe queue turn failed")
            await asyncio.sleep(IDLE_POLL_S)
