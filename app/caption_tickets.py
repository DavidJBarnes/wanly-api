"""Caption tickets: describe returns at once, and the caption happens in the background
(console#564).

WHAT WAS WRONG

    POST /images/scene held its HTTP request open for its whole turn in the caption queue
    (app/caption_queue.py). Behind a dataset's captioning, or a dozen held jobs, that turn is
    many minutes away, and a request that stays open that long is fragile: a proxy timeout, a
    phone going to sleep, or simply navigating away. Navigating away was the visible failure:
    the modal's "Describing..." lived only in that page's state, so coming back showed the
    image as idle while its caption was still in line, with no sign it was coming.

WHAT THIS DOES

    A ticket is the scene caption of ONE image -- an asyncio task that takes a turn in the
    scene lane, captions it, and saves those words on the image's ImageMeta row. The person
    asking gets the ticket back at once and polls it; the task does not care whether anyone
    is still listening.

    SINGLE-FLIGHT PER IMAGE. While a scene ticket for an image is queued or running, every
    other request for that image's scene -- the modal, a second tab, the New Job dialog, a
    held job (app/caption_hold.py) -- gets THAT ticket. Two captions would write two different
    descriptions, and the second would silently replace the words the first person (or job)
    already used.

    Finished tickets are kept for RESULT_TTL_S so a page opened later can still show "Failed:
    retry" or pick up the finished words.

    HALVES. A ticket still carries `half` ("scene", always): there used to be a second,
    motion half with its own lane (console#572/#590), removed in wanly-api#481. The field
    and the per-half keys stay so the API's shape does not change under the console.

WHAT IT DOES NOT DO

    Survive a restart. Tickets live in this process, like the queue itself: a deploy drops
    the queued describes. Held jobs do not care (their state is in the database and the hold
    sweep re-asks), and a dropped describe simply shows as not captioned, so it can be asked
    for again. A table for one user's describe queue was not worth a migration.

NEVER HOLDS A CONNECTION ACROSS A CAPTION

    The #559 rule. Every database touch is a short session of its own, none of them open while
    the ticket waits in line or while the captioner works.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app import caption_queue as cq
from app import s3
from app.database import async_session
from app.joycaption import CaptionError, CaptionerBusy
from app.models import ImageMeta

logger = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

#: The halves of an image's caption. Only the scene is left (the motion half went in
#: wanly-api#481); kept as a tuple so callers that loop over halves need not change.
SCENE = "scene"
HALVES = (SCENE,)

#: How long a finished ticket is remembered, for a page opened after the caption finished.
RESULT_TTL_S = 6 * 3600


class ImageUnreadable(CaptionError):
    """The image itself could not be fetched -- that image's problem, not the captioner's."""


@dataclass(eq=False)
class Ticket:
    path: str
    half: str
    origin: str
    params: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = QUEUED
    error: str | None = None
    #: The captioner refused because the box beside it is rendering (or in render mode).
    #: Waited out by the hold; shown as the reason to the person.
    busy: bool = False
    #: ...and the refusal is "no GPU in <mode> mode" (wanly-api#392): waited out with no time
    #: limit, because only a person switching a box ends it.
    mode_wait: bool = False
    unreadable: bool = False
    #: Taken out of line before it ran, because nothing needed it any more (withdraw()).
    withdrawn: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: Monotonic finish time, for the TTL.
    _finished_mono: float | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    _entry: object = None
    _queue: object = None
    #: Asked at the front of the line, before captioning: "is this still needed?" The caption
    #: hold passes one -- whatever was ahead of it in line may have been this image. Dropped
    #: as soon as a describe joins: a person asked for a fresh caption, and gets one.
    precheck: object = None
    #: Done without captioning, because the precheck said the words were already there.
    skipped: bool = False
    #: The held jobs this caption is for (console#590): [{"job_id", "name"}]. Shown on the
    #: image as "requested by job ...", because a caption nobody clicked for should say who
    #: asked.
    requested_by: list = field(default_factory=list)
    _image: bytes | None = None
    _cfg: dict | None = None

    @property
    def lane(self) -> str:
        return cq.SCENE_LANE

    @property
    def finished(self) -> bool:
        return self.status in (DONE, FAILED)

    async def wait(self) -> "Ticket":
        await self._done.wait()
        return self

    def add_requesters(self, jobs) -> None:
        """Note held jobs this caption is for, once each."""
        have = {r["job_id"] for r in self.requested_by}
        for j in jobs or []:
            if j["job_id"] not in have:
                self.requested_by.append(dict(j))
                have.add(j["job_id"])


#: The ticket in flight for each (image, half), queued or running. The single-flight table.
_active: dict[tuple[str, str], Ticket] = {}
#: Every ticket still remembered, by id.
_by_id: dict[str, Ticket] = {}
#: The most recent FINISHED ticket of each (image, half).
_last: dict[tuple[str, str], Ticket] = {}


def reset() -> None:
    """Forget everything. Tests only: each test has its own event loop."""
    _active.clear()
    _by_id.clear()
    _last.clear()


def _prune() -> None:
    cutoff = time.monotonic() - RESULT_TTL_S
    for tid, t in list(_by_id.items()):
        if t.finished and t._finished_mono is not None and t._finished_mono < cutoff:
            del _by_id[tid]
            if _last.get((t.path, t.half)) is t:
                del _last[(t.path, t.half)]


def _check_half(half: str) -> None:
    if half not in HALVES:
        raise ValueError(f"unknown caption half {half!r}")


# ---------------------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------------------

def active(path: str, half: str | None = None) -> Ticket | None:
    """The caption of this image that is queued or running right now, if any.

    `half` is accepted for the old per-half callers; anything but the scene has none."""
    t = _active.get((path, half or SCENE))
    return t if t is not None and not t.finished else None


def get(ticket_id: str) -> Ticket | None:
    _prune()
    return _by_id.get(ticket_id)


def latest(path: str, half: str | None = None) -> Ticket | None:
    """What a view should show for this image: the caption in flight, else the last finished.
    `half` is accepted for the old per-half callers; anything but the scene has none."""
    _prune()
    half = half or SCENE
    return active(path, half) or _last.get((path, half))


def request(path: str, half: str = SCENE, *, origin: str = "describe",
            params: dict | None = None, precheck=None,
            requested_by=None) -> tuple[Ticket, bool]:
    """Get this half of this image captioned. Returns (ticket, joined).

    joined=True: a caption of this half was already in flight and this is it -- nothing new
    was queued. Its params (style, instruction) are the first asker's; a second describe that
    named different ones gets the caption already coming rather than a second one.
    """
    _check_half(half)
    _prune()
    current = active(path, half)
    if current is not None:
        if origin == "describe" and current.status == QUEUED:
            current.precheck = None
        current.add_requesters(requested_by)
        logger.info("Caption ticket %s on %s (%s): %s joined it (%s)", current.id, path, half,
                    origin, current.status)
        return current, True
    t = _start(path, half, origin, params, precheck)
    t.add_requesters(requested_by)
    return t, False


def _start(path: str, half: str, origin: str, params: dict | None, precheck=None) -> Ticket:
    t = Ticket(path=path, half=half, origin=origin, params=dict(params or {}),
               precheck=precheck)
    # In line before the request answers, so its first report already has a position.
    _enter(t)
    _active[(path, half)] = t
    _by_id[t.id] = t
    t.task = asyncio.create_task(_run(t), name=f"caption-ticket {half} {path}")
    logger.info("Caption ticket %s on %s: %s queued (for a %s; lane depth %d)",
                t.id, path, half, origin, t._queue.depth())
    return t


def _enter(t: Ticket) -> None:
    """Take a place in the lane now (synchronously, so a position exists at once)."""
    q = cq.queue
    t._queue = q
    t._entry = q.reserve(t.path, kind="hold" if t.origin == "hold" else "describe",
                         token=t.id)


def withdraw(t: Ticket, reason: str) -> bool:
    """Take a QUEUED ticket out of line. Only the caption hold does this, for its own ticket,
    when the words it was queued for turned out to be saved already. Returns whether it was.
    """
    if t.status != QUEUED or t.task is None or t.task.done():
        return False
    t.withdrawn = True
    t.status = FAILED  # out of active() at once, so no describe can join it now
    t.error = f"withdrawn: {reason}"
    if _active.get((t.path, t.half)) is t:
        del _active[(t.path, t.half)]
    # Finished here rather than in _run: a task cancelled before its first step never runs
    # its body at all, so _run's finally would never set the event its waiter is on.
    t._queue.discard(t._entry)
    t.finished_at = datetime.now(timezone.utc)
    t._finished_mono = time.monotonic()
    t._done.set()
    t.task.cancel()
    logger.info("Caption ticket %s on %s (%s): withdrawn (%s)", t.id, t.path, t.half, reason)
    return True


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------

def view(t: Ticket | None) -> dict:
    """A ticket as the API reports it. Position is computed now, never stored."""
    q = t._queue if t is not None and t._queue is not None else cq.queue
    if t is None:
        return {"ticket_id": None, "status": None, "position": None, "depth": q.depth(),
                "half": None, "mode": None, "origin": None, "error": None, "busy": False,
                "mode_wait": False,
                "created_at": None, "started_at": None, "finished_at": None, "lane": None,
                "requested_by": []}
    position = None
    if t.status == QUEUED:
        position = q.token_status(t.id)["position"]
    elif t.status == RUNNING:
        position = 0
    return {"ticket_id": t.id, "status": t.status, "position": position, "depth": q.depth(),
            "half": t.half, "mode": t.half, "origin": t.origin, "error": t.error,
            "busy": t.busy, "mode_wait": t.mode_wait, "created_at": t.created_at, "started_at": t.started_at,
            "finished_at": t.finished_at, "lane": None if t.finished else t.lane,
            "requested_by": [dict(r) for r in t.requested_by]}


def recent() -> list[Ticket]:
    """The last finished ticket of every (image, half) still remembered with nothing newer in
    flight, newest first."""
    _prune()
    out = [t for (p, h), t in _last.items() if active(p, h) is None]
    out.sort(key=lambda t: t.finished_at or t.created_at, reverse=True)
    return out


# ---------------------------------------------------------------------------------------
# The work
# ---------------------------------------------------------------------------------------

async def _run(t: Ticket) -> None:
    try:
        async with t._queue.turn(t.path, reserved=t._entry):
            t.status = RUNNING
            t.started_at = datetime.now(timezone.utc)
            if t.precheck is not None and await t.precheck():
                t.skipped = True
            else:
                await _scene_step(t)
        if not t.skipped:
            # Release what this caption completes now, not when the hold next looks.
            await _settle(t)
        t.status = DONE
        t._image = None  # not kept for RESULT_TTL_S
        logger.info("Caption ticket %s on %s: %s done%s", t.id, t.path, t.half,
                    " (not needed any more: the words were saved while it waited)"
                    if t.skipped else "")
    except asyncio.CancelledError:
        t.status = FAILED
        if not t.withdrawn:
            t.error = "cancelled: the API stopped before this caption finished"
            raise
        # Withdrawn: an ordinary ending, not a shutdown -- do not propagate.
    except CaptionerBusy as e:
        t.status, t.error, t.busy = FAILED, str(e), True
        t.mode_wait = bool(getattr(e, "mode_wait", False))
        logger.info("Caption ticket %s on %s: %s refused, the captioner is busy (%s)",
                    t.id, t.path, t.half, e)
    except ImageUnreadable as e:
        t.status, t.error, t.unreadable = FAILED, str(e), True
        logger.warning("Caption ticket %s on %s: %s", t.id, t.path, e)
    except CaptionError as e:
        t.status, t.error = FAILED, str(e)
        logger.warning("Caption ticket %s on %s: %s failed: %s", t.id, t.path, t.half, e)
    except Exception as e:  # noqa: BLE001 - a person has to hear about it, whatever it was
        t.status, t.error = FAILED, f"{type(e).__name__}: {e}"
        logger.exception("Caption ticket %s on %s (%s) failed", t.id, t.path, t.half)
    finally:
        t._queue.discard(t._entry)  # cancelled before its turn began: no phantom in line
        t.finished_at = datetime.now(timezone.utc)
        t._finished_mono = time.monotonic()
        key = (t.path, t.half)
        if _active.get(key) is t:
            del _active[key]
        prev = _last.get(key)
        # A withdrawn ticket is not a result worth showing: nothing was asked of the
        # captioner, and "failed" on the image would be a lie.
        if not t.withdrawn and (
                prev is None or (prev.finished_at or prev.created_at) <= t.finished_at):
            _last[key] = t
        t._done.set()


async def _load(t: Ticket) -> str:
    """The settings, and the image's saved scene, on a short session; then the image itself.
    Returns the saved scene ("" when there is none). Never holds a connection across S3."""
    from app.routes import captions
    async with async_session() as db:
        if t._cfg is None:
            t._cfg = await captions._get_all_settings(db)
        saved_scene = await _saved_scene(db, t.path)
    if t._image is None:
        try:
            t._image = await asyncio.to_thread(s3.download_bytes, t.path)
        except Exception as e:
            raise ImageUnreadable(f"could not read {t.path}: {e}") from e
    return saved_scene


async def _saved_scene(db, path: str) -> str:
    meta = await db.get(ImageMeta, path, populate_existing=True)
    return (meta.scene_description or "").strip() if meta else ""


async def _fallback_base() -> str:
    """The image-description captioner's URL -- the scene's fallback when the scene service is
    down -- refusing (CaptionerBusy) while its box renders. Its own short session.
    interactive=True routes it exactly as the modal's was routed."""
    from app.routes import captions
    async with async_session() as db:
        return await captions._caption_base(db, interactive=True)


async def _scene_step(t: Ticket) -> None:
    """Make the scene and save it. Raises CaptionError and its kinds."""
    from app.routes import captions

    await _load(t)
    p = t.params
    scene, instruction = await captions.caption_scene(
        t._image, t._cfg, _fallback_base, style=p.get("style"), instruction=p.get("instruction"))
    if not (scene or "").strip():
        # A blank caption is a failure wearing a success's clothes. Storing it would mark
        # the image described and stop anything ever asking again.
        raise CaptionError("the captioner returned nothing for this image")
    async with async_session() as db:
        meta = await db.get(ImageMeta, t.path)
        if meta is None:
            meta = ImageMeta(path=t.path)
            db.add(meta)
        apply_scene(meta, scene, instruction)
        await db.commit()
    logger.info("Caption ticket %s: scene of %s saved (%d words)", t.id, t.path,
                len(scene.split()))


def apply_scene(meta: ImageMeta, scene: str, instruction: str | None) -> None:
    """Write the scene onto a row, and nothing else."""
    meta.scene_description = scene.strip()
    meta.scene_instruction = instruction
    meta.scene_described_at = datetime.now(timezone.utc)


async def _settle(t: Ticket) -> None:
    """Release every held segment on the image whose scene is now saved. Non-fatal: the
    hold's own waiter and sweep release it anyway, this only makes it immediate."""
    from app import caption_hold
    try:
        await caption_hold.settle(t.path, finishing=t)
    except Exception:  # noqa: BLE001
        logger.exception("Caption ticket %s: could not release holds on %s", t.id, t.path)
