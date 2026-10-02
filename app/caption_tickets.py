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

    A ticket is ONE caption of ONE image: an asyncio task that takes a turn in the ordinary
    caption queue, captions the image, and saves the words on its ImageMeta row exactly as
    the modal always did. The person asking gets the ticket back at once and polls it; the
    task does not care whether anyone is still listening.

    SINGLE-FLIGHT PER IMAGE. While a ticket for an image is queued or running, every other
    request for that image -- the modal, a second tab, the New Job dialog, a held job
    (app/caption_hold.py) -- gets THAT ticket. Two captions of one image would write two
    different descriptions, and the second would silently replace the words the first
    person (or job) already used.

    Two modes, because a held job missing only its motion paragraph must not have its scene
    regenerated (the person already read it):

      * "pair": the scene and the motion paragraph, from one call pair. Describe and re-roll.
      * "motion": the motion paragraph alone, grounded on the saved scene. Only the caption
        hold asks for this. A describe that arrives while one is still QUEUED upgrades it to
        a pair (a re-roll was asked for, and the hold then uses the re-rolled words); one that
        arrives while it is RUNNING queues a pair behind it.

    Finished tickets are kept for RESULT_TTL_S so a page opened later can still show "Failed:
    retry" or pick up the finished words.

TWO LANES (wanly-console#572)

    The halves are made by different captioners -- the scene by JoyCaption on the scene
    service (seconds), the motion paragraph by Qwen3-VL on a 3090 (much longer) -- and each
    has its own line (app/caption_queue.py). A pair takes a turn in the SCENE lane, saves the
    scene (clearing any old motion, which would no longer match it), releases every held
    segment that needed only the scene, and then joins the MOTION lane. A motion-only ticket
    goes straight to the motion lane. So a scene never waits behind somebody's motion
    paragraph, and a held job's <SCENE> is released the moment its scene is saved while its
    <MOTION> waits for the motion.

    A motion paragraph is saved only beside the scene it was grounded on: if a re-roll
    replaced the scene while the motion was being made, the paragraph is dropped (the
    re-roll makes its own).

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
from app.config import settings
from app.database import async_session
from app.joycaption import CaptionError, CaptionerBusy, describe_motion
from app.models import ImageMeta

logger = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

PAIR = "pair"
MOTION = "motion"

#: How long a finished ticket is remembered, for a page opened after the caption finished.
RESULT_TTL_S = 6 * 3600


class ImageUnreadable(CaptionError):
    """The image itself could not be fetched -- that image's problem, not the captioner's."""


@dataclass(eq=False)
class Ticket:
    path: str
    mode: str
    origin: str
    params: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = QUEUED
    error: str | None = None
    #: The captioner refused because the box beside it is rendering (or in render mode).
    #: Waited out by the hold; shown as the reason to the person.
    busy: bool = False
    unreadable: bool = False
    #: Taken out of line before it ran, because nothing needed it any more (withdraw()).
    withdrawn: bool = False
    #: The scene was saved but the motion half failed -- a partial success.
    motion_error: str | None = None
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
    #: Which line it is in now: "scene", then "motion" (cq.SCENE_LANE / cq.MOTION_LANE).
    lane: str = cq.SCENE_LANE
    #: The motion captioner refused for a render while the scene was saved: the hold waits
    #: it out like any refusal, rather than failing the job.
    motion_busy: bool = False
    #: Carried from the scene step to the motion step, so the image is fetched once.
    _image: bytes | None = None
    _cfg: dict | None = None
    _scene: str | None = None

    @property
    def finished(self) -> bool:
        return self.status in (DONE, FAILED)

    async def wait(self) -> "Ticket":
        await self._done.wait()
        return self


#: The ticket in flight for each image (queued or running). The single-flight table.
_active: dict[str, Ticket] = {}
#: Every ticket still remembered, by id.
_by_id: dict[str, Ticket] = {}
#: The most recent FINISHED ticket of each image.
_last: dict[str, Ticket] = {}


def reset() -> None:
    """Forget everything. Tests only: each test has its own event loop."""
    _active.clear()
    _by_id.clear()
    _last.clear()
    cq.motion_queue = cq.CaptionQueue()


def _prune() -> None:
    cutoff = time.monotonic() - RESULT_TTL_S
    for tid, t in list(_by_id.items()):
        if t.finished and t._finished_mono is not None and t._finished_mono < cutoff:
            del _by_id[tid]
            if _last.get(t.path) is t:
                del _last[t.path]


# ---------------------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------------------

def active(path: str) -> Ticket | None:
    """The caption of this image that is queued or running right now, if any."""
    t = _active.get(path)
    return t if t is not None and not t.finished else None


def get(ticket_id: str) -> Ticket | None:
    _prune()
    return _by_id.get(ticket_id)


def latest(path: str) -> Ticket | None:
    """What a view should show for this image: the one in flight, else the last finished."""
    _prune()
    return active(path) or _last.get(path)


def request(path: str, *, mode: str = PAIR, origin: str = "describe",
            params: dict | None = None, precheck=None) -> tuple[Ticket, bool]:
    """Get this image captioned. Returns (ticket, joined).

    joined=True: a caption of this image was already in flight and this is it -- nothing new
    was queued. Its params (style, instruction) are the first asker's; a second describe that
    named different ones gets the caption already coming rather than a second one.
    """
    _prune()
    current = active(path)
    if current is not None:
        if origin == "describe" and current.status == QUEUED:
            current.precheck = None
        if mode == PAIR and current.mode == MOTION:
            # A re-roll asked for while only the motion paragraph is coming. A motion ticket
            # waits in the motion lane and cannot turn into a scene caption there, so the
            # re-roll is a pair of its own, and it becomes the image's ticket: anyone asking
            # from now on -- and a held job waiting on the motion ticket -- joins it. A motion
            # ticket that has not started is withdrawn (its paragraph would be grounded on
            # the scene about to be replaced); a running one finishes, and its words are
            # dropped if the scene changed underneath it.
            if current.status == QUEUED:
                withdraw(current, f"a {origin} re-roll replaced it")
            return _start(path, PAIR, origin, params, precheck), False
        logger.info("Caption ticket %s on %s: %s joined it (%s)", current.id, path, origin,
                    current.status)
        return current, True
    return _start(path, mode, origin, params, precheck), False


def _start(path: str, mode: str, origin: str, params: dict | None, precheck=None) -> Ticket:
    t = Ticket(path=path, mode=mode, origin=origin, params=dict(params or {}),
               precheck=precheck)
    # In line before the request answers, so its first report already has a position. A
    # pair starts in the scene lane; a motion-only ticket has no scene to make.
    if mode == MOTION:
        _enter(t, cq.MOTION_LANE)
    else:
        _enter(t, cq.SCENE_LANE)
    _active[path] = t
    _by_id[t.id] = t
    t.task = asyncio.create_task(_run(t), name=f"caption-ticket {path}")
    logger.info("Caption ticket %s on %s: queued (%s, for a %s; %s lane depth %d)",
                t.id, path, mode, origin, t.lane, t._queue.depth())
    return t


def _enter(t: Ticket, lane: str) -> None:
    """Take a place in `lane` now (synchronously, so a position exists at once)."""
    q = cq.motion_queue if lane == cq.MOTION_LANE else cq.queue
    t.lane = lane
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
    if _active.get(t.path) is t:
        del _active[t.path]
    # Finished here rather than in _run: a task cancelled before its first step never runs
    # its body at all, so _run's finally would never set the event its waiter is on.
    t._queue.discard(t._entry)
    t.finished_at = datetime.now(timezone.utc)
    t._finished_mono = time.monotonic()
    t._done.set()
    t.task.cancel()
    logger.info("Caption ticket %s on %s: withdrawn (%s)", t.id, t.path, reason)
    return True


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------

def view(t: Ticket | None) -> dict:
    """A ticket as the API reports it. Position is computed now, never stored."""
    q = t._queue if t is not None and t._queue is not None else cq.queue
    if t is None:
        return {"ticket_id": None, "status": None, "position": None, "depth": q.depth(),
                "mode": None, "origin": None, "error": None, "busy": False,
                "motion_error": None, "created_at": None, "started_at": None,
                "finished_at": None, "lane": None}
    position = None
    if t.status == QUEUED:
        position = q.token_status(t.id)["position"]
    elif t.status == RUNNING:
        position = 0
    return {"ticket_id": t.id, "status": t.status, "position": position, "depth": q.depth(),
            "mode": t.mode, "origin": t.origin, "error": t.error, "busy": t.busy,
            "motion_error": t.motion_error, "created_at": t.created_at,
            "started_at": t.started_at, "finished_at": t.finished_at,
            "lane": None if t.finished else t.lane}


def recent() -> list[Ticket]:
    """The last finished ticket of every image that has one still remembered, newest first."""
    _prune()
    out = [t for p, t in _last.items() if active(p) is None]
    out.sort(key=lambda t: t.finished_at or t.created_at, reverse=True)
    return out


# ---------------------------------------------------------------------------------------
# The work
# ---------------------------------------------------------------------------------------

async def _run(t: Ticket) -> None:
    try:
        if t.lane == cq.SCENE_LANE:
            async with t._queue.turn(t.path, reserved=t._entry):
                t.status = RUNNING
                t.started_at = datetime.now(timezone.utc)
                if t.precheck is not None and await t.precheck():
                    t.skipped = True
                else:
                    await _scene_step(t)
            if not t.skipped and settings.motion_caption_enabled:
                # Out of the scene lane BEFORE the motion is made: the next image's scene
                # goes now, not after this image's paragraph.
                t.status = QUEUED
                _enter(t, cq.MOTION_LANE)
                await _settle_scene(t)
        if t.lane == cq.MOTION_LANE and not t.skipped:
            async with t._queue.turn(t.path, reserved=t._entry):
                t.status = RUNNING
                t.started_at = t.started_at or datetime.now(timezone.utc)
                if t.mode == MOTION and t.precheck is not None and await t.precheck():
                    t.skipped = True
                else:
                    await _motion_step(t)
        t.status = DONE
        t._image = None  # not kept for RESULT_TTL_S
        logger.info("Caption ticket %s on %s: done%s", t.id, t.path,
                    " (not needed any more: the words were saved while it waited)"
                    if t.skipped else
                    f" (motion failed: {t.motion_error})" if t.motion_error else "")
    except asyncio.CancelledError:
        t.status = FAILED
        if not t.withdrawn:
            t.error = "cancelled: the API stopped before this caption finished"
            raise
        # Withdrawn: an ordinary ending, not a shutdown -- do not propagate.
    except CaptionerBusy as e:
        t.status, t.error, t.busy = FAILED, str(e), True
        logger.info("Caption ticket %s on %s: refused, the captioner is busy (%s)",
                    t.id, t.path, e)
    except ImageUnreadable as e:
        t.status, t.error, t.unreadable = FAILED, str(e), True
        logger.warning("Caption ticket %s on %s: %s", t.id, t.path, e)
    except CaptionError as e:
        t.status, t.error = FAILED, str(e)
        logger.warning("Caption ticket %s on %s: failed: %s", t.id, t.path, e)
    except Exception as e:  # noqa: BLE001 - a person has to hear about it, whatever it was
        t.status, t.error = FAILED, f"{type(e).__name__}: {e}"
        logger.exception("Caption ticket %s on %s failed", t.id, t.path)
    finally:
        t._queue.discard(t._entry)  # cancelled before its turn began: no phantom in line
        t.finished_at = datetime.now(timezone.utc)
        t._finished_mono = time.monotonic()
        if _active.get(t.path) is t:
            del _active[t.path]
        # Only if nothing newer has finished: a chained pair can finish after the motion
        # ticket it queued behind, never before, but say so rather than assume it.
        prev = _last.get(t.path)
        # A withdrawn ticket is not a result worth showing: nothing was asked of the
        # captioner, and "failed" on the image would be a lie.
        if not t.withdrawn and (
                prev is None or (prev.finished_at or prev.created_at) <= t.finished_at):
            _last[t.path] = t
        t._done.set()


async def _load(t: Ticket) -> str:
    """The settings, and the image's saved scene, on a short session; then the image itself.
    Returns the saved scene ("" when there is none). Never holds a connection across S3."""
    from app.routes import captions
    async with async_session() as db:
        if t._cfg is None:
            t._cfg = await captions._get_all_settings(db)
        meta = await db.get(ImageMeta, t.path)
        saved_scene = (meta.scene_description or "").strip() if meta else ""
    if t._image is None:
        try:
            t._image = await asyncio.to_thread(s3.download_bytes, t.path)
        except Exception as e:
            raise ImageUnreadable(f"could not read {t.path}: {e}") from e
    return saved_scene


async def _motion_base() -> str:
    """The motion captioner's URL, refusing (CaptionerBusy) while its box renders. Its own
    short session. interactive=True routes it exactly as the modal's was routed."""
    from app.routes import captions
    async with async_session() as db:
        return await captions._caption_base(db, interactive=True)


async def _scene_step(t: Ticket) -> None:
    """Make the scene and save it. Raises CaptionError and its kinds.

    Saved with the modal's own writer and an empty motion half, so an old motion paragraph is
    cleared rather than left beside a scene it was not grounded on -- a held job needing both
    must not be released with mismatched halves between here and the motion step.
    """
    from app.routes import captions
    from app.routes.captions import ScenePair
    from app.routes.images import _apply_scene_pair

    await _load(t)
    p = t.params
    scene, instruction = await captions.caption_scene(
        t._image, t._cfg, _motion_base, style=p.get("style"), instruction=p.get("instruction"))
    if not (scene or "").strip():
        # A blank caption is a failure wearing a success's clothes. Storing it would mark
        # the image described and stop anything ever asking again.
        raise CaptionError("the captioner returned nothing for this image")
    async with async_session() as db:
        meta = await db.get(ImageMeta, t.path)
        if meta is None:
            meta = ImageMeta(path=t.path)
            db.add(meta)
        _apply_scene_pair(meta, ScenePair(scene=scene, scene_instruction=instruction))
        await db.commit()
    t._scene = scene.strip()
    logger.info("Caption ticket %s: scene of %s saved (%d words)%s", t.id, t.path,
                len(scene.split()),
                "; motion next, in the motion lane" if settings.motion_caption_enabled else "")


async def _settle_scene(t: Ticket) -> None:
    """Release every held segment on the image that needed only the scene. Non-fatal: the
    hold's own waiter and sweep release it anyway, this only makes it immediate."""
    from app import caption_hold
    try:
        await caption_hold.settle(t.path)
    except Exception:  # noqa: BLE001
        logger.exception("Caption ticket %s: could not release scene-only holds on %s",
                         t.id, t.path)


async def _motion_step(t: Ticket) -> None:
    """Make the motion paragraph, grounded on the scene, and save it beside that scene.

    In a PAIR a motion failure -- the captioner erring or refusing for a render -- is the
    pair's partial success (motion_error), never a reason to lose the scene. A MOTION ticket
    has nothing else to show for itself, so there it raises.
    """
    saved_scene = await _load(t)
    scene = t._scene or saved_scene
    if not scene:
        # A motion ticket whose scene vanished (deleted, or never made). Ground it on a
        # fresh one; rare enough that doing it here, outside the scene lane, costs nothing.
        await _scene_step(t)
        scene = t._scene
    p, cfg = t.params, t._cfg or {}
    try:
        base = await _motion_base()
        motion_style = p.get("motion_style") or cfg.get("motion_style", "")
        custom = (p["motion_instruction"] if p.get("motion_instruction") is not None
                  else cfg.get("motion_instruction", ""))
        motion, instruction = await describe_motion(t._image, scene, style=motion_style,
                                                    custom=custom, base_url=base)
        motion = (motion or "").strip()
        if not motion:
            raise CaptionError("the captioner returned no motion paragraph")
    except CaptionerBusy as e:
        if t.mode == MOTION:
            raise
        t.motion_error, t.motion_busy = str(e), True
        logger.info("Caption ticket %s: motion of %s refused (scene kept): %s", t.id, t.path, e)
        return
    except CaptionError as e:
        if t.mode == MOTION:
            raise
        t.motion_error = str(e)
        logger.warning("Caption ticket %s: motion of %s failed (scene kept): %s",
                       t.id, t.path, e)
        return
    async with async_session() as db:
        meta = await db.get(ImageMeta, t.path)
        if meta is None:  # deleted in the meantime; the scene it was grounded on went too
            raise CaptionError(f"{t.path} lost its saved description while captioning")
        if (meta.scene_description or "").strip() != scene.strip():
            # A re-roll replaced the scene while this paragraph was being made. Saving it
            # would pair it, invisibly, with a scene it never saw; the re-roll makes its own.
            logger.info("Caption ticket %s: the scene of %s changed while its motion was "
                        "made; that paragraph is dropped", t.id, t.path)
            t.motion_error = "the scene was replaced while this motion paragraph was made"
            return
        meta.motion_description = motion
        meta.motion_instruction = instruction
        meta.motion_described_at = datetime.now(timezone.utc)
        await db.commit()
    logger.info("Caption ticket %s: added the motion paragraph to %s (%d words)",
                t.id, t.path, len(motion.split()))
